#!/usr/bin/env python3
"""dsh-metabolomics — Host 元数据/引擎冒烟测试（只用标准库）

拉起 `python3 -u lib/mb_engine.py --workdir <tmp>`，按 DESIGN.md §2.1 的 JSON-lines 协议
逐条发命令，验证 Host 层与 Python worker 之间的契约没断：

  * ping      → ok、capabilities.steps 恰好 12 个且顺序正确
  * formats   → 格式清单非空（mb_io 未就绪时也要有静态兜底）
  * demo      → 内置示例可载入，n_samples > 0
  * analyses  → 静态分析目录 ≥ 30 项（模块未落盘也要列出来）
  * bogus     → 未知命令返回 bad_request（不能崩 worker）

用法：`python3 test/host_smoke_test.py`（通过打印 PASS，失败打印原因并退出码 1）
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "lib")
ENGINE = os.path.join(LIB, "mb_engine.py")

EXPECTED_STEPS = [
    "qc", "filter", "impute", "normalize", "transform", "pca",
    "plsda", "de", "batch", "enrich", "biomarker", "annotate",
]
MIN_ANALYSES = 30
START_TIMEOUT = 60.0
PING_TIMEOUT = 180.0
DEMO_TIMEOUT = 240.0
DEFAULT_TIMEOUT = 120.0


class EngineFailure(AssertionError):
    """引擎超时/退出等基础设施问题（与断言失败区分，方便看日志）。"""


class Worker:
    """极简 JSON-lines 客户端：自己按行缓冲 + select 超时，不依赖服务端 EOF。"""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.log_path = os.path.join(workdir, "engine.stderr.log")
        self._log = open(self.log_path, "wb")
        self._buffer = b""
        self._next_id = 0
        self.noise: list[str] = []
        self.env = dict(os.environ)
        self.env.setdefault("MPLBACKEND", "Agg")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", ENGINE, "--workdir", workdir],
            cwd=LIB,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            bufsize=0,
            env=self.env,
        )

    # -- 低层读写 ---------------------------------------------------------
    def _readline(self, deadline: float) -> str:
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineFailure(f"等待引擎响应超时（{self._stderr_tail()}）")
            readable, _, _ = select.select([self.proc.stdout], [], [], remaining)
            if not readable:
                raise EngineFailure(f"等待引擎响应超时（{self._stderr_tail()}）")
            chunk = self.proc.stdout.read(65536)
            if not chunk:
                raise EngineFailure(
                    f"引擎进程提前退出（code={self.proc.poll()}）：{self._stderr_tail()}"
                )
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", "replace")

    def _stderr_tail(self) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            return " ".join(text.strip().splitlines()[-4:])[:600] or "无 stderr 输出"
        except OSError:
            return "无法读取 stderr 日志"

    def _send(self, cmd: str, args: dict | None = None) -> int:
        self._next_id += 1
        payload = {"id": self._next_id, "cmd": cmd, "args": args or {}}
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise EngineFailure(f"写入引擎失败（{cmd}）：{error}；{self._stderr_tail()}") from error
        return self._next_id

    def request(self, cmd: str, args: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict:
        """发一条命令并等到同 id 的响应（progress 事件跳过）。"""
        request_id = self._send(cmd, args)
        deadline = time.monotonic() + timeout
        while True:
            line = self._readline(deadline).strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # 契约要求 stdout 只有协议行；这里不因噪声直接失败，但记录下来
                self.noise.append(line[:200])
                continue
            if not isinstance(message, dict):
                self.noise.append(line[:200])
                continue
            if message.get("event"):
                continue  # 进度事件
            if str(message.get("id")) != str(request_id):
                self.noise.append(line[:200])
                continue
            return message

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.request("shutdown", timeout=10.0)
        except Exception:  # noqa: BLE001 - 关停失败不影响结论
            pass
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            self._log.close()
        except OSError:
            pass


def check(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def ok_result(message: dict, label: str) -> dict:
    check(message.get("ok") is True, f"{label}: 期望 ok=true，实际 {json.dumps(message, ensure_ascii=False)[:400]}")
    result = message.get("result")
    check(isinstance(result, dict), f"{label}: result 不是对象：{type(result).__name__}")
    return result


def run_case(worker: Worker) -> dict:
    """按顺序发 ping/formats/demo/analyses/bogus，返回收集到的证据。"""
    evidence: dict = {}

    # 1) ping：引擎握手 + 能力清单
    ping = ok_result(worker.request("ping", timeout=PING_TIMEOUT), "ping")
    capabilities = ping.get("capabilities") or {}
    steps = capabilities.get("steps") or []
    step_ids = [item.get("id") for item in steps if isinstance(item, dict)]
    check(
        len(steps) == 12,
        f"ping: capabilities.steps 期望 12 个，实际 {len(steps)} 个：{step_ids}",
    )
    check(
        step_ids == EXPECTED_STEPS,
        f"ping: 步骤顺序不对\n  期望 {EXPECTED_STEPS}\n  实际 {step_ids}",
    )
    check(bool(ping.get("python")), "ping: 缺少 python 版本字段")
    check(ping.get("geo") is True, f"ping: geo 期望 true，实际 {ping.get('geo')!r}")
    evidence["ping"] = {
        "python": ping.get("python"),
        "mode": ping.get("mode"),
        "steps": len(steps),
        "analyses": len(capabilities.get("analyses") or []),
        "formats": len(capabilities.get("formats") or []),
        "geo": ping.get("geo"),
        "metabolights": ping.get("metabolights"),
    }

    # 2) formats：数据格式清单（mb_io 未就绪时也必须有静态兜底）
    formats = ok_result(worker.request("formats"), "formats")
    ids = [str(item) for item in (formats.get("ids") or [])]
    entries = formats.get("formats") or []
    check(bool(ids), f"formats: ids 为空（formats={json.dumps(formats, ensure_ascii=False)[:300]}）")
    check(bool(entries), "formats: formats 列表为空")
    check("matrix_text" in ids, f"formats: 缺少 matrix_text，实际 {ids}")
    evidence["formats"] = {"ids": len(ids), "first": ids[:4]}

    # 3) demo：内置代谢组学示例（72 样本 × 1600 特征）
    demo = ok_result(worker.request("demo", timeout=DEMO_TIMEOUT), "demo")
    dataset = demo.get("dataset") or {}
    n_samples = dataset.get("n_samples")
    n_features = dataset.get("n_features")
    check(isinstance(n_samples, int) and n_samples > 0, f"demo: n_samples 期望 > 0，实际 {n_samples!r}")
    check(isinstance(n_features, int) and n_features > 0, f"demo: n_features 期望 > 0，实际 {n_features!r}")
    preview = demo.get("preview") or {}
    check(bool(preview.get("features")) or bool(preview.get("features_preview")),
          "demo: preview 里没有特征列表")
    evidence["demo"] = {
        "id": dataset.get("id"),
        "n_samples": n_samples,
        "n_features": n_features,
        "source_format": dataset.get("source_format"),
        "has_mz": bool((dataset.get("feature_meta") or {}).get("has_mz")),
    }

    # 4) analyses：静态目录（DESIGN §4 共 36 项，模块未落盘也要列出来）
    analyses = ok_result(worker.request("analyses", {"dataset": evidence["demo"]["id"]}), "analyses")
    items = analyses.get("analyses") or []
    analysis_ids = [str(item.get("id")) for item in items if isinstance(item, dict)]
    check(
        len(items) >= MIN_ANALYSES,
        f"analyses: 期望 ≥ {MIN_ANALYSES} 项，实际 {len(items)} 项",
    )
    for expected in ("qc_summary", "plsda_vip", "univariate_de", "mummichog_pathway", "rf_importance"):
        check(expected in analysis_ids, f"analyses: 缺少 {expected}（实际 {analysis_ids}）")
    check(bool(analyses.get("groups")), "analyses: groups 为空")
    evidence["analyses"] = {"count": len(items), "groups": len(analyses.get("groups") or [])}

    # 5) bogus：未知命令必须是可读的 bad_request（worker 不能崩）
    bogus = worker.request("bogus")
    check(bogus.get("ok") is False, f"bogus: 未知命令竟然成功：{json.dumps(bogus, ensure_ascii=False)[:300]}")
    error = bogus.get("error") or {}
    check(
        error.get("code") == "bad_request",
        f"bogus: 期望 error.code=bad_request，实际 {error.get('code')!r}（message={error.get('message')!r}）",
    )
    evidence["bogus"] = {"code": error.get("code"), "message": str(error.get("message"))[:80]}

    # bogus 之后引擎仍应可用（单条命令失败不拖垮 worker）
    pong = ok_result(worker.request("ping", timeout=PING_TIMEOUT), "ping-after-bogus")
    check(len(((pong.get("capabilities") or {}).get("steps") or [])) == 12, "ping: bogus 之后能力清单异常")
    return evidence


def main() -> int:
    if not os.path.isfile(ENGINE):
        print(f"FAIL: 找不到引擎脚本 {ENGINE}", file=sys.stderr)
        return 2
    workdir = tempfile.mkdtemp(prefix="dsh-mb-smoke-")
    worker = None
    try:
        started = time.monotonic()
        worker = Worker(workdir)
        evidence = run_case(worker)
        elapsed = time.monotonic() - started
        noise = list(worker.noise)
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        if worker is not None:
            tail = worker._stderr_tail()
            if tail:
                print(f"engine stderr: {tail}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - 基础设施异常也给出可读输出
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if worker is not None:
            worker.close()
        shutil.rmtree(workdir, ignore_errors=True)

    for label, value in evidence.items():
        print(f"  {label}: {json.dumps(value, ensure_ascii=False)}")
    if noise:
        # 契约要求 stdout 只有协议行；噪声不算致命（可能来自第三方库），但要报出来
        print(f"  warn: stdout 出现 {len(noise)} 行非协议输出（例：{noise[0][:120]}）", file=sys.stderr)
    print(f"PASS  dsh-metabolomics host smoke（{elapsed:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
