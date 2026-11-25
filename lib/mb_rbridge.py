"""
dsh-metabolomics — 通用 Rscript 桥（ropls 等官方 R 实现的调用通道）

用途：把 Python 侧数据打包成 bundle，调
`Rscript --vanilla lib/<script>.R --in <bundle> --out <out> --params <json>`，
再把 R 写出的 result.json / tables/*.csv 读回来。

约定（与 dsh-geobulk/lib/rbridge.py 同一套设计，这里按代谢组学裁剪，不 import 它）：
  * bundle 目录：expr.tsv（样本×特征；首行特征名、首列样本名）、info.csv（首列样本名）、
    params.json、extra/<name>.tsv（可选附加表）
  * R 脚本必须写 <out>/result.json，表写 <out>/tables/<name>.csv（首列=序号列名，Python 侧
    按 index_col=0 读回；这样 samples/features 名不会因为 R 的 check.names 被改写）
  * 退出码：0=成功；3=缺前置（转 prerequisite）；4=运行期程序错误（转 runtime，**不得**据此
    改换统计方法）；5=缺依赖包（转 unsupported，中文原因）；其它=失败（stderr 尾部作消息）
  * stdout 一律只进返回值，不进协议：桥里没有任何 print()，R 的 stdout/stderr 都被 capture
  * 超时上限 600s（ropls 含置换检验时仍应远小于该值）

本文件只依赖标准库 + numpy/pandas + mb_common。禁止新增第三方依赖。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from mb_common import EngineError, jsonable

# Rscript 候选（顺序即优先级）
R_CANDIDATES = [
    os.environ.get("DS_MB_RSCRIPT") or "",
    "/Library/Frameworks/R.framework/Resources/bin/Rscript",
    "/usr/local/bin/Rscript",
    "/opt/homebrew/bin/Rscript",
    "/usr/bin/Rscript",
]

#: 探测时检查的 R 包（ropls 是本模块唯一需要的官方实现）
R_PACKAGES = ["ropls"]

#: 单次 R 调用允许的最长等待（协议要求 ≤600s）
MAX_TIMEOUT = 600
DEFAULT_TIMEOUT = 300
CACHE_SECONDS = 300

_LOCK = threading.Lock()
_STATE: dict = {
    "checked_at": 0.0,
    "available": False,
    "script": None,
    "version": None,
    "packages": {},
    "ropls_version": None,
    "error": None,
}


class RUnavailable(EngineError):
    """R / 依赖包不可用：调用方可以据此回退内置实现（其它错误不得回退）。"""

    def __init__(self, message: str, **details):
        super().__init__("unsupported", message, **details)


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------


def candidate_scripts() -> list[str]:
    out: list[str] = []
    for cand in R_CANDIDATES:
        if cand and os.path.exists(cand):
            out.append(cand)
    which = shutil.which("Rscript")
    if which:
        out.append(which)
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ))


def probe(force: bool = False) -> dict:
    """探测 Rscript 与 ropls，结果缓存 5 分钟。永不抛异常。"""
    now = time.time()
    with _LOCK:
        if not force and _STATE["checked_at"] and now - _STATE["checked_at"] < CACHE_SECONDS:
            return dict(_STATE)
        state = {
            "checked_at": now,
            "available": False,
            "script": None,
            "version": None,
            "packages": {},
            "ropls_version": None,
            "error": None,
        }
        for script in candidate_scripts():
            try:
                res = _run([script, "--version"], timeout=20)
                text = (res.stdout or "") + (res.stderr or "")
                if "Rscript" not in text and "R version" not in text:
                    continue
                state["script"] = script
                state["available"] = True
                raw = text.strip()
                match = re.search(r"version\s+([0-9]+\.[0-9]+\.[0-9]+)", raw)
                state["version"] = match.group(1) if match else (raw.splitlines()[0] if raw else "R")
                break
            except Exception as error:  # noqa: BLE001 - 探测失败只记录
                state["error"] = str(error)
                continue
        if state["available"]:
            code = (
                "cat(paste(vapply(c(%s), function(p) requireNamespace(p, quietly=TRUE), logical(1)),"
                " collapse=','))" % (",".join('"%s"' % p for p in R_PACKAGES))
            )
            try:
                res = _run([state["script"], "--vanilla", "-e", code], timeout=120)
                flags = (res.stdout or "").strip().split(",")
                if len(flags) == len(R_PACKAGES):
                    state["packages"] = {p: flags[i].strip() == "TRUE" for i, p in enumerate(R_PACKAGES)}
                else:
                    state["packages"] = {}
            except Exception as error:  # noqa: BLE001
                state["packages"] = {}
                state["error"] = str(error)
            if (state["packages"] or {}).get("ropls"):
                try:
                    res = _run([state["script"], "--vanilla", "-e",
                                'cat(as.character(packageVersion("ropls")))'], timeout=60)
                    text = (res.stdout or "").strip()
                    state["ropls_version"] = text.splitlines()[-1].strip() if text else None
                except Exception:  # noqa: BLE001
                    state["ropls_version"] = None
        _STATE.update(state)
        return dict(_STATE)


def r_available() -> bool:
    return bool(probe().get("available"))


def has_pkg(name: str) -> bool:
    return bool((probe().get("packages") or {}).get(name))


def ropls_available() -> bool:
    return r_available() and has_pkg("ropls")


def r_info() -> dict:
    state = probe()
    return {
        "available": bool(state.get("available")),
        "version": state.get("version"),
        "script": state.get("script"),
        "packages": state.get("packages") or {},
        "ropls_version": state.get("ropls_version"),
        "error": state.get("error"),
    }


def unavailable_reason() -> str:
    """R / ropls 不可用的中文原因（用于回退时的 method 字符串）。"""
    state = probe()
    if not state.get("available"):
        return "本机没有可用的 Rscript"
    if not (state.get("packages") or {}).get("ropls"):
        return "R 已找到但 ropls 包未安装（Bioconductor）"
    return ""


# ---------------------------------------------------------------------------
# bundle / 结果读写
# ---------------------------------------------------------------------------


def build_bundle(payload: dict, bundle_dir: str) -> str:
    """把 payload 写成 R 侧可读的 bundle：expr.tsv / info.csv / params.json / extra/*.tsv。"""
    os.makedirs(bundle_dir, exist_ok=True)
    expr = payload.get("expr")
    features = payload.get("features")
    samples = payload.get("samples")
    info = payload.get("info")
    if expr is not None:
        arr = np.asarray(expr, dtype=np.float64)
        if arr.ndim != 2:
            raise EngineError("internal", "run_r: expr 必须是二维数组（样本×特征）")
        if samples is None:
            samples = [f"S{i + 1}" for i in range(arr.shape[0])]
        if features is None:
            features = [f"F{i + 1}" for i in range(arr.shape[1])]
        if len(samples) != arr.shape[0] or len(features) != arr.shape[1]:
            raise EngineError("internal", "run_r: 样本/特征名数量与 expr 形状不一致")
        frame = pd.DataFrame(arr, index=[str(s) for s in samples],
                             columns=[str(f) for f in features])
        # 样本×特征 → TSV（首行特征名，首列样本名）
        frame.to_csv(os.path.join(bundle_dir, "expr.tsv"), sep="\t", index=True, encoding="utf-8")
    if info is not None:
        frame = info if isinstance(info, pd.DataFrame) else pd.DataFrame(info)
        frame.to_csv(os.path.join(bundle_dir, "info.csv"), sep="\t", index=True, encoding="utf-8")
    params = payload.get("params") or {}
    with open(os.path.join(bundle_dir, "params.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(params), handle, ensure_ascii=False, indent=2)
    extra = payload.get("extra") or {}
    if extra:
        extra_dir = os.path.join(bundle_dir, "extra")
        os.makedirs(extra_dir, exist_ok=True)
        for name, value in extra.items():
            frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            frame.to_csv(os.path.join(extra_dir, f"{name}.tsv"), sep="\t", index=True,
                         encoding="utf-8")
    return bundle_dir


def _read_tables(tables_dir: str) -> dict:
    """读 tables/*.csv（首列是行名）；解析失败的表跳过但不丢整个结果。"""
    tables: dict[str, pd.DataFrame] = {}
    if not os.path.isdir(tables_dir):
        return tables
    for name in sorted(os.listdir(tables_dir)):
        if not name.endswith((".csv", ".tsv")):
            continue
        path = os.path.join(tables_dir, name)
        key = os.path.splitext(name)[0]
        try:
            if name.endswith(".tsv"):
                tables[key] = pd.read_csv(path, sep="\t", index_col=0)
            else:
                tables[key] = pd.read_csv(path, index_col=0)
        except Exception:  # noqa: BLE001 - 单表读回失败不影响其它表
            continue
    return tables


def _read_result(outdir: str) -> tuple[dict, dict]:
    result_path = os.path.join(outdir, "result.json")
    payload: dict = {}
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as error:  # noqa: BLE001
            raise EngineError("internal", f"R 输出 result.json 解析失败：{error}")
    if not isinstance(payload, dict):
        payload = {}
    tables = _read_tables(os.path.join(outdir, "tables"))
    payload.setdefault("warnings", [])
    payload.setdefault("tables", [])
    payload.setdefault("summary", {"title": "", "bullets": [], "metrics": {}})
    return payload, tables


def _read_error(outdir: str) -> dict | None:
    path = os.path.join(outdir, "error.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 调用
# ---------------------------------------------------------------------------


def run_r(script_name: str, payload: dict, outdir: str,
          timeout: int = DEFAULT_TIMEOUT) -> dict:
    """调 R 脚本并读回结果。

    返回 {"ok":True,"result":dict,"tables":{name:DataFrame},"warnings":[],
          "ms":int,"stdout":str,"stderr":str,"r_version":str,"script":str}
    失败抛 EngineError：unsupported（缺 R / 缺包）、prerequisite（数据不满足）、
    runtime（程序错误，调用方不得静默换方法）、timeout、internal。
    """
    state = probe()
    if not state.get("available"):
        raise RUnavailable(
            "本机没有可用的 Rscript：ropls 官方实现不可用（已尝试 "
            + "、".join(candidate_scripts() or ["PATH"]) + "）"
        )
    if not (state.get("packages") or {}).get("ropls"):
        raise RUnavailable(
            "R 包 ropls 未安装：无法调用 opls::opls 官方实现。安装方式（Bioconductor 主站不可达时）："
            "克隆 git.bioconductor.org/packages/ropls，从 DESCRIPTION/NAMESPACE 去掉 "
            "MultiAssayExperiment/MultiDataSet（对象互操作层，核心算法不依赖）后 "
            "R CMD INSTALL；Biobase/SummarizedExperiment 需已在 R 库里。"
        )
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(lib_dir, script_name)
    if not os.path.exists(script):
        raise EngineError("unsupported", f"R 脚本不存在：{script_name}")
    timeout = int(max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT)))
    outdir = os.path.abspath(outdir)
    bundle = os.path.join(outdir, "r_bundle")
    r_out = os.path.join(outdir, "r_out")
    # 每次调用前清掉上一轮的 bundle / 输出，避免残留表被当成新结果读回
    for path in (bundle, r_out):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    build_bundle(payload, bundle)
    params_json = json.dumps(jsonable(payload.get("params") or {}), ensure_ascii=False)
    cmd = [state["script"], "--vanilla", script, "--in", bundle, "--out", r_out,
           "--params", params_json]
    t0 = time.time()
    try:
        res = _run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise EngineError("timeout", f"R 计算超时（>{timeout}s）：{script_name}")
    except Exception as error:  # noqa: BLE001
        raise EngineError("internal", f"R 调用失败：{error}")
    elapsed = int((time.time() - t0) * 1000)
    stderr_text = (res.stderr or "")
    stdout_text = (res.stdout or "")
    stderr_tail = stderr_text[-2000:]
    error_json = _read_error(r_out)
    if res.returncode != 0:
        message = ""
        if isinstance(error_json, dict) and error_json.get("message"):
            message = str(error_json["message"])
        reason = message or stderr_tail.strip() or stdout_text.strip() or f"Rscript 退出码 {res.returncode}"
        if res.returncode == 5 or "MISSING_PACKAGE:" in stderr_text:
            raise RUnavailable(f"R 依赖包缺失：{reason[-800:]}")
        if res.returncode == 3:
            raise EngineError("prerequisite", reason[-800:])
        if res.returncode == 4:
            # 运行期程序错误：调用方不得据此改换统计方法（GEO-02 同一原则）
            raise EngineError("runtime", f"R 计算内部错误：{reason[-800:]}")
        raise EngineError("internal", f"R 计算失败（退出码 {res.returncode}）：{reason[-800:]}")
    parsed, tables = _read_result(r_out)
    return {
        "ok": True,
        "result": parsed,
        "tables": tables,
        "warnings": parsed.get("warnings") or [],
        "ms": elapsed,
        "stdout": stdout_text[-2000:],
        "stderr": stderr_tail,
        "r_version": state.get("version"),
        "ropls_version": state.get("ropls_version"),
        "script": script,
    }


def try_run_r(script_name: str, payload: dict, outdir: str,
              timeout: int = DEFAULT_TIMEOUT) -> dict | None:
    """run_r 的软版本：R / ropls 不可用时返回 None（调用方可回退内置实现）。

    注意：prerequisite / runtime / timeout **必须上抛** —— 数据不满足要让用户看到真实原因，
    程序错误绝不能被静默换成另一种算法。
    """
    try:
        return run_r(script_name, payload, outdir, timeout=timeout)
    except RUnavailable:
        return None


__all__ = [
    "MAX_TIMEOUT",
    "DEFAULT_TIMEOUT",
    "RUnavailable",
    "probe",
    "r_available",
    "r_info",
    "has_pkg",
    "ropls_available",
    "unavailable_reason",
    "build_bundle",
    "run_r",
    "try_run_r",
]
