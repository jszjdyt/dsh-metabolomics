#!/usr/bin/env python3
"""
dsh-metabolomics — Python worker（长驻子进程）

协议（DESIGN.md §2.1，与 dsh-geobulk 完全一致）：
  stdin  : 一行一个 JSON 请求 {"id":"..","cmd":"..","args":{..}}
  stdout : 一行一个 JSON 响应 {"id":"..","ok":true,"result":{..}} /
           {"id":"..","ok":false,"error":{"code":"..","message":"..","trace":".."}}
           可选 {"id":"..","event":"progress","message":"..","pct":0-100}
  stdout 只允许协议行；日志一律 stderr。

启动：python3 -u lib/mb_engine.py --workdir <abs_dir>

设计原则：
  * 命令各自懒导入对应模块（mb_state / mb_io / mb_geo / mb_registry），
    任一模块缺失或导入失败只影响用到它的命令，ping/formats/steps 仍可回答。
  * 单条命令异常只影响该条；shutdown 或 stdin EOF 退出。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PLUGIN_VERSION = "0.1.0"
PACKAGE_NAMES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "statsmodels", "openpyxl", "h5py"]
OPTIONAL_PACKAGES = ["lifelines", "PIL", "networkx", "anndata", "seaborn", "matchms", "rdkit", "scanpy"]
METABOLIGHTS_PING_URL = "https://www.ebi.ac.uk/metabolights/ws/studies"


def _prepare_env(workdir: str) -> None:
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(workdir, "mpl-cache"))
    os.environ["DS_MB_WORKDIR"] = workdir
    # 兼容 mb_common 里的旧变量名（其它工作台共用同一份工具层写法）
    os.environ.setdefault("DS_BG_WORKDIR", workdir)
    try:
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    except OSError:
        os.environ["MPLCONFIGDIR"] = "/tmp"
    for sub in ("figures", "exports", "uploads", "downloads", "analyses", "unpack", "genesets", "ms2"):
        try:
            os.makedirs(os.path.join(workdir, sub), exist_ok=True)
        except OSError:
            pass


class Worker:
    def __init__(self, workdir: str):
        self.workdir = workdir
        self.started = time.time()
        self.state = None
        self._engine_info_cache: dict | None = None
        self._metabolights_cache: bool | None = None
        try:
            self.config = json.loads(os.environ.get("DS_MB_CONFIG") or "{}")
        except Exception:  # noqa: BLE001
            self.config = {}

    # -- 基础设施 ---------------------------------------------------------
    def load_state(self):
        if self.state is None:
            import mb_state

            self.state = mb_state.State()
        return self.state

    def engine_info(self) -> dict:
        if self._engine_info_cache is not None:
            return self._engine_info_cache
        import platform

        import numpy
        import pandas
        import scipy
        import sklearn

        packages = {
            "numpy": getattr(numpy, "__version__", "?"),
            "pandas": getattr(pandas, "__version__", "?"),
            "scipy": getattr(scipy, "__version__", "?"),
            "sklearn": getattr(sklearn, "__version__", "?"),
        }
        for name in PACKAGE_NAMES[4:] + OPTIONAL_PACKAGES:
            try:
                module = __import__(name)
                packages[name] = getattr(module, "__version__", "?")
            except Exception:  # noqa: BLE001
                packages[name] = None
        info = {
            "python": platform.python_version(),
            "platform": sys.platform,
            "mode": "builtin",
            "workdir": self.workdir,
            "packages": packages,
            "plugin": {"name": "dsh-metabolomics", "version": PLUGIN_VERSION},
            "started_at": self.started,
        }
        self._engine_info_cache = info
        return info

    def _probe_metabolights(self) -> bool:
        try:
            request = urllib.request.Request(METABOLIGHTS_PING_URL, headers={
                "User-Agent": f"dsh-metabolomics/{PLUGIN_VERSION}",
                "Accept": "application/json",
            })
            # 本机 openssl 缺根证书时裸 urlopen 会 CERTIFICATE_VERIFY_FAILED：
            # 与 mb_geo 保持一致，优先用 certifi 的 CA 包（没有就退回默认上下文）。
            context = None
            try:
                import ssl

                import certifi

                context = ssl.create_default_context(cafile=certifi.where())
            except Exception:  # noqa: BLE001
                context = None
            try:
                response = urllib.request.urlopen(request, timeout=6, context=context)
            except TypeError:  # pragma: no cover - context 不被支持时退回
                response = urllib.request.urlopen(request, timeout=6)
            with response:
                status = int(getattr(response, "status", 200) or 200)
                return status < 400
        except Exception:  # noqa: BLE001 - 网络不可达/被墙都只是降级
            return False

    def metabolights_available(self) -> bool:
        """MetaboLights（EBI）数据源是否可用：模块在 + 端点可达（结果缓存）。"""
        if self._metabolights_cache is not None:
            return self._metabolights_cache
        capable = False
        try:
            import mb_geo

            capable = callable(getattr(mb_geo, "ml_search", None))
        except Exception:  # noqa: BLE001
            capable = False
        self._metabolights_cache = bool(capable and self._probe_metabolights())
        return self._metabolights_cache

    def _registry(self):
        import mb_registry

        return mb_registry

    # -- ping / 目录 ------------------------------------------------------
    def cmd_ping(self, args):
        registry = self._registry()
        info = dict(self.engine_info())
        info["modules"] = registry.module_status()
        info["capabilities"] = {
            "steps": registry.steps_catalog(),
            "analyses": [item["id"] for item in registry.analyses_catalog(None)["analyses"]],
            "plots": registry.available_plots(),
            "formats": registry.formats_catalog()["ids"],
        }
        info["geo"] = True
        info["metabolights"] = self.metabolights_available()
        info["config"] = {"workdir": self.workdir}
        return info

    def cmd_steps(self, args):
        return self._registry().steps_payload(reload=bool(args.get("reload")))

    def cmd_formats(self, args):
        return self._registry().formats_catalog()

    # -- 数据集 -----------------------------------------------------------
    def cmd_open(self, args):
        registry = self._registry()
        state = self.load_state()
        if args.get("demo"):
            import mb_state

            dataset = mb_state.make_demo(state, name=args.get("name") or "demo-metabo")
            registry.ensure_fields(dataset)
            try:
                state.select(dataset.id)   # 新建/载入的数据集应当成为当前数据集
            except Exception:  # noqa: BLE001
                state.active = dataset.id
            return {"dataset": registry.dataset_summary(dataset), "preview": registry.preview(dataset)}

        raw_path = args.get("path") or args.get("url")
        if not raw_path:
            raise _err("bad_request", "请提供 path（本地文件/目录）或 url")
        is_url = str(raw_path).startswith(("http://", "https://", "ftp://"))
        path = str(raw_path)
        if is_url:
            path = self._download_url(path)
        else:
            path = os.path.abspath(path)
            existing = state.find_by_path(path) if hasattr(state, "find_by_path") else None
            if existing is not None:
                registry.ensure_fields(existing)
                return {"dataset": registry.dataset_summary(existing),
                        "preview": registry.preview(existing), "reused": True}
            if not os.path.exists(path):
                raise _err("not_found", f"路径不存在：{path}")

        import mb_io

        options = {
            "data_format": args.get("data_format") or args.get("format") or "auto",
            "member": args.get("member"),
            "object": args.get("object"),
            "transpose": args.get("transpose"),
            "delimiter": args.get("delimiter"),
            "sheet": args.get("sheet"),
            "sample_sheet": args.get("sample_sheet"),
            "clinical": args.get("clinical"),
            "feature_col": args.get("feature_col"),
            "mz_col": args.get("mz_col"),
            "rt_col": args.get("rt_col"),
            "group_col": args.get("group_col"),
            "zero_as_missing": args.get("zero_as_missing"),
        }
        options = {k: v for k, v in options.items() if v is not None}
        loaded = mb_io.load_source(path, workdir=self.workdir, options=options)
        dataset = registry.build_dataset(state, loaded, name=args.get("name"), source=path, options=options)
        return {"dataset": registry.dataset_summary(dataset), "preview": registry.preview(dataset)}

    def _download_url(self, url: str) -> str:
        """把远程直链下载到 workdir/downloads 再按本地文件导入。"""
        name = os.path.basename(url.split("?", 1)[0]) or "download.bin"
        name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:120]
        dest_dir = os.path.join(self.workdir, "downloads")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{int(time.time())}-{name}")
        request = urllib.request.Request(url, headers={"User-Agent": f"dsh-metabolomics/{PLUGIN_VERSION}"})
        with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise _err("unsupported", f"下载失败或文件为空：{url}")
        return dest

    def cmd_demo(self, args):
        return self.cmd_open({"demo": True, "name": args.get("name")})

    def cmd_list(self, args):
        state = self.load_state()
        registry = self._registry()
        for dataset in getattr(state, "datasets", {}).values():
            registry.ensure_fields(dataset)
        return {"datasets": state.summaries(), "active": state.active}

    def cmd_select(self, args):
        state = self.load_state()
        target = str(args.get("dataset") or "")
        datasets = getattr(state, "datasets", {}) or {}
        if target not in datasets:
            raise _err("not_found", f"数据集 {target} 不存在")
        state.active = target
        registry = self._registry()
        registry.ensure_fields(datasets[target])
        return {"active": state.active, "dataset": registry.dataset_summary(datasets[target])}

    def cmd_close(self, args):
        state = self.load_state()
        target = str(args.get("dataset") or state.active or "")
        datasets = getattr(state, "datasets", {}) or {}
        if not target or target not in datasets:
            return {"closed": False, "datasets": state.summaries(), "active": state.active}
        state.close(target)
        return {"closed": True, "datasets": state.summaries(), "active": state.active}

    def cmd_reset(self, args):
        state = self.load_state()
        state.reset()
        return {"reset": True, "datasets": [], "active": None}

    def cmd_preview(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        return registry.preview(dataset)

    def cmd_samples(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        payload = dict(args)
        payload.pop("dataset", None)
        return registry.handle_samples(dataset, payload)

    # -- 执行 -------------------------------------------------------------
    def cmd_run(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        step = str(args.get("step") or "")
        if not step:
            raise _err("bad_request", "请提供 step")
        return registry.run_step(dataset, step, args.get("params") or {}, self.workdir, self.engine_info())

    def cmd_plot(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        kind = str(args.get("kind") or "")
        if not kind:
            raise _err("bad_request", "请提供 kind（图类型）")
        return registry.run_plot(dataset, kind, args.get("params") or {}, self.workdir, self.engine_info())

    def cmd_analyses(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        return registry.analyses_catalog(dataset)

    def cmd_analysis(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        analysis_id = str(args.get("id") or "")
        if not analysis_id:
            raise _err("bad_request", "请提供分析 id")
        return registry.run_analysis(dataset, analysis_id, args.get("params") or {}, self.workdir,
                                     self.engine_info())

    def cmd_export(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        return registry.export_dataset(dataset, str(args.get("what") or "summary_json"),
                                       args.get("path") or args.get("out_path"), self.workdir,
                                       self.engine_info())

    def cmd_audit(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        return registry.audit(dataset)

    def cmd_suggest(self, args):
        registry = self._registry()
        dataset = self.load_state().get(args.get("dataset"))
        return registry.suggest(dataset)

    def cmd_recipe(self, args):
        registry = self._registry()
        return registry.recipe(args.get("id") or None)

    # -- 导入探测 ---------------------------------------------------------
    def cmd_inspect(self, args):
        raw_path = args.get("path") or args.get("url")
        if not raw_path:
            raise _err("bad_request", "请提供 path 或 url")
        path = str(raw_path)
        if path.startswith(("http://", "https://", "ftp://")):
            path = self._download_url(path)
        else:
            path = os.path.abspath(path)
            if not os.path.exists(path):
                raise _err("not_found", f"路径不存在：{path}")
        import mb_io

        options = {
            "data_format": args.get("data_format") or args.get("format") or "auto",
            "member": args.get("member"),
            "object": args.get("object"),
            "delimiter": args.get("delimiter"),
            "sheet": args.get("sheet"),
            "transpose": args.get("transpose"),
        }
        options = {k: v for k, v in options.items() if v is not None}
        return mb_io.inspect_source(path, workdir=self.workdir, options=options)

    # -- GEO --------------------------------------------------------------
    def cmd_geo_search(self, args):
        import mb_geo

        return mb_geo.geo_search(
            term=str(args.get("term") or ""),
            species=str(args.get("species") or "all"),
            method=str(args.get("method") or "all"),
            study_type=str(args.get("study_type") or "all"),
            min_samples=int(args.get("min_samples") or 0),
            has_clinical=bool(args.get("has_clinical")),
            limit=int(args.get("limit") or 20),
            page=int(args.get("page") or 0),
            sort=str(args.get("sort") or "relevance"),
        )

    def cmd_geo_detail(self, args):
        import mb_geo

        return mb_geo.geo_detail(str(args.get("gse") or ""))

    def cmd_geo_download(self, args):
        import mb_geo

        progress = self._progress_callback(args.get("__request_id__"))
        gse = str(args.get("gse") or "")
        files = args.get("files") or []
        dest = str(args.get("dest_dir") or os.path.join(self.workdir, "downloads", "GEO"))
        return mb_geo.geo_download(gse, files, dest, overwrite=bool(args.get("overwrite")), progress=progress)

    # -- MetaboLights -----------------------------------------------------
    def cmd_ml_search(self, args):
        import mb_geo

        return mb_geo.ml_search(
            query=str(args.get("query") or args.get("term") or ""),
            limit=int(args.get("limit") or 20),
            page=int(args.get("page") or 0),
            assay_type=args.get("assay_type"),
        )

    def cmd_ml_detail(self, args):
        import mb_geo

        return mb_geo.ml_detail(str(args.get("accession") or ""))

    def cmd_ml_download(self, args):
        import mb_geo

        progress = self._progress_callback(args.get("__request_id__"))
        accession = str(args.get("accession") or "")
        files = args.get("files") or []
        dest = str(args.get("dest_dir") or os.path.join(self.workdir, "downloads", "MetaboLights"))
        return mb_geo.ml_download(accession, files, dest,
                                  overwrite=bool(args.get("overwrite")), progress=progress)

    def _progress_callback(self, request_id):
        if not request_id:
            return None

        def progress(message, pct=None):
            payload = {"id": request_id, "event": "progress", "message": str(message)}
            if pct is not None:
                try:
                    payload["pct"] = float(pct)
                except (TypeError, ValueError):
                    pass
            _emit(payload)

        return progress

    # -- 关停 -------------------------------------------------------------
    def cmd_shutdown(self, args):
        return {"bye": True}

    # -- 分发 -------------------------------------------------------------
    def dispatch(self, cmd: str, args: dict):
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None or not callable(handler):
            raise _err("bad_request", f"未知命令：{cmd}")
        return handler(args or {})


class _EngineFailure(Exception):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        for key, value in (self.details or {}).items():
            out[key] = _jsonable(value)
        return out


def _err(code: str, message: str, **details):
    return _EngineFailure(code, message, **details)


def _jsonable(value):
    """引擎层自己的兜底序列化（mb_common 导入失败时也能用）。"""
    try:
        from mb_common import jsonable

        return jsonable(value)
    except Exception:  # noqa: BLE001
        return value


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=os.path.join(os.path.dirname(HERE), "work"))
    parsed, _ = parser.parse_known_args()
    workdir = os.path.abspath(parsed.workdir)
    _prepare_env(workdir)

    # mb_common 必须在 MPLBACKEND/MPLCONFIGDIR 就位之后再导入（它会锁 Agg 后端）
    import mb_common

    worker = Worker(workdir)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            payload = json.loads(line)
            request_id = payload.get("id")
            cmd = str(payload.get("cmd") or "")
            args = payload.get("args") or {}
            if cmd in ("geo_download", "ml_download"):
                args = dict(args)
                args["__request_id__"] = request_id
            result = worker.dispatch(cmd, args)
            _emit({"id": request_id, "ok": True, "result": mb_common.jsonable(result)})
            if cmd == "shutdown":
                return 0
        except _EngineFailure as error:
            _emit({"id": request_id, "ok": False, "error": error.to_dict()})
        except mb_common.EngineError as error:
            _emit({"id": request_id, "ok": False, "error": error.to_dict()})
        except (BrokenPipeError, KeyboardInterrupt):
            return 0
        except Exception as error:  # noqa: BLE001
            trace = traceback.format_exc()
            sys.stderr.write(trace + "\n")
            sys.stderr.flush()
            _emit({"id": request_id, "ok": False,
                   "error": {"code": "internal", "message": f"{type(error).__name__}: {error}",
                             "trace": trace[-2000:]}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
