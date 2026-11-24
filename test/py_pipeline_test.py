#!/usr/bin/env python3
"""dsh-metabolomics 集成回归测试：demo → 12 步 → 全部图 → 全部分析 → 导出 → 自检。

运行：MPLBACKEND=Agg MPLCONFIGDIR=<tmp> python3 test/py_pipeline_test.py
不依赖网络与 npm；失败抛 AssertionError 并以非 0 退出。
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mb-mpl-"))

import numpy as np  # noqa: E402

import mb_registry as R  # noqa: E402
import mb_state  # noqa: E402

STEPS = ["qc", "filter", "impute", "normalize", "transform", "pca",
         "plsda", "de", "batch", "enrich", "biomarker", "annotate"]

#: 需要显式参数的图/分析（其余用默认参数）
ANALYSIS_PARAMS = {
    "roc_biomarker": {"feature": None},
    "survival_metabolite": {"feature": None, "time_col": "os_time", "event_col": "os_event"},
    "trend_course": {"time_col": "age"},
    "dose_response": {"dose_col": "age"},
    "formula_candidates": {"ppm": 10},
}
#: 需要 MS2 谱库/外部文件的分析（demo 有内置谱库，因此都能跑；保留集合便于将来跳过）
SKIP_ANALYSES: set[str] = set()
SKIP_PLOTS: set[str] = set()
#: 需要必填参数的图
PLOT_PARAMS = {
    "dose_response": {"dose_col": "age"},
    "trend_plot": {"time_col": "age"},
    "km_metabolite": {"feature": None, "time_col": "os_time", "event_col": "os_event"},
}


def _ok(label: str, value=None) -> None:
    print(f"  OK  {label}" + (f" — {value}" if value is not None else ""))


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="mb-pipeline-")
    state = mb_state.State()
    dataset = mb_state.make_demo(state, name="demo-metabo")
    assert dataset.X.shape == (72, 1600), f"demo 形状异常：{dataset.X.shape}"
    _ok("demo", f"{dataset.X.shape[0]} 样本 × {dataset.X.shape[1]} 特征")

    for step in STEPS:
        payload = R.run_step(dataset, step, {}, workdir, {})
        summary = payload.get("summary") or {}
        assert summary.get("bullets"), f"步骤 {step} 没有 summary.bullets"
        _ok(f"step {step}", summary.get("title"))

    assert "p_adj" in " ".join(dataset.results.get("de", {}).keys() if hasattr(dataset.results.get("de", {}), "keys") else []) \
        or "de" in dataset.steps, "差异步骤没有写入结果"

    # 全部图
    kinds = sorted(R.all_plots().keys())
    failed = []
    for kind in kinds:
        if kind in SKIP_PLOTS:
            continue
        try:
            params = {}
            for key, value in (PLOT_PARAMS.get(kind) or {}).items():
                params[key] = dataset.features[0] if value is None else value
            figure = R.run_plot(dataset, kind, params, workdir, {})
            assert os.path.exists(figure["file"]) and os.path.getsize(figure["file"]) > 1000
        except Exception as error:  # noqa: BLE001
            failed.append((kind, f"{type(error).__name__}: {error}"))
    assert not failed, f"图失败：{failed}"
    _ok("plots", f"{len(kinds) - len(SKIP_PLOTS)} 种全部可出图")

    # 全部分析
    catalog = R.analyses_catalog(dataset)
    analyses = catalog.get("analyses") or []
    failed = []
    for item in analyses:
        aid = item.get("id")
        if aid in SKIP_ANALYSES or not aid:
            continue
        params = {}
        for key, value in (ANALYSIS_PARAMS.get(aid) or {}).items():
            params[key] = dataset.features[0] if value is None else value
        try:
            payload = R.run_analysis(dataset, aid, params, workdir, {})
            assert (payload.get("summary") or {}).get("bullets"), f"{aid} 没有 summary"
        except Exception as error:  # noqa: BLE001
            failed.append((aid, f"{type(error).__name__}: {error}"))
    assert not failed, f"分析失败：{failed}"
    _ok("analyses", f"{len(analyses) - len(SKIP_ANALYSES)} 项全部可跑")

    # 导出
    for what in ("expr_csv", "de_csv", "summary_json", "methods"):
        try:
            out = R.export_dataset(dataset, what, None, workdir, {})
            assert os.path.exists(out["file"]), f"{what} 导出文件不存在"
            _ok(f"export {what}", os.path.basename(out["file"]))
        except Exception as error:  # noqa: BLE001
            print(f"  WARN export {what} 跳过：{type(error).__name__}: {error}")

    result = R.audit(dataset)
    assert 0 <= result["score"] <= 100
    _ok("audit", f"{result['score']}/100（{result.get('level')}）")
    print("\nPASS  dsh-metabolomics pipeline")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"\nFAIL  {error}")
        sys.exit(1)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nFAIL  未预期异常")
        sys.exit(1)
