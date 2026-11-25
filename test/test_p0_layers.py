#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — P0-1 回归测试：变换/标度三层分离与效应量口径

对应 BIO_P0_BACKLOG.md §4 P0-1：
  * `step_transform` 必须显式产出并保存三层 linear / log / model（含 transform/offset/base、
    center/denom、features/samples 快照与数组预算上限）；
  * `X_kind` 语义：有标度 → scaled；仅 log → lognorm；仅 cuberoot → transformed；
  * 已知二倍信号 log2FC = 1.0（从未标度层算），且切换 PCA scaling 不改变差异效应；
  * `detected_frac` 用 ctx["raw"] 的原始缺失掩码；
  * 无层可用的旧「标度回退」warning 仍然保留（mb_de 兜底路径）。

用法：cd plugins/dsh-metabolomics && MPLBACKEND=Agg python3 test/test_p0_layers.py
成功打印 PASS 并以 0 退出；失败打印 FAIL 并以 1 退出。
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mb-mpl-"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mb_de as D  # noqa: E402
import mb_prep as P  # noqa: E402
from mb_common import EngineError  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="mb-p0-layers-")
PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  PASS {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))


def out(tag: str) -> str:
    path = os.path.join(OUTDIR, tag)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# ctx 构造
# ---------------------------------------------------------------------------


def make_ctx(X, samples, groups=None, X_kind="raw", raw=None, results=None,
             group_key="group", features=None):
    arr = np.asarray(X, dtype=np.float64)
    n, p = arr.shape
    features = list(features) if features is not None else [f"F{j + 1:04d}" for j in range(p)]
    info = pd.DataFrame(index=pd.Index([str(s) for s in samples], name="sample"))
    if groups is not None:
        info["group"] = [str(g) for g in groups]
    feature_info = pd.DataFrame(
        {"name": [f"M{j + 1}" for j in range(p)], "mz": np.nan, "rt": np.nan},
        index=pd.Index(features, name="feature"),
    )
    return {
        "dataset_id": "ds_p0_layers",
        "name": "p0-layers",
        "features": features,
        "samples": [str(s) for s in samples],
        "X": np.asarray(arr, dtype=np.float32),
        "X_kind": X_kind,
        "raw": None if raw is None else np.asarray(raw, dtype=np.float32),
        "feature_info": feature_info,
        "sample_info": info,
        "group_key": group_key if groups is not None else None,
        "groups": sorted(set(map(str, groups))) if groups is not None else None,
        "matrix_order": "samples_by_features",
        "species": "human",
        "results": {} if results is None else results,
        "library": None,
        "rng": np.random.default_rng(0),
        "workdir": OUTDIR,
        "outdir": OUTDIR,
        "engine": {"python": sys.executable, "mode": "builtin"},
        "notes": [],
        "warnings": [],
    }


def two_group_data(seed: int = 7, n_per_group: int = 12, n_features: int = 60):
    """线性强度：对照 random，实验组前 10 个特征严格 = 2×对照（已知二倍信号）。"""
    rng = np.random.default_rng(seed)
    n_control = n_per_group
    n_case = n_per_group
    control = rng.uniform(2000.0, 80000.0, (n_control, n_features))
    case = rng.uniform(2000.0, 80000.0, (n_case, n_features))
    case[:, :10] = 2.0 * control[:, :10]
    linear = np.vstack([control, case])
    labels = ["control"] * n_control + ["case"] * n_case
    samples = [f"S{i + 1:02d}" for i in range(linear.shape[0])]
    return linear, labels, samples


# ---------------------------------------------------------------------------
# 1) step_transform 产出三层
# ---------------------------------------------------------------------------


def test_layers_saved() -> dict:
    print("\n[1] step_transform 保存 linear / log / model 三层")
    linear, labels, samples = two_group_data()
    ctx = make_ctx(linear, samples, groups=labels)
    try:
        payload = P.step_transform(ctx, {"transform": "log2", "scale": "auto"}, out("layers"))
    except Exception:  # noqa: BLE001
        check("step_transform 可跑", False, traceback.format_exc(limit=2))
        return {}
    check("step_transform 可跑", True)
    store = payload.get("store") or {}
    transform = store.get("transform") or {}
    layers = transform.get("layers")
    check("store['transform']['layers'] 存在", isinstance(layers, dict), type(layers).__name__)
    if not isinstance(layers, dict):
        return {"payload": payload, "ctx": ctx, "linear": linear}

    values = layers.get("values") or {}
    check("三层 values 键齐全（linear/log/model）",
          {"linear", "log", "model"} <= set(values.keys()), str(sorted(values.keys())))
    check("features 快照与 ctx 一致", list(layers.get("features") or []) == ctx["features"])
    check("samples 快照与 ctx 一致", list(layers.get("samples") or []) == ctx["samples"])

    spec = layers.get("transform") or {}
    check("transform 规格含 method/offset/base",
          {"method", "offset", "base"} <= set(spec.keys()), str(spec))
    check("transform.method = log2", str(spec.get("method")) == "log2", str(spec.get("method")))
    check("transform.base = 2", float(spec.get("base") or 0) == 2.0, str(spec.get("base")))

    center = layers.get("center")
    denom = layers.get("denom")
    check("标度参数 center/denom 已保存",
          center is not None and denom is not None
          and np.asarray(center).size == len(ctx["features"])
          and np.asarray(denom).size == len(ctx["features"]),
          f"center={None if center is None else np.shape(center)}, denom={None if denom is None else np.shape(denom)}")

    model = values.get("model")
    replace_X = np.asarray(payload.get("replace", {}).get("X"), dtype=np.float64)
    check("model 层 = replace['X']",
          model is not None and np.allclose(np.asarray(model, dtype=np.float64), replace_X,
                                            rtol=1e-5, atol=1e-6))
    log_layer = values.get("log")
    offset = float(spec.get("offset") or 0.0)
    expected_log = np.log2(np.asarray(linear, dtype=np.float64) + offset)
    check("log 层 ≈ log2(linear + offset)",
          log_layer is not None and np.allclose(np.asarray(log_layer, dtype=np.float64),
                                                expected_log, rtol=1e-5, atol=1e-5))
    lin_layer = values.get("linear")
    check("linear 层 = 输入线性矩阵",
          lin_layer is not None and np.allclose(np.asarray(lin_layer, dtype=np.float64),
                                                np.asarray(linear, dtype=np.float64),
                                                rtol=1e-6, atol=1e-6))

    budget = layers.get("budget") or {}
    check("数组预算字段存在（limit/per_layer/truncated）",
          {"limit_bytes", "per_layer_bytes", "truncated"} <= set(budget.keys()), str(budget))
    check("小数据未触发预算截断", bool(budget.get("truncated")) is False, str(budget.get("truncated")))
    return {"payload": payload, "ctx": ctx, "linear": linear}


# ---------------------------------------------------------------------------
# 2) X_kind 语义
# ---------------------------------------------------------------------------


def test_kind_semantics() -> None:
    print("\n[2] X_kind 语义（scaled / lognorm / transformed）")
    linear, labels, samples = two_group_data(seed=11)
    cases = [
        ("log2", "none", "lognorm"),
        ("log10", "none", "lognorm"),
        ("cuberoot", "none", "transformed"),
        ("none", "auto", "scaled"),
        ("log2", "pareto", "scaled"),
        ("log2", "auto", "scaled"),
    ]
    for transform, scale, expect in cases:
        ctx = make_ctx(linear, samples, groups=labels)
        try:
            payload = P.step_transform(ctx, {"transform": transform, "scale": scale},
                                       out(f"kind-{transform}-{scale}"))
            got = payload.get("replace", {}).get("X_kind")
            check(f"transform={transform} scale={scale} → X_kind={expect}", got == expect, str(got))
            layers = (payload.get("store") or {}).get("transform", {}).get("layers") or {}
            check(f"  · {transform}/{scale} 仍保存三层",
                  isinstance(layers.get("values"), dict)
                  and layers.get("values", {}).get("model") is not None)
        except Exception:  # noqa: BLE001
            check(f"transform={transform} scale={scale} 可跑", False, traceback.format_exc(limit=2))

    ctx = make_ctx(linear, samples, groups=labels)
    payload = P.step_transform(ctx, {"transform": "none", "scale": "none"}, out("kind-none"))
    check("transform=none scale=none 不写 X_kind", "X_kind" not in (payload.get("replace") or {}),
          str(payload.get("replace", {}).get("X_kind")))
    layers = (payload.get("store") or {}).get("transform", {}).get("layers") or {}
    values = layers.get("values") or {}
    check("none/none 时 linear=log=model 三层仍齐全",
          values.get("linear") is not None and values.get("log") is not None
          and values.get("model") is not None)


# ---------------------------------------------------------------------------
# 3) 二倍信号 log2FC = 1.0；切换 scaling 不改变差异效应
# ---------------------------------------------------------------------------


def _de_on_transformed(linear, labels, samples, transform, scale, tag):
    ctx = make_ctx(linear, samples, groups=labels)
    payload = P.step_transform(ctx, {"transform": transform, "scale": scale}, out(f"de-{tag}"))
    new_X = payload["replace"]["X"]
    new_kind = payload["replace"].get("X_kind", "raw")
    de_ctx = make_ctx(new_X, samples, groups=labels, X_kind=new_kind,
                      results={"transform": payload["store"]["transform"]})
    return D.step_de(de_ctx, {"case": "case", "control": "control", "fc": 1.5, "fdr": 0.05},
                     out(f"de-run-{tag}"))


def test_log2fc_on_layers(shared: dict) -> None:
    print("\n[3] 二倍信号 log2FC = 1.0（从未标度层计算）")
    if not shared:
        check("依赖 [1] 的 layer 数据", False, "skipped")
        return
    linear = shared["linear"]
    labels = ["control"] * 12 + ["case"] * 12
    samples = [f"S{i + 1:02d}" for i in range(24)]
    for scale in ("auto", "pareto", "none"):
        try:
            payload = _de_on_transformed(linear, labels, samples, "log2", scale, f"log2-{scale}")
        except Exception:  # noqa: BLE001
            check(f"scale={scale} 差异可跑", False, traceback.format_exc(limit=3))
            continue
        frame = payload["store"]["de"]["table"]
        got = pd.to_numeric(frame.set_index("feature")["log2FC"], errors="coerce")
        first = [got[f"F{j + 1:04d}"] for j in range(10)]
        ok = all(abs(float(v) - 1.0) < 1e-4 for v in first)
        check(f"scale={scale}：10 个 2× 特征 log2FC=1.0", ok,
              ", ".join(f"{float(v):.6f}" for v in first[:3]))
        mean_case = pd.to_numeric(frame.set_index("feature")["mean_case"], errors="coerce")
        check(f"scale={scale}：mean_case 来自未标度层（非标度单位）",
              float(mean_case["F0001"]) > 1.0, f"{float(mean_case['F0001']):.4f}")
        basis = str(payload["store"]["de"].get("log2fc_basis") or "")
        check(f"scale={scale}：log2FC 口径标注为未标度层", "未标度" in basis or "log" in basis, basis)

    # 切换 scaling：auto vs pareto 的 log2FC 完全一致
    try:
        fa = _de_on_transformed(linear, labels, samples, "log2", "auto", "cmp-auto")["store"]["de"]["table"]
        fp = _de_on_transformed(linear, labels, samples, "log2", "pareto", "cmp-pareto")["store"]["de"]["table"]
        va = pd.to_numeric(fa.set_index("feature")["log2FC"].reindex(fa["feature"]), errors="coerce")
        vp = pd.to_numeric(fp.set_index("feature")["log2FC"].reindex(fa["feature"]), errors="coerce")
        check("切换 PCA scaling（auto→pareto）不改变 log2FC",
              bool(np.allclose(va.to_numpy(float), vp.to_numpy(float), atol=1e-9, equal_nan=True)),
              f"max|Δ|={np.nanmax(np.abs(va.to_numpy(float) - vp.to_numpy(float))):.3g}")
    except Exception:  # noqa: BLE001
        check("切换 PCA scaling 不改变 log2FC", False, traceback.format_exc(limit=2))

    # 无 layer 的旧标度矩阵：保留回退 warning
    rng = np.random.default_rng(3)
    Xs = rng.normal(size=(24, 30))
    legacy = make_ctx(Xs, samples, groups=labels, X_kind="scaled")
    try:
        res = D.step_de(legacy, {"case": "case", "control": "control"}, out("legacy-scaled"))
        check("无层可用的 scaled 矩阵 → 保留标度回退 warning",
              any("标度" in str(w) for w in res.get("warnings") or []),
              str(res.get("warnings"))[:120])
        check("无层可用时 log2FC 口径标注 scaled-difference",
              "scaled" in str(res["store"]["de"].get("log2fc_basis")),
              str(res["store"]["de"].get("log2fc_basis")))
    except Exception:  # noqa: BLE001
        check("无层可用的 scaled 矩阵可跑", False, traceback.format_exc(limit=2))


# ---------------------------------------------------------------------------
# 4) detected_frac 用 raw 缺失掩码
# ---------------------------------------------------------------------------


def test_detected_fraction() -> None:
    print("\n[4] detected_frac 来自 ctx['raw'] 的原始缺失掩码")
    linear, labels, samples = two_group_data(seed=21, n_per_group=12, n_features=30)
    X = linear.copy()
    raw = linear.copy()
    # 特征 5：case 组（后 12 行）前 4 个样本原始缺失，随后被填补（X 无 NaN）
    raw[12:16, 5] = np.nan
    X[12:16, 5] = 0.5 * np.nanmin(linear[:, 5])
    ctx = make_ctx(X, samples, groups=labels, X_kind="lognorm", raw=raw)
    res = D.step_de(ctx, {"case": "case", "control": "control"}, out("detected"))
    frame = res["store"]["de"]["table"].set_index("feature")
    got = float(frame.loc["F0006", "detected_frac"])
    expected = (12 - 4 + 12) / 24.0
    check("detected_frac = 原始检出比例（含 raw NaN）", abs(got - expected) < 1e-9,
          f"{got:.6f} vs {expected:.6f}")
    other = float(frame.loc["F0001", "detected_frac"])
    check("无原始缺失的特征 detected_frac = 1", abs(other - 1.0) < 1e-9, f"{other:.6f}")

    # raw 形状不匹配时回退到当前矩阵并在 notes/warnings 里说明
    bad_ctx = make_ctx(X, samples, groups=labels, X_kind="lognorm",
                       raw=np.ones((4, 3), dtype=np.float32))
    try:
        res2 = D.step_de(bad_ctx, {"case": "case", "control": "control"}, out("detected-bad"))
        check("raw 形状不匹配时回退当前矩阵（不崩）",
              float(pd.to_numeric(res2["store"]["de"]["table"]["detected_frac"], errors="coerce")
                    .min()) == 1.0)
    except Exception:  # noqa: BLE001
        check("raw 形状不匹配时回退当前矩阵（不崩）", False, traceback.format_exc(limit=2))


# ---------------------------------------------------------------------------
# 5) 三层数组预算 + 只存 model 层时的反演
# ---------------------------------------------------------------------------


def test_layer_budget() -> None:
    print("\n[5] 三层数组预算：超预算不伪造，truncated=True；只存 model 层可反演")
    linear, labels, samples = two_group_data(seed=31, n_per_group=12, n_features=60)
    ctx = make_ctx(linear, samples, groups=labels)
    payload = P.step_transform(
        ctx, {"transform": "log2", "scale": "auto", "layer_budget_bytes": 10000}, out("budget"))
    layers = (payload.get("store") or {}).get("transform", {}).get("layers") or {}
    budget = layers.get("budget") or {}
    check("budget 字段名为 limit_bytes / per_layer_bytes / truncated",
          {"limit_bytes", "per_layer_bytes", "truncated"} <= set(budget.keys()), str(budget))
    check("小预算触发 truncated=True", bool(budget.get("truncated")) is True, str(budget))
    values = layers.get("values") or {}
    stored = layers.get("stored") or []
    check("超预算时只保存放得下的层（model 优先），其余为 None 而不是伪造",
          values.get("model") is not None and values.get("linear") is None
          and values.get("log") is None and stored == ["model"],
          f"stored={stored}")
    check("预算截断写 warning", any("预算" in str(w) for w in payload.get("warnings") or []),
          str(payload.get("warnings"))[:120])

    # 只存 model 层时，mb_de 必须能从 model 反演出未标度 log 层，效果量与完整层一致
    try:
        de_ctx = make_ctx(payload["replace"]["X"], samples, groups=labels, X_kind="scaled",
                          results={"transform": payload["store"]["transform"]})
        res = D.step_de(de_ctx, {"case": "case", "control": "control"}, out("budget-de"))
        frame = res["store"]["de"]["table"].set_index("feature")
        got = [float(frame.loc[f"F{j + 1:04d}", "log2FC"]) for j in range(10)]
        check("仅存 model 层时反演未标度层，2× 特征 log2FC 仍 ≈1.0",
              all(abs(value - 1.0) < 1e-3 for value in got),
              ", ".join(f"{value:.6f}" for value in got[:3]))
        check("反演口径同样标注未标度层",
              "未标度" in str(res["store"]["de"].get("log2fc_basis")),
              str(res["store"]["de"].get("log2fc_basis")))
    except Exception:  # noqa: BLE001
        check("仅存 model 层时反演未标度层", False, traceback.format_exc(limit=2))



# ---------------------------------------------------------------------------
# 6) 真实 registry 链路：三层随 store 进入 dataset.results，de 走同一口径
# ---------------------------------------------------------------------------


def test_registry_integration() -> None:
    print("\n[6] registry 集成：三层经 run_step 写入 results，de 从层取效应量")
    try:
        import mb_registry as registry
    except Exception:  # noqa: BLE001
        check("registry 可导入", False, traceback.format_exc(limit=2))
        return
    from types import SimpleNamespace

    linear, labels, samples = two_group_data(seed=41)
    ctx = make_ctx(linear, samples, groups=labels)
    workdir = out("registry-work")
    dataset = SimpleNamespace(
        id="ds_p0_layers_reg", name="p0-layers-reg", path=None, source_format="matrix_text",
        X=np.asarray(ctx["X"], dtype=np.float32).copy(),
        raw=np.asarray(ctx["X"], dtype=np.float32).copy(),
        X_kind="raw", value_kind="raw",
        features=list(ctx["features"]), genes=list(ctx["features"]), samples=list(ctx["samples"]),
        feature_info=ctx["feature_info"].copy(), sample_info=ctx["sample_info"].copy(),
        matrix_order="samples_by_features", species="human", polarity="pos", library=None,
        group_key="group", groups=list(labels and sorted(set(labels))),
        steps=["normalize"], history=[], analyses_run=[], results={},
        notes=[], warnings=[], current="normalize",
    )
    try:
        registry.run_step(dataset, "transform", {"transform": "log2", "scale": "auto"}, workdir)
    except Exception:  # noqa: BLE001
        check("registry transform 可跑", False, traceback.format_exc(limit=2))
        return
    check("registry transform 可跑", True)
    store = (dataset.results or {}).get("transform") or {}
    layers = store.get("layers") if isinstance(store, dict) else None
    values = (layers or {}).get("values") or {}
    log_layer = values.get("log")
    check("dataset.results['transform']['layers'] 在 registry 路径下保留数组",
          layers is not None and log_layer is not None and np.asarray(log_layer).ndim == 2,
          f"keys={sorted((layers or {}).keys())[:6]}")
    check("registry 后 X_kind = scaled", str(dataset.X_kind) == "scaled", str(dataset.X_kind))
    try:
        payload = registry.run_step(dataset, "de", {"case": "case", "control": "control",
                                                    "fc": 1.5, "fdr": 0.05}, workdir)
    except Exception:  # noqa: BLE001
        check("registry de 可跑", False, traceback.format_exc(limit=2))
        return
    check("registry de 可跑", True)
    metrics = (payload.get("summary") or {}).get("metrics") or {}
    check("registry de 口径 = 未标度层",
          "未标度" in str(metrics.get("log2fc_basis")), str(metrics.get("log2fc_basis")))
    frame = ((dataset.results or {}).get("de") or {}).get("table")
    try:
        got = [float(frame.set_index("feature").loc[f"F{j + 1:04d}", "log2FC"]) for j in range(10)]
        check("registry 链路下 2× 特征 log2FC=1.0",
              all(abs(value - 1.0) < 1e-4 for value in got),
              ", ".join(f"{value:.6f}" for value in got[:3]))
    except Exception:  # noqa: BLE001
        check("registry 链路下 2× 特征 log2FC=1.0", False, traceback.format_exc(limit=2))


# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("P0-1 回归：变换/标度三层分离 + 效应量口径")
    print("=" * 78)
    shared = test_layers_saved()
    test_kind_semantics()
    test_log2fc_on_layers(shared)
    test_detected_fraction()
    test_layer_budget()
    test_registry_integration()
    print("\n" + "=" * 78)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
