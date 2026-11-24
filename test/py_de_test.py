#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — mb_de（单变量差异 / 多组 / 趋势 / 剂量）回归测试

覆盖（DESIGN §3 step de / §4 差异与趋势 / §5 图 / §11 mb-stat 验收）：
  * 60 样本 × 300 特征、两组各 30、注入 80 个真差异特征（|log2FC| 0.8–1.5）、5% NaN 已填补；
  * Welch 差异 top100 命中率 ≥ 60%、BH 单调（p_adj ≥ p）、火山图点数 = 表行数（300）；
  * wilcoxon / limma / paired 分支可跑；不满足配对条件 → prerequisite；
  * 多组 ANOVA（+Tukey/ Games-Howell / Dunn 事后）、时间趋势、4PL 剂量响应分支可跑；
  * 5 个 PLOTS（volcano_de / fc_bar / heatmap_top / trend_plot / dose_response）都能出图且 size > 1000。

用法：cd plugins/dsh-metabolomics && python3 test/py_de_test.py
成功打印 PASS 并以 0 退出；失败打印 FAIL 并以 1 退出。
"""

from __future__ import annotations

import math
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
from scipy import stats  # noqa: E402

import mb_de as D  # noqa: E402
from mb_common import EngineError  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="mb-de-test-")
PASSED: list[str] = []
FAILED: list[str] = []

DOSE_LEVELS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
TIME_LEVELS = [0.0, 6.0, 24.0]
TRUE_EC50 = 3.0
TRUE_HILL = 1.5


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
# 合成数据
# ---------------------------------------------------------------------------


def build_ctx(seed: int = 5, n_per_group: int = 30, n_features: int = 300,
              n_diff: int = 80, fill_nan: bool = True, n_groups: int = 2,
              group_names: tuple = ("control", "case")) -> tuple:
    rng = np.random.default_rng(seed)
    names = list(group_names)[:n_groups]
    p = n_features
    labels = np.array([g for g in names for _ in range(n_per_group)], dtype=object)
    n = labels.size
    samples = [f"S{i + 1:02d}" for i in range(n)]
    case_rows = labels == names[-1]

    mzs = np.round(rng.uniform(100.0, 900.0, p), 4)
    rts = np.round(rng.uniform(0.5, 18.0, p), 2)
    features = [f"mz{mz:.4f}_rt{rt:.2f}" for mz, rt in zip(mzs, rts)]
    feature_info = pd.DataFrame(
        {"mz": mzs, "rt": rts, "name": [f"Metabolite_{i + 1}" for i in range(p)],
         "class": ["Organic acid"] * p},
        index=pd.Index(features, name="feature"),
    )

    base = rng.normal(10.0, 1.2, p)
    subject_effect = rng.normal(0.0, 0.45, n_per_group)
    subject_full = np.concatenate([subject_effect for _ in range(len(names))])
    X = base[None, :] + subject_full[:, None] + rng.normal(0.0, 0.35, (n, p))

    # 注入差异特征（case 组：一半上调、一半下调）
    diff_idx = rng.choice(p, size=min(n_diff, p), replace=False)
    half = max(n_diff // 2, 1)
    log2fc = np.zeros(p)
    log2fc[diff_idx[:half]] = rng.uniform(0.8, 1.5, min(half, diff_idx.size))
    log2fc[diff_idx[half:]] = -rng.uniform(0.8, 1.5, max(diff_idx.size - half, 0))
    X[case_rows] += log2fc[None, :]

    # 时间趋势特征（30 个）+ 剂量响应特征（1 个 4PL 响应）
    time_values = np.array([TIME_LEVELS[i % len(TIME_LEVELS)] for i in range(n)], dtype=float)
    trend_idx = rng.choice(np.setdiff1d(np.arange(p), diff_idx), size=min(30, p - diff_idx.size),
                           replace=False)
    for j, feature_idx in enumerate(trend_idx):
        slope = rng.uniform(0.02, 0.06) * (1 if j % 2 == 0 else -1)
        X[:, feature_idx] += slope * time_values
    dose_values = np.array([DOSE_LEVELS[i % len(DOSE_LEVELS)] for i in range(n)], dtype=float)
    dose_idx = int(rng.choice(np.setdiff1d(np.arange(p), np.concatenate([diff_idx, trend_idx]))))
    response = 2.0 + (10.0 - 2.0) / (1.0 + (dose_values / TRUE_EC50) ** TRUE_HILL)
    X[:, dose_idx] += (response - response.mean()) * 1.5
    noise_only_idx = int(np.setdiff1d(np.arange(p), np.concatenate([diff_idx, trend_idx,
                                                                   [dose_idx]]))[0])

    # 5% NaN 后按 half-min 填补（模拟 impute 步骤已完成）
    mask = rng.random((n, p)) < 0.05
    X[mask] = np.nan
    if fill_nan:
        col_min = np.nanmin(X, axis=0)
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = col_min[cols] * 0.5
    X[:, noise_only_idx] = 5.0  # 常数特征（放在填补之后，保证真正常数 → 4PL 应给可读错误）

    subject_ids = [f"P{i + 1:02d}" for i in range(n_per_group)] * len(names)
    sample_info = pd.DataFrame(
        {"group": labels, "batch": ["B1" if i % 2 else "B2" for i in range(n)],
         "subject": subject_ids, "time": time_values, "dose": dose_values,
         "age": np.round(rng.normal(50, 8, n), 1)},
        index=pd.Index(samples, name="sample"),
    )
    ctx = {
        "dataset_id": "ds_de",
        "name": "synthetic-de",
        "features": features,
        "samples": samples,
        "X": np.asarray(X, dtype=np.float32),
        "X_kind": "lognorm",
        "raw": None,
        "feature_info": feature_info,
        "sample_info": sample_info,
        "group_key": "group",
        "groups": names,
        "matrix_order": "samples_by_features",
        "species": "human",
        "results": {},
        "library": None,
        "rng": np.random.default_rng(0),
        "workdir": OUTDIR,
        "outdir": OUTDIR,
        "engine": {"python": sys.executable, "mode": "builtin"},
        "notes": [],
        "warnings": [],
    }
    injected = {features[i] for i in diff_idx.tolist()}
    injected_up = {features[i] for i in diff_idx[:half].tolist()}
    injected_down = {features[i] for i in diff_idx[half:].tolist()}
    extra = {"dose_feature": features[dose_idx], "constant_feature": features[noise_only_idx],
             "trend_features": [features[i] for i in trend_idx]}
    return ctx, features, injected, injected_up, injected_down, extra


def run_step(ctx, params, tag):
    return D.step_de(ctx, params, out(tag))


def figure_ok(fig) -> bool:
    path = (fig or {}).get("file")
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# 1) 注册表结构
# ---------------------------------------------------------------------------


def test_catalog():
    print("\n[1] 注册表结构")
    check("STEP 只有 de", [s["id"] for s in D.STEPS] == ["de"], str([s["id"] for s in D.STEPS]))
    step = D.STEPS[0]
    check("step.requires == ['transform']", step["requires"] == ["transform"])
    params = {p["key"] for p in step["params"]}
    check("step 参数齐全", {"group_col", "case", "control", "method", "paired", "fdr", "fc"} <= params,
          str(sorted(params)))
    analyses = {a["id"]: a for a in D.ANALYSES}
    expect = ["univariate_de", "volcano_de", "fc_bar", "heatmap_top", "multi_group_de",
              "trend_course", "dose_response"]
    check("7 个分析注册", list(analyses) == expect, str(list(analyses)))
    check("分析 group 全为差异与趋势", all(a["group"] == "差异与趋势" for a in D.ANALYSES))
    check("needs 符合契约",
          analyses["univariate_de"]["needs"] == ["groups"]
          and analyses["volcano_de"]["needs"] == ["de"]
          and analyses["dose_response"]["needs"] == ["groups"])
    check("5 个 PLOTS 归属正确（de_heatmap_top 归本模块）",
          set(D.PLOTS) == {"volcano_de", "fc_bar", "de_heatmap_top", "trend_plot", "dose_response"},
          str(sorted(D.PLOTS)))
    check("不抢占 mb_plots 的 heatmap_top kind",
          "heatmap_top" not in D.PLOTS and "de_heatmap_top" in D.PLOTS, str(sorted(D.PLOTS)))
    check("PLOTS 都可调用", all(callable(f) for f in D.PLOTS.values()))


# ---------------------------------------------------------------------------
# 2) 统计核心自检
# ---------------------------------------------------------------------------


def test_math():
    print("\n[2] 统计核心自检")
    rng = np.random.default_rng(9)
    A = rng.normal(1.0, 0.5, (20, 40))
    B = rng.normal(0.8, 0.6, (22, 40))
    t, p, _df = D._welch_core(A, B)
    ref = stats.ttest_ind(A, B, axis=0, equal_var=False)
    check("Welch t 与 scipy 一致", np.allclose(t, ref.statistic, rtol=1e-9, atol=1e-9),
          f"max|dt|={np.nanmax(np.abs(t - ref.statistic)):.2e}")
    check("Welch p 与 scipy 一致", np.allclose(p, ref.pvalue, rtol=1e-8, atol=1e-12),
          f"max|dp|={np.nanmax(np.abs(p - ref.pvalue)):.2e}")
    t2, p2, label = D._welch_p(A, B, "case", "control")
    check("无缺失时走共享实现 welch_de", "mb_common" in label, label)
    check("共享实现与本模块一致", np.allclose(p, p2, rtol=1e-8, atol=1e-12))

    # nan-aware：把部分值设为 NaN 后仍能出有限 p
    A2 = A.copy()
    A2[0, :10] = np.nan
    t3, p3, _ = D._welch_core(A2, B)
    check("含缺失仍能算 Welch", int(np.isfinite(p3).sum()) == 40, str(int(np.isfinite(p3).sum())))

    # Mann-Whitney 与 scipy 一致
    z, pm = D._mannwhitney(A[:, :20], B[:, :20])
    ref_mw = [stats.mannwhitneyu(A[:, j], B[:, j], alternative="two-sided").pvalue for j in range(20)]
    check("Mann-Whitney p 与 scipy 一致", np.allclose(pm, ref_mw, rtol=1e-9, atol=1e-12))

    # limma moderated t
    t_l, p_l, info = D._limma(A, B)
    check("limma 输出 p ∈ [0,1]", bool(np.nanmin(p_l) >= 0 and np.nanmax(p_l) <= 1))
    check("limma 估计 d0 有限", np.isfinite(info["d0"]) and info["d0"] > 0, f"d0={info['d0']:.3g}")
    check("limma 与普通 t 同向（相关 > 0.9）",
          float(np.corrcoef(t_l, t)[0, 1]) > 0.9, f"{float(np.corrcoef(t_l, t)[0, 1]):.3f}")

    # 差值检验：配对 t
    base = rng.normal(5.0, 1.0, (12, 15))
    d_case = base + rng.normal(0.7, 0.2, (12, 15))
    stat, pp = D._paired_core(d_case, base, "t")
    check("配对 t 检出方向正确", float(np.nanmean(stat)) > 0, f"mean t={float(np.nanmean(stat)):.2f}")

    # 4PL 恢复已知参数
    dose = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0] * 4)
    y = 2.0 + (10.0 - 2.0) / (1.0 + (dose / TRUE_EC50) ** TRUE_HILL) + rng.normal(0, 0.05, dose.size)
    fit = D._fit_4pl(dose, y)
    check("4PL 恢复 EC50", abs(fit["ec50"] - TRUE_EC50) / TRUE_EC50 < 0.2, f"EC50={fit['ec50']:.3f}")
    check("4PL 恢复 Hill 斜率", abs(fit["hill"] - TRUE_HILL) < 0.4, f"hill={fit['hill']:.3f}")
    check("4PL R² > 0.98", fit["r2"] > 0.98, f"{fit['r2']:.4f}")
    # 真值 y = 2 + 8/(1+(dose/3)^1.5) 随剂量下降（低剂量高、高剂量低）→ 方向应为 decreasing
    check("4PL 方向判定", fit["direction"] == "decreasing", fit["direction"])
    try:
        D._fit_4pl(dose, np.full(dose.size, 5.0))
        check("常数响应 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("常数响应 → prerequisite", error.code == "prerequisite", f"{error.code}")

    # 4PL 数值稳健性：极端参数/极端指数不得抛 OverflowError（标量与数组都测）
    for t in (0.0, np.array([-1e6, 0.0, 1e6])):
        value = D._four_pl(t, 0.0, 1.0, 1e6, -1e6)
        check("4PL 极端指数不溢出", bool(np.all(np.isfinite(value))), str(value)[:40])
    check("4PL 标量输入返回有限值",
          math.isfinite(float(D._four_pl(-1e7, 0.0, 1.0, 1e6, 1e6))))


# ---------------------------------------------------------------------------
# 3) STEP de（Welch）
# ---------------------------------------------------------------------------


def test_step_de():
    print("\n[3] STEP de（Welch）")
    ctx, features, injected, up_set, down_set, extra = build_ctx()
    res = run_step(ctx, {"method": "welch", "fdr": 0.05, "fc": 1.5}, "step-welch")
    frame = res["store"]["de"]["table"]
    metrics = res["summary"]["metrics"]
    print(f"    metrics: up={metrics['n_up']} down={metrics['n_down']} tested={metrics['n_tested']} "
          f"method={metrics['method']}")

    check("表行数=特征数", len(frame) == len(features), f"{len(frame)}")
    check("store 结构符合 schema",
          res["store"]["de"]["table"] is res["store"]["de"]["de_table"]
          and isinstance(res["store"]["de"]["up"], list)
          and res["store"]["de"]["method"] == "welch (mb_common.welch_de)",
          str(res["store"]["de"]["method"]))
    check("表列符合契约（前 12 列）",
          [c["key"] for c in res["tables"][0]["columns"]] ==
          ["feature", "name", "mz", "rt", "log2FC", "mean_case", "mean_control",
           "p", "p_adj", "regulation", "fc_hit", "detected_frac"],
          str([c["key"] for c in res["tables"][0]["columns"]]))
    check("regulation 取值合法", set(frame["regulation"]) <= {"up", "down", "ns"},
          str(set(frame["regulation"])))
    check("上调/下调都能检出", metrics["n_up"] > 0 and metrics["n_down"] > 0,
          f"{metrics['n_up']}/{metrics['n_down']}")
    check("log2FC 口径 = log2-mean-difference", metrics["log2fc_basis"] == "log2-mean-difference",
          str(metrics["log2fc_basis"]))
    check("FDR/FC 设置生效", metrics["fdr"] == 0.05 and metrics["fc"] == 1.5)

    p = pd.to_numeric(frame["p"], errors="coerce").to_numpy()
    padj = pd.to_numeric(frame["p_adj"], errors="coerce").to_numpy()
    finite = np.isfinite(p) & np.isfinite(padj)
    check("BH 单调：p_adj ≥ p", bool(np.all(padj[finite] >= p[finite] - 1e-12)),
          f"违例 {int(np.sum(padj[finite] < p[finite] - 1e-12))}")
    check("BH 不超 1", bool(np.all(padj[finite] <= 1.0 + 1e-12)))
    check("全部特征有效检验", int(finite.sum()) == len(frame), f"{int(finite.sum())}/{len(frame)}")
    check("按 FDR 升序排序", bool(np.all(np.diff(padj[finite]) >= -1e-12)))

    top100 = set(frame.head(100)["feature"].astype(str))
    hit = len(top100 & injected) / 100.0
    check("Welch top100 命中注入差异特征 ≥ 60%", hit >= 0.6, f"hit={hit:.2f}")

    sig = frame[frame["regulation"] != "ns"]
    up_hits = set(sig["feature"].astype(str)) & up_set
    down_hits = set(sig["feature"].astype(str)) & down_set
    up_correct = float(np.mean([frame.loc[frame["feature"] == f, "regulation"].iloc[0] == "up"
                                for f in up_hits])) if up_hits else 0.0
    down_correct = float(np.mean([frame.loc[frame["feature"] == f, "regulation"].iloc[0] == "down"
                                  for f in down_hits])) if down_hits else 0.0
    check("显著特征方向正确（up）", up_correct >= 0.9, f"{up_correct:.2f} ({len(up_hits)} 个)")
    check("显著特征方向正确（down）", down_correct >= 0.9, f"{down_correct:.2f} ({len(down_hits)} 个)")
    check("fc_hit 与 |log2FC| 一致",
          bool(np.all(np.asarray(frame["fc_hit"]) ==
                      (np.abs(pd.to_numeric(frame["log2FC"], errors="coerce").to_numpy()) >= np.log2(1.5)))))

    # 图：火山图点数 = 表行数
    ids = [f["id"] for f in res["figures"]]
    check("步骤出 3 张图（volcano/fc/heatmap）", len(res["figures"]) == 3, str(ids))
    check("图文件都 > 1000 字节", all(figure_ok(f) for f in res["figures"]))
    volcano = next((f for f in res["figures"] if "volcano" in str(f.get("file"))), None)
    check("火山图点数 = 表行数", volcano is not None and volcano.get("n_points") == len(frame),
          f"n_points={volcano.get('n_points') if volcano else None}, rows={len(frame)}")
    check("火山图 n_points 与 result 表 total 一致",
          volcano.get("n_points") == res["tables"][0]["total"])

    # data_out
    data_out = res.get("data_out") or []
    names = [os.path.basename(d["path"]) for d in data_out]
    check("data_out 有 de_all.csv / de_sig.csv",
          "de_all.csv" in names and "de_sig.csv" in names, str(names))
    all_rows = next((d for d in data_out if d["path"].endswith("de_all.csv")), None)
    check("de_all.csv 行数=特征数", all_rows is not None and len(pd.read_csv(all_rows["path"])) == len(features))

    # 确定性 + 显式 case/control 翻转方向
    res2 = run_step(ctx, {"method": "welch", "fdr": 0.05, "fc": 1.5}, "step-welch-2")
    check("重复运行结果一致", np.allclose(
        pd.to_numeric(res2["store"]["de"]["table"]["p"], errors="coerce").to_numpy(),
        p, equal_nan=True))
    res3 = run_step(ctx, {"method": "welch", "case": "control", "control": "case",
                          "fdr": 0.05, "fc": 1.5}, "step-welch-swap")
    swapped = res3["store"]["de"]["table"].set_index("feature")
    orig = frame.set_index("feature")
    common = swapped.index[:50]
    check("交换 case/control 后 log2FC 变号",
          np.allclose(swapped.loc[common, "log2FC"], -orig.loc[common, "log2FC"], atol=1e-9))
    check("自动推断对照组 = control", metrics["contrast"].startswith("case vs control"), metrics["contrast"])
    return ctx, res


# ---------------------------------------------------------------------------
# 4) 其他方法：wilcoxon / limma / paired
# ---------------------------------------------------------------------------


def test_other_methods(ctx):
    print("\n[4] wilcoxon / limma / paired 分支")
    res = run_step(ctx, {"method": "wilcoxon", "fdr": 0.05, "fc": 1.5}, "step-wilcoxon")
    frame = res["store"]["de"]["table"]
    check("wilcoxon 可跑", len(frame) == len(ctx["features"]), f"{len(frame)}")
    check("wilcoxon 有显著特征", res["summary"]["metrics"]["n_up"] + res["summary"]["metrics"]["n_down"] > 0,
          str(res["summary"]["metrics"]))
    check("wilcoxon 方法标注", "mann-whitney" in str(res["store"]["de"]["method"]),
          str(res["store"]["de"]["method"]))

    res = run_step(ctx, {"method": "limma", "fdr": 0.05, "fc": 1.5}, "step-limma")
    frame = res["store"]["de"]["table"]
    p = pd.to_numeric(frame["p"], errors="coerce").to_numpy()
    padj = pd.to_numeric(frame["p_adj"], errors="coerce").to_numpy()
    finite = np.isfinite(p)
    check("limma 可跑", int(finite.sum()) == len(frame), f"{int(finite.sum())}/{len(frame)}")
    check("limma p ∈ [0,1]", bool(np.all((p[finite] >= 0) & (p[finite] <= 1))))
    check("limma p_adj ≥ p", bool(np.all(padj[finite] >= p[finite] - 1e-12)))
    check("limma 有显著特征", res["summary"]["metrics"]["n_up"] > 5,
          str(res["summary"]["metrics"]["n_up"]))
    check("limma 给出近似说明", any("矩估计" in w for w in res["warnings"]), str(res["warnings"])[:120])

    res = run_step(ctx, {"method": "welch", "paired": True, "fdr": 0.05, "fc": 1.5}, "step-paired")
    check("paired 可跑（有 subject 列）", "paired" in str(res["store"]["de"]["method"]),
          str(res["store"]["de"]["method"]))
    check("paired 记录 paired=True", bool(res["store"]["de"]["paired"]))
    check("paired 有显著特征", res["summary"]["metrics"]["n_up"] > 0,
          str(res["summary"]["metrics"]["n_up"]))

    # 不满足配对条件：去掉配对列 + 组间样本不等
    bad = dict(ctx)
    info = ctx["sample_info"].drop(columns=["subject"]).copy()
    bad["sample_info"] = info
    bad["X"] = ctx["X"][:-3]
    bad["samples"] = ctx["samples"][:-3]
    bad["feature_info"] = ctx["feature_info"]
    try:
        run_step(bad, {"paired": True}, "step-paired-bad")
        check("无配对列且样本数不等 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("无配对列且样本数不等 → prerequisite", error.code == "prerequisite",
              f"{error.code}: {error.message[:60]}")

    # 非法方法
    try:
        run_step(ctx, {"method": "randomforest"}, "step-bad-method")
        check("非法 method → bad_request", False, "没有抛错")
    except EngineError as error:
        check("非法 method → bad_request", error.code == "bad_request", error.code)


# ---------------------------------------------------------------------------
# 5) 组解析边界
# ---------------------------------------------------------------------------


def test_group_edges():
    print("\n[5] 组解析边界")
    ctx, _f, _i, _u, _d, _e = build_ctx(seed=31, n_per_group=20)

    one = dict(ctx)
    info = ctx["sample_info"].copy()
    info["group"] = "case"
    one["sample_info"] = info
    try:
        D.step_de(one, {}, out("edge-one-group"))
        check("单组 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("单组 → prerequisite", error.code == "prerequisite", error.code)

    no_group = dict(ctx)
    no_group["group_key"] = None
    no_group["sample_info"] = ctx["sample_info"].drop(columns=["group"])
    try:
        D.step_de(no_group, {}, out("edge-no-group"))
        check("无分组列 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("无分组列 → prerequisite", error.code == "prerequisite", error.code)

    # 三组但未指定 case/control → prerequisite（提示用多组 ANOVA）
    multi, _f, _i, _u, _d, _e = build_ctx(seed=33, n_per_group=15, n_groups=3,
                                          group_names=("control", "low", "high"))
    try:
        D.step_de(multi, {}, out("edge-three-group"))
        check("三组未指定 case/control → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("三组未指定 case/control → prerequisite", error.code == "prerequisite", error.code)
    res = D.step_de(multi, {"case": "high", "control": "control"}, out("edge-three-explicit"))
    check("三组显式指定 case/control 可跑", res["store"]["de"]["n_tested"] > 0,
          str(res["store"]["de"]["n_tested"]))

    # 缺失值：不填补也能跑并说明
    nan_ctx, _f, _i, _u, _d, _e = build_ctx(seed=35, fill_nan=False)
    res = D.step_de(nan_ctx, {}, out("edge-nan"))
    check("含 NaN 仍能跑并说明", any("缺失" in w for w in res["warnings"]),
          str(res["warnings"])[:100])
    check("含 NaN 时仍检出差异", res["summary"]["metrics"]["n_up"] > 0,
          str(res["summary"]["metrics"]["n_up"]))

    # 标度矩阵：log2FC 口径应给出说明
    scaled = dict(ctx)
    scaled["X_kind"] = "scaled"
    res = D.step_de(scaled, {}, out("edge-scaled"))
    check("标度矩阵 → 给出 log2FC 口径 warning",
          any("标度" in w for w in res["warnings"]), str(res["warnings"])[:100])


# ---------------------------------------------------------------------------
# 6) 多组 ANOVA
# ---------------------------------------------------------------------------


def test_multi_group():
    print("\n[6] 多组 ANOVA")
    ctx, features, injected, _u, _d, _e = build_ctx(seed=41, n_per_group=20, n_groups=3,
                                                    group_names=("control", "low", "high"))
    r = D.analysis_multi_group_de(ctx, {"method": "anova", "posthoc": "tukey"}, out("an-anova"))
    frame = r["tables"][0]
    check("ANOVA 表行数=特征数", frame["total"] == len(features), f"{frame['total']}")
    check("ANOVA 有显著特征", r["summary"]["metrics"]["n_sig"] > 0,
          str(r["summary"]["metrics"]["n_sig"]))
    check("ANOVA 事后表存在", len(r["tables"]) == 2 and r["tables"][1]["total"] > 0,
          str([t["id"] for t in r["tables"]]))
    post = pd.DataFrame(r["tables"][1]["rows"])
    check("事后 p 值合法", bool(np.all((post["p"] >= 0) & (post["p"] <= 1))))
    check("事后校正 p ≥ 原始 p", bool(np.all(post["p_adj"] >= post["p"] - 1e-12)))
    check("ANOVA 出图", all(figure_ok(f) for f in r["figures"]))

    for method, posthoc in (("welch", "games_howell"), ("kruskal", "dunn")):
        r = D.analysis_multi_group_de(ctx, {"method": method, "posthoc": posthoc},
                                      out(f"an-{method}"))
        check(f"{method} + {posthoc} 可跑", r["tables"][0]["total"] == len(features),
              str(r["tables"][0]["total"]))
        check(f"{method} 有显著特征", r["summary"]["metrics"]["n_sig"] > 0,
              str(r["summary"]["metrics"]["n_sig"]))
        if len(r["tables"]) > 1:
            check(f"{method} 事后表非空", r["tables"][1]["total"] > 0, str(r["tables"][1]["total"]))
    # 分组差异明显时，注入特征应排在前列
    first = pd.DataFrame(r["tables"][0]["rows"]).head(30)["feature"].astype(str)
    check("多组检验 top30 命中注入特征 ≥ 30%", len(set(first) & injected) / 30 >= 0.3,
          f"{len(set(first) & injected) / 30:.2f}")


# ---------------------------------------------------------------------------
# 7) 时间趋势
# ---------------------------------------------------------------------------


def test_trend():
    print("\n[7] 时间趋势")
    ctx, _features, _injected, _u, _d, extra = build_ctx(seed=51)
    r = D.analysis_trend_course(ctx, {"time_col": "time", "top_n": 8}, out("an-trend"))
    frame = pd.DataFrame(r["tables"][0]["rows"])
    check("趋势表有结果", len(frame) > 0, str(len(frame)))
    check("趋势列齐全",
          {"slope", "p_slope", "rho_spearman", "trend", "span_change"} <= set(frame.columns),
          str(list(frame.columns)))
    check("时间点识别正确", r["summary"]["metrics"]["n_timepoints"] == len(TIME_LEVELS),
          str(r["summary"]["metrics"]["n_timepoints"]))
    check("注入的时间趋势特征进入结果",
          len(set(frame["feature"].astype(str)) & set(extra["trend_features"])) > 0,
          str(len(set(frame["feature"].astype(str)) & set(extra["trend_features"]))))
    check("趋势出图", all(figure_ok(f) for f in r["figures"]))

    # 自动选特征（features 为空、没有 de 缓存时按相关性挑）
    r2 = D.analysis_trend_course(ctx, {"time_col": "time"}, out("an-trend-auto"))
    check("趋势自动选特征可跑", r2["tables"][0]["total"] > 0, str(r2["tables"][0]["total"]))

    # 没有时间列 → prerequisite
    no_time = dict(ctx)
    no_time["sample_info"] = ctx["sample_info"].drop(columns=["time", "dose"])
    try:
        D.analysis_trend_course(no_time, {}, out("an-trend-bad"))
        check("无时间列 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("无时间列 → prerequisite", error.code == "prerequisite", error.code)

    # 指定特征 + 时间列不存在
    try:
        D.analysis_trend_course(ctx, {"time_col": "not_a_col"}, out("an-trend-bad2"))
        check("时间列不存在 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("时间列不存在 → prerequisite", error.code == "prerequisite", error.code)


# ---------------------------------------------------------------------------
# 8) 剂量响应（4PL）
# ---------------------------------------------------------------------------


def test_dose():
    print("\n[8] 剂量响应（4PL）")
    ctx, _features, _injected, _u, _d, extra = build_ctx(seed=61)
    feature = extra["dose_feature"]
    r = D.analysis_dose_response(ctx, {"dose_col": "dose", "feature": feature}, out("an-dose"))
    metrics = r["summary"]["metrics"]
    print(f"    EC50={metrics['ec50']:.3f} hill={metrics['hill']:.3f} R2={metrics['r2']:.3f}")
    check("EC50 接近真值（±30%）", abs(metrics["ec50"] - TRUE_EC50) / TRUE_EC50 < 0.3,
          f"{metrics['ec50']:.3f}")
    check("Hill 斜率接近真值", abs(metrics["hill"] - TRUE_HILL) < 0.6, f"{metrics['hill']:.3f}")
    check("4PL R² > 0.8", metrics["r2"] > 0.8, f"{metrics['r2']:.3f}")
    table_frame = pd.DataFrame(r["tables"][0]["rows"])
    ec50_row = table_frame[table_frame["parameter"] == "EC50"].iloc[0]
    check("EC50 有置信区间",
          ec50_row["ci_low"] is not None and ec50_row["ci_low"] < ec50_row["estimate"] < ec50_row["ci_high"],
          f"{ec50_row['ci_low']:.3f}–{ec50_row['ci_high']:.3f}")
    check("剂量响应出图", all(figure_ok(f) for f in r["figures"]))

    # 自动选特征
    r2 = D.analysis_dose_response(ctx, {"dose_col": "dose"}, out("an-dose-auto"))
    check("剂量响应自动选特征可跑", r2["summary"]["metrics"]["ec50"] is not None,
          str(r2["summary"]["metrics"]["feature"]))

    # 常数特征 → prerequisite（可读错误）
    try:
        D.analysis_dose_response(ctx, {"dose_col": "dose", "feature": extra["constant_feature"]},
                                 out("an-dose-const"))
        check("常数特征 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("常数特征 → prerequisite", error.code == "prerequisite",
              f"{error.code}: {error.message[:60]}")

    # 特征不存在 → bad_request
    try:
        D.analysis_dose_response(ctx, {"dose_col": "dose", "feature": "nope"}, out("an-dose-bad"))
        check("未知特征 → bad_request", False, "没有抛错")
    except EngineError as error:
        check("未知特征 → bad_request", error.code == "bad_request", error.code)

    # 无关特征 × 无关剂量列（demo 上 dose_col="age" 的形态）：只允许可读错误或弱拟合，绝不 OverflowError
    try:
        unrelated = D.analysis_dose_response(ctx, {"dose_col": "age", "feature": _features[0]},
                                             out("an-dose-irrelevant"))
        weak = unrelated["summary"]["metrics"].get("r2", 0.0)
        warned = any("质量" in w or "弱" in w for w in unrelated["warnings"])
        check("无关剂量列不崩（返回弱拟合或可读错误）", True,
              f"r2={weak:.3f} warning={warned}")
    except EngineError as error:
        check("无关剂量列不崩（返回弱拟合或可读错误）",
              error.code in ("prerequisite", "bad_request"), f"{error.code}: {error.message[:50]}")
    except OverflowError as error:  # 明确禁止：这是老 bug
        check("无关剂量列不崩（返回弱拟合或可读错误）", False, f"OverflowError: {error}")

    no_dose = dict(ctx)
    no_dose["sample_info"] = ctx["sample_info"].drop(columns=["dose"])
    try:
        D.analysis_dose_response(no_dose, {}, out("an-dose-nodose"))
        check("无剂量列 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("无剂量列 → prerequisite", error.code == "prerequisite", error.code)


# ---------------------------------------------------------------------------
# 9) 基于 de 缓存的分析 + 5 个 PLOTS
# ---------------------------------------------------------------------------


def test_cached_analyses_and_plots(ctx, res):
    print("\n[9] 缓存分析 + 5 个 PLOTS")
    ctx = dict(ctx)
    ctx["results"] = dict(res.get("store") or {})

    r = D.analysis_volcano_de(ctx, {"n_labels": 15, "fc": 1.5, "fdr": 0.05}, out("an-volcano"))
    check("volcano_de 标注表 15 行", r["tables"][0]["total"] == 15, str(r["tables"][0]["total"]))
    check("volcano_de 出图", all(figure_ok(f) for f in r["figures"]))
    check("volcano_de 点数=特征数", r["summary"]["metrics"]["n_points"] == len(ctx["features"]),
          str(r["summary"]["metrics"]["n_points"]))

    r = D.analysis_fc_bar(ctx, {"top_n": 20}, out("an-fcbar"))
    check("fc_bar 表 20 行", r["tables"][0]["total"] == 20, str(r["tables"][0]["total"]))
    check("fc_bar 出图", all(figure_ok(f) for f in r["figures"]))

    r = D.analysis_heatmap_top(ctx, {"top_n": 50, "cluster": True}, out("an-heat"))
    check("heatmap_top 出图", all(figure_ok(f) for f in r["figures"]))
    n_heat = r["summary"]["metrics"]["n_features"]
    check("heatmap_top 特征数 ≤ top_n 且 > 0", 0 < n_heat <= 50, str(n_heat))

    r = D.analysis_univariate_de(ctx, {"method": "welch"}, out("an-univariate"))
    check("univariate_de 可跑（重算）", r["tables"][0]["total"] == len(ctx["features"]),
          str(r["tables"][0]["total"]))
    check("univariate_de 更新 store", "de" in r.get("store", {}), str(list(r.get("store", {}))))

    # PLOTS：5 个都能从缓存/现场出图
    plotted = out("plots")
    for kind, fn in D.PLOTS.items():
        params = {"dose_col": "dose", "time_col": "time"}
        fig = fn(ctx, params, plotted)
        check(f"PLOTS {kind} 出图 >1000B", figure_ok(fig),
              f"{os.path.basename(fig.get('file') or '')} "
              f"{os.path.getsize(fig['file']) if figure_ok(fig) else 0}B")
    # 火山图 PLOT 的 n_points 必须等于 DE 表行数
    fig = D.plot_volcano_de(ctx, {}, plotted)
    frame = ctx["results"]["de"]["table"]
    check("PLOT 火山图 n_points = DE 表行数", fig.get("n_points") == len(frame),
          f"{fig.get('n_points')} vs {len(frame)}")


def test_plots_from_scratch():
    print("\n[10] 5 个 PLOTS（无 de 缓存 → 现场算）")
    ctx, _features, _i, _u, _d, _e = build_ctx(seed=71, n_per_group=20, n_features=150)
    plotted = out("plots-scratch")
    for kind, fn in D.PLOTS.items():
        params = {"dose_col": "dose", "time_col": "time"}
        fig = fn(ctx, params, plotted)
        check(f"PLOTS {kind} 无缓存出图", figure_ok(fig))




# ---------------------------------------------------------------------------
# 11) demo 形态：case/control/QC 三组，未指定 case/control 时自动排除 QC
# ---------------------------------------------------------------------------


def add_qc_samples(ctx: dict, n_qc: int = 8, seed: int = 99) -> dict:
    """在两组数据里追加 QC 样本（模拟 mb_state demo：group=QC + qc=1 列）。"""
    rng = np.random.default_rng(seed)
    X = np.asarray(ctx["X"], dtype=float)
    base = np.nanmean(X, axis=0)
    qc_x = base[None, :] + rng.normal(0.0, 0.12, (n_qc, X.shape[1]))
    samples = list(ctx["samples"]) + [f"QC{i + 1:02d}" for i in range(n_qc)]
    info = ctx["sample_info"].copy()
    info["qc"] = 0
    qc_info = pd.DataFrame(
        {"group": ["QC"] * n_qc, "batch": ["B1"] * n_qc,
         "subject": [f"Q{i + 1:02d}" for i in range(n_qc)],
         "time": [0.0] * n_qc, "dose": [np.nan] * n_qc, "age": [np.nan] * n_qc,
         "qc": [1] * n_qc},
        index=pd.Index([f"QC{i + 1:02d}" for i in range(n_qc)], name="sample"),
    )
    new_info = pd.concat([info, qc_info]).reindex(pd.Index(samples, name="sample"))
    out = dict(ctx)
    out["X"] = np.vstack([X, qc_x]).astype(np.float32)
    out["samples"] = samples
    out["sample_info"] = new_info
    out["groups"] = ["case", "control", "QC"]
    return out


def test_qc_exclusion():
    print("\n[11] demo 形态：QC 自动排除")
    ctx, _features, _i, _u, _d, _e = build_ctx(seed=81, n_per_group=20)
    qc_ctx = add_qc_samples(ctx, n_qc=8)

    res = D.step_de(qc_ctx, {}, out("qc-auto"))
    store = res["store"]["de"]
    metrics = res["summary"]["metrics"]
    check("QC 组存在时未指定 case/control 仍能跑", metrics["n_tested"] > 0, str(metrics["n_tested"]))
    check("自动排除 QC：只比较 case vs control",
          metrics["contrast"] == "case vs control", str(metrics["contrast"]))
    check("QC 样本数不进入两组样本量",
          metrics["n_case"] == 20 and metrics["n_control"] == 20,
          f"{metrics['n_case']}/{metrics['n_control']}")
    check("warning 说明已排除 QC", any("QC" in w for w in res["warnings"]),
          str([w for w in res["warnings"] if "QC" in w])[:120])
    check("QC 排除后仍检出差异", metrics["n_up"] > 0, str(metrics["n_up"]))
    check("store 记录 comparison", store["comparison"] == {"case": "case", "control": "control"},
          str(store.get("comparison")))

    # 只有分组标签是 QC、没有 qc 列也能识别
    label_only = dict(qc_ctx)
    label_only["sample_info"] = qc_ctx["sample_info"].drop(columns=["qc"])
    res2 = D.step_de(label_only, {}, out("qc-label-only"))
    check("仅靠 group=QC 标签也能排除 QC",
          res2["summary"]["metrics"]["contrast"] == "case vs control"
          and any("QC" in w for w in res2["warnings"]),
          str(res2["summary"]["metrics"]["contrast"]))

    # 显式指定 case=QC 时不排除 QC（用户明确要比 QC）
    res3 = D.step_de(qc_ctx, {"case": "QC", "control": "control"}, out("qc-explicit"))
    check("显式 case=QC 时可与 QC 比较",
          res3["summary"]["metrics"]["n_case"] == 8 and "QC" in str(res3["summary"]["metrics"]["contrast"]),
          str(res3["summary"]["metrics"]["contrast"]))

    # 排除 QC 后仍有 >2 组 → prerequisite 且文案提到 QC
    three, _f, _i, _u, _d, _e = build_ctx(seed=83, n_per_group=12, n_groups=3,
                                          group_names=("control", "low", "high"))
    try:
        D.step_de(add_qc_samples(three, n_qc=6), {}, out("qc-three"))
        check("排除 QC 后仍 >2 组 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("排除 QC 后仍 >2 组 → prerequisite",
              error.code == "prerequisite" and "QC" in error.message,
              f"{error.code}: {error.message[:70]}")




def test_volcano_without_padj():
    print("\n[12] 火山图容错：DE 表缺 p_adj 列时不崩")
    ctx, _f, _i, _u, _d, _e = build_ctx(seed=85, n_per_group=15, n_features=120)
    frame = D._de_compute(ctx, {})["frame"]
    stripped = frame.drop(columns=[c for c in ("p_adj", "regulation", "fc_hit") if c in frame.columns])
    try:
        fig = D._render_volcano(stripped, {"n_labels": 10}, out("volcano-no-padj"))
        check("缺 p_adj 列时火山图不崩", figure_ok(fig),
              os.path.basename(fig.get("file") or ""))
        check("缺 p_adj 列时点数=行数", fig.get("n_points") == len(stripped),
              f"{fig.get('n_points')} vs {len(stripped)}")
    except Exception as error:  # noqa: BLE001
        check("缺 p_adj 列时火山图不崩", False, f"{type(error).__name__}: {error}")
    # 缺 p_adj 也要能走 PLOT 入口（plot_volcano_de 内部会经过 _ensure_de 归一化）
    fig = D.plot_volcano_de(ctx, {"n_labels": 5}, out("volcano-no-padj-plot"))
    check("plot_volcano_de 容错可出图", figure_ok(fig))


def test_demo_three_group_step():
    print("\n[13] demo 形态（case/control/QC）step_de 不抛 prerequisite")
    ctx, _f, _i, _u, _d, _e = build_ctx(seed=87, n_per_group=18)
    demo_ctx = add_qc_samples(ctx, n_qc=6)
    try:
        res = D.step_de(demo_ctx, {}, out("demo-step"))
    except EngineError as error:
        check("3 组 demo 上 step_de 不抛 prerequisite", False, f"{error.code}: {error.message[:80]}")
        return
    metrics = res["summary"]["metrics"]
    bullets = "；".join(res["summary"]["bullets"])
    check("3 组 demo 上 step_de 不抛 prerequisite", True,
          f"contrast={metrics['contrast']}")
    check("summary.bullets 含差异数与阈值", ("上调" in bullets and "下调" in bullets
                                            and "FDR" in bullets), bullets[:100])
    check("demo 上火山图正常出图", any("volcano" in str(f.get("file")) for f in res["figures"])
          and all(figure_ok(f) for f in res["figures"]))
    check("demo 上 store 写入 results['de'] 结构",
          isinstance(res["store"]["de"], dict) and isinstance(res["store"]["de"]["table"], pd.DataFrame),
          str(list(res["store"].keys())))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("dsh-metabolomics / mb_de 回归测试")
    print("=" * 72)
    test_catalog()
    test_math()
    ctx, res = test_step_de()
    test_other_methods(ctx)
    test_group_edges()
    test_multi_group()
    test_trend()
    test_dose()
    test_cached_analyses_and_plots(ctx, res)
    test_plots_from_scratch()
    test_qc_exclusion()
    test_volcano_without_padj()
    test_demo_three_group_step()

    print("\n" + "=" * 72)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL（异常中断）")
        sys.exit(1)
