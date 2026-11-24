#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — mb_plsda（PLS-DA / OPLS-DA）回归测试

覆盖（DESIGN §3 step plsda / §4 多变量判别 / §5 图 / §11 mb-stat 验收）：
  * 60 样本 × 300 特征、两组各 30、注入 60 个真差异特征（log2FC 0.8–1.6）、5% NaN 已填补；
  * 断言 Q² > 0.3、置换检验 p < 0.05、VIP 前 20 命中注入特征 ≥ 50%；
  * 6 个分析都能跑（plsda_vip / oplsda_splot / plsda_permutation / plsda_cv /
    plsda_multiclass / pca_loadings）；
  * 4 个 PLOTS（plsda_scores / oplsda_splot / vip_bar / permutation_test）都能出图且 size > 1000；
  * 契约细节：store schema、OPLS p(corr) ∈ [-1,1]、单组 → prerequisite、NaN 兜底、确定性。

用法：cd plugins/dsh-metabolomics && python3 test/py_plsda_test.py
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

import mb_plsda as P  # noqa: E402
from mb_common import EngineError  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="mb-plsda-test-")
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
# 合成数据：60 × 300，两组各 30，60 个注入差异特征，5% NaN（已按 half-min 填补）
# ---------------------------------------------------------------------------


def build_ctx(seed: int = 7, n_per_group: int = 30, n_features: int = 300,
              n_diff: int = 60, nan_frac: float = 0.05,
              groups: tuple = ("control", "case"), fill_nan: bool = True,
              n_groups: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    g_names = list(groups)[:n_groups]
    n = n_per_group * len(g_names)
    p = n_features
    samples = [f"S{i + 1:02d}" for i in range(n)]
    labels = np.array([g for g in g_names for _ in range(n_per_group)], dtype=object)

    # 特征理化性质
    mzs = np.round(rng.uniform(100.0, 900.0, p), 4)
    rts = np.round(rng.uniform(0.5, 18.0, p), 2)
    features = [f"mz{mz:.4f}_rt{rt:.2f}" for mz, rt in zip(mzs, rts)]
    feature_info = pd.DataFrame(
        {
            "mz": mzs,
            "rt": rts,
            "name": [f"Metabolite_{i + 1}" for i in range(p)],
            "class": ["Lipid"] * p,
        },
        index=pd.Index(features, name="feature"),
    )

    # log2 强度矩阵：基础丰度 + 样本因子 + 噪声
    base = rng.normal(10.0, 1.2, p)
    sample_factor = rng.normal(0.0, 0.30, n)
    X = base[None, :] + sample_factor[:, None] + rng.normal(0.0, 0.45, (n, p))

    # 注入差异特征（第一个组作为"case"上调/下调混合）
    diff_idx = rng.choice(p, size=min(n_diff, p), replace=False)
    log2fc = np.zeros(p)
    for j in diff_idx[: max(n_diff // 2, 1)]:
        log2fc[j] = rng.uniform(0.8, 1.6)
    for j in diff_idx[max(n_diff // 2, 1):]:
        log2fc[j] = -rng.uniform(0.8, 1.6)
    case_rows = labels == g_names[-1]
    X[case_rows] += log2fc[None, :]

    # 5% NaN 后按特征 half-min 填补（模拟 impute 步骤已完成）
    mask = rng.random((n, p)) < nan_frac
    X[mask] = np.nan
    if fill_nan:
        col_min = np.nanmin(X, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(X))
        X[nan_rows, nan_cols] = col_min[nan_cols] * 0.5

    sample_info = pd.DataFrame(
        {"group": labels, "batch": ["B1" if i % 2 else "B2" for i in range(n)],
         "age": np.round(rng.normal(50, 8, n), 1)},
        index=pd.Index(samples, name="sample"),
    )
    return {
        "dataset_id": "ds_plsda",
        "name": "synthetic-plsda",
        "features": features,
        "samples": samples,
        "X": np.asarray(X, dtype=np.float32),
        "X_kind": "lognorm",
        "raw": None,
        "feature_info": feature_info,
        "sample_info": sample_info,
        "group_key": "group",
        "groups": g_names,
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
    }, features, {features[i] for i in diff_idx.tolist()}


def run_step(ctx, params, tag):
    return P.step_plsda(ctx, params, out(tag))


def figure_ok(fig) -> bool:
    path = (fig or {}).get("file")
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# 1) 注册表结构
# ---------------------------------------------------------------------------


def test_catalog():
    print("\n[1] 注册表结构")
    step_ids = [s["id"] for s in P.STEPS]
    check("STEP 只有 plsda", step_ids == ["plsda"], str(step_ids))
    step = P.STEPS[0]
    check("step.requires == ['transform']", step["requires"] == ["transform"])
    check("step.run 可调用", callable(step["run"]))
    params = {p["key"] for p in step["params"]}
    check("step 参数齐全",
          {"method", "n_comp", "cv_folds", "permutations", "center_scale"} <= params, str(sorted(params)))
    analyses = {a["id"]: a for a in P.ANALYSES}
    expect = ["plsda_vip", "oplsda_splot", "plsda_permutation", "plsda_cv",
              "plsda_multiclass", "pca_loadings"]
    check("6 个分析注册", list(analyses) == expect, str(list(analyses)))
    check("分析 group 全为多变量判别",
          all(a["group"] == "多变量判别" for a in P.ANALYSES))
    check("needs 符合契约",
          analyses["plsda_vip"]["needs"] == ["plsda"]
          and analyses["plsda_multiclass"]["needs"] == ["groups"]
          and analyses["pca_loadings"]["needs"] == ["transform"])
    check("4 个 PLOTS 归属正确",
          set(P.PLOTS) == {"plsda_scores", "oplsda_splot", "vip_bar", "permutation_test"},
          str(sorted(P.PLOTS)))
    check("PLOTS 都可调用", all(callable(f) for f in P.PLOTS.values()))


# ---------------------------------------------------------------------------
# 2) 数值算法自检
# ---------------------------------------------------------------------------


def test_math():
    print("\n[2] 算法自检（NIPALS / VIP / 椭圆）")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 60))
    y = np.array([1.0] * 20 + [-1.0] * 20)
    Y = np.column_stack([(y > 0).astype(float), (y < 0).astype(float)])
    Y = Y - Y.mean(axis=0)
    fit = P._pls2_nipals(X, Y, 3)
    check("NIPALS 提取 3 个成分", fit["n_comp"] == 3, str(fit["n_comp"]))
    check("R²Y 单调不减", bool(np.all(np.diff(fit["r2y"]) >= -1e-9)), str(fit["r2y"].round(3)))
    check("R²X 单调不减", bool(np.all(np.diff(fit["r2x"]) >= -1e-9)), str(fit["r2x"].round(3)))
    vip = P._vip(fit["W"], fit["T"], fit["Q"])
    check("VIP 长度=特征数", vip.shape == (60,), str(vip.shape))
    check("mean(VIP²)≈1", abs(float(np.mean(vip ** 2)) - 1.0) < 1e-6,
          f"{float(np.mean(vip ** 2)):.6f}")
    check("W 列单位化", np.allclose(np.linalg.norm(fit["W"], axis=0), 1.0, atol=1e-8))
    ell = P._hotelling_ellipse(rng.normal(size=(20, 2)))
    check("95% 椭圆返回点集", isinstance(ell, np.ndarray) and ell.shape[1] == 2, str(ell.shape))
    check("样本不足时椭圆返回 None", P._hotelling_ellipse(np.zeros((2, 2))) is None)
    # OPLS 结构自洽
    opls = P._opls_fit(X, Y, 2)
    check("OPLS 有 1 预测 + 2 正交成分", opls["n_orth"] == 2 and opls["n_comp"] == 3,
          f"n_orth={opls['n_orth']}, n_comp={opls['n_comp']}")
    check("OPLS p(corr) ∈ [-1,1]",
          bool(np.all(np.abs(opls["p_corr"]) <= 1.0 + 1e-9)), f"max={np.abs(opls['p_corr']).max():.3f}")
    check("OPLS R²Y ∈ [0,1]", 0.0 <= float(opls["r2y"]) <= 1.0 + 1e-9, f"{opls['r2y']:.3f}")
    coef = opls["coef"]
    Xc = X - X.mean(axis=0)
    pred = Xc @ coef
    check("OPLS 预测系数与拟合一致",
          float(np.abs((pred - np.outer(opls["t_pred"], opls["q"])).mean())) < 1e-8)


# ---------------------------------------------------------------------------
# 3) STEP plsda：Q² / 置换 / VIP 命中
# ---------------------------------------------------------------------------


def test_step_plsda():
    print("\n[3] STEP plsda（PLS-DA 全量断言）")
    ctx, features, injected = build_ctx()
    res = run_step(ctx, {"method": "plsda", "n_comp": 3, "cv_folds": 7, "permutations": 200},
                   "step-plsda")
    metrics = res["summary"]["metrics"]
    store = res.get("store") or {}
    plsda = store.get("plsda") or {}
    print(f"    metrics: r2x={metrics.get('r2x'):.3f} r2y={metrics.get('r2y'):.3f} "
          f"q2={metrics.get('q2'):.3f} perm_p_r2y={metrics.get('perm_p_r2y'):.4g} "
          f"perm_p_q2={metrics.get('perm_p_q2'):.4g} acc={metrics.get('cv_accuracy')}")

    check("Q² > 0.3", float(metrics.get("q2") or 0.0) > 0.3, f"{metrics.get('q2')}")
    check("R²Y > 0.5", float(metrics.get("r2y") or 0.0) > 0.5, f"{metrics.get('r2y')}")
    check("置换 p(R²Y) < 0.05", float(metrics.get("perm_p_r2y") or 1.0) < 0.05,
          f"{metrics.get('perm_p_r2y'):.4g}")
    check("置换 p(Q²) < 0.05", float(metrics.get("perm_p_q2") or 1.0) < 0.05,
          f"{metrics.get('perm_p_q2'):.4g}")
    check("置换次数=200", int(metrics.get("perm_n") or 0) == 200, str(metrics.get("perm_n")))
    check("CV 准确率 > 0.8", float(metrics.get("cv_accuracy") or 0.0) > 0.8,
          f"{metrics.get('cv_accuracy')}")

    vip_store = store.get("vip")
    check("store['vip'] 是 dict（registry export 要求，含 table/vip 键）",
          isinstance(vip_store, dict) and isinstance(vip_store.get("table"), pd.DataFrame),
          type(vip_store).__name__)
    vip = vip_store["table"]
    check("store['plsda']['vip'] 与 store['vip']['table'] 是同一张表",
          plsda.get("vip") is vip, f"n={len(vip)}")
    check("VIP 表列符合 schema",
          list(vip.columns)[:6] == ["rank", "feature", "name", "mz", "rt", "vip"], str(list(vip.columns)))
    top20 = set(vip.head(20)["feature"].astype(str))
    hit = len(top20 & injected) / 20.0
    check("VIP 前 20 命中注入差异特征 ≥ 50%", hit >= 0.5, f"hit={hit:.2f}")
    check("VIP 降序", bool(np.all(np.diff(vip["vip"].to_numpy()) <= 1e-12)))
    check("store['plsda']['labels'] 与样本对齐",
          len(plsda.get("labels") or []) == len(ctx["samples"]))
    check("store['plsda']['scores'] 行列正确",
          isinstance(plsda.get("scores"), pd.DataFrame) and len(plsda["scores"]) == len(ctx["samples"])
          and "T1" in plsda["scores"].columns)
    check("store['plsda']['permutation'] 有零分布",
          isinstance(plsda.get("permutation"), dict) and len(plsda["permutation"]["q2_null"]) == 200)
    check("step 返回 ≥3 张图", len(res["figures"]) >= 3, str([f["id"] for f in res["figures"]]))
    check("图文件都 > 1000 字节", all(figure_ok(f) for f in res["figures"]))
    data_out = res.get("data_out") or []
    check("data_out 有 VIP 全表 csv",
          any(d["kind"] == "csv" and "vip" in d["path"].lower() for d in data_out),
          str([os.path.basename(d["path"]) for d in data_out]))
    check("VIP 全表行数=特征数",
          any(d["path"].endswith("plsda_vip.csv") and len(pd.read_csv(d["path"])) == len(features)
              for d in data_out))

    # 确定性
    res2 = run_step(ctx, {"method": "plsda", "n_comp": 3, "cv_folds": 7, "permutations": 200},
                    "step-plsda-2")
    check("重复运行结果一致（确定性）",
          abs(float(res2["summary"]["metrics"]["q2"]) - float(metrics["q2"])) < 1e-12
          and abs(float(res2["summary"]["metrics"]["perm_p_q2"]) - float(metrics["perm_p_q2"])) < 1e-12)
    return ctx, res


# ---------------------------------------------------------------------------
# 4) OPLS-DA 分支
# ---------------------------------------------------------------------------


def test_step_opls():
    print("\n[4] STEP plsda（OPLS-DA 分支）")
    ctx, _features, _injected = build_ctx(seed=13)
    res = run_step(ctx, {"method": "oplsda", "n_comp": 3, "cv_folds": 7, "permutations": 100},
                   "step-opls")
    store = res.get("store") or {}
    plsda = store.get("plsda") or {}
    metrics = res["summary"]["metrics"]
    check("method=oplsda", plsda.get("method") == "oplsda", str(plsda.get("method")))
    check("正交成分数=2（n_comp=3）", int(plsda.get("n_orth") or 0) == 2, str(plsda.get("n_orth")))
    splot = store.get("splot")
    check("store['splot'] 是 DataFrame", isinstance(splot, pd.DataFrame), type(splot).__name__)
    check("S-plot 表列正确",
          {"feature", "loading_p1", "p_corr", "vip"} <= set(splot.columns), str(list(splot.columns)))
    check("p(corr) ∈ [-1,1]", bool(np.all(np.abs(splot["p_corr"].to_numpy()) <= 1.0 + 1e-9)))
    check("OPLS Q² > 0.3", float(metrics.get("q2") or 0.0) > 0.3, str(metrics.get("q2")))
    check("OPLS 置换 p(Q²) < 0.05", float(metrics.get("perm_p_q2") or 1.0) < 0.05,
          f"{metrics.get('perm_p_q2'):.4g}")
    ids = [f["id"] for f in res["figures"]]
    check("OPLS 步骤含 4 张图（含 S-plot）", len(res["figures"]) == 4, str(ids))
    check("S-plot 图已生成", any("splot" in str(f.get("file")) for f in res["figures"]))
    check("图文件都 > 1000 字节", all(figure_ok(f) for f in res["figures"]))
    return ctx


# ---------------------------------------------------------------------------
# 5) 6 个分析
# ---------------------------------------------------------------------------


def test_analyses(ctx, res):
    print("\n[5] 6 个分析")
    ctx = dict(ctx)
    ctx["results"] = dict(res.get("store") or {})
    results = {}

    r = P.analysis_plsda_vip(ctx, {"top_n": 30, "threshold": 1}, out("an-vip"))
    results["plsda_vip"] = r
    check("plsda_vip：表 30 行", (r["tables"][0]["total"] == 30), str(r["tables"][0]["total"]))
    check("plsda_vip：出图", all(figure_ok(f) for f in r["figures"]))
    check("plsda_vip：metrics 有 n_above", "n_above" in (r["summary"]["metrics"] or {}),
          str(r["summary"]["metrics"].get("n_above")))

    r = P.analysis_oplsda_splot(ctx, {"top_n": 20}, out("an-splot"))
    results["oplsda_splot"] = r
    check("oplsda_splot：表 20 行", r["tables"][0]["total"] == 20, str(r["tables"][0]["total"]))
    check("oplsda_splot：出图", all(figure_ok(f) for f in r["figures"]))
    check("oplsda_splot：n_orth=2", r["summary"]["metrics"].get("n_orth") == 2,
          str(r["summary"]["metrics"].get("n_orth")))

    r = P.analysis_plsda_permutation(ctx, {"n_perms": 150}, out("an-perm"))
    results["plsda_permutation"] = r
    check("plsda_permutation：p<0.05", float(r["summary"]["metrics"]["p_q2"]) < 0.05,
          f"{r['summary']['metrics']['p_q2']:.4g}")
    check("plsda_permutation：表行数=置换次数",
          r["tables"][0]["total"] == int(r["summary"]["metrics"]["n_perms"]),
          f"{r['tables'][0]['total']}")
    check("plsda_permutation：出图", all(figure_ok(f) for f in r["figures"]))

    r = P.analysis_plsda_cv(ctx, {"folds": 7}, out("an-cv"))
    results["plsda_cv"] = r
    check("plsda_cv：准确率 > 0.8", float(r["summary"]["metrics"]["accuracy"]) > 0.8,
          str(r["summary"]["metrics"]["accuracy"]))
    check("plsda_cv：混淆矩阵表存在", len(r["tables"]) == 2 and r["tables"][0]["total"] == 2,
          str([t["id"] for t in r["tables"]]))
    check("plsda_cv：出图", all(figure_ok(f) for f in r["figures"]))

    r = P.analysis_plsda_multiclass(ctx, {"n_comp": 3}, out("an-multi"))
    results["plsda_multiclass"] = r
    check("plsda_multiclass：两组时给 warning",
          any("2 个分组" in w for w in r["warnings"]), str(r["warnings"])[:120])
    check("plsda_multiclass：OVR 表 2 行", r["tables"][0]["total"] == 2, str(r["tables"][0]["total"]))
    check("plsda_multiclass：出 2 张图", len(r["figures"]) == 2)

    r = P.analysis_pca_loadings(ctx, {"pc": 1, "top_n": 20}, out("an-pca"))
    results["pca_loadings"] = r
    check("pca_loadings：表 20 行", r["tables"][0]["total"] == 20, str(r["tables"][0]["total"]))
    check("pca_loadings：解释方差 ∈ (0,1]",
          0.0 < float(r["summary"]["metrics"]["explained_variance"]) <= 1.0,
          str(r["summary"]["metrics"]["explained_variance"]))
    check("pca_loadings：出图", all(figure_ok(f) for f in r["figures"]))
    return results


# ---------------------------------------------------------------------------
# 6) 多组（3 类）PLS-DA
# ---------------------------------------------------------------------------


def test_multiclass():
    print("\n[6] 三组 PLS-DA")
    ctx, _features, _injected = build_ctx(seed=21, n_per_group=20, n_groups=3,
                                          groups=("control", "low", "high"))
    res = run_step(ctx, {"method": "plsda", "n_comp": 3, "cv_folds": 5, "permutations": 50},
                   "step-multi")
    store = res.get("store") or {}
    plsda = store.get("plsda") or {}
    check("3 类 one-hot：classes 长度 3", len(plsda.get("classes") or []) == 3,
          str(plsda.get("classes")))
    check("3 类 PLS-DA Q² > 0.2", float((res["summary"]["metrics"] or {}).get("q2") or 0) > 0.2,
          str((res["summary"]["metrics"] or {}).get("q2")))
    ctx2 = dict(ctx)
    ctx2["results"] = dict(store)
    r = P.analysis_plsda_multiclass(ctx2, {"n_comp": 3, "cv_folds": 5}, out("an-multi3"))
    check("多类 OVR 表 3 行", r["tables"][0]["total"] == 3, str(r["tables"][0]["total"]))
    check("多类准确率 > 0.6", float(r["summary"]["metrics"]["accuracy"]) > 0.6,
          str(r["summary"]["metrics"]["accuracy"]))


# ---------------------------------------------------------------------------
# 7) 4 个 PLOTS 从零出图（ctx 无缓存 → 现场拟合）
# ---------------------------------------------------------------------------


def test_plots_from_scratch():
    print("\n[7] 4 个 PLOTS（无缓存 → 现场拟合）")
    ctx, _features, _injected = build_ctx(seed=31, n_per_group=25, n_features=200)
    plotted = os.path.join(OUTDIR, "plots")
    os.makedirs(plotted, exist_ok=True)
    for kind, fn in P.PLOTS.items():
        params = {"n_perms": 60} if kind == "permutation_test" else {}
        fig = fn(ctx, params, plotted)
        ok = figure_ok(fig)
        check(f"PLOTS {kind} 出图 >1000B", ok, f"{os.path.basename(fig.get('file') or '')} "
                                              f"{os.path.getsize(fig['file']) if ok else 0}B")
    # 4 个 kind 的图都能从缓存再出一次（回归：_cached_model 路径）
    res = run_step(ctx, {"method": "plsda", "n_comp": 3, "cv_folds": 5, "permutations": 50},
                   "step-plots")
    ctx2 = dict(ctx)
    ctx2["results"] = dict(res.get("store") or {})
    for kind, fn in P.PLOTS.items():
        fig = fn(ctx2, {}, plotted)
        check(f"PLOTS {kind} 从缓存出图", figure_ok(fig))


# ---------------------------------------------------------------------------
# 8) 契约边界：prerequisite / NaN / 标度
# ---------------------------------------------------------------------------


def test_edges():
    print("\n[8] 契约边界")
    # 单组 → prerequisite
    ctx1, _f, _i = build_ctx(seed=41, groups=("control", "case"))
    info = ctx1["sample_info"].copy()
    info["group"] = "case"
    ctx1["sample_info"] = info
    try:
        P.step_plsda(ctx1, {"permutations": 10}, out("edge-one-group"))
        check("单组 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("单组 → prerequisite", error.code == "prerequisite", f"{error.code}: {error.message}")

    # 没分组列 → prerequisite
    ctx2, _f, _i = build_ctx(seed=42)
    ctx2["group_key"] = None
    ctx2["sample_info"] = ctx2["sample_info"].drop(columns=["group"])
    try:
        P.plot_plsda_scores(ctx2, {}, out("edge-nogroup"))
        check("无分组 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("无分组 → prerequisite", error.code == "prerequisite", f"{error.code}")

    # NaN 未填补 → 仍能跑并给 warning
    ctx3, _f, _i = build_ctx(seed=43, fill_nan=False)
    res = P.step_plsda(ctx3, {"n_comp": 2, "cv_folds": 5, "permutations": 30}, out("edge-nan"))
    check("含 NaN 仍能跑并解释缺失",
          any("缺失" in w for w in res["warnings"]), str(res["warnings"])[:120])
    check("含 NaN 时 Q² 仍 > 0.3", float((res["summary"]["metrics"] or {}).get("q2") or 0) > 0.3,
          str((res["summary"]["metrics"] or {}).get("q2")))

    # 非法参数
    ctx4, _f, _i = build_ctx(seed=44)
    try:
        P.step_plsda(ctx4, {"method": "randomforest"}, out("edge-bad"))
        check("非法 method → bad_request", False, "没有抛错")
    except EngineError as error:
        check("非法 method → bad_request", error.code == "bad_request", f"{error.code}")
    try:
        P.analysis_pca_loadings(ctx4, {"pc": 999}, out("edge-pc"))
        check("pc 超范围 → prerequisite", False, "没有抛错")
    except EngineError as error:
        check("pc 超范围 → prerequisite", error.code == "prerequisite", f"{error.code}")

    # 标度形式
    for mode, expect_label in (("pareto", "Pareto"), ("none", "centering"), ("auto", "unit variance")):
        prep = P._prepare(dict(ctx4), center_scale=mode)
        check(f"标度 {mode} 生效", expect_label in prep["scale_label"], prep["scale_label"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("dsh-metabolomics / mb_plsda 回归测试")
    print("=" * 72)
    test_catalog()
    test_math()
    ctx, res = test_step_plsda()
    test_step_opls()
    test_analyses(ctx, res)
    test_multiclass()
    test_plots_from_scratch()
    test_edges()

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
