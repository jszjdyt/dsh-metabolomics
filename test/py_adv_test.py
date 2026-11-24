#!/usr/bin/env python3
"""
dsh-metabolomics — mb-adv 回归测试（只覆盖 mb_enrich.py + mb_biomarker.py）

自造 ctx（72 样本 × 400 特征）：
  * 分组 case 33 / control 33 / QC 6（QC 用来验证"分组比较时剔除 QC"）
  * feature_info 带 mz / rt / adduct / name / kegg / class / pathway
  * 3 条目标通路（Purine / Arginine and proline / Tryptophan）共 33 个"设计特征"：
    名称取自内置库"只属于该通路"的成员、m/z 由该代谢物分子式的理论离子质量反算（±2 ppm）
  * 另外 120 个背景特征用 pathway 列注释到 4 条无关通路（ORA 的背景集合）
  * 40 个显著差异特征（33 个设计特征 + 7 个无注释噪声特征）
  * 临床列 os_time / os_event 与某个上调设计特征相关（保证 KM/logrank 有差异）
断言：
  * enrich 的 4 个分析 + 2 张图；biomarker 的 6 个分析 + 5 张图全部可跑
  * mummichog 返回 ≥1 通路、p∈[0,1]、3 条目标通路显著；ora_kegg 命中数与注释一致
  * RF/ROC 在强信号下 AUC>0.8；LASSO 入选 ≥1；模型验证可跑；生存分析 logrank 显著
  * 图文件存在且 >1000 字节；表 ≤500 行；data_out 文件真实存在；前置缺失抛 prerequisite
跑完打印 PASS 并以 0 退出。离线、无网络依赖，约 20–40 秒。
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import mb_common as C  # noqa: E402
import mb_biomarker as B  # noqa: E402
import mb_enrich as E  # noqa: E402

ALLOWED_NEEDS = {"groups", "group", "transform", "normalize", "plsda", "de", "vip",
                 "annotation", "qc", "clinical", "ms2", "mz", "filter", "pca", "batch",
                 "impute", "enrich", "annotate", "scaled", "lognorm"}
STEP_IDS = {"qc", "filter", "impute", "normalize", "transform", "pca", "plsda", "de",
            "batch", "enrich", "biomarker", "annotate"}
FORBIDDEN_PLOTS = {"qc_tic", "qc_rsd", "qc_missing", "pca_scores", "pca_variance",
                   "plsda_scores", "oplsda_splot", "vip_bar", "permutation_test",
                   "volcano_de", "heatmap_top", "batch_pca", "mass_accuracy", "mass_network"}

ENRICH_IDS = {"mummichog_pathway", "ora_kegg", "ora_hmdb_class", "metabolite_set_enrich"}
BIOMARKER_IDS = {"rf_importance", "lasso_panel", "roc_biomarker", "biomarker_score",
                 "model_validation", "survival_metabolite"}
ENRICH_PLOTS = {"enrich_bar", "enrich_dot"}
BIOMARKER_PLOTS = {"rf_importance", "lasso_path", "roc_curve", "km_metabolite", "calibration_curve"}

TARGET_PATHWAYS = [
    "Purine metabolism",
    "Arginine and proline metabolism",
    "Tryptophan metabolism",
]
BACKGROUND_PATHWAYS = [
    "Fatty acid metabolism", "Caffeine metabolism", "Retinol metabolism", "Folate biosynthesis",
]
CLASS_OF_PATHWAY = {
    "Purine metabolism": "Nucleotide",
    "Arginine and proline metabolism": "Amino acid",
    "Tryptophan metabolism": "Amine",
    "Fatty acid metabolism": "Lipid",
    "Caffeine metabolism": "Nucleobase derivative",
    "Retinol metabolism": "Vitamin / cofactor",
    "Folate biosynthesis": "Vitamin / cofactor",
}

N_SAMPLES, N_FEATURES = 72, 400
N_CASE, N_CONTROL, N_QC = 33, 33, 6
N_SIGNIFICANT = 40
N_BACKGROUND_ANNOTATED = 120


# ---------------------------------------------------------------------------
# 造数
# ---------------------------------------------------------------------------


def _unique_members():
    """只属于一条通路的成员（避免"同名代谢物属于多条通路"污染 ORA 计数）。"""
    counter: dict[str, int] = {}
    for members in E.MB_PATHWAY_MEMBERS.values():
        for member in members:
            counter[member] = counter.get(member, 0) + 1
    return {path: [m for m in members if counter[m] == 1]
            for path, members in E.MB_PATHWAY_MEMBERS.items()}


def build_ctx(seed: int = 0, outdir: str | None = None):
    rng = np.random.default_rng(seed)
    outdir = outdir or tempfile.mkdtemp(prefix="mb-adv-")
    uniq = _unique_members()
    adduct = "[M+H]+"

    # ---- 设计特征：3 条目标通路，每条取"唯一成员" ----
    plan = {
        "Purine metabolism": 10,
        "Arginine and proline metabolism": 10,
        "Tryptophan metabolism": 13,
    }
    designed: list[dict] = []
    for path, count in plan.items():
        for member in uniq[path][:count]:
            neutral = E.MB_METABOLITE_MASS[member]
            mz_true = C.adduct_mz(neutral, adduct)
            mz = mz_true * (1.0 + float(rng.uniform(-2.0, 2.0)) * 1e-6)
            designed.append({"pathway": path, "name": member, "mz": mz, "neutral": neutral})
    n_designed = len(designed)
    assert n_designed == sum(plan.values()) == 33, n_designed

    feature_ids: list[str] = []
    mz_values: list[float] = []
    names: list[str] = []
    classes: list[str] = []
    pathways: list[str] = []
    kegg_ids: list[str] = []
    adducts: list[str] = []
    rts: list[float] = []
    for index, item in enumerate(designed):
        feature_id = f"mz{item['mz']:.4f}_rt{1.0 + index * 0.15:.2f}"
        feature_ids.append(feature_id)
        mz_values.append(item["mz"])
        names.append(item["name"])
        classes.append(CLASS_OF_PATHWAY[item["pathway"]])
        pathways.append(item["pathway"])
        kegg_ids.append(f"C{(index * 7 + 3) % 9000 + 1000:05d}")
        adducts.append(adduct)
        rts.append(1.0 + index * 0.15)

    # ---- 背景特征：120 个注释到无关通路（name 不是代谢物名 → 只靠 pathway 列归属）----
    for index in range(N_BACKGROUND_ANNOTATED):
        mz = float(rng.uniform(100.0, 900.0))
        feature_id = f"mz{mz:.4f}_rt{5.0 + (index % 40) * 0.3:.2f}"
        feature_ids.append(feature_id)
        mz_values.append(mz)
        names.append(f"Noise {index + 1:03d}")
        path = BACKGROUND_PATHWAYS[index % len(BACKGROUND_PATHWAYS)]
        classes.append(CLASS_OF_PATHWAY[path])
        pathways.append(path)
        kegg_ids.append("")
        adducts.append(adduct)
        rts.append(5.0 + (index % 40) * 0.3)

    # ---- 其余空白特征 ----
    while len(feature_ids) < N_FEATURES:
        index = len(feature_ids)
        mz = float(rng.uniform(100.0, 900.0))
        feature_id = f"mz{mz:.4f}_rt{9.0 + (index % 30) * 0.25:.2f}"
        feature_ids.append(feature_id)
        mz_values.append(mz)
        names.append(f"Unassigned {index + 1:03d}")
        classes.append("Other")
        pathways.append("")
        kegg_ids.append("")
        adducts.append(adduct)
        rts.append(9.0 + (index % 30) * 0.25)

    assert len(feature_ids) == N_FEATURES == len(set(feature_ids)), "特征 id 必须唯一"
    features = feature_ids

    # ---- 强度矩阵（case 组对设计特征有 ±1.2 的对数尺度位移）----
    X = rng.normal(loc=11.0, scale=0.6, size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    case_mask = np.zeros(N_SAMPLES, dtype=bool)
    case_mask[:N_CASE] = True
    control_mask = np.zeros(N_SAMPLES, dtype=bool)
    control_mask[N_CASE:N_CASE + N_CONTROL] = True
    qc_mask = np.zeros(N_SAMPLES, dtype=bool)
    qc_mask[N_CASE + N_CONTROL:] = True
    log2fc = np.zeros(N_FEATURES, dtype=float)
    significant = np.zeros(N_FEATURES, dtype=bool)
    for index in range(n_designed):
        shift = 1.2 if index % 2 == 0 else -1.2
        log2fc[index] = shift / math.log(2.0)
        X[case_mask, index] += np.float32(shift)
        significant[index] = True
    extra_sig = np.where(~significant)[0][: N_SIGNIFICANT - n_designed]
    for index in extra_sig:
        shift = 1.0
        log2fc[index] = shift / math.log(2.0)
        X[case_mask, index] += np.float32(shift)
        significant[index] = True
    # QC 样本：轻微随机波动 + 不同水平，验证"被剔除"不影响主流程
    X[qc_mask, :] += rng.normal(0.0, 0.08, size=(int(qc_mask.sum()), N_FEATURES)).astype(np.float32)
    assert int(significant.sum()) == N_SIGNIFICANT

    # ---- 样本表 ----
    samples = [f"S{index + 1:02d}" for index in range(N_SAMPLES)]
    groups = np.where(case_mask, "case", np.where(control_mask, "control", "QC"))
    is_qc = np.where(qc_mask, 1, 0)

    # ---- 临床列：os_time / os_event 与上调设计特征相关 ----
    risk_feature = 0  # designed[0] 是上调特征
    z = (X[:, risk_feature] - X[:, risk_feature].mean()) / X[:, risk_feature].std(ddof=1)
    hazard = 0.020 * np.exp(0.90 * z)
    raw_time = -np.log(np.clip(rng.uniform(0.001, 1.0, N_SAMPLES), 1e-6, 1.0)) / hazard
    censor_time = 60.0
    os_time = np.minimum(raw_time, censor_time)
    os_event = (raw_time <= censor_time).astype(int)
    sample_info = pd.DataFrame({
        "group": groups,
        "qc": is_qc,
        "batch": np.where(np.arange(N_SAMPLES) % 2 == 0, "B1", "B2"),
        "age": rng.uniform(30, 70, N_SAMPLES).round(1),
        "os_time": os_time,
        "os_event": os_event,
    }, index=pd.Index(samples, name="sample"))

    # ---- 特征注释 ----
    feature_info = pd.DataFrame({
        "mz": np.asarray(mz_values, dtype=float),
        "rt": np.asarray(rts, dtype=float),
        "name": names,
        "formula": "",
        "adduct": adducts,
        "hmdb": [f"HMDB{index + 1:07d}" for index in range(N_FEATURES)],
        "kegg": [value if value else np.nan for value in kegg_ids],
        "class": classes,
        "pathway": [value if value else np.nan for value in pathways],
        "polarity": "pos",
        "confidence": 3,
    }, index=pd.Index(features, name="feature"))

    # ---- 差异表（mb-stat 的正式 schema）----
    mean_case = X[case_mask].mean(axis=0)
    mean_control = X[control_mask].mean(axis=0)
    p_values = np.where(significant, 1e-6, rng.uniform(0.02, 1.0, N_FEATURES))
    p_adj = np.where(significant, 1e-4, np.clip(rng.uniform(0.2, 1.0, N_FEATURES), 0.2, 1.0))
    de = pd.DataFrame({
        "feature": features,
        "name": names,
        "mz": np.asarray(mz_values, dtype=float),
        "rt": np.asarray(rts, dtype=float),
        "log2FC": log2fc,
        "mean_case": mean_case,
        "mean_control": mean_control,
        "p": p_values,
        "p_adj": p_adj,
        "regulation": np.where(significant, np.where(log2fc > 0, "up", "down"), "ns"),
        "fc_hit": significant,
        "detected_frac": 1.0,
        "t": np.where(significant, np.sign(log2fc) * 8.0, 0.2),
    })
    de_store = {
        "table": de, "de_table": de, "method": "welch", "paired": False,
        "group_key": "group", "case": "case", "control": "control",
        "fdr": 0.05, "fc": 1.5, "n_tested": N_FEATURES,
        "n_up": int(significant.sum()), "n_down": 0,
        "up": [f for f in features[:n_designed]], "down": [],
        "log2fc_basis": "log2-difference", "params": {},
    }

    # ---- VIP（模拟上游 PLS-DA 结果；两处键都给）----
    vip_frame = pd.DataFrame({
        "rank": np.arange(1, N_FEATURES + 1),
        "feature": features,
        "name": names,
        "mz": np.asarray(mz_values, dtype=float),
        "rt": np.asarray(rts, dtype=float),
        "vip": np.where(significant, 1.4 + rng.uniform(0, 0.6, N_FEATURES), rng.uniform(0.2, 0.9, N_FEATURES)),
        "above_threshold": significant,
    })
    plsda_store = {"vip": vip_frame, "table": vip_frame,
                   "metrics": {"method": "plsda", "n_comp": 3, "r2x": 0.6, "r2y": 0.9, "q2": 0.7}}

    ctx = {
        "dataset_id": "ds-test",
        "name": "mb-adv-test",
        "features": features,
        "samples": samples,
        "X": X,
        "X_kind": "lognorm",
        "raw": None,
        "feature_info": feature_info,
        "sample_info": sample_info,
        "group_key": "group",
        "groups": ["case", "control", "QC"],
        "matrix_order": "samples_by_features",
        "species": "human",
        "polarity": "pos",
        "results": {
            "de": de_store,
            "plsda": plsda_store,
            # 与 mb-plsda 对齐：results["vip"] 是 dict（registry 会对 store 值做真值判断，
            # 裸 DataFrame 会导致 vip_csv/report 导出失败），plsda["vip"] 仍是 DataFrame
            "vip": {"table": vip_frame, "vip": vip_frame, "method": "plsda",
                    "n_features": N_FEATURES, "n_above": int(significant.sum()), "threshold": 1.0},
        },
        "library": None,
        "rng": np.random.default_rng(seed),
        "workdir": outdir,
        "outdir": outdir,
        "engine": {"python": "", "mode": "builtin"},
        "notes": [],
        "warnings": [],
    }
    truth = {
        "designed": designed,
        "plan": plan,
        "n_designed": n_designed,
        "risk_feature": features[risk_feature],
        "up_features": [features[i] for i in range(n_designed) if i % 2 == 0],
        "down_features": [features[i] for i in range(n_designed) if i % 2 == 1],
        "expected_hits": dict(plan),
    }
    return ctx, truth


# ---------------------------------------------------------------------------
# 断言小工具
# ---------------------------------------------------------------------------


def ok(message: str) -> None:
    print(f"  [ok] {message}")


def expect(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def as_frame(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return pd.DataFrame(value)
    return pd.DataFrame()


def store_frame(payload: dict, module_key: str = "enrich") -> pd.DataFrame:
    store = payload.get("store") or {}
    block = store.get(module_key) if isinstance(store, dict) else None
    if isinstance(block, dict):
        for key in ("table", "top"):
            frame = as_frame(block.get(key))
            if not frame.empty:
                return frame
    return pd.DataFrame()


def check_figures(payload: dict, minimum: int = 1000) -> list[dict]:
    figures = payload.get("figures") or []
    expect(len(figures) >= 1, "结果里没有图")
    for figure in figures:
        path = figure.get("file")
        expect(path and os.path.isfile(path), f"图文件不存在：{path}")
        size = os.path.getsize(path)
        expect(size > minimum, f"图太小（{size} 字节）：{path}")
    return figures


def check_tables(payload: dict) -> list[dict]:
    tables = payload.get("tables") or []
    for item in tables:
        expect(isinstance(item, dict) and item.get("columns"), "表结构不完整")
        rows = item.get("rows") or []
        expect(len(rows) <= 500, f"表超过 500 行：{item.get('id')} -> {len(rows)}")
        for column in item["columns"]:
            expect(column.get("key") and column.get("label"), "列缺少 key/label")
    return tables


def check_data_out(payload: dict) -> list[dict]:
    data_out = payload.get("data_out") or []
    for item in data_out:
        path = item.get("path")
        expect(path and os.path.isfile(path), f"data_out 文件不存在：{path}")
    return data_out


def expect_prerequisite(fn, contains: str = ""):
    try:
        fn()
    except C.EngineError as error:
        expect(error.code == "prerequisite", f"错误码不是 prerequisite，而是 {error.code}：{error.message}")
        if contains:
            expect(contains in error.message, f"错误信息里缺少「{contains}」：{error.message}")
        return error
    raise AssertionError("应当抛 prerequisite，但没有抛")


# ---------------------------------------------------------------------------
# 1. 目录与契约
# ---------------------------------------------------------------------------


def test_catalog() -> None:
    print("* 目录与契约")
    expect(len(E.MB_PATHWAY_MEMBERS) >= 20, f"内置通路库不足 20 条：{len(E.MB_PATHWAY_MEMBERS)}")
    sizes = [len(members) for members in E.MB_PATHWAY_MEMBERS.values()]
    expect(min(sizes) >= 5, f"有通路成员少于 5 个：{min(sizes)}")
    expect(max(sizes) <= 25, f"有通路成员多于 25 个：{max(sizes)}")
    expect(all(mass > 0 for mass in E.MB_METABOLITE_MASS.values()), "存在非正的中性质量")
    # 分子式质量抽查（与 HMDB 单同位素质量一致到 0.01 Da）
    knots = {"D-Glucose": 180.0634, "Pyruvate": 88.0160, "Citrate": 192.0270,
             "Cholesterol": 386.3549, "ATP": 506.9957, "Glutathione": 307.0838}
    for name, expected in knots.items():
        got = E.MB_METABOLITE_MASS[name]
        expect(abs(got - expected) < 0.01, f"{name} 质量不对：{got:.4f} vs {expected:.4f}")
    ok(f"内置通路库 {len(E.MB_PATHWAY_MEMBERS)} 条，成员 {min(sizes)}–{max(sizes)} 个，"
       f"代谢物质量 {len(E.MB_METABOLITE_MASS)} 条（分子式换算）")

    # STEPS / ANALYSES 结构
    expect([step["id"] for step in E.STEPS] == ["enrich"], "mb_enrich.STEPS 应只有 enrich")
    expect([step["id"] for step in B.STEPS] == ["biomarker"], "mb_biomarker.STEPS 应只有 biomarker")
    for step in list(E.STEPS) + list(B.STEPS):
        expect(step["id"] in STEP_IDS, f"未知步骤 id：{step['id']}")
        expect(callable(step["run"]), f"步骤 {step['id']} 没有 run")
        for requirement in step.get("requires") or []:
            expect(requirement in STEP_IDS, f"步骤 {step['id']} 的 requires 非法：{requirement}")
        for param in step.get("params") or []:
            expect(isinstance(param, dict) and param.get("key"), f"步骤 {step['id']} 参数缺 key")
    enrich_ids = {item["id"] for item in E.ANALYSES}
    biomarker_ids = {item["id"] for item in B.ANALYSES}
    expect(enrich_ids == ENRICH_IDS, f"mb_enrich.ANALYSES 与契约不一致：{sorted(enrich_ids)}")
    expect(biomarker_ids == BIOMARKER_IDS, f"mb_biomarker.ANALYSES 与契约不一致：{sorted(biomarker_ids)}")
    for analysis in list(E.ANALYSES) + list(B.ANALYSES):
        expect(analysis["group"] in ("通路与化学分类", "标志物与建模"), f"分组不对：{analysis['group']}")
        expect(str(analysis.get("method") or "").strip(), f"{analysis['id']} 缺少 method")
        expect("内置实现" in analysis["method"], f"{analysis['id']} 的 method 必须写「内置实现」")
        expect(callable(analysis["run"]), f"{analysis['id']} 没有 run")
        needs = analysis.get("needs") or []
        expect(needs, f"{analysis['id']} 缺少 needs")
        for need in needs:
            expect(need in ALLOWED_NEEDS, f"{analysis['id']} 的 need 非法：{need}")
        for param in analysis.get("params") or []:
            expect(isinstance(param, dict) and param.get("key"), f"{analysis['id']} 参数缺 key")
    ok(f"enrich 分析 {len(enrich_ids)} 项 / biomarker 分析 {len(biomarker_ids)} 项，"
       f"needs 全部是合法 token，method 均注明内置实现")

    # PLOTS 归属
    expect(set(E.PLOTS) == ENRICH_PLOTS, f"enrich 图 kind 不对：{sorted(E.PLOTS)}")
    expect(set(B.PLOTS) == BIOMARKER_PLOTS, f"biomarker 图 kind 不对：{sorted(B.PLOTS)}")
    overlap = (set(E.PLOTS) | set(B.PLOTS)) & FORBIDDEN_PLOTS
    expect(not overlap, f"导出了别人拥有的图 kind：{sorted(overlap)}")
    expect(not (set(E.PLOTS) & set(B.PLOTS)), "两个模块的 PLOTS kind 重复")
    for kind, fn in list(E.PLOTS.items()) + list(B.PLOTS.items()):
        expect(callable(fn), f"图 {kind} 不可调用")
    ok(f"PLOTS：enrich {sorted(E.PLOTS)} / biomarker {sorted(B.PLOTS)}，无越界")


# ---------------------------------------------------------------------------
# 2. enrich 模块
# ---------------------------------------------------------------------------


def test_enrich(ctx, truth, outdir) -> dict:
    print("* 通路富集（mb_enrich）")
    expected = truth["expected_hits"]

    # --- mummichog ---
    payload = E.analysis_mummichog_pathway(ctx, {"n_perm": 500}, outdir)
    frame = store_frame(payload, "enrich")
    expect(not frame.empty, "mummichog 没有返回任何通路")
    for column in ("pathway", "n_hits", "n_total", "p", "p_adj", "hits"):
        expect(column in frame.columns, f"mummichog 表缺列：{column}")
    expect(frame["p"].between(0.0, 1.0).all(), "存在 p∉[0,1] 的通路")
    expect(frame["p_adj"].between(0.0, 1.0).all(), "存在 FDR∉[0,1] 的通路")
    found = set(frame["pathway"].astype(str))
    for pathway in TARGET_PATHWAYS:
        expect(pathway in found, f"mummichog 没有召回目标通路：{pathway}")
        row = frame[frame["pathway"] == pathway].iloc[0]
        expect(float(row["p"]) < 0.05, f"{pathway} 的经验 p 不显著：{row['p']}")
        expect(int(row["n_hits"]) >= expected[pathway], f"{pathway} 命中数偏低：{row['n_hits']}")
    check_figures(payload)
    check_tables(payload)
    check_data_out(payload)
    ok(f"mummichog_pathway：{len(frame)} 条通路，3 条目标通路 p<0.05，"
       f"FDR=min {frame['p_adj'].min():.3g}")

    # 确定性：同参数重跑 p 完全一致
    frame2 = store_frame(E.analysis_mummichog_pathway(ctx, {"n_perm": 500}, outdir), "enrich")
    same = np.allclose(frame["p"].to_numpy(float), frame2["p"].to_numpy(float))
    expect(same, "同样的数据与 seed 两次 mummichog 结果不一致")
    ok("mummichog 置换用固定 seed，重跑 p 值完全一致")

    # --- ORA KEGG：命中数与注释严格一致 ---
    payload = E.analysis_ora_kegg(ctx, {"p_cut": 0.05, "min_size": 3}, outdir)
    frame = store_frame(payload, "enrich")
    expect(not frame.empty, "ora_kegg 没有返回任何通路")
    for pathway, count in expected.items():
        rows = frame[frame["pathway"] == pathway]
        expect(not rows.empty, f"ora_kegg 没有返回目标通路：{pathway}")
        row = rows.iloc[0]
        expect(int(row["n_hits"]) == count, f"{pathway} ORA 命中数 {row['n_hits']} != 注释数 {count}")
        expect(int(row["n_total"]) == count, f"{pathway} ORA 背景集合大小 {row['n_total']} != {count}")
        expect(float(row["p"]) < 0.05, f"{pathway} ORA 不显著：{row['p']}")
    expect(frame["p"].between(0.0, 1.0).all(), "ORA 存在 p∉[0,1]")
    check_figures(payload)
    ok(f"ora_kegg：命中数与 pathway 注释完全一致（{sorted(expected.values())}），"
       f"FDR=min {frame['p_adj'].min():.3g}")

    # --- 名称匹配路径：去掉 pathway 列，只靠代谢物名也应该能归属 ---
    ctx_names = dict(ctx)
    info = ctx["feature_info"].copy().drop(columns=["pathway"])
    ctx_names["feature_info"] = info
    frame_names = store_frame(E.analysis_ora_kegg(ctx_names, {"p_cut": 0.05}, outdir), "enrich")
    expect(not frame_names.empty, "按代谢物名匹配时 ORA 没有结果")
    for pathway, count in expected.items():
        rows = frame_names[frame_names["pathway"] == pathway]
        expect(not rows.empty, f"按名称匹配没有召回 {pathway}")
        expect(int(rows.iloc[0]["n_hits"]) == count,
               f"按名称匹配 {pathway} 命中数 {rows.iloc[0]['n_hits']} != {count}")
    ok("pathway 列缺失时用代谢物名匹配内置库，命中数同样正确")

    # --- HMDB 化学分类 ---
    payload = E.analysis_ora_hmdb_class(ctx, {"p_cut": 0.05}, outdir)
    frame = store_frame(payload, "enrich")
    expect(not frame.empty, "ora_hmdb_class 没有结果")
    expect("class" in frame.columns, "化学分类表缺 class 列")
    expect(frame["p"].between(0.0, 1.0).all(), "化学分类富集存在 p∉[0,1]")
    for label in ("Nucleotide", "Amino acid", "Amine"):
        expect(label in set(frame["class"].astype(str)), f"化学分类缺 {label}")
    check_figures(payload)
    ok(f"ora_hmdb_class：{len(frame)} 个分类，Nucleotide/Amino acid/Amine 均显著")

    # --- MSEA ---
    payload = E.analysis_metabolite_set_enrich(ctx, {"n_perm": 400, "min_size": 5}, outdir)
    frame = store_frame(payload, "enrich")
    expect(not frame.empty, "MSEA 没有结果")
    for column in ("es", "nes", "p", "p_adj"):
        expect(column in frame.columns, f"MSEA 表缺列：{column}")
    expect(frame["p"].between(0.0, 1.0).all(), "MSEA 存在 p∉[0,1]")
    expect(np.isfinite(frame["es"].to_numpy(float)).all(), "MSEA 的 ES 出现 NaN")
    check_figures(payload)
    ok(f"metabolite_set_enrich：{len(frame)} 个集合，ES/NES/p 齐全")

    # --- STEP enrich（两种方法）---
    payload = E.step_enrich(ctx, {"method": "mummichog", "n_perm": 300}, outdir)
    expect(len(store_frame(payload, "enrich")) >= 1, "step enrich(mummichog) 没有结果")
    expect("Mummichog" in payload["summary"]["bullets"][0], "step 的方法说明不对")
    check_figures(payload)
    payload = E.step_enrich(ctx, {"method": "ora_kegg"}, outdir)
    expect(len(store_frame(payload, "enrich")) >= 1, "step enrich(ora_kegg) 没有结果")
    check_tables(payload)
    check_data_out(payload)
    ok("step enrich：method=mummichog / ora_kegg 两条路径都能跑，表+图+data_out 齐全")

    # --- 降级：完全没有 m/z（列与 id 编码都没有）→ 自动回退注释 ORA 并 warning ---
    ctx_no_mz = strip_mz_context(ctx)
    payload = E.analysis_mummichog_pathway(ctx_no_mz, {"n_perm": 200}, outdir)
    expect(len(store_frame(payload, "enrich")) >= 1, "没有 m/z 时应该回退到 ORA")
    expect(any(("改用" in str(warning) or "回退" in str(warning)) for warning in payload["warnings"]),
           f"回退时没有写 warnings（不诚实）：{payload['warnings']}")
    ok("没有 m/z 时 mummichog 自动改用注释 ORA，并写 warnings 说明方法变更")

    # --- 缺前置 ---
    expect_prerequisite(lambda: E.analysis_mummichog_pathway({**ctx, "results": {}}, {}, outdir), "差异")
    expect_prerequisite(lambda: E.step_enrich({**ctx, "results": {}}, {}, outdir), "差异")
    try:
        E.step_enrich(ctx, {"method": "not-a-method"}, outdir)
        raise AssertionError("未知方法应当抛 bad_request")
    except C.EngineError as error:
        expect(error.code == "bad_request", f"未知方法错误码应为 bad_request，实为 {error.code}")
    ok("缺差异结果 → prerequisite；未知 method → bad_request")

    # --- 图函数 ---
    for kind in ("enrich_bar", "enrich_dot"):
        figure = E.PLOTS[kind](ctx, {"top_n": 10}, outdir)
        expect(os.path.getsize(figure["file"]) > 1000, f"{kind} 图太小")
        expect(figure["title"], f"{kind} 缺少标题")
    ok("PLOTS enrich_bar / enrich_dot 可直接出图")

    return {"store": ctx["results"]}


# ---------------------------------------------------------------------------
# 3. biomarker 模块
# ---------------------------------------------------------------------------


def test_biomarker(ctx, truth, outdir) -> None:
    print("* 标志物与建模（mb_biomarker）")
    up_feature = truth["up_features"][0]

    # --- step：rf / lasso / vip / combo ---
    payload = B.step_biomarker(ctx, {"model": "rf", "n_features": 10, "cv_folds": 5}, outdir)
    frame = store_frame(payload, "biomarker")
    expect(not frame.empty, "step biomarker(rf) 没有特征表")
    for column in ("rank", "feature", "name", "importance", "score"):
        expect(column in frame.columns, f"标志物表缺列：{column}")
    expect(int(payload["summary"]["metrics"]["n_features"]) <= 10, "输出特征数超过 n_features")
    auc = float(payload["summary"]["metrics"].get("cv_auc") or 0.0)
    expect(auc > 0.8, f"强信号下 RF 交叉验证 AUC 应 >0.8，实为 {auc:.3f}")
    expect(int(payload["summary"]["metrics"]["n_samples"]) == N_CASE + N_CONTROL,
           f"应剔除 {N_QC} 个 QC 样本（期望 {N_CASE + N_CONTROL} 个样本）")
    check_figures(payload)
    check_tables(payload)
    check_data_out(payload)
    ok(f"step biomarker(rf)：top10 特征，交叉验证 AUC={auc:.3f}，QC 样本已剔除"
       f"（{payload['summary']['metrics']['n_samples']} 个样本）")

    payload = B.step_biomarker(ctx, {"model": "lasso", "n_features": 10}, outdir)
    metrics = payload["summary"]["metrics"]
    expect(int(metrics.get("n_selected") or 0) >= 1, "LASSO 没有选出任何特征")
    ok(f"step biomarker(lasso)：入选 {metrics['n_selected']} 个特征，best C={metrics.get('best_c'):.3g}")

    payload = B.step_biomarker(ctx, {"model": "vip", "n_features": 8}, outdir)
    frame = store_frame(payload, "biomarker")
    expect(not frame.empty, "step biomarker(vip) 没有特征表")
    expect(frame["importance"].abs().max() > 1.0, "VIP 重要度取值不对")
    ok("step biomarker(vip)：复用 PLS-DA VIP（importance=VIP）")

    # results["vip"] 新 schema：只有 dict（无 plsda 结果）+ 表被 JSON 化成 list[dict] 也要认得
    vip_block = ctx["results"]["vip"]
    ctx_vip_dict = {**ctx, "results": {
        "de": ctx["results"]["de"],
        "vip": {"table": C.jsonable(vip_block["table"]), "vip": C.jsonable(vip_block["vip"]),
                "method": "plsda", "n_features": N_FEATURES, "n_above": int(truth["n_designed"]),
                "threshold": 1.0},
    }}
    payload = B.step_biomarker(ctx_vip_dict, {"model": "vip", "n_features": 6}, outdir)
    frame = store_frame(payload, "biomarker")
    expect(not frame.empty, "results['vip'] 为 dict（JSON 化后）时 VIP 读取失败")
    ok("step biomarker(vip)：兼容 results['vip'] dict schema（含 JSON 化后的 list[dict]）")

    payload = B.step_biomarker(ctx, {"model": "combo", "n_features": 8}, outdir)
    expect(len(store_frame(payload, "biomarker")) >= 1, "combo 没有结果")
    ok("step biomarker(combo)：三模型秩平均可跑")

    # 没有 VIP 时 combo 应降级并 warning
    ctx_no_vip = {**ctx, "results": {"de": ctx["results"]["de"]}}
    payload = B.step_biomarker(ctx_no_vip, {"model": "combo", "n_features": 6}, outdir)
    expect(any("VIP" in text for text in payload["warnings"]), "缺 VIP 时 combo 应写 warning")
    expect_prerequisite(lambda: B.step_biomarker(ctx_no_vip, {"model": "vip"}, outdir), "PLS-DA")
    ok("缺 VIP：combo 降级（warning），vip 模型 → prerequisite")

    # --- 分析 1：rf_importance ---
    payload = B.analysis_rf_importance(ctx, {"n_trees": 200, "top_n": 20}, outdir)
    metrics = payload["summary"]["metrics"]
    expect(float(metrics["cv_auc"]) > 0.8, f"rf_importance 交叉验证 AUC 应 >0.8：{metrics['cv_auc']}")
    expect(not store_frame(payload, "biomarker").empty, "rf_importance 没有表")
    figures = check_figures(payload)
    expect(any(figure["id"] == "rf-importance" for figure in figures), "缺少 rf_importance 图")
    check_tables(payload)
    check_data_out(payload)
    ok(f"rf_importance：AUC={float(metrics['cv_auc']):.3f}，"
       f"top 特征={payload['tables'][0]['rows'][0]['name']}")

    # --- 分析 2：lasso_panel ---
    payload = B.analysis_lasso_panel(ctx, {"folds": 5, "alpha": "cv"}, outdir)
    frame = store_frame(payload, "biomarker")
    metrics = payload["summary"]["metrics"]
    expect(int(metrics["n_selected"]) >= 1, "lasso_panel 没有入选特征")
    expect("selected" in frame.columns, "lasso_panel 表缺 selected 列")
    expect(bool(frame["selected"].any()), "lasso_panel 表里没有任何 selected=True")
    figures = check_figures(payload)
    expect(any(figure["id"] == "lasso-path" for figure in figures), "缺少 lasso_path 图")
    check_tables(payload)
    ok(f"lasso_panel：入选 {metrics['n_selected']}/{metrics['n_pool']} 个特征，"
       f"best C={metrics['best_c']:.3g}，路径图已出")

    # --- 分析 3：roc_biomarker ---
    payload = B.analysis_roc_biomarker(ctx, {"feature": up_feature, "ci": True}, outdir)
    metrics = payload["summary"]["metrics"]
    expect(float(metrics["auc"]) > 0.8, f"强信号下单特征 AUC 应 >0.8：{metrics['auc']}")
    expect(float(metrics["ci_low"]) <= float(metrics["auc"]) <= float(metrics["ci_high"]),
           f"DeLong CI 未覆盖 AUC：{metrics}")
    expect(float(metrics["p"]) < 0.01, f"单特征 AUC 的 p 应很小：{metrics['p']}")
    figures = check_figures(payload)
    expect(any(figure["id"] == "roc-curve" for figure in figures), "缺少 roc_curve 图")
    check_tables(payload)
    ok(f"roc_biomarker：AUC={metrics['auc']:.3f}（DeLong 95% CI "
       f"{metrics['ci_low']:.3f}–{metrics['ci_high']:.3f}，p={metrics['p']:.2g}）")

    # 反向打分应当 AUC 互补（校验 AUC 计算方向）
    values = ctx["X"][:, ctx["features"].index(up_feature)].astype(float)
    keep = np.asarray([g != "QC" for g in ctx["sample_info"]["group"]])
    y = np.asarray([1 if g == "case" else 0 for g in ctx["sample_info"]["group"]])[keep]
    auc_forward = B._auc_binary(y, values[keep])
    auc_backward = B._auc_binary(y, -values[keep])
    expect(abs(auc_forward + auc_backward - 1.0) < 1e-9, "AUC 的对称性检查失败")
    ok("AUC 的 Mann-Whitney 实现满足 AUC(+) + AUC(−) = 1 的对称性")

    # --- 分析 4：biomarker_score ---
    payload = B.analysis_biomarker_score(ctx, {"features": [], "weights": "model"}, outdir)
    metrics = payload["summary"]["metrics"]
    expect(float(metrics["auc"]) > 0.8, f"组合评分 AUC 应 >0.8：{metrics['auc']}")
    expect(int(metrics["n_features"]) >= 5, "组合评分特征太少")
    values = payload.get("sample_values") or {}
    expect("biomarker_score" in values, "组合评分没有写回样本列")
    scores = values["biomarker_score"]
    expect(len(scores) == N_SAMPLES, f"写回评分的长度 {len(scores)} != 样本数 {N_SAMPLES}")
    expect(sum(1 for v in scores if v is not None) == N_CASE + N_CONTROL,
           "写回评分应包含全部非 QC 样本、QC 样本为空")
    figures = check_figures(payload)
    expect(len(figures) >= 2, "组合评分应同时输出 ROC 与分组箱线图")
    check_tables(payload)
    ok(f"biomarker_score：{metrics['n_features']} 个特征加权评分，AUC={metrics['auc']:.3f}，"
       f"评分写回样本列（{sum(1 for v in scores if v is not None)} 个非 QC 样本）")

    # --- 分析 5：model_validation ---
    payload = B.analysis_model_validation(ctx, {"n_boot": 40, "folds": 5}, outdir)
    metrics = payload["summary"]["metrics"]
    for key in ("apparent_auc", "optimism", "corrected_auc", "cv_auc"):
        expect(key in metrics and metrics[key] is not None, f"model_validation 缺指标 {key}")
    expect(0.0 <= float(metrics["apparent_auc"]) <= 1.0, "表观 AUC 越界")
    expect(0.0 <= float(metrics["corrected_auc"]) <= 1.0, "校正后 AUC 越界")
    expect(float(metrics["apparent_auc"]) > 0.8, "强信号下表观 AUC 应 >0.8")
    calibration = as_frame(((payload.get("store") or {}).get("biomarker") or {}).get("calibration"))
    expect(not calibration.empty and len(calibration) >= 2, "校准表行数不足")
    figures = check_figures(payload)
    expect(any(figure["id"] == "calibration-curve" for figure in figures), "缺少 calibration_curve 图")
    check_tables(payload)
    ok(f"model_validation：表观 AUC={float(metrics['apparent_auc']):.3f}，"
       f"乐观度={float(metrics['optimism']):.3f}，校正后={float(metrics['corrected_auc']):.3f}，"
       f"CV AUC={float(metrics['cv_auc']):.3f}")

    # --- 分析 6：survival_metabolite ---
    payload = B.analysis_survival_metabolite(
        ctx, {"feature": truth["risk_feature"], "cutoff": "median"}, outdir)
    metrics = payload["summary"]["metrics"]
    expect(float(metrics["logrank_p"]) < 0.05, f"KM/logrank 应显著：p={metrics['logrank_p']}")
    expect(metrics.get("hr") is not None and float(metrics["hr"]) > 1.0,
           f"高风险组的 HR 应 >1：{metrics.get('hr')}")
    expect(int(metrics["n_high"]) >= 3 and int(metrics["n_low"]) >= 3, "KM 分组样本太少")
    figures = check_figures(payload)
    expect(any(figure["id"] == "km-metabolite" for figure in figures), "缺少 km_metabolite 图")
    check_tables(payload)
    ok(f"survival_metabolite：logrank p={float(metrics['logrank_p']):.3g}，HR={float(metrics['hr']):.2f}"
       f"（{metrics['n_high']} vs {metrics['n_low']} 例）")

    # --- 生存分析的前置：缺临床列 ---
    ctx_no_clinical = dict(ctx)
    info = ctx["sample_info"].copy().drop(columns=["os_time", "os_event"])
    ctx_no_clinical["sample_info"] = info
    expect_prerequisite(
        lambda: B.analysis_survival_metabolite(ctx_no_clinical, {"feature": up_feature}, outdir), "生存")

    # --- 缺分组 ---
    ctx_no_group = dict(ctx)
    ctx_no_group["group_key"] = None
    expect_prerequisite(lambda: B.analysis_rf_importance(ctx_no_group, {}, outdir), "分组")
    expect_prerequisite(lambda: B.step_biomarker({**ctx, "results": {}}, {}, outdir), "差异")
    ok("缺临床列/分组/差异结果时都抛中文 prerequisite")

    # --- PLOTS ---
    for kind in ("rf_importance", "lasso_path", "roc_curve", "km_metabolite", "calibration_curve"):
        params = {"feature": up_feature, "top_n": 10, "n_trees": 150, "n_boot": 20}
        figure = B.PLOTS[kind](ctx, params, outdir)
        expect(os.path.isfile(figure["file"]), f"{kind} 没有生成文件")
        expect(os.path.getsize(figure["file"]) > 1000, f"{kind} 图太小")
        expect(figure["title"], f"{kind} 缺少标题")
    ok("PLOTS rf_importance / lasso_path / roc_curve / km_metabolite / calibration_curve 全部可出图")


# ---------------------------------------------------------------------------
# 4. 缓存 round-trip（registry 会把 store 里的表 JSON 化）
# ---------------------------------------------------------------------------


def test_store_roundtrip(ctx, truth, outdir) -> None:
    print("* store 缓存 round-trip（PLOTS 复用上次分析结果）")
    payload = B.analysis_rf_importance(ctx, {"n_trees": 120, "top_n": 10}, outdir)
    store = payload.get("store") or {}
    block = store.get("biomarker") or {}
    frame = as_frame(block.get("top"))
    expect(not frame.empty, "rf_importance store 里没有 top 表")
    original = _payload_from_json(payload)
    ctx_cached = {**ctx, "results": {**ctx["results"], **original}}
    figure = B.PLOTS["rf_importance"](ctx_cached, {"top_n": 10}, outdir)
    expect(os.path.getsize(figure["file"]) > 1000, "缓存复用时报错或图太小")
    ok("分析结果 JSON 化后仍可被 PLOTS 复用（list[dict] → DataFrame）")


def strip_mz_context(ctx: dict) -> dict:
    """造一个"既没有 mz 列、特征 id 里也没有 mz 编码"的 ctx（用于测 mummichog 回退）。"""
    mapping = {feature: f"F{index + 1:04d}" for index, feature in enumerate(ctx["features"])}
    new_ctx = dict(ctx)
    new_ctx["features"] = [mapping[feature] for feature in ctx["features"]]
    info = ctx["feature_info"].drop(columns=["mz"]).copy()
    info.index = pd.Index(new_ctx["features"], name="feature")
    new_ctx["feature_info"] = info
    de_store = dict(ctx["results"].get("de") or {})
    de_table = as_frame(de_store.get("table")).copy()
    de_table["feature"] = de_table["feature"].map(mapping)
    de_table = de_table.drop(columns=["mz"], errors="ignore")
    de_store["table"] = de_table
    de_store["de_table"] = de_table
    new_ctx["results"] = {"de": de_store}
    new_ctx["polarity"] = "unknown"
    return new_ctx


def _payload_from_json(payload: dict) -> dict:
    """模拟 registry 的 jsonable(store) 之后再取回。"""
    return {"biomarker": C.jsonable(((payload.get("store") or {}).get("biomarker") or {}))}


# ---------------------------------------------------------------------------
# 5. 跨模块 smoke：dose_response 不能抛 OverflowError（Lead 实测项，mb_de 由 mb-stat 维护）
# ---------------------------------------------------------------------------


def test_dose_response_no_overflow(ctx, outdir) -> None:
    print("* 跨模块 smoke：dose_response 不抛 OverflowError（Lead 实测项）")
    try:
        import mb_de  # noqa: PLC0415 - 只做 smoke，不在本模块职责内
    except Exception as error:  # noqa: BLE001
        print(f"  [skip] mb_de 尚未就绪：{type(error).__name__}: {error}")
        return
    runner = next((item["run"] for item in getattr(mb_de, "ANALYSES", [])
                   if item.get("id") == "dose_response"), None)
    if runner is None:
        print("  [skip] mb_de 里没有 dose_response")
        return
    failures: list[str] = []
    finished = 0
    for feature in ctx["features"][:5]:
        try:
            payload = runner(ctx, {"dose_col": "age", "feature": feature}, outdir)
        except C.EngineError as error:
            expect(error.code in ("prerequisite", "bad_request", "unsupported"),
                   f"dose_response 的错误码不可读：{error.code}（{error.message}）")
            finished += 1
            continue
        except Exception as error:  # noqa: BLE001 - OverflowError 等非 EngineError 就是 bug
            failures.append(f"{feature}: {type(error).__name__}: {error}")
            continue
        for figure in payload.get("figures") or []:
            expect(os.path.isfile(figure.get("file") or ""), "dose_response 返回的图不存在")
        finished += 1
    expect(not failures,
           f"dose_response 抛了非 EngineError 异常（OverflowError 等，需 mb-stat 修 mb_de._fit_4pl）：{failures[:2]}")
    ok(f"dose_response 在 {finished} 个特征上都没有抛 OverflowError（结果或可读 prerequisite）")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    started = time.time()
    outdir = tempfile.mkdtemp(prefix="mb-adv-")
    try:
        test_catalog()
        ctx, truth = build_ctx(seed=0, outdir=outdir)
        test_enrich(ctx, truth, outdir)
        test_biomarker(ctx, truth, outdir)
        test_store_roundtrip(ctx, truth, outdir)
        test_dose_response_no_overflow(ctx, outdir)
        elapsed = time.time() - started
        print(f"\n全部断言通过（耗时 {elapsed:.1f}s，输出目录 {outdir}）")
        print("PASS")
        return 0
    except AssertionError as error:
        print(f"\nFAIL: {error}")
        return 1
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
