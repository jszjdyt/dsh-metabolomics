#!/usr/bin/env python3
"""
py_prep_test.py — mb-base（mb_prep.py + mb_plots.py）确定性自测（DESIGN §11）

不依赖 mb_state.py / mb_io.py（那两位由别的组在写），自己造一个最小 ctx：
  72 样本（case 30 / control 30 / QC 12）× 1600 特征
  * 120 个注入差异特征（case 上 log2FC 0.8–2.0，正负各半）
  * 200 个批次效应特征（B2 上调/下调 ~25%）
  * 每样本独立稀释因子（ln SD 0.20）→ 用于验证 PQN
  * 5%–10% 随机缺失（实测 7.5%）
  * 样本表：group / batch / qc / age / bmi；特征表：mz / rt / name / class / kegg

断言：
  1. 6 个步骤依次跑通，每步信封合法（summary.bullets 非空、tables/figures 是 list、图文件 >1000B）
  2. impute 后无 NaN
  3. normalize=PQN 后 TIC 的 CV 明显下降
  4. pca 前两个主成分能把 case/control 分开（AUC > 0.6）
  5. 10 个 PLOTS 都能出 PNG（size > 1000）
  6. 5 个 ANALYSES 都能返回合法信封，且 method 标注“内置实现”
  7. registry 集成（可选）：用自造 Dataset 影子对象走 registry.run_step 六步，
     验证 replace/store/sample_columns 写回契约。registry 未就绪时跳过并说明。

运行：
  MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d) python3 test/py_prep_test.py
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import traceback

# matplotlib 只能用 Agg；缓存目录必须是可写临时目录（~/.matplotlib 不可写）
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.mkdtemp(prefix="mb-mpl-"), "cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mb_common as C  # noqa: E402
import mb_plots  # noqa: E402
import mb_prep  # noqa: E402

# ---------------------------------------------------------------------------
# 常量：与 DESIGN §9 的 demo 规模一致（72 × 1600）
# ---------------------------------------------------------------------------

N_CASE, N_CTRL, N_QC = 30, 30, 12
N_SAMPLES = N_CASE + N_CTRL + N_QC
N_FEATURES = 1600
N_DIFF = 120
N_BATCH_EFFECT = 200
NAN_FRAC = 0.075
SEED = 0

STEP_ORDER = ["qc", "filter", "impute", "normalize", "transform", "pca"]
STEP_PARAMS = {
    "qc": {"qc_label": "QC"},
    "filter": {"min_sample_frac": 0.5},
    "impute": {"method": "half_min"},
    "normalize": {"method": "pqn"},
    "transform": {"transform": "log2", "scale": "auto"},
    "pca": {"n_comp": 3, "color_by": "group"},
}
ANALYSES_TO_RUN = [
    ("qc_summary", {}),
    ("rsd_qc", {"qc_label": "QC"}),
    ("missing_pattern", {"max_features": 200}),
    ("sample_outlier", {"n_comp": 3, "alpha": 0.95}),
    ("sample_corr", {"method": "pearson"}),
]
PLOT_KINDS = ["qc_tic", "qc_rsd", "qc_missing", "sample_corr", "sample_dendro",
              "pca_scores", "pca_variance", "hotelling_t2", "heatmap_top", "batch_pca"]


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def make_dataset(seed: int = SEED):
    """造 72×1600 的模拟代谢组学矩阵 + 样本表 + 特征表；返回 (X, features, samples, feature_info, sample_info)。"""
    rng = np.random.default_rng(seed)
    samples = ([f"CASE_{index + 1:02d}" for index in range(N_CASE)]
               + [f"CTRL_{index + 1:02d}" for index in range(N_CTRL)]
               + [f"QC_{index + 1:02d}" for index in range(N_QC)])
    groups = np.array(["case"] * N_CASE + ["control"] * N_CTRL + ["QC"] * N_QC)
    # 批次与分组交错分配：避免批次与分组完全混杂（也保证两类样本两个批次都有）
    batches = np.array(["B1" if index % 2 == 0 else "B2" for index in range(N_SAMPLES)])
    qc_flag = (groups == "QC").astype(int)

    base_mean = rng.uniform(8.0, 13.0, size=N_FEATURES)
    base_sd = rng.uniform(0.12, 0.45, size=N_FEATURES)

    diff_index = rng.choice(N_FEATURES, size=N_DIFF, replace=False)
    sign = np.where(rng.random(N_DIFF) < 0.5, -1.0, 1.0)
    diff_log2fc = rng.uniform(0.8, 2.0, size=N_DIFF) * sign
    diff_effect = np.zeros(N_FEATURES)
    diff_effect[diff_index] = diff_log2fc * np.log(2.0)

    batch_index = rng.choice(np.setdiff1d(np.arange(N_FEATURES), diff_index),
                             size=N_BATCH_EFFECT, replace=False)
    batch_effect = np.zeros(N_FEATURES)
    batch_effect[batch_index] = np.log(1.25) * np.where(rng.random(N_BATCH_EFFECT) < 0.5, -1.0, 1.0)

    dilution = np.exp(rng.normal(0.0, 0.20, size=N_SAMPLES))
    is_case = (groups == "case").astype(float)
    is_batch2 = (batches == "B2").astype(float)
    is_qc = (groups == "QC").astype(float)

    log_matrix = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    for index in range(N_SAMPLES):
        noise_sd = 0.06 if is_qc[index] else 1.0
        noise = rng.normal(0.0, base_sd * noise_sd)
        log_matrix[index] = (base_mean + noise + batch_effect * is_batch2[index]
                            + diff_effect * is_case[index])
    log_matrix += np.log(dilution)[:, None]
    X = np.exp(log_matrix)

    mask = rng.random((N_SAMPLES, N_FEATURES)) < NAN_FRAC
    X[mask] = np.nan

    rt = rng.uniform(0.4, 16.0, size=N_FEATURES)
    mz = rng.uniform(90.0, 1000.0, size=N_FEATURES)
    features = []
    seen = set()
    for index in range(N_FEATURES):
        ident = f"mz{mz[index]:.4f}_rt{rt[index]:.2f}"
        while ident in seen:            # 极少数四舍五入碰撞：加后缀保证唯一
            ident = ident + "x"
        seen.add(ident)
        features.append(ident)
    classes = np.array(["Amino acid", "Lipid", "Bile acid", "Carbohydrate",
                        "Nucleotide", "Organic acid", "Amine", "Other"])
    feature_info = pd.DataFrame({
        "mz": mz,
        "rt": rt,
        "name": [f"Metabolite_{index + 1:04d}" for index in range(N_FEATURES)],
        "formula": [None] * N_FEATURES,
        "adduct": [None] * N_FEATURES,
        "hmdb": [None] * N_FEATURES,
        "kegg": [f"cpd:C{index + 1:05d}" for index in range(N_FEATURES)],
        "class": [classes[index % len(classes)] for index in range(N_FEATURES)],
        "polarity": ["pos"] * N_FEATURES,
        "confidence": [2 if index < N_DIFF else None for index in range(N_FEATURES)],
    }, index=pd.Index(features, name="feature"))

    sample_info = pd.DataFrame({
        "group": groups,
        "batch": batches,
        "qc": qc_flag,
        "age": rng.integers(30, 70, size=N_SAMPLES),
        "bmi": np.round(rng.normal(24.0, 3.0, size=N_SAMPLES), 1),
    }, index=pd.Index(samples, name="sample"))
    return X, features, samples, feature_info, sample_info, groups


def make_ctx(workdir: str):
    X, features, samples, feature_info, sample_info, groups = make_dataset()
    ctx = {
        "dataset_id": "ds1",
        "name": "synthetic-metabo",
        "features": list(features),
        "samples": list(samples),
        "X": np.asarray(X, dtype=np.float32),
        "X_kind": "raw",
        "raw": np.asarray(X, dtype=np.float32),
        "feature_info": feature_info.copy(),
        "sample_info": sample_info.copy(),
        "group_key": "group",
        "groups": list(dict.fromkeys(groups.tolist())),
        "matrix_order": "samples_by_features",
        "species": "human",
        "polarity": "pos",
        "results": {},
        "library": None,
        "rng": np.random.default_rng(SEED),
        "workdir": workdir,
        "outdir": os.path.join(workdir, "out"),
        "engine": {"python": sys.version.split()[0], "mode": "builtin"},
        "notes": [],
        "warnings": [],
    }
    return ctx


# ---------------------------------------------------------------------------
# 断言小工具
# ---------------------------------------------------------------------------


def check_envelope(payload, label: str) -> None:
    """校验 §2.5 结果信封：summary.bullets 非空、tables/figures 结构合法、图文件存在且 >1000B。"""
    assert isinstance(payload, dict), f"{label}: 返回不是 dict"
    summary = payload.get("summary")
    assert isinstance(summary, dict), f"{label}: 缺少 summary dict"
    bullets = summary.get("bullets")
    assert isinstance(bullets, list) and len(bullets) > 0, f"{label}: summary.bullets 为空"
    assert all(isinstance(item, str) and item for item in bullets), f"{label}: bullets 不是非空字符串"
    assert isinstance(summary.get("metrics"), dict), f"{label}: summary.metrics 缺失"
    for key in ("tables", "figures", "warnings", "notes"):
        assert isinstance(payload.get(key), list), f"{label}: {key} 不是 list"
    for entry in payload["tables"]:
        assert isinstance(entry.get("columns"), list) and entry["columns"], f"{label}: table 缺少 columns"
        assert isinstance(entry.get("rows"), list), f"{label}: table 缺少 rows"
        for column in entry["columns"]:
            assert column.get("key") and column.get("label"), f"{label}: table 列缺少 key/label"
    for entry in payload["figures"]:
        path = entry.get("file")
        assert path and os.path.exists(path), f"{label}: 图文件不存在 {path}"
        size = os.path.getsize(path)
        assert size > 1000, f"{label}: 图文件太小 {path} ({size}B)"


def check_figure(fig, label: str) -> int:
    assert isinstance(fig, dict), f"{label}: PLOTS 返回不是 dict"
    path = fig.get("file")
    assert path and os.path.exists(path), f"{label}: 图文件不存在 {path}"
    size = os.path.getsize(path)
    assert size > 1000, f"{label}: 图文件太小 ({size}B)"
    return int(size)


def auc_score(values, positive) -> float:
    """Mann-Whitney AUC（方向无关：<0.5 时取 1-AUC）。"""
    values = np.asarray(values, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    auc = (float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def apply_payload(ctx, payload) -> None:
    """把步骤返回的 replace / store / sample_columns 应用到测试用 ctx（模拟 registry §2.5 消费）。"""
    replace = payload.get("replace") or {}
    if replace.get("X") is not None:
        ctx["X"] = np.asarray(replace["X"], dtype=np.float32)
    if replace.get("features") is not None:
        ctx["features"] = [str(item) for item in replace["features"]]
    if replace.get("feature_info") is not None:
        frame = replace["feature_info"]
        if isinstance(frame, pd.DataFrame):
            ctx["feature_info"] = frame
    if replace.get("raw") is not None:
        ctx["raw"] = np.asarray(replace["raw"], dtype=np.float32)
    kind = replace.get("X_kind") or replace.get("value_kind")
    if kind:
        ctx["X_kind"] = str(kind)
    for key, value in (payload.get("store") or {}).items():
        ctx["results"][str(key)] = value
    columns = payload.get("sample_columns") or []
    values = payload.get("sample_values") or {}
    if columns and values:
        info = ctx["sample_info"]
        for entry in columns:
            name = entry.get("key") if isinstance(entry, dict) else entry
            series = values.get(name) if name else None
            if series is not None and len(series) == len(info):
                info[name] = list(series)
        ctx["sample_info"] = info


# ---------------------------------------------------------------------------
# 1. 步骤链
# ---------------------------------------------------------------------------


def check_step_effects(ctx, payloads) -> dict:
    """步骤语义断言 + 关键指标打印。"""
    facts = {}

    # --- impute：填补后必须无 NaN
    X = np.asarray(ctx["X"], dtype=np.float64)
    assert np.all(np.isfinite(X)), "impute 后矩阵里仍有 NaN/Inf"
    before = payloads["impute"]["summary"]["metrics"]["n_missing_before"]
    assert before > 0, "测试数据本应有缺失值"
    facts["missing_before"] = before
    facts["missing_after"] = int((~np.isfinite(X)).sum())
    print(f"[check] impute   : 缺失 {before} -> {facts['missing_after']}（无 NaN ✓）")

    # --- normalize：PQN 后 TIC 的 CV 下降
    metrics = payloads["normalize"]["summary"]["metrics"]
    cv_before = float(metrics["tic_cv_before"])
    cv_after = float(metrics["tic_cv_after"])
    assert cv_after < cv_before, f"PQN 后 TIC 的 CV 没有下降：{cv_before:.2f}% -> {cv_after:.2f}%"
    assert cv_after < 0.8 * cv_before, f"PQN 后 TIC 的 CV 下降不明显：{cv_before:.2f}% -> {cv_after:.2f}%"
    facts["tic_cv_before"], facts["tic_cv_after"] = cv_before, cv_after
    print(f"[check] normalize: TIC CV {cv_before:.1f}% -> {cv_after:.1f}%（下降 ✓）")

    # --- transform：X_kind=scaled
    assert ctx["X_kind"] == "scaled", f"transform 后 X_kind 应为 scaled，实际 {ctx['X_kind']}"
    print(f"[check] transform: X_kind={ctx['X_kind']} "
          f"(|skew| {payloads['transform']['summary']['metrics']['median_abs_skew_before']:.2f}"
          f" -> {payloads['transform']['summary']['metrics']['median_abs_skew_after']:.2f})")

    # --- pca：得分/方差/载荷齐全，且 case/control 可分
    stored = ctx["results"].get("pca")
    assert isinstance(stored, dict), "pca 步骤没有写 store['pca']"
    for key in ("scores", "ratio", "loadings"):
        assert stored.get(key) is not None, f"store['pca'] 缺少 {key}"
        # result() 会把 store 过一遍 jsonable（ndarray → list），两种形态都要能读
        np.asarray(stored[key], dtype=np.float64)
    scores = np.asarray(stored["scores"], dtype=np.float64)
    ratio = np.asarray(stored["ratio"], dtype=np.float64)
    assert scores.shape[0] == N_SAMPLES and scores.shape[1] >= 2
    # ratio 只含保留的前 n_comp 个主成分（不是全部），所以和 ≤ 1 且单调下降
    assert 0.0 < float(ratio[0]) <= 1.0, f"PC1 解释方差异常：{ratio[0]}"
    assert float(ratio.sum()) <= 1.0 + 1e-6, f"解释方差和 > 1：{ratio.sum()}"
    assert np.all(np.diff(ratio) <= 1e-9), f"解释方差不是单调下降：{ratio}"
    groups = np.asarray([str(value) for value in ctx["sample_info"]["group"].tolist()])
    biological = groups != "QC"
    positive = groups[biological] == "case"
    sub = scores[biological]
    auc_pc1 = auc_score(sub[:, 0], positive)
    auc_pc2 = auc_score(sub[:, 1], positive)
    best = max(auc_pc1, auc_pc2)
    assert best > 0.6, f"PCA 前两主成分分不开 case/control：AUC PC1={auc_pc1:.3f} PC2={auc_pc2:.3f}"
    facts["auc"] = best
    facts["pc1_pct"] = float(ratio[0] * 100)
    print(f"[check] pca      : PC1 {ratio[0] * 100:.1f}% / PC1+PC2 {ratio[:2].sum() * 100:.1f}%；"
          f"AUC(PC1)={auc_pc1:.3f} AUC(PC2)={auc_pc2:.3f} -> max {best:.3f}（>0.6 ✓）")

    # --- qc：QC 样本被识别，RSD 表有值
    qc_metrics = payloads["qc"]["summary"]["metrics"]
    assert qc_metrics["n_qc"] == N_QC, f"QC 样本识别数不对：{qc_metrics['n_qc']}"
    assert payloads["qc"]["summary"]["metrics"]["median_rsd_pct"] < 30.0, "QC RSD 中位异常"
    print(f"[check] qc       : 识别 QC {qc_metrics['n_qc']} 个，QC-RSD 中位 "
          f"{qc_metrics['median_rsd_pct']:.1f}%")
    return facts


# ---------------------------------------------------------------------------
# 2. PLOTS
# ---------------------------------------------------------------------------


def check_qc_plots_after_transform(ctx, outdir_root: str) -> None:
    """回归 bug（Lead 实测）：transform 之后 qc 系列图若回退到 ctx["X"] 会把标度值当峰面积，
    算出中位 RSD=239% 这种荒谬数字。必须是 raw 数据源，且中位 RSD < 100%。"""
    poisoned = copy.copy(ctx)
    poisoned["results"] = copy.deepcopy(ctx["results"])
    # 人为制造缓存长度不匹配（filter 掉特征后就是这个情形）→ 强制走回退分支
    poisoned["results"]["qc"]["rsd"] = [0.0, 0.0, 0.0]
    block = mb_plots._qc_block(poisoned, {})
    rsd = np.asarray(block["rsd"], dtype=np.float64)
    finite = rsd[np.isfinite(rsd)]
    assert block.get("source") == "raw", (
        f"qc 系列图的数据源必须回退到 ctx['raw']，实际 {block.get('source')}")
    assert finite.size > 0, "回退后没有可用的 RSD"
    median_rsd = float(np.median(finite))
    assert median_rsd < 100.0, (
        f"transform 之后 qc_rsd 的中位 RSD={median_rsd:.1f}%（明显不合理：可能用了标度后的 X）")
    tic = np.asarray(block["tic"], dtype=np.float64)
    assert np.nanmedian(tic) > 0, f"回退后 TIC 中位应 > 0，实际 {np.nanmedian(tic)}"
    outdir = os.path.join(outdir_root, "plot-qc_rsd_after_transform")
    os.makedirs(outdir, exist_ok=True)
    fig = mb_plots.PLOTS["qc_rsd"](poisoned, {"dpi": 110}, outdir)
    check_figure(fig, "plot:qc_rsd_after_transform")
    print(f"[check] qc_rsd   : transform 后回退 raw，中位 RSD={median_rsd:.1f}%（<100% ✓），"
          f"TIC 中位={np.nanmedian(tic):.3g}")


def run_plots(ctx, outdir_root: str) -> None:
    assert set(mb_plots.PLOTS) == set(PLOT_KINDS), (
        f"PLOTS 与 DESIGN §5 基础图清单不一致：多 {set(mb_plots.PLOTS) - set(PLOT_KINDS)}，"
        f"少 {set(PLOT_KINDS) - set(mb_plots.PLOTS)}")
    for kind in PLOT_KINDS:
        outdir = os.path.join(outdir_root, f"plot-{kind}")
        os.makedirs(outdir, exist_ok=True)
        params = {"top_n": 20, "color_by": "group", "dpi": 120}
        fig = mb_plots.PLOTS[kind](copy.copy(ctx), params, outdir)
        size = check_figure(fig, f"plot:{kind}")
        print(f"[plot] {kind:<14} {size:>7}B  {os.path.basename(fig['file'])}")


# ---------------------------------------------------------------------------
# 3. ANALYSES
# ---------------------------------------------------------------------------


def run_analyses(ctx, outdir_root: str, missing_ctx) -> None:
    entries = {item["id"]: item for item in mb_prep.ANALYSES}
    assert set(entries) == {item[0] for item in ANALYSES_TO_RUN}, (
        f"ANALYSES 清单不一致：{sorted(entries)}")
    for analysis_id, params in ANALYSES_TO_RUN:
        entry = entries[analysis_id]
        assert entry.get("group") == "质控与基础", f"{analysis_id}: group 应为「质控与基础」"
        method = str(entry.get("method") or "")
        assert method.startswith("内置实现（依据"), f"{analysis_id}: method 未标注内置实现/文献依据"
        assert callable(entry.get("run")), f"{analysis_id}: 缺少 run"
        target = missing_ctx if analysis_id == "missing_pattern" else ctx
        outdir = os.path.join(outdir_root, f"an-{analysis_id}")
        os.makedirs(outdir, exist_ok=True)
        payload = entry["run"](copy.copy(target), dict(params), outdir)
        check_envelope(payload, f"analysis:{analysis_id}")
        metrics = payload["summary"].get("metrics") or {}
        if analysis_id == "qc_summary":
            assert float(metrics.get("median_tic") or 0) > 0, "qc_summary 的 TIC 中位应 > 0（说明用了 raw）"
        if analysis_id == "rsd_qc":
            assert float(metrics.get("median_rsd_qc") or 999) < 100.0, (
                f"transform 之后 rsd_qc 的中位 RSD 不合理：{metrics.get('median_rsd_qc')}")
        print(f"[analysis] {analysis_id:<16} bullets={len(payload['summary']['bullets'])} "
              f"tables={len(payload['tables'])} figures={len(payload['figures'])} | "
              f"{payload['summary']['bullets'][0][:64]}")


# ---------------------------------------------------------------------------
# 4. 诚实失败 / 无分组降级
# ---------------------------------------------------------------------------


def run_edge_cases(ctx_pre_impute, ctx_final) -> None:
    """阈值过严 → prerequisite；无分组列 → pca 降级为单色并给 warning（不崩）。"""
    try:
        mb_prep.step_filter(copy.copy(ctx_pre_impute), {"min_sample_frac": 1.0}, tempfile.mkdtemp())
    except C.EngineError as error:
        assert error.code == "prerequisite", f"过滤全灭时应为 prerequisite，实际 {error.code}"
        print(f"[edge] filter 全灭 -> prerequisite ✓：{str(error.message)[:56]}")
    else:
        raise AssertionError("过滤阈值过严时没有抛 prerequisite")

    no_group = copy.copy(ctx_final)          # 用变换后的 ctx（否则 pca 会因 X_kind=raw 拒绝）
    no_group["sample_info"] = ctx_final["sample_info"].drop(columns=["group"])
    no_group["group_key"] = None
    no_group["groups"] = None
    payload = mb_prep.step_pca(no_group, {"n_comp": 3, "color_by": "group"}, tempfile.mkdtemp())
    check_envelope(payload, "edge:pca_no_group")
    assert payload["warnings"], "没有分组列时 pca 应该给 warning（降级为单色）"
    print(f"[edge] pca 无分组 -> warning ✓：{payload['warnings'][0][:56]}")


# ---------------------------------------------------------------------------
# 4b. 参数分支冒烟：每个 method / transform / scale 分支至少跑一次
# ---------------------------------------------------------------------------


def run_branch_smoke(ctx_initial) -> None:
    """覆盖 impute/normalize/transform 的全部 method 分支 + filter 的 QC-RSD 分支 + pca 着色分支。"""
    steps = {item["id"]: item for item in mb_prep.STEPS}
    root = tempfile.mkdtemp(prefix="mb-branch-")
    raw_ctx = copy.copy(ctx_initial)
    raw_ctx["results"] = {}

    # impute 四个方法：都必须在有缺失的矩阵上跑通且填完无 NaN
    for method in ("half_min", "median", "knn", "ppca"):
        outdir = os.path.join(root, f"impute-{method}")
        os.makedirs(outdir, exist_ok=True)
        payload = steps["impute"]["run"](copy.copy(raw_ctx), {"method": method, "k": 3}, outdir)
        check_envelope(payload, f"branch:impute:{method}")
        filled = np.asarray(payload["replace"]["X"], dtype=np.float64)
        assert np.all(np.isfinite(filled)), f"impute={method} 后有 NaN"
        print(f"[branch] impute method={method:<8} -> 无 NaN ✓")

    # 先做一份无缺失矩阵，供 normalize / transform 分支使用
    filled_ctx = copy.copy(raw_ctx)
    filled_ctx["X"] = np.asarray(steps["impute"]["run"](
        copy.copy(raw_ctx), {"method": "half_min"}, os.path.join(root, "impute-base"))["replace"]["X"],
        dtype=np.float32)

    # filter 的 QC-RSD 分支
    qc_filter = steps["filter"]["run"](copy.copy(raw_ctx),
                                       {"min_sample_frac": 0.3, "rsd_max": 40}, root)
    check_envelope(qc_filter, "branch:filter:rsd_max")
    n_keep = int(qc_filter["summary"]["metrics"]["n_features_after"])
    assert 10 <= n_keep <= N_FEATURES, f"QC-RSD 过滤后特征数异常：{n_keep}"
    print(f"[branch] filter rsd_max=40 -> {N_FEATURES} → {n_keep} 特征 ✓")

    # normalize 五个方法
    norm_ctx = copy.copy(filled_ctx)
    norm_ctx["results"] = {}
    for method in ("pqn", "tic", "median", "quantile", "none"):
        outdir = os.path.join(root, f"normalize-{method}")
        os.makedirs(outdir, exist_ok=True)
        payload = steps["normalize"]["run"](copy.copy(norm_ctx), {"method": method}, outdir)
        check_envelope(payload, f"branch:normalize:{method}")
        assert np.all(np.isfinite(np.asarray(payload["replace"]["X"], dtype=np.float64)))
        print(f"[branch] normalize method={method:<9} -> 无 NaN ✓")

    # transform × scale 分支
    for transform in ("log2", "log10", "cuberoot", "none"):
        for scale in ("auto", "pareto", "range", "none"):
            outdir = os.path.join(root, f"transform-{transform}-{scale}")
            os.makedirs(outdir, exist_ok=True)
            payload = steps["transform"]["run"](copy.copy(filled_ctx),
                                                {"transform": transform, "scale": scale}, outdir)
            check_envelope(payload, f"branch:transform:{transform}:{scale}")
            kind = payload["replace"].get("X_kind")
            # P0-1 三值语义（BIO_P0_BACKLOG §4）：有标度 → scaled；仅 log → lognorm；
            # 仅 cuberoot → transformed；transform+scale 都为 none 时不写 X_kind。
            if transform == "none" and scale == "none":
                assert kind is None, "transform+scale 都为 none 时不应强行写 X_kind"
            elif scale != "none":
                assert kind == "scaled", f"transform={transform} scale={scale} 后 X_kind 应为 scaled"
            elif transform in ("log2", "log10"):
                assert kind == "lognorm", f"仅 {transform} 未标度时 X_kind 应为 lognorm，得到 {kind}"
            else:
                assert kind == "transformed", f"仅 {transform} 未标度时 X_kind 应为 transformed，得到 {kind}"
            # P0-1：三层快照必须随 transform 步骤写入 store
            layers = (payload.get("store") or {}).get("transform", {}).get("layers") or {}
            values = layers.get("values") or {}
            assert isinstance(values, dict) and values.get("model") is not None, \
                f"transform={transform} scale={scale} 未保存 model 层快照"
    print("[branch] transform × scale 16 个组合 -> 全部跑通 ✓")

    # pca 的 color_by 分支（batch / qc / none）：先变换到 scaled（pca 会拒绝 raw）
    os.makedirs(os.path.join(root, "transform-base"), exist_ok=True)
    scaled = steps["transform"]["run"](copy.copy(filled_ctx),
                                       {"transform": "log2", "scale": "auto"},
                                       os.path.join(root, "transform-base"))
    check_envelope(scaled, "branch:transform:base")
    pca_ctx = copy.copy(filled_ctx)
    pca_ctx["X"] = np.asarray(scaled["replace"]["X"], dtype=np.float32)
    pca_ctx["X_kind"] = "scaled"
    pca_ctx["results"] = {}
    for color_by in ("batch", "qc", "none"):
        outdir = os.path.join(root, f"pca-{color_by}")
        os.makedirs(outdir, exist_ok=True)
        payload = steps["pca"]["run"](copy.copy(pca_ctx),
                                      {"n_comp": 4, "color_by": color_by}, outdir)
        check_envelope(payload, f"branch:pca:{color_by}")
        print(f"[branch] pca color_by={color_by:<6} -> {len(payload['figures'])} 图 ✓")


# ---------------------------------------------------------------------------
# 5. registry 集成（可选：registry 由 Host 组维护，未就绪则跳过）
# ---------------------------------------------------------------------------


def run_registry_pass(ctx_initial, workdir: str) -> None:
    try:
        import mb_registry as registry
    except Exception as error:  # noqa: BLE001
        print(f"[registry] 跳过（{type(error).__name__}: {error}）")
        return
    from types import SimpleNamespace

    dataset = SimpleNamespace(
        id="ds1",
        name="synthetic-metabo",
        path=None,
        source_format="matrix_text",
        X=np.asarray(ctx_initial["X"], dtype=np.float32).copy(),
        raw=np.asarray(ctx_initial["raw"], dtype=np.float32).copy(),
        X_kind="raw",
        value_kind="raw",
        features=list(ctx_initial["features"]),
        genes=list(ctx_initial["features"]),
        samples=list(ctx_initial["samples"]),
        feature_info=ctx_initial["feature_info"].copy(),
        sample_info=ctx_initial["sample_info"].copy(),
        matrix_order="samples_by_features",
        species="human",
        polarity="pos",
        library=None,
        group_key="group",
        groups=list(ctx_initial["groups"]),
        steps=[],
        history=[],
        analyses_run=[],
        results={},
        notes=[],
        warnings=[],
        current=None,
    )
    summary = registry.module_status()
    prep = summary.get("mb_prep") or {}
    if not prep.get("ok"):
        print(f"[registry] 跳过（mb_prep 未注册：{prep.get('error')}）")
        return
    plots_module = summary.get("mb_plots") or {}
    registered = set(plots_module.get("plots") or [])
    conflicts = registry.load_modules().get("conflicts") or []
    missing = set(PLOT_KINDS) - registered
    # DESIGN §1 把 heatmap_top 归 mb_plots；若被别的模块抢注（registry 先加载 MODULES），
    # 只告警不判失败——这属于跨模块归属问题，由 Lead 协调。
    assert missing <= {"heatmap_top"}, (
        f"registry 没拿到这些基础图：{sorted(missing)}")
    if missing:
        print(f"[registry] 注意：{sorted(missing)} 被其它模块抢注；conflicts={conflicts}")

    for step in STEP_ORDER:
        payload = registry.run_step(dataset, step, dict(STEP_PARAMS[step]), workdir)
        assert payload.get("step") == step
        assert payload["summary"]["bullets"], f"registry {step}: bullets 为空"
        assert step in (payload.get("steps") or []), f"registry {step}: dataset.steps 没记录"
    assert dataset.X_kind == "scaled", f"registry 集成后 X_kind={dataset.X_kind}"
    assert not np.isnan(np.asarray(dataset.X, dtype=np.float64)).any(), "registry 集成后仍有 NaN"
    assert set(dataset.steps) == set(STEP_ORDER), f"registry steps 不完整：{dataset.steps}"
    assert len(dataset.features) == np.asarray(dataset.X).shape[1]
    assert "PC1" in dataset.sample_info.columns, "pca 的 sample_columns 没写回 sample_info"
    print(f"[registry] 六步 run_step 全通：steps={dataset.steps} X_kind={dataset.X_kind} "
          f"shape={np.asarray(dataset.X).shape} sample_info+={list(dataset.sample_info.columns)[:6]}…")

    # 图分发：registry.run_plot 按 kind 找 owner 并返回带 kind/bytes 的 dict
    plot_payload = registry.run_plot(dataset, "pca_scores", {"color_by": "group"}, workdir)
    assert plot_payload.get("kind") == "pca_scores"
    assert os.path.getsize(plot_payload["file"]) > 1000
    print(f"[registry] run_plot('pca_scores') -> {plot_payload['bytes']}B")


# ---------------------------------------------------------------------------
# 6. 真实 demo 数据冒烟（mb_state 由别的组维护：就绪才跑，断言保持宽松）
# ---------------------------------------------------------------------------


def run_demo_smoke(workdir: str) -> None:
    """在 mb_state.make_demo 的真实 demo 上跑 6 步 + QC 图，验证跨模块集成（可选）。"""
    try:
        import mb_registry as registry
        import mb_state
        state = mb_state.State()
        dataset = mb_state.make_demo(state)
    except Exception as error:  # noqa: BLE001 - mb_state 未就绪则跳过
        print(f"[demo] 跳过（{type(error).__name__}: {error}）")
        return
    n_samples, n_features = int(dataset.n_samples), int(dataset.n_features)
    if (n_samples, n_features) != (N_SAMPLES, N_FEATURES):
        print(f"[demo] 跳过：demo 规模变成 {n_samples}×{n_features}（本测试按 DESIGN §9 的 72×1600 描述）")
        return
    for step in STEP_ORDER:
        payload = registry.run_step(dataset, step, dict(STEP_PARAMS[step]), workdir)
        assert payload["summary"]["bullets"], f"demo {step}: bullets 为空"
    x = np.asarray(dataset.X, dtype=np.float64)
    assert not np.isnan(x).any(), "demo 流程后仍有 NaN"
    assert dataset.X_kind == "scaled"
    # transform 之后再画 qc_rsd：必须回退 raw（Lead 实测的 bug 场景）
    outdir = os.path.join(workdir, "demo-qc-rsd")
    os.makedirs(outdir, exist_ok=True)
    ctx = registry.build_ctx(dataset, outdir)
    block = mb_plots._qc_block(ctx, {})
    rsd = np.asarray(block["rsd"], dtype=np.float64)
    finite = rsd[np.isfinite(rsd)]
    assert block.get("source") == "raw"
    assert finite.size and float(np.median(finite)) < 100.0
    fig = registry.run_plot(dataset, "qc_rsd", {"dpi": 120}, workdir)
    assert os.path.getsize(fig["file"]) > 1000
    print(f"[demo] mb_state demo {n_samples}×{n_features} 六步全通；transform 后 qc_rsd 用 raw，"
          f"中位 RSD={float(np.median(finite)):.1f}%（<100% ✓）")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("mb-base 自测：mb_prep.py（6 步 + 5 分析） + mb_plots.py（10 张基础图）")
    print(f"Python {sys.version.split()[0]} / numpy {np.__version__} / pandas {pd.__version__}")
    print("=" * 78)
    root = tempfile.mkdtemp(prefix="mb-prep-test-")
    workdir = os.path.join(root, "work")
    os.makedirs(workdir, exist_ok=True)

    ctx_initial = make_ctx(workdir)
    X = np.asarray(ctx_initial["X"], dtype=np.float64)
    n_missing = int((~np.isfinite(X)).sum())
    frac = n_missing / X.size * 100.0
    print(f"[data] {X.shape[0]} 样本 × {X.shape[1]} 特征；NaN {n_missing} 个（{frac:.2f}%）；"
          f"分组 {dict(ctx_initial['sample_info']['group'].value_counts())}；"
          f"批次 {dict(ctx_initial['sample_info']['batch'].value_counts())}")
    assert 5.0 <= frac <= 12.0, f"缺失率应在 5–10% 附近，实际 {frac:.2f}%"

    ctx = copy.copy(ctx_initial)
    ctx["results"] = {}
    payloads = {}
    steps = {item["id"]: item for item in mb_prep.STEPS}
    for index, step in enumerate(STEP_ORDER):
        outdir = os.path.join(root, f"step-{step}")
        os.makedirs(outdir, exist_ok=True)
        payload = steps[step]["run"](ctx, dict(STEP_PARAMS[step]), outdir)
        check_envelope(payload, f"step:{step}")
        apply_payload(ctx, payload)
        payloads[step] = payload
        print(f"[step {index + 1}/6] {step:<9} bullets={len(payload['summary']['bullets'])} "
              f"tables={len(payload['tables'])} figures={len(payload['figures'])} | "
              f"{payload['summary']['bullets'][0][:76]}")
        if step == "filter":
            missing_ctx = copy.copy(ctx)       # 保留“有缺失”的快照给 missing_pattern 测试
    facts = check_step_effects(ctx, payloads)

    run_edge_cases(missing_ctx, ctx)
    run_branch_smoke(ctx_initial)
    check_qc_plots_after_transform(ctx, root)
    run_plots(ctx, root)
    run_analyses(ctx, root, missing_ctx)
    run_registry_pass(ctx_initial, os.path.join(root, "registry"))
    run_demo_smoke(os.path.join(root, "demo"))

    print("-" * 78)
    print(f"SUMMARY  missing {facts['missing_before']} -> {facts['missing_after']}; "
          f"TIC CV {facts['tic_cv_before']:.1f}% -> {facts['tic_cv_after']:.1f}%; "
          f"PC1 {facts['pc1_pct']:.1f}%; AUC {facts['auc']:.3f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"FAIL: {error}")
        traceback.print_exc()
        sys.exit(1)
