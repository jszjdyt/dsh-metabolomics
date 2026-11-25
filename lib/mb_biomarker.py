"""
dsh-metabolomics — 标志物筛选与建模（DESIGN.md §3 step biomarker / §4 group=标志物与建模 / §5 图）

本模块提供：
  STEPS   : biomarker（model=rf | lasso | vip | combo）
  ANALYSES: rf_importance / lasso_panel / roc_biomarker / biomarker_score /
            model_validation / survival_metabolite
  PLOTS   : rf_importance / lasso_path / roc_curve / km_metabolite / calibration_curve

方法（全部内置实现；本机没有 MetaboAnalystR / ropls / glmnet，sklearn + lifelines 已装）：
  * RandomForest 分类 + 置换重要度（准确度/ROC-AUC 下降）——Breiman 2001。
  * L1 逻辑回归（LASSO）+ 交叉验证选 C ——Tibshirani 1996；正则化路径画入选特征数与系数。
  * PLS-DA VIP 直接复用上游结果（Wold 1995；来自 ctx["results"]["vip"] 或
    ctx["results"]["plsda"]["vip"]），combo = 三种重要度的秩平均（稳健融合思想）。
  * ROC/AUC + DeLong 置信区间（DeLong, DeLong & Clarke-Pearson 1988 的置换值方差公式，
    内置实现；样本量过大时退回 Hanley-McNeil 近似并标注）。
  * 自助法乐观度校正 + 校准曲线 ——Harrell 1996（bootstrap optimism）；校准曲线为
    交叉验证外推预测概率的分位分箱 vs 观测事件率。
  * KM / logrank / Cox：优先 lifelines（Therneau 2000），不可用或拟合失败时用内置
    Newton-Raphson（Breslow ties）单变量 Cox 并标注来源。
  * 多标志物线性评分：标准化后按权重加权（weights=model 用 L2 逻辑回归系数）。

约定（DESIGN §2.3/§2.5）：
  * 只读 ctx；结果用 mb_common.result()/table()/save_fig() 构造；图上文字一律英文。
  * 缺前置抛 prerequisite；参数非法抛 bad_request；单个可选步骤（如 Cox）失败只写 warnings。
  * 分组比较前会剔除 QC 样本（sample_info["qc"]==1 或 group 取值像 QC），避免稀释统计。
"""

from __future__ import annotations

import math
import os
import warnings as _warnings

import numpy as np
import pandas as pd

from mb_common import (
    PALETTE,
    EngineError,
    bad_request,
    bh_fdr,
    data_file,
    export_csv,
    new_fig,
    palette,
    prerequisite,
    result,
    save_fig,
    table,
)

SEED = 0
DEFAULT_N_FEATURES = 10
DEFAULT_CV_FOLDS = 5
MAX_POOL = 20000          # DESIGN §2.3：特征数上限
MAX_LASSO_POOL = 2000     # LASSO 前先按单变量信号筛到这个规模（否则 saga 太慢）
MAX_BOOT = 500
N_PERM_REPEATS = 5        # RF 置换重要度的重复次数

QC_TOKENS = {"qc", "quality control", "pool", "pooled", "blank", "blank qc", "mix", "qualitycontrol"}


# ---------------------------------------------------------------------------
# 1. ctx 读取小工具（只依赖 mb_common）
# ---------------------------------------------------------------------------


def _features(ctx) -> list[str]:
    return [str(item) for item in (ctx.get("features") or [])]


def _samples(ctx) -> list[str]:
    return [str(item) for item in (ctx.get("samples") or [])]


def _info(ctx) -> pd.DataFrame:
    info = ctx.get("sample_info")
    if isinstance(info, pd.DataFrame):
        return info
    return pd.DataFrame(index=pd.Index(_samples(ctx), name="sample"))


def _feature_info(ctx) -> pd.DataFrame:
    info = ctx.get("feature_info")
    if isinstance(info, pd.DataFrame):
        return info
    return pd.DataFrame(index=pd.Index(_features(ctx), name="feature"))


def _raw_X(ctx) -> np.ndarray:
    X = ctx.get("X")
    if X is None:
        raise prerequisite("数据集没有强度矩阵（X），请先导入特征表或载入示例数据")
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise EngineError("internal", f"强度矩阵维度异常：{arr.shape}（应为 样本×特征）")
    return arr


def _as_number(value, default=float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sqlite_bool(value) -> int:
    """事件列解析：1/0、True/False、dead/alive、yes/no、event 文本。"""
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    text = str(value).strip().lower()
    if text in ("1", "1.0", "true", "yes", "y", "dead", "death", "event", "deceased", "recurrence"):
        return 1
    if text in ("0", "0.0", "false", "no", "n", "alive", "censored", "censor", "living", "none"):
        return 0
    number = _as_number(value)
    if math.isfinite(number):
        return 1 if number > 0.5 else 0
    return 0


#: 事件列显式映射（BIO_P0_BACKLOG §4 P0-3）。只认下面这些显式取值，其余一律视为未知：
#: 旧实现把未知文本静默记成 0（删失），等于把"不知道"当成"没发生"，会系统性高估生存、低估风险。
EVENT_POSITIVE = ("1", "1.0", "true", "yes", "y", "dead", "death", "deceased", "event",
                  "recurrence", "recur", "relapse", "progressed", "progression")
EVENT_NEGATIVE = ("0", "0.0", "false", "no", "n", "alive", "censored", "censor", "living",
                  "none", "disease_free", "df", "no_event")


def _event_value(value) -> str | None:
    """把事件列原始值规整成小写文本；空/NaN/None 返回 None。"""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return str(int(number)) if number.is_integer() else str(number)
    text = str(value).strip().lower()
    if text in ("", "nan", "none", "null", "na", "n/a", "missing", "unknown", "?"):
        return None
    return text


def _event_code(value, event_map: dict[str, int] | None = None) -> int | None:
    """显式事件编码：返回 1/0；无法识别返回 None（调用方决定 bad_request 还是按 NA 丢弃）。"""
    text = _event_value(value)
    if text is None:
        return None
    if event_map:
        if text in event_map:
            mapped = event_map[text]
            return 1 if float(mapped) >= 0.5 else 0
        # 事件映射支持数值键（如 {1:1, 2:0}）
        try:
            number = int(float(text))
            if number in event_map:
                return 1 if float(event_map[number]) >= 0.5 else 0
        except (TypeError, ValueError):
            pass
    if text in EVENT_POSITIVE:
        return 1
    if text in EVENT_NEGATIVE:
        return 0
    return None


def _feature_column(ctx, column: str) -> dict[str, object]:
    """feature_info 的某列 → 特征 → 取值（缺列返回空 dict）。"""
    info = _feature_info(ctx)
    if column not in info.columns:
        return {}
    out: dict[str, object] = {}
    for feature in _features(ctx):
        try:
            if feature in info.index:
                out[feature] = info.at[feature, column]
        except (KeyError, IndexError):
            continue
    return out


def _feature_meta(ctx, features: list[str]) -> pd.DataFrame:
    """特征 → name/mz/rt/class 的表（缺列填空）。"""
    name_map = _feature_column(ctx, "name")
    mz_map = _feature_column(ctx, "mz")
    rt_map = _feature_column(ctx, "rt")
    class_map = _feature_column(ctx, "class")
    rows = []
    for feature in features:
        name = name_map.get(feature)
        rows.append({
            "feature": feature,
            "name": "" if name is None or str(name).lower() in ("nan", "none") else str(name),
            "mz": _as_number(mz_map.get(feature)),
            "rt": _as_number(rt_map.get(feature)),
            "class": "" if class_map.get(feature) is None else str(class_map.get(feature)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. 差异结果（pool 来源）与分组
# ---------------------------------------------------------------------------

DE_FEATURE_CANDIDATES = ("feature", "feature_id", "gene", "metabolite", "id", "name")
DE_FC_CANDIDATES = ("log2FC", "logFC", "log2_fc", "log2FoldChange", "log_fc")
DE_P_CANDIDATES = ("p", "P.Value", "pvalue", "p_value", "P")
DE_PADJ_CANDIDATES = ("p_adj", "adj.P.Val", "padj", "FDR", "fdr", "qvalue", "p.adjust", "P.adj")


def _de_frame(ctx) -> pd.DataFrame:
    """取 results['de'] 的表（兼容 DataFrame / dict(table) / list[dict]），统一列名。"""
    results = ctx.get("results")
    de = results.get("de") if isinstance(results, dict) else None
    if de is None:
        raise prerequisite("请先跑「单变量差异」（步骤 de）再筛选标志物")
    frame: pd.DataFrame | None = None
    if isinstance(de, pd.DataFrame):
        frame = de.copy()
    elif isinstance(de, dict):
        for key in ("table", "de_table", "tables", "rows", "de"):
            value = de.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                frame = value.copy()
                break
            if isinstance(value, list) and value and isinstance(value[0], dict):
                frame = pd.DataFrame(value)
                break
    if frame is None or frame.empty:
        raise prerequisite("差异分析结果为空，请重跑「单变量差异」")

    if "feature" not in frame.columns:
        renamed = False
        for cand in DE_FEATURE_CANDIDATES:
            if cand in frame.columns:
                frame = frame.rename(columns={cand: "feature"})
                renamed = True
                break
        if not renamed:
            index_name = str(frame.index.name or "feature")
            frame = frame.reset_index()
            frame = frame.rename(columns={index_name: "feature"} if index_name in frame.columns
                                else {frame.columns[0]: "feature"})
    renames: dict[str, str] = {}
    for target, cands in (("log2FC", DE_FC_CANDIDATES), ("p", DE_P_CANDIDATES), ("p_adj", DE_PADJ_CANDIDATES)):
        if target in frame.columns:
            continue
        for cand in cands:
            if cand in frame.columns:
                renames[cand] = target
                break
    if renames:
        frame = frame.rename(columns=renames)
    frame = frame.copy()
    frame["feature"] = frame["feature"].astype(str)
    for column in ("log2FC", "p", "p_adj", "mean_case", "mean_control"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "log2FC" not in frame.columns:
        frame["log2FC"] = 0.0
    if "p_adj" not in frame.columns and "p" in frame.columns:
        frame["p_adj"] = bh_fdr(frame["p"].to_numpy(float))
    if "p_adj" not in frame.columns:
        frame["p_adj"] = 1.0
    frame["log2FC"] = frame["log2FC"].fillna(0.0)
    frame["p_adj"] = frame["p_adj"].fillna(1.0)
    return frame.drop_duplicates("feature").reset_index(drop=True)


def _de_hint(ctx) -> dict:
    """差异步骤里记录的分组提示（case/control/fdr），用于确定 ROC 的正类。"""
    results = ctx.get("results")
    de = results.get("de") if isinstance(results, dict) else None
    if not isinstance(de, dict):
        return {}
    hint = {}
    for key in ("case", "control", "group_key", "fdr", "fc"):
        if de.get(key) is not None:
            hint[key] = de[key]
    return hint


def _sample_mask(ctx) -> np.ndarray:
    """分组比较用的样本掩码：剔除 QC 样本与空分组（DESIGN demo 的 group 含 QC 水平）。"""
    samples = _samples(ctx)
    info = _info(ctx)
    n = len(samples)
    mask = np.ones(n, dtype=bool)
    if n == 0:
        raise prerequisite("数据集没有样本")
    if "qc" in info.columns:
        values = pd.to_numeric(info["qc"], errors="coerce").fillna(0.0).to_numpy()
        if values.size == n:
            mask &= values.astype(float) < 0.5
    key = str(ctx.get("group_key") or "")
    if key and key in info.columns:
        labels = info[key].astype(str).str.strip()
        lower = labels.str.lower()
        bad = labels.isin(["", "nan", "none", "na", "None"]) | lower.isin(QC_TOKENS)
        if bad.size == n:
            mask &= ~bad.to_numpy()
    return mask


CASE_TOKENS = ("case", "tumor", "tumour", "disease", "patient", "treated", "treatment", "high", "case_group")
CONTROL_TOKENS = ("control", "ctrl", "healthy", "normal", "vehicle", "untreated", "low", "wt", "sham")


def _pick_case_control(ctx, params, groups: list[str]) -> tuple[str, str, str]:
    """确定二分类的正类（case）与负类（control）与依据。"""
    params = dict(params or {})
    for key in ("case", "positive"):
        value = str(params.get(key) or "").strip()
        if value in groups:
            neg = str(params.get("control") or "").strip()
            if neg not in groups:
                neg = next((g for g in groups if g != value), groups[0])
            return value, neg, "params 指定"
    hint = _de_hint(ctx)
    case_hint = str(hint.get("case") or "").strip()
    control_hint = str(hint.get("control") or "").strip()
    if case_hint in groups and control_hint in groups and case_hint != control_hint:
        return case_hint, control_hint, "差异分析的 case/control"
    if case_hint in groups and len(groups) == 2:
        other = next((g for g in groups if g != case_hint), None)
        if other:
            return case_hint, other, "差异分析的 case"
    for token in CASE_TOKENS:
        match = next((g for g in groups if token in g.lower()), None)
        if match:
            other = next((g for g in groups if g != match), None)
            if other:
                return match, other, "分组名关键词"
    for token in CONTROL_TOKENS:
        match = next((g for g in groups if token in g.lower()), None)
        if match:
            other = next((g for g in groups if g != match), None)
            if other:
                return other, match, "分组名关键词（对照组）"
    ordered = sorted(groups)
    return ordered[-1], ordered[0], "按字母序（建议显式指定 case/control）"


def _group_setup(ctx, params) -> dict:
    """样本级分组设置：剔除 QC 后的标签、二分类目标、多分类编码。"""
    params = dict(params or {})
    key = str(ctx.get("group_key") or "")
    info = _info(ctx)
    samples = _samples(ctx)
    if not key:
        raise prerequisite("还没有指定分组列：请先在样本表里把某列设为分组（至少 2 组）")
    if key not in info.columns:
        raise prerequisite(f"分组列「{key}」不在样本信息里，请重新设置分组列")
    mask = _sample_mask(ctx)
    labels = info[key].astype(str).str.strip().to_numpy()
    keep = np.asarray([bool(mask[i]) and labels[i] not in ("", "nan", "None") for i in range(len(labels))],
                      dtype=bool)
    kept_labels = labels[keep]
    groups = [str(g) for g in dict.fromkeys(kept_labels.tolist())]
    if len(groups) < 2:
        raise prerequisite(
            f"分组列「{key}」在剔除 QC 样本后只剩 {len(groups)} 组，至少需要 2 组才能建模"
        )
    n_qc = int((~mask).sum())
    pos, neg, basis = _pick_case_control(ctx, params, groups)
    y = np.asarray([1 if lab == pos else 0 for lab in kept_labels], dtype=int)
    multiclass = len(groups) > 2
    codes = np.asarray([groups.index(lab) for lab in kept_labels], dtype=int)
    return {
        "keep": keep, "labels": labels, "groups": groups, "pos": pos, "neg": neg,
        "basis": basis, "y": y, "codes": codes, "multiclass": multiclass, "n_qc": n_qc,
        "group_key": key, "samples": samples,
    }


def _binary_subset(setup: dict, params: dict, notes: list[str]) -> tuple[np.ndarray, np.ndarray, str, str]:
    """ROC/Cox 这类二分类分析用：返回 (样本掩码, 已对齐掩码的 0/1 标签, 正类, 负类)。

    多组时只保留正/负两类样本（掩码会变窄），标签始终满足 ``len(y) == keep.sum()``。
    """
    labels = setup["labels"]
    pos, neg = setup["pos"], setup["neg"]
    if not setup["multiclass"]:
        return setup["keep"], setup["y"], pos, neg
    keep = np.asarray([bool(setup["keep"][i]) and labels[i] in (pos, neg) for i in range(len(labels))],
                      dtype=bool)
    y = np.asarray([1 if labels[i] == pos else 0 for i in range(int(keep.sum()))], dtype=int)
    notes.append(
        f"分组有 {len(setup['groups'])} 类，本分析按二分类处理：{pos} vs {neg}（保留 {int(keep.sum())} 个样本）"
    )
    return keep, y, pos, neg


# ---------------------------------------------------------------------------
# 3. 矩阵准备（缺失填补 + 候选特征池）
# ---------------------------------------------------------------------------


def _matrix_subset(ctx, indices: np.ndarray, keep: np.ndarray, warnings: list[str],
                   impute: bool = True):
    """取候选特征子矩阵（按样本掩码筛选；默认中位数填补），返回 (X, 特征名)。

    ``impute=False`` 时保留 NaN，供嵌套交叉验证/自助法在**每个训练折内部**重新估计
    填补参数（P0-2：填补也是"看数据"的决策，不能在全样本上先做）。
    """
    X = _raw_X(ctx)
    features = _features(ctx)
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        raise prerequisite("候选特征为空：差异表与当前矩阵没有交集，请重跑差异分析")
    sub = X[keep][:, idx]
    n_imputed = int(np.isnan(sub).sum())
    if n_imputed and impute:
        medians = np.nanmedian(np.where(np.isfinite(sub), sub, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        rows, cols = np.where(~np.isfinite(sub))
        sub = sub.copy()
        sub[rows, cols] = medians[cols]
        frac = n_imputed / max(sub.size, 1)
        warnings.append(f"候选矩阵含 {n_imputed} 个缺失值（{frac * 100:.1f}%），已用特征中位数填补")
    names = [features[i] for i in idx]
    return np.ascontiguousarray(sub, dtype=np.float64), names


def _candidate_pool(ctx, params: dict, notes: list[str], warnings: list[str]):
    """候选特征池：优先"差异显著特征"，否则按 |log2FC| 取前若干；返回 (下标, 说明)。"""
    params = dict(params or {})
    features = _features(ctx)
    position = {f: i for i, f in enumerate(features)}
    p_cut = float(params.get("p_cut") or 0.05)
    pool_mode = str(params.get("pool") or "de").strip().lower()
    try:
        de = _de_frame(ctx)
        de = de[de["feature"].isin(position.keys())]
    except EngineError:
        de = pd.DataFrame(columns=["feature", "log2FC", "p_adj"])
        pool_mode = "all"
    if not de.empty and pool_mode != "all":
        sig = de[de["p_adj"] < p_cut].copy()
        sig["_abs"] = sig["log2FC"].abs()
        sig = sig.sort_values(["_abs", "p_adj"], ascending=[False, True])
        if len(sig) >= 5:
            chosen = [f for f in sig["feature"].tolist() if f in position]
            notes.append(f"候选特征池 = 差异显著特征（p_adj<{p_cut:g}，n={len(chosen)}）")
            return np.asarray([position[f] for f in chosen], dtype=int), "de"
        warnings.append(f"显著差异特征只有 {len(sig)} 个（<5），改用全部特征按 |log2FC| 排序")
    if not de.empty:
        ordered = de.reindex(de["log2FC"].abs().sort_values(ascending=False).index)
        chosen = [f for f in ordered["feature"].tolist() if f in position][:MAX_POOL]
        notes.append(f"候选特征池 = 全矩阵按 |log2FC| 排序取前 {len(chosen)} 个")
        return np.asarray([position[f] for f in chosen], dtype=int), "all"
    # 没有差异表可用时的最终回退：按方差取前 2000
    X = _raw_X(ctx)
    variance = np.nanvar(np.where(np.isfinite(X), X, np.nan), axis=0)
    variance = np.where(np.isfinite(variance), variance, -np.inf)
    order = np.argsort(-variance)[: min(2000, len(features))]
    notes.append(f"没有可用差异表，候选特征池 = 方差最大的前 {len(order)} 个特征")
    return np.asarray(order, dtype=int), "variance"


def _cap_pool(ctx, indices: np.ndarray, y_kept: np.ndarray, limit: int) -> np.ndarray:
    """候选池过大时按单变量 |Cohen's d| 预筛（只为了让 RF/LASSO 跑得动，notes 里会说明）。"""
    if indices.size <= limit:
        return indices
    X = _raw_X(ctx)
    y_kept = np.asarray(y_kept, dtype=int)
    if y_kept.size != X.shape[0]:
        raise EngineError("internal", "候选特征预筛的标签长度与样本数不一致")
    sub = X[:, indices]
    pos = sub[y_kept == 1]
    neg = sub[y_kept == 0]
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        return indices[:limit]
    mean_pos = np.nanmean(pos, axis=0)
    mean_neg = np.nanmean(neg, axis=0)
    sd_pos = np.nanvar(pos, axis=0, ddof=1)
    sd_neg = np.nanvar(neg, axis=0, ddof=1)
    pooled = np.sqrt(np.maximum((sd_pos + sd_neg) / 2.0, 0.0))
    score = np.where(pooled > 0, np.abs(mean_pos - mean_neg) / pooled,
                     np.where(np.isfinite(mean_pos - mean_neg), np.inf, -np.inf))
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:limit]
    return indices[np.sort(order)]


# ---------------------------------------------------------------------------
# 4. 重要度 / 模型
# ---------------------------------------------------------------------------


def _rf_model(X: np.ndarray, y: np.ndarray, n_trees: int = 300, seed: int = SEED):
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415 - sklearn 已装

    model = RandomForestClassifier(
        n_estimators=max(20, int(n_trees)), random_state=seed, n_jobs=1,
        min_samples_leaf=2, class_weight="balanced_subsample", max_features="sqrt",
    )
    model.fit(X, y)
    return model


def _rf_importance(model, X: np.ndarray, y: np.ndarray, multiclass: bool, seed: int = SEED):
    """RF 置换重要度（Breiman 2001 的 permutation importance；sklearn 的 O(n_repeats) 实现）。"""
    from sklearn.inspection import permutation_importance  # noqa: PLC0415

    scoring = "roc_auc" if (not multiclass and len(np.unique(y)) == 2) else "accuracy"
    try:
        output = permutation_importance(model, X, y, n_repeats=N_PERM_REPEATS,
                                       random_state=seed, scoring=scoring, n_jobs=1)
        mean = np.asarray(output.importances_mean, dtype=float)
        std = np.asarray(output.importances_std, dtype=float)
    except Exception:  # noqa: BLE001 - 置换重要度失败时退回不纯度重要度
        mean = np.asarray(model.feature_importances_, dtype=float)
        std = np.zeros_like(mean)
    return mean, std


def _cv_folds_for(y: np.ndarray, folds: int) -> int:
    """分层折数不能超过最小类样本数。"""
    counts = np.bincount(np.asarray(y, dtype=int))
    counts = counts[counts > 0]
    if counts.size < 2:
        raise prerequisite("目标变量只有 1 个类别，无法做交叉验证/ROC，请检查分组列")
    return int(max(2, min(int(folds), int(counts.min()))))


def _oof_proba(X: np.ndarray, y: np.ndarray, folds: int, make_model, multiclass: bool):
    """交叉验证外推概率（ROC/校准曲线用），返回 (proba, folds_used)。"""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: PLC0415

    folds_used = _cv_folds_for(y, folds)
    splitter = StratifiedKFold(n_splits=folds_used, shuffle=True, random_state=SEED)
    model = make_model()
    proba = cross_val_predict(model, X, y, cv=splitter, method="predict_proba", n_jobs=1)
    return np.asarray(proba, dtype=float), folds_used


def _auc_binary(y: np.ndarray, score: np.ndarray) -> float:
    """二分类 AUC（Mann-Whitney 秩公式）；退化时返回 NaN。"""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _delong(y: np.ndarray, score: np.ndarray, alpha: float = 0.05):
    """ROC-AUC 与 DeLong 1988 方差置信区间（内置实现，置换值 formulation）。

    Var(AUC) = Var(V10)/n_case + Var(V01)/n_control，其中
    V10 = 每个 case 相对全部 control 的 psi 均值、V01 = 每个 control 相对全部 case 的 psi 均值。
    样本量过大（> 4e6 对）时退回 Hanley-McNeil 近似并标注。
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    n_pos, n_neg = int(pos.size), int(neg.size)
    if n_pos < 2 or n_neg < 2:
        auc = _auc_binary(y, score)
        return {"auc": auc, "se": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "method": "样本不足，未做 DeLong CI"}
    if n_pos * n_neg > 4_000_000:
        auc = _auc_binary(y, score)
        q1 = auc / (2.0 - auc)
        q2 = 2.0 * auc * auc / (1.0 + auc)
        se = math.sqrt(
            (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc))
            / max(n_pos * n_neg, 1)
        )
        method = "Hanley-McNeil 近似（样本对过多，未逐对计算 DeLong 方差）"
    else:
        psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :]).astype(float)
        auc = float(psi.mean())
        v_case = psi.mean(axis=1)
        v_control = psi.mean(axis=0)
        var = float(np.var(v_case, ddof=1) / n_pos + np.var(v_control, ddof=1) / n_neg)
        se = math.sqrt(max(var, 0.0))
        method = "DeLong 1988（置换值方差公式，内置实现）"
    try:
        from scipy import stats as _stats  # noqa: PLC0415

        z_alpha = float(_stats.norm.ppf(1.0 - alpha / 2.0))
        z_stat = (auc - 0.5) / se if se > 0 else 0.0
        p_value = float(2.0 * _stats.norm.sf(abs(z_stat))) if se > 0 else float("nan")
    except Exception:  # noqa: BLE001 - 没有 scipy 时用正态近似
        z_alpha = 1.959963985
        z_stat = (auc - 0.5) / se if se > 0 else 0.0
        p_value = math.erfc(abs(z_stat) / math.sqrt(2.0)) if se > 0 else float("nan")
    if se <= 0:
        # 完全分离时 DeLong 标准误为 0：p 无定义，按"极显著"报告并明确标注
        p_value = 0.0
        method = method + "；SE=0（两类完全分离，CI 退化为点）"
    return {
        "auc": auc, "se": se, "lo": float(np.clip(auc - z_alpha * se, 0.0, 1.0)),
        "hi": float(np.clip(auc + z_alpha * se, 0.0, 1.0)), "p": p_value, "method": method,
    }


def _roc_points(y: np.ndarray, score: np.ndarray):
    """ROC 折线（FPR/TPR/阈值），去掉重复点。"""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    s_sorted = score[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos = max(int(y_sorted.sum()), 1)
    n_neg = max(int((1 - y_sorted).sum()), 1)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    thresholds = np.concatenate([[float("inf")], s_sorted])
    keep = np.ones(tpr.size, dtype=bool)
    keep[1:-1] = ~((fpr[1:-1] == fpr[:-2]) & (tpr[1:-1] == tpr[:-2]))
    return fpr[keep], tpr[keep], thresholds[keep]


def _vip_from_value(value, depth: int = 0) -> pd.Series | None:
    """把各种 VIP 容器（DataFrame / Series / dict / list[dict] / JSON 化后的结构）转成 Series。

    支持的 schema（与 mb-plsda 对齐）：
      * results["plsda"]["vip"] / ["table"]：DataFrame，列 rank/feature/name/mz/rt/vip/above_threshold；
      * results["vip"]：dict {"table": DataFrame, "vip": DataFrame, "method", ...}（registry 会把
        "table" JSON 化后还原、而 "vip" 键可能仍是 list[dict]，都要认）；
      * 兜底：{feature: vip} 的扁平 dict / Series。
    """
    if value is None or depth > 3:
        return None
    if isinstance(value, pd.Series):
        series = pd.to_numeric(value, errors="coerce").dropna()
        return series.astype(float) if series.size else None
    if isinstance(value, (pd.DataFrame, list)):
        frame = value if isinstance(value, pd.DataFrame) else (
            pd.DataFrame(value) if value and isinstance(value[0], dict) else pd.DataFrame())
        if frame.empty:
            return None
        if "feature" in frame.columns and "vip" in frame.columns:
            series = pd.Series(pd.to_numeric(frame["vip"], errors="coerce").to_numpy(),
                               index=frame["feature"].astype(str).to_numpy()).dropna()
            return series.astype(float) if series.size else None
        if "vip" in frame.columns:  # index 就是特征 id
            series = pd.to_numeric(frame["vip"], errors="coerce").dropna()
            return series.astype(float) if series.size else None
        return None
    if isinstance(value, dict):
        # 嵌套：{"table": ..., "vip": ..., "vip_table": ...} → 逐层试
        for key in ("vip", "table", "vip_table", "rows", "data", "scores"):
            inner = _vip_from_value(value.get(key), depth + 1)
            if inner is not None:
                return inner
        flat = {str(k): _as_number(v) for k, v in value.items()}
        series = pd.Series(flat).dropna()
        if series.size and np.isfinite(series.to_numpy(float)).any():
            return series.astype(float)
    return None


def _vip_series(ctx, notes: list[str]) -> pd.Series:
    """从 ctx['results'] 里取 VIP（results['plsda']['vip'] / results['vip']（dict 或表）/ 扁平字典）。"""
    results = ctx.get("results")
    if not isinstance(results, dict):
        raise prerequisite("请先跑「PLS-DA / OPLS-DA」（步骤 plsda）得到 VIP")
    candidates: list[tuple[str, object]] = []
    plsda = results.get("plsda")
    if isinstance(plsda, dict):
        candidates.extend([("results['plsda']['vip']", plsda.get("vip")),
                           ("results['plsda']['table']", plsda.get("table"))])
    candidates.extend([("results['vip']", results.get("vip")),
                       ("results['plsda_vip']", results.get("plsda_vip"))])
    for source, candidate in candidates:
        series = _vip_from_value(candidate)
        if series is not None and series.size:
            notes.append(f"VIP 来自 {source}（{series.size} 个特征）")
            return series
    raise prerequisite("请先跑「PLS-DA / OPLS-DA」（步骤 plsda）得到 VIP 再做 VIP 建模")


def _rank_average(series_list: list[pd.Series]) -> pd.Series:
    """秩平均融合：每个模型内按重要度降序给秩（1 最优），再取平均（越小越重要）。"""
    ranks = []
    for series in series_list:
        values = series.astype(float)
        if values.empty:
            continue
        ranks.append(values.rank(ascending=False, method="average"))
    if not ranks:
        raise prerequisite("没有可融合的重要度（RF/LASSO/VIP 都不可用）")
    joined = pd.concat(ranks, axis=1)
    return joined.mean(axis=1, skipna=True)


def _lasso_fit(X: np.ndarray, y: np.ndarray, folds: int, multiclass: bool, alpha):
    """L1 逻辑回归 + 交叉验证选 C（Tibshirani 1996）；返回标准化器/模型/系数。"""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    folds_used = _cv_folds_for(y, folds)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    splitter = StratifiedKFold(n_splits=folds_used, shuffle=True, random_state=SEED)
    fixed_c = None
    if alpha not in (None, "", "cv", "auto", "CV"):
        try:
            fixed_c = max(float(alpha), 1e-4)
        except (TypeError, ValueError):
            fixed_c = None
    if fixed_c is not None:
        model = LogisticRegression(penalty="l1", solver="saga", C=fixed_c, max_iter=6000,
                                   tol=1e-3, class_weight="balanced", random_state=SEED)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            model.fit(Xs, y)
        best_c = fixed_c
        cv_score = float("nan")
    else:
        model = LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 7), cv=splitter, penalty="l1", solver="saga",
            scoring="roc_auc" if not multiclass else "accuracy", max_iter=6000, tol=1e-3,
            class_weight="balanced", random_state=SEED, n_jobs=1, refit=True,
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            model.fit(Xs, y)
        try:
            best_c = float(np.ravel(model.C_)[0])
        except Exception:  # noqa: BLE001
            best_c = float("nan")
        try:
            scores = model.scores_[list(model.classes_)[0]]
            cv_score = float(np.mean(scores)) if np.size(scores) else float("nan")
        except Exception:  # noqa: BLE001
            cv_score = float("nan")
    coefs = np.asarray(model.coef_, dtype=float)
    if coefs.ndim == 2 and coefs.shape[0] > 1:
        magnitude = np.max(np.abs(coefs), axis=0)
        sign_source = coefs[np.argmax(np.abs(coefs), axis=0), np.arange(coefs.shape[1])]
    else:
        magnitude = np.abs(coefs[0] if coefs.ndim == 2 else coefs)
        sign_source = coefs[0] if coefs.ndim == 2 else coefs
    return {"scaler": scaler, "model": model, "coefs": coefs, "magnitude": magnitude,
            "signed": sign_source, "best_c": best_c, "cv_score": cv_score, "folds": folds_used}


def _lasso_path(X: np.ndarray, y: np.ndarray, n_points: int = 8):
    """正则化路径：C 从大到小，记录入选特征数与系数（用于 lasso_path 图）。"""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    grid = np.logspace(1.0, -2.0, max(4, int(n_points)))
    n_selected: list[int] = []
    coef_rows: list[np.ndarray] = []
    for c in grid:
        try:
            model = LogisticRegression(penalty="l1", solver="saga", C=float(c), max_iter=4000,
                                       tol=1e-3, class_weight="balanced", random_state=SEED)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                model.fit(Xs, y)
            coefs = np.asarray(model.coef_, dtype=float)
            coef = np.max(np.abs(coefs), axis=0) * np.sign(np.max(np.abs(coefs), axis=0))
            if coefs.ndim == 2 and coefs.shape[0] == 1:
                coef = coefs[0]
            coef_rows.append(np.asarray(coef, dtype=float))
            n_selected.append(int(np.sum(np.abs(coef) > 1e-8)))
        except Exception:  # noqa: BLE001 - 某个 C 不收敛就跳过
            coef_rows.append(np.full(Xs.shape[1], np.nan))
            n_selected.append(0)
    return {"Cs": grid, "n_selected": np.asarray(n_selected, dtype=int),
            "coefs": np.vstack(coef_rows) if coef_rows else np.zeros((0, Xs.shape[1]))}


# ---------------------------------------------------------------------------
# 5. 生存分析（KM / logrank / Cox）
# ---------------------------------------------------------------------------


def _clinical_columns(ctx, params) -> tuple[str | None, str | None]:
    """找生存时间/事件列：params 优先，其次按列名 + 数值/0-1 启发式。"""
    params = dict(params or {})
    info = _info(ctx)
    time_col = str(params.get("time_col") or "").strip() or None
    event_col = str(params.get("event_col") or "").strip() or None
    time_patterns = ("time", "survival", "follow", "month", "day", "os", "dfs", "rfs", "pfs", "duration")
    event_patterns = ("event", "status", "dead", "death", "censor", "vital", "recur")
    if time_col is None:
        for col in info.columns:
            key = str(col).lower()
            if any(p in key for p in time_patterns):
                values = pd.to_numeric(info[col], errors="coerce")
                if values.notna().sum() >= max(3, int(len(values) * 0.5)):
                    time_col = str(col)
                    break
    if event_col is None:
        for col in info.columns:
            key = str(col).lower()
            if any(p in key for p in event_patterns):
                # 事件列可以是 0/1 数值，也可以是 dead/alive 这类显式文本；未知取值不算"可解析"
                codes = [_event_code(value) for value in info[col].tolist()]
                known = int(sum(code is not None for code in codes))
                if known >= max(3, int(len(codes) * 0.5)):
                    event_col = str(col)
                    break
    return time_col, event_col


def _survival_columns(ctx, params, warnings: list[str] | None = None
                      ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """取 (时间, 事件, time_col, event_col)。

    事件列必须显式映射（BIO_P0_BACKLOG §4 P0-3）：
      * ``event_map`` 可显式指定 ``{"progression": 1, "stable": 0}``；
      * 内置只认 dead/alive/censored/1/0 等明确取值；
      * 未知取值默认 ``bad_request``（列出取值与样本数），不允许静默当删失；
      * ``event_unknown="na"``：未知事件样本的时间/事件置 NaN（下游自动剔除）并写 warning。
    """
    params = dict(params or {})
    time_col, event_col = _clinical_columns(ctx, params)
    info = _info(ctx)
    if not time_col or time_col not in info.columns:
        raise prerequisite("样本信息里缺少生存时间列（time/os_time/months 等），请先导入临床表或指定 time_col")
    if not event_col or event_col not in info.columns:
        raise prerequisite("样本信息里缺少生存事件列（event/status/dead 等），请先导入临床表或指定 event_col")
    times = pd.to_numeric(info[time_col], errors="coerce").to_numpy(dtype=float)
    times = np.where(np.isfinite(times) & (times > 0), times, np.nan)
    raw_map = params.get("event_map") or {}
    event_map: dict[str, int] = {}
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            text = _event_value(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if text is not None:
                event_map[text] = 1 if number >= 0.5 else 0
    unknown_mode = str(params.get("event_unknown") or "error").strip().lower()
    drop_unknown = unknown_mode in ("na", "nan", "drop", "na_drop", "drop_unknown")
    codes: list[float] = []
    unknown_positions: list[int] = []
    unknown_counts: dict[str, int] = {}
    for position, value in enumerate(info[event_col].tolist()):
        code = _event_code(value, event_map)
        if code is None:
            codes.append(np.nan)
            unknown_positions.append(position)
            label = "空值" if _event_value(value) is None else str(value).strip()
            unknown_counts[label] = unknown_counts.get(label, 0) + 1
        else:
            codes.append(float(code))
    events = np.asarray(codes, dtype=float)
    if unknown_positions:
        shown = "、".join(f"{value}×{count}" for value, count in list(unknown_counts.items())[:6])
        total = len(info)
        message = (
            f"事件列「{event_col}」有 {len(unknown_positions)}/{total} 个无法识别的取值（{shown}）："
            f"生存事件必须显式映射为 0/1 或 dead/alive/censored 等明确取值，不能把未知当删失"
        )
        if not drop_unknown:
            raise bad_request(
                message + "；请传 event_map（如 {'progression':1,'stable':0}），"
                          "或显式设置 event_unknown='na' 把这些样本按 NA 丢弃"
            )
        times = times.copy()
        times[unknown_positions] = np.nan
        if warnings is not None:
            warnings.append(message + "；按 event_unknown='na' 已将这些样本置 NA（下游自动剔除）")
    valid = np.isfinite(times) & np.isfinite(events) & (times > 0)
    times = np.where(valid, times, np.nan)
    events = np.where(valid, events, np.nan)
    return times, events, str(time_col), str(event_col)


def _km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier 乘积极限估计（Kaplan & Meier 1958）；返回阶梯曲线。"""
    order = np.argsort(times, kind="mergesort")
    t = np.asarray(times, dtype=float)[order]
    e = np.asarray(events, dtype=int)[order]
    xs: list[float] = [0.0]
    ys: list[float] = [1.0]
    survival = 1.0
    n = t.size
    at_risk = n
    for i in range(n):
        if e[i] == 1 and at_risk > 0:
            survival *= (1.0 - 1.0 / at_risk)
            xs.append(float(t[i]))
            ys.append(float(survival))
        at_risk -= 1
    return np.asarray(xs), np.asarray(ys)


def _km_median(times: np.ndarray, events: np.ndarray) -> float:
    xs, ys = _km_curve(times, events)
    for x, y in zip(xs, ys):
        if y <= 0.5:
            return float(x)
    return float("nan")


def _logrank(t1, e1, t2, e2) -> tuple[float, float, str]:
    """logrank 检验（Mantel-Haenszel 形式）；返回 (chi2, p, 方法说明)。"""
    t1 = np.asarray(t1, float)
    t2 = np.asarray(t2, float)
    e1 = np.asarray(e1, int)
    e2 = np.asarray(e2, int)
    try:
        from lifelines.statistics import logrank_test  # noqa: PLC0415

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            outcome = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
        return float(outcome.test_statistic), float(outcome.p_value), "lifelines logrank_test"
    except Exception:  # noqa: BLE001 - lifelines 不可用时用内置实现
        times = np.unique(np.concatenate([t1, t2]))
        observed = 0.0
        expected = 0.0
        variance = 0.0
        for t in times:
            n1 = float((t1 >= t).sum())
            n2 = float((t2 >= t).sum())
            n = n1 + n2
            d1 = float(((t1 == t) & (e1 == 1)).sum())
            d2 = float(((t2 == t) & (e2 == 1)).sum())
            d = d1 + d2
            if d <= 0 or n <= 1:
                continue
            observed += d1
            expected += d * n1 / n
            variance += d * (n1 / n) * (n2 / n) * (n - d) / (n - 1.0)
        chi2 = (observed - expected) ** 2 / variance if variance > 0 else 0.0
        try:
            from scipy import stats as _stats  # noqa: PLC0415

            p_value = float(_stats.chi2.sf(chi2, 1))
        except Exception:  # noqa: BLE001
            p_value = float("nan")
        return float(chi2), p_value, "内置实现（logrank 卡方）"


def _cox_univariate_internal(times: np.ndarray, events: np.ndarray, x: np.ndarray) -> dict:
    """单变量 Cox 的 Newton-Raphson 求解（Breslow ties），lifelines 不可用时的回退。"""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x) & (t > 0)
    t, e, x = t[ok], e[ok], x[ok]
    if t.size < 5 or e.sum() < 2:
        raise prerequisite("生存数据太少（有效样本 <5 或事件数 <2），无法拟合 Cox")
    if np.nanstd(x) <= 0:
        raise prerequisite("该特征在所有样本里取值相同，无法做 Cox 回归")
    beta = 0.0
    info = 0.0
    for _ in range(50):
        score = 0.0
        info = 0.0
        for event_time in np.unique(t[e == 1]):
            risk = t >= event_time
            d = float(((t == event_time) & (e == 1)).sum())
            xr = x[risk]
            with np.errstate(over="ignore"):
                weights = np.exp(np.clip(beta * xr, -50, 50))
            s0 = float(weights.sum())
            s1 = float((xr * weights).sum())
            s2 = float((xr ** 2 * weights).sum())
            x_events = float(x[(t == event_time) & (e == 1)].sum())
            score += x_events - d * s1 / max(s0, 1e-12)
            info += d * (s2 / max(s0, 1e-12) - (s1 / max(s0, 1e-12)) ** 2)
        if not np.isfinite(info) or info <= 0:
            break
        step = score / info
        beta += float(step)
        if abs(step) < 1e-8:
            break
    se = math.sqrt(1.0 / info) if info > 0 else float("nan")
    hr = math.exp(beta) if abs(beta) < 50 else float("inf")
    z = beta / se if se and math.isfinite(se) and se > 0 else float("nan")
    try:
        from scipy import stats as _stats  # noqa: PLC0415

        p_value = float(2.0 * _stats.norm.sf(abs(z))) if math.isfinite(z) else float("nan")
    except Exception:  # noqa: BLE001
        p_value = math.erfc(abs(z) / math.sqrt(2.0)) if math.isfinite(z) else float("nan")
    lo = math.exp(beta - 1.959963985 * se) if se and math.isfinite(se) else float("nan")
    hi = math.exp(beta + 1.959963985 * se) if se and math.isfinite(se) else float("nan")
    return {"hr": hr, "lo": lo, "hi": hi, "p": p_value, "beta": beta, "se": se,
            "method": "内置实现（Newton-Raphson，Breslow ties）"}


def _cox_univariate(times: np.ndarray, events: np.ndarray, x: np.ndarray) -> dict:
    """单变量 Cox：优先 lifelines（Therneau 2000），失败/缺失时退回内置实现。"""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x) & (t > 0)
    if int(ok.sum()) < 5 or int(e[ok].sum()) < 2:
        raise prerequisite("生存数据太少（有效样本 <5 或事件数 <2），无法拟合 Cox")
    try:
        from lifelines import CoxPHFitter  # noqa: PLC0415

        frame = pd.DataFrame({"T": t[ok], "E": e[ok], "x": x[ok]})
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            fitter = CoxPHFitter()
            fitter.fit(frame, duration_col="T", event_col="E")
        summary = fitter.summary.loc["x"]
        return {
            "hr": float(summary["exp(coef)"]),
            "lo": float(summary["exp(coef) lower 95%"]),
            "hi": float(summary["exp(coef) upper 95%"]),
            "p": float(summary["p"]),
            "beta": float(summary["coef"]),
            "se": float(summary["se(coef)"]),
            "method": "lifelines CoxPHFitter",
        }
    except prerequisite:
        raise
    except Exception:  # noqa: BLE001 - lifelines 缺失或拟合失败 → 内置实现
        return _cox_univariate_internal(t, e, x)


def _cutoff_threshold(score: np.ndarray, times: np.ndarray, events: np.ndarray,
                      mode: str) -> tuple[float, str]:
    """分组阈值：median（中位数）或 optimal（扫描分位点最大化 logrank 统计量）。"""
    text = str(mode or "median").strip().lower()
    if text in ("median", "median_split", "中位"):
        value = float(np.nanmedian(score))
        return value, "中位数切分（median split）"
    if text in ("optimal", "best", "最优"):
        quantiles = np.linspace(0.25, 0.75, 21)
        candidates = np.unique(np.nanquantile(score, quantiles))
        best_value = float(np.nanmedian(score))
        best_score = -1.0
        for value in candidates:
            group_high = score > value
            if group_high.sum() < 3 or (~group_high).sum() < 3:
                continue
            try:
                chi2, _, _ = _logrank(times[group_high], events[group_high],
                                      times[~group_high], events[~group_high])
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(chi2) and chi2 > best_score:
                best_score = float(chi2)
                best_value = float(value)
        return best_value, "最优切分（扫描 25–75 分位点最大化 logrank 统计量；注意多重比较偏倚）"
    number = _as_number(mode)
    if math.isfinite(number):
        return float(number), "用户指定阈值"
    value = float(np.nanmedian(score))
    return value, "中位数切分（median split）"


# ---------------------------------------------------------------------------
# 6. 表 / 图
# ---------------------------------------------------------------------------

BIOMARKER_COLUMNS = [
    {"key": "rank", "label": "排名", "type": "number"},
    {"key": "feature", "label": "特征", "type": "string"},
    {"key": "name", "label": "代谢物", "type": "string"},
    {"key": "mz", "label": "m/z", "type": "number"},
    {"key": "rt", "label": "RT", "type": "number"},
    {"key": "importance", "label": "重要度", "type": "number"},
    {"key": "score", "label": "标准化得分", "type": "number"},
    {"key": "log2FC", "label": "log2FC", "type": "number"},
    {"key": "mean_case", "label": "均值(实验组)", "type": "number"},
    {"key": "mean_control", "label": "均值(对照组)", "type": "number"},
    {"key": "regulation", "label": "方向", "type": "string"},
]

METRIC_COLUMNS = [
    {"key": "metric", "label": "指标", "type": "string"},
    {"key": "value", "label": "数值", "type": "number"},
    {"key": "detail", "label": "说明", "type": "string"},
]

SURVIVAL_COLUMNS = [
    {"key": "group", "label": "分组", "type": "string"},
    {"key": "n", "label": "样本数", "type": "number"},
    {"key": "n_events", "label": "事件数", "type": "number"},
    {"key": "median_survival", "label": "中位生存时间", "type": "number"},
]

COX_COLUMNS = [
    {"key": "feature", "label": "特征", "type": "string"},
    {"key": "name", "label": "代谢物", "type": "string"},
    {"key": "hr", "label": "HR", "type": "number"},
    {"key": "hr_low", "label": "HR 95% 下限", "type": "number"},
    {"key": "hr_high", "label": "HR 95% 上限", "type": "number"},
    {"key": "p", "label": "P 值", "type": "number"},
    {"key": "method", "label": "方法", "type": "string"},
]

ROC_COLUMNS = [
    {"key": "fpr", "label": "FPR", "type": "number"},
    {"key": "tpr", "label": "TPR", "type": "number"},
    {"key": "threshold", "label": "阈值", "type": "number"},
]

CALIBRATION_COLUMNS = [
    {"key": "bin", "label": "分箱", "type": "number"},
    {"key": "n", "label": "样本数", "type": "number"},
    {"key": "mean_pred", "label": "平均预测概率", "type": "number"},
    {"key": "observed", "label": "观测事件率", "type": "number"},
    {"key": "ci_low", "label": "95% 下限", "type": "number"},
    {"key": "ci_high", "label": "95% 上限", "type": "number"},
]


def _importance_frame(ctx, features: list[str], importance, de: pd.DataFrame | None = None,
                      binary_means: dict | None = None) -> pd.DataFrame:
    """特征 → (rank/feature/name/mz/rt/importance/score/log2FC/均值/方向) 的输出表。"""
    meta = _feature_meta(ctx, features)
    values = pd.Series(np.asarray(importance, dtype=float), index=features)
    frame = meta.copy()
    frame["importance"] = [float(values.get(f, np.nan)) for f in features]
    ordered = frame.assign(_abs=frame["importance"].abs()).sort_values("_abs", ascending=False)
    ordered = ordered.drop(columns=["_abs"]).reset_index(drop=True)
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    # 标准化得分：重要度 min-max 到 [0,1]
    raw = ordered["importance"].to_numpy(float)
    lo, hi = np.nanmin(raw), np.nanmax(raw)
    if math.isfinite(lo) and math.isfinite(hi) and hi - lo > 0:
        ordered["score"] = (raw - lo) / (hi - lo)
    else:
        ordered["score"] = 1.0
    logfc_map: dict[str, float] = {}
    mean_case: dict[str, float] = {}
    mean_control: dict[str, float] = {}
    regulation: dict[str, str] = {}
    if isinstance(de, pd.DataFrame) and not de.empty:
        de_index = de.drop_duplicates("feature").set_index("feature")
        for feature in ordered["feature"]:
            if feature in de_index.index:
                row = de_index.loc[feature]
                logfc_map[feature] = _as_number(row.get("log2FC"))
                mean_case[feature] = _as_number(row.get("mean_case")) if "mean_case" in de_index.columns else float("nan")
                mean_control[feature] = _as_number(row.get("mean_control")) if "mean_control" in de_index.columns else float("nan")
                regulation[feature] = str(row.get("regulation") or "")
    ordered["log2FC"] = [_as_number(logfc_map.get(f)) for f in ordered["feature"]]
    ordered["mean_case"] = [_as_number(mean_case.get(f)) for f in ordered["feature"]]
    ordered["mean_control"] = [_as_number(mean_control.get(f)) for f in ordered["feature"]]
    if not regulation:
        ordered["regulation"] = np.where(ordered["log2FC"] > 0, "up",
                                         np.where(ordered["log2FC"] < 0, "down", ""))
    else:
        ordered["regulation"] = [regulation.get(f, "") for f in ordered["feature"]]
    return ordered[["rank", "feature", "name", "mz", "rt", "importance", "score",
                    "log2FC", "mean_case", "mean_control", "regulation"]]


def _fig_importance(frame: pd.DataFrame, outdir: str, name: str, title: str,
                    top_n: int = 20, xlabel: str = "Importance") -> dict:
    sub = frame.head(max(1, int(top_n))).iloc[::-1]
    labels = []
    for _, row in sub.iterrows():
        label = str(row["name"]) if row["name"] else str(row["feature"])
        labels.append(label if len(label) <= 38 else label[:36] + "…")
    values = sub["importance"].to_numpy(float)
    height = max(3.4, 0.36 * len(labels) + 1.5)
    fig, ax = new_fig(8.8, height)
    positions = np.arange(len(labels))
    colors = [PALETTE[0] if v >= 0 else PALETTE[2] for v in values]
    ax.barh(positions, values, color=colors, height=0.68, edgecolor="white", linewidth=0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.axvline(0.0, color="#8a9099", linewidth=0.8)
    span = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not math.isfinite(span) or span <= 0:
        span = 1.0
    for pos, value in zip(positions, values):
        ax.text(value + (span * 0.02 if value >= 0 else -span * 0.02), pos, f"{value:.3g}",
                va="center", ha="left" if value >= 0 else "right", fontsize=7.2, color="#5a6068")
    low = float(np.nanmin(values)) if values.size else 0.0
    high = float(np.nanmax(values)) if values.size else 1.0
    ax.set_xlim(min(0.0, low - span * 0.12), max(0.0, high + span * 0.18))
    ax.grid(axis="x", color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    return save_fig(fig, outdir, name, title)


def _fig_roc(fpr, tpr, auc: float, lo, hi, outdir: str, name: str = "roc-curve",
             title: str = "ROC curve", label: str = "") -> dict:
    fig, ax = new_fig(5.0, 4.8)
    ax.plot(fpr, tpr, color=PALETTE[0], linewidth=2.0,
            label=f"{label + ' ' if label else ''}AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#b9bfc8", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title, fontsize=10.5, loc="left")
    text = f"AUC = {auc:.3f}"
    if math.isfinite(lo) and math.isfinite(hi):
        text += f"\n95% CI {lo:.3f}-{hi:.3f}"
    ax.text(0.97, 0.05, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
            color="#3b4048")
    ax.legend(loc="lower right", frameon=False, fontsize=8.4)
    ax.grid(color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    return save_fig(fig, outdir, name, title)


def _fig_boxplot(ctx, scores: np.ndarray, labels: np.ndarray, outdir: str, name: str,
                 title: str, ylabel: str = "Score", groups_order: list[str] | None = None) -> dict:
    uniq = groups_order or [str(v) for v in dict.fromkeys(labels.tolist())]
    fig, ax = new_fig(5.2, 4.6)
    data = [scores[labels == group] for group in uniq]
    colors = palette(max(len(uniq), 1))
    box = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                     medianprops={"color": "#2b2f36", "linewidth": 1.2})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(SEED)
    for i, group in enumerate(uniq, start=1):
        values = data[i - 1]
        jitter = rng.uniform(-0.12, 0.12, values.size)
        ax.scatter(np.full(values.size, i) + jitter, values, s=12, color=colors[i - 1],
                   alpha=0.8, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks(np.arange(1, len(uniq) + 1))
    ax.set_xticklabels([str(g) for g in uniq], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(axis="y", color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    return save_fig(fig, outdir, name, title)


def _fig_lasso_path(path: dict, selected: list[str], all_features: list[str], outdir: str,
                    name: str = "lasso-path", title: str = "LASSO regularization path") -> dict:
    fig, axes = new_fig(9.6, 4.6)
    ax = axes
    try:
        fig.subplots_adjust(wspace=0.32)
    except Exception:  # noqa: BLE001
        pass
    log_c = np.log10(np.clip(path["Cs"], 1e-6, None))
    ax.plot(log_c, path["n_selected"], color=PALETTE[0], linewidth=2.0, marker="o", markersize=3.4)
    ax.set_xlabel("log10(C)  (larger C = weaker penalty)")
    ax.set_ylabel("Number of selected metabolites")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.invert_xaxis()
    if path["coefs"].size and path["coefs"].shape[1] == len(all_features):
        strongest = np.argsort(-np.nanmax(np.abs(path["coefs"]), axis=0))[:8]
        for rank, column in enumerate(strongest):
            values = path["coefs"][:, column]
            if not np.isfinite(values).any():
                continue
            ax.plot(log_c, values, color=palette(rank + 1)[-1], linewidth=1.1, alpha=0.9)
    return save_fig(fig, outdir, name, title)


def _fig_km(groups: dict, logrank_p: float, hr: float | None, feature: str, outdir: str,
            name: str = "km-metabolite", title: str = "Kaplan-Meier by metabolite level") -> dict:
    fig, ax = new_fig(5.8, 4.9)
    for index, (label, (times, events)) in enumerate(groups.items()):
        xs, ys = _km_curve(times, events)
        color = PALETTE[index % len(PALETTE)]
        ax.step(xs, ys, where="post", color=color, linewidth=1.9,
                label=f"{label} (n={len(times)})")
        censor = events == 0
        if censor.any():
            t_c = times[censor]
            surv = np.interp(t_c, xs, ys)
            ax.scatter(t_c, surv, marker="|", s=26, color=color, alpha=0.85)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title(title, fontsize=10.5, loc="left")
    text = f"logrank p = {logrank_p:.3g}"
    if hr is not None and math.isfinite(hr):
        text += f"\nCox HR = {hr:.2f}"
    ax.text(0.03, 0.06, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4,
            color="#3b4048")
    ax.legend(loc="upper right", frameon=False, fontsize=8.4)
    ax.grid(color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    return save_fig(fig, outdir, name, title)


def _fig_calibration(frame: pd.DataFrame, auc: float, outdir: str, name: str = "calibration-curve",
                     title: str = "Calibration curve") -> dict:
    fig, ax = new_fig(5.0, 4.6)
    if frame.empty:
        ax.text(0.5, 0.5, "no calibration bins", transform=ax.transAxes, ha="center", va="center")
    else:
        pred = frame["mean_pred"].to_numpy(float)
        observed = frame["observed"].to_numpy(float)
        low = frame["ci_low"].to_numpy(float)
        high = frame["ci_high"].to_numpy(float)
        ax.errorbar(pred, observed, yerr=[np.clip(observed - low, 0, None), np.clip(high - observed, 0, None)],
                    fmt="o-", color=PALETTE[0], linewidth=1.8, markersize=5.5,
                    ecolor="#9aa0a8", elinewidth=1.0, capsize=2.5, label="Observed vs predicted")
        ax.plot([0, 1], [0, 1], color="#b9bfc8", linestyle="--", linewidth=1.0, label="Perfect calibration")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.text(0.03, 0.93, f"CV AUC = {auc:.3f}" if math.isfinite(auc) else "",
                transform=ax.transAxes, fontsize=8.4, color="#3b4048", va="top")
        ax.legend(loc="lower right", frameon=False, fontsize=8.4)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    return save_fig(fig, outdir, name, title)


def _calibration_frame(proba: np.ndarray, y: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """校准表：按预测概率分位分箱，比较平均预测概率与观测事件率（Wilson 95% CI）。"""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(proba)
    n_bins = int(max(2, min(n_bins, proba.size)))
    bins = np.array_split(order, n_bins)
    rows = []
    for index, idx in enumerate(bins, start=1):
        if idx.size == 0:
            continue
        p_hat = float(proba[idx].mean())
        observed = float(y[idx].mean())
        n = int(idx.size)
        z = 1.959963985
        denom = 1.0 + z * z / max(n, 1)
        centre = (observed + z * z / (2 * n)) / denom
        half = z * math.sqrt(max(observed * (1 - observed) / n + z * z / (4 * n * n), 0.0)) / denom
        rows.append({"bin": index, "n": n, "mean_pred": p_hat, "observed": observed,
                     "ci_low": max(0.0, centre - half), "ci_high": min(1.0, centre + half)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. 自助法乐观度校正（Harrell 1996）
# ---------------------------------------------------------------------------


def _logit_model(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    # 不显式传 penalty：sklearn 1.8 起 penalty 参数弃用，默认就是 L2（ridge）
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs",
                                     class_weight="balanced", random_state=SEED)),
    ])


# ---------------------------------------------------------------------------
# 7b. 嵌套交叉验证：训练折内完成 填补 → 标度 → 筛选 → 模型 → 阈值（P0-2 防泄漏）
#
# 旧实现的泄漏路径：全样本做特征筛选/填补/标准化 → 再交叉验证。纯噪声下全样本
# |Cohen's d| 预筛能把 AUC 抬到 ~0.99。下面的实现保证任何"看标签"的决策（筛选、阈值、
# 超参）都只在训练折内做，测试折只做 predict；有受试者 ID 时按个体整块划分。
# ---------------------------------------------------------------------------

SUBJECT_COL_HINTS = ("subject", "subject_id", "patient", "patient_id", "donor", "donor_id",
                     "individual", "participant", "case_id", "animal", "mouse")
DEFAULT_SCREEN_LIMIT = 500   # 高维时折内单变量预筛上限（只为控制模型成本，不改变 fold 边界）


def _find_column(info: pd.DataFrame, hints) -> str | None:
    """按关键词找一列（先精确匹配，再包含匹配）。"""
    if not isinstance(info, pd.DataFrame):
        return None
    columns = [str(column) for column in info.columns]
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for hint in hints:
        for column in columns:
            if hint in column.lower():
                return column
    return None


def _subject_vector(ctx, params: dict) -> tuple[np.ndarray | None, str | None]:
    """取受试者/患者 ID 向量，供按个体分组划分与自助法整块重抽；无则返回 (None, None)。"""
    params = dict(params or {})
    info = _info(ctx)
    samples = _samples(ctx)
    explicit = str(params.get("subject_col") or "").strip()
    column: str | None = None
    if explicit:
        if explicit not in info.columns:
            raise bad_request(
                f"subject_col「{explicit}」不在样本信息里；可选列：{'、'.join(map(str, info.columns))}"
            )
        column = explicit
    if column is None:
        column = _find_column(info, SUBJECT_COL_HINTS)
    if not column or len(info) != len(samples):
        return None, None
    values = [str(value).strip() for value in info[column].tolist()]
    if any(value == "" or value.lower() in ("nan", "none", "na") for value in values):
        return None, None
    return np.asarray(values, dtype=object), column


def _impute_medians(train: np.ndarray) -> np.ndarray:
    """训练折逐特征中位数（只在训练折上估计，测试折复用）。"""
    arr = np.asarray(train, dtype=float)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    return np.where(np.isfinite(median), median, 0.0)


def _impute_apply(arr: np.ndarray, median: np.ndarray) -> np.ndarray:
    out = np.array(np.asarray(arr, dtype=float), copy=True)
    rows, cols = np.where(~np.isfinite(out))
    if rows.size:
        out[rows, cols] = np.asarray(median, dtype=float)[cols]
    return out


def _cohen_d(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐特征 Cohen's d（正类 − 负类），只应传入训练折。"""
    y = np.asarray(y, dtype=int)
    pos = X[y == 1]
    neg = X[y == 0]
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        raise prerequisite("训练折内某一类样本少于 2 个，无法做单变量筛选")
    mean_pos = pos.mean(axis=0)
    mean_neg = neg.mean(axis=0)
    var_pos = pos.var(axis=0, ddof=1)
    var_neg = neg.var(axis=0, ddof=1)
    pooled = np.sqrt(np.maximum((var_pos + var_neg) / 2.0, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(pooled > 0, (mean_pos - mean_neg) / pooled, 0.0)
    return np.where(np.isfinite(score), score, 0.0)


def _select_in_fold(X_train: np.ndarray, y_train: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """**只在训练折内**用 |Cohen's d| 选特征；测试折不得参与（P0-2 核心约束）。"""
    scores = _cohen_d(X_train, y_train)
    k = int(max(1, min(int(n_features), X_train.shape[1])))
    order = np.argsort(-np.abs(scores), kind="stable")[:k]
    return np.sort(order), scores


def _classifier(model: str, n_trees: int = 100, seed: int = SEED):
    kind = str(model or "logit").strip().lower()
    if kind in ("rf", "randomforest", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

        return RandomForestClassifier(
            n_estimators=max(20, int(n_trees)), random_state=seed, n_jobs=1,
            min_samples_leaf=2, class_weight="balanced_subsample", max_features="sqrt")
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    return LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs",
                              class_weight="balanced", random_state=seed)


def _fit_pipeline(X_train: np.ndarray, y_train: np.ndarray, n_features: int = DEFAULT_N_FEATURES,
                  model: str = "logit", n_trees: int = 100) -> dict:
    """训练折内的完整流程：中位数填补 → 标准化 → 折内筛选 → 模型。测试折只做 transform。"""
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    median = _impute_medians(X_train)
    filled = _impute_apply(X_train, median)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)
    picked, _scores = _select_in_fold(scaled, y_train, n_features)
    estimator = _classifier(model, n_trees=n_trees)
    estimator.fit(scaled[:, picked], np.asarray(y_train, dtype=int))
    return {"median": median, "scaler": scaler, "picked": picked, "model": estimator,
            "model_kind": str(model)}


def _predict_pipeline(pipe: dict, X: np.ndarray) -> np.ndarray:
    filled = _impute_apply(X, pipe["median"])
    scaled = pipe["scaler"].transform(filled)
    proba = pipe["model"].predict_proba(scaled[:, pipe["picked"]])
    return np.asarray(proba[:, 1], dtype=float)


def _apparent_auc(X: np.ndarray, y: np.ndarray, n_features: int = DEFAULT_N_FEATURES,
                  model: str = "logit", n_trees: int = 100) -> float:
    """表观 AUC：全样本拟合后在**同一批样本**上评估（乐观，明确标为 apparent）。"""
    pipe = _fit_pipeline(X, y, n_features=n_features, model=model, n_trees=n_trees)
    return float(_auc_binary(y, _predict_pipeline(pipe, X)))


def _youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Youden J 最大化的阈值（Sensitivity + Specificity − 1），只用训练折内层 OOF。"""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    ok = np.isfinite(proba)
    if int(ok.sum()) < 4 or np.unique(y_true[ok]).size < 2:
        return 0.5
    from sklearn.metrics import roc_curve  # noqa: PLC0415

    fpr, tpr, thresholds = roc_curve(y_true[ok], proba[ok])
    index = int(np.argmax(tpr - fpr))
    value = float(thresholds[index])
    if not (0.0 < value < 1.0) or not math.isfinite(value):
        value = float(np.median(proba[ok]))
    return value


def _cv_splitter(y: np.ndarray, groups, folds: int, seed: int):
    """按是否有受试者 ID 选择分层样本划分或分层个体划分。"""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: PLC0415

    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    counts = counts[counts > 0]
    if counts.size < 2:
        raise prerequisite("目标变量只有 1 个类别，无法做交叉验证/ROC，请检查分组列")
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        folds_used = int(max(2, min(int(folds), int(np.unique(groups).size))))
        try:
            return StratifiedGroupKFold(n_splits=folds_used, shuffle=True, random_state=seed), folds_used, True
        except Exception:  # noqa: BLE001 - 极老 sklearn 回退到 GroupKFold
            from sklearn.model_selection import GroupKFold  # noqa: PLC0415

            return GroupKFold(n_splits=folds_used), folds_used, True
    folds_used = int(max(2, min(int(folds), int(counts.min()))))
    return StratifiedKFold(n_splits=folds_used, shuffle=True, random_state=seed), folds_used, False


def _split_folds(splitter, X, y, groups, folds_used: int):
    """生成划分；分层个体划分在极端分布下会失败，回退 GroupKFold。"""
    try:
        return list(splitter.split(X, y, groups))
    except Exception:  # noqa: BLE001
        from sklearn.model_selection import GroupKFold  # noqa: PLC0415

        if groups is None:
            raise
        return list(GroupKFold(n_splits=folds_used).split(X, y, groups))


def _inner_oof_scores(X: np.ndarray, y: np.ndarray, folds: int, n_features: int,
                      model: str, n_trees: int, groups=None, seed: int = SEED + 1) -> np.ndarray:
    """内层交叉验证的外推概率（每个内层训练折内部再走一遍完整 pipeline）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups_arr = None if groups is None else np.asarray(groups, dtype=object)
    splitter, folds_used, _grouped = _cv_splitter(y, groups_arr, folds, seed)
    oof = np.full(y.size, np.nan)
    for inner_train, inner_test in _split_folds(splitter, X, y, groups_arr, folds_used):
        y_train = y[inner_train]
        if np.unique(y_train).size < 2:
            continue
        try:
            pipe = _fit_pipeline(X[inner_train], y_train, n_features=n_features,
                                 model=model, n_trees=n_trees)
            oof[inner_test] = _predict_pipeline(pipe, X[inner_test])
        except (EngineError, ValueError):
            continue
    return oof


def nested_cv_auc(X, y, groups=None, folds: int = DEFAULT_CV_FOLDS, seed: int = SEED,
                  n_features: int = DEFAULT_N_FEATURES, model: str = "logit",
                  n_trees: int = 100, inner_folds: int = 3) -> dict:
    """嵌套交叉验证 AUC（外层评估、内层取阈值）：训练折内完成填补/标度/筛选/模型/阈值。

    每个外层折：训练折上估中位数填补参数 → 标准化 → 用训练折 |Cohen's d| 选 top n_features
    → 训练折内层 CV 的 OOF 概率取 Youden 阈值 → 模型拟合 → 只在测试折上预测；测试折的标签
    从不参与任何选择。传入受试者 ``groups`` 时用分层个体划分（同个体重复样本不跨折）。

    返回 ``{"auc","folds","fold_aucs","fold_thresholds","n_features","grouped","oof_proba",...}``。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    if X.ndim != 2:
        raise bad_request(f"X 必须是二维矩阵（样本×特征），收到 {X.shape}")
    if X.shape[0] != y.size:
        raise bad_request(f"X 的样本数 {X.shape[0]} 与标签数 {y.size} 不一致")
    if y.size < 8:
        raise prerequisite(f"样本数只有 {y.size} 个，至少需要 8 个才能做嵌套交叉验证")
    groups_arr = None if groups is None else np.asarray(groups, dtype=object)
    if groups_arr is not None and groups_arr.size != y.size:
        raise bad_request(f"groups 长度 {groups_arr.size} 与样本数 {y.size} 不一致")
    splitter, folds_used, grouped = _cv_splitter(y, groups_arr, folds, seed)
    splits = _split_folds(splitter, X, y, groups_arr, folds_used)
    oof = np.full(y.size, np.nan)
    fold_aucs: list[float] = []
    fold_thresholds: list[float] = []
    fold_features: list[list[int]] = []
    for train_index, test_index in splits:
        y_train = y[train_index]
        if np.unique(y_train).size < 2:
            continue
        inner_groups = groups_arr[train_index] if groups_arr is not None else None
        inner_oof = _inner_oof_scores(X[train_index], y_train, inner_folds, n_features,
                                      model, n_trees, groups=inner_groups, seed=seed + 1)
        threshold = _youden_threshold(y_train, inner_oof)
        pipe = _fit_pipeline(X[train_index], y_train, n_features=n_features,
                             model=model, n_trees=n_trees)
        proba = _predict_pipeline(pipe, X[test_index])
        oof[test_index] = proba
        fold_aucs.append(float(_auc_binary(y[test_index], proba)))
        fold_thresholds.append(float(threshold))
        fold_features.append([int(i) for i in pipe["picked"]])
    finite = np.isfinite(oof)
    auc = float(_auc_binary(y[finite], oof[finite])) if finite.any() else float("nan")
    return {
        "auc": auc, "folds": int(folds_used),
        "fold_aucs": fold_aucs, "fold_thresholds": fold_thresholds,
        "fold_features": fold_features,
        "n_features": int(max(1, min(int(n_features), X.shape[1]))),
        "model": str(model), "grouped": bool(grouped),
        "n_groups": int(np.unique(groups_arr).size) if grouped else None,
        "n_samples": int(y.size), "inner_folds": int(inner_folds),
        "oof_proba": oof,
    }


def _bootstrap_optimism(X: np.ndarray, y: np.ndarray, n_boot: int, notes: list[str],
                        warnings: list[str], groups=None, n_features: int = DEFAULT_N_FEATURES,
                        model: str = "logit", n_trees: int = 100) -> dict:
    """Harrell 自助法乐观度（Harrell 1996）：每次重抽样**重做全流程**（填补/标度/筛选/模型/阈值）。

    有受试者 ID 时按个体整块重抽（同个体重复样本一起进出）；返回 bootstrap_grouped 与
    n_subjects，区分 apparent（全样本表观）与 corrected（乐观度校正后）。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    n_boot = int(max(10, min(MAX_BOOT, int(n_boot))))
    if np.unique(y).size < 2:
        raise prerequisite("目标变量只有 1 个类别，无法做自助法验证")
    groups_arr = None if groups is None else np.asarray(groups, dtype=object)
    if groups_arr is not None and groups_arr.size != y.size:
        groups_arr = None
    subject_ids = np.unique(groups_arr) if groups_arr is not None else None
    apparent_pipe = _fit_pipeline(X, y, n_features=n_features, model=model, n_trees=n_trees)
    apparent = float(_auc_binary(y, _predict_pipeline(apparent_pipe, X)))
    rng = np.random.default_rng(SEED)
    optimism_values: list[float] = []
    threshold_values: list[float] = []
    for _ in range(n_boot):
        if subject_ids is not None:
            drawn = rng.choice(subject_ids, size=subject_ids.size, replace=True)
            idx = np.concatenate([np.flatnonzero(groups_arr == subject) for subject in drawn])
        else:
            idx = rng.integers(0, y.size, y.size)
        if idx.size == 0 or np.unique(y[idx]).size < 2:
            continue
        y_boot = y[idx]
        # 重做全流程：先用自助样本内层 CV 取 Youden 阈值（仍然只用训练数据）
        inner_groups = groups_arr[idx] if groups_arr is not None else None
        try:
            inner_oof = _inner_oof_scores(X[idx], y_boot, 3, n_features, model, n_trees,
                                          groups=inner_groups, seed=SEED + 1)
            threshold_values.append(float(_youden_threshold(y_boot, inner_oof)))
            pipe = _fit_pipeline(X[idx], y_boot, n_features=n_features, model=model, n_trees=n_trees)
            auc_boot = _auc_binary(y_boot, _predict_pipeline(pipe, X[idx]))
            auc_orig = _auc_binary(y, _predict_pipeline(pipe, X))
        except (EngineError, ValueError):
            continue
        if np.isfinite(auc_boot) and np.isfinite(auc_orig):
            optimism_values.append(float(auc_boot - auc_orig))
    valid = len(optimism_values)
    mean_optimism = float(np.mean(optimism_values)) if valid else float("nan")
    corrected = float(apparent - mean_optimism) if valid else float("nan")
    notes.append(
        f"自助法乐观度校正（Harrell 1996）：{valid}/{n_boot} 次重抽样有效"
        + (f"（按 {len(subject_ids)} 个受试者整块重抽）" if subject_ids is not None else "（按样本重抽）")
        + f"，表观 AUC={apparent:.3f}、乐观度={mean_optimism:.3f}、校正后 AUC={corrected:.3f}"
    )
    if valid < n_boot * 0.5:
        warnings.append(f"只有 {valid}/{n_boot} 次自助抽样成功，校正结果不稳定")
    return {"apparent": apparent, "optimism": mean_optimism, "corrected": corrected,
            "n_boot": n_boot, "n_valid": valid, "grouped": bool(subject_ids is not None),
            "n_subjects": int(len(subject_ids)) if subject_ids is not None else None,
            "threshold": float(np.mean(threshold_values)) if threshold_values else None,
            "n_features": int(max(1, min(int(n_features), X.shape[1])))}


# ---------------------------------------------------------------------------
# 8. 步骤：biomarker
# ---------------------------------------------------------------------------


def _model_importance(ctx, params: dict, feature_names: list[str], X: np.ndarray, y: np.ndarray,
                      setup: dict, notes: list[str], warnings: list[str]) -> tuple[pd.Series, dict]:
    """按 model 计算重要度序列（rf/lasso/vip/combo），返回 (importance Series, 指标 dict)。"""
    model = str(params.get("model") or "rf").strip().lower()
    n_trees = int(params.get("n_trees") or 300)
    folds = int(params.get("cv_folds") or DEFAULT_CV_FOLDS)
    multiclass = setup["multiclass"]
    target = setup["codes"] if multiclass else y
    metrics: dict = {}
    if model in ("rf", "randomforest", "random_forest"):
        forest = _rf_model(X, target, n_trees=n_trees)
        mean, std = _rf_importance(forest, X, target, multiclass)
        notes.append(f"RandomForest 置换重要度（Breiman 2001；{n_trees} 棵树，{N_PERM_REPEATS} 次置换，"
                     f"评分={'accuracy' if multiclass else 'roc_auc'}）")
        try:
            proba, folds_used = _oof_proba(X, target, folds, lambda: _rf_model(X, target, n_trees=n_trees), multiclass)
            metrics["cv_folds"] = folds_used
            if multiclass:
                metrics["cv_accuracy"] = float(np.mean(np.argmax(proba, axis=1) == target))
            else:
                metrics["cv_auc"] = _auc_binary(target, proba[:, 1])
        except Exception as error:  # noqa: BLE001 - 交叉验证失败不阻塞主结果
            warnings.append(f"RF 交叉验证失败：{error}")
        series = pd.Series(mean, index=feature_names)
        nz = series[series.abs() > 1e-12]
        series = nz if not nz.empty else series
        return series, metrics
    if model in ("lasso", "l1", "logistic_l1"):
        fit = _lasso_fit(X, y, folds, multiclass, params.get("alpha"))
        notes.append(
            f"L1 逻辑回归（LASSO，Tibshirani 1996；saga + {fit['folds']} 折交叉验证选 C，"
            f"best C={fit['best_c']:.3g}）"
        )
        metrics["best_c"] = fit["best_c"]
        metrics["n_selected"] = int(np.sum(np.abs(fit["magnitude"]) > 1e-8))
        if np.isfinite(fit["cv_score"]):
            metrics["cv_auc"] = float(fit["cv_score"])
        series = pd.Series(fit["magnitude"], index=feature_names)
        return series, metrics
    if model in ("vip", "plsda"):
        vip = _vip_series(ctx, notes)
        series = vip.reindex(feature_names).fillna(0.0)
        notes.append(f"复用上游 PLS-DA VIP（{len(vip)} 个特征；Wold 1995）")
        return series, metrics
    if model in ("combo", "rank", "rank_average"):
        pool = feature_names
        series_list = []
        try:
            forest = _rf_model(X, target, n_trees=n_trees)
            mean, _ = _rf_importance(forest, X, target, multiclass)
            series_list.append(pd.Series(mean, index=pool, name="rf"))
        except Exception as error:  # noqa: BLE001
            warnings.append(f"combo 里的 RF 分支失败：{error}")
        try:
            fit = _lasso_fit(X, y, folds, multiclass, params.get("alpha"))
            series_list.append(pd.Series(fit["magnitude"], index=pool, name="lasso"))
        except Exception as error:  # noqa: BLE001
            warnings.append(f"combo 里的 LASSO 分支失败：{error}")
        try:
            vip = _vip_series(ctx, notes).reindex(pool).fillna(0.0)
            series_list.append(pd.Series(vip.to_numpy(float), index=pool, name="vip"))
        except EngineError:
            warnings.append("缺少 PLS-DA VIP，combo 只用 RF + LASSO 两种重要度做秩平均")
        if not series_list:
            raise prerequisite("combo 需要至少一种可用模型，但 RF / LASSO / VIP 都失败了")
        rank_mean = _rank_average(series_list)
        notes.append(f"combo = {len(series_list)} 种重要度的秩平均（1 = 最重要）")
        series = (-rank_mean).sort_values(ascending=False)
        return series, metrics
    raise bad_request(f"未知模型「{model}」（可选：rf / lasso / vip / combo）")


def _cox_table(ctx, features: list[str], times: np.ndarray, events: np.ndarray,
               X: np.ndarray, notes: list[str], warnings: list[str]) -> pd.DataFrame:
    """对 top 特征做单变量 Cox（HR/p），返回表。times/events/X 需已按样本掩码对齐。"""
    meta = _feature_meta(ctx, features)
    rows = []
    for j, feature in enumerate(features):
        try:
            fit = _cox_univariate(times, events, X[:, j])
        except EngineError as error:
            warnings.append(f"特征 {feature} 的 Cox 拟合失败：{error.message}")
            continue
        rows.append({
            "feature": feature,
            "name": meta.iloc[j]["name"] if j < len(meta) else "",
            "hr": fit.get("hr"), "hr_low": fit.get("lo"), "hr_high": fit.get("hi"),
            "p": fit.get("p"), "method": fit.get("method", ""),
        })
    if rows:
        notes.append(f"额外做了 {len(rows)} 个 top 特征的单变量 Cox（HR/p）")
    return pd.DataFrame(rows)


def step_biomarker(ctx, params, outdir):
    """STEP biomarker：RF / LASSO / VIP / combo 选特征，可选单变量 Cox。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    setup = _group_setup(ctx, params)
    de_frame = _de_frame(ctx)  # 步骤契约 requires=["de"]：缺差异结果必须给可读前置错误
    n_features = int(params.get("n_features") or DEFAULT_N_FEATURES)
    n_features = int(max(1, min(n_features, 500)))
    folds = int(params.get("cv_folds") or DEFAULT_CV_FOLDS)
    pool_indices, pool_mode = _candidate_pool(ctx, params, notes, warnings)
    if pool_indices.size > MAX_POOL:
        warnings.append(f"候选特征 {pool_indices.size} 个超过上限，按 |Cohen's d| 预筛到 {MAX_POOL} 个")
        pool_indices = _cap_pool(ctx, pool_indices, setup["y"], MAX_POOL)
    X, feature_names = _matrix_subset(ctx, pool_indices, setup["keep"], warnings)
    importance, metrics = _model_importance(ctx, params, feature_names, X, setup["y"], setup, notes, warnings)
    top = importance.sort_values(ascending=False).head(n_features)
    frame = _importance_frame(ctx, list(top.index), top.to_numpy(float), de_frame)
    dropped = [f for f in feature_names if f not in set(top.index)]
    if dropped:
        # 其余特征也进 data_out 的完整表（表只展示 top n_features）
        extra = _importance_frame(ctx, dropped, importance.reindex(dropped).fillna(0.0).to_numpy(float), de_frame)
        full = pd.concat([frame, extra], ignore_index=True)
        full["rank"] = np.arange(1, len(full) + 1)
    else:
        full = frame

    figures: list[dict] = []
    data_out: list[dict] = []
    model = str(params.get("model") or "rf").strip().lower()
    xlabel = {"vip": "VIP", "combo": "Rank-average importance (sign-flipped)"}.get(model, "Importance")
    figures.append(_fig_importance(frame, outdir, "biomarker-importance",
                                   f"Biomarker importance ({model})", top_n=max(n_features, 10), xlabel=xlabel))
    try:
        path = export_csv(full, outdir, "biomarker-features.csv", index=False)
        data_out.append(data_file("biomarker_csv", path, "标志物完整重要度表"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出标志物表失败：{error}")

    tables = [table("biomarker", f"标志物特征表（top {len(frame)}）", BIOMARKER_COLUMNS,
                    frame.to_dict(orient="records"),
                    note="重要度来自所选模型（内置实现：RF 置换重要度 / LASSO 系数 / VIP / 秩平均）")]

    # 可选：单变量 Cox
    time_col, event_col = _clinical_columns(ctx, params)
    cox_frame = pd.DataFrame()
    if time_col and event_col:
        try:
            times, events, time_col, event_col = _survival_columns(
                ctx, {"time_col": time_col, "event_col": event_col,
                      "event_map": params.get("event_map"),
                      "event_unknown": params.get("event_unknown")}, warnings)
            top_features = list(frame["feature"].head(min(len(frame), max(5, n_features))))
            keep_arr = np.asarray(setup["keep"], dtype=bool)
            cox_full = keep_arr & np.isfinite(times) & np.isfinite(events) & (times > 0)
            if int(cox_full.sum()) < 4:
                warnings.append(f"可用生存样本只有 {int(cox_full.sum())} 个（<4），跳过单变量 Cox")
            else:
                cox_frame = _cox_table(ctx, top_features, times[cox_full],
                                       events[cox_full].astype(int), X[cox_full[keep_arr]],
                                       notes, warnings)
        except EngineError as error:
            warnings.append(f"未做 Cox 回归：{error.message}")
    elif params.get("time_col") or params.get("event_col"):
        warnings.append("指定了 time_col/event_col 但样本信息里缺列，跳过 Cox 回归")
    else:
        notes.append("样本信息里没有生存时间/事件列，跳过单变量 Cox")
    if not cox_frame.empty:
        tables.append(table("biomarker_cox", "top 特征单变量 Cox", COX_COLUMNS,
                            cox_frame.to_dict(orient="records"), note="lifelines CoxPHFitter（不可用时内置实现）"))

    # ------------------------------------------------------------------
    # P0-2：指标拆分 —— apparent（表观，乐观）/ nested_cv（训练折内全流程）/ external（未提供）
    # 旧实现先在全样本上做 |Cohen's d| 预筛再 CV：纯噪声 60×6000 能报 AUC≈0.9967。
    # ------------------------------------------------------------------
    p2_notes: list[str] = []
    p2_warnings: list[str] = []
    subjects, subject_col = _subject_vector(ctx, params)
    subjects_kept = None if subjects is None else subjects[np.asarray(setup["keep"], dtype=bool)]
    if not setup["multiclass"]:
        try:
            X_pool_raw, _pool_names = _matrix_subset(ctx, pool_indices, setup["keep"],
                                                     p2_warnings, impute=False)
            nested_model = "rf" if model in ("rf", "randomforest", "random_forest") else "logit"
            nested_trees = int(max(50, min(int(params.get("n_trees") or 300), 150)))
            apparent_auc = _apparent_auc(X_pool_raw, setup["y"], n_features=n_features,
                                         model=nested_model, n_trees=nested_trees)
            nested = nested_cv_auc(X_pool_raw, setup["y"], groups=subjects_kept, folds=folds,
                                   seed=SEED, n_features=n_features, model=nested_model,
                                   n_trees=nested_trees)
            old_cv = metrics.get("cv_auc")
            metrics["pool_cv_auc"] = (float(old_cv) if isinstance(old_cv, (int, float))
                                      and math.isfinite(float(old_cv)) else None)
            metrics["cv_auc"] = float(nested["auc"])
            metrics["cv_folds"] = int(nested["folds"])
            metrics["apparent_auc"] = float(apparent_auc)
            metrics["nested_cv_auc"] = float(nested["auc"])
            metrics["nested_cv_n_features"] = int(nested["n_features"])
            metrics["nested_cv_grouped"] = bool(nested["grouped"])
            if nested["grouped"]:
                metrics["n_subjects"] = int(nested["n_groups"])
            p2_notes.append(
                f"表观 AUC={apparent_auc:.3f}（全样本拟合，乐观偏差未校正）；"
                f"嵌套交叉验证 AUC={nested['auc']:.3f}（{nested['folds']} 折；训练折内完成"
                f"填补/标度/Cohen's d 筛选/模型/Youden 阈值）"
            )
            p2_notes.append(
                "未提供独立外部验证集：external_auc 不报告；apparent_auc 不是验证，"
                "nested_cv_auc 才是泛化估计"
            )
            if nested["grouped"]:
                p2_notes.append(f"划分与重抽按受试者「{subject_col}」整块进行（{nested['n_groups']} 个个体）")
            else:
                p2_warnings.append(
                    "样本表里没有受试者/患者 ID 列（subject/patient/donor 等）：嵌套 CV 按样本划分；"
                    "若同一样本有重复测量，请先补 ID 列，否则泛化误差会被低估"
                )
        except EngineError as error:
            p2_warnings.append(f"嵌套交叉验证未完成：{error.message}")
    else:
        p2_notes.append("多分类任务不做二分类 AUC 的 apparent/nested 拆分；重要度表仍来自全样本模型")
    notes.extend(p2_notes)
    warnings.extend(p2_warnings)

    bullets = [
        f"模型：{model}；候选池 {len(feature_names)} 个特征（{pool_mode}），输出前 {len(frame)} 个标志物",
        f"分组：{setup['pos']} vs {setup['neg']}（{setup['basis']}），有效样本 {int(setup['keep'].sum())} 个"
        + (f"；已剔除 {setup['n_qc']} 个 QC 样本" if setup["n_qc"] else ""),
        f"top 特征：{'、'.join(str(v) for v in frame['name'].head(5).tolist() if str(v)) or '、'.join(frame['feature'].head(5))}",
    ]
    if "nested_cv_auc" in metrics:
        bullets.append(
            f"表观 AUC {float(metrics['apparent_auc']):.3f} vs 嵌套 CV AUC "
            f"{float(metrics['nested_cv_auc']):.3f}（差值即乐观偏差；外部验证集未提供）"
        )
    elif "cv_auc" in metrics and math.isfinite(float(metrics["cv_auc"])):
        bullets.append(f"交叉验证 AUC = {float(metrics['cv_auc']):.3f}")
    if "n_selected" in metrics:
        bullets.append(f"LASSO 入选特征 {metrics['n_selected']} 个（best C={metrics.get('best_c', float('nan')):.3g}）")
    bullets.append("方法：内置实现（RF 置换重要度 Breiman 2001 / LASSO Tibshirani 1996 / VIP Wold 1995）")

    summary_metrics = {
        "model": model, "n_pool": int(len(feature_names)), "n_features": int(len(frame)),
        "n_samples": int(setup["keep"].sum()), "pos": setup["pos"], "neg": setup["neg"],
        **{k: v for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(float(v))},
    }
    store = {"biomarker": {"model": model, "table": full, "top": frame, "metrics": summary_metrics,
                           "features": list(frame["feature"]), "params": dict(params)}}
    return result(title="标志物筛选", bullets=bullets, metrics=summary_metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


# ---------------------------------------------------------------------------
# 9. 分析（group=标志物与建模）
# ---------------------------------------------------------------------------


def analysis_rf_importance(ctx, params, outdir):
    """随机森林 + 置换重要度（Breiman 2001）：top_n 特征 + 交叉验证 AUC。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    setup = _group_setup(ctx, params)
    n_trees = int(params.get("n_trees") or 300)
    top_n = int(params.get("top_n") or 20)
    pool_indices, _ = _candidate_pool(ctx, params, notes, warnings)
    if pool_indices.size > MAX_POOL:
        pool_indices = _cap_pool(ctx, pool_indices, setup["y"], MAX_POOL)
        warnings.append(f"候选特征过多，已按 |Cohen's d| 预筛到 {MAX_POOL} 个")
    X, feature_names = _matrix_subset(ctx, pool_indices, setup["keep"], warnings)
    target = setup["codes"] if setup["multiclass"] else setup["y"]
    forest = _rf_model(X, target, n_trees=n_trees)
    mean, std = _rf_importance(forest, X, target, setup["multiclass"])
    oof_auc = float("nan")
    folds_used = _cv_folds_for(target, int(params.get("folds") or DEFAULT_CV_FOLDS))
    try:
        proba, folds_used = _oof_proba(X, target, folds_used,
                                       lambda: _rf_model(X, target, n_trees=n_trees), setup["multiclass"])
        oof_auc = (float(np.mean(np.argmax(proba, axis=1) == target)) if setup["multiclass"]
                   else _auc_binary(target, proba[:, 1]))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"交叉验证失败：{error}")
    de_frame = _de_frame(ctx) if "de" in (ctx.get("results") or {}) else None
    frame = _importance_frame(ctx, feature_names, mean, de_frame)
    top = frame.head(top_n)
    figures = [_fig_importance(top, outdir, "rf-importance", "Random forest permutation importance",
                              top_n=top_n)]
    data_out: list[dict] = []
    try:
        path = export_csv(frame, outdir, "rf-importance.csv", index=False)
        data_out.append(data_file("rf_csv", path, "RF 重要度（全部特征）"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出 RF 重要度表失败：{error}")
    metrics = {
        "n_trees": n_trees, "n_pool": len(feature_names), "cv_folds": int(folds_used),
        "cv_auc": oof_auc if not setup["multiclass"] else float("nan"),
        "cv_accuracy": oof_auc if setup["multiclass"] else float("nan"),
        "n_samples": int(setup["keep"].sum()),
    }
    bullets = [
        f"RandomForest（{n_trees} 棵树）+ {N_PERM_REPEATS} 次置换重要度（Breiman 2001；内置实现）",
        f"候选池 {len(feature_names)} 个特征、{int(setup['keep'].sum())} 个样本（{setup['pos']} vs {setup['neg']}）",
    ]
    if math.isfinite(oof_auc):
        bullets.append(f"{folds_used} 折交叉验证 {'准确率' if setup['multiclass'] else 'AUC'} = {oof_auc:.3f}")
    if not top.empty:
        bullets.append(f"最重要特征：{'、'.join(str(v) for v in top['name'].head(5).tolist() if str(v))}")
    tables = [table("rf_importance", f"RF 置换重要度（top {len(top)}）", BIOMARKER_COLUMNS,
                    top.to_dict(orient="records"), note="置换重要度 = 打乱该特征后模型评分下降的均值")]
    store = {"biomarker": {"kind": "rf_importance", "table": frame, "top": top,
                           "metrics": metrics, "params": params}}
    return result(title="随机森林重要度", bullets=bullets, metrics=metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


def analysis_lasso_panel(ctx, params, outdir):
    """L1 逻辑回归 + 交叉验证选 C（Tibshirani 1996）：入选特征 + 正则化路径。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    setup = _group_setup(ctx, params)
    folds = int(params.get("folds") or DEFAULT_CV_FOLDS)
    alpha = params.get("alpha", "cv")
    pool_indices, _ = _candidate_pool(ctx, params, notes, warnings)
    if pool_indices.size > MAX_LASSO_POOL:
        warnings.append(f"候选特征 {pool_indices.size} 个超过 LASSO 预算，按 |Cohen's d| 预筛到 {MAX_LASSO_POOL} 个")
        pool_indices = _cap_pool(ctx, pool_indices, setup["y"], MAX_LASSO_POOL)
    X, feature_names = _matrix_subset(ctx, pool_indices, setup["keep"], warnings)
    fit = _lasso_fit(X, setup["y"], folds, setup["multiclass"], alpha)
    coefs = fit["magnitude"]
    selected = np.isfinite(coefs) & (np.abs(coefs) > 1e-8)
    notes.append(
        f"LASSO 正则化路径：C 网格 {np.logspace(-3, 1, 7).size} 个、{fit['folds']} 折交叉验证；"
        f"入选 {int(selected.sum())}/{len(feature_names)} 个特征"
    )
    de_frame = _de_frame(ctx) if "de" in (ctx.get("results") or {}) else None
    frame = _importance_frame(ctx, feature_names, np.abs(coefs), de_frame)
    frame["coefficient"] = [float(fit["signed"][feature_names.index(f)]) for f in frame["feature"]]
    frame["selected"] = [bool(abs(float(fit["signed"][feature_names.index(f)])) > 1e-8)
                         for f in frame["feature"]]
    columns = BIOMARKER_COLUMNS + [
        {"key": "coefficient", "label": "系数（标准化）", "type": "number"},
        {"key": "selected", "label": "入选", "type": "boolean"},
    ]
    top = frame.head(int(params.get("top_n") or 20))
    path = _lasso_path(X, setup["y"], n_points=int(params.get("path_points") or 8))
    figures = [_fig_lasso_path(path, list(frame.loc[frame["selected"], "feature"]), feature_names,
                               outdir, "lasso-path", "LASSO regularization path")]
    data_out: list[dict] = []
    try:
        path_csv = export_csv(frame, outdir, "lasso-coefficients.csv", index=False)
        data_out.append(data_file("lasso_csv", path_csv, "LASSO 系数表（全部候选特征）"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出 LASSO 系数表失败：{error}")
    metrics = {"best_c": fit["best_c"], "n_selected": int(selected.sum()),
               "n_pool": len(feature_names), "cv_auc": float(fit["cv_score"]),
               "folds": int(fit["folds"]), "n_samples": int(setup["keep"].sum())}
    bullets = [
        f"L1 逻辑回归（LASSO，Tibshirani 1996）+ {fit['folds']} 折交叉验证选 C = {fit['best_c']:.3g}"
        "（内置实现；saga 坐标下降）",
        f"候选池 {len(feature_names)} 个特征，入选 {int(selected.sum())} 个（|系数|>1e-8）",
    ]
    if np.isfinite(fit["cv_score"]):
        bullets.append(f"交叉验证评分（{'准确率' if setup['multiclass'] else 'ROC-AUC'}）= {fit['cv_score']:.3f}")
    tables = [table("lasso", f"LASSO 代谢物组合（top {len(top)}）", columns,
                    top.to_dict(orient="records"),
                    note="系数来自标准化后的 L1 逻辑回归；入选 = |系数|>1e-8")]
    store = {"biomarker": {"kind": "lasso_panel", "table": frame, "top": top,
                           "metrics": metrics, "params": params}}
    return result(title="LASSO 代谢物组合", bullets=bullets, metrics=metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


def _pick_feature(ctx, params: dict, notes: list[str]) -> str:
    """ROC/生存分析的默认特征：params.feature → 差异里上调最明显的特征（正类=case）。"""
    feature = str((params or {}).get("feature") or "").strip()
    features = _features(ctx)
    if feature:
        if feature not in set(features):
            raise bad_request(f"特征「{feature}」不在当前矩阵里（可用预览命令查看特征 id）")
        return feature
    try:
        de = _de_frame(ctx)
        de = de[de["feature"].isin(set(features))]
        if not de.empty:
            up = de[de["log2FC"] > 0].sort_values("log2FC", ascending=False)
            if not up.empty:
                chosen = str(up.iloc[0]["feature"])
                notes.append(f"未指定特征，自动选择上调幅度最大的特征（与正类方向一致）：{chosen}")
                return chosen
            ordered = de.reindex(de["log2FC"].abs().sort_values(ascending=False).index)
            chosen = str(ordered.iloc[0]["feature"])
            notes.append(f"未指定特征，自动选择 |log2FC| 最大的特征：{chosen}")
            return chosen
    except EngineError:
        pass
    if not features:
        raise prerequisite("数据集没有特征")
    chosen = features[0]
    notes.append(f"未指定特征，回退到第一个特征：{chosen}")
    return chosen


def _feature_vector(ctx, feature: str, keep: np.ndarray, warnings: list[str]) -> np.ndarray:
    X = _raw_X(ctx)
    index = _features(ctx).index(feature)
    values = X[:, index].astype(float)
    n_nan = int(np.isnan(values[keep]).sum())
    if n_nan:
        median = float(np.nanmedian(values[keep])) if np.isfinite(values[keep]).any() else 0.0
        values = np.where(np.isfinite(values), values, median)
        warnings.append(f"特征「{feature}」有 {n_nan} 个缺失值，已用中位数填补")
    return values


def analysis_roc_biomarker(ctx, params, outdir):
    """单标志物 ROC：AUC + DeLong 1988 置信区间（内置实现）。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    setup = _group_setup(ctx, params)
    keep, y, pos, neg = _binary_subset(setup, params, notes)
    feature = _pick_feature(ctx, params, notes)
    values = _feature_vector(ctx, feature, keep, warnings)
    score = values[keep]
    target = y
    if int(target.sum()) < 2 or int((1 - target).sum()) < 2:
        raise prerequisite("正/负类样本各需至少 2 个才能做 ROC")
    stats = _delong(target, score, alpha=0.05)
    if not bool(params.get("ci", True)):
        stats = {**stats, "lo": float("nan"), "hi": float("nan"), "method": stats["method"] + "；未报告 CI"}
    fpr, tpr, thresholds = _roc_points(target, score)
    frame = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    tables = [table("roc", "ROC 曲线坐标", ROC_COLUMNS, frame.to_dict(orient="records"),
                    note=f"AUC={stats['auc']:.3f}；{stats['method']}")]
    figures = [_fig_roc(fpr, tpr, stats["auc"], stats["lo"], stats["hi"], outdir,
                        "roc-curve", f"ROC: {feature}")]
    meta = _feature_meta(ctx, [feature])
    label = str(meta.iloc[0]["name"]) if not meta.empty and str(meta.iloc[0]["name"]) else feature
    metrics = {"auc": stats["auc"], "ci_low": stats["lo"], "ci_high": stats["hi"],
               "p": stats["p"], "n_pos": int(target.sum()), "n_neg": int((1 - target).sum())}
    direction = "该特征在正类里更高（AUC>0.5 为正向标志物）"
    if float(stats["auc"]) < 0.5:
        direction = (f"该特征在对照组（{neg}）里更高：AUC<0.5 表示反向标志物，"
                     f"可取 1-AUC={1.0 - float(stats['auc']):.3f} 作为同向判别力解释")
    bullets = [
        f"特征 {label}（{feature}）：AUC = {stats['auc']:.3f}",
        f"DeLong 95% CI {stats['lo']:.3f}–{stats['hi']:.3f}（{stats['method']}）",
        f"{pos}（n={int(target.sum())}）vs {neg}（n={int((1 - target).sum())}）；{direction}",
    ]
    data_out: list[dict] = []
    try:
        path = export_csv(frame, outdir, "roc-points.csv", index=False)
        data_out.append(data_file("roc_csv", path, "ROC 坐标表"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出 ROC 表失败：{error}")
    store = {"biomarker": {"kind": "roc_biomarker", "feature": feature, "metrics": metrics,
                           "roc": frame, "params": params}}
    return result(title="单标志物 ROC", bullets=bullets, metrics=metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


def analysis_biomarker_score(ctx, params, outdir):
    """多标志物线性评分：标准化后按权重加权，ROC + 分组箱线 + 写回样本列。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    setup = _group_setup(ctx, params)
    keep, y, pos, neg = _binary_subset(setup, params, notes)
    features = params.get("features") or params.get("feature")
    if isinstance(features, str):
        features = [features]
    features = [str(f) for f in (features or []) if str(f).strip()]
    all_features = _features(ctx)
    if features:
        missing = [f for f in features if f not in set(all_features)]
        if missing:
            raise bad_request(f"指定特征不在矩阵里：{'、'.join(missing[:5])}")
    else:
        de_frame = _de_frame(ctx)
        de_frame = de_frame[de_frame["feature"].isin(set(all_features))]
        ordered = de_frame.reindex(de_frame["log2FC"].abs().sort_values(ascending=False).index)
        features = [str(f) for f in ordered["feature"].head(10).tolist()]
        notes.append(f"未指定 features，自动取 |log2FC| 最大的 {len(features)} 个特征")
    if len(features) < 1:
        raise prerequisite("没有可用于评分的特征")
    X = _raw_X(ctx)
    indices = [all_features.index(f) for f in features]
    sub = X[keep][:, indices]
    if np.isnan(sub).any():
        medians = np.nanmedian(sub, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        rows, cols = np.where(~np.isfinite(sub))
        sub = sub.copy()
        sub[rows, cols] = medians[cols]
        warnings.append(f"评分特征矩阵有 {len(rows)} 个缺失值，已用中位数填补")
    means = sub.mean(axis=0)
    stds = sub.std(axis=0, ddof=1)
    stds = np.where(stds > 0, stds, 1.0)
    z = (sub - means) / stds
    weights_mode = str(params.get("weights") or "model").strip().lower()
    if weights_mode in ("model", "lasso", "logit"):
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

            model = LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs",
                                       class_weight="balanced", random_state=SEED)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                model.fit(z, y)
            weights = np.asarray(model.coef_[0], dtype=float)
            notes.append("权重来自 L2 逻辑回归的标准化系数（weights=model）")
        except Exception as error:  # noqa: BLE001
            weights = np.ones(len(features), dtype=float)
            warnings.append(f"模型权重拟合失败（{error}），改用等权")
    elif weights_mode in ("de", "logfc"):
        de_frame = _de_frame(ctx).set_index("feature")
        weights = np.asarray([_as_number(de_frame["log2FC"].get(f, 0.0)) for f in features], dtype=float)
        notes.append("权重来自差异分析的 log2FC（weights=de）")
    else:
        weights = np.ones(len(features), dtype=float)
        notes.append("等权平均（weights=equal）")
    denom = float(np.sum(np.abs(weights)))
    if not math.isfinite(denom) or denom <= 0:
        weights = np.ones(len(features), dtype=float)
        denom = float(len(features))
    score = z @ weights / denom
    stats = _delong(y, score, alpha=0.05)
    fpr, tpr, thresholds = _roc_points(y, score)
    roc_frame = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    labels_kept = setup["labels"][keep]
    tables = []
    sample_rows = pd.DataFrame({
        "sample": np.asarray(setup["samples"])[keep],
        "group": labels_kept,
        "score": score,
    })
    tables.append(table("biomarker_score", "样本评分表", [
        {"key": "sample", "label": "样本", "type": "string"},
        {"key": "group", "label": "分组", "type": "string"},
        {"key": "score", "label": "评分", "type": "number"},
    ], sample_rows.head(200).to_dict(orient="records"),
        note=f"共 {len(sample_rows)} 个样本（表内最多 200 行，完整表见 data_out）"))
    weight_frame = pd.DataFrame({
        "feature": features,
        "weight": weights,
        "abs_weight": np.abs(weights),
    }).sort_values("abs_weight", ascending=False)
    tables.append(table("score_weights", "评分权重", [
        {"key": "feature", "label": "特征", "type": "string"},
        {"key": "weight", "label": "权重", "type": "number"},
        {"key": "abs_weight", "label": "|权重|", "type": "number"},
    ], weight_frame.to_dict(orient="records"), note=f"权重模式：{weights_mode}"))
    figures = [
        _fig_roc(fpr, tpr, stats["auc"], stats["lo"], stats["hi"], outdir,
                 "roc-curve", "ROC: biomarker score"),
        _fig_boxplot(ctx, score, labels_kept, outdir, "biomarker-score-box",
                     "Biomarker score by group", "Score", [pos, neg]),
    ]
    data_out: list[dict] = []
    try:
        path = export_csv(sample_rows, outdir, "biomarker-score.csv", index=False)
        data_out.append(data_file("score_csv", path, "样本评分表（完整）"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出评分表失败：{error}")
    # 把评分写回样本表（registry 消费 sample_columns/sample_values）
    full_scores = np.full(len(_samples(ctx)), np.nan, dtype=float)
    full_scores[np.where(keep)[0]] = score
    metrics = {"auc": stats["auc"], "ci_low": stats["lo"], "ci_high": stats["hi"], "p": stats["p"],
               "n_features": len(features), "weights": weights_mode,
               "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    bullets = [
        f"{len(features)} 个特征的标准化加权评分（权重：{weights_mode}）；AUC = {stats['auc']:.3f}"
        f"（DeLong 95% CI {stats['lo']:.3f}–{stats['hi']:.3f}）",
        f"{pos} vs {neg}：{int(y.sum())} vs {int((1 - y).sum())} 个样本",
        "评分已写回样本表（列名 biomarker_score），可直接用于分组/生存分析",
    ]
    store = {"biomarker": {"kind": "biomarker_score", "features": features, "weights": weights.tolist(),
                           "metrics": metrics, "params": params}}
    payload = result(title="组合评分", bullets=bullets, metrics=metrics, tables=tables,
                     figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store,
                     sample_columns=[{"key": "biomarker_score", "label": "标志物评分",
                                      "kind": "numeric"}],
                     sample_values={"biomarker_score": full_scores})
    return payload


def analysis_model_validation(ctx, params, outdir):
    """模型验证：按患者/样本的嵌套交叉验证 + 分组自助法乐观度校正（Harrell 1996）。

    P0-2 修正：不再"全样本筛特征/填补后 CV"。所有看标签的步骤（Cohen's d 筛选、Youden 阈值、
    模型）都只在训练折内完成；自助法按受试者整块重抽并重做全流程；指标拆分为
    apparent（表观，乐观）/ cv_auc = nested_cv（嵌套交叉验证）/ external（未提供）。
    """
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    setup = _group_setup(ctx, params)
    keep, y, pos, neg = _binary_subset(setup, params, notes)
    n_boot = int(params.get("n_boot") or 200)
    folds = int(params.get("folds") or DEFAULT_CV_FOLDS)
    n_features = int(params.get("n_features") or DEFAULT_N_FEATURES)
    n_features = int(max(1, min(n_features, 100)))
    pool_indices, _ = _candidate_pool(ctx, params, notes, warnings)
    if pool_indices.size > MAX_LASSO_POOL:
        pool_indices = _cap_pool(ctx, pool_indices, setup["y"], MAX_LASSO_POOL)
        warnings.append(f"候选特征过多，已按 |Cohen's d| 预筛到 {MAX_LASSO_POOL} 个")
    X_raw, feature_names = _matrix_subset(ctx, pool_indices, keep, warnings, impute=False)
    if int(y.sum()) < 3 or int((1 - y).sum()) < 3:
        raise prerequisite("正/负类样本各需至少 3 个才能做模型验证")
    subjects, subject_col = _subject_vector(ctx, params)
    subjects_kept = None if subjects is None else subjects[np.asarray(keep, dtype=bool)]
    if subjects_kept is not None:
        notes.append(f"按受试者「{subject_col}」分组划分与整块重抽（{len(np.unique(subjects_kept))} 个个体）")
    else:
        warnings.append(
            "样本表里没有受试者/患者 ID 列（subject/patient/donor 等）：嵌套 CV 按样本划分、"
            "自助法按样本重抽；若同一样本有重复测量，泛化误差会被低估"
        )

    # 嵌套交叉验证（训练折内：填补 → 标度 → Cohen's d 筛选 → 模型 → 内层 CV Youden 阈值）
    nested = nested_cv_auc(X_raw, y, groups=subjects_kept, folds=folds, seed=SEED,
                           n_features=n_features, model="logit")
    nested_auc = float(nested["auc"])
    oof = np.asarray(nested["oof_proba"], dtype=float)
    finite_oof = np.isfinite(oof)
    if finite_oof.any():
        oof = np.where(finite_oof, oof, float(np.nanmedian(oof[finite_oof])))
    else:
        oof = np.full(y.size, 0.5)
    calibration = _calibration_frame(oof, y, n_bins=int(params.get("bins") or 5))
    threshold_values = [value for value in nested.get("fold_thresholds") or [] if math.isfinite(float(value))]
    threshold_mean = float(np.mean(threshold_values)) if threshold_values else 0.5

    # 自助法：按受试者整块重抽 + 每次重做全流程
    boot = _bootstrap_optimism(X_raw, y, n_boot, notes, warnings, groups=subjects_kept,
                               n_features=n_features, model="logit")
    metrics = {
        "apparent_auc": float(boot["apparent"]),
        "optimism": float(boot["optimism"]),
        "corrected_auc": float(boot["corrected"]),
        "cv_auc": nested_auc,
        "nested_cv_auc": nested_auc,
        "external_auc": None,
        "folds": int(nested["folds"]),
        "n_boot": int(boot["n_boot"]),
        "n_boot_valid": int(boot["n_valid"]),
        "bootstrap_grouped": bool(boot["grouped"]),
        "n_pool": len(feature_names),
        "n_samples": int(y.size),
        "n_features": int(nested["n_features"]),
        "threshold_youden": threshold_mean,
    }
    if boot["n_subjects"] is not None:
        metrics["n_subjects"] = int(boot["n_subjects"])
    rows = [
        {"metric": "apparent_auc", "value": metrics["apparent_auc"],
         "detail": "全样本拟合的表观 AUC（乐观，不等于验证）"},
        {"metric": "optimism", "value": metrics["optimism"],
         "detail": "自助法平均乐观度（Harrell 1996；按受试者整块重抽）"},
        {"metric": "corrected_auc", "value": metrics["corrected_auc"], "detail": "乐观度校正后 AUC"},
        {"metric": "nested_cv_auc", "value": nested_auc,
         "detail": f"{metrics['folds']} 折嵌套 CV（训练折内完成填补/标度/筛选/模型/阈值）"},
        {"metric": "threshold_youden", "value": threshold_mean, "detail": "训练折内层 CV 的平均 Youden 阈值"},
        {"metric": "n_boot", "value": float(boot["n_boot"]), "detail": "自助重抽样次数"},
        {"metric": "n_boot_valid", "value": float(boot["n_valid"]), "detail": "有效自助重抽样次数"},
        {"metric": "n_samples", "value": float(y.size), "detail": "有效样本数"},
    ]
    if boot["n_subjects"] is not None:
        rows.append({"metric": "n_subjects", "value": float(boot["n_subjects"]), "detail": "参与重抽的个体数"})
    tables = [
        table("model_validation", "模型验证指标", METRIC_COLUMNS, rows,
              note="模型：标准化 + L2 逻辑回归；乐观度 = 自助样本内 AUC − 原样本 AUC 的均值；"
                   "外部验证集未提供（external_auc 不报告）"),
        table("calibration", "校准曲线分箱", CALIBRATION_COLUMNS, calibration.to_dict(orient="records"),
              note="按嵌套交叉验证外推预测概率分位分箱；CI 为 Wilson 95%"),
    ]
    figures = [_fig_calibration(calibration, nested_auc, outdir, "calibration-curve", "Calibration curve")]
    data_out: list[dict] = []
    try:
        path = export_csv(calibration, outdir, "calibration.csv", index=False)
        data_out.append(data_file("calibration_csv", path, "校准表"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出校准表失败：{error}")
    bullets = [
        f"表观 AUC {metrics['apparent_auc']:.3f}；自助法乐观度 {metrics['optimism']:.3f}，"
        f"校正后 AUC {metrics['corrected_auc']:.3f}（Harrell 1996；"
        f"{metrics['n_boot_valid']}/{metrics['n_boot']} 次有效"
        + (f"，{metrics['n_subjects']} 个受试者" if boot["n_subjects"] is not None else "") + "）",
        f"嵌套交叉验证 AUC {nested_auc:.3f}（{metrics['folds']} 折；训练折内完成全部看标签的步骤）；"
        f"平均 Youden 阈值 {threshold_mean:.3f}",
        "未提供独立外部验证集：external_auc 不报告；表观 AUC 不等于外部验证",
        "方法：内置实现（嵌套 CV + bootstrap optimism + 交叉验证校准；模型为标准化 + L2 逻辑回归）",
    ]
    store = {"biomarker": {"kind": "model_validation", "metrics": metrics,
                           "calibration": calibration, "params": params}}
    return result(title="模型验证（自助法）", bullets=bullets, metrics=metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


def analysis_survival_metabolite(ctx, params, outdir):
    """代谢物分组生存：中位/最优 cutoff + logrank + Cox（lifelines）+ KM。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    times, events, time_col, event_col = _survival_columns(ctx, params, warnings)
    feature = _pick_feature(ctx, params, notes)
    # 生存分析也剔除 QC 样本；未知事件按 NA 处理（event_unknown='na' 时由 _survival_columns 置 NaN）
    n_unknown_event = int((~np.isfinite(events)).sum())
    mask = _sample_mask(ctx) & np.isfinite(times) & np.isfinite(events) & (times > 0)
    if int(mask.sum()) < 8:
        raise prerequisite(f"可用生存样本只有 {int(mask.sum())} 个（需要 ≥8），请检查临床表")
    values = _feature_vector(ctx, feature, mask, warnings)
    score_all = values[mask]
    times = times[mask]
    events = np.asarray(events, dtype=float)[mask].astype(int)
    if int(events.sum()) < 2:
        raise prerequisite("生存事件（event=1）少于 2 个，无法做 logrank / Cox 分析")
    center = float(np.nanmedian(score_all))
    scale = float(np.nanstd(score_all, ddof=1))
    z = (score_all - center) / (scale if scale > 0 else 1.0)
    cutoff, cutoff_note = _cutoff_threshold(z, times, events, params.get("cutoff") or "median")
    high = z > cutoff
    if high.sum() < 3 or (~high).sum() < 3:
        cutoff = float(np.nanmedian(z))
        high = z > cutoff
        warnings.append("按当前 cutoff 分组后某组样本少于 3 个，已退回中位数切分")
    if high.sum() < 2 or (~high).sum() < 2:
        raise prerequisite("按该阈值分组后每组样本太少（<2），无法做生存分析")
    chi2, logrank_p, logrank_method = _logrank(times[high], events[high], times[~high], events[~high])
    try:
        cox = _cox_univariate(times, events, z)
    except EngineError as error:
        cox = {}
        warnings.append(f"Cox 回归跳过：{error.message}")
    rows = []
    for label, selected in ((f"High ({feature})", high), (f"Low ({feature})", ~high)):
        rows.append({
            "group": label, "n": int(selected.sum()), "n_events": int(events[selected].sum()),
            "median_survival": _km_median(times[selected], events[selected]),
        })
    groups_curve = {
        "High": (times[high], events[high]),
        "Low": (times[~high], events[~high]),
    }
    figures = [_fig_km(groups_curve, logrank_p, cox.get("hr"), feature, outdir,
                       "km-metabolite", f"Kaplan-Meier: {feature}")]
    tables = [
        table("km_groups", "生存分组汇总", SURVIVAL_COLUMNS, rows,
              note=f"cutoff={cutoff:.3f}（{cutoff_note}）；时间列 {time_col}、事件列 {event_col}"),
        table("km_test", "logrank 与 Cox", [
            {"key": "test", "label": "检验", "type": "string"},
            {"key": "value", "label": "统计量", "type": "number"},
            {"key": "p", "label": "P 值", "type": "number"},
            {"key": "detail", "label": "说明", "type": "string"},
        ], [
            {"test": "logrank", "value": chi2, "p": logrank_p, "detail": logrank_method},
            {"test": "Cox HR (per SD)", "value": cox.get("hr"), "p": cox.get("p"),
             "detail": cox.get("method", "未拟合")},
            {"test": "Cox HR 95% 下限", "value": cox.get("lo"), "p": None, "detail": ""},
            {"test": "Cox HR 95% 上限", "value": cox.get("hi"), "p": None, "detail": ""},
        ], note="Cox 以 z 标准化后的特征值作为协变量（HR 为每 1 个 SD 的风险比）"),
    ]
    data_out: list[dict] = []
    try:
        sample_rows = pd.DataFrame({
            "sample": np.asarray(setup_samples := _samples(ctx))[mask],
            "time": times, "event": events, "feature_z": z,
            "group": np.where(high, "High", "Low"),
        })
        path = export_csv(sample_rows, outdir, "survival-groups.csv", index=False)
        data_out.append(data_file("survival_csv", path, "样本生存分组表"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出生存分组表失败：{error}")
    metrics = {
        "logrank_chi2": chi2, "logrank_p": logrank_p, "hr": cox.get("hr"),
        "hr_low": cox.get("lo"), "hr_high": cox.get("hi"), "cox_p": cox.get("p"),
        "cutoff": cutoff, "n_samples": int(mask.sum()), "n_events": int(events.sum()),
        "n_high": int(high.sum()), "n_low": int((~high).sum()),
        "n_unknown_event": int(n_unknown_event),
    }
    bullets = [
        f"特征 {feature} 按 {cutoff_note} 分组：High {int(high.sum())} 例 / Low {int((~high).sum())} 例",
        f"logrank p = {logrank_p:.3g}（{logrank_method}）",
    ]
    if cox:
        bullets.append(f"Cox HR = {cox.get('hr'):.3f}（95% CI {cox.get('lo'):.3f}–{cox.get('hi'):.3f}，"
                       f"p = {cox.get('p'):.3g}；{cox.get('method')}）")
    bullets.append("方法：KM 乘积极限 + logrank + Cox（lifelines，缺失时内置 Newton-Raphson）")
    store = {"biomarker": {"kind": "survival_metabolite", "feature": feature, "metrics": metrics,
                           "cutoff": cutoff, "params": params}}
    return result(title="代谢物分组生存", bullets=bullets, metrics=metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


# ---------------------------------------------------------------------------
# 10. PLOTS（rf_importance / lasso_path / roc_curve / km_metabolite / calibration_curve）
# ---------------------------------------------------------------------------


def _as_frame(value) -> pd.DataFrame:
    """缓存里的表可能已被 JSON 化成 list[dict]（registry jsonable），这里统一还原。"""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _stored(ctx, kind: str):
    results = ctx.get("results")
    if not isinstance(results, dict):
        return None
    store = results.get("biomarker")
    if isinstance(store, dict) and str(store.get("kind")) == kind:
        return store
    return None


def plot_rf_importance(ctx, params, outdir):
    """RF 置换重要度条形图（无缓存时现场重算）。"""
    params = dict(params or {})
    stored = _stored(ctx, "rf_importance")
    top_cached = _as_frame(stored.get("top")) if stored is not None else pd.DataFrame()
    if stored is not None and not top_cached.empty:
        top = top_cached
        top_n = int(params.get("top_n") or len(top))
        return _fig_importance(top, outdir, "rf-importance", "Random forest permutation importance",
                               top_n=top_n)
    setup = _group_setup(ctx, params)
    notes: list[str] = []
    warnings: list[str] = []
    n_trees = int(params.get("n_trees") or 300)
    pool_indices, _ = _candidate_pool(ctx, params, notes, warnings)
    if pool_indices.size > MAX_POOL:
        pool_indices = _cap_pool(ctx, pool_indices, setup["y"], MAX_POOL)
    X, feature_names = _matrix_subset(ctx, pool_indices, setup["keep"], warnings)
    target = setup["codes"] if setup["multiclass"] else setup["y"]
    forest = _rf_model(X, target, n_trees=n_trees)
    mean, _ = _rf_importance(forest, X, target, setup["multiclass"])
    de_frame = _de_frame(ctx) if "de" in (ctx.get("results") or {}) else None
    frame = _importance_frame(ctx, feature_names, mean, de_frame)
    return _fig_importance(frame, outdir, "rf-importance", "Random forest permutation importance",
                           top_n=int(params.get("top_n") or 20))


def plot_lasso_path(ctx, params, outdir):
    """LASSO 正则化路径图（入选特征数 + 系数轨迹）。"""
    params = dict(params or {})
    stored = _stored(ctx, "lasso_panel")
    setup = _group_setup(ctx, params)
    notes: list[str] = []
    warnings: list[str] = []
    pool_indices, _ = _candidate_pool(ctx, params, notes, warnings)
    if pool_indices.size > MAX_LASSO_POOL:
        pool_indices = _cap_pool(ctx, pool_indices, setup["y"], MAX_LASSO_POOL)
    X, feature_names = _matrix_subset(ctx, pool_indices, setup["keep"], warnings)
    path = _lasso_path(X, setup["y"], n_points=int(params.get("path_points") or 8))
    return _fig_lasso_path(path, [], feature_names, outdir, "lasso-path", "LASSO regularization path")


def plot_roc_curve(ctx, params, outdir):
    """单特征或组合评分的 ROC 曲线。"""
    params = dict(params or {})
    setup = _group_setup(ctx, params)
    notes: list[str] = []
    warnings: list[str] = []
    keep, y, pos, neg = _binary_subset(setup, params, notes)
    feature = _pick_feature(ctx, params, notes)
    values = _feature_vector(ctx, feature, keep, warnings)
    score = values[keep]
    target = y
    if int(target.sum()) < 2 or int((1 - target).sum()) < 2:
        raise prerequisite("正/负类样本各需至少 2 个才能画 ROC")
    stats = _delong(target, score, alpha=0.05)
    fpr, tpr, _ = _roc_points(target, score)
    return _fig_roc(fpr, tpr, stats["auc"], stats["lo"], stats["hi"], outdir,
                    "roc-curve", f"ROC: {feature}")


def plot_km_metabolite(ctx, params, outdir):
    """代谢物高低分组的 KM 曲线。"""
    params = dict(params or {})
    warnings: list[str] = []
    times, events, time_col, event_col = _survival_columns(ctx, params)
    feature = _pick_feature(ctx, params, [])
    mask = _sample_mask(ctx) & np.isfinite(times) & np.isfinite(events) & (times > 0)
    if int(mask.sum()) < 8:
        raise prerequisite(f"可用生存样本只有 {int(mask.sum())} 个（需要 ≥8）")
    values = _feature_vector(ctx, feature, mask, warnings)
    score = values[mask]
    times = times[mask]
    events = np.asarray(events, dtype=float)[mask].astype(int)
    center = float(np.nanmedian(score))
    scale = float(np.nanstd(score, ddof=1)) or 1.0
    z = (score - center) / scale
    cutoff, _ = _cutoff_threshold(z, times, events, params.get("cutoff") or "median")
    high = z > cutoff
    if high.sum() < 2 or (~high).sum() < 2:
        cutoff = float(np.nanmedian(z))
        high = z > cutoff
    chi2, logrank_p, _ = _logrank(times[high], events[high], times[~high], events[~high])
    try:
        cox = _cox_univariate(times, events, z)
    except EngineError:
        cox = {}
    return _fig_km({"High": (times[high], events[high]), "Low": (times[~high], events[~high])},
                   logrank_p, cox.get("hr"), feature, outdir, "km-metabolite",
                   f"Kaplan-Meier: {feature}")


def plot_calibration_curve(ctx, params, outdir):
    """校准曲线（模型验证的内部步骤：交叉验证外推概率分箱）。"""
    params = dict(params or {})
    stored = _stored(ctx, "model_validation")
    cal_cached = _as_frame(stored.get("calibration")) if stored is not None else pd.DataFrame()
    if stored is not None and not cal_cached.empty:
        frame = cal_cached
        metrics = stored.get("metrics") or {}
        auc = float(metrics.get("cv_auc") or float("nan"))
        return _fig_calibration(frame, auc, outdir, "calibration-curve", "Calibration curve")
    setup = _group_setup(ctx, params)
    notes: list[str] = []
    warnings: list[str] = []
    keep, y, _, _ = _binary_subset(setup, params, notes)
    pool_indices, _ = _candidate_pool(ctx, params, notes, warnings)
    if pool_indices.size > MAX_LASSO_POOL:
        pool_indices = _cap_pool(ctx, pool_indices, keep, MAX_LASSO_POOL)
    X, _features_used = _matrix_subset(ctx, pool_indices, keep, warnings)
    proba, _ = _oof_proba(X, y, int(params.get("folds") or DEFAULT_CV_FOLDS), lambda: _logit_model(X, y), False)
    auc = _auc_binary(y, proba[:, 1])
    frame = _calibration_frame(proba[:, 1], y, n_bins=int(params.get("bins") or 5))
    return _fig_calibration(frame, auc, outdir, "calibration-curve", "Calibration curve")


# ---------------------------------------------------------------------------
# 11. 模块导出（DESIGN §2.5）
# ---------------------------------------------------------------------------

STEPS = [
    {
        "id": "biomarker",
        "title": "标志物筛选",
        "requires": ["de"],
        "params": [
            {"key": "model", "label": "模型", "type": "select", "default": "rf",
             "options": [{"value": "rf", "label": "随机森林置换重要度"},
                         {"value": "lasso", "label": "LASSO（L1 逻辑回归）"},
                         {"value": "vip", "label": "复用 PLS-DA VIP"},
                         {"value": "combo", "label": "三模型秩平均"}]},
            {"key": "n_features", "label": "输出特征数", "type": "number", "default": DEFAULT_N_FEATURES},
            {"key": "cv_folds", "label": "交叉验证折数", "type": "number", "default": DEFAULT_CV_FOLDS},
            {"key": "time_col", "label": "生存时间列", "type": "text", "default": None},
            {"key": "event_col", "label": "生存事件列", "type": "text", "default": None},
        ],
        "run": step_biomarker,
    },
]

ANALYSES = [
    {
        "id": "rf_importance",
        "title": "随机森林重要度",
        "group": "标志物与建模",
        "method": "RF + 置换重要度（Breiman 2001；内置实现）",
        "needs": ["groups"],
        "params": [
            {"key": "n_trees", "label": "树数", "type": "number", "default": 300},
            {"key": "top_n", "label": "Top N", "type": "number", "default": 20},
        ],
        "run": analysis_rf_importance,
    },
    {
        "id": "lasso_panel",
        "title": "LASSO 代谢物组合",
        "group": "标志物与建模",
        "method": "L1 逻辑回归 + 交叉验证选 C（Tibshirani 1996；内置实现）",
        "needs": ["groups"],
        "params": [
            {"key": "folds", "label": "折数", "type": "number", "default": DEFAULT_CV_FOLDS},
            {"key": "alpha", "label": "λ/C 选择", "type": "text", "default": "cv"},
        ],
        "run": analysis_lasso_panel,
    },
    {
        "id": "roc_biomarker",
        "title": "单标志物 ROC",
        "group": "标志物与建模",
        "method": "ROC/AUC + DeLong 1988 置信区间（内置实现）",
        "needs": ["groups"],
        "params": [
            {"key": "feature", "label": "特征", "type": "text", "default": ""},
            {"key": "ci", "label": "置信区间", "type": "bool", "default": True},
        ],
        "run": analysis_roc_biomarker,
    },
    {
        "id": "biomarker_score",
        "title": "组合评分",
        "group": "标志物与建模",
        "method": "多标志物标准化线性评分 + ROC + 分组（内置实现）",
        "needs": ["groups"],
        "params": [
            {"key": "features", "label": "特征列表", "type": "list", "default": []},
            {"key": "weights", "label": "权重", "type": "select", "default": "model",
             "options": [{"value": "model", "label": "逻辑回归系数"},
                         {"value": "de", "label": "差异 log2FC"},
                         {"value": "equal", "label": "等权"}]},
        ],
        "run": analysis_biomarker_score,
    },
    {
        "id": "model_validation",
        "title": "模型验证（自助法）",
        "group": "标志物与建模",
        "method": "bootstrap 乐观度校正 + 交叉验证校准（Harrell 1996；内置实现）",
        "needs": ["groups"],
        "params": [
            {"key": "n_boot", "label": "自助次数", "type": "number", "default": 200},
            {"key": "folds", "label": "折数", "type": "number", "default": DEFAULT_CV_FOLDS},
        ],
        "run": analysis_model_validation,
    },
    {
        "id": "survival_metabolite",
        "title": "代谢物分组生存",
        "group": "标志物与建模",
        "method": "KM/logrank 内置实现 + Cox（lifelines；Therneau 2000，缺失时内置 Newton-Raphson）",
        "needs": ["clinical"],
        "params": [
            {"key": "feature", "label": "特征", "type": "text", "default": ""},
            {"key": "time_col", "label": "时间列", "type": "text", "default": None},
            {"key": "event_col", "label": "事件列", "type": "text", "default": None},
            {"key": "cutoff", "label": "分组方式", "type": "text", "default": "median"},
        ],
        "run": analysis_survival_metabolite,
    },
]


PLOTS = {
    "rf_importance": plot_rf_importance,
    "lasso_path": plot_lasso_path,
    "roc_curve": plot_roc_curve,
    "km_metabolite": plot_km_metabolite,
    "calibration_curve": plot_calibration_curve,
}


__all__ = ["STEPS", "ANALYSES", "PLOTS", "step_biomarker"]
