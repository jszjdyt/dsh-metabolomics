"""
dsh-metabolomics — 基础流程步骤与分析（mb-base）

本模块提供（DESIGN.md §3 前 6 步 + §4 group=质控与基础）：
  STEPS   : qc / filter / impute / normalize / transform / pca
  ANALYSES: qc_summary / rsd_qc / missing_pattern / sample_outlier / sample_corr

Method 全部用 numpy/scipy/sklearn 自实现（本机无 xcms / MetaboAnalystR / ropls / pyopenms）：
  * qc        : 每样本 TIC（列和）、检出特征数、缺失率；QC 样本逐特征 RSD（Want 2010；Dunn 2011）
  * filter    : 非缺失比例 / 平均强度 / QC-RSD 三阈值过滤（Dunn 2011 数据清洗原则）
  * impute    : half-min / 特征中位数 / KNN（Troyanskaya 2001）/ 迭代 PCA（Josse & Husson 2012）
  * normalize : PQN 概率商归一化（Dieterle 2006）/ TIC / 中位数 / 分位数（Bolstad 2003）
  * transform : log2 / log10 / cuberoot + auto(UV)/pareto/range 标度（van den Berg 2006）
  * pca       : SVD + Hotelling T²（Hotelling 1931；SIMCA 语义）+ DModX（Wold 1994）

约定（DESIGN §2.3 / §2.5）：
  * ctx 只读；写回一律通过 result(...) 的 replace / store / sample_columns。
  * 图上文字一律英文（服务器缺 CJK 字体）；中文只出现在 summary.bullets / tables.label。
  * 缺前置 → prerequisite(中文原因)；参数非法 → bad_request(中文原因)。
  * 确定性：不引入随机性（KNN/PPCA 都是确定的初始化）。
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd

from mb_common import (
    EngineError,
    PALETTE,
    bad_request,
    data_file,
    export_csv,
    new_fig,
    palette,
    prerequisite,
    result,
    save_fig,
    table,
)

from matplotlib.patches import Ellipse, Patch  # noqa: E402 - mb_common 已设好 Agg 后端

DEFAULT_DPI = 140
DEFAULT_RSD_CUT = 30.0          # QC 可接受 RSD 上限（%，Want 2010 常用 30% 经验阈值）
MAX_MULTIVARIATE_FEATURES = 20000   # DESIGN §2.3：超过就先按方差筛（避免 SVD 爆内存）


# ---------------------------------------------------------------------------
# ctx 读取小工具（只读；不修改 ctx）
# ---------------------------------------------------------------------------


def _X(ctx) -> np.ndarray:
    arr = (ctx or {}).get("X")
    if arr is None:
        raise prerequisite("数据集没有强度矩阵，请先导入数据或载入示例数据")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise EngineError("internal", f"强度矩阵形状异常：{arr.shape}")
    return arr


def _features(ctx) -> list[str]:
    return [str(item) for item in ((ctx or {}).get("features") or [])]


def _samples(ctx) -> list[str]:
    return [str(item) for item in ((ctx or {}).get("samples") or [])]


def _info(ctx) -> pd.DataFrame:
    frame = (ctx or {}).get("sample_info")
    samples = _samples(ctx)
    if isinstance(frame, pd.DataFrame) and len(frame) == len(samples):
        return frame
    return pd.DataFrame(index=pd.Index(samples, name="sample"))


def _feature_info(ctx) -> pd.DataFrame:
    frame = (ctx or {}).get("feature_info")
    features = _features(ctx)
    if isinstance(frame, pd.DataFrame) and len(frame) == len(features):
        return frame
    return pd.DataFrame(index=pd.Index(features, name="feature"))


def _param(params, key, default=None):
    value = (params or {}).get(key)
    return default if value is None else value


def _as_float(params, key, default=None):
    value = _param(params, key, None)
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "nan", "auto"):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _as_int(params, key, default: int) -> int:
    value = _as_float(params, key, None)
    if value is None:
        return int(default)
    try:
        return int(round(value))
    except (TypeError, ValueError):
        return int(default)


def _fmt(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    if number != 0 and (abs(number) >= 1e5 or abs(number) < 1e-3):
        return f"{number:.{max(digits - 1, 1)}e}"
    return f"{number:.{digits}g}"


def _sorted_labels(values) -> list[str]:
    """标签排序：数字按数值升序，其余按字典序（保证图例/配色稳定）。"""
    uniq = list(dict.fromkeys([str(item) for item in values]))

    def key(value: str):
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, value)

    return sorted(uniq, key=key)


def _cat_colors(labels) -> dict:
    """分组 → 颜色：同一标签永远同色（按排序后的索引取 PALETTE）。"""
    order = _sorted_labels(labels)
    colors = palette(len(order))
    return {label: colors[index % len(colors)] for index, label in enumerate(order)}


def _group_labels(ctx, params=None, required: bool = False):
    """当前分组标签 (labels, key)：params.group_col 优先，其次 ctx.group_key。"""
    params = params or {}
    key = str(params.get("group_col") or (ctx or {}).get("group_key") or "").strip()
    info = _info(ctx)
    if key and key in info.columns:
        return np.asarray([str(value) for value in info[key].tolist()], dtype=object), key
    if key:
        raise prerequisite(f"样本信息里没有分组列「{key}」，请先在样本表里设置分组")
    if required:
        raise prerequisite("请先设置分组列（样本 → 设置分组）")
    return None, None


def _qc_mask(ctx, qc_label=None):
    """QC 样本掩码：qc_label 正则匹配样本名，或 sample_info 的 qc 列 = 1。"""
    samples = _samples(ctx)
    info = _info(ctx)
    mask = np.zeros(len(samples), dtype=bool)
    used = []
    label = str(qc_label or "").strip()
    if label:
        try:
            regex = re.compile(label)
        except re.error as error:
            raise bad_request(f"qc_label 正则编译失败：{error}") from error
        mask |= np.array([bool(regex.search(sample)) for sample in samples], dtype=bool)
        used.append(f"样本名匹配 /{label}/")
    for col in ("qc", "is_qc", "qc_flag", "qc_sample"):
        if col in info.columns:
            values = info[col].tolist()
            mask |= np.array([str(value).strip().lower() in ("1", "1.0", "true", "yes", "y", "qc")
                              for value in values], dtype=bool)
            used.append(f"样本列 {col}=1")
    for col in ("sample_type", "sampletype", "type"):
        if col == "type":
            continue
        if col in info.columns:
            values = [str(value).strip().lower() for value in info[col].tolist()]
            mask |= np.array([value in ("qc", "pool", "pooled", "quality control", "qc sample")
                              for value in values], dtype=bool)
            used.append(f"样本列 {col}=QC")
    return mask, "；".join(used)


# ---------------------------------------------------------------------------
# 统计小工具
# ---------------------------------------------------------------------------


def _qc_matrix(ctx):
    """QC 统计（TIC / RSD / 缺失）用的矩阵：优先 ctx.raw（原始强度，DESIGN §2.3 明确给 TIC/RSD 用）。

    返回 (matrix, source)。shape 与当前 X 不一致或 raw 缺失时回退当前矩阵。
    """
    X = _X(ctx)
    raw = (ctx or {}).get("raw")
    if raw is not None:
        try:
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 2 and arr.shape == X.shape and np.isfinite(arr).any():
                return arr, "raw"
        except (TypeError, ValueError):
            pass
    return X, "current"


def _filled_log_matrix(ctx):
    """相关性/样本一致性用的矩阵：优先 log2(raw)，缺失用特征中位数填。

    Pearson 对每样本的乘性稀释不敏感（log 后只是常数偏移，中心化即消掉），
    所以在 raw 的 log2 上算样本相关既能反映真实性又不受归一化缩放影响。
    """
    matrix, source = _qc_matrix(ctx)
    arr = np.asarray(matrix, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if not finite.all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(np.where(finite, arr, np.nan), axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        rows, cols = np.where(~finite)
        arr[rows, cols] = median[cols]
    positive = arr[arr > 0]
    offset = 0.0
    if arr.size and float(np.min(arr)) <= 0:
        offset = float(positive.min()) / 2.0 if positive.size else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log2(np.where(arr + offset > 0, arr + offset, 2.0 ** -20))
    return out, source


def _sample_metrics(X: np.ndarray):
    """每样本 (TIC, 检出特征数, 缺失率%)；NaN 容错。"""
    finite = np.isfinite(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tic = np.nansum(X, axis=1)
    tic = np.where(np.isfinite(tic), tic, 0.0)
    detected = finite.sum(axis=1).astype(np.int64)
    missing_pct = 100.0 * (1.0 - finite.mean(axis=1))
    return tic, detected, missing_pct


def _feature_rsd(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐特征 RSD(%) 与均值（NaN 容错；RSD = SD/mean × 100，Want 2010）。"""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        n = arr.shape[1] if arr.ndim == 2 else 0
        return np.full(n, np.nan), np.full(n, np.nan)
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        total = np.nansum(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
    mean = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
    sd = np.where(counts > 1, sd, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rsd = np.where(np.isfinite(mean) & (mean > 0), sd / mean * 100.0, np.nan)
    return rsd, mean


def _missing_pct_features(X: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 - np.isfinite(X).mean(axis=0))


def _skew_kurt(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐特征偏度与超峰度（Fisher；内置实现，避免 scipy 版本差异）。"""
    arr = np.asarray(mat, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(arr, axis=0)
        centered = arr - mu
        m2 = np.nanmean(centered ** 2, axis=0)
        m3 = np.nanmean(centered ** 3, axis=0)
        m4 = np.nanmean(centered ** 4, axis=0)
    sd = np.sqrt(np.maximum(m2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.where(sd > 0, m3 / np.maximum(sd, 1e-12) ** 3, 0.0)
        kurt = np.where(sd > 0, m4 / np.maximum(sd, 1e-12) ** 4 - 3.0, 0.0)
    return np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(
        kurt, nan=0.0, posinf=0.0, neginf=0.0)


def _chi2_quantile(alpha: float, df: int = 2) -> float:
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(float(alpha), int(df)))
    except Exception:  # noqa: BLE001 - scipy 不可用时用卡方表近似
        return 5.991 if int(df) == 2 else float(df) * 2.0


def _f_quantile(alpha: float, dfn: float, dfd: float) -> float:
    try:
        from scipy.stats import f

        return float(f.ppf(float(alpha), float(dfn), float(dfd)))
    except Exception:  # noqa: BLE001 - 回退：F ≈ chi2/dfn
        return _chi2_quantile(alpha, max(1, int(round(dfn)))) / max(float(dfn), 1.0)


def _ellipse_patch(points: np.ndarray, color: str, alpha: float = 0.95, linestyle: str = "--"):
    """组内 Hotelling 95% 置信椭圆（基于 2×2 协方差 + 卡方分位数）。"""
    pts = np.asarray(points, dtype=np.float64)[:, :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values[0] <= 0:
        return None
    chi2 = _chi2_quantile(alpha, 2)
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width = 2.0 * math.sqrt(values[0] * chi2)
    height = 2.0 * math.sqrt(max(values[1] * chi2, 0.0))
    return Ellipse((float(pts[:, 0].mean()), float(pts[:, 1].mean())), width, height,
                   angle=angle, facecolor="none", edgecolor=color, linewidth=1.1,
                   linestyle=linestyle, alpha=0.9)


def _draw_group_ellipses(ax, scores: np.ndarray, labels) -> None:
    if labels is None:
        return
    labels = np.asarray([str(item) for item in labels], dtype=object)
    colors = _cat_colors(labels)
    for label in _sorted_labels(labels):
        mask = labels == label
        patch = _ellipse_patch(np.asarray(scores, dtype=np.float64)[mask], colors[label])
        if patch is not None:
            ax.add_patch(patch)


# ---------------------------------------------------------------------------
# PCA / 离群诊断核心（pca 步骤、sample_outlier、基础图共用同一套定义）
# ---------------------------------------------------------------------------


def _variance_subset(X: np.ndarray, limit: int = MAX_MULTIVARIATE_FEATURES):
    """按方差挑 ≤limit 个特征（避免 SVD 在 20 万特征上爆内存）；返回 (列索引, 是否筛过)。"""
    n_features = X.shape[1]
    if n_features <= limit:
        return np.arange(n_features), False
    variances = np.zeros(n_features, dtype=np.float64)
    step = max(1, int(4_000_000 / max(X.shape[0], 1)))
    for start in range(0, n_features, step):
        chunk = np.asarray(X[:, start:start + step], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            variances[start:start + step] = np.nan_to_num(chunk.var(axis=0), nan=0.0)
    order = np.argsort(-variances, kind="stable")[:int(limit)]
    return np.sort(order), True


def _pca_core(X: np.ndarray, n_comp: int = 3, feature_limit: int = MAX_MULTIVARIATE_FEATURES) -> dict:
    """SVD PCA：返回 scores / ratio / loadings / 残差平方和（Wold 1994 语义）。"""
    arr = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise prerequisite("矩阵里还有缺失值：请先跑「缺失值填补」再做 PCA")
    n_samples, n_features = arr.shape
    if n_samples < 3 or n_features < 2:
        raise prerequisite(f"样本（{n_samples}）或特征（{n_features}）太少，无法做 PCA")
    columns, subsampled = _variance_subset(arr, feature_limit)
    sub = arr[:, columns]
    center = sub.mean(axis=0)
    centered = sub - center
    max_comp = max(2, min(n_samples - 1, centered.shape[1]))
    want = int(max(2, min(int(n_comp), max_comp, 15)))
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = int(min(want, s.size))
    scores = u[:, :k] * s[:k]
    total = float((s ** 2).sum())
    ratio = (s[:k] ** 2) / (total if total > 0 else 1.0)
    loadings = vt[:k, :]
    # 残差平方和：||Xc_i||² - Σ_j t_ij²（载荷行正交归一，交叉项为 0）
    row_norm2 = (centered ** 2).sum(axis=1)
    sse = np.maximum(row_norm2 - (scores ** 2).sum(axis=1), 0.0)
    return {
        "scores": scores,
        "ratio": ratio,
        "loadings": loadings,
        "columns": columns,
        "center": center,
        "subsampled": bool(subsampled),
        "n_features_total": int(n_features),
        "sse": sse,
        "n_comp": int(k),
    }


def _hotelling_t2(scores: np.ndarray, alpha: float = 0.95) -> tuple[np.ndarray, float, float]:
    """Hotelling T²（NIPALS/SIMCA 语义）与 95%/99% 临界值。"""
    mat = np.asarray(scores, dtype=np.float64)
    n_samples, k = mat.shape
    if k == 0 or n_samples <= k:
        return np.zeros(n_samples), float("nan"), float("nan")
    variance = mat.var(axis=0, ddof=1)
    variance = np.where(variance > 0, variance, 1.0)
    t2 = ((mat ** 2) / variance[None, :]).sum(axis=1)
    # 经典观测级 T² 控制限：k(n²-1)/(n(n-k)) · F(α; k, n-k)
    scale = k * (n_samples ** 2 - 1) / (n_samples * (n_samples - k))
    crit95 = scale * _f_quantile(0.95, k, n_samples - k)
    crit99 = scale * _f_quantile(0.99, k, n_samples - k)
    return t2, float(crit95), float(crit99)


def _dmodx(sse: np.ndarray, n_features: int, n_comp: int, alpha: float = 0.95):
    """DModX（Wold 1994；SIMCA 的 F 近似临界值）与残差标准差 s0。"""
    sse = np.asarray(sse, dtype=np.float64)
    n_samples = sse.size
    k_eff = max(int(n_features) - int(n_comp), 1)
    dmodx = np.sqrt(np.maximum(sse, 0.0) / k_eff)
    dof = max(n_samples - int(n_comp) - 1, 1)
    s0 = math.sqrt(float(np.maximum(sse, 0.0).sum()) / (dof * k_eff)) if sse.size else float("nan")
    crit = s0 * math.sqrt(_f_quantile(alpha, k_eff, max(dof * k_eff, 1))) if s0 > 0 else float("nan")
    return dmodx, float(crit), float(s0)


def _pca_from_ctx(ctx, n_comp: int = 3) -> dict:
    """优先复用 pca 步骤缓存（results["pca"]），否则现场算一遍。

    注意：result() 会把 store 过一遍 jsonable（numpy → list），所以这里统一再转回 ndarray。
    """
    cached = ((ctx or {}).get("results") or {}).get("pca")
    if isinstance(cached, dict):
        scores = _as_array(cached.get("scores"))
        if scores is not None and scores.ndim == 2 and scores.shape[0] == _X(ctx).shape[0]:
            out = dict(cached)
            ratio = _as_array(cached.get("ratio"))
            out["scores"] = scores
            out["ratio"] = ratio if ratio is not None else np.zeros(scores.shape[1])
            out["loadings"] = _as_array(cached.get("loadings"))
            out["n_comp"] = int(scores.shape[1])
            if cached.get("sse") is not None:
                out["sse"] = _as_array(cached.get("sse"))
            if cached.get("columns") is not None:
                out["columns"] = [int(value) for value in cached["columns"]]
            return out
    return _pca_core(_X(ctx), n_comp=n_comp)


def _as_array(value):
    """JSON 化过的 store（list/None）→ ndarray；失败返回 None。"""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


def _residual_sse(ctx, scores: np.ndarray, columns=None) -> np.ndarray:
    """残差平方和：||Xc_i||² − Σ_j t_ij²（载荷行正交归一，交叉项为 0）。"""
    arr = _X(ctx)
    if columns is not None:
        try:
            index = np.asarray([int(value) for value in columns], dtype=np.int64)
            if index.size and index.max() < arr.shape[1]:
                arr = arr[:, index]
        except (TypeError, ValueError):
            arr = _X(ctx)
    centered = arr - arr.mean(axis=0)
    return np.maximum((centered ** 2).sum(axis=1) - (scores ** 2).sum(axis=1), 0.0)


def _color_labels(ctx, params, color_by, qc_mask=None):
    """把 color_by（group|batch|qc|none|任意样本列）解析成 (labels, title)。"""
    lowered = str(color_by or "group").strip().lower()
    info = _info(ctx)
    if lowered in ("", "none", "no", "false"):
        return None, "none"
    if lowered in ("group", "groups", "group_key", "condition"):
        labels, key = _group_labels(ctx, params)
        return labels, (key or "group")
    if lowered in ("qc", "quality", "qc_label"):
        if qc_mask is None or not np.asarray(qc_mask).any():
            return None, "none"
        return np.asarray(["QC" if bool(flag) else "Sample" for flag in qc_mask], dtype=object), "qc"
    if lowered in ("batch", "batch_id", "plate", "run", "cohort"):
        for col in (color_by, "batch", "batch_id", "plate", "run", "cohort"):
            if col and str(col) in info.columns:
                return np.asarray([str(value) for value in info[str(col)].tolist()], dtype=object), str(col)
        return None, "none"
    if str(color_by) in info.columns:
        return np.asarray([str(value) for value in info[str(color_by)].tolist()], dtype=object), str(color_by)
    return None, "none"


# ---------------------------------------------------------------------------
# 绘图小工具（步骤自带图；kind 级基础图在 mb_plots.py）
# ---------------------------------------------------------------------------


def _fig_tic(samples, tic, labels, qc_mask, outdir: str, name: str, title: str) -> dict:
    """每样本 TIC 柱状（按 TIC 降序；分组上色，QC 样本描边）。"""
    tic = np.asarray(tic, dtype=np.float64)
    order = np.argsort(-np.nan_to_num(tic, nan=-np.inf), kind="stable")
    fig, ax = new_fig(9.6, 4.3)
    positions = np.arange(tic.size)
    handles = []
    if labels is not None:
        colors = _cat_colors(labels)
        bar_colors = [colors[str(labels[index])] for index in order]
        handles = [Patch(facecolor=colors[label], label=label) for label in _sorted_labels(labels)]
    else:
        bar_colors = ["#8a919c" if bool(qc_mask[index]) else PALETTE[0] for index in order]
        if qc_mask is not None and np.asarray(qc_mask).any():
            handles = [Patch(facecolor=PALETTE[0], label="sample"),
                       Patch(facecolor="#8a919c", label="QC")]
    ax.bar(positions, tic[order], color=bar_colors, width=0.82, linewidth=0)
    if qc_mask is not None and np.asarray(qc_mask).any():
        hit = np.array([bool(qc_mask[index]) for index in order])
        ax.scatter(positions[hit], tic[order][hit], s=18, facecolors="none",
                   edgecolors="#2b2f36", linewidths=0.8, zorder=3)
    median = float(np.nanmedian(tic)) if tic.size else 0.0
    ax.axhline(median, color=PALETTE[2], linestyle="--", linewidth=1.0)
    handles.append(Patch(facecolor="none", edgecolor=PALETTE[2], linestyle="--",
                         label=f"median {_fmt(median, 3)}"))
    step = max(1, tic.size // 24)
    ax.set_xticks(positions[::step])
    ax.set_xticklabels([samples[index] for index in order[::step]], rotation=90, fontsize=6)
    ax.set_ylabel("TIC (sum of intensities)")
    ax.set_title(title, fontsize=10)
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right", ncols=2)
    return save_fig(fig, outdir, name, title=None)


def _fig_hist(series, outdir: str, name: str, title: str, xlabel: str,
              bins: int = 48, vlines=()) -> dict:
    """多组直方图（series=[(label, values), ...]）+ 可选阈值竖线。"""
    fig, ax = new_fig(7.2, 4.2)
    for index, (label, values) in enumerate(series):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, alpha=0.55, color=PALETTE[index % len(PALETTE)],
                label=label)
    for value, text in vlines:
        if value is None or not np.isfinite(value):
            continue
        ax.axvline(float(value), color="#e15759", linestyle="--", linewidth=1.0, label=text)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    return save_fig(fig, outdir, name, title=None)


def _fig_missing(samples, sample_missing, feature_missing, outdir: str, name: str) -> dict:
    """缺失率：左=每样本缺失率（排序柱状），右=特征缺失率分布。"""
    fig, axes = new_fig(10.4, 4.0, ncols=2)
    axes = np.atleast_1d(axes).ravel()
    sample_missing = np.asarray(sample_missing, dtype=np.float64)
    order = np.argsort(-sample_missing, kind="stable")
    positions = np.arange(sample_missing.size)
    axes[0].bar(positions, sample_missing[order], color=PALETTE[0], width=0.82, linewidth=0)
    axes[0].axhline(float(np.nanmedian(sample_missing)), color=PALETTE[2], linestyle="--",
                    linewidth=1.0, label=f"median {_fmt(np.nanmedian(sample_missing), 3)}%")
    step = max(1, sample_missing.size // 14)
    axes[0].set_xticks(positions[::step])
    axes[0].set_xticklabels([samples[index] for index in order[::step]], rotation=90, fontsize=6)
    axes[0].set_ylabel("missing (%)")
    axes[0].set_title("Missing values per sample", fontsize=10)
    axes[0].legend(fontsize=7)
    values = np.asarray(feature_missing, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size:
        axes[1].hist(values, bins=40, color=PALETTE[1], alpha=0.75)
        axes[1].axvline(50.0, color="#e15759", linestyle="--", linewidth=1.0, label="50% line")
        axes[1].legend(fontsize=7)
    axes[1].set_xlabel("missing (%) per feature", fontsize=9)
    axes[1].set_ylabel("features", fontsize=9)
    axes[1].set_title("Feature-level missingness", fontsize=10)
    return save_fig(fig, outdir, name, title=None)


def _fig_pca_scores(scores, ratio, labels, outdir: str, name: str, title: str) -> dict:
    """PC1–PC2 散点：分组上色 + 组内 95% Hotelling 椭圆。"""
    scores = np.asarray(scores, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    fig, ax = new_fig(6.8, 5.2)
    if labels is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=26, color=PALETTE[0], linewidths=0,
                   alpha=0.85, label="samples")
    else:
        labels = np.asarray([str(item) for item in labels], dtype=object)
        colors = _cat_colors(labels)
        for label in _sorted_labels(labels):
            mask = labels == label
            ax.scatter(scores[mask, 0], scores[mask, 1], s=26, color=colors[label],
                       linewidths=0, alpha=0.85, label=f"{label} (n={int(mask.sum())})")
        _draw_group_ellipses(ax, scores, labels)
    pc1 = ratio[0] * 100 if ratio.size else 0.0
    pc2 = ratio[1] * 100 if ratio.size > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0.0, color="#dfe3e8", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#dfe3e8", linewidth=0.8, zorder=0)
    if labels is not None:
        ax.legend(fontsize=7, loc="best")
    return save_fig(fig, outdir, name, title=None)


def _fig_pca_variance(ratio, outdir: str, name: str, title: str) -> dict:
    ratio = np.asarray(ratio, dtype=np.float64)
    n_show = int(min(ratio.size, 15))
    fig, ax = new_fig(7.0, 4.0)
    positions = np.arange(1, n_show + 1)
    ax.bar(positions, ratio[:n_show] * 100, color=PALETTE[0], width=0.7, alpha=0.9,
           label="individual")
    ax.plot(positions, np.cumsum(ratio[:n_show]) * 100, color=PALETTE[2], marker="o",
            markersize=3, linewidth=1.2, label="cumulative")
    ax.set_xlabel("principal component", fontsize=9)
    ax.set_ylabel("explained variance (%)", fontsize=9)
    ax.set_xticks(positions)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    return save_fig(fig, outdir, name, title=None)


def _fig_outliers(t2, t2_crit95, t2_crit99, dmodx, dmodx_crit, labels, outdir: str,
                  name: str, title: str) -> dict:
    """两联图：Hotelling T²（含 95/99% 限）与 DModX（含 F 近似限）。"""
    t2 = np.asarray(t2, dtype=np.float64)
    dmodx = np.asarray(dmodx, dtype=np.float64)
    fig, axes = new_fig(10.6, 4.2, ncols=2)
    axes = np.atleast_1d(axes).ravel()
    positions = np.arange(t2.size)
    colors = np.where(t2 > t2_crit99, "#c0392b", np.where(t2 > t2_crit95, "#e67e22", "#4e79a7"))
    axes[0].bar(positions, t2, color=colors, width=0.85, linewidth=0)
    if np.isfinite(t2_crit95):
        axes[0].axhline(t2_crit95, color="#e67e22", linestyle="--", linewidth=1.0,
                        label=f"95% limit {_fmt(t2_crit95, 3)}")
    if np.isfinite(t2_crit99):
        axes[0].axhline(t2_crit99, color="#c0392b", linestyle=":", linewidth=1.0,
                        label=f"99% limit {_fmt(t2_crit99, 3)}")
    axes[0].set_xlabel("sample index", fontsize=9)
    axes[0].set_ylabel("Hotelling T$^2$", fontsize=9)
    axes[0].set_title("Hotelling T$^2$ (colored: over limit)", fontsize=10)
    axes[0].legend(fontsize=7)
    colors2 = np.where(dmodx > dmodx_crit, "#c0392b", "#76b7b2")
    axes[1].bar(positions, dmodx, color=colors2, width=0.85, linewidth=0)
    if np.isfinite(dmodx_crit):
        axes[1].axhline(dmodx_crit, color="#c0392b", linestyle="--", linewidth=1.0,
                        label=f"alpha limit {_fmt(dmodx_crit, 3)}")
    axes[1].set_xlabel("sample index", fontsize=9)
    axes[1].set_ylabel("DModX", fontsize=9)
    axes[1].set_title("DModX (model residual distance)", fontsize=10)
    axes[1].legend(fontsize=7)
    if labels is not None:
        for ax in axes:
            ax.set_xticks(positions[::max(1, positions.size // 12)])
    fig.suptitle(title, fontsize=11)
    return save_fig(fig, outdir, name, title=None)


# ---------------------------------------------------------------------------
# 步骤：qc（样本质控）
# ---------------------------------------------------------------------------


def step_qc(ctx, params, outdir):
    """样本质控：TIC / 检出特征数 / 缺失率 + QC 样本逐特征 RSD（Want 2010；Dunn 2011）。"""
    params = dict(params or {})
    X, qc_source = _qc_matrix(ctx)
    samples = _samples(ctx)
    features = _features(ctx)
    finfo = _feature_info(ctx)
    min_features = max(_as_int(params, "min_features", 0), 0)

    tic, detected, missing_pct = _sample_metrics(X)
    qc_mask, qc_desc = _qc_mask(ctx, params.get("qc_label"))
    n_qc = int(qc_mask.sum())
    labels, group_key = _group_labels(ctx, params)

    if n_qc >= 2:
        rsd, _mean = _feature_rsd(X[qc_mask])
        rsd_scope = f"QC({n_qc})"
    elif X.shape[0] >= 2:
        rsd, _mean = _feature_rsd(X)
        rsd_scope = "all samples"
    else:
        rsd = np.full(X.shape[1], np.nan)
        rsd_scope = "none"
    rsd_finite = rsd[np.isfinite(rsd)]
    median_rsd = float(np.median(rsd_finite)) if rsd_finite.size else float("nan")
    pct_pass = float((rsd_finite <= DEFAULT_RSD_CUT).mean() * 100.0) if rsd_finite.size else 0.0
    feature_missing = _missing_pct_features(X)

    median_tic = float(np.median(tic)) if tic.size else 0.0
    tic_cv = float(np.std(tic) / np.mean(tic) * 100.0) if tic.size and np.mean(tic) > 0 else 0.0
    below = np.where(detected < min_features)[0] if min_features > 0 else np.array([], dtype=int)

    rows = []
    for index, sample in enumerate(samples):
        row = {"sample": sample,
               "role": "QC" if qc_mask[index] else "sample",
               "tic": float(tic[index]),
               "n_features": int(detected[index]),
               "missing_pct": float(missing_pct[index]),
               "flag": bool(min_features > 0 and detected[index] < min_features)}
        if labels is not None:
            row["group"] = str(labels[index])
        rows.append(row)
    columns = [{"key": "sample", "label": "样本", "type": "string"},
               {"key": "role", "label": "类型", "type": "string"}]
    if labels is not None:
        columns.append({"key": "group", "label": f"分组（{group_key}）", "type": "string"})
    columns += [{"key": "tic", "label": "TIC", "type": "number"},
                {"key": "n_features", "label": "检出特征数", "type": "number"},
                {"key": "missing_pct", "label": "缺失率%", "type": "number"},
                {"key": "flag", "label": "低于最少特征", "type": "boolean"}]

    feature_rows = []
    if rsd_finite.size:
        order = np.argsort(-np.nan_to_num(rsd, nan=-np.inf), kind="stable")[:200]
    else:
        order = np.argsort(-feature_missing, kind="stable")[:200]
    for index in order:
        index = int(index)
        item = {"feature": features[index] if index < len(features) else str(index),
                "rsd_pct": float(rsd[index]) if np.isfinite(rsd[index]) else None,
                "missing_pct": float(feature_missing[index])}
        for col in ("mz", "rt", "name"):
            if col in finfo.columns:
                try:
                    value = finfo[col].iloc[index]
                    if pd.notna(value):
                        item[col] = float(value) if col in ("mz", "rt") else str(value)
                except Exception:  # noqa: BLE001
                    continue
        feature_rows.append(item)
    feature_columns = [{"key": "feature", "label": "特征", "type": "string"}]
    for col, label in (("mz", "m/z"), ("rt", "RT"), ("name", "名称")):
        if feature_rows and col in feature_rows[0]:
            feature_columns.append({"key": col, "label": label,
                                    "type": "number" if col in ("mz", "rt") else "string"})
    feature_columns += [{"key": "rsd_pct", "label": f"RSD%（{rsd_scope}）", "type": "number"},
                        {"key": "missing_pct", "label": "缺失率%", "type": "number"}]

    figures = [
        _fig_tic(samples, tic, labels, qc_mask, outdir, "qc_tic",
                 "Total ion current (TIC) per sample"),
        _fig_hist([(f"feature RSD ({rsd_scope})", rsd)], outdir, "qc_rsd",
                  "Feature RSD distribution", "RSD (%)", bins=48,
                  vlines=[(DEFAULT_RSD_CUT, f"{DEFAULT_RSD_CUT:.0f}% threshold"),
                          (median_rsd, f"median {_fmt(median_rsd, 3)}%")]),
        _fig_missing(samples, missing_pct, feature_missing, outdir, "qc_missing"),
    ]

    bullets = [
        f"{len(samples)} 个样本 × {X.shape[1]} 个特征；每样本检出特征中位 "
        f"{float(np.median(detected)):.0f}，缺失率中位 {float(np.median(missing_pct)):.1f}%",
        f"TIC 范围 {_fmt(np.min(tic), 4)}–{_fmt(np.max(tic), 4)}（中位 {_fmt(median_tic, 4)}），"
        f"样本间 CV {tic_cv:.1f}%",
    ]
    if n_qc >= 2:
        bullets.append(
            f"识别到 {n_qc} 个 QC 样本（{qc_desc or 'qc 列'}）：QC 特征 RSD 中位 {median_rsd:.1f}%，"
            f"{pct_pass:.0f}% 的特征 RSD ≤ {DEFAULT_RSD_CUT:.0f}%")
    else:
        bullets.append(
            "没有识别到 QC 样本（qc_label 为空且样本表无 qc 列）：上面 RSD 按全部样本计算，"
            "只能作为整体变异参考")
    if min_features > 0:
        if below.size:
            names = "、".join(samples[index] for index in below[:8])
            bullets.append(f"检出特征数 < {min_features} 的样本 {below.size} 个（{names}"
                           + ("…" if below.size > 8 else "") + "），建议在样本表里剔除后重跑")
        else:
            bullets.append(f"所有样本检出特征数都 ≥ {min_features}")
    if X.shape[1] and pct_pass < 60 and n_qc >= 2:
        bullets.append("QC 特征 RSD 合格比例偏低（<60%）：检查进样稳定性，或在「特征过滤」里按 QC-RSD 剔除")

    metrics = {
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]), "n_qc": n_qc,
        "median_tic": median_tic, "tic_cv_pct": tic_cv,
        "median_detected": float(np.median(detected)) if detected.size else 0.0,
        "median_missing_pct": float(np.median(missing_pct)) if missing_pct.size else 0.0,
        "median_rsd_pct": median_rsd, "pct_rsd_pass": pct_pass,
        "n_below_min_features": int(below.size), "rsd_scope": rsd_scope,
    }

    store = {"qc": {
        "samples": list(samples),
        "tic": [float(value) for value in tic],
        "detected": [int(value) for value in detected],
        "missing_pct": [float(value) for value in missing_pct],
        "qc_mask": [bool(value) for value in qc_mask],
        "qc_desc": qc_desc,
        "rsd": [float(value) if np.isfinite(value) else None for value in rsd],
        "rsd_scope": rsd_scope,
        "median_rsd_pct": median_rsd,
        "pct_rsd_pass": pct_pass,
        "n_features": int(X.shape[1]),
    }}

    data_out = []
    try:
        path = export_csv(pd.DataFrame(rows), outdir, "qc-samples.csv", index=False)
        data_out.append(data_file("qc_csv", path, "样本级质控表"))
    except Exception:  # noqa: BLE001 - 导出失败不影响结果
        pass
    try:
        if feature_rows:
            path = export_csv(pd.DataFrame(feature_rows), outdir, "qc-features.csv", index=False)
            data_out.append(data_file("qc_csv", path, "特征级 RSD/缺失表"))
    except Exception:  # noqa: BLE001
        pass

    return result(
        title="样本质控",
        bullets=bullets,
        metrics=metrics,
        tables=[
            table("qc_samples", "样本级质控指标", columns, rows,
                  note=f"QC 样本识别：{qc_desc or '未识别'}"),
            table("qc_features", "特征级变异与缺失（Top 200）", feature_columns, feature_rows),
        ],
        figures=figures,
        data_out=data_out,
        store=store,
        warnings=[] if n_qc >= 2 else ["没有识别到 QC 样本：QC-RSD 相关过滤与诊断不可用"],
    )


# ---------------------------------------------------------------------------
# 步骤：filter（特征过滤）
# ---------------------------------------------------------------------------


def step_filter(ctx, params, outdir):
    """特征过滤：非缺失比例 / 平均强度 / QC-RSD 三阈值（Dunn 2011 数据清洗原则）。"""
    params = dict(params or {})
    X = _X(ctx)
    features = _features(ctx)
    finfo = _feature_info(ctx)
    if not features or len(features) != X.shape[1]:
        features = [f"F{index}" for index in range(X.shape[1])]

    min_frac = _as_float(params, "min_sample_frac", 0.5)
    min_frac = float(np.clip(0.0 if min_frac is None else min_frac, 0.0, 1.0))
    min_intensity = _as_float(params, "min_intensity", 0.0) or 0.0
    rsd_max = _as_float(params, "rsd_max", None)

    finite = np.isfinite(X)
    counts = finite.sum(axis=0)
    frac = finite.mean(axis=0) if X.shape[0] else np.zeros(X.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        total = np.nansum(X, axis=0)
    mean_intensity = np.where(counts > 0, total / np.maximum(counts, 1), 0.0)

    keep = (frac >= min_frac) & (mean_intensity >= min_intensity)
    fail_frac = (frac < min_frac)
    fail_intensity = (~fail_frac) & (mean_intensity < min_intensity)
    fail_rsd = np.zeros(X.shape[1], dtype=bool)
    rsd = np.full(X.shape[1], np.nan)
    if rsd_max is not None:
        qc_mask, qc_desc = _qc_mask(ctx, params.get("qc_label"))
        if int(qc_mask.sum()) < 2:
            raise prerequisite(
                "按 QC-RSD 过滤需要至少 2 个 QC 样本：请用 qc_label 指定样本名正则，"
                "或在样本表里加 qc 列（1=QC）")
        rsd, _mean = _feature_rsd(X[qc_mask])
        fail_rsd = keep & ~np.isfinite(rsd)
        fail_rsd |= keep & np.isfinite(rsd) & (rsd > float(rsd_max))
        keep = keep & np.isfinite(rsd) & (rsd <= float(rsd_max))
    else:
        qc_desc = ""
    warnings_list: list[str] = []

    n_keep = int(keep.sum())
    if n_keep < 10:
        raise prerequisite(
            f"按当前阈值只剩 {n_keep} 个特征（<10）：请放宽 min_sample_frac / min_intensity"
            + (" / 提高 rsd_max" if rsd_max is not None else ""))

    X2 = np.asarray(X[:, keep], dtype=np.float32)
    features2 = [features[index] for index in range(len(features)) if keep[index]]
    info2 = None
    try:
        info2 = finfo.loc[[f for f in features2 if f in set(finfo.index)]]
        info2 = info2.reindex(pd.Index(features2, name=getattr(finfo.index, "name", None) or "feature"))
    except Exception:  # noqa: BLE001
        info2 = None

    replace = {"X": X2, "features": features2, "feature_info": info2}
    raw = (ctx or {}).get("raw")
    if raw is not None:
        try:
            raw_arr = np.asarray(raw, dtype=np.float64)
            if raw_arr.shape == X.shape:
                replace["raw"] = np.asarray(raw_arr[:, keep], dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass

    rows = []
    for index in range(X.shape[1]):
        item = {"feature": features[index] if index < len(features) else str(index),
                "detected_frac": float(frac[index]),
                "mean_intensity": float(mean_intensity[index]),
                "kept": bool(keep[index])}
        if rsd_max is not None:
            item["rsd_qc"] = float(rsd[index]) if np.isfinite(rsd[index]) else None
        if not keep[index]:
            if fail_rsd[index]:
                item["reason"] = "QC RSD 超限"
            elif fail_frac[index]:
                item["reason"] = "检出比例不足"
            elif fail_intensity[index]:
                item["reason"] = "平均强度过低"
            else:
                item["reason"] = "其它"
        rows.append(item)
    columns = [{"key": "feature", "label": "特征", "type": "string"},
               {"key": "detected_frac", "label": "检出比例", "type": "number"},
               {"key": "mean_intensity", "label": "平均强度", "type": "number"}]
    if rsd_max is not None:
        columns.append({"key": "rsd_qc", "label": "QC RSD%", "type": "number"})
    columns += [{"key": "kept", "label": "保留", "type": "boolean"},
                {"key": "reason", "label": "剔除原因", "type": "string"}]

    fig, ax = new_fig(7.2, 4.0)
    labels_bar = ["retained", "low detection", "low intensity"]
    values_bar = [n_keep, int(fail_frac.sum()), int(fail_intensity.sum())]
    if rsd_max is not None:
        labels_bar.append("QC RSD over limit")
        values_bar.append(int(fail_rsd.sum()))
    colors = [PALETTE[4], PALETTE[1], PALETTE[3], PALETTE[2]][:len(labels_bar)]
    positions = np.arange(len(labels_bar))
    ax.bar(positions, values_bar, color=colors, width=0.66)
    for position, value in zip(positions, values_bar):
        ax.text(position, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_bar, fontsize=8)
    ax.set_ylabel("features", fontsize=9)
    ax.set_title(f"Feature filtering: {X.shape[1]} -> {n_keep}", fontsize=10)
    ax.margins(y=0.16)
    figure = save_fig(fig, outdir, "filter_features", title=None)

    bullets = [
        f"特征：{X.shape[1]} → {n_keep}（剔除 {X.shape[1] - n_keep}）",
        f"阈值：检出比例 ≥ {min_frac:.2f}、平均强度 ≥ {_fmt(min_intensity, 3)}"
        + (f"、QC RSD ≤ {_fmt(rsd_max, 3)}%" if rsd_max is not None else ""),
        f"剔除构成：检出比例不足 {int(fail_frac.sum())} 个、平均强度过低 {int(fail_intensity.sum())} 个"
        + (f"、QC RSD 超限 {int(fail_rsd.sum())} 个" if rsd_max is not None else ""),
    ]
    if rsd_max is not None:
        bullets.append(f"QC 样本：{int(_qc_mask(ctx, params.get('qc_label'))[0].sum())} 个"
                       f"（{qc_desc or 'qc 列'}）")
    if n_keep < 0.2 * X.shape[1]:
        warnings_list.append(f"过滤后只剩 {n_keep / X.shape[1] * 100:.0f}% 的特征，阈值可能过严")

    data_out = []
    try:
        path = export_csv(pd.DataFrame(rows), outdir, "feature-filter.csv", index=False)
        data_out.append(data_file("filter_csv", path, "特征过滤明细"))
    except Exception:  # noqa: BLE001
        pass

    return result(
        title="特征过滤",
        bullets=bullets,
        metrics={"n_features_before": int(X.shape[1]), "n_features_after": n_keep,
                 "removed_low_detection": int(fail_frac.sum()),
                 "removed_low_intensity": int(fail_intensity.sum()),
                 "removed_qc_rsd": int(fail_rsd.sum()),
                 "min_sample_frac": min_frac, "min_intensity": float(min_intensity),
                 "rsd_max": rsd_max},
        tables=[table("filter", "特征过滤明细", columns, rows)],
        figures=[figure],
        data_out=data_out,
        replace=replace,
        store={"filter": {"n_before": int(X.shape[1]), "n_after": n_keep,
                          "min_sample_frac": min_frac, "min_intensity": float(min_intensity),
                          "rsd_max": rsd_max}},
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# 步骤：impute（缺失值填补）
# ---------------------------------------------------------------------------


def _half_min_impute(X: np.ndarray) -> np.ndarray:
    """half-min：每特征用 min/2 填补（MetaboAnalyst 默认；Wei 2018 比较过其稳健性）。"""
    arr = np.array(X, dtype=np.float64)
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        minimum = np.nanmin(np.where(missing, np.nan, arr), axis=0)
    global_min = np.nanmin(np.where(missing, np.nan, arr)) if missing.any() else 0.0
    if not np.isfinite(global_min):
        global_min = 0.0
    fill = np.where(np.isfinite(minimum), minimum, global_min) / 2.0
    fill = np.where(np.isfinite(fill), fill, 0.0)
    arr[missing] = np.take(fill, np.where(missing)[1])
    return arr


def _median_impute(X: np.ndarray) -> np.ndarray:
    """特征中位数填补（对偏态数据比均值稳）。"""
    arr = np.array(X, dtype=np.float64)
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(np.where(missing, np.nan, arr), axis=0)
        global_median = np.nanmedian(np.where(missing, np.nan, arr))
    if not np.isfinite(global_median):
        global_median = 0.0
    fill = np.where(np.isfinite(median), median, global_median)
    arr[missing] = np.take(fill, np.where(missing)[1])
    return arr


def _knn_impute(X: np.ndarray, k: int = 5) -> np.ndarray:
    """KNN 填补（Troyanskaya 2001；sklearn KNNImputer，样本间距离按特征加权）。"""
    try:
        from sklearn.impute import KNNImputer
    except Exception as error:  # noqa: BLE001
        raise prerequisite(f"KNN 填补需要 scikit-learn（{error}）") from error
    arr = np.asarray(X, dtype=np.float64)
    n_samples = arr.shape[0]
    neighbors = int(max(1, min(int(k), max(1, n_samples - 1))))
    model = KNNImputer(n_neighbors=neighbors, weights="uniform")
    out = np.asarray(model.fit_transform(arr), dtype=np.float64)
    return out


def _ppca_impute(X: np.ndarray, n_comp: int = 3, max_iter: int = 40, tol: float = 1e-6) -> np.ndarray:
    """迭代 PCA（EM-PCA）填补近似：Josse & Husson 2012 的 regularized/iterative PCA 思想。"""
    arr = np.array(X, dtype=np.float64)
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        filled_mean = np.nanmean(np.where(missing, np.nan, arr), axis=0)
    filled_mean = np.where(np.isfinite(filled_mean), filled_mean, 0.0)
    filled = np.where(missing, filled_mean[None, :], arr)
    n_samples, n_features = filled.shape
    k = int(max(1, min(int(n_comp), n_samples - 1, n_features)))
    for _iteration in range(int(max_iter)):
        center = filled.mean(axis=0)
        centered = filled - center
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        recon = (u[:, :k] * s[:k]) @ vt[:k, :] + center
        updated = np.where(missing, recon, arr)
        change = float(np.max(np.abs(updated[missing] - filled[missing]))) if missing.any() else 0.0
        filled = updated
        if change < tol:
            break
    return filled


def step_impute(ctx, params, outdir):
    """缺失值填补：half-min / 中位数 / KNN / 迭代 PCA；填补后断言无 NaN（DESIGN §3）。"""
    params = dict(params or {})
    X = _X(ctx)
    method = str(params.get("method") or "half_min").strip().lower()
    k = max(_as_int(params, "k", 5), 1)

    before_missing = ~np.isfinite(X)
    n_missing = int(before_missing.sum())
    before_pct = float(100.0 * before_missing.mean()) if X.size else 0.0
    per_feature_before = _missing_pct_features(X)
    per_sample_before = 100.0 * (1.0 - np.isfinite(X).mean(axis=1))

    if method in ("half_min", "half-min", "halfmin", "min_half", "min"):
        X2 = _half_min_impute(X)
        method_label = "half-min（每特征 min/2）"
        note = "half-min 用每特征最小值的一半替代缺失：假设缺失来自低丰度（低于检测限），MetaboAnalyst 默认策略"
    elif method in ("median", "feature_median"):
        X2 = _median_impute(X)
        method_label = "特征中位数"
        note = "特征中位数填补：稳健但对高缺失特征会压缩方差"
    elif method in ("knn", "k-nn"):
        if X.shape[0] > 3000:
            raise prerequisite(
                f"KNN 填补需要计算样本间距离（当前 {X.shape[0]} 个样本）：请改用 half_min 或 median")
        X2 = _knn_impute(X, k)
        method_label = f"KNN（k={k}）"
        note = "KNN 填补利用样本间相似性（Troyanskaya 2001）；特征数极多时距离会被噪声稀释"
    elif method in ("ppca", "bpca", "iterative_pca", "pca"):
        X2 = _ppca_impute(X, n_comp=int(max(1, min(_as_int(params, "n_comp", 3), 10))))
        method_label = "迭代 PCA（EM-PCA 近似）"
        note = "迭代 PCA 填补：用低维主成分子空间重构缺失值（Josse & Husson 2012 思想）"
    else:
        raise bad_request(
            f"不支持的填补方法：{method}（可选 half_min / median / knn / ppca）")

    warnings_list = []
    if method in ("knn", "k-nn"):
        remaining = ~np.isfinite(X2)
        if remaining.any():
            X2 = _median_impute(X2)
            warnings_list.append(
                f"KNN 后仍有 {int(remaining.sum())} 个缺失（全缺失特征无法用邻居估计），已用中位数兜底")
    if not np.all(np.isfinite(X2)):
        raise EngineError("internal", "缺失值填补失败：矩阵里仍然存在 NaN/Inf")
    if n_missing == 0:
        method_label = method_label + "（无缺失，未修改）"

    figures = []
    try:
        fig, axes = new_fig(10.2, 4.0, ncols=2)
        axes = np.atleast_1d(axes).ravel()
        positions = np.arange(X.shape[1])
        axes[0].plot(positions, per_feature_before, color=PALETTE[0], linewidth=0.8)
        axes[0].set_xlabel("feature index", fontsize=9)
        axes[0].set_ylabel("missing (%)", fontsize=9)
        axes[0].set_title("Missing rate per feature (before)", fontsize=10)
        axes[1].bar(np.arange(X.shape[0]), per_sample_before, color=PALETTE[1], width=0.85)
        axes[1].set_xlabel("sample index", fontsize=9)
        axes[1].set_ylabel("missing (%)", fontsize=9)
        axes[1].set_title("Missing rate per sample (before)", fontsize=10)
        figures.append(save_fig(fig, outdir, "impute_missing", title=None))
    except Exception:  # noqa: BLE001 - 图失败不应阻塞填补
        pass

    after_pct = float(100.0 * (1.0 - np.isfinite(X2).mean())) if X2.size else 0.0
    bullets = [
        f"填补方法：{method_label}；缺失 {n_missing} 个（{before_pct:.2f}%）→ 0（{after_pct:.2f}%）",
        f"受影响样本 {int((per_sample_before > 0).sum())} 个，受影响特征 {int((per_feature_before > 0).sum())} 个",
        note,
    ]
    if warnings_list:
        bullets.append("注意：" + "；".join(warnings_list))

    rows = [{"stage": "before", "missing_cells": n_missing, "missing_pct": before_pct,
             "features_with_missing": int((per_feature_before > 0).sum()),
             "samples_with_missing": int((per_sample_before > 0).sum())},
            {"stage": "after", "missing_cells": int((~np.isfinite(X2)).sum()),
             "missing_pct": after_pct, "features_with_missing": 0, "samples_with_missing": 0}]

    return result(
        title="缺失值填补",
        bullets=bullets,
        metrics={"method": method, "n_missing_before": n_missing,
                 "missing_pct_before": before_pct, "missing_pct_after": after_pct},
        tables=[table("impute", "缺失率变化",
                      columns=[{"key": "stage", "label": "阶段", "type": "string"},
                               {"key": "missing_cells", "label": "缺失单元", "type": "number"},
                               {"key": "missing_pct", "label": "缺失率%", "type": "number"},
                               {"key": "features_with_missing", "label": "有缺失的特征", "type": "number"},
                               {"key": "samples_with_missing", "label": "有缺失的样本", "type": "number"}],
                      rows=rows,
                      note=note)],
        figures=figures,
        replace={"X": np.asarray(X2, dtype=np.float32)},
        store={"impute": {"method": method, "n_missing_before": n_missing,
                          "missing_pct_before": before_pct, "missing_pct_after": after_pct}},
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# 步骤：normalize（归一化）
# ---------------------------------------------------------------------------


def _pqn_factors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PQN 稀释因子：样本对参考谱（逐特征中位数）商的中位数（Dieterle 2006）。"""
    arr = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reference = np.nanmedian(np.where(finite, arr, np.nan), axis=0)
    reference = np.where(np.isfinite(reference), reference, np.nan)
    factors = np.ones(arr.shape[0], dtype=np.float64)
    valid_features = np.isfinite(reference) & (reference > 0)
    for index in range(arr.shape[0]):
        mask = valid_features & finite[index] & (arr[index] > 0)
        if not mask.any():
            continue
        quotients = arr[index, mask] / reference[mask]
        quotients = quotients[np.isfinite(quotients)]
        if quotients.size:
            factors[index] = float(np.median(quotients))
    factors[~np.isfinite(factors) | (factors <= 0)] = 1.0
    return factors, reference


def _quantile_normalize(X: np.ndarray) -> np.ndarray:
    """分位数标准化（Bolstad 2003）：每样本排序后取秩均值，再映射回原位置。"""
    arr = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        column_median = np.nanmedian(filled, axis=0)
    column_median = np.where(np.isfinite(column_median), column_median, 0.0)
    rows, cols = np.where(~finite)
    filled[rows, cols] = column_median[cols]
    order = np.argsort(filled, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_index = np.arange(filled.shape[0])[:, None]
    ranks[row_index, order] = np.arange(filled.shape[1])[None, :]
    reference = np.sort(filled, axis=1).mean(axis=0)
    out = np.empty_like(filled)
    out[row_index, ranks] = reference[None, :]
    out[~finite] = np.nan
    return out


def step_normalize(ctx, params, outdir):
    """归一化：PQN（Dieterle 2006）/ TIC / 中位数 / 分位数；输出稀释因子表与前后 TIC 图。"""
    params = dict(params or {})
    X = _X(ctx)
    method = str(params.get("method") or "pqn").strip().lower()
    tic_before, _detected, _missing = _sample_metrics(X)
    valid_tic = tic_before[tic_before > 0]
    median_tic = float(np.median(valid_tic)) if valid_tic.size else 1.0
    median_tic = median_tic if median_tic > 0 else 1.0

    factors = np.ones(X.shape[0], dtype=np.float64)
    warnings_list = []
    if method in ("pqn", "probabilistic"):
        factors, _reference = _pqn_factors(X)
        factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)
        X2 = X / factors[:, None]
        method_label = "PQN 概率商归一化（Dieterle 2006）"
        note = ("PQN：以逐特征中位数谱为参考，每样本计算比值的中位数作为稀释因子；"
                "对稀释/进样量差异稳健，受真实生物变化影响小")
    elif method in ("tic", "total", "sum", "ms_total"):
        row = np.where(tic_before > 0, tic_before, median_tic)
        factors = row / median_tic
        X2 = X / factors[:, None]
        method_label = "TIC 总离子流归一化"
        note = "TIC：每样本除以总离子流（等价于把各样本 TIC 校正到中位数）；简单但对高丰度特征敏感"
    elif method in ("median", "median_intensity"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_median = np.nanmedian(X, axis=1)
        row_median = np.where(np.isfinite(row_median) & (row_median > 0), row_median, 1.0)
        global_median = float(np.median(row_median))
        global_median = global_median if global_median > 0 else 1.0
        factors = row_median / global_median
        X2 = X / factors[:, None]
        method_label = "中位数归一化"
        note = "中位数：每样本除以自身特征中位数；比 TIC 稳健，但忽略丰度分布差异"
    elif method in ("quantile", "quantiles", "qn"):
        X2 = _quantile_normalize(X)
        factors = np.ones(X.shape[0], dtype=np.float64)
        method_label = "分位数标准化"
        note = "分位数标准化（Bolstad 2003）：所有样本的强度分布被映射到同一参考分布"
    elif method in ("none", "no"):
        X2 = np.array(X, dtype=np.float64)
        method_label = "跳过归一化"
        note = "按参数跳过归一化：后续多变量分析可能被进样量差异主导"
        warnings_list.append("未做归一化：PCA/PLS-DA 前建议至少做 TIC 或 PQN")
    else:
        raise bad_request(
            f"不支持的归一化方法：{method}（可选 pqn / tic / median / quantile / none）")

    X2 = np.where(np.isfinite(X2), X2, np.nan)
    tic_after, _d2, _m2 = _sample_metrics(X2)

    def _cv(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[values > 0]
        if values.size < 2:
            return 0.0
        return float(values.std(ddof=1) / values.mean() * 100.0)

    cv_before = _cv(tic_before)
    cv_after = _cv(tic_after)

    figures = []
    try:
        fig, axes = new_fig(10.4, 4.1, ncols=2)
        axes = np.atleast_1d(axes).ravel()
        for ax, values, title, color in (
            (axes[0], tic_before, "TIC before normalization", PALETTE[0]),
            (axes[1], tic_after, "TIC after normalization", PALETTE[4]),
        ):
            ax.hist(values, bins=30, color=color, alpha=0.8)
            ax.axvline(float(np.median(values)), color="#e15759", linestyle="--", linewidth=1.0,
                       label=f"median {_fmt(np.median(values), 3)}")
            ax.set_xlabel("TIC", fontsize=9)
            ax.set_ylabel("samples", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7)
        figures.append(save_fig(fig, outdir, "normalize_tic", title=None))
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for index, sample in enumerate(_samples(ctx)):
        rows.append({"sample": sample, "factor": float(factors[index]),
                     "tic_before": float(tic_before[index]),
                     "tic_after": float(tic_after[index]) if np.isfinite(tic_after[index]) else None})
    rows_sorted = sorted(rows, key=lambda item: -item["factor"])[:500]

    bullets = [
        f"归一化方法：{method_label}",
        f"TIC 变异（CV）：{cv_before:.1f}% → {cv_after:.1f}%",
        f"稀释因子范围 {_fmt(np.min(factors), 3)}–{_fmt(np.max(factors), 3)}"
        f"（中位 {_fmt(np.median(factors), 3)}）",
        note,
    ]
    if cv_after > cv_before * 0.9 and method not in ("none", "no"):
        bullets.append("归一化后 TIC 变异没有明显下降：可能本身稀释差异很小，或存在强生物学差异主导 TIC")
    if method in ("none", "no"):
        warnings_list.append("跳过归一化：TIC 变异未校正（CV %.1f%%）" % cv_before)

    return result(
        title="归一化",
        bullets=bullets,
        metrics={"method": method, "tic_cv_before": cv_before, "tic_cv_after": cv_after,
                 "factor_min": float(np.min(factors)), "factor_max": float(np.max(factors)),
                 "factor_median": float(np.median(factors))},
        tables=[table("normalize", "稀释因子（按大小排序）",
                      columns=[{"key": "sample", "label": "样本", "type": "string"},
                               {"key": "factor", "label": "稀释因子", "type": "number"},
                               {"key": "tic_before", "label": "TIC 前", "type": "number"},
                               {"key": "tic_after", "label": "TIC 后", "type": "number"}],
                      rows=rows_sorted,
                      note="PQN/TIC/中位数方法下因子 = 每样本的缩放系数（X 除以该值）")],
        figures=figures,
        replace={"X": np.asarray(X2, dtype=np.float32)},
        store={"normalize": {"method": method, "factors": [float(v) for v in factors],
                             "tic_cv_before": cv_before, "tic_cv_after": cv_after}},
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# 步骤：transform（变换 / 标度）
# ---------------------------------------------------------------------------


#: P0-1 三层（linear / log / model）默认内存预算：float32 × n × p × 3 层。
#: 超预算时不伪造，按 model → linear → log 的优先级尽量保存，budget.truncated=True。
LAYER_BUDGET_BYTES = 512 * 1024 * 1024

_LOG_METHODS = ("log2", "log10", "ln")
_LOG_BASE = {"log2": 2.0, "log10": 10.0, "ln": float(math.e)}


def _transform_values(X: np.ndarray, transform: str) -> tuple[np.ndarray, str, dict]:
    """log2 / log10 / ln / cuberoot；对 ≤0 的值加半个最小正值（保证有定义）。

    返回 (变换后矩阵, 人类可读标签, 规格 dict)。规格 dict 是 P0-1 三层分离的元数据：
    ``method``（none/log2/log10/ln/cuberoot）、``offset``、``base``、``label``。
    """
    arr = np.asarray(X, dtype=np.float64)
    lowered = str(transform or "none").strip().lower()
    if lowered in ("none", "no", ""):
        return arr.copy(), "none", {"method": "none", "offset": 0.0, "base": None, "label": "none"}
    positive = arr[np.isfinite(arr) & (arr > 0)]
    offset = 0.0
    finite = arr[np.isfinite(arr)]
    if positive.size and finite.size and float(np.nanmin(finite)) <= 0:
        offset = float(positive.min()) / 2.0
    shifted = arr + offset
    with np.errstate(invalid="ignore", divide="ignore"):
        if lowered == "log2":
            out = np.log2(np.where(shifted > 0, shifted, np.nan))
            method = "log2"
        elif lowered == "log10":
            out = np.log10(np.where(shifted > 0, shifted, np.nan))
            method = "log10"
        elif lowered in ("cuberoot", "cbrt", "cube_root"):
            out = np.cbrt(arr)
            method = "cuberoot"
        elif lowered in ("log", "ln", "loge"):
            out = np.log(np.where(shifted > 0, shifted, np.nan))
            method = "ln"
        else:
            raise bad_request(f"不支持的变换：{transform}（可选 log2 / log10 / cuberoot / none）")
    label = transform if offset == 0 else f"{transform}(x+{_fmt(offset, 3)})"
    meta = {"method": method, "offset": float(offset), "base": _LOG_BASE.get(method), "label": label}
    return out, label, meta


def _scale_values(X: np.ndarray, scale: str) -> tuple[np.ndarray, str, dict]:
    """auto(UV) / pareto / range / none 标度（van den Berg 2006）。

    返回 (标度后矩阵, 标签, 规格 dict)。规格里的 ``center`` / ``denom`` 逐特征保存
    "x → (x − center) / denom" 的实际参数（长度 = 特征数），供 P0-1 三层审计与反演。
    """
    arr = np.asarray(X, dtype=np.float64)
    lowered = str(scale or "none").strip().lower()
    n_features = int(arr.shape[1]) if arr.ndim == 2 else 0
    if lowered in ("none", "no", ""):
        return (arr.copy(), "none",
                {"method": "none", "chosen": "none", "label": "none",
                 "center": np.zeros(n_features, dtype=np.float64),
                 "denom": np.ones(n_features, dtype=np.float64)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
    center = np.where(np.isfinite(center), center, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
    if lowered in ("auto", "uv", "unit", "unit_variance", "autoscaling"):
        denom = np.where(np.isfinite(sd), sd, 1.0)
        label = "auto(UV, mean-center + /SD)"
        method = "uv"
    elif lowered in ("pareto", "par"):
        denom = np.where(np.isfinite(sd), np.sqrt(sd), 1.0)
        label = "pareto (mean-center + /sqrt(SD))"
        method = "pareto"
    elif lowered in ("range", "rangescaling"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lo = np.nanmin(arr, axis=0)
            hi = np.nanmax(arr, axis=0)
        span = hi - lo
        denom = np.where(np.isfinite(span) & (span > 0), span, 1.0)
        label = "range (mean-center + /(max-min))"
        method = "range"
    else:
        raise bad_request(f"不支持的标度：{scale}（可选 auto / pareto / range / none）")
    meta = {"method": method, "chosen": lowered, "label": label,
            "center": center, "denom": np.asarray(denom, dtype=np.float64)}
    return (arr - center) / denom, label, meta


def _build_transform_layers(linear: np.ndarray, transformed: np.ndarray, model: np.ndarray,
                            transform_meta: dict, scale_meta: dict, value_kind: str,
                            ctx, params: dict) -> tuple[dict, list[str]]:
    """P0-1：把 linear（归一化后原始强度）/ log（未标度变换层）/ model（建模标度层）三层
    连同 features/samples 快照、变换/标度参数与数组预算写进 store["transform"]["layers"]。

    预算：三层 float32 超过 ``LAYER_BUDGET_BYTES``（可用参数 layer_budget_bytes 覆盖）时，
    按 model → linear → log 的优先级尽量保存数组，未保存的层为 None，并置 truncated=True
    与 warning；绝不把不同层静默混用。
    """
    warnings_out: list[str] = []
    n, p = np.asarray(model).shape
    raw_limit = params.get("layer_budget_bytes") if isinstance(params, dict) else None
    try:
        limit = int(raw_limit) if raw_limit is not None else int(LAYER_BUDGET_BYTES)
    except (TypeError, ValueError):
        limit = int(LAYER_BUDGET_BYTES)
    limit = max(1024, limit)
    per_layer = max(1, limit // 3)
    bytes_one = int(n) * int(p) * 4  # float32

    arrays = {"linear": np.asarray(linear, dtype=np.float64),
              "log": np.asarray(transformed, dtype=np.float64),
              "model": np.asarray(model, dtype=np.float64)}
    values: dict[str, np.ndarray | None] = {"linear": None, "log": None, "model": None}
    stored: list[str] = []
    if bytes_one <= per_layer:
        for key in ("linear", "log", "model"):
            values[key] = np.asarray(arrays[key], dtype=np.float32)
            stored.append(key)
        truncated = False
    else:
        budget = limit
        for key in ("model", "linear", "log"):  # model 优先：后续分析/反演都先看它
            if bytes_one <= budget:
                values[key] = np.asarray(arrays[key], dtype=np.float32)
                stored.append(key)
                budget -= bytes_one
        truncated = True
        warnings_out.append(
            f"矩阵 {n}×{p}（每层约 {bytes_one / 1048576:.1f} MB）超出三层数组预算 "
            f"{limit / 1048576:.0f} MB，只保存了 {('/'.join(stored)) or '无'} 层；"
            f"其余层为 None（不是空值伪造），需要现算请重跑「变换/标度」")

    center = np.asarray(scale_meta.get("center", np.zeros(p)), dtype=np.float64)
    denom = np.asarray(scale_meta.get("denom", np.ones(p)), dtype=np.float64)
    layers = {
        "schema": "mb-transform-layers/1",
        "features": [str(f) for f in (ctx.get("features") or [])],
        "samples": [str(s) for s in (ctx.get("samples") or [])],
        "value_kind": str(value_kind or "raw"),
        "transform": dict(transform_meta),
        "scale": {k: v for k, v in dict(scale_meta).items() if k not in ("center", "denom")},
        "center": np.asarray(center, dtype=np.float32),
        "denom": np.asarray(denom, dtype=np.float32),
        "values": values,
        "stored": stored,
        "budget": {"limit_bytes": int(limit), "per_layer_bytes": int(per_layer),
                   "truncated": bool(truncated)},
    }
    return layers, warnings_out


def step_transform(ctx, params, outdir):
    """变换 + 标度：log2/log10/cuberoot + auto/pareto/range（van den Berg 2006）。"""
    params = dict(params or {})
    X = _X(ctx)
    transform = str(params.get("transform") or "log2").strip().lower()
    scale = str(params.get("scale") or "auto").strip().lower()
    current_kind = str((ctx or {}).get("X_kind") or "raw")

    skew_before, kurt_before = _skew_kurt(X)
    Xt, transform_label, transform_meta = _transform_values(X, transform)
    skew_mid, kurt_mid = _skew_kurt(Xt)
    finite_skew = skew_mid[np.isfinite(skew_mid)]
    median_abs_skew = float(np.median(np.abs(finite_skew))) if finite_skew.size else 0.0

    chosen_scale = scale
    auto_note = ""
    if scale == "auto":
        chosen_scale = "pareto" if median_abs_skew > 1.0 else "auto"
        auto_note = (f"auto 标度：变换后 |偏度| 中位 {median_abs_skew:.2f} "
                     f"{'> 1 → pareto' if median_abs_skew > 1.0 else '≤ 1 → UV'}")
    Xs, scale_label, scale_meta = _scale_values(Xt, chosen_scale)
    skew_after, kurt_after = _skew_kurt(Xs)

    if not np.all(np.isfinite(Xs)):
        remaining = int((~np.isfinite(Xs)).sum())
        raise prerequisite(
            f"变换/标度后有 {remaining} 个非有限值：请先跑「缺失值填补」，或把强度矩阵里的 0/负值处理掉")

    rows = [
        {"stage": "before (current)", "median_abs_skew": _median_abs(skew_before),
         "median_kurtosis": _median_kurt(kurt_before)},
        {"stage": f"after transform ({transform_label})", "median_abs_skew": _median_abs(skew_mid),
         "median_kurtosis": _median_kurt(kurt_mid)},
        {"stage": f"after scale ({scale_label})", "median_abs_skew": _median_abs(skew_after),
         "median_kurtosis": _median_kurt(kurt_after)},
    ]

    figures = []
    try:
        fig, axes = new_fig(10.4, 4.0, ncols=2)
        axes = np.atleast_1d(axes).ravel()
        axes[0].hist(np.clip(skew_before[np.isfinite(skew_before)], -10, 10), bins=48,
                     color=PALETTE[2], alpha=0.75)
        axes[0].set_xlabel("skewness (before)", fontsize=9)
        axes[0].set_ylabel("features", fontsize=9)
        axes[0].set_title("Feature skewness before transform", fontsize=10)
        axes[1].hist(np.clip(skew_after[np.isfinite(skew_after)], -10, 10), bins=48,
                     color=PALETTE[4], alpha=0.75)
        axes[1].set_xlabel("skewness (after transform + scale)", fontsize=9)
        axes[1].set_ylabel("features", fontsize=9)
        axes[1].set_title("Feature skewness after", fontsize=10)
        figures.append(save_fig(fig, outdir, "transform_skew", title=None))
    except Exception:  # noqa: BLE001
        pass

    # X_kind 三值语义：有标度→scaled；仅 log→lognorm；仅 cuberoot→transformed
    _t = str(transform or "").strip().lower()
    _did_transform = _t not in ("none", "no", "")
    _did_scale = str(scale_label or "none").strip().lower() != "none"
    applied = _did_transform or _did_scale
    if _did_scale:
        kind = "scaled"
    elif _t in ("log2", "log10", "ln", "log", "log1p"):
        kind = "lognorm"
    elif _t in ("cuberoot", "cbrt"):
        kind = "transformed"
    else:
        kind = current_kind
    replace = {"X": np.asarray(Xs, dtype=np.float32)}
    if applied:
        replace["X_kind"] = kind
    warnings_list = []
    if not applied:
        warnings_list.append("变换与标度都设为 none：X_kind 保持原值")
    elif rows[2]["median_abs_skew"] > rows[0]["median_abs_skew"] * 1.15:
        warnings_list.append(
            f"变换后 |偏度| 中位反而上升（{rows[0]['median_abs_skew']:.2f} → "
            f"{rows[2]['median_abs_skew']:.2f}）：常见于 half-min 填补出的极低值；"
            "可改用 median/PPCA 填补，或检查是否存在异常低强度")

    bullets = [
        f"变换：{transform_label} → 标度：{scale_label}"
        + (f"；{auto_note}" if auto_note else ""),
        f"特征 |偏度| 中位：{rows[0]['median_abs_skew']:.2f} → {rows[2]['median_abs_skew']:.2f}"
        f"（超峰度中位 {rows[2]['median_kurtosis']:.2f}）",
        {"scaled": "输出 X_kind = scaled：后续 PCA/PLS-DA 在标度后的矩阵上计算",
         "lognorm": "输出 X_kind = lognorm：只做了对数变换、未标度；PCA 等多元分析前建议再跑一次标度",
         "transformed": "输出 X_kind = transformed：只做了 cuberoot、未标度；PCA 等多元分析前建议再跑一次标度",
         }.get(str(kind), f"未做变换/标度：X_kind 保持 {current_kind}（下一步 PCA 会拒绝 raw 矩阵）"),
    ]
    if transform in ("none", "no", "") and scale == "none":
        bullets.append("注意：未做任何变换/标度，PCA 可能被少数高丰度特征主导")

    # P0-1：三层分离 —— linear（归一化后原始强度）/ log（未标度变换层）/ model（建模标度层）
    # 三层 + 快照 + 变换/标度参数 + 数组预算一起进 store["transform"]["layers"]，供 mb_de 等
    # 下游按"效应量用未标度层、统计量用当前矩阵"的口径取数（BIO_P0_BACKLOG §4 P0-1）。
    layer_warnings: list[str] = []
    try:
        layers, layer_warnings = _build_transform_layers(
            np.asarray(X, dtype=np.float64), np.asarray(Xt, dtype=np.float64),
            np.asarray(Xs, dtype=np.float64), transform_meta, scale_meta, kind, ctx or {}, params)
    except Exception as error:  # noqa: BLE001 - 层快照失败不阻塞主步骤，但必须明说
        layers = None
        layer_warnings = [f"三层快照保存失败（{type(error).__name__}: {error}），下游将回退当前矩阵口径"]
    warnings_list.extend(layer_warnings)

    store_transform = {
        "transform": transform_label,
        "scale": scale_label,
        "median_abs_skew_before": rows[0]["median_abs_skew"],
        "median_abs_skew_after": rows[2]["median_abs_skew"],
        "layers": layers,
    }
    return result(
        title="变换/标度",
        bullets=bullets,
        metrics={"transform": transform_label, "scale": scale_label,
                 "median_abs_skew_before": rows[0]["median_abs_skew"],
                 "median_abs_skew_after": rows[2]["median_abs_skew"],
                 "X_kind": kind,
                 "layers_stored": list((layers or {}).get("stored") or []),
                 "layers_truncated": bool(((layers or {}).get("budget") or {}).get("truncated"))},
        tables=[table("transform", "偏度 / 峰度前后对比",
                      columns=[{"key": "stage", "label": "阶段", "type": "string"},
                               {"key": "median_abs_skew", "label": "|偏度| 中位", "type": "number"},
                               {"key": "median_kurtosis", "label": "超峰度 中位", "type": "number"}],
                      rows=rows,
                      note="偏度/超峰度逐特征计算（Fisher）；接近 0 表示分布更对称、更接近高斯")],
        figures=figures,
        replace=replace,
        store={"transform": store_transform},
        warnings=warnings_list,
    )


def _median_abs(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(np.abs(arr))) if arr.size else 0.0


def _median_kurt(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else 0.0


# ---------------------------------------------------------------------------
# 步骤：pca（主成分分析）
# ---------------------------------------------------------------------------


def step_pca(ctx, params, outdir):
    """PCA：SVD 得分/载荷/解释方差 + Hotelling T² 与 DModX 离群诊断（Wold 1994）。"""
    params = dict(params or {})
    X = _X(ctx)
    if str((ctx or {}).get("X_kind") or "raw").lower() == "raw":
        raise prerequisite("请先完成「变换/标度」步骤：raw 强度直接做 PCA 会被高丰度特征主导")
    n_comp = _as_int(params, "n_comp", 3)
    color_by = str(params.get("color_by") or "group")
    qc_mask, _qc_desc = _qc_mask(ctx, params.get("qc_label"))

    core = _pca_core(X, n_comp=n_comp)
    scores = np.asarray(core["scores"], dtype=np.float64)
    ratio = np.asarray(core["ratio"], dtype=np.float64)
    loadings = np.asarray(core["loadings"], dtype=np.float64)
    labels, label_title = _color_labels(ctx, params, color_by, qc_mask)
    warnings_list = []
    if labels is None and str(color_by).lower() in ("group", "groups", "group_key"):
        warnings_list.append("没有分组列：PCA 按单色显示；请先在样本表里设置分组")
    if core["subsampled"]:
        warnings_list.append(
            f"特征数 {core['n_features_total']} 超过 {MAX_MULTIVARIATE_FEATURES}："
            f"PCA 只用了方差最大的 {len(core['columns'])} 个特征")

    t2, t2_crit95, t2_crit99 = _hotelling_t2(scores, alpha=0.95)
    dmodx, dmodx_crit, _s0 = _dmodx(core["sse"], n_features=len(core["columns"]),
                                    n_comp=scores.shape[1], alpha=0.95)
    over_t2 = np.where(t2 > t2_crit95)[0]
    over_dmodx = np.where(dmodx > dmodx_crit)[0] if np.isfinite(dmodx_crit) else np.array([], dtype=int)
    both = np.intersect1d(over_t2, over_dmodx)

    figures = [
        _fig_pca_scores(scores, ratio, labels, outdir, "pca_scores",
                        f"PCA scores (color = {label_title})"),
        _fig_pca_variance(ratio, outdir, "pca_variance", "PCA scree / cumulative variance"),
        _fig_outliers(t2, t2_crit95, t2_crit99, dmodx, dmodx_crit, labels, outdir,
                      "hotelling_t2", "Sample outliers: Hotelling T2 and DModX"),
    ]

    n_show = int(min(scores.shape[1], 5))
    score_rows = []
    samples = _samples(ctx)
    for index, sample in enumerate(samples):
        row = {"sample": sample}
        if labels is not None:
            row["group"] = str(labels[index])
        for comp in range(n_show):
            row[f"PC{comp + 1}"] = float(scores[index, comp])
        row["t2"] = float(t2[index])
        row["dmodx"] = float(dmodx[index])
        row["t2_flag"] = "over 95%" if t2[index] > t2_crit95 else ""
        row["dmodx_flag"] = "over limit" if (np.isfinite(dmodx_crit) and dmodx[index] > dmodx_crit) else ""
        score_rows.append(row)
    score_columns = [{"key": "sample", "label": "样本", "type": "string"}]
    if labels is not None:
        score_columns.append({"key": "group", "label": f"分组（{label_title}）", "type": "string"})
    score_columns += [{"key": f"PC{comp + 1}", "label": f"PC{comp + 1}", "type": "number"}
                      for comp in range(n_show)]
    score_columns += [{"key": "t2", "label": "Hotelling T²", "type": "number"},
                      {"key": "dmodx", "label": "DModX", "type": "number"},
                      {"key": "t2_flag", "label": "T²", "type": "string"},
                      {"key": "dmodx_flag", "label": "DModX", "type": "string"}]

    features_here = _features(ctx)
    order = np.argsort(-np.abs(loadings[0]))[:200]
    loading_rows = []
    for column in order:
        column = int(column)
        try:
            feature = features_here[int(core["columns"][column])]
        except Exception:  # noqa: BLE001
            feature = f"F{column}"
        row = {"feature": feature}
        for comp in range(loadings.shape[0]):
            row[f"PC{comp + 1}"] = float(loadings[comp, column])
        loading_rows.append(row)
    loading_columns = [{"key": "feature", "label": "特征", "type": "string"}]
    loading_columns += [{"key": f"PC{comp + 1}", "label": f"PC{comp + 1} 载荷", "type": "number"}
                        for comp in range(loadings.shape[0])]

    outlier_rows = []
    for index in np.argsort(-t2)[:50]:
        index = int(index)
        if t2[index] <= t2_crit95 and not (np.isfinite(dmodx_crit) and dmodx[index] > dmodx_crit):
            continue
        outlier_rows.append({"sample": samples[index] if index < len(samples) else str(index),
                             "group": str(labels[index]) if labels is not None else "",
                             "t2": float(t2[index]), "t2_limit95": float(t2_crit95),
                             "dmodx": float(dmodx[index]),
                             "dmodx_limit": float(dmodx_crit) if np.isfinite(dmodx_crit) else None,
                             "both": bool(index in set(both.tolist()))})

    bullets = [
        f"PCA：{scores.shape[1]} 个主成分；PC1 解释 {ratio[0] * 100:.1f}%，"
        f"PC1+PC2 累计 {float(np.sum(ratio[:2])) * 100:.1f}%"
        + (f"，前 5 个累计 {float(np.sum(ratio[:5])) * 100:.1f}%" if ratio.size > 2 else ""),
        f"离群诊断：T² 超 95% 限 {over_t2.size} 个样本，DModX 超限 {over_dmodx.size} 个"
        + (f"，两者同时超限 {both.size} 个" if both.size else ""),
    ]
    if over_t2.size or over_dmodx.size:
        names = "、".join(samples[index] for index in np.union1d(over_t2, over_dmodx)[:8])
        bullets.append(f"可疑样本：{names}" + ("…" if np.union1d(over_t2, over_dmodx).size > 8 else ""))
    else:
        bullets.append("没有样本超过 T²/DModX 控制限：样本层面一致性良好")
    if over_dmodx.size > 0.10 * max(t2.size, 1):
        bullets.append(
            f"DModX 超限比例偏高（{over_dmodx.size}/{t2.size}）：提示残差方差在样本间不均一"
            "（如 QC 与生物样本噪声不同），DModX 限只作相对参考；重点看 T² 与两者同时超限的样本")
    if labels is not None:
        bullets.append(f"得分图按「{label_title}」着色，虚线椭圆为各组 95% Hotelling 置信域")

    data_out = []
    try:
        frame = pd.DataFrame(score_rows).set_index("sample")
        path = export_csv(frame, outdir, "pca-scores.csv")
        data_out.append(data_file("pca_csv", path, "PCA 得分 + T²/DModX"))
    except Exception:  # noqa: BLE001
        pass
    try:
        frame = pd.DataFrame(loading_rows).set_index("feature")
        path = export_csv(frame, outdir, "pca-loadings.csv")
        data_out.append(data_file("pca_csv", path, "PCA 载荷（PC1 Top200）"))
    except Exception:  # noqa: BLE001
        pass

    sample_columns = [{"key": f"PC{comp + 1}", "kind": "numeric"} for comp in range(n_show)]
    sample_columns += [{"key": "pc1", "kind": "numeric"}, {"key": "pc2", "kind": "numeric"},
                       {"key": "t2", "kind": "numeric"}, {"key": "dmodx", "kind": "numeric"}]
    sample_values = {f"PC{comp + 1}": scores[:, comp].tolist() for comp in range(n_show)}
    sample_values.update({"pc1": scores[:, 0].tolist(),
                          "pc2": scores[:, 1].tolist() if scores.shape[1] > 1 else scores[:, 0].tolist(),
                          "t2": t2.tolist(), "dmodx": dmodx.tolist()})

    return result(
        title="主成分分析",
        bullets=bullets,
        metrics={"n_comp": int(scores.shape[1]), "pc1_variance": float(ratio[0]),
                 "pc12_variance": float(np.sum(ratio[:2])),
                 "n_t2_outlier": int(over_t2.size), "n_dmodx_outlier": int(over_dmodx.size),
                 "t2_limit95": float(t2_crit95), "dmodx_limit": float(dmodx_crit),
                 "n_features_used": int(len(core["columns"])),
                 "feature_subsampled": bool(core["subsampled"])},
        tables=[
            table("pca_scores", "样本得分与离群诊断", score_columns, score_rows),
            table("pca_loadings", "载荷（PC1 Top 200）", loading_columns, loading_rows),
            table("pca_outliers", "疑似离群样本", [
                {"key": "sample", "label": "样本", "type": "string"},
                {"key": "group", "label": "分组", "type": "string"},
                {"key": "t2", "label": "T²", "type": "number"},
                {"key": "t2_limit95", "label": "T² 95% 限", "type": "number"},
                {"key": "dmodx", "label": "DModX", "type": "number"},
                {"key": "dmodx_limit", "label": "DModX 限", "type": "number"},
                {"key": "both", "label": "两者同时超限", "type": "boolean"},
            ], outlier_rows),
        ],
        figures=figures,
        data_out=data_out,
        store={"pca": {"scores": scores, "ratio": ratio, "loadings": loadings,
                       "t2": t2, "t2_crit95": t2_crit95, "t2_crit99": t2_crit99,
                       "dmodx": dmodx, "dmodx_crit": dmodx_crit,
                       "columns": core["columns"], "subsampled": bool(core["subsampled"])}},
        sample_columns=sample_columns,
        sample_values=sample_values,
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# 分析：qc_summary（质控汇总）
# ---------------------------------------------------------------------------


def _qc_block(ctx, params):
    """qc 分析共用的指标块（TIC/检出/缺失/RSD）：在原始强度（raw）上算。"""
    X, qc_source = _qc_matrix(ctx)
    tic, detected, missing_pct = _sample_metrics(X)
    qc_mask, qc_desc = _qc_mask(ctx, params.get("qc_label"))
    feature_missing = _missing_pct_features(X)
    rsd_qc = rsd_bio = None
    if int(qc_mask.sum()) >= 2:
        rsd_qc, _m = _feature_rsd(X[qc_mask])
    if int((~qc_mask).sum()) >= 2:
        rsd_bio, _m = _feature_rsd(X[~qc_mask])
    return {
        "X": X, "source": qc_source, "tic": tic, "detected": detected,
        "missing_pct": missing_pct, "qc_mask": qc_mask, "qc_desc": qc_desc,
        "feature_missing": feature_missing, "rsd_qc": rsd_qc, "rsd_bio": rsd_bio,
    }


def analysis_qc_summary(ctx, params, outdir):
    """质控汇总：按样本角色的 TIC/检出/缺失/RSD 汇总表 + TIC/RSD 图（Dunn 2011；Want 2010）。"""
    params = dict(params or {})
    block = _qc_block(ctx, params)
    X = block["X"]
    samples = _samples(ctx)
    groups = [("all", np.ones(X.shape[0], dtype=bool)),
              ("QC", block["qc_mask"]),
              ("biological", ~block["qc_mask"])]
    rows = []
    for name, mask in groups:
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"role": name, "n_samples": n,
               "median_tic": float(np.median(block["tic"][mask])),
               "median_detected": float(np.median(block["detected"][mask])),
               "median_missing_pct": float(np.median(block["missing_pct"][mask]))}
        rsd = None
        if n >= 3:
            rsd, _mean = _feature_rsd(X[mask])
        if rsd is not None:
            finite = rsd[np.isfinite(rsd)]
            row["median_rsd_pct"] = float(np.median(finite)) if finite.size else None
        rows.append(row)

    figures = [
        _fig_tic(samples, block["tic"], None, block["qc_mask"], outdir, "qc_summary_tic",
                 "TIC per sample (outlined = QC)"),
        _fig_hist([(f"QC RSD ({int(block['qc_mask'].sum())})",
                    block["rsd_qc"] if block["rsd_qc"] is not None else np.array([])),
                   ("biological RSD",
                    block["rsd_bio"] if block["rsd_bio"] is not None else np.array([]))],
                  outdir, "qc_summary_rsd", "Feature RSD: QC vs biological samples", "RSD (%)",
                  bins=48, vlines=[(DEFAULT_RSD_CUT, f"{DEFAULT_RSD_CUT:.0f}% threshold")]),
    ]

    source_text = "原始强度（raw）" if block.get("source") == "raw" else "当前矩阵"
    bullets = [
        f"{X.shape[0]} 个样本 × {X.shape[1]} 个特征；"
        f"TIC 中位 {_fmt(np.median(block['tic']), 4)}，缺失率中位 {float(np.median(block['missing_pct'])):.1f}%",
        f"统计基于{source_text}：TIC/RSD 不受后续归一化与变换标度影响",
    ]
    if int(block["qc_mask"].sum()) >= 2:
        rsd = block["rsd_qc"]
        finite = rsd[np.isfinite(rsd)]
        bullets.append(f"QC 样本 {int(block['qc_mask'].sum())} 个：特征 RSD 中位 "
                       f"{float(np.median(finite)):.1f}%，"
                       f"{float((finite <= DEFAULT_RSD_CUT).mean() * 100):.0f}% 的特征 RSD ≤ 30%")
        if block["rsd_bio"] is not None:
            bio = block["rsd_bio"][np.isfinite(block["rsd_bio"])]
            if bio.size:
                bullets.append(f"生物样本特征 RSD 中位 {float(np.median(bio)):.1f}%"
                               "（通常高于 QC：QC 变异应主要来自分析过程）")
    else:
        bullets.append("没有 QC 样本：无法把分析变异与生物学变异分开，建议在样本表里标注 QC 池样本")
    bullets.append("判读标准：QC 特征 RSD ≤ 30% 视为方法稳定（Want 2010；Dunn 2011）")

    return result(
        title="质控汇总",
        bullets=bullets,
        metrics={"n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
                 "n_qc": int(block["qc_mask"].sum()),
                 "median_tic": float(np.median(block["tic"])),
                 "median_missing_pct": float(np.median(block["missing_pct"]))},
        tables=[table("qc_summary", "按样本角色的质控汇总", [
            {"key": "role", "label": "样本角色", "type": "string"},
            {"key": "n_samples", "label": "样本数", "type": "number"},
            {"key": "median_tic", "label": "TIC 中位", "type": "number"},
            {"key": "median_detected", "label": "检出特征中位", "type": "number"},
            {"key": "median_missing_pct", "label": "缺失率中位%", "type": "number"},
            {"key": "median_rsd_pct", "label": "特征 RSD 中位%", "type": "number"},
        ], rows, note="内置实现（依据 Want 2010 / Dunn 2011）")],
        figures=figures,
    )


# ---------------------------------------------------------------------------
# 分析：rsd_qc（QC 样本 RSD）
# ---------------------------------------------------------------------------


def analysis_rsd_qc(ctx, params, outdir):
    """QC 样本 RSD 与生物样本对比（Want 2010：QC 特征 RSD ≤ 30% 为方法可接受）。"""
    params = dict(params or {})
    threshold = _as_float(params, "threshold", 30.0) or 30.0
    block = _qc_block(ctx, params)
    X = block["X"]
    if int(block["qc_mask"].sum()) < 2:
        raise prerequisite(
            "没有识别到 QC 样本：请用 qc_label 指定样本名正则（如 QC），"
            "或在样本表里加一列 qc（1=QC 样本）")
    rsd_qc = block["rsd_qc"]
    rsd_bio = block["rsd_bio"]
    finite_qc = rsd_qc[np.isfinite(rsd_qc)]
    n_pass = int((finite_qc <= threshold).sum())
    pct_pass = float(n_pass / finite_qc.size * 100.0) if finite_qc.size else 0.0

    features = _features(ctx)
    order = np.argsort(-np.nan_to_num(rsd_qc, nan=-np.inf), kind="stable")[:200]
    rows = []
    for index in order:
        index = int(index)
        row = {"feature": features[index] if index < len(features) else str(index),
               "rsd_qc": float(rsd_qc[index]) if np.isfinite(rsd_qc[index]) else None,
               "rsd_bio": float(rsd_bio[index]) if rsd_bio is not None and np.isfinite(rsd_bio[index]) else None,
               "pass": bool(np.isfinite(rsd_qc[index]) and rsd_qc[index] <= threshold)}
        if row["rsd_qc"] is not None and row["rsd_bio"] not in (None, 0):
            row["qc_over_bio"] = float(row["rsd_qc"] / row["rsd_bio"])
        rows.append(row)

    figures = []
    try:
        fig, axes = new_fig(10.4, 4.1, ncols=2)
        axes = np.atleast_1d(axes).ravel()
        values_qc = rsd_qc[np.isfinite(rsd_qc)]
        values_qc = values_qc[values_qc <= np.quantile(values_qc, 0.99)] if values_qc.size else values_qc
        axes[0].hist(values_qc, bins=48, color=PALETTE[2], alpha=0.7,
                     label=f"QC (n={int(block['qc_mask'].sum())})")
        if rsd_bio is not None:
            values_bio = rsd_bio[np.isfinite(rsd_bio)]
            if values_bio.size:
                values_bio = values_bio[values_bio <= np.quantile(values_bio, 0.99)]
                axes[0].hist(values_bio, bins=48, color=PALETTE[0], alpha=0.45, label="biological")
        axes[0].axvline(threshold, color="#e15759", linestyle="--", linewidth=1.0,
                        label=f"{threshold:.0f}% threshold")
        axes[0].set_xlabel("RSD (%)", fontsize=9)
        axes[0].set_ylabel("features", fontsize=9)
        axes[0].set_title("Feature RSD: QC vs biological", fontsize=10)
        axes[0].legend(fontsize=7)
        if rsd_bio is not None:
            mask = np.isfinite(rsd_qc) & np.isfinite(rsd_bio) & (rsd_bio > 0) & (rsd_qc > 0)
            if mask.any():
                axes[1].scatter(rsd_bio[mask], rsd_qc[mask], s=9, color=PALETTE[4],
                                alpha=0.6, linewidths=0)
                limit = float(max(np.quantile(rsd_bio[mask], 0.98), np.quantile(rsd_qc[mask], 0.98)))
                axes[1].plot([0, limit], [0, limit], color="#8a919c", linestyle=":", linewidth=1.0,
                             label="y = x")
                axes[1].set_xlabel("biological RSD (%)", fontsize=9)
                axes[1].set_ylabel("QC RSD (%)", fontsize=9)
                axes[1].set_title("QC vs biological RSD per feature", fontsize=10)
                axes[1].legend(fontsize=7)
        figures.append(save_fig(fig, outdir, "rsd_qc", title=None))
    except Exception:  # noqa: BLE001
        pass

    bullets = [
        f"QC 样本 {int(block['qc_mask'].sum())} 个（{block['qc_desc'] or 'qc 列'}）："
        f"特征 RSD 中位 {float(np.median(finite_qc)):.1f}%，{pct_pass:.0f}% 的特征 RSD ≤ {threshold:.0f}%",
        f"RSD 基于{'原始强度（raw）' if block.get('source') == 'raw' else '当前矩阵'}计算；"
        "峰面积的 RSD 才可直接对比 30% 经验阈值",
    ]
    if rsd_bio is not None:
        finite_bio = rsd_bio[np.isfinite(rsd_bio)]
        if finite_bio.size:
            bullets.append(f"生物样本特征 RSD 中位 {float(np.median(finite_bio)):.1f}%："
                           "若与 QC 接近，说明总变异主要由分析过程贡献")
    if pct_pass < 70:
        bullets.append(f"通过率偏低（{pct_pass:.0f}% < 70%）：建议在「特征过滤」设置 rsd_max="
                       f"{threshold:.0f} 剔除不稳定特征，或优化样品前处理/进样顺序")
    bullets.append("方法：内置实现（依据 Want 2010 / Dunn 2011）")

    return result(
        title="QC 样本 RSD",
        bullets=bullets,
        metrics={"n_qc": int(block["qc_mask"].sum()), "threshold": float(threshold),
                 "median_rsd_qc": float(np.median(finite_qc)) if finite_qc.size else None,
                 "n_pass": n_pass, "pct_pass": pct_pass},
        tables=[table("rsd_qc", f"QC RSD 最高的 200 个特征（阈值 {threshold:.0f}%）", [
            {"key": "feature", "label": "特征", "type": "string"},
            {"key": "rsd_qc", "label": "QC RSD%", "type": "number"},
            {"key": "rsd_bio", "label": "生物样本 RSD%", "type": "number"},
            {"key": "qc_over_bio", "label": "QC/生物", "type": "number"},
            {"key": "pass", "label": "通过", "type": "boolean"},
        ], rows, note="方法：内置实现（依据 Want 2010 / Dunn 2011）")],
        figures=figures,
    )


# ---------------------------------------------------------------------------
# 分析：missing_pattern（缺失模式）
# ---------------------------------------------------------------------------


def analysis_missing_pattern(ctx, params, outdir):
    """缺失模式：缺失率分布 + 样本×特征缺失热图（Want 2010 / Dunn 2011 缺失机制排查）。"""
    params = dict(params or {})
    max_features = max(_as_int(params, "max_features", 500), 10)
    X, qc_source = _qc_matrix(ctx)
    samples = _samples(ctx)
    features = _features(ctx)
    feature_missing = _missing_pct_features(X)
    sample_missing = 100.0 * (1.0 - np.isfinite(X).mean(axis=1))
    missing_mask = ~np.isfinite(X)

    n_missing = int(missing_mask.sum())
    overall = float(100.0 * missing_mask.mean()) if X.size else 0.0
    order = np.argsort(-feature_missing, kind="stable")[:min(max_features, X.shape[1])]
    sub = missing_mask[:, order]
    # 样本按缺失率排序，便于看“整行/整段”缺失模式
    sample_order = np.argsort(-sample_missing, kind="stable")

    figures = []
    try:
        fig, axes = new_fig(11.0, 4.6, ncols=2, gridspec_kw={"width_ratios": [1.25, 1.0]})
        axes = np.atleast_1d(axes).ravel()
        image = axes[0].imshow(sub[sample_order].T, aspect="auto", cmap="magma_r",
                              vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_xlabel(f"samples (n={len(sample_order)}, sorted by missing rate)", fontsize=9)
        axes[0].set_ylabel(f"top {order.size} features by missing", fontsize=9)
        axes[0].set_title("Missingness pattern (dark = missing)", fontsize=10)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(image, ax=axes[0], fraction=0.04, pad=0.02)
        axes[1].hist(feature_missing[np.isfinite(feature_missing)], bins=40,
                     color=PALETTE[1], alpha=0.8)
        axes[1].axvline(50.0, color="#e15759", linestyle="--", linewidth=1.0,
                        label="50% (filter default)")
        axes[1].set_xlabel("missing (%) per feature", fontsize=9)
        axes[1].set_ylabel("features", fontsize=9)
        axes[1].set_title("Feature missing-rate distribution", fontsize=10)
        axes[1].legend(fontsize=7)
        figures.append(save_fig(fig, outdir, "missing_pattern", title=None))
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for index in order[:200]:
        index = int(index)
        rows.append({"feature": features[index] if index < len(features) else str(index),
                     "missing_pct": float(feature_missing[index]),
                     "present_samples": int((~missing_mask[:, index]).sum())})
    sample_rows = [{"sample": samples[index], "missing_pct": float(sample_missing[index]),
                    "present_features": int((~missing_mask[index]).sum())}
                   for index in sample_order[:200]]

    high_feature = int((feature_missing > 50).sum())
    bad_samples = int((sample_missing > 30).sum())
    bullets = [
        f"整体缺失 {overall:.2f}%（{n_missing} 个单元）；有缺失的特征 {int((feature_missing > 0).sum())} 个，"
        f"缺失率 > 50% 的特征 {high_feature} 个",
        f"缺失统计基于{'原始强度（raw）' if qc_source == 'raw' else '当前矩阵'}："
        + ("反映导入时的原始缺失模式" if qc_source == "raw" else "当前矩阵已填补，缺失率可能为 0"),
        f"缺失率 > 30% 的样本 {bad_samples} 个"
        + ("：建议检查进样/前处理，或在样本表里剔除" if bad_samples else "：样本层面缺失可控"),
        "判读：随机缺失（MCAR）可用 half-min/KNN 填补；若整段/整行缺失更像批次或进样问题，"
        "应优先处理批次（批次校正）再做填补",
        "方法：内置实现（依据 Want 2010 / Dunn 2011）",
    ]

    return result(
        title="缺失模式",
        bullets=bullets,
        metrics={"missing_pct": overall, "n_missing_cells": n_missing,
                 "n_features_with_missing": int((feature_missing > 0).sum()),
                 "n_features_over_50pct": high_feature, "n_samples_over_30pct": bad_samples},
        tables=[
            table("missing_features", "缺失率最高的特征", [
                {"key": "feature", "label": "特征", "type": "string"},
                {"key": "missing_pct", "label": "缺失率%", "type": "number"},
                {"key": "present_samples", "label": "有值样本数", "type": "number"},
            ], rows),
            table("missing_samples", "缺失率最高的样本", [
                {"key": "sample", "label": "样本", "type": "string"},
                {"key": "missing_pct", "label": "缺失率%", "type": "number"},
                {"key": "present_features", "label": "有值特征数", "type": "number"},
            ], sample_rows),
        ],
        figures=figures,
    )


# ---------------------------------------------------------------------------
# 分析：sample_outlier（样本离群）
# ---------------------------------------------------------------------------


def analysis_sample_outlier(ctx, params, outdir):
    """样本离群：PCA + Hotelling T² + DModX（Wold 1994；SIMCA 语义）。"""
    params = dict(params or {})
    n_comp = _as_int(params, "n_comp", 3)
    alpha = _as_float(params, "alpha", 0.95) or 0.95
    if not (0.5 <= alpha < 1.0):
        raise bad_request("alpha 需要在 [0.5, 1) 之间（常用 0.95 / 0.99）")
    core = _pca_from_ctx(ctx, n_comp=n_comp)
    scores = np.asarray(core["scores"], dtype=np.float64)
    ratio = np.asarray(core["ratio"], dtype=np.float64)
    t2, t2_crit95, t2_crit99 = _hotelling_t2(scores, alpha=alpha)
    sse = core.get("sse")
    if sse is None:
        sse = _residual_sse(ctx, scores, core.get("columns"))
    n_features = core.get("columns")
    n_features = len(n_features) if n_features is not None else _X(ctx).shape[1]
    dmodx, dmodx_crit, s0 = _dmodx(np.asarray(sse, dtype=np.float64), n_features,
                                   scores.shape[1], alpha=alpha)

    samples = _samples(ctx)
    labels, label_title = _color_labels(ctx, params, params.get("color_by", "group"),
                                        _qc_mask(ctx, params.get("qc_label"))[0])
    over_t2 = np.where(t2 > t2_crit95)[0]
    over_dmodx = np.where(dmodx > dmodx_crit)[0] if np.isfinite(dmodx_crit) else np.array([], dtype=int)
    flagged = np.union1d(over_t2, over_dmodx)

    figures = [
        _fig_pca_scores(scores, ratio, labels, outdir, "sample_outlier_pca",
                        "PCA scores (outliers: over-limit samples)"),
        _fig_outliers(t2, t2_crit95, t2_crit99, dmodx, dmodx_crit, labels, outdir,
                      "sample_outlier_t2", "Hotelling T2 / DModX outlier screening"),
    ]
    try:
        fig, ax = new_fig(6.6, 4.8)
        ax.scatter(t2, dmodx, s=24, color=PALETTE[0], alpha=0.8, linewidths=0)
        if np.isfinite(t2_crit95):
            ax.axvline(t2_crit95, color="#e67e22", linestyle="--", linewidth=1.0,
                       label=f"T2 {alpha * 100:.0f}% limit")
        if np.isfinite(dmodx_crit):
            ax.axhline(dmodx_crit, color="#c0392b", linestyle="--", linewidth=1.0,
                       label=f"DModX {alpha * 100:.0f}% limit")
        if flagged.size:
            ax.scatter(t2[flagged], dmodx[flagged], s=42, facecolors="none",
                       edgecolors="#c0392b", linewidths=1.1, label=f"flagged ({flagged.size})")
        ax.set_xlabel("Hotelling T$^2$", fontsize=9)
        ax.set_ylabel("DModX", fontsize=9)
        ax.set_title("Outlier map (T2 vs DModX)", fontsize=10)
        ax.legend(fontsize=7)
        figures.append(save_fig(fig, outdir, "sample_outlier_map", title=None))
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for index in np.argsort(-t2)[:200]:
        index = int(index)
        rows.append({"sample": samples[index] if index < len(samples) else str(index),
                     "t2": float(t2[index]),
                     "dmodx": float(dmodx[index]),
                     "t2_over": bool(t2[index] > t2_crit95),
                     "dmodx_over": bool(np.isfinite(dmodx_crit) and dmodx[index] > dmodx_crit)})

    bullets = [
        f"PCA {scores.shape[1]} 个主成分（PC1 {ratio[0] * 100:.1f}%）："
        f"T² {alpha * 100:.0f}% 限 {_fmt(t2_crit95, 4)}，DModX 限 {_fmt(dmodx_crit, 4)}",
        f"超限样本：T² {over_t2.size} 个、DModX {over_dmodx.size} 个、并集 {flagged.size} 个",
    ]
    if flagged.size:
        names = "、".join(samples[index] for index in flagged[:10])
        bullets.append(f"建议复核：{names}" + ("…" if flagged.size > 10 else ""))
        bullets.append("离群样本可能来自进样异常/前处理失误：先看 QC 样本是否同样偏离；"
                       "确认后再决定剔除（剔除会改变分组平衡）")
    else:
        bullets.append("没有样本同时超过 T²/DModX 控制限：样本层面一致")
    if over_dmodx.size > 0.10 * max(t2.size, 1):
        bullets.append(
            f"DModX 超限比例偏高（{over_dmodx.size}/{t2.size}）：残差方差不均一时 DModX 限偏严，"
            "它只反映“离模型多远”，判读要与 T² 一起看")
    bullets.append("方法：内置实现（依据 Wold 1994；SIMCA 语义；Want 2010 / Dunn 2011 QC 原则）")

    return result(
        title="样本离群",
        bullets=bullets,
        metrics={"n_comp": int(scores.shape[1]), "alpha": float(alpha),
                 "t2_limit": float(t2_crit95), "dmodx_limit": float(dmodx_crit),
                 "n_t2_over": int(over_t2.size), "n_dmodx_over": int(over_dmodx.size),
                 "n_flagged": int(flagged.size), "s0": float(s0)},
        tables=[table("sample_outlier", "样本离群指标（按 T² 降序，Top 200）", [
            {"key": "sample", "label": "样本", "type": "string"},
            {"key": "t2", "label": "Hotelling T²", "type": "number"},
            {"key": "dmodx", "label": "DModX", "type": "number"},
            {"key": "t2_over", "label": "T² 超限", "type": "boolean"},
            {"key": "dmodx_over", "label": "DModX 超限", "type": "boolean"},
        ], rows, note="方法：内置实现（依据 Wold 1994；SIMCA 语义）")],
        figures=figures,
        sample_columns=[{"key": "t2", "kind": "numeric"}, {"key": "dmodx", "kind": "numeric"}],
        sample_values={"t2": t2.tolist(), "dmodx": dmodx.tolist()},
    )


# ---------------------------------------------------------------------------
# 分析：sample_corr（样本相关性）
# ---------------------------------------------------------------------------


def _corr_matrix(X: np.ndarray, method: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if method in ("spearman", "spear", "rank"):
        ranks = np.argsort(np.argsort(arr, axis=1, kind="stable"), axis=1, kind="stable")
        arr = ranks.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(arr)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def analysis_sample_corr(ctx, params, outdir):
    """样本相关性 + 层次聚类（Pearson/Spearman；内置实现，依据 Want 2010 / Dunn 2011）。"""
    params = dict(params or {})
    method = str(params.get("method") or "pearson").strip().lower()
    if method not in ("pearson", "spearman", "spear", "rank"):
        raise bad_request(f"不支持的相关系数：{method}（可选 pearson / spearman）")
    method = "spearman" if method in ("spear", "rank") else method
    matrix, corr_source = _filled_log_matrix(ctx)
    samples = _samples(ctx)
    corr = _corr_matrix(matrix, method)
    n = corr.shape[0]
    off = corr[~np.eye(n, dtype=bool)].reshape(n, n - 1) if n > 1 else np.zeros((n, 0))
    mean_corr = off.mean(axis=1) if n > 1 else np.ones(n)
    min_corr = off.min(axis=1) if n > 1 else np.ones(n)
    max_corr = off.max(axis=1) if n > 1 else np.ones(n)
    labels, label_title = _color_labels(ctx, params, params.get("color_by", "group"),
                                        _qc_mask(ctx, params.get("qc_label"))[0])

    q1, q3 = np.percentile(mean_corr, [25, 75]) if n > 3 else (mean_corr.mean(), mean_corr.mean())
    iqr = max(float(q3 - q1), 1e-6)
    low_cut = float(q1 - 1.5 * iqr)
    flagged = np.where(mean_corr < low_cut)[0]

    # 层次聚类排序
    order = np.arange(n)
    dendro_ok = False
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        distance = 1.0 - corr
        np.fill_diagonal(distance, 0.0)
        distance = np.clip((distance + distance.T) / 2.0, 0.0, 2.0)
        link = linkage(squareform(distance, checks=False), method="average")
        order = np.asarray(leaves_list(link), dtype=int)
        dendro_ok = True
    except Exception:  # noqa: BLE001
        order = np.arange(n)

    figures = []
    try:
        show = order[:200]
        fig, ax = new_fig(7.6, 6.6)
        image = ax.imshow(corr[np.ix_(show, show)], cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                          interpolation="nearest")
        ax.set_title(f"Sample-sample {method} correlation (hierarchical order)", fontsize=10)
        step = max(1, show.size // 20)
        ax.set_xticks(np.arange(0, show.size, step))
        ax.set_xticklabels([samples[index] for index in show[::step]], rotation=90, fontsize=6)
        ax.set_yticks(np.arange(0, show.size, step))
        ax.set_yticklabels([samples[index] for index in show[::step]], fontsize=6)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="correlation")
        figures.append(save_fig(fig, outdir, "sample_corr_heatmap", title=None))
    except Exception:  # noqa: BLE001
        pass
    if dendro_ok:
        try:
            from scipy.cluster.hierarchy import dendrogram

            fig, ax = new_fig(10.0, 4.6)
            dendrogram(link, labels=[samples[index] for index in range(n)], ax=ax,
                       leaf_font_size=6, color_threshold=0.7 * float(link[:, 2].max()),
                       above_threshold_color="#8a919c")
            ax.set_title(f"Sample clustering (1 - {method} correlation, average linkage)", fontsize=10)
            ax.set_ylabel("distance", fontsize=9)
            ax.grid(False)
            figures.append(save_fig(fig, outdir, "sample_corr_dendro", title=None))
        except Exception:  # noqa: BLE001
            pass

    rows = []
    for index in np.argsort(mean_corr):
        index = int(index)
        rows.append({"sample": samples[index],
                     "mean_corr": float(mean_corr[index]),
                     "min_corr": float(min_corr[index]),
                     "max_corr": float(max_corr[index]),
                     "flag": bool(mean_corr[index] < low_cut)})

    bullets = [
        f"{method} 相关：样本间平均相关 {float(np.mean(mean_corr)):.3f}，"
        f"最低 {float(np.min(min_corr)):.3f}",
        f"低相关样本阈值 {low_cut:.3f}（Q1 − 1.5×IQR）：命中 {flagged.size} 个",
        ("在原始强度的 log2 上计算（" if corr_source == "raw" else "在当前矩阵上计算（")
        + "Pearson 对每样本乘性稀释不敏感，log 后常数偏移会被中心化消掉）",
    ]
    if flagged.size:
        bullets.append("低相关样本：" + "、".join(samples[index] for index in flagged[:10])
                       + "（提示前处理异常或样本混淆）")
    else:
        bullets.append("没有明显低相关样本：样本间整体一致性良好")
    if labels is not None:
        bullets.append(f"相关性矩阵/聚类按「{label_title}」可以进一步核对分组是否成簇")
    bullets.append("方法：内置实现（依据 Want 2010 / Dunn 2011）")

    return result(
        title="样本相关性",
        bullets=bullets,
        metrics={"method": method, "mean_corr": float(np.mean(mean_corr)),
                 "low_cut": low_cut, "n_flagged": int(flagged.size)},
        tables=[table("sample_corr", f"样本间 {method} 相关（按平均相关升序）", [
            {"key": "sample", "label": "样本", "type": "string"},
            {"key": "mean_corr", "label": "平均相关", "type": "number"},
            {"key": "min_corr", "label": "最低相关", "type": "number"},
            {"key": "max_corr", "label": "最高相关", "type": "number"},
            {"key": "flag", "label": "低相关标记", "type": "boolean"},
        ], rows, note="方法：内置实现（依据 Want 2010 / Dunn 2011）")],
        figures=figures,
    )


# ---------------------------------------------------------------------------
# 注册（DESIGN §2.5 / §3 / §4）
# ---------------------------------------------------------------------------

STEPS = [
    {
        "id": "qc", "title": "样本质控", "requires": [],
        "method": "每样本 TIC / 检出特征数 / 缺失率 + QC 样本 RSD（Want 2010；Dunn 2011）",
        "params": [
            {"key": "group_col", "label": "分组列（着色用）", "type": "text", "default": None},
            {"key": "qc_label", "label": "QC 样本标签（正则）", "type": "text", "default": ""},
            {"key": "min_features", "label": "每样本最少特征", "type": "number", "default": 0},
        ],
        "run": step_qc,
    },
    {
        "id": "filter", "title": "特征过滤", "requires": ["qc"],
        "method": "非缺失比例 / 平均强度 / QC-RSD 过滤（Dunn 2011）",
        "params": [
            {"key": "min_sample_frac", "label": "最少非缺失比例", "type": "number", "default": 0.5},
            {"key": "min_intensity", "label": "最小平均强度", "type": "number", "default": 0},
            {"key": "rsd_max", "label": "QC RSD 上限(%)", "type": "number", "default": None},
        ],
        "run": step_filter,
    },
    {
        "id": "impute", "title": "缺失值填补", "requires": ["filter"],
        "method": "half-min / 中位数 / KNN（Troyanskaya 2001）/ 迭代 PCA（Josse & Husson 2012）",
        "params": [
            {"key": "method", "label": "方法", "type": "select", "default": "half_min",
             "options": [{"value": "half_min", "label": "half-min（每特征 min/2）"},
                         {"value": "median", "label": "特征中位数"},
                         {"value": "knn", "label": "KNN"},
                         {"value": "ppca", "label": "迭代 PCA"}]},
            {"key": "k", "label": "KNN 邻居数", "type": "number", "default": 5},
        ],
        "run": step_impute,
    },
    {
        "id": "normalize", "title": "归一化", "requires": ["impute"],
        "method": "PQN（Dieterle 2006）/ TIC / 中位数 / 分位数（Bolstad 2003）",
        "params": [
            {"key": "method", "label": "方法", "type": "select", "default": "pqn",
             "options": [{"value": "pqn", "label": "PQN 概率商归一化"},
                         {"value": "tic", "label": "TIC 总离子流"},
                         {"value": "median", "label": "中位数"},
                         {"value": "quantile", "label": "分位数标准化"},
                         {"value": "none", "label": "跳过"}]},
        ],
        "run": step_normalize,
    },
    {
        "id": "transform", "title": "变换/标度", "requires": ["normalize"],
        "method": "log2/log10/cuberoot + auto/pareto/range 标度（van den Berg 2006）",
        "params": [
            {"key": "transform", "label": "变换", "type": "select", "default": "log2",
             "options": [{"value": "log2", "label": "log2"},
                         {"value": "log10", "label": "log10"},
                         {"value": "cuberoot", "label": "立方根"},
                         {"value": "none", "label": "不变换"}]},
            {"key": "scale", "label": "标度", "type": "select", "default": "auto",
             "options": [{"value": "auto", "label": "auto（按偏度选 UV/pareto）"},
                         {"value": "pareto", "label": "Pareto"},
                         {"value": "range", "label": "Range"},
                         {"value": "none", "label": "不标度"}]},
        ],
        "run": step_transform,
    },
    {
        "id": "pca", "title": "主成分分析", "requires": ["transform"],
        "method": "SVD + Hotelling T² + DModX（Hotelling 1931；Wold 1994）",
        "params": [
            {"key": "n_comp", "label": "主成分数", "type": "number", "default": 3},
            {"key": "color_by", "label": "着色", "type": "select", "default": "group",
             "options": [{"value": "group", "label": "分组"},
                         {"value": "batch", "label": "批次"},
                         {"value": "qc", "label": "QC/样本"},
                         {"value": "none", "label": "单色"}]},
        ],
        "run": step_pca,
    },
]

ANALYSES = [
    {
        "id": "qc_summary", "title": "质控汇总", "group": "质控与基础",
        "method": "内置实现（依据 Want 2010 / Dunn 2011）：TIC/检出/缺失/RSD 汇总表 + TIC/RSD 图",
        "needs": ["qc"], "params": [],
        "run": analysis_qc_summary,
    },
    {
        "id": "rsd_qc", "title": "QC 样本 RSD", "group": "质控与基础",
        "method": "内置实现（依据 Want 2010 / Dunn 2011）：RSD = SD/mean，QC 与生物样本对比",
        "needs": ["qc"],
        "params": [{"key": "qc_label", "label": "QC 样本标签（正则）", "type": "text", "default": ""},
                   {"key": "threshold", "label": "RSD 阈值(%)", "type": "number", "default": 30}],
        "run": analysis_rsd_qc,
    },
    {
        "id": "missing_pattern", "title": "缺失模式", "group": "质控与基础",
        "method": "内置实现（依据 Want 2010 / Dunn 2011）：缺失率分布 + 样本×特征缺失热图",
        "needs": ["qc"],
        "params": [{"key": "max_features", "label": "热图最多特征数", "type": "number", "default": 500}],
        "run": analysis_missing_pattern,
    },
    {
        "id": "sample_outlier", "title": "样本离群", "group": "质控与基础",
        "method": "内置实现（依据 Wold 1994；SIMCA 语义；Want 2010 / Dunn 2011）：PCA + Hotelling T² + DModX",
        "needs": ["transform"],
        "params": [{"key": "n_comp", "label": "主成分数", "type": "number", "default": 3},
                   {"key": "alpha", "label": "置信度", "type": "number", "default": 0.95}],
        "run": analysis_sample_outlier,
    },
    {
        "id": "sample_corr", "title": "样本相关性", "group": "质控与基础",
        "method": "内置实现（依据 Want 2010 / Dunn 2011）：Pearson/Spearman 相关 + 层次聚类",
        "needs": ["transform"],
        "params": [{"key": "method", "label": "相关系数", "type": "select", "default": "pearson",
                    "options": [{"value": "pearson", "label": "Pearson"},
                                {"value": "spearman", "label": "Spearman"}]}],
        "run": analysis_sample_corr,
    },
]

__all__ = ["STEPS", "ANALYSES"]
