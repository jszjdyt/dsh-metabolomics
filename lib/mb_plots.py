"""
dsh-metabolomics — 基础绘图模块（mb-base）

导出 `PLOTS`：`kind -> fn(ctx, params, outdir) -> save_fig 返回的 dict`
（registry 把各模块的 PLOTS 合并成全局表，`plot` 命令按 kind 分发，DESIGN §2.5 / §5）。

本文件覆盖 DESIGN §5 白名单里属于基础图的部分（10 个 kind）：
    qc_tic        每样本 TIC（总离子流）柱状，分组/QC 上色
    qc_rsd        QC（或全体）特征 RSD 分布 + 30% 阈值
    qc_missing    缺失率：每样本柱状 + 每特征分布
    sample_corr   样本×样本相关热图（Pearson/Spearman）+ 层次聚类排序
    sample_dendro 样本层次聚类树（1 - 相关系数距离，average linkage）
    pca_scores    PC1–PC2 得分图 + 组内 95% Hotelling 椭圆
    pca_variance  碎石图（单主成分 + 累计解释方差）
    hotelling_t2  Hotelling T² 与 DModX 离群图（Wold 1994）
    heatmap_top   Top 特征热图（行 z-score + 行列层次聚类排序）
    batch_pca     批次 vs 分组 PCA 对照（看批次是否主导结构）

通用参数（DESIGN §5）：`color_by`（group|batch|qc|none，也接受样本列名）、`top_n=20`、`dpi=140`。
约定：
  * 图上文字一律英文（服务器可能缺 CJK 字体）；中文只放在返回 dict 的 title/warnings。
  * 只读 ctx：不修改 ctx.X / results；需要子集时新建数组。
  * 缺前置 → prerequisite(中文原因)；参数非法 → bad_request(中文原因)。
  * 配色用 mb_common.PALETTE：同一分组值在同一个数据集里永远同色（按标签排序取色）。
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
    new_fig,
    palette,
    prerequisite,
    save_fig,
    zscore_rows,
)

from matplotlib.colors import ListedColormap  # noqa: E402 - mb_common 已设好 Agg 后端
from matplotlib.patches import Ellipse, Patch  # noqa: E402

DEFAULT_DPI = 140
DEFAULT_TOP_N = 20
DEFAULT_RSD_CUT = 30.0
MAX_MULTIVARIATE_FEATURES = 20000
MAX_HEATMAP_SAMPLES = 200


# ---------------------------------------------------------------------------
# ctx / 参数小工具
# ---------------------------------------------------------------------------


def _X(ctx) -> np.ndarray:
    arr = (ctx or {}).get("X")
    if arr is None:
        raise prerequisite("数据集没有强度矩阵，请先导入数据或载入示例数据")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise EngineError("internal", f"强度矩阵形状异常：{arr.shape}")
    return arr


def _samples(ctx) -> list[str]:
    return [str(item) for item in ((ctx or {}).get("samples") or [])]


def _features(ctx) -> list[str]:
    return [str(item) for item in ((ctx or {}).get("features") or [])]


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


def _as_int(params, key, default: int) -> int:
    value = _param(params, key, None)
    if value is None:
        return int(default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _as_float(params, key, default: float) -> float:
    value = _param(params, key, None)
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _dpi(params) -> int:
    return int(max(60, min(_as_int(params, "dpi", DEFAULT_DPI), 400)))


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
    uniq = list(dict.fromkeys([str(item) for item in values]))

    def key(value: str):
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, value)

    return sorted(uniq, key=key)


def _cat_colors(labels) -> dict:
    order = _sorted_labels(labels)
    colors = palette(len(order))
    return {label: colors[index % len(colors)] for index, label in enumerate(order)}


def _group_labels(ctx, params=None):
    params = params or {}
    key = str(params.get("group_col") or (ctx or {}).get("group_key") or "").strip()
    info = _info(ctx)
    if key and key in info.columns:
        return np.asarray([str(value) for value in info[key].tolist()], dtype=object), key
    return None, None


def _qc_mask(ctx, qc_label=None):
    """QC 样本掩码：qc_label 正则匹配样本名，或 sample_info 的 qc 列=1。"""
    samples = _samples(ctx)
    info = _info(ctx)
    mask = np.zeros(len(samples), dtype=bool)
    label = str(qc_label or "").strip()
    if label:
        try:
            regex = re.compile(label)
        except re.error as error:
            raise bad_request(f"qc_label 正则编译失败：{error}") from error
        mask |= np.array([bool(regex.search(sample)) for sample in samples], dtype=bool)
    for col in ("qc", "is_qc", "qc_flag", "qc_sample"):
        if col in info.columns:
            mask |= np.array([str(value).strip().lower() in ("1", "1.0", "true", "yes", "y", "qc")
                              for value in info[col].tolist()], dtype=bool)
    for col in ("sample_type", "sampletype"):
        if col in info.columns:
            mask |= np.array([str(value).strip().lower() in ("qc", "pool", "pooled", "quality control")
                              for value in info[col].tolist()], dtype=bool)
    return mask


def _batch_labels(ctx, params=None):
    """批次标签：color_by/params.batch_key 指定列 → 名字像批次的列。"""
    params = params or {}
    info = _info(ctx)
    requested = str(params.get("batch_key") or params.get("batch_col") or "").strip()
    lowered = [str(col).lower() for col in info.columns]
    for candidate in (requested, "batch", "batch_id", "batchid", "plate", "run", "cohort",
                      "library_batch", "extraction_batch"):
        if not candidate:
            continue
        if candidate in info.columns:
            return np.asarray([str(value) for value in info[candidate].tolist()], dtype=object), candidate
        if candidate.lower() in lowered:
            column = info.columns[lowered.index(candidate.lower())]
            return np.asarray([str(value) for value in info[column].tolist()], dtype=object), str(column)
    return None, None


def _color_labels(ctx, params, color_by=None):
    """color_by → (labels, title)；labels=None 表示单色。"""
    lowered = str(color_by if color_by is not None else _param(params, "color_by", "group") or "group")
    lowered = lowered.strip()
    key = lowered.lower()
    if key in ("", "none", "no", "false"):
        return None, "none"
    if key in ("group", "groups", "group_key", "condition"):
        labels, column = _group_labels(ctx, params)
        return labels, (column or "group")
    if key in ("qc", "quality", "qc_label"):
        mask = _qc_mask(ctx, _param(params, "qc_label", ""))
        if not mask.any():
            return None, "none"
        return np.asarray(["QC" if bool(flag) else "Sample" for flag in mask], dtype=object), "qc"
    if key in ("batch", "batch_id", "plate", "run", "cohort"):
        labels, column = _batch_labels(ctx, params)
        return labels, (column or "batch")
    info = _info(ctx)
    if lowered in info.columns:
        return np.asarray([str(value) for value in info[lowered].tolist()], dtype=object), lowered
    return None, "none"


# ---------------------------------------------------------------------------
# QC / PCA 数值（与 mb_prep 的步骤实现保持一致；模块之间不互相 import）
# ---------------------------------------------------------------------------


def _qc_matrix(ctx):
    """QC 图（TIC/RSD/缺失）用的矩阵：优先 ctx.raw（原始强度；DESIGN §2.3）。"""
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
    """样本相关用矩阵：优先 log2(raw)（Pearson 对每样本乘性稀释不敏感），缺失用特征中位数填。"""
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
        return np.log2(np.where(arr + offset > 0, arr + offset, 2.0 ** -20)), source


def _sample_metrics(X: np.ndarray):
    finite = np.isfinite(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tic = np.nan_to_num(np.nansum(X, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    return tic, finite.sum(axis=1).astype(np.int64), 100.0 * (1.0 - finite.mean(axis=1))


def _feature_rsd(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.full(arr.shape[1] if arr.ndim == 2 else 0, np.nan)
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nansum(arr, axis=0) / np.maximum(counts, 1)
        sd = np.nanstd(arr, axis=0, ddof=1)
    sd = np.where(counts > 1, sd, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rsd = np.where(np.isfinite(mean) & (mean > 0), sd / mean * 100.0, np.nan)
    return rsd


def _missing_pct_features(X: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 - np.isfinite(X).mean(axis=0))


def _cached_qc(ctx):
    cached = ((ctx or {}).get("results") or {}).get("qc")
    if not isinstance(cached, dict):
        return None
    tic = cached.get("tic")
    if not isinstance(tic, (list, tuple)) or len(tic) != _X(ctx).shape[0]:
        return None
    try:
        out = {
            "tic": np.asarray(tic, dtype=np.float64),
            "detected": np.asarray(cached.get("detected") or [], dtype=np.float64),
            "missing_pct": np.asarray(cached.get("missing_pct") or [], dtype=np.float64),
            "qc_mask": np.asarray(cached.get("qc_mask") or [], dtype=bool),
            "rsd": np.asarray([np.nan if value is None else float(value)
                               for value in (cached.get("rsd") or [])], dtype=np.float64),
            "rsd_scope": str(cached.get("rsd_scope") or ""),
        }
    except (TypeError, ValueError):
        return None
    if out["qc_mask"].size != _X(ctx).shape[0] or out["missing_pct"].size != _X(ctx).shape[0]:
        return None
    return out


def _qc_block(ctx, params):
    """TIC / 检出 / 缺失 / RSD：优先复用 qc 步骤缓存，否则在原始强度（raw）上现场算。"""
    X, source = _qc_matrix(ctx)
    cached = _cached_qc(ctx)
    if cached is not None:
        block = dict(cached)
        if block["rsd"].size != X.shape[1]:
            block["rsd"] = _feature_rsd(X[block["qc_mask"]]) if int(block["qc_mask"].sum()) >= 2 \
                else _feature_rsd(X)
            block["rsd_scope"] = f"QC({int(block['qc_mask'].sum())})" \
                if int(block["qc_mask"].sum()) >= 2 else "all samples"
    else:
        tic, detected, missing_pct = _sample_metrics(X)
        qc_mask = _qc_mask(ctx, _param(params, "qc_label", ""))
        n_qc = int(qc_mask.sum())
        rsd = _feature_rsd(X[qc_mask]) if n_qc >= 2 else _feature_rsd(X)
        block = {"tic": tic, "detected": detected.astype(np.float64),
                 "missing_pct": missing_pct, "qc_mask": qc_mask, "rsd": rsd,
                 "rsd_scope": f"QC({n_qc})" if n_qc >= 2 else "all samples"}
    block["feature_missing"] = _missing_pct_features(X)
    block["source"] = source
    return block


def _variance_subset(X: np.ndarray, limit: int = MAX_MULTIVARIATE_FEATURES):
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
    sse = np.maximum((centered ** 2).sum(axis=1) - (scores ** 2).sum(axis=1), 0.0)
    return {"scores": scores, "ratio": ratio, "loadings": vt[:k, :], "columns": columns,
            "center": center, "subsampled": bool(subsampled), "n_features_total": int(n_features),
            "sse": sse, "n_comp": int(k)}


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


def _pca_from_ctx(ctx, n_comp: int = 3) -> dict:
    """优先复用 pca 步骤缓存（results["pca"]）；store 已被 jsonable 成 list，这里再转回 ndarray。"""
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


def _chi2_quantile(alpha: float, df: int = 2) -> float:
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(float(alpha), int(df)))
    except Exception:  # noqa: BLE001
        return 5.991 if int(df) == 2 else float(df) * 2.0


def _f_quantile(alpha: float, dfn: float, dfd: float) -> float:
    try:
        from scipy.stats import f

        return float(f.ppf(float(alpha), float(dfn), float(dfd)))
    except Exception:  # noqa: BLE001
        return _chi2_quantile(alpha, max(1, int(round(dfn)))) / max(float(dfn), 1.0)


def _hotelling_t2(scores: np.ndarray):
    mat = np.asarray(scores, dtype=np.float64)
    n_samples, k = mat.shape
    if k == 0 or n_samples <= k:
        return np.zeros(n_samples), float("nan"), float("nan")
    variance = mat.var(axis=0, ddof=1)
    variance = np.where(variance > 0, variance, 1.0)
    t2 = ((mat ** 2) / variance[None, :]).sum(axis=1)
    scale = k * (n_samples ** 2 - 1) / (n_samples * (n_samples - k))
    return t2, float(scale * _f_quantile(0.95, k, n_samples - k)), \
        float(scale * _f_quantile(0.99, k, n_samples - k))


def _dmodx(sse: np.ndarray, n_features: int, n_comp: int):
    sse = np.asarray(sse, dtype=np.float64)
    n_samples = sse.size
    k_eff = max(int(n_features) - int(n_comp), 1)
    dmodx = np.sqrt(np.maximum(sse, 0.0) / k_eff)
    dof = max(n_samples - int(n_comp) - 1, 1)
    s0 = math.sqrt(float(np.maximum(sse, 0.0).sum()) / (dof * k_eff)) if sse.size else float("nan")
    crit = s0 * math.sqrt(_f_quantile(0.95, k_eff, max(dof * k_eff, 1))) if s0 > 0 else float("nan")
    return dmodx, float(crit)


def _ellipse_patch(points: np.ndarray, color: str, alpha: float = 0.95):
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
    return Ellipse((float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                   2.0 * math.sqrt(values[0] * chi2), 2.0 * math.sqrt(max(values[1] * chi2, 0.0)),
                   angle=angle, facecolor="none", edgecolor=color, linewidth=1.1,
                   linestyle="--", alpha=0.9)


def _scatter_groups(ax, x, y, labels, colors, point_size: float = 26.0, alpha: float = 0.85,
                    legend: bool = True, count_label: bool = True):
    """按标签分组散点 + 图例（颜色稳定）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if labels is None:
        ax.scatter(x, y, s=point_size, color=PALETTE[0], linewidths=0, alpha=alpha,
                   label="samples")
        return
    labels = np.asarray([str(item) for item in labels], dtype=object)
    for label in _sorted_labels(labels):
        mask = labels == label
        text = f"{label} (n={int(mask.sum())})" if count_label else str(label)
        ax.scatter(x[mask], y[mask], s=point_size, color=colors[label], linewidths=0,
                   alpha=alpha, label=text)
    if legend and labels.size:
        ax.legend(fontsize=7, loc="best")


def _draw_ellipses(ax, x, y, labels, colors) -> None:
    if labels is None:
        return
    labels = np.asarray([str(item) for item in labels], dtype=object)
    points = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    for label in _sorted_labels(labels):
        mask = labels == label
        patch = _ellipse_patch(points[mask], colors[label])
        if patch is not None:
            ax.add_patch(patch)


def _de_frame(ctx) -> pd.DataFrame | None:
    """读 results["de"]（DataFrame / dict / list[dict]）——用于 heatmap_top 选特征。"""
    stored = ((ctx or {}).get("results") or {}).get("de")
    if stored is None:
        return None
    if isinstance(stored, pd.DataFrame):
        return stored if not stored.empty else None
    if isinstance(stored, dict):
        for key in ("table", "de_table", "rows", "data"):
            value = stored.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                return value
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.DataFrame(value)
        try:
            frame = pd.DataFrame(stored)
            return frame if not frame.empty else None
        except Exception:  # noqa: BLE001
            return None
    if isinstance(stored, list) and stored and isinstance(stored[0], dict):
        return pd.DataFrame(stored)
    return None


def _top_feature_indices(ctx, top_n: int) -> tuple[np.ndarray, str]:
    """挑用于画图的特征：优先差异结果（按 q/p 升序），否则按方差（降序）。"""
    features = _features(ctx)
    n_features = _X(ctx).shape[1]
    limit = int(max(2, min(max(int(top_n), 2), 400)))
    frame = _de_frame(ctx)
    if frame is not None and not frame.empty:
        feature_column = None
        for candidate in ("feature", "feature_id", "metabolite", "id", "name"):
            if candidate in frame.columns:
                feature_column = candidate
                break
        if feature_column is not None:
            rank_column = None
            for candidate in ("q", "fdr", "padj", "adj.P.Val", "qval", "q_value",
                              "p", "pval", "p_value", "P.Value"):
                if candidate in frame.columns:
                    rank_column = candidate
                    break
            ids = [str(value) for value in frame[feature_column].tolist()]
            if rank_column is not None:
                ranks = pd.to_numeric(frame[rank_column], errors="coerce").to_numpy(dtype=np.float64)
                order = np.argsort(np.nan_to_num(ranks, nan=np.inf), kind="stable")
            else:
                order = np.arange(len(ids))
            index = {name: position for position, name in enumerate(features)}
            picked = [index[ids[position]] for position in order if ids[position] in index]
            picked = list(dict.fromkeys(picked))[:limit]
            if picked:
                return np.asarray(picked, dtype=np.int64), f"top {len(picked)} differential features"
    # 兜底：方差最大的特征
    arr = _X(ctx)
    variances = np.zeros(n_features, dtype=np.float64)
    step = max(1, int(4_000_000 / max(arr.shape[0], 1)))
    for start in range(0, n_features, step):
        chunk = np.asarray(arr[:, start:start + step], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            variances[start:start + step] = np.nan_to_num(chunk.var(axis=0), nan=0.0)
    picked = np.argsort(-variances, kind="stable")[:limit]
    return np.sort(picked).astype(np.int64), f"top {len(picked)} high-variance features"


def _hier_order(matrix: np.ndarray) -> np.ndarray:
    """行方向的 average-linkage 层次聚类叶子顺序（失败保持原序）。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return np.arange(matrix.shape[0] if matrix.ndim >= 1 else 0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        return np.asarray(leaves_list(linkage(pdist(matrix, metric="euclidean"),
                                              method="average")), dtype=np.int64)
    except Exception:  # noqa: BLE001 - 聚类失败保持原序
        return np.arange(matrix.shape[0])


def _contrast_text(hex_color: str) -> str:
    """浅色块上用黑字、深色块上用白字（色带上的组名可读性）。"""
    try:
        text = str(hex_color).lstrip("#")
        r, g, b = (int(text[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.55 else "white"
    except Exception:  # noqa: BLE001
        return "white"


def _short(text: str, limit: int = 26) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _feature_label(ctx, index: int) -> str:
    """特征显示名：有 name 用 name(mz)，否则用特征 id（截断）。"""
    features = _features(ctx)
    finfo = _feature_info(ctx)
    base = features[index] if index < len(features) else f"F{index}"
    try:
        if "name" in finfo.columns:
            name = finfo["name"].iloc[index]
            if isinstance(name, str) and name.strip() and name.strip().isascii():
                return name.strip()
    except Exception:  # noqa: BLE001
        pass
    return base


# ---------------------------------------------------------------------------
# 1. qc_tic
# ---------------------------------------------------------------------------


def plot_qc_tic(ctx, params, outdir):
    """每样本 TIC 柱状图（按 TIC 降序；分组上色，QC 样本描边）。"""
    params = dict(params or {})
    X = _X(ctx)
    samples = _samples(ctx)
    block = _qc_block(ctx, params)
    tic = np.asarray(block["tic"], dtype=np.float64)
    qc_mask = np.asarray(block["qc_mask"], dtype=bool)
    labels, title_label = _color_labels(ctx, params, _param(params, "color_by", "group"))
    dpi = _dpi(params)

    order = np.argsort(-np.nan_to_num(tic, nan=-np.inf), kind="stable")
    fig, ax = new_fig(9.6, 4.3)
    positions = np.arange(tic.size)
    if labels is not None:
        colors = _cat_colors(labels)
        bar_colors = [colors[str(labels[index])] for index in order]
        handles = [Patch(facecolor=colors[label], label=label) for label in _sorted_labels(labels)]
    else:
        bar_colors = ["#8a919c" if qc_mask[index] else PALETTE[0] for index in order]
        handles = ([Patch(facecolor=PALETTE[0], label="sample"),
                    Patch(facecolor="#8a919c", label="QC")] if qc_mask.any() else [])
    ax.bar(positions, tic[order], color=bar_colors, width=0.82, linewidth=0)
    if qc_mask.any():
        hit = np.array([bool(qc_mask[index]) for index in order])
        ax.scatter(positions[hit], tic[order][hit], s=18, facecolors="none",
                   edgecolors="#2b2f36", linewidths=0.8, zorder=3)
    median = float(np.nanmedian(tic)) if tic.size else 0.0
    ax.axhline(median, color="#e15759", linestyle="--", linewidth=1.0)
    handles.append(Patch(facecolor="none", edgecolor="#e15759", linestyle="--",
                         label=f"median {_fmt(median, 3)}"))
    step = max(1, tic.size // 24)
    ax.set_xticks(positions[::step])
    ax.set_xticklabels([samples[index] for index in order[::step]], rotation=90, fontsize=6)
    ax.set_ylabel("TIC (sum of intensities)")
    ax.set_title(f"Total ion current per sample (color = {title_label})")
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right", ncols=2)
    return save_fig(fig, outdir, "qc_tic", title=None, dpi=dpi)


# ---------------------------------------------------------------------------
# 2. qc_rsd
# ---------------------------------------------------------------------------


def plot_qc_rsd(ctx, params, outdir):
    """特征 RSD 分布（有 QC 用 QC，否则全体）+ 30% 阈值 + 累计曲线。"""
    params = dict(params or {})
    block = _qc_block(ctx, params)
    threshold = _as_float(params, "threshold", DEFAULT_RSD_CUT)
    rsd = np.asarray(block["rsd"], dtype=np.float64)
    values = rsd[np.isfinite(rsd)]
    if values.size == 0:
        raise prerequisite("没有可计算的 RSD（至少需要 2 个样本，且特征均值需要 > 0）")
    dpi = _dpi(params)
    clipped = np.clip(values, 0, np.quantile(values, 0.995) if values.size > 20 else values.max())

    fig, axes = new_fig(10.4, 4.1, ncols=2)
    axes = np.atleast_1d(axes).ravel()
    axes[0].hist(clipped, bins=48, color=PALETTE[2], alpha=0.75,
                 label=f"RSD ({block['rsd_scope']}, n={values.size})")
    axes[0].axvline(threshold, color="#c0392b", linestyle="--", linewidth=1.0,
                    label=f"{threshold:.0f}% threshold")
    axes[0].axvline(float(np.median(values)), color="#4e79a7", linestyle=":", linewidth=1.0,
                    label=f"median {_fmt(np.median(values), 3)}%")
    axes[0].set_xlabel("feature RSD (%)")
    axes[0].set_ylabel("features")
    axes[0].set_title(f"Feature RSD distribution ({block.get('source', 'current')} matrix)")
    axes[0].legend(fontsize=7)

    ordered = np.sort(values)
    cumulative = np.arange(1, ordered.size + 1) / ordered.size * 100.0
    axes[1].plot(ordered, cumulative, color=PALETTE[0], linewidth=1.3)
    axes[1].axvline(threshold, color="#c0392b", linestyle="--", linewidth=1.0,
                    label=f"{threshold:.0f}% threshold")
    axes[1].axhline(float((values <= threshold).mean() * 100.0), color="#8a919c",
                    linestyle=":", linewidth=1.0,
                    label=f"{float((values <= threshold).mean() * 100.0):.0f}% pass")
    axes[1].set_xlabel("feature RSD (%)")
    axes[1].set_ylabel("cumulative features (%)")
    axes[1].set_title("Cumulative RSD curve")
    axes[1].legend(fontsize=7)
    return save_fig(fig, outdir, "qc_rsd", title=None, dpi=dpi)


# ---------------------------------------------------------------------------
# 3. qc_missing
# ---------------------------------------------------------------------------


def plot_qc_missing(ctx, params, outdir):
    """缺失率：左=每样本缺失率（排序），右=每特征缺失率分布。"""
    params = dict(params or {})
    X = _X(ctx)
    samples = _samples(ctx)
    block = _qc_block(ctx, params)
    dpi = _dpi(params)
    sample_missing = np.asarray(block["missing_pct"], dtype=np.float64)
    feature_missing = np.asarray(block["feature_missing"], dtype=np.float64)

    fig, axes = new_fig(10.4, 4.0, ncols=2)
    axes = np.atleast_1d(axes).ravel()
    order = np.argsort(-sample_missing, kind="stable")
    positions = np.arange(sample_missing.size)
    axes[0].bar(positions, sample_missing[order], color=PALETTE[0], width=0.82, linewidth=0)
    axes[0].axhline(float(np.nanmedian(sample_missing)), color="#e15759", linestyle="--",
                    linewidth=1.0, label=f"median {_fmt(np.nanmedian(sample_missing), 3)}%")
    step = max(1, sample_missing.size // 14)
    axes[0].set_xticks(positions[::step])
    axes[0].set_xticklabels([samples[index] for index in order[::step]], rotation=90, fontsize=6)
    axes[0].set_ylabel("missing (%)")
    axes[0].set_title(f"Missing values per sample ({block.get('source', 'current')} matrix, "
                      f"overall {_fmt(np.mean(sample_missing), 3)}%)")
    axes[0].legend(fontsize=7)
    values = feature_missing[np.isfinite(feature_missing)]
    if values.size:
        axes[1].hist(values, bins=40, color=PALETTE[1], alpha=0.8)
        axes[1].axvline(50.0, color="#c0392b", linestyle="--", linewidth=1.0,
                        label="50% (filter threshold)")
        axes[1].legend(fontsize=7)
    axes[1].set_xlabel("missing (%) per feature")
    axes[1].set_ylabel("features")
    axes[1].set_title(f"Feature-level missingness (n={values.size})")
    return save_fig(fig, outdir, "qc_missing", title=None, dpi=dpi)


# ---------------------------------------------------------------------------
# 4/5. sample_corr / sample_dendro
# ---------------------------------------------------------------------------


def _corr_matrix(ctx, method: str):
    """返回 (相关矩阵, 矩阵来源)；在 log2(原始强度) 上算（对乘性稀释不敏感）。"""
    arr, source = _filled_log_matrix(ctx)
    if method in ("spearman", "spear", "rank"):
        arr = np.argsort(np.argsort(arr, axis=1, kind="stable"), axis=1, kind="stable")
        arr = arr.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(arr)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr, source


def _corr_method(params) -> str:
    method = str(_param(params, "method", "pearson") or "pearson").strip().lower()
    if method in ("spear", "rank"):
        method = "spearman"
    if method not in ("pearson", "spearman"):
        raise bad_request(f"不支持的相关系数：{method}（可选 pearson / spearman）")
    return method


def _cluster_order(corr: np.ndarray):
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        distance = 1.0 - np.asarray(corr, dtype=np.float64)
        np.fill_diagonal(distance, 0.0)
        distance = np.clip((distance + distance.T) / 2.0, 0.0, 2.0)
        link = linkage(squareform(distance, checks=False), method="average")
        return np.asarray(leaves_list(link), dtype=int), link
    except Exception:  # noqa: BLE001
        return np.arange(corr.shape[0]), None


def plot_sample_corr(ctx, params, outdir):
    """样本×样本相关热图（Pearson/Spearman；层次聚类排序；>200 样本自动下采样）。"""
    params = dict(params or {})
    method = _corr_method(params)
    corr, source = _corr_matrix(ctx, method)
    samples = _samples(ctx)
    labels, title_label = _color_labels(ctx, params, _param(params, "color_by", "group"))
    dpi = _dpi(params)
    order, _link = _cluster_order(corr)
    if order.size > MAX_HEATMAP_SAMPLES:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(order.size, size=MAX_HEATMAP_SAMPLES, replace=False))
        order = order[keep]
    shown = corr[np.ix_(order, order)]

    height = min(9.0, max(5.2, 0.055 * order.size + 3.4))
    fig, ax = new_fig(min(9.6, max(6.0, 0.07 * order.size + 3.0)), height)
    lo = float(np.nanmin(shown))
    hi = float(np.nanmax(shown))
    vmin = max(-1.0, float(np.floor(lo * 10) / 10))
    vmax = 1.0 if hi > 0.999 else float(np.ceil(hi * 10) / 10)
    if vmax - vmin < 0.02:
        vmax = min(1.0, vmin + 0.02)
    image = ax.imshow(shown, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    step = max(1, order.size // 20)
    ax.set_xticks(np.arange(0, order.size, step))
    ax.set_xticklabels([samples[index] for index in order[::step]], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(0, order.size, step))
    ax.set_yticklabels([samples[index] for index in order[::step]], fontsize=6)
    ax.grid(False)
    ax.set_title(f"Sample-sample {method} correlation on log2({'raw' if source == 'raw' else 'current'}) "
                 f"intensity (n={order.size}, color = {title_label})")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="correlation")
    return save_fig(fig, outdir, "sample_corr", title=None, dpi=dpi)


def plot_sample_dendro(ctx, params, outdir):
    """样本层次聚类树：距离 = 1 − 相关系数，average linkage；叶子按分组上色。"""
    params = dict(params or {})
    method = _corr_method(params)
    corr, source = _corr_matrix(ctx, method)
    samples = _samples(ctx)
    labels, title_label = _color_labels(ctx, params, _param(params, "color_by", "group"))
    dpi = _dpi(params)

    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except Exception as error:  # noqa: BLE001
        raise prerequisite(f"层次聚类需要 scipy（{error}）") from error

    distance = 1.0 - corr
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 2.0)
    link = linkage(squareform(distance, checks=False), method="average")

    width = min(13.0, max(6.4, 0.16 * len(samples) + 3.4))
    fig, ax = new_fig(width, 5.2)
    try:
        tree = dendrogram(link, labels=samples, ax=ax,
                          leaf_rotation=90 if len(samples) > 20 else 0,
                          leaf_font_size=6 if len(samples) > 40 else 8,
                          color_threshold=0.0, above_threshold_color="#9aa1ab")
    except Exception as error:  # noqa: BLE001
        raise prerequisite(f"层次聚类绘图失败（{error}）") from error
    ax.set_ylabel("distance (1 - correlation)")
    ax.set_title(f"Sample clustering (1 - {method} correlation on log2 "
                 f"{'raw' if source == 'raw' else 'current'} intensity, average linkage)")
    ax.grid(False)
    if labels is not None:
        colors = _cat_colors(labels)
        mapping = {sample: str(labels[index]) for index, sample in enumerate(samples)}
        for tick in ax.get_xticklabels():
            group = mapping.get(tick.get_text())
            if group in colors:
                tick.set_color(colors[group])
        ax.legend(handles=[Patch(facecolor=colors[label], label=label)
                           for label in _sorted_labels(labels)],
                  fontsize=7, loc="upper right", title=f"color = {title_label}",
                  title_fontsize=7)
    ax.margins(x=0.01)
    return save_fig(fig, outdir, "sample_dendro", title=None, dpi=dpi)


# ---------------------------------------------------------------------------
# 6/7/8. pca_scores / pca_variance / hotelling_t2
# ---------------------------------------------------------------------------


def plot_pca_scores(ctx, params, outdir):
    """PC1–PC2 得分图：分组上色 + 组内 95% Hotelling 置信椭圆。"""
    params = dict(params or {})
    core = _pca_from_ctx(ctx, n_comp=max(_as_int(params, "n_comp", 3), 3))
    scores = np.asarray(core["scores"], dtype=np.float64)
    ratio = np.asarray(core["ratio"], dtype=np.float64)
    labels, title_label = _color_labels(ctx, params, _param(params, "color_by", "group"))
    dpi = _dpi(params)

    fig, ax = new_fig(6.8, 5.2)
    colors = _cat_colors(labels) if labels is not None else {}
    _scatter_groups(ax, scores[:, 0], scores[:, 1], labels, colors,
                    point_size=_as_float(params, "point_size", 28.0))
    _draw_ellipses(ax, scores[:, 0], scores[:, 1], labels, colors)
    pc1 = float(ratio[0] * 100.0) if ratio.size else 0.0
    pc2 = float(ratio[1] * 100.0) if ratio.size > 1 else 0.0
    ax.axhline(0.0, color="#dfe3e8", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#dfe3e8", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title(f"PCA scores (color = {title_label}; dashed = 95% ellipse)")
    return save_fig(fig, outdir, "pca_scores", title=None, dpi=dpi)


def plot_pca_variance(ctx, params, outdir):
    """PCA 碎石图：单主成分解释方差柱 + 累计折线。"""
    params = dict(params or {})
    core = _pca_from_ctx(ctx, n_comp=max(_as_int(params, "n_comp", 3), 3))
    ratio = np.asarray(core["ratio"], dtype=np.float64)
    if ratio.size == 0:
        raise prerequisite("PCA 没有解释方差信息，请重跑「主成分分析」步骤")
    dpi = _dpi(params)
    n_show = int(min(ratio.size, max(_as_int(params, "top_n", 15), 2)))
    positions = np.arange(1, n_show + 1)
    fig, ax = new_fig(7.2, 4.1)
    ax.bar(positions, ratio[:n_show] * 100, color=PALETTE[0], width=0.7, alpha=0.9,
           label="individual")
    ax.plot(positions, np.cumsum(ratio[:n_show]) * 100, color="#e15759", marker="o",
            markersize=3, linewidth=1.2, label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance (%)")
    ax.set_xticks(positions)
    ax.set_title(f"PCA scree plot (PC1 = {ratio[0] * 100:.1f}%, "
                 f"first {n_show} = {float(np.sum(ratio[:n_show])) * 100:.1f}%)")
    ax.legend(fontsize=8)
    return save_fig(fig, outdir, "pca_variance", title=None, dpi=dpi)


def plot_hotelling_t2(ctx, params, outdir):
    """Hotelling T² 与 DModX 离群图（95%/99% 控制限；Wold 1994 SIMCA 语义）。"""
    params = dict(params or {})
    core = _pca_from_ctx(ctx, n_comp=max(_as_int(params, "n_comp", 3), 3))
    scores = np.asarray(core["scores"], dtype=np.float64)
    t2, t2_crit95, t2_crit99 = _hotelling_t2(scores)
    sse = core.get("sse")
    if sse is None:
        sse = _residual_sse(ctx, scores, core.get("columns"))
    columns = core.get("columns")
    n_features = len(columns) if columns is not None else _X(ctx).shape[1]
    dmodx, dmodx_crit = _dmodx(np.asarray(sse, dtype=np.float64), n_features, scores.shape[1])
    dpi = _dpi(params)
    samples = _samples(ctx)

    fig, axes = new_fig(11.2, 4.3, ncols=2)
    axes = np.atleast_1d(axes).ravel()
    positions = np.arange(t2.size)
    colors = np.where(t2 > t2_crit99, "#c0392b", np.where(t2 > t2_crit95, "#e67e22", "#4e79a7"))
    axes[0].bar(positions, t2, color=colors, width=0.86, linewidth=0)
    if np.isfinite(t2_crit95):
        axes[0].axhline(t2_crit95, color="#e67e22", linestyle="--", linewidth=1.0,
                        label=f"95% limit {_fmt(t2_crit95, 3)}")
    if np.isfinite(t2_crit99):
        axes[0].axhline(t2_crit99, color="#c0392b", linestyle=":", linewidth=1.0,
                        label=f"99% limit {_fmt(t2_crit99, 3)}")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("Hotelling T$^2$")
    axes[0].set_title(f"Hotelling T$^2$ ({int((t2 > t2_crit95).sum())} over 95% limit)")
    axes[0].legend(fontsize=7, loc="upper left")
    colors2 = np.where(np.isfinite(dmodx_crit) & (dmodx > dmodx_crit), "#c0392b", "#76b7b2")
    axes[1].bar(positions, dmodx, color=colors2, width=0.86, linewidth=0)
    if np.isfinite(dmodx_crit):
        axes[1].axhline(dmodx_crit, color="#c0392b", linestyle="--", linewidth=1.0,
                        label=f"95% limit {_fmt(dmodx_crit, 3)}")
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("DModX")
    axes[1].set_title("DModX (model residual distance)")
    axes[1].legend(fontsize=7, loc="upper left")
    flagged = np.where((t2 > t2_crit95) | (np.isfinite(dmodx_crit) & (dmodx > dmodx_crit)))[0]
    if flagged.size:
        for ax in axes:
            ax.set_xticks(positions[::max(1, positions.size // 12)])
        axes[0].set_xlabel(f"sample index ({flagged.size} flagged, red/orange)", fontsize=9)
    return save_fig(fig, outdir, "hotelling_t2", title=None, dpi=dpi)


# ---------------------------------------------------------------------------
# 9. heatmap_top
# ---------------------------------------------------------------------------


def plot_heatmap_top(ctx, params, outdir):
    """Top 特征热图：行 z-score（±2.5），行层次聚类 + 列按分组块聚类，顶部加分组色带。"""
    params = dict(params or {})
    top_n = max(_as_int(params, "top_n", DEFAULT_TOP_N), 2)
    dpi = _dpi(params)
    X = _X(ctx)
    samples = _samples(ctx)
    labels, title_label = _color_labels(ctx, params, _param(params, "color_by", "group"))
    columns, selection = _top_feature_indices(ctx, top_n)

    matrix = np.asarray(X[:, columns], dtype=np.float64).T  # (features, samples)
    z = zscore_rows(matrix)
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=2.5, neginf=-2.5), -2.5, 2.5)

    sample_index = np.arange(len(samples))
    if sample_index.size > MAX_HEATMAP_SAMPLES:
        rng = np.random.default_rng(0)
        sample_index = np.sort(rng.choice(sample_index.size, size=MAX_HEATMAP_SAMPLES,
                                          replace=False))
    z = z[:, sample_index]
    kept_samples = [samples[index] for index in sample_index]
    kept_labels = (np.asarray([str(labels[index]) for index in sample_index], dtype=object)
                   if labels is not None else None)

    row_order = _hier_order(z)
    if kept_labels is not None:
        # 列：先按分组块（块的先后按标签排序，配色/图例稳定），块内再层次聚类
        col_order: list[int] = []
        for label in _sorted_labels(kept_labels):
            block = np.where(kept_labels == label)[0]
            if block.size > 2:
                block = block[_hier_order(z[:, block].T)]
            col_order.extend(int(index) for index in block)
        col_order = np.asarray(col_order, dtype=np.int64)
    else:
        col_order = _hier_order(z.T)

    shown = z[np.ix_(row_order, col_order)]
    n_rows, n_cols = shown.shape
    width = min(13.5, max(6.2, 0.15 * n_cols + 5.2))
    height = min(13.0, max(4.2, 0.26 * n_rows + 2.8))

    fig, axes = new_fig(width, height, nrows=2, ncols=2,
                        # 不传 wspace/hspace：显式间距会让 tight_layout 认为不兼容（见 save_fig）
                        gridspec_kw={"width_ratios": [50, 1], "height_ratios": [1.4, 24]})
    axes = np.asarray(axes)
    strip_ax, ax, cax = axes[0][0], axes[1][0], axes[1][1]
    axes[0][1].set_visible(False)

    image = ax.imshow(shown, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                      interpolation="nearest")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_short(_feature_label(ctx, int(columns[index])), 26)
                        for index in row_order], fontsize=6)
    step = max(1, n_cols // 36)
    ticks = np.arange(0, n_cols, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_short(kept_samples[int(col_order[index])], 18) for index in ticks],
                       rotation=90, fontsize=5.6)
    ax.set_xlabel(f"samples (n={n_cols}, grouped then clustered)")
    ax.set_ylabel("features (row z-score)")
    ax.set_title(f"Top features heatmap ({selection}; row z-score, hierarchical order)")
    ax.grid(False)
    ax.set_xlim(-0.5, n_cols - 0.5)

    if kept_labels is not None:
        ordered_labels = kept_labels[col_order]
        classes = _sorted_labels(ordered_labels)
        colors = _cat_colors(ordered_labels)
        codes = np.array([classes.index(str(value)) for value in ordered_labels], dtype=np.int64)
        strip_ax.imshow(codes[None, :], aspect="auto",
                        cmap=ListedColormap([colors[label] for label in classes]),
                        vmin=-0.5, vmax=len(classes) - 0.5, interpolation="nearest")
        strip_ax.set_xlim(-0.5, n_cols - 0.5)
        strip_ax.set_xticks([])
        strip_ax.set_yticks([])
        strip_ax.grid(False)
        for label in classes:
            block = np.where(ordered_labels == label)[0]
            if block.size == 0:
                continue
            strip_ax.text(float(block.mean()), 0.0, _short(label, 14), ha="center", va="center",
                          fontsize=6.5, color=_contrast_text(colors[label]))
        strip_ax.set_title(f"group annotation (color = {title_label})", fontsize=7, pad=2)
    else:
        strip_ax.set_visible(False)

    bar = fig.colorbar(image, cax=cax)
    bar.set_label("row z-score", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    return save_fig(fig, outdir, "heatmap_top", title=None, dpi=dpi)


# ---------------------------------------------------------------------------
# 10. batch_pca
# ---------------------------------------------------------------------------


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """单变量 eta²（组间平方和 / 总平方和）——批次解释度。"""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray([str(item) for item in labels], dtype=object)
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for label in _sorted_labels(labels):
        mask = labels == label
        if mask.any():
            between += float(mask.sum()) * float((values[mask].mean() - values.mean()) ** 2)
    return float(np.clip(between / total, 0.0, 1.0))


def plot_batch_pca(ctx, params, outdir):
    """批次 vs 分组 PCA 对照：看批次是否主导结构（批次混杂诊断）。"""
    params = dict(params or {})
    batch_labels, batch_column = _batch_labels(ctx, params)
    if batch_labels is None:
        raise prerequisite(
            "样本信息里没有批次列：请在样本表里加一列 batch（或 plate/run）后再看批次 PCA")
    core = _pca_from_ctx(ctx, n_comp=max(_as_int(params, "n_comp", 3), 3))
    scores = np.asarray(core["scores"], dtype=np.float64)
    ratio = np.asarray(core["ratio"], dtype=np.float64)
    group_labels, group_column = _color_labels(ctx, params, _param(params, "color_by", "group"))
    dpi = _dpi(params)
    n_pcs = int(min(scores.shape[1], 5))
    batch_eta = float(np.mean([_eta_squared(scores[:, comp], batch_labels)
                               for comp in range(n_pcs)]))
    group_eta = None
    if group_labels is not None:
        group_eta = float(np.mean([_eta_squared(scores[:, comp], group_labels)
                                   for comp in range(n_pcs)]))

    fig, axes = new_fig(11.6, 5.2, ncols=2)
    axes = np.atleast_1d(axes).ravel()
    batch_colors = _cat_colors(batch_labels)
    _scatter_groups(axes[0], scores[:, 0], scores[:, 1], batch_labels, batch_colors,
                    point_size=28.0, count_label=False)
    _draw_ellipses(axes[0], scores[:, 0], scores[:, 1], batch_labels, batch_colors)
    axes[0].set_title(f"PCA by {batch_column} (batch eta2 = {batch_eta:.2f})")
    if group_labels is not None:
        group_colors = _cat_colors(group_labels)
        _scatter_groups(axes[1], scores[:, 0], scores[:, 1], group_labels, group_colors,
                        point_size=28.0)
        _draw_ellipses(axes[1], scores[:, 0], scores[:, 1], group_labels, group_colors)
        axes[1].set_title(f"PCA by {group_column} (group eta2 = {group_eta:.2f})")
    else:
        axes[1].text(0.5, 0.5, "no group column", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=9, color="#8a919c")
        axes[1].set_title("PCA by group (not available)")
    pc1 = float(ratio[0] * 100.0) if ratio.size else 0.0
    pc2 = float(ratio[1] * 100.0) if ratio.size > 1 else 0.0
    for ax in axes:
        ax.axhline(0.0, color="#dfe3e8", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="#dfe3e8", linewidth=0.8, zorder=0)
        ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    if batch_eta > 0.5:
        axes[0].text(0.02, 0.02, "batch dominates structure", transform=axes[0].transAxes,
                     fontsize=7, color="#c0392b", va="bottom")
    return save_fig(fig, outdir, "batch_pca", title=None, dpi=dpi)


# ---------------------------------------------------------------------------
# PLOTS 注册表（kind 必须出现在 DESIGN §5 白名单里；每个 kind 只能一个 owner）
# ---------------------------------------------------------------------------

PLOTS = {
    "qc_tic": plot_qc_tic,
    "qc_rsd": plot_qc_rsd,
    "qc_missing": plot_qc_missing,
    "sample_corr": plot_sample_corr,
    "sample_dendro": plot_sample_dendro,
    "pca_scores": plot_pca_scores,
    "pca_variance": plot_pca_variance,
    "hotelling_t2": plot_hotelling_t2,
    "heatmap_top": plot_heatmap_top,
    "batch_pca": plot_batch_pca,
}

__all__ = ["PLOTS"]
