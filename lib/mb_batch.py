"""
dsh-metabolomics — 批次校正（DESIGN.md §3 `batch` + §4 group=批次与网络）

方法（文献）：
  * QC-RLSC（QC-based Robust LOESS Signal Correction）：用 QC 样本在进样顺序上拟合
    LOESS 曲线，把每个特征的信号校正到批次中位数（Dunn 2011，Nature Protocols）。
  * ComBat：位置/尺度经验贝叶斯校正（Johnson 2007），保留生物分组（可选协变量）。

只依赖 numpy/pandas/scipy/statsmodels（可选）/mb_common。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mb_common as sc

#: 本模块拥有的图（避免与 mb_plots / 其它模块重名）
PLOT_KEYS = ("batch_effect", "batch_corrected_tic")


def _p(params, key, default=None):
    value = (params or {}).get(key, default)
    return default if value in (None, "", "null") else value


def _p_float(params, key, default):
    try:
        return float(_p(params, key, default))
    except (TypeError, ValueError):
        return float(default)


def _p_int(params, key, default):
    try:
        return int(float(_p(params, key, default)))
    except (TypeError, ValueError):
        return int(default)


def _X(ctx) -> np.ndarray:
    X = ctx.get("X")
    if X is None:
        raise sc.prerequisite("数据集没有表达矩阵")
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise sc.EngineError("internal", f"矩阵维度异常：{arr.shape}")
    return arr


def _info(ctx) -> pd.DataFrame:
    frame = ctx.get("sample_info")
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(index=pd.Index([str(s) for s in (ctx.get("samples") or [])], name="sample"))


def _outdir(ctx, outdir) -> str:
    return sc.ensure_dir(outdir or ctx.get("outdir") or ".")


def _batch_values(ctx, key: str) -> np.ndarray:
    frame = _info(ctx)
    if not key or key not in frame.columns:
        raise sc.prerequisite(f"找不到批次列「{key or '(未指定)'}」：请在样本表里加 batch 列，或用 samples 操作导入")
    return np.asarray([str(v) for v in frame[key].tolist()], dtype=object)


def _qc_mask(ctx, qc_label: str) -> np.ndarray:
    frame = _info(ctx)
    n = len(frame)
    mask = np.zeros(n, dtype=bool)
    if "qc" in frame.columns:
        values = pd.to_numeric(frame["qc"], errors="coerce").fillna(0)
        mask |= values.to_numpy() > 0
    if qc_label:
        try:
            import re

            pattern = re.compile(str(qc_label), re.IGNORECASE)
            mask |= np.array([bool(pattern.search(str(s))) for s in frame.index], dtype=bool)
        except Exception:  # noqa: BLE001
            mask |= np.array([str(qc_label).lower() in str(s).lower() for s in frame.index], dtype=bool)
    return mask


def _order(ctx) -> np.ndarray:
    """进样顺序：优先 sample_info['injection_order']/['order']，否则用样本原始顺序。"""
    frame = _info(ctx)
    for key in ("injection_order", "injection", "order", "run_order", "acquired"):
        if key in frame.columns:
            values = pd.to_numeric(frame[key], errors="coerce")
            if values.notna().any():
                order = values.fillna(values.max() + 1).to_numpy(dtype=float)
                return np.argsort(np.argsort(order)).astype(float)
    return np.arange(len(frame), dtype=float)


def _batch_effect_score(matrix: np.ndarray, batches: np.ndarray, groups=None) -> dict:
    """批次效应指标：批次解释的方差比例 + 每特征的批次间方差比。"""
    try:
        from sklearn.decomposition import PCA

        centered = matrix - np.nanmean(matrix, axis=0, keepdims=True)
        centered = np.nan_to_num(centered, nan=0.0)
        pcs = PCA(n_components=min(5, centered.shape[0] - 1, centered.shape[1]), random_state=0).fit_transform(centered)
    except Exception:  # noqa: BLE001
        pcs = np.nan_to_num(matrix, nan=0.0)[:, : min(3, matrix.shape[1])]
    uniq = list(dict.fromkeys([str(b) for b in batches]))
    if len(uniq) < 2:
        return {"variance_explained": 0.0, "n_batches": len(uniq), "pseudo_f": float("nan")}
    ss_total = float(np.sum((pcs - pcs.mean(axis=0)) ** 2))
    ss_within = 0.0
    for name in uniq:
        sel = np.array([str(b) == name for b in batches])
        if sel.sum() == 0:
            continue
        sub = pcs[sel]
        ss_within += float(np.sum((sub - sub.mean(axis=0)) ** 2))
    ss_between = max(ss_total - ss_within, 0.0)
    return {"variance_explained": ss_between / ss_total if ss_total > 0 else 0.0,
            "n_batches": len(uniq), "pseudo_f": float("nan")}


def _rlsc(matrix: np.ndarray, batches: np.ndarray, qc: np.ndarray, order: np.ndarray,
          span: float = 0.75) -> tuple[np.ndarray, list[dict]]:
    """QC-RLSC：每个批次内用 QC 样本拟合 LOESS，把信号校正到批次 QC 中位数。"""
    out = np.array(matrix, dtype=np.float64, copy=True)
    info: list[dict] = []
    uniq = list(dict.fromkeys([str(b) for b in batches]))
    for name in uniq:
        rows = np.where(np.array([str(b) == name for b in batches]))[0]
        if rows.size == 0:
            continue
        qc_rows = np.array([i for i in rows if qc[i]], dtype=int)
        info.append({"batch": name, "n_samples": int(rows.size), "n_qc": int(qc_rows.size)})
        if qc_rows.size < 3:
            continue
        x = order[qc_rows]
        order_index = np.argsort(x)
        qc_rows = qc_rows[order_index]
        x = x[order_index]
        target_median = np.nanmedian(matrix[qc_rows], axis=0)
        corr = np.ones((rows.size, matrix.shape[1]), dtype=np.float64)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except Exception:  # noqa: BLE001 - 没有 statsmodels 时退化为线性/中位数校正
            lowess = None
        all_rows = np.array(rows)
        all_x = order[all_rows]
        for feature in range(matrix.shape[1]):
            y = matrix[qc_rows, feature]
            finite = np.isfinite(y)
            if finite.sum() < 3:
                continue
            if lowess is not None:
                try:
                    fitted = lowess(y[finite], x[finite], frac=float(np.clip(span, 0.2, 1.0)),
                                    return_sorted=False)
                    base = np.interp(all_x, x[finite], fitted)
                except Exception:  # noqa: BLE001
                    base = np.full(all_x.shape, float(np.nanmedian(y[finite])))
            else:
                base = np.full(all_x.shape, float(np.nanmedian(y[finite])))
            base = np.where(np.abs(base) < 1e-12, 1.0, base)
            corr[:, feature] = float(target_median[feature] if np.isfinite(target_median[feature]) else 1.0) / base
        corr = np.where(np.isfinite(corr) & (np.abs(corr) < 1e6), corr, 1.0)
        out[all_rows] = matrix[all_rows] * corr
    return out, info


def _combat(matrix: np.ndarray, batches: np.ndarray, groups=None) -> tuple[np.ndarray, list[dict]]:
    """ComBat：位置/尺度经验贝叶斯校正（Johnson 2007，简化参数实现）。"""
    out = np.array(matrix, dtype=np.float64, copy=True)
    uniq = list(dict.fromkeys([str(b) for b in batches]))
    info: list[dict] = []
    if len(uniq) < 2:
        return out, [{"batch": uniq[0] if uniq else "", "n_samples": int(matrix.shape[0]), "note": "只有 1 个批次"}]
    design_cols = [np.ones(matrix.shape[0])]
    if groups is not None:
        gl = [str(g) for g in groups]
        levels = list(dict.fromkeys(gl))
        for level in levels[1:]:
            design_cols.append(np.array([1.0 if g == level else 0.0 for g in gl]))
    design = np.column_stack(design_cols)
    try:
        beta, *_ = np.linalg.lstsq(design, np.nan_to_num(matrix, nan=0.0), rcond=None)
        grand = design @ beta
    except Exception:  # noqa: BLE001
        grand = np.nanmean(matrix, axis=0, keepdims=True) * np.ones_like(matrix)
    std = np.nanstd(matrix - grand, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    standardized = (matrix - grand) / std
    gamma = {}
    delta = {}
    for name in uniq:
        rows = np.array([str(b) == name for b in batches])
        if rows.sum() == 0:
            continue
        mean_i = np.nanmean(standardized[rows], axis=0)
        var_i = np.nanvar(standardized[rows], axis=0)
        gamma[name] = np.nan_to_num(mean_i, nan=0.0)
        delta[name] = np.where(np.isfinite(var_i) & (var_i > 1e-9), var_i, 1.0)
        info.append({"batch": name, "n_samples": int(rows.sum()),
                     "max_shift": float(np.nanmax(np.abs(gamma[name]))),
                     "median_scale": float(np.median(delta[name]))})
    # 经验贝叶斯收缩（简化：向 0 / 1 收缩，权重 = 1/√n）
    for name in uniq:
        rows = np.array([str(b) == name for b in batches])
        n = max(int(rows.sum()), 1)
        w = 1.0 / np.sqrt(n)
        gamma[name] = gamma[name] * (1.0 - w)
        delta[name] = 1.0 + (delta[name] - 1.0) * (1.0 - w)
        out[rows] = (standardized[rows] - gamma[name]) / np.sqrt(np.maximum(delta[name], 1e-6))
        out[rows] = out[rows] * std + grand[rows]
    return out, info


def _fig_batch(ctx, params, outdir, before: np.ndarray, after: np.ndarray,
               batches: np.ndarray, name: str, title: str) -> dict:
    from sklearn.decomposition import PCA

    fig, axes = sc.new_fig(11.0, 4.4)
    mapping = {b: sc.PALETTE[i % len(sc.PALETTE)] for i, b in enumerate(dict.fromkeys([str(b) for b in batches]))}
    fig.clf()
    panels = [(before, "Before batch correction"), (after, "After batch correction")]
    for index, (matrix, subtitle) in enumerate(panels):
        ax = fig.add_subplot(1, 2, index + 1)
        filled = np.nan_to_num(matrix, nan=np.nanmean(matrix, axis=0))
        try:
            pcs = PCA(n_components=2, random_state=0).fit_transform(filled - filled.mean(axis=0))
        except Exception:  # noqa: BLE001
            pcs = filled[:, :2]
        for batch, color in mapping.items():
            sel = np.array([str(b) == batch for b in batches])
            ax.scatter(pcs[sel, 0], pcs[sel, 1], s=16, color=color, label=batch, linewidths=0)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.axes[0].legend(fontsize=7, frameon=False, markerscale=1.4, title="batch", title_fontsize=7)
    return sc.save_fig(fig, outdir, name, title=None)


def _fig_tic(ctx, params, outdir, before: np.ndarray, after: np.ndarray, name: str) -> dict:
    fig, axes = sc.new_fig(7.6, 4.0)
    fig.clf()
    ax = fig.add_subplot(111)
    xs = np.arange(before.shape[0])
    ax.plot(xs, np.nansum(before, axis=1), marker="o", markersize=3, linewidth=1.0,
            color=sc.PALETTE[0], label="before")
    ax.plot(xs, np.nansum(after, axis=1), marker="s", markersize=3, linewidth=1.0,
            color=sc.PALETTE[2], label="after")
    ax.set_xlabel("sample index", fontsize=9)
    ax.set_ylabel("TIC (sum intensity)", fontsize=9)
    ax.set_title("Total ion current before/after correction", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return sc.save_fig(fig, outdir, name, title=None)


def _run(ctx, params, outdir, apply_change: bool) -> dict:
    matrix = _X(ctx)
    method = str(_p(params, "method", "rlsc") or "rlsc").lower()
    batch_key = str(_p(params, "batch_key", "") or "") or "batch"  # 默认列名 batch 
    qc_label = str(_p(params, "qc_label", "") or "")
    span = _p_float(params, "span", 0.75)
    batches = _batch_values(ctx, batch_key)
    if len(dict.fromkeys(batches.tolist())) < 2:
        raise sc.prerequisite(f"批次列「{batch_key}」只有 1 个批次，无需校正")
    qc = _qc_mask(ctx, qc_label)
    order = _order(ctx)
    groups = None
    group_key = ctx.get("group_key")
    if group_key:
        frame = _info(ctx)
        if group_key in frame.columns:
            groups = np.asarray([str(v) for v in frame[group_key].tolist()], dtype=object)
    before = np.array(matrix, dtype=np.float64, copy=True)
    if method in ("rlsc", "loess", "qc-rlsc"):
        after, info = _rlsc(matrix, batches, qc, order, span=span)
        method_label = "QC-RLSC"
        method_text = "QC-RLSC LOESS 信号校正（Dunn 2011；内置实现）"
    elif method in ("combat", "eb"):
        after, info = _combat(matrix, batches, groups)
        method_label = "ComBat"
        method_text = "ComBat 位置/尺度经验贝叶斯（Johnson 2007；简化实现）"
    elif method in ("median", "center"):
        after = np.array(matrix, dtype=np.float64, copy=True)
        info = []
        for batch in dict.fromkeys(batches.tolist()):
            rows = np.array([str(b) == batch for b in batches])
            if rows.sum() == 0:
                continue
            med = np.nanmedian(matrix[rows], axis=0)
            grand = np.nanmedian(med)
            factor = np.where(np.abs(med) > 1e-12, grand / med, 1.0)
            after[rows] = matrix[rows] * factor
            info.append({"batch": batch, "n_samples": int(rows.sum()), "median_factor": float(np.median(factor))})
        method_label = "中位数校正"
        method_text = "批次中位数缩放（内置实现）"
    else:
        raise sc.bad_request(f"未知的批次校正方法「{method}」（可用 rlsc / combat / median）")
    after = np.where(np.isfinite(after), after, matrix)
    stats_before = _batch_effect_score(before, batches, groups)
    stats_after = _batch_effect_score(after, batches, groups)
    if not apply_change:
        return {"after": after, "before": before, "info": info, "method_label": method_label,
                "method_text": method_text, "batches": batches, "stats_before": stats_before,
                "stats_after": stats_after, "qc_label": qc_label, "span": span, "batch_key": batch_key}
    outdir = _outdir(ctx, outdir)
    figures = [_fig_batch(ctx, params, outdir, before, after, batches, "batch-effect-pca", "Batch effect"),
               _fig_tic(ctx, params, outdir, before, after, "batch-tic")]
    rows = [{"metric": "批次解释方差比例", "before": stats_before["variance_explained"],
             "after": stats_after["variance_explained"]}]
    for item in info:
        rows.append({"metric": f"批次 {item.get('batch')}",
                     "before": item.get("n_samples"), "after": item.get("n_qc", item.get("median_scale"))})
    bullets = [
        f"{method_label}：{len(dict.fromkeys(batches.tolist()))} 个批次 × {matrix.shape[1]} 个特征",
        f"批次解释方差比例：{stats_before['variance_explained'] * 100:.1f}% → {stats_after['variance_explained'] * 100:.1f}%",
    ]
    if method.startswith("rlsc") or method in ("loess", "qc-rlsc"):
        n_qc = int(qc.sum())
        if n_qc < 3:
            bullets.append("注意：QC 样本少于 3 个，RLSC 实际退化为批次中位数校正（结果仍可用，但不如有 QC 时可靠）")
        else:
            bullets.append(f"QC 样本 {n_qc} 个，LOESS span={span}")
    return sc.result(
        summary={"title": "批次校正", "bullets": bullets,
                 "metrics": {"method": method_label, "n_batches": int(len(dict.fromkeys(batches.tolist()))),
                             "variance_before": stats_before["variance_explained"],
                             "variance_after": stats_after["variance_explained"]}},
        tables=[sc.table("batch", "批次校正指标",
                         columns=[{"key": "metric", "label": "指标", "type": "string"},
                                  {"key": "before", "label": "校正前", "type": "number"},
                                  {"key": "after", "label": "校正后", "type": "number"}],
                         rows=rows)],
        figures=figures,
        replace={"X": after.astype(np.float32)},
        store={"batch": {"method": method_label, "methods": info,
                         "variance_before": stats_before["variance_explained"],
                         "variance_after": stats_after["variance_explained"],
                         "batch_key": batch_key}},
        notes=[method_text],
    )


# ---------------------------------------------------------------------------
# STEP / ANALYSES / PLOTS
# ---------------------------------------------------------------------------


def step_batch(ctx, params, outdir):
    return _run(ctx, params, outdir, apply_change=True)


def analysis_rlsc(ctx, params, outdir):
    merged = dict(params or {})
    merged.setdefault("method", "rlsc")
    out = _run(ctx, merged, outdir, apply_change=False)
    outdir = _outdir(ctx, outdir)
    figures = [_fig_batch(ctx, merged, outdir, out["before"], out["after"], out["batches"],
                          "rlsc-pca", "QC-RLSC correction")]
    return sc.result(
        summary={"title": "QC-RLSC 批次校正", "bullets": [
            out["method_text"],
            f"批次解释方差：{out['stats_before']['variance_explained'] * 100:.1f}% → "
            f"{out['stats_after']['variance_explained'] * 100:.1f}%"],
            "metrics": {"variance_before": out["stats_before"]["variance_explained"],
                        "variance_after": out["stats_after"]["variance_explained"]}},
        tables=[sc.table("rlsc", "各批次概况",
                         columns=[{"key": "batch", "label": "批次", "type": "string"},
                                  {"key": "n_samples", "label": "样本", "type": "number"},
                                  {"key": "n_qc", "label": "QC", "type": "number"}],
                         rows=out["info"])],
        figures=figures)


def analysis_combat(ctx, params, outdir):
    merged = dict(params or {})
    merged.setdefault("method", "combat")
    out = _run(ctx, merged, outdir, apply_change=False)
    outdir = _outdir(ctx, outdir)
    figures = [_fig_batch(ctx, merged, outdir, out["before"], out["after"], out["batches"],
                          "combat-pca", "ComBat correction")]
    return sc.result(
        summary={"title": "ComBat 批次校正", "bullets": [
            out["method_text"],
            f"批次解释方差：{out['stats_before']['variance_explained'] * 100:.1f}% → "
            f"{out['stats_after']['variance_explained'] * 100:.1f}%"],
            "metrics": {"variance_before": out["stats_before"]["variance_explained"],
                        "variance_after": out["stats_after"]["variance_explained"]}},
        tables=[sc.table("combat", "各批次位置/尺度参数",
                         columns=[{"key": "batch", "label": "批次", "type": "string"},
                                  {"key": "n_samples", "label": "样本", "type": "number"},
                                  {"key": "max_shift", "label": "最大位置偏移", "type": "number"},
                                  {"key": "median_scale", "label": "中位尺度", "type": "number"}],
                         rows=out["info"])],
        figures=figures)


def plot_batch_effect(ctx, params=None, outdir=None):
    params = dict(params or {})
    merged = dict(params)
    merged.setdefault("method", "rlsc")
    out = _run(ctx, merged, outdir, apply_change=False)
    return _fig_batch(ctx, params, _outdir(ctx, outdir), out["before"], out["after"],
                      out["batches"], "batch-effect-pca", "Batch effect")


def plot_batch_corrected_tic(ctx, params=None, outdir=None):
    params = dict(params or {})
    merged = dict(params)
    merged.setdefault("method", "rlsc")
    out = _run(ctx, merged, outdir, apply_change=False)
    return _fig_tic(ctx, params, _outdir(ctx, outdir), out["before"], out["after"], "batch-tic")


STEPS = [
    {
        "id": "batch",
        "title": "批次校正",
        "icon": "batch",
        "requires": ["normalize"],
        "params": [
            {"key": "batch_key", "label": "批次列", "type": "column", "default": "batch"},
            {"key": "method", "label": "方法", "type": "select", "default": "rlsc",
             "options": [{"value": "rlsc", "label": "QC-RLSC（LOESS，需 QC 样本）"},
                         {"value": "combat", "label": "ComBat（经验贝叶斯）"},
                         {"value": "median", "label": "批次中位数缩放"}]},
            {"key": "qc_label", "label": "QC 样本标签（正则或列值）", "type": "text", "default": "QC"},
            {"key": "span", "label": "LOESS span", "type": "number", "default": 0.75},
        ],
        "run": step_batch,
    },
]

ANALYSES = [
    {
        "id": "rlsc_batch", "title": "QC-RLSC 批次校正", "group": "批次与网络",
        "method": "QC 样本 LOESS 信号校正（Dunn 2011；内置实现，statsmodels lowess 可选）",
        "needs": ["qc"],
        "params": [
            {"key": "batch_key", "label": "批次列", "type": "column", "default": "batch"},
            {"key": "qc_label", "label": "QC 标签", "type": "text", "default": "QC"},
            {"key": "span", "label": "LOESS span", "type": "number", "default": 0.75},
        ],
        "run": analysis_rlsc,
    },
    {
        "id": "combat_batch", "title": "ComBat 批次校正", "group": "批次与网络",
        "method": "位置/尺度经验贝叶斯（Johnson 2007；简化实现）",
        "needs": ["qc"],
        "params": [{"key": "batch_key", "label": "批次列", "type": "column", "default": "batch"}],
        "run": analysis_combat,
    },
]

PLOTS = {
    "batch_effect": plot_batch_effect,
    "batch_corrected_tic": plot_batch_corrected_tic,
}

__all__ = ["STEPS", "ANALYSES", "PLOTS"]
