"""
dsh-metabolomics — 共表达网络与代谢物模块（DESIGN.md §4 group=批次与网络）

方法（文献）：
  * 代谢物共表达网络：|Pearson/Spearman r| 阈值建图（代谢组学网络分析的常用做法，如
    MetaboAnalyst 的 correlation network；Xia & Wishart 2011）。
  * 代谢物模块：层次聚类切分 + 模块特征值（module eigengene），借鉴 WGCNA 的思路
    （Langfelder & Horvath 2008），但用简化相关度量，不声称等同 WGCNA。
  * 模块-表型相关：模块特征值与样本性状的 Pearson 相关 + BH 校正。

只依赖 numpy/pandas/scipy（networkx 可选）/mb_common。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mb_common as sc

#: 本模块拥有的图（避免与其它模块重名）
PLOT_KEYS = ("corr_heatmap", "corr_network")


def _p(params, key, default=None):
    value = (params or {}).get(key, default)
    return default if value in (None, "", "null") else value


def _p_int(params, key, default):
    try:
        return int(float(_p(params, key, default)))
    except (TypeError, ValueError):
        return int(default)


def _p_float(params, key, default):
    try:
        return float(_p(params, key, default))
    except (TypeError, ValueError):
        return float(default)


def _outdir(ctx, outdir) -> str:
    return sc.ensure_dir(outdir or ctx.get("outdir") or ".")


def _features(ctx) -> list[str]:
    return [str(item) for item in (ctx.get("features") or [])]


def _X(ctx) -> np.ndarray:
    X = ctx.get("X")
    if X is None:
        raise sc.prerequisite("数据集没有表达矩阵")
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise sc.EngineError("internal", f"矩阵维度异常：{arr.shape}")
    return arr


def _info_frame(ctx) -> pd.DataFrame:
    frame = ctx.get("feature_info")
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(index=pd.Index(_features(ctx), name="feature"))


def _select_matrix(ctx, max_features: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """挑方差最大的 max_features 个特征（返回 样本×特征 子矩阵 + 特征名 + 原始索引）。"""
    X = _X(ctx)
    names = _features(ctx)
    if X.shape[1] != len(names):
        names = [f"F{i + 1}" for i in range(X.shape[1])]
    filled = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    var = np.nanvar(filled, axis=0)
    order = np.argsort(-np.nan_to_num(var, nan=0.0))
    keep = np.sort(order[: max(5, min(int(max_features), X.shape[1]))])
    return filled[:, keep], [names[i] for i in keep], keep


def _corr(matrix: np.ndarray, method: str) -> np.ndarray:
    method = str(method or "pearson").lower()
    if method in ("spearman", "rank"):
        ranks = np.apply_along_axis(lambda col: pd.Series(col).rank().to_numpy(), 0, matrix)
        corr = np.corrcoef(ranks, rowvar=False)
    else:
        corr = np.corrcoef(matrix, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def _modules(corr: np.ndarray, threshold: float, min_size: int) -> tuple[np.ndarray, dict]:
    """层次聚类切模块：先按 |r| 距离做 average linkage，再按高度/阈值切树。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    adjacency = np.abs(corr)
    np.fill_diagonal(adjacency, 1.0)
    distance = 1.0 - adjacency
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    # 用 1-threshold 作为切树高度：相关系数低于阈值就分开
    cut_height = max(1e-6, 1.0 - float(np.clip(threshold, 0.05, 0.99)))
    labels = fcluster(tree, t=cut_height, criterion="distance")
    # 小模块并入 -1（未分配）
    counts = pd.Series(labels).value_counts()
    small = {label for label, count in counts.items() if count < min_size}
    labels = np.array([-1 if label in small else label for label in labels], dtype=int)
    info: dict = {}
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue
        members = np.where(labels == label)[0]
        sub = adjacency[np.ix_(members, members)]
        info[int(label)] = {"size": int(members.size),
                            "mean_abs_r": float(np.mean(sub[np.triu_indices(members.size, 1)])) if members.size > 1 else float("nan")}
    return labels, info


def _eigengene(matrix: np.ndarray, members: np.ndarray) -> np.ndarray:
    """模块特征值：成员特征标准化后的第一主成分得分。"""
    sub = matrix[:, members]
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0)
    sd[sd <= 1e-9] = 1.0
    z = (sub - mu) / sd
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=1, random_state=0).fit_transform(z)[:, 0]
    except Exception:  # noqa: BLE001
        return z.mean(axis=1)


def _network_table(corr: np.ndarray, names: list[str], threshold: float, top_n: int) -> list[dict]:
    rows = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            value = float(corr[i, j])
            if abs(value) >= threshold:
                rows.append({"feature_a": names[i], "feature_b": names[j],
                             "correlation": value, "abs_correlation": abs(value),
                             "direction": "positive" if value > 0 else "negative"})
    rows.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return rows[: max(50, top_n * 20)]


def _hubs(corr: np.ndarray, names: list[str], threshold: float, top_n: int) -> list[dict]:
    adjacency = (np.abs(corr) >= threshold).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    strength = (np.abs(corr) * (np.abs(corr) >= threshold)).sum(axis=1)
    rows = [{"feature": names[i], "degree": int(degree[i]), "strength": float(strength[i])}
            for i in range(len(names))]
    rows.sort(key=lambda item: (-item["degree"], -item["strength"]))
    return rows[: max(10, top_n)]


def _fig_network(corr: np.ndarray, names: list[str], labels: np.ndarray, threshold: float,
                 outdir: str, name: str = "corr-network") -> dict:
    fig, ax = sc.new_fig(7.4, 6.2)
    edges = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr[i, j]) >= threshold:
                edges.append((i, j, corr[i, j]))
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(len(names)))
        for i, j, value in edges:
            graph.add_edge(int(i), int(j), weight=float(abs(value)), sign=float(value))
        positions = nx.spring_layout(graph, seed=0, k=0.9) if graph.number_of_nodes() else {}
        xs = [positions.get(i, (0, 0))[0] for i in range(len(names))]
        ys = [positions.get(i, (0, 0))[1] for i in range(len(names))]
        for i, j, value in edges:
            if i in positions and j in positions:
                ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]],
                        color=("#d1495b" if value > 0 else "#2e6f95"),
                        linewidth=0.3 + 1.2 * abs(value) * 0.6, alpha=0.5, zorder=1)
        palette = {}
        for label in sorted(set(labels.tolist())):
            palette[label] = "#bab0ac" if label == -1 else sc.PALETTE[len(palette) % len(sc.PALETTE)]
        colors = [palette.get(int(label), "#bab0ac") for label in labels]
        ax.scatter(xs, ys, s=22, c=colors, zorder=2, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    except Exception:  # noqa: BLE001
        ax.hist([abs(value) for _i, _j, value in edges] or [0], bins=30, color=sc.PALETTE[0])
        ax.set_xlabel("|r|", fontsize=9); ax.set_ylabel("pairs", fontsize=9)
    ax.set_title(f"Metabolite co-expression network (|r| ≥ {threshold})", fontsize=10)
    return sc.save_fig(fig, outdir, name, title=None)


def _fig_heatmap(corr: np.ndarray, names: list[str], labels: np.ndarray, outdir: str,
                 name: str = "corr-heatmap", max_features: int = 80) -> dict:
    order = np.argsort(labels, kind="stable")[:max_features]
    sub = corr[np.ix_(order, order)]
    fig, ax = sc.new_fig(7.0, 6.2)
    image = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1)
    step = max(1, len(order) // 12)
    ax.set_xticks(np.arange(0, len(order), step))
    ax.set_xticklabels([names[i] for i in order[::step]], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(0, len(order), step))
    ax.set_yticklabels([names[i] for i in order[::step]], fontsize=6)
    ax.set_title("Metabolite correlation (ordered by module)", fontsize=10)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Pearson r", fontsize=8)
    return sc.save_fig(fig, outdir, name, title=None)


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------


def analysis_correlation_network(ctx, params, outdir):
    max_features = _p_int(params, "top_n", 800)
    threshold = _p_float(params, "threshold", 0.6)
    method = str(_p(params, "method", "pearson") or "pearson")
    matrix, names, _keep = _select_matrix(ctx, max_features)
    if matrix.shape[1] < 5:
        raise sc.prerequisite("特征少于 5 个，无法构建共表达网络")
    corr = _corr(matrix, method)
    labels, info = _modules(corr, threshold, _p_int(params, "min_size", 5))
    rows = _network_table(corr, names, threshold, _p_int(params, "top_n", 30))
    outdir = _outdir(ctx, outdir)
    figure = _fig_network(corr, names, labels, threshold, outdir)
    n_modules = len([label for label in set(labels.tolist()) if label != -1])
    bullets = [f"{len(names)} 个特征、|r|≥{threshold} 的边 {len(rows)} 条（{method}）",
               f"识别 {n_modules} 个模块（每模块 ≥{_p_int(params, 'min_size', 5)} 个特征）"]
    if rows:
        bullets.append("最强相关对：" + "、".join(
            f"{row['feature_a']}~{row['feature_b']}({row['correlation']:.2f})" for row in rows[:3]))
    return sc.result(
        summary={"title": "共表达网络", "bullets": bullets,
                 "metrics": {"n_features": len(names), "n_edges": len(rows),
                             "n_modules": n_modules, "threshold": threshold, "method": method}},
        tables=[sc.table("network", "共表达边（按 |r| 排序）",
                         columns=[{"key": "feature_a", "label": "特征 A", "type": "string"},
                                  {"key": "feature_b", "label": "特征 B", "type": "string"},
                                  {"key": "correlation", "label": "r", "type": "number"},
                                  {"key": "abs_correlation", "label": "|r|", "type": "number"},
                                  {"key": "direction", "label": "方向", "type": "string"}],
                         rows=rows)],
        figures=[figure],
        store={"network": {"names": names, "labels": labels.tolist(), "modules": info,
                           "n_edges": len(rows), "threshold": threshold}},
    )


def analysis_metabolite_modules(ctx, params, outdir):
    max_features = _p_int(params, "top_n", 800)
    threshold = _p_float(params, "threshold", 0.6)
    min_size = _p_int(params, "min_size", 5)
    matrix, names, _keep = _select_matrix(ctx, max_features)
    if matrix.shape[1] < 10:
        raise sc.prerequisite("特征太少（<10），无法做模块分析")
    corr = _corr(matrix, str(_p(params, "method", "pearson") or "pearson"))
    labels, info = _modules(corr, threshold, min_size)
    outdir = _outdir(ctx, outdir)
    if not info:
        raise sc.prerequisite(f"在 |r|≥{threshold} 下没有形成 ≥{min_size} 个特征的模块：请降低 threshold 或 min_size")
    # 模块成员表 + 特征值
    frame = _info_frame(ctx)
    rows = []
    eigengenes = {}
    for label, meta in sorted(info.items()):
        members = np.where(labels == label)[0]
        eigengenes[label] = _eigengene(matrix, members)
        for position in members:
            name = names[position]
            rows.append({"module": f"M{label}", "feature": name,
                         "name": frame["name"].get(name) if "name" in frame.columns else None,
                         "class": frame["class"].get(name) if "class" in frame.columns else None,
                         "module_size": meta["size"], "mean_abs_r": meta["mean_abs_r"]})
    if rows:
        frame_rows = pd.DataFrame(rows)
        path = sc.export_csv(frame_rows, outdir, "metabolite-modules.csv", index=False)
        data_out = [{"kind": "modules_csv", "path": path, "label": "代谢物模块成员表"}]
    else:
        data_out = []
    figure = _fig_heatmap(corr, names, labels, outdir)
    biggest = max(info.items(), key=lambda item: item[1]["size"])
    bullets = [f"{len(info)} 个模块，最大模块 M{biggest[0]}（{biggest[1]['size']} 个特征）",
               "模块内平均 |r|：" + "，".join(f"M{label}:{meta['mean_abs_r']:.2f}" for label, meta in list(info.items())[:5])]
    return sc.result(
        summary={"title": "代谢物模块", "bullets": bullets,
                 "metrics": {"n_modules": len(info), "largest": int(biggest[1]["size"]),
                             "threshold": threshold, "min_size": min_size}},
        tables=[sc.table("modules", "模块与成员",
                         columns=[{"key": "module", "label": "模块", "type": "string"},
                                  {"key": "feature", "label": "特征", "type": "string"},
                                  {"key": "name", "label": "名称", "type": "string"},
                                  {"key": "class", "label": "化学分类", "type": "string"},
                                  {"key": "module_size", "label": "模块大小", "type": "number"},
                                  {"key": "mean_abs_r", "label": "模块内平均 |r|", "type": "number"}],
                         rows=rows)],
        figures=[figure],
        data_out=data_out,
        store={"modules": {"modules": info, "labels": labels.tolist(), "names": names,
                           "eigengenes": {int(key): value.tolist() for key, value in eigengenes.items()}}},
    )


def analysis_network_hubs(ctx, params, outdir):
    threshold = _p_float(params, "threshold", 0.6)
    top_n = _p_int(params, "top_n", 20)
    matrix, names, _keep = _select_matrix(ctx, _p_int(params, "max_features", 800))
    corr = _corr(matrix, str(_p(params, "method", "pearson") or "pearson"))
    rows = _hubs(corr, names, threshold, top_n)
    if not rows:
        raise sc.prerequisite("没有超过阈值的相关，无法计算 hub")
    outdir = _outdir(ctx, outdir)
    fig, ax = sc.new_fig(6.8, 4.2)
    values = [row["degree"] for row in rows][::-1]
    labels = [row["feature"] for row in rows][::-1]
    ax.barh(np.arange(len(rows)), values, color=[sc.PALETTE[i % len(sc.PALETTE)] for i in range(len(rows))])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("degree (|r| ≥ threshold)", fontsize=9)
    ax.set_title("Network hub metabolites", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure = sc.save_fig(fig, outdir, "network-hubs", title=None)
    bullets = [f"按度中心性排出的 hub 特征（阈值 |r|≥{threshold}）",
               "前 3 个：" + "、".join(f"{row['feature']}(degree {row['degree']})" for row in rows[:3])]
    return sc.result(summary={"title": "网络 hub 代谢物", "bullets": bullets,
                              "metrics": {"n_hubs": len(rows), "threshold": threshold}},
                     tables=[sc.table("hubs", "hub 特征",
                                      columns=[{"key": "feature", "label": "特征", "type": "string"},
                                               {"key": "degree", "label": "度", "type": "number"},
                                               {"key": "strength", "label": "加权强度", "type": "number"}],
                                      rows=rows)],
                     figures=[figure])


def analysis_module_trait(ctx, params, outdir):
    store = ctx.get("results") or {}
    modules = store.get("modules")
    if not isinstance(modules, dict) or not modules.get("eigengenes"):
        raise sc.prerequisite("请先跑「代谢物模块」分析（module_trait_corr 需要模块特征值）")
    eigengenes = modules["eigengenes"]
    info = ctx.get("sample_info")
    if not isinstance(info, pd.DataFrame) or info.shape[1] == 0:
        raise sc.prerequisite("样本信息为空：需要分组/临床列才能做模块-表型相关")
    rows = []
    for key in list(info.columns)[:20]:
        series = info[key]
        if series.nunique(dropna=True) > max(10, int(len(series) * 0.5)):
            continue
        if series.dtype == object or str(series.dtype).startswith("category"):
            values = pd.get_dummies(series.astype(str), drop_first=True).to_numpy(dtype=float)
            labels = [str(v) for v in pd.get_dummies(series.astype(str), drop_first=True).columns]
        else:
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            values = numeric.reshape(-1, 1)
            labels = [str(key)]
        for index, label in enumerate(labels):
            trait = values[:, index]
            for module_id, eigen in eigengenes.items():
                eigen_arr = np.asarray(eigen, dtype=float)
                mask = np.isfinite(trait) & np.isfinite(eigen_arr)
                if mask.sum() < 5:
                    continue
                r = float(np.corrcoef(trait[mask], eigen_arr[mask])[0, 1])
                rows.append({"trait": label, "module": f"M{module_id}", "r": r,
                             "n": int(mask.sum())})
    if not rows:
        raise sc.prerequisite("没有可用于相关的数值/分类表型列")
    frame = pd.DataFrame(rows)
    try:
        frame["p"] = [float(2 * (1 - _t_cdf(abs(r) * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - r * r, 1e-12)),
                                           max(n - 2, 1)))) for r, n in zip(frame["r"], frame["n"])]
        frame["p_adj"] = sc.bh_fdr(frame["p"].to_numpy())
    except Exception:  # noqa: BLE001
        frame["p"] = np.nan
        frame["p_adj"] = np.nan
    outdir = _outdir(ctx, outdir)
    traits = list(dict.fromkeys(frame["trait"].tolist()))
    module_ids = list(dict.fromkeys(frame["module"].tolist()))
    matrix = np.zeros((len(module_ids), len(traits)))
    for _, row in frame.iterrows():
        matrix[module_ids.index(row["module"]), traits.index(row["trait"])] = float(row["r"])
    fig, ax = sc.new_fig(max(6.0, 0.9 * len(traits) + 3.2), max(3.0, 0.5 * len(module_ids) + 2.0))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels(traits, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(module_ids)))
    ax.set_yticklabels(module_ids, fontsize=8)
    ax.set_title("Module eigengene vs trait correlation", fontsize=10)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Pearson r", fontsize=8)
    figure = sc.save_fig(fig, outdir, "module-trait", title=None)
    significant = frame[frame["p_adj"] < 0.05] if "p_adj" in frame.columns else frame.iloc[0:0]
    bullets = [f"{len(module_ids)} 个模块 × {len(traits)} 个表型：{len(frame)} 个相关",
               f"p_adj<0.05 的：{len(significant)} 个"
               + ("（" + "、".join(f"{row['module']}~{row['trait']}({row['r']:.2f})"
                                  for _, row in significant.head(3).iterrows()) + "）" if len(significant) else "")]
    return sc.result(summary={"title": "模块-表型相关", "bullets": bullets,
                              "metrics": {"n_modules": len(module_ids), "n_traits": len(traits),
                                          "n_significant": int(len(significant))}},
                     tables=[sc.table("module_trait", "模块与表型相关",
                                      columns=[{"key": "module", "label": "模块", "type": "string"},
                                               {"key": "trait", "label": "表型", "type": "string"},
                                               {"key": "r", "label": "r", "type": "number"},
                                               {"key": "p", "label": "p", "type": "number"},
                                               {"key": "p_adj", "label": "p_adj", "type": "number"},
                                               {"key": "n", "label": "样本", "type": "number"}],
                                      rows=frame.sort_values("r", key=lambda s: -s.abs()).to_dict(orient="records"))],
                     figures=[figure])


def _t_cdf(t: float, df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.cdf(t, df))
    except Exception:  # noqa: BLE001
        return 0.5


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


def plot_corr_heatmap(ctx, params=None, outdir=None):
    params = dict(params or {})
    threshold = _p_float(params, "threshold", 0.6)
    matrix, names, _keep = _select_matrix(ctx, _p_int(params, "max_features", 300))
    corr = _corr(matrix, str(_p(params, "method", "pearson") or "pearson"))
    labels, _info = _modules(corr, threshold, _p_int(params, "min_size", 5))
    return _fig_heatmap(corr, names, labels, _outdir(ctx, outdir))


def plot_corr_network(ctx, params=None, outdir=None):
    params = dict(params or {})
    threshold = _p_float(params, "threshold", 0.6)
    matrix, names, _keep = _select_matrix(ctx, _p_int(params, "max_features", 300))
    corr = _corr(matrix, str(_p(params, "method", "pearson") or "pearson"))
    labels, _info = _modules(corr, threshold, _p_int(params, "min_size", 5))
    return _fig_network(corr, names, labels, threshold, _outdir(ctx, outdir))


ANALYSES = [
    {
        "id": "correlation_network", "title": "共表达网络", "group": "批次与网络",
        "method": "|r| 阈值建图 + 层次模块（Xia & Wishart 2011 语义；内置实现）",
        "needs": ["transform"],
        "params": [
            {"key": "threshold", "label": "相关阈值 |r|", "type": "number", "default": 0.6},
            {"key": "top_n", "label": "参与计算的特征数", "type": "number", "default": 800},
            {"key": "min_size", "label": "最小模块大小", "type": "number", "default": 5},
            {"key": "method", "label": "相关方法", "type": "select", "default": "pearson",
             "options": [{"value": "pearson", "label": "Pearson"},
                         {"value": "spearman", "label": "Spearman"}]},
        ],
        "run": analysis_correlation_network,
    },
    {
        "id": "metabolite_modules", "title": "代谢物模块", "group": "批次与网络",
        "method": "层次聚类切模块 + 模块特征值（WGCNA 思想，Langfelder 2008；简化实现）",
        "needs": ["transform"],
        "params": [
            {"key": "threshold", "label": "相关阈值 |r|", "type": "number", "default": 0.6},
            {"key": "top_n", "label": "参与计算的特征数", "type": "number", "default": 800},
            {"key": "min_size", "label": "最小模块大小", "type": "number", "default": 5},
        ],
        "run": analysis_metabolite_modules,
    },
    {
        "id": "network_hubs", "title": "网络 hub 代谢物", "group": "批次与网络",
        "method": "度中心性 / 加权强度（内置实现）",
        "needs": ["transform"],
        "params": [
            {"key": "threshold", "label": "相关阈值 |r|", "type": "number", "default": 0.6},
            {"key": "top_n", "label": "返回 hub 数", "type": "number", "default": 20},
        ],
        "run": analysis_network_hubs,
    },
    {
        "id": "module_trait_corr", "title": "模块-表型相关", "group": "批次与网络",
        "method": "模块特征值与表型相关 + BH（WGCNA 语义；内置实现）",
        "needs": ["transform"],
        "params": [],
        "run": analysis_module_trait,
    },
]

PLOTS = {
    "corr_heatmap": plot_corr_heatmap,
    "corr_network": plot_corr_network,
}

__all__ = ["ANALYSES", "PLOTS"]
