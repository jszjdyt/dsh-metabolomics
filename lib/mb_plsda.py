"""
dsh-metabolomics — PLS-DA / OPLS-DA 判别分析（mb-stat）

本模块提供：
  STEPS    : plsda（requires=["transform"]）
  ANALYSES : plsda_vip / oplsda_splot / plsda_permutation / plsda_cv / plsda_multiclass / pca_loadings
  PLOTS    : plsda_scores / oplsda_splot / vip_bar / permutation_test

方法（PLS-DA / OPLS-DA 优先调 R 的 ropls 官方实现，缺失时回退内置 NIPALS）：
  * **ropls（官方实现，优先）**：lib/r_ropls.R 通过 mb_rbridge 调 `ropls::opls`——PCA / PLS-DA
    （predI=k, orthoI=0）/ OPLS-DA（predI=1, orthoI=m）；R²X/R²Y/Q²、VIP、置换检验（permI，
    含 pR2Y/pQ2 与零分布）、p(corr) 全部来自 ropls 自己的 modelDF/summaryDF/vipVn/permMN。
    多分组时 ropls 的 OPLS-DA 只支持二分类，按"阳性类 vs 其余"一对其余构造（显式记录）。
  * **内置回退**（R 或 ropls 不可用时；method 字符串会写明）：
    PLS2（NIPALS，Wold 1966；Geladi & Kowalski 1986）：X 中心化 + Pareto/auto 标度，Y 用
    one-hot 编码（两类 → 2 列；多类 → g 列），逐成分 deflation，R²X/R²Y 为累计解释率。
  * 交叉验证 Q²：分层 k 折（每折只用训练折拟合标度参数，避免信息泄漏），
    Q² = 1 − PRESS/TSS（Wold 1978；SIMCA/ropls 语义）。走 ropls 时 Q² 以 ropls 的
    crossvalI 结果为准，Python 侧 k 折只负责混淆矩阵/准确率/AUC。
  * VIP（Variable Importance in Projection，Wold 1995）：
    VIP_j = sqrt( p · Σ_a SS_a·(w_ja/||w_a||)² / Σ_a SS_a )，SS_a = (t_aᵀt_a)·Σ_y q_{a,y}²；
    全部成分时 mean(VIP²) = 1。
  * 置换检验（Westerhuis 2008）：打乱 Y 的行 → 每次重新拟合 + 交叉验证，得到 R²Y/Q² 零分布，
    经验 p = (1 + #{null ≥ obs}) / (1 + n_perm)；同时给出置换图的 R²Y/Q² 回归截距。
  * OPLS-DA（Trygg & Wold 2002）：1 个预测成分 + k 个正交成分。正交成分取"减去预测成分后的
    残差矩阵"的主成分（正交信号校正 OSC 语义，Wold 1998），在正交校正后的矩阵上重拟合预测成分；
    输出 p1 / p(corr) 供 S-plot（Bylesjö 2006）使用。多类时用 XᵀY 的首个左奇异向量作为预测方向。
  * 判别性能：混淆矩阵 + 准确率 + 一类对其余 AUC（sklearn 仅在用到时惰性 import）。
  * 95% 置信椭圆：Hotelling T²（Johnson & Wichern《Applied Multivariate Statistical Analysis》）。

约定（DESIGN.md §2.3/§2.5）：
  * 只读 ctx，不修改 X/feature_info/sample_info；写回走返回的 store。
  * 缺前置 → prerequisite（中文原因）；参数非法 → bad_request；不吞异常。
  * 图上文字一律英文；返回给模型/UI 的文字是中文。
  * 确定性：所有随机过程都用固定种子（SEED=0）的 np.random.default_rng，重复运行结果一致。
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import mb_rbridge as rb
from mb_common import (
    PALETTE,
    EngineError,
    Timer,
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

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SEED = 0
MAX_MODEL_FEATURES = 20000      # DESIGN §2.3：超过则按方差筛到 20000 并 warning
PERM_BUDGET = 3.0e10            # 置换检验的"运算预算"（用于确定性降级置换次数）
MIN_PERMS = 50                  # 降级后至少保留的置换次数（50 次的 p 下限≈0.02，仍可判 0.05）
DEFAULT_DPI = 140

CONTROL_HINTS = ("control", "ctrl", "ctl", "normal", "healthy", "nc", "wt", "con",
                 "vehicle", "sham", "baseline", "untreated", "placebo", "mock")

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _pdict(params) -> dict:
    return dict(params or {})


def _int_param(params: dict, key: str, default, lo: int = 1, hi: int = 100000) -> int:
    raw = params.get(key)
    if raw is None or raw == "":
        value = int(default)
    else:
        try:
            value = int(round(float(raw)))
        except (TypeError, ValueError):
            raise bad_request(f"参数「{key}」必须是整数，收到：{raw!r}")
    return int(max(lo, min(hi, value)))


def _float_param(params: dict, key: str, default: float, lo: float | None = None,
                 hi: float | None = None) -> float:
    raw = params.get(key)
    if raw is None or raw == "":
        value = float(default)
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise bad_request(f"参数「{key}」必须是数字，收到：{raw!r}")
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _short(text, limit: int = 26) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _feature_meta_frame(ctx, features: list[str]) -> pd.DataFrame:
    """特征元数据（name/mz/rt/label），索引与 features 对齐；缺列用空值兜底。"""
    frame = pd.DataFrame(index=pd.Index([str(f) for f in features], name="feature"))
    info = ctx.get("feature_info")
    for col in ("name", "mz", "rt"):
        if isinstance(info, pd.DataFrame) and col in info.columns:
            try:
                series = info[col].reindex(frame.index)
            except Exception:  # noqa: BLE001 - index 重复等异常输入
                series = pd.Series(np.nan, index=frame.index)
            if col in ("mz", "rt"):
                frame[col] = pd.to_numeric(series, errors="coerce")
            else:
                frame[col] = series.where(series.notna(), "").astype(object)
        else:
            frame[col] = np.nan if col in ("mz", "rt") else ""
    names = frame["name"].astype(str)
    frame["label"] = names.where(names.str.len() > 0, frame.index.astype(str))
    return frame


def _matrix(ctx) -> tuple[np.ndarray, list[str], list[str]]:
    """取 ctx 的强度矩阵与特征/样本名（拷贝，绝不原地改 ctx）。"""
    X = ctx.get("X")
    if X is None:
        raise prerequisite("数据集没有强度矩阵：先导入特征表或载入示例数据")
    try:
        arr = np.array(X, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise EngineError("internal", f"强度矩阵无法转成数值：{error}")
    if arr.ndim != 2:
        raise EngineError("internal", f"强度矩阵维度异常：{arr.shape}（应为 样本×特征）")
    features = [str(f) for f in (ctx.get("features") or [])]
    samples = [str(s) for s in (ctx.get("samples") or [])]
    if not features:
        features = [f"F{i + 1}" for i in range(arr.shape[1])]
    if not samples:
        samples = [f"S{i + 1}" for i in range(arr.shape[0])]
    if len(features) != arr.shape[1] or len(samples) != arr.shape[0]:
        raise EngineError(
            "internal",
            f"特征/样本名数量与矩阵形状不一致：{len(samples)}×{len(features)} vs {arr.shape[0]}×{arr.shape[1]}",
        )
    return arr, features, samples


def _sample_info_frame(ctx, samples: list[str]) -> pd.DataFrame:
    info = ctx.get("sample_info")
    index = pd.Index(samples, name="sample")
    if isinstance(info, pd.DataFrame):
        if list(map(str, info.index)) == samples:
            return info
        try:
            return info.reindex(index)
        except Exception:  # noqa: BLE001
            return pd.DataFrame(index=index)
    return pd.DataFrame(index=index)


def _guess_group_col(frame: pd.DataFrame) -> str | None:
    """没有 group_key 时猜一个像分组的列（2–20 组、避开批次/样本号/时间列）。"""
    if not isinstance(frame, pd.DataFrame):
        return None
    prefer = ("group", "groups", "condition", "treatment", "class", "type", "phenotype",
              "disease", "status", "arm", "cohort", "label", "category")
    avoid = ("batch", "plate", "run", "qc", "sample", "id", "time", "age", "bmi", "dose", "event")
    best, best_score = None, -1
    for col in frame.columns:
        key = str(col).lower()
        try:
            nunique = int(frame[col].nunique(dropna=True))
        except Exception:  # noqa: BLE001
            continue
        if not (1 < nunique <= 20):
            continue
        score = 10 if key in prefer else (6 if any(p in key for p in prefer) else 0)
        if any(p in key for p in avoid):
            score -= 8
        if score > best_score:
            best, best_score = str(col), score
    return best if best_score > 0 else None


def _labels(ctx, group_col=None, samples=None, require=True):
    """取分组标签。返回 (labels, group_key, groups)；无分组且 require=False 时返回 (None,None,None)。"""
    samples = [str(s) for s in (samples or ctx.get("samples") or [])]
    frame = _sample_info_frame(ctx, samples)
    key = str(group_col or ctx.get("group_key") or "").strip()
    if not key:
        key = _guess_group_col(frame) or ""
    if not key:
        if require:
            raise prerequisite("还没有分组列：先在「样本」里设置分组列（group_col），判别分析需要监督标签")
        return None, None, None
    if key not in frame.columns:
        if require:
            raise prerequisite(f"样本信息里没有分组列「{key}」；可选列：{'、'.join(map(str, frame.columns)) or '无'}")
        return None, None, None
    raw = frame[key].tolist()
    labels = []
    for value in raw:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            labels.append("NA")
        else:
            labels.append(str(value))
    labels = np.array(labels, dtype=object)
    groups: list[str] = []
    for value in labels:
        if value not in groups:
            groups.append(value)
    if require:
        if len(groups) < 2:
            raise prerequisite(f"分组列「{key}」只有 {len(groups)} 个分组（{'、'.join(groups)}），"
                               f"PLS-DA 至少需要两个分组")
        tiny = [g for g in groups if int(np.sum(labels == g)) < 2]
        if tiny:
            detail = "、".join(f"{g}({int(np.sum(labels == g))})" for g in tiny)
            raise prerequisite(f"以下分组样本数不足 2：{detail}；判别分析无法做交叉验证")
    return labels, key, groups


# ---------------------------------------------------------------------------
# 预处理 / 矩阵准备
# ---------------------------------------------------------------------------


def _preprocess(X: np.ndarray, center_scale: str = "auto"):
    """中心化 + 标度。返回 (Xp, mean, scale, 说明文字)。

    auto/uv → 单位方差（auto scaling）；pareto → 除以 √SD（Pareto scaling，代谢组学常用，
    保留高丰度特征的相对权重）；none → 仅中心化。
    """
    arr = np.asarray(X, dtype=np.float64)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sd = arr.std(axis=0, ddof=1)
    else:
        sd = np.zeros(arr.shape[1])
    mode = str(center_scale or "auto").lower()
    if mode in ("none", "center", "centering", "no"):
        scale = np.ones_like(sd)
        label = "centering only"
    elif mode in ("pareto", "par"):
        scale = np.sqrt(np.maximum(sd, 0.0))
        label = "Pareto scaling"
    elif mode in ("uv", "standard", "auto", "unit", "zscore", "z-score"):
        scale = sd
        label = "auto scaling (unit variance)"
    else:
        raise bad_request(f"未知的标度方式「{center_scale}」（可选 auto / pareto / none）")
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (arr - mean) / scale, mean, scale, label


def _onehot(labels, groups) -> np.ndarray:
    """Y one-hot（n×g）；多类时即 g 列 0/1。"""
    g = len(groups)
    index = {str(c): i for i, c in enumerate(groups)}
    Y = np.zeros((len(labels), g), dtype=np.float64)
    for i, lab in enumerate(labels):
        j = index.get(str(lab))
        if j is not None:
            Y[i, j] = 1.0
    return Y


def _prepare(ctx, center_scale: str = "auto", require_groups: bool = True,
             group_col=None) -> dict:
    """把 ctx 变成模型输入：缺失值兜底、按方差筛特征、中心化/标度、one-hot Y。"""
    X, features, samples = _matrix(ctx)
    warnings: list[str] = []
    n, p = X.shape
    if n < 3 or p < 2:
        raise prerequisite(f"矩阵太小（{n} 样本 × {p} 特征），无法做 PLS-DA")

    missing = int(np.isnan(X).sum())
    if missing:
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        rows, cols = np.where(np.isnan(X))
        X = X.copy()
        X[rows, cols] = med[cols]
        warnings.append(
            f"矩阵含 {missing} 个缺失值（{missing / max(n * p, 1):.1%}）：多变量模型按特征中位数临时填补；"
            f"建议先跑「缺失值填补」"
        )

    feature_pos = np.arange(p)
    if p > MAX_MODEL_FEATURES:
        var = np.var(X, axis=0)
        keep = np.sort(np.argsort(-var)[:MAX_MODEL_FEATURES])
        X = X[:, keep]
        features = [features[i] for i in keep]
        feature_pos = keep
        warnings.append(f"特征数 {p} 超过上限：按方差筛到 {MAX_MODEL_FEATURES} 个特征进入多变量模型")

    labels = groups = group_key = None
    if require_groups:
        labels, group_key, groups = _labels(ctx, group_col=group_col, samples=samples, require=True)

    Xp, xmean, xscale, scale_label = _preprocess(X, center_scale)
    Y = _onehot(labels, groups) if require_groups else np.zeros((n, 0))
    Yc = Y - Y.mean(axis=0, keepdims=True) if Y.size else Y

    return {
        "X": X,
        "Xp": Xp,
        "Y": Yc,
        "Yraw": Y,
        "labels": labels,
        "groups": list(groups or []),
        "group_key": group_key,
        "features": features,
        "samples": samples,
        "feature_pos": feature_pos,
        "feature_meta": _feature_meta_frame(ctx, features),
        "xmean": xmean,
        "xscale": xscale,
        "scale_label": scale_label,
        "center_scale": str(center_scale or "auto").lower(),
        "warnings": warnings,
        "n_samples": int(n),
        "n_features": int(len(features)),
        "n_features_total": int(p),
    }


# ---------------------------------------------------------------------------
# NIPALS PLS2
# ---------------------------------------------------------------------------


def _pls2_nipals(Xp: np.ndarray, Yc: np.ndarray, n_comp: int,
                 tol: float = 1e-9, max_iter: int = 200) -> dict:
    """NIPALS PLS2（Wold 1966；Geladi & Kowalski 1986）：逐成分提取 t/w/p/q 并 deflation。"""
    X = np.array(Xp, dtype=np.float64, copy=True)
    F = np.array(Yc, dtype=np.float64, copy=True)
    n, p = X.shape
    m = F.shape[1]
    A = int(max(1, min(int(n_comp), n - 1, p)))
    ss_x_total = float(np.sum(X ** 2)) or 1.0
    ss_y_total = float(np.sum(F ** 2)) or 1.0
    T = np.zeros((n, A))
    W = np.zeros((p, A))
    P = np.zeros((p, A))
    Q = np.zeros((m, A))
    r2x = np.zeros(A)
    r2y = np.zeros(A)
    used = 0
    for a in range(A):
        col = int(np.argmax(np.var(X, axis=0)))
        t = X[:, col].copy()
        if not np.any(np.abs(t) > 1e-15):
            break
        w = np.zeros(p)
        for _ in range(max_iter):
            w = X.T @ t
            norm_w = float(np.linalg.norm(w))
            if norm_w < 1e-12:
                break
            w = w / norm_w
            t_new = X @ w
            delta = float(np.linalg.norm(t_new - t))
            t = t_new
            if delta <= tol * max(float(np.linalg.norm(t)), 1e-12):
                break
        tt = float(t @ t)
        if not np.isfinite(tt) or tt < 1e-18:
            break
        pvec = X.T @ t / tt
        qvec = F.T @ t / tt
        X = X - np.outer(t, pvec)
        F = F - np.outer(t, qvec)
        T[:, a] = t
        W[:, a] = w
        P[:, a] = pvec
        Q[:, a] = qvec
        r2x[a] = 1.0 - float(np.sum(X ** 2)) / ss_x_total
        r2y[a] = 1.0 - float(np.sum(F ** 2)) / ss_y_total
        used = a + 1
    if used == 0:
        raise EngineError("internal", "PLS 拟合没有提取到有效成分（矩阵可能全是常数）")
    return {
        "T": T[:, :used], "W": W[:, :used], "P": P[:, :used], "Q": Q[:, :used],
        "r2x": r2x[:used], "r2y": r2y[:used], "n_comp": used,
        "ss_x_total": ss_x_total, "ss_y_total": ss_y_total,
    }


def _pls_coef(W: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """PLS 回归系数 B = W (PᵀW)⁻¹ Qᵀ（p×m）；用于交叉验证与置换检验的预测。"""
    if W.shape[1] == 0:
        return np.zeros((W.shape[0], Q.shape[0]))
    core = P.T @ W
    try:
        inv = np.linalg.inv(core)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(core)
    return W @ inv @ Q.T


def _vip(W: np.ndarray, T: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """VIP（Wold 1995）：VIP_j = sqrt(p·Σ_a SS_a·w_ja² / Σ_a SS_a)，w 已归一化。"""
    p = W.shape[0]
    if W.shape[1] == 0:
        return np.ones(p)
    ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)   # 每个成分的解释平方和
    total = float(ss.sum())
    if not np.isfinite(total) or total <= 1e-18:
        return np.ones(p)
    return np.sqrt(np.maximum(p * ((W ** 2) @ ss) / total, 0.0))


# ---------------------------------------------------------------------------
# 交叉验证
# ---------------------------------------------------------------------------


def _cv_folds(labels, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """分层 k 折（按类别逐层轮流分配），返回每折的测试样本下标。"""
    labels = np.asarray(labels, dtype=object)
    n = int(labels.size)
    k = int(max(2, min(int(k), n)))
    folds: list[list[int]] = [[] for _ in range(k)]
    classes: list[str] = []
    for lab in labels:
        if str(lab) not in classes:
            classes.append(str(lab))
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        idx = rng.permutation(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    out = [np.array(sorted(f), dtype=int) for f in folds if len(f) > 0]
    return out


def _cv_pls(prep: dict, n_comp: int, folds: list[np.ndarray]) -> dict:
    """k 折交叉验证：每折只用训练折拟合标度与模型，预测测试折。"""
    X = prep["X"]
    Y = prep["Y"]
    n = X.shape[0]
    center_scale = prep["center_scale"]
    A = int(max(1, min(int(n_comp), n - 1)))
    press = np.zeros(A)
    tss = float(np.sum(Y ** 2)) or 1.0
    oof = np.full(Y.shape, np.nan)
    for test in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test] = False
        train = np.flatnonzero(train_mask)
        if train.size < 3:
            continue
        Xtr, xmean, xscale, _ = _preprocess(X[train], center_scale)
        try:
            fit = _pls2_nipals(Xtr, Y[train], A)
        except EngineError:
            continue
        Xte = (X[test] - xmean) / xscale
        for a in range(1, fit["n_comp"] + 1):
            coef = _pls_coef(fit["W"][:, :a], fit["P"][:, :a], fit["Q"][:, :a])
            pred = Xte @ coef
            press[a - 1] += float(np.sum((Y[test] - pred) ** 2))
            if a == fit["n_comp"]:
                oof[test] = pred
    q2 = np.where(tss > 0, 1.0 - press / max(tss, 1e-18), 0.0)
    used = int(min(A, q2.size))
    return {"q2": [float(v) for v in q2[:used]], "press": [float(v) for v in press[:used]],
            "tss": float(tss), "oof": oof, "folds": folds, "n_comp": used}


def _classification(score: np.ndarray, labels, groups) -> dict:
    """由预测得分（n×g）算准确率/混淆矩阵/AUC（AUC 惰性 import sklearn）。"""
    classes = [str(g) for g in groups]
    truth = np.array([classes.index(str(lab)) if str(lab) in classes else -1 for lab in labels])
    ok = truth >= 0
    out = {"accuracy": float("nan"), "confusion": [], "auc": None, "classes": classes,
           "pred_labels": [], "n": int(ok.sum())}
    if not ok.any() or score.size == 0:
        return out
    pred_idx = np.argmax(score[ok], axis=1)
    truth_ok = truth[ok]
    accuracy = float(np.mean(pred_idx == truth_ok))
    g = len(classes)
    confusion = np.zeros((g, g), dtype=int)
    for t, p in zip(truth_ok, pred_idx):
        confusion[int(t), int(p)] += 1
    out["accuracy"] = accuracy
    out["confusion"] = confusion.tolist()
    out["pred_labels"] = [classes[i] for i in pred_idx]
    auc = None
    try:
        from sklearn.metrics import roc_auc_score  # 惰性 import：只有该分支才需要

        if g == 2:
            auc = float(roc_auc_score(truth_ok, score[ok][:, 1]))
        elif g > 2:
            auc = float(roc_auc_score(truth_ok, score[ok], multi_class="ovr", average="macro"))
    except Exception:  # noqa: BLE001 - 单一类别 / 退化预测等
        auc = None
    out["auc"] = auc
    return out


# ---------------------------------------------------------------------------
# 置换检验
# ---------------------------------------------------------------------------


def _perm_plan(n: int, p: int, n_comp: int, folds: int, n_perms: int) -> tuple[int, str | None]:
    """按确定性"运算预算"决定实际置换次数（大数据自动降级并给出可读说明）。"""
    requested = int(max(1, n_perms))
    per_perm = 40.0 * float(max(p, 1)) * float(max(n, 1)) * float(max(n_comp, 1)) * (1.0 + float(max(folds, 2)))
    allowed = int(PERM_BUDGET / max(per_perm, 1.0))
    if allowed >= requested:
        return requested, None
    effective = int(max(min(requested, MIN_PERMS), max(allowed, 1)))
    effective = int(min(effective, requested))
    note = (f"置换次数按计算预算从 {requested} 降为 {effective}"
            f"（{n} 样本 × {p} 特征 × {n_comp} 成分 × {folds} 折 CV 的估计开销过大；"
            f"可减少特征或用更小的 cv_folds 重跑）")
    return effective, note


def _permutation(prep: dict, n_comp: int, folds: list[np.ndarray], n_perms: int) -> dict:
    """置换检验（Westerhuis 2008）：打乱 Y 行，重算 R²Y 与 CV Q² 零分布 + 经验 p。"""
    n = prep["X"].shape[0]
    Xp = prep["Xp"]
    Y = prep["Y"]
    requested = int(max(1, n_perms))
    base_fit = _pls2_nipals(Xp, Y, n_comp)
    A = base_fit["n_comp"]
    observed_r2y = float(base_fit["r2y"][-1])
    observed_q2 = float(_cv_pls(prep, A, folds)["q2"][-1])
    effective, note = _perm_plan(n, prep["n_features"], A, len(folds), requested)
    rng = np.random.default_rng(SEED)
    null_r2 = np.zeros(effective)
    null_q2 = np.zeros(effective)
    corr = np.zeros(effective)
    tss = float(np.sum(Y ** 2)) or 1.0
    y_flat = Y.reshape(-1)
    for b in range(effective):
        perm = rng.permutation(n)
        Yperm = Y[perm]
        try:
            fit = _pls2_nipals(Xp, Yperm, A)
            null_r2[b] = float(fit["r2y"][-1])
        except EngineError:
            null_r2[b] = 0.0
        press = 0.0
        for test in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[test] = False
            train = np.flatnonzero(train_mask)
            if train.size < 3:
                continue
            Xtr, xmean, xscale, _ = _preprocess(prep["X"][train], prep["center_scale"])
            try:
                fit = _pls2_nipals(Xtr, Yperm[train], A)
            except EngineError:
                continue
            coef = _pls_coef(fit["W"], fit["P"], fit["Q"])
            pred = ((prep["X"][test] - xmean) / xscale) @ coef
            press += float(np.sum((Yperm[test] - pred) ** 2))
        null_q2[b] = 1.0 - press / tss
        perm_flat = Yperm.reshape(-1)
        denom = float(np.linalg.norm(perm_flat) * np.linalg.norm(y_flat))
        corr[b] = float(perm_flat @ y_flat / denom) if denom > 1e-18 else 0.0

    p_r2y = float((1 + int(np.sum(null_r2 >= observed_r2y))) / (1 + effective))
    p_q2 = float((1 + int(np.sum(null_q2 >= observed_q2))) / (1 + effective))

    def _intercept(values: np.ndarray) -> float:
        try:
            slope, intercept = np.polyfit(corr, values, 1)
            return float(intercept)
        except Exception:  # noqa: BLE001
            return float("nan")

    return {
        "n_perms": effective,
        "requested": requested,
        "r2y_obs": observed_r2y,
        "q2_obs": observed_q2,
        "r2y_null": null_r2.tolist(),
        "q2_null": null_q2.tolist(),
        "corr": corr.tolist(),
        "p_r2y": p_r2y,
        "p_q2": p_q2,
        "r2y_intercept": _intercept(null_r2),
        "q2_intercept": _intercept(null_q2),
        "note": note,
    }


# ---------------------------------------------------------------------------
# OPLS-DA
# ---------------------------------------------------------------------------


def _first_direction(mat: np.ndarray) -> np.ndarray:
    """p×m 矩阵的首个方向向量（m=1 直接归一化；m>1 取首个左奇异向量，与 PLS 首权一致）。"""
    if mat.shape[1] == 1:
        vec = mat[:, 0]
    else:
        try:
            u, _s, _vt = np.linalg.svd(mat, full_matrices=False)
            vec = u[:, 0]
        except np.linalg.LinAlgError:
            vec = mat.sum(axis=1)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.zeros(mat.shape[0])
    return vec / norm


def _opls_fit(Xp: np.ndarray, Yc: np.ndarray, n_orth: int = 2) -> dict:
    """OPLS-DA（Trygg & Wold 2002）：预测成分 + 正交成分（OSC 语义）。

    步骤：
      1) 预测成分：w1 ∝ XᵀY（多列取首个左奇异向量），t = Xw1，p1 = Xᵀt/(tᵀt)；
      2) 残差矩阵做主成分 → 正交成分 (T_orth, V_orth)，X_clean = X − T_orth V_orthᵀ；
      3) 在 X_clean 上重拟合预测成分，得到最终 t_pred / p_pred / q；预测系数
         coef = (I − V Vᵀ)(w_pred ⊗ q)，测试样本 Ŷ = X_test coef。
    """
    X = np.array(Xp, dtype=np.float64, copy=True)
    Y = np.array(Yc, dtype=np.float64, copy=True)
    n, p = X.shape
    ss_x_total = float(np.sum(X ** 2)) or 1.0
    ss_y_total = float(np.sum(Y ** 2)) or 1.0
    n_orth = int(max(0, n_orth))

    w1 = _first_direction(X.T @ Y)
    if not np.any(w1):
        raise EngineError("internal", "OPLS 无法提取预测成分：X 与 Y 没有共变")
    t1 = X @ w1
    p1 = X.T @ t1 / max(float(t1 @ t1), 1e-18)
    X_res = X - np.outer(t1, p1)

    V = np.zeros((p, 0))
    T_orth = np.zeros((n, 0))
    sv = np.zeros(0)
    k = int(min(n_orth, max(1, n - 1), p))
    if n_orth > 0:
        try:
            U, S, Vt = np.linalg.svd(X_res, full_matrices=False)
            V = Vt[:k].T
            T_orth = U[:, :k] * S[:k]
            sv = S[:k]
        except np.linalg.LinAlgError:
            V = np.zeros((p, 0))
            T_orth = np.zeros((n, 0))
            sv = np.zeros(0)

    if V.shape[1] > 0:
        X_clean = X - T_orth @ V.T
    else:
        X_clean = X

    w_pred = _first_direction(X_clean.T @ Y)
    if not np.any(w_pred):
        w_pred = w1
    t_pred = X_clean @ w_pred
    tt = max(float(t_pred @ t_pred), 1e-18)
    p_pred = X_clean.T @ t_pred / tt
    q = Y.T @ t_pred / tt                      # m
    Yhat = np.outer(t_pred, q)
    r2y = 1.0 - float(np.sum((Y - Yhat) ** 2)) / ss_y_total

    coef = np.outer(w_pred, q)                 # p×m
    if V.shape[1] > 0:
        coef = coef - V @ (V.T @ coef)

    ss_pred = float(np.sum(np.outer(t_pred, p_pred) ** 2))
    ss_orth = float(np.sum(sv ** 2))
    final_res = X_clean - np.outer(t_pred, p_pred)
    ss_res = float(np.sum(final_res ** 2))
    r2x_pred = ss_pred / ss_x_total
    r2x_orth = [float(s ** 2 / ss_x_total) for s in sv]
    r2x_total = 1.0 - ss_res / ss_x_total

    t_center = t_pred - t_pred.mean()
    denom_t = float(np.linalg.norm(t_center))
    if denom_t > 1e-15:
        X_center = X_clean - X_clean.mean(axis=0, keepdims=True)
        num = X_center.T @ t_center
        den = np.linalg.norm(X_center, axis=0) * denom_t
        p_corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-15)
    else:
        p_corr = np.zeros(p)

    vip = _vip(w_pred.reshape(-1, 1), t_pred.reshape(-1, 1), q.reshape(-1, 1))

    return {
        "kind": "oplsda",
        "n_comp": int(1 + V.shape[1]),
        "n_orth": int(V.shape[1]),
        "w_pred": w_pred,
        "p_pred": p_pred,
        "t_pred": t_pred,
        "t_orth": T_orth,
        "v_orth": V,
        "s_orth": sv,
        "p_corr": np.asarray(p_corr, dtype=float).reshape(-1),
        "coef": coef,
        "q": q,
        "r2y": r2y,
        "r2x_pred": float(r2x_pred),
        "r2x_orth": r2x_orth,
        "r2x_total": float(r2x_total),
        "vip": vip,
        "r2x": [float(1.0 - (ss_x_total - ss_pred - sum(s ** 2 for s in sv)) / ss_x_total)],
    }


def _cv_opls(prep: dict, n_orth: int, folds: list[np.ndarray]) -> dict:
    """OPLS-DA 的 k 折交叉验证 Q²（每折拟合完整 OPLS，用预测系数预测测试折）。"""
    X = prep["X"]
    Y = prep["Y"]
    n = X.shape[0]
    tss = float(np.sum(Y ** 2)) or 1.0
    press = 0.0
    oof = np.full(Y.shape, np.nan)
    for test in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test] = False
        train = np.flatnonzero(train_mask)
        if train.size < 4:
            continue
        Xtr, xmean, xscale, _ = _preprocess(X[train], prep["center_scale"])
        try:
            fit = _opls_fit(Xtr, Y[train], n_orth)
        except EngineError:
            continue
        Xte = (X[test] - xmean) / xscale
        pred = Xte @ fit["coef"]
        press += float(np.sum((Y[test] - pred) ** 2))
        oof[test] = pred
    q2 = 1.0 - press / tss
    return {"q2": [float(q2)], "press": [press], "tss": tss, "oof": oof, "folds": folds}


# ---------------------------------------------------------------------------
# 模型装配
# ---------------------------------------------------------------------------


def _base_params(ctx, params: dict) -> dict:
    """补全参数：method / n_comp / center_scale 未显式给出时继承上一次 PLS-DA 结果。"""
    out = _pdict(params)
    store = _cached_store(ctx) or {}
    if not out.get("method"):
        out["method"] = store.get("method") or "plsda"
    if out.get("n_comp") is None:
        out["n_comp"] = store.get("n_comp") or 3
    if not out.get("center_scale"):
        out["center_scale"] = store.get("center_scale") or "auto"
    if out.get("orth_comp") is None and store.get("opls"):
        out["orth_comp"] = (store.get("opls") or {}).get("n_orth")
    return out


def _resolve_method(params: dict) -> str:
    raw = str(params.get("method") or "plsda").strip().lower()
    if raw in ("opls", "oplsda", "opls-da", "opls_da"):
        return "oplsda"
    if raw in ("pls", "plsda", "pls-da", "pls_da", "pls2"):
        return "plsda"
    raise bad_request(f"未知的判别方法「{params.get('method')}」（可选 plsda / oplsda）")


# ---------------------------------------------------------------------------
# ropls 官方实现（R 桥）——PLS-DA / OPLS-DA / PCA 的优先路径
# ---------------------------------------------------------------------------

R_SCRIPT = "r_ropls.R"
R_TIMEOUT = 300              # 秒；mb_rbridge 内部再夹到 ≤600
R_MAX_PERMS = 1000           # ropls 置换是 R 侧循环，超过这个量级确定性降级
R_FALLBACK_LABEL = "回退内置 NIPALS（ropls 不可用）"


def _impl_param(params: dict) -> str:
    """implementation 参数：auto（默认，优先 ropls）/ ropls / builtin。"""
    raw = str((params or {}).get("implementation") or "auto").strip().lower()
    if raw in ("ropls", "r", "official"):
        return "ropls"
    if raw in ("builtin", "python", "numpy", "internal"):
        return "builtin"
    return "auto"


def ropls_status() -> dict:
    """ropls 可用性（自检/UI/方法字符串共用；永不抛异常）。"""
    try:
        info = rb.r_info()
    except Exception as error:  # noqa: BLE001 - 探测失败不影响主流程
        return {"available": False, "reason": f"R 探测失败：{error}", "rscript": None,
                "r_version": None, "ropls_version": None}
    return {
        "available": bool(info.get("available")) and bool((info.get("packages") or {}).get("ropls")),
        "rscript": info.get("script"),
        "r_version": info.get("version"),
        "ropls_version": info.get("ropls_version"),
        "reason": "" if (info.get("available") and (info.get("packages") or {}).get("ropls"))
        else rb.unavailable_reason(),
    }


def _r_outdir(ctx, tag: str) -> str:
    base = str(ctx.get("outdir") or os.getcwd())
    return os.path.join(base, "ropls", tag)


def _r_info_frame(ctx, prep: dict) -> pd.DataFrame:
    """样本注释 + 明确的 `_mb_group` 分组列（避免依赖 group_col 猜测）。"""
    samples = [str(s) for s in prep["samples"]]
    frame = _sample_info_frame(ctx, samples).copy()
    frame.index = pd.Index(samples, name="sample")
    labels = prep.get("labels")
    if labels is None:
        labels = ["NA"] * len(samples)
    frame["_mb_group"] = [str(x) for x in labels]
    return frame


def _ropls_label(result_payload: dict, method: str) -> str:
    """真实实现的方法字符串（含 n_comp / ortho 值与 ropls 版本）。"""
    version = str(result_payload.get("ropls_version") or "").strip()
    version_txt = f"ropls {version}" if version else "ropls"
    if method == "oplsda":
        ortho = int(result_payload.get("orthoI") or 0)
        return f"ropls::opls（OPLS-DA，predI=1，orthoI={ortho}，{version_txt}）"
    n_comp = int(result_payload.get("predI") or 0)
    return f"ropls::opls（PLS-DA 模式，predI={n_comp}，orthoI=0，{version_txt}）"


def _label_method(model: dict) -> str:
    """模型真实实现的方法字符串（ropls::opls … 或 回退内置 NIPALS（ropls 不可用）…）。"""
    label = str(model.get("impl_label") or "").strip()
    if label:
        return label
    if model.get("implementation") == "ropls":
        return "ropls::opls"
    return R_FALLBACK_LABEL


def _with_method(payload: dict, model: dict) -> dict:
    """把真实实现写进结果信封（summary.method，供 UI / Agent / 审计读取）。"""
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary["method"] = _label_method(model)
    return payload


def _align_rows(frame: pd.DataFrame, index, what: str) -> pd.DataFrame:
    """把 R 写回的表按样本/特征名对齐；整行缺失视为协议对不上（不静默补 0）。"""
    target = pd.Index([str(x) for x in index])
    if frame.index.has_duplicates:
        frame = frame[~frame.index.duplicated(keep="first")]
    out = frame.reindex(target)
    if len(out) and out.isna().all(axis=1).any():
        missing = int(out.isna().all(axis=1).sum())
        raise EngineError("internal", f"ropls 返回的 {what} 表与数据对不上：{missing} 行缺失")
    return out


def _first_row(frame: pd.DataFrame, component: str) -> dict:
    if frame is None or frame.empty:
        return {}
    if "component" in frame.columns and component in set(map(str, frame["component"])):
        return frame.loc[frame["component"] == component].iloc[0].to_dict()
    return {}


def _num(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _perm_from_ropls(perm_table: pd.DataFrame, perm_meta: dict,
                     requested: int) -> dict | None:
    """ropls permMN（第 1 行=观测模型，其余=打乱标签）→ 与内置实现同构的置换 dict。"""
    if perm_table is None or len(perm_table) < 2:
        return None
    null = perm_table.iloc[1:]
    r2y = pd.to_numeric(null.get("R2Y(cum)"), errors="coerce").to_numpy(dtype=float)
    q2 = pd.to_numeric(null.get("Q2(cum)"), errors="coerce").to_numpy(dtype=float)
    corr = pd.to_numeric(null.get("sim"), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(r2y) & np.isfinite(q2)
    r2y = r2y[valid]
    q2 = q2[valid]
    corr = corr[valid]

    def _intercept(values):
        try:
            finite = np.isfinite(values) & np.isfinite(corr)
            if int(finite.sum()) < 2:
                return float("nan")
            _slope, intercept = np.polyfit(corr[finite], values[finite], 1)
            return float(intercept)
        except Exception:  # noqa: BLE001
            return float("nan")

    n_perms = int(len(r2y))
    return {
        "n_perms": n_perms,
        "requested": int(requested),
        "r2y_obs": _num(perm_meta.get("r2y_obs")),
        "q2_obs": _num(perm_meta.get("q2_obs")),
        "r2y_null": r2y.tolist(),
        "q2_null": q2.tolist(),
        "corr": corr.tolist(),
        "p_r2y": _num(perm_meta.get("p_r2y"), 1.0),
        "p_q2": _num(perm_meta.get("p_q2"), 1.0),
        "r2y_intercept": _intercept(r2y),
        "q2_intercept": _intercept(q2),
        "note": None,
        "source": "ropls",
    }


def _cv_python(ctx, prep: dict, method: str, n_comp: int, n_orth: int,
               cv_folds) -> dict:
    """Python 侧分层 k 折：只提供混淆矩阵/准确率/AUC 与逐成分 Q²（ropls 不返回 OOF 预测）。"""
    folds = _cv_folds(prep["labels"], int(cv_folds or 7), np.random.default_rng(SEED))
    if method == "oplsda":
        cv = _cv_opls(prep, n_orth, folds)
    else:
        cv = _cv_pls(prep, int(max(1, n_comp)), folds)
    cv["classification"] = _classification(cv["oof"], prep["labels"], prep["groups"])
    cv["source"] = "python-kfold"
    return cv


def _fit_model_ropls(ctx, params: dict, *, with_cv: bool = False, cv_folds=None,
                     with_permutation: bool = False, n_perms=None) -> dict:
    """调 r_ropls.R（ropls::opls）拟合，装配成与内置实现同构的 model dict。

    ropls 不可用时抛 mb_rbridge.RUnavailable，由 _fit_model 决定是否回退；
    ropls 运行期报错按 runtime 上抛（不静默换算法）。
    """
    method = _resolve_method(params)
    center_scale = str(params.get("center_scale") or "auto")
    prep = _prepare(ctx, center_scale=center_scale, group_col=params.get("group_col"))
    n = prep["n_samples"]
    n_comp = _int_param(params, "n_comp", 3, 1, 50)
    if method == "oplsda":
        orth = params.get("orth_comp")
        n_orth = int(max(1, min(int(orth) if orth is not None else n_comp - 1, n - 2, 20)))
        pred_i = 1
    else:
        n_orth = 0
        pred_i = int(max(1, min(n_comp, n - 1, prep["n_features"])))

    folds = int(max(3, min(int(cv_folds or params.get("cv_folds") or 7), n - 1)))
    requested_perms = 0
    perms = 0
    if with_permutation:
        requested_perms = int(max(1, n_perms or params.get("permutations") or 200))
        perms = int(min(requested_perms, R_MAX_PERMS))

    positive = ""
    if method == "oplsda" and len(prep["groups"]) > 2:
        positive = str(params.get("positive_class") or "").strip() or str(prep["groups"][0])

    payload = {
        "expr": prep["X"],
        "features": prep["features"],
        "samples": prep["samples"],
        "info": _r_info_frame(ctx, prep),
        "params": {
            "mode": "oplsda" if method == "oplsda" else "plsda",
            "n_comp": int(pred_i + n_orth) if method == "oplsda" else int(pred_i),
            "orth_comp": int(n_orth),
            "center_scale": str(prep["center_scale"]),
            "cv_folds": int(folds),
            "permutations": int(perms),
            "seed": int(SEED),
            "group_col": "_mb_group",
            "positive_class": positive,
        },
    }
    rres = rb.run_r(R_SCRIPT, payload, outdir=_r_outdir(ctx, method), timeout=R_TIMEOUT)
    r_payload = rres.get("result") or {}
    tables = rres.get("tables") or {}
    scores_r = tables.get("scores")
    loadings_r = tables.get("loadings")
    modeldf = tables.get("modeldf")
    if scores_r is None or loadings_r is None:
        raise EngineError("internal", "ropls 返回缺少 scores/loadings 表")

    scores_r = _align_rows(scores_r, prep["samples"], "scores")
    loadings_r = loadings_r.reindex(pd.Index([str(f) for f in prep["features"]]))
    for column in ("vip", "loading_p1", "p_corr1"):
        if column not in loadings_r.columns:
            loadings_r[column] = np.nan

    vip_values = pd.to_numeric(loadings_r["vip"], errors="coerce").to_numpy(dtype=float)
    label = _ropls_label(r_payload, method)
    summary_r = r_payload.get("summary") or {}
    model: dict = {
        "method": method,
        "prep": prep,
        "classes": prep["groups"],
        "labels": prep["labels"],
        "center_scale": prep["center_scale"],
        "scale_label": prep["scale_label"],
        "warnings": list(prep["warnings"]),
        "params": params,
        "implementation": "ropls",
        "impl_label": label,
        "ropls": {
            "version": r_payload.get("ropls_version"),
            "r_version": r_payload.get("r_version"),
            "model_type": r_payload.get("model_type"),
            "predI": r_payload.get("predI"),
            "orthoI": r_payload.get("orthoI"),
            "crossvalI": r_payload.get("crossvalI"),
            "positive_class": r_payload.get("positive_class") or None,
            "binary": bool(r_payload.get("binary")),
            "ms": int(rres.get("ms") or 0),
        },
        "vip_values": vip_values,
    }

    scores = scores_r.copy()
    scores.index = pd.Index([str(s) for s in prep["samples"]], name="sample")
    scores["group"] = prep["labels"]
    model["scores"] = scores
    model["vip"] = _vip_frame(prep, vip_values)
    model["loadings"] = loadings_r

    if method == "oplsda":
        total_row = _first_row(modeldf, "sum")
        pred_row = _first_row(modeldf, "p1")
        ortho_rows = []
        if modeldf is not None and "component" in modeldf.columns:
            ortho_rows = [row for row in modeldf.to_dict(orient="records")
                          if str(row.get("component", "")).startswith("o")]
        n_orth_actual = int(r_payload.get("orthoI") or len(ortho_rows) or n_orth)
        model["n_comp"] = int(pred_i + n_orth_actual)
        model["n_orth"] = n_orth_actual
        model["r2x_pred"] = _num(pred_row.get("r2x"), 0.0)
        model["r2x_orth"] = [_num(row.get("r2x"), 0.0) for row in ortho_rows]
        model["r2x_total"] = _num(total_row.get("r2x_cum"), _num(summary_r.get("r2x_cum"), 0.0))
        model["r2x_cum"] = float(model["r2x_total"] or 0.0)
        model["r2y_cum"] = float(_num(summary_r.get("r2y_cum"), 0.0) or 0.0)
        model["r2y_pred"] = float(_num(pred_row.get("r2y_cum"), 0.0) or 0.0)
        model["q2_cum"] = _num(summary_r.get("q2_cum"))
        model["q2"] = [model["q2_cum"]] if model["q2_cum"] is not None else []
        model["r2y"] = [model["r2y_cum"]]
        model["r2x"] = [model["r2x_total"]]
        p1 = pd.to_numeric(loadings_r["loading_p1"], errors="coerce").to_numpy(dtype=float)
        p_corr = pd.to_numeric(loadings_r["p_corr1"], errors="coerce").to_numpy(dtype=float)
        model["splot"] = _splot_frame(prep, {"p_pred": p1, "p_corr": p_corr, "vip": vip_values})
        model["p_corr"] = p_corr.tolist()
        model["p_pred"] = p1.tolist()
    else:
        model["n_comp"] = int(r_payload.get("n_comp") or pred_i)
        model["n_orth"] = 0
        model["r2x_cum"] = float(_num(summary_r.get("r2x_cum"), 0.0) or 0.0)
        model["r2y_cum"] = float(_num(summary_r.get("r2y_cum"), 0.0) or 0.0)
        model["q2_cum"] = _num(summary_r.get("q2_cum"))
        r2x_list: list[float] = []
        r2y_list: list[float] = []
        q2_list: list[float] = []
        if modeldf is not None and len(modeldf):
            pred_rows = [row for row in modeldf.to_dict(orient="records")
                         if str(row.get("component", "")).startswith("p")]
            for row in pred_rows:
                r2x_list.append(_num(row.get("r2x_cum"), 0.0))
                r2y_list.append(_num(row.get("r2y_cum"), 0.0))
                q2_value = _num(row.get("q2_cum"))
                if q2_value is not None:
                    q2_list.append(q2_value)
        model["r2x"] = r2x_list or [model["r2x_cum"]]
        model["r2y"] = r2y_list or [model["r2y_cum"]]
        model["q2"] = q2_list or ([model["q2_cum"]] if model["q2_cum"] is not None else [])
        model["splot"] = None
        loading_cols = [c for c in loadings_r.columns if str(c).startswith("loading_p")]
        p_columns = loadings_r[loading_cols].apply(pd.to_numeric, errors="coerce")
        corr_columns = loadings_r[[c for c in loadings_r.columns if str(c).startswith("p_corr")]]
        model["p_pred"] = p_columns.iloc[:, 0].to_numpy(dtype=float) if loading_cols else None
        model["p_corr"] = (corr_columns.iloc[:, 0].to_numpy(dtype=float)
                           if len(corr_columns.columns) else None)

    perm = _perm_from_ropls(tables.get("perm"), r_payload.get("perm") or {}, requested_perms)
    if perm is not None:
        model["permutation"] = perm
    if with_cv:
        model["cv"] = _cv_python(ctx, prep, method, model["n_comp"], model.get("n_orth") or 0,
                                 folds)
    if perms and perm is None:
        model["warnings"].append("ropls 没有返回置换零分布（permI=0 或响应不是单列），置换检验未运行")
    if requested_perms > perms:
        model["warnings"].append(
            f"ropls 置换次数从 {requested_perms} 降到 {perms}（R 侧置换循环的确定性上限）")
    # R 侧 warnings/notes 全部上抛（含"多分类按一对其余"这类必须让用户看到的说明）
    for text in list(rres.get("warnings") or []) + list(r_payload.get("notes") or []):
        text = str(text).strip()
        if text and text not in model["warnings"]:
            model["warnings"].append(text)
    _attach_metrics(model)
    model["metrics"]["implementation"] = label
    model["metrics"]["ropls_version"] = r_payload.get("ropls_version")
    if method == "oplsda":
        model["metrics"]["binary"] = bool(r_payload.get("binary"))
        model["metrics"]["positive_class"] = r_payload.get("positive_class") or None
        model["metrics"]["r2y_pred"] = model.get("r2y_pred")
    return model


def _fit_model_builtin(ctx, params: dict, *, with_cv: bool = False, cv_folds=None,
                       with_permutation: bool = False, n_perms=None) -> dict:
    """内置 NIPALS 实现（ropls 不可用时的回退路径）。"""
    params = _base_params(ctx, params)
    method = _resolve_method(params)
    center_scale = str(params.get("center_scale") or "auto")
    prep = _prepare(ctx, center_scale=center_scale, group_col=params.get("group_col"))
    n_comp = _int_param(params, "n_comp", 3, 1, 50)
    warnings = list(prep["warnings"])
    n = prep["n_samples"]

    model: dict = {
        "method": method,
        "prep": prep,
        "classes": prep["groups"],
        "labels": prep["labels"],
        "center_scale": prep["center_scale"],
        "scale_label": prep["scale_label"],
        "warnings": warnings,
        "params": params,
    }

    if method == "oplsda":
        n_orth_default = max(0, n_comp - 1)
        orth = params.get("orth_comp")
        n_orth = int(max(0, min(int(orth) if orth is not None else n_orth_default, n - 2, 20)))
        fit = _opls_fit(prep["Xp"], prep["Y"], n_orth)
        model.update(fit)
        model["n_comp"] = int(fit["n_comp"])
        model["n_orth"] = int(fit["n_orth"])
        model["r2x_cum"] = float(fit["r2x_total"])
        model["r2y_cum"] = float(fit["r2y"])
        model["vip_values"] = fit["vip"]
        scores = pd.DataFrame(
            {"T_pred": fit["t_pred"]},
            index=pd.Index(prep["samples"], name="sample"),
        )
        for j in range(fit["t_orth"].shape[1]):
            scores[f"T_orth{j + 1}"] = fit["t_orth"][:, j]
        scores["group"] = prep["labels"]
        model["scores"] = scores
        model["vip"] = _vip_frame(prep, fit["vip"])
        model["splot"] = _splot_frame(prep, fit)
        if with_cv:
            folds = _cv_folds(prep["labels"], int(cv_folds or params.get("cv_folds") or 7),
                              np.random.default_rng(SEED))
            cv = _cv_opls(prep, n_orth, folds)
            cv["classification"] = _classification(cv["oof"], prep["labels"], prep["groups"])
            model["cv"] = cv
            model["q2_cum"] = float(cv["q2"][-1])
            model["q2"] = [float(cv["q2"][-1])]
        if with_permutation:
            perm = _permutation_opls(prep, n_orth, int(cv_folds or params.get("cv_folds") or 7),
                                     int(n_perms or params.get("permutations") or 200))
            model["permutation"] = perm
    else:
        fit = _pls2_nipals(prep["Xp"], prep["Y"], n_comp)
        vip = _vip(fit["W"], fit["T"], fit["Q"])
        model.update({
            "T": fit["T"], "W": fit["W"], "P": fit["P"], "Q": fit["Q"],
            "r2x": fit["r2x"].tolist(), "r2y": fit["r2y"].tolist(),
            "n_comp": int(fit["n_comp"]),
            "r2x_cum": float(fit["r2x"][-1]), "r2y_cum": float(fit["r2y"][-1]),
            "coef": _pls_coef(fit["W"], fit["P"], fit["Q"]),
            "vip_values": vip,
        })
        scores = pd.DataFrame(fit["T"], index=pd.Index(prep["samples"], name="sample"))
        scores.columns = [f"T{j + 1}" for j in range(fit["T"].shape[1])]
        scores["group"] = prep["labels"]
        model["scores"] = scores
        model["vip"] = _vip_frame(prep, vip)
        model["splot"] = None
        if with_cv:
            folds = _cv_folds(prep["labels"], int(cv_folds or params.get("cv_folds") or 7),
                              np.random.default_rng(SEED))
            cv = _cv_pls(prep, fit["n_comp"], folds)
            cv["classification"] = _classification(cv["oof"], prep["labels"], prep["groups"])
            model["cv"] = cv
            model["q2"] = [float(v) for v in cv["q2"]]
            model["q2_cum"] = float(cv["q2"][-1])
        if with_permutation:
            model["permutation"] = _permutation(prep, fit["n_comp"], _cv_folds(
                prep["labels"], int(cv_folds or params.get("cv_folds") or 7),
                np.random.default_rng(SEED)), int(n_perms or params.get("permutations") or 200))

    _attach_metrics(model)
    model["implementation"] = "builtin"
    model["impl_label"] = str(model.get("fallback_label") or R_FALLBACK_LABEL)
    model["metrics"]["implementation"] = model["impl_label"]
    return model


def _fit_model(ctx, params: dict, *, with_cv: bool = False, cv_folds=None,
               with_permutation: bool = False, n_perms=None) -> dict:
    """拟合 PLS-DA / OPLS-DA：优先 ropls 官方实现，R/ropls 不可用时回退内置 NIPALS。

    method 字符串始终反映真实实现（ropls::opls … 或 回退内置 NIPALS（ropls 不可用）…）。
    ropls 运行期报错（runtime/prerequisite/timeout）**不**回退——不静默换统计方法。
    """
    params = _base_params(ctx, params)
    impl = _impl_param(params)
    fallback_reason = ""
    if impl in ("auto", "ropls"):
        try:
            return _fit_model_ropls(ctx, params, with_cv=with_cv, cv_folds=cv_folds,
                                    with_permutation=with_permutation, n_perms=n_perms)
        except rb.RUnavailable as error:
            if impl == "ropls":
                raise EngineError("unsupported",
                                  f"ropls 官方实现不可用（implementation=ropls）：{error.message}")
            fallback_reason = error.message
        except EngineError as error:
            # prerequisite / runtime / timeout / internal 一律上抛，不允许静默改方法
            raise
    elif impl == "builtin":
        fallback_reason = "参数显式指定 implementation=builtin"
    model = _fit_model_builtin(ctx, params, with_cv=with_cv, cv_folds=cv_folds,
                               with_permutation=with_permutation, n_perms=n_perms)
    if fallback_reason:
        model["fallback_label"] = f"{R_FALLBACK_LABEL}—{fallback_reason}"
        model["impl_label"] = model["fallback_label"]
        model["metrics"]["implementation"] = model["impl_label"]
    return model


def _permutation_opls(prep: dict, n_orth: int, folds_k: int, n_perms: int) -> dict:
    """OPLS-DA 的置换检验（与 PLS-DA 同一框架：打乱 Y、重算 R²Y 与 CV Q²）。"""
    n = prep["X"].shape[0]
    Y = prep["Y"]
    base = _opls_fit(prep["Xp"], Y, n_orth)
    folds = _cv_folds(prep["labels"], folds_k, np.random.default_rng(SEED))
    obs_r2y = float(base["r2y"])
    obs_q2 = float(_cv_opls(prep, n_orth, folds)["q2"][-1])
    effective, note = _perm_plan(n, prep["n_features"], max(1, base["n_comp"]), len(folds), n_perms)
    rng = np.random.default_rng(SEED)
    null_r2 = np.zeros(effective)
    null_q2 = np.zeros(effective)
    corr = np.zeros(effective)
    tss = float(np.sum(Y ** 2)) or 1.0
    y_flat = Y.reshape(-1)
    for b in range(effective):
        perm = rng.permutation(n)
        Yperm = Y[perm]
        try:
            fit = _opls_fit(prep["Xp"], Yperm, n_orth)
            null_r2[b] = float(fit["r2y"])
        except EngineError:
            null_r2[b] = 0.0
        press = 0.0
        for test in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[test] = False
            train = np.flatnonzero(train_mask)
            if train.size < 4:
                continue
            Xtr, xmean, xscale, _ = _preprocess(prep["X"][train], prep["center_scale"])
            try:
                sub = _opls_fit(Xtr, Yperm[train], n_orth)
            except EngineError:
                continue
            pred = ((prep["X"][test] - xmean) / xscale) @ sub["coef"]
            press += float(np.sum((Yperm[test] - pred) ** 2))
        null_q2[b] = 1.0 - press / tss
        perm_flat = Yperm.reshape(-1)
        denom = float(np.linalg.norm(perm_flat) * np.linalg.norm(y_flat))
        corr[b] = float(perm_flat @ y_flat / denom) if denom > 1e-18 else 0.0
    p_r2y = float((1 + int(np.sum(null_r2 >= obs_r2y))) / (1 + effective))
    p_q2 = float((1 + int(np.sum(null_q2 >= obs_q2))) / (1 + effective))

    def _intercept(values):
        try:
            _slope, intercept = np.polyfit(corr, values, 1)
            return float(intercept)
        except Exception:  # noqa: BLE001
            return float("nan")

    return {
        "n_perms": effective, "requested": int(n_perms),
        "r2y_obs": obs_r2y, "q2_obs": obs_q2,
        "r2y_null": null_r2.tolist(), "q2_null": null_q2.tolist(), "corr": corr.tolist(),
        "p_r2y": p_r2y, "p_q2": p_q2,
        "r2y_intercept": _intercept(null_r2), "q2_intercept": _intercept(null_q2),
        "note": note,
    }


def _attach_metrics(model: dict) -> None:
    prep = model["prep"]
    metrics = {
        "method": "OPLS-DA" if model["method"] == "oplsda" else "PLS-DA",
        "n_samples": prep["n_samples"],
        "n_features": prep["n_features"],
        "n_features_total": prep["n_features_total"],
        "n_comp": int(model.get("n_comp") or 0),
        "r2x": float(model.get("r2x_cum") or 0.0),
        "r2y": float(model.get("r2y_cum") or 0.0),
        "q2": float(model.get("q2_cum") or 0.0) if model.get("q2_cum") is not None else None,
        "scaling": model.get("scale_label"),
        "classes": len(model["classes"]),
    }
    if model["method"] == "oplsda":
        metrics["n_orth"] = int(model.get("n_orth") or 0)
        metrics["r2x_pred"] = float(model.get("r2x_pred") or 0.0)
    if model.get("cv"):
        cls = model["cv"].get("classification") or {}
        metrics["cv_accuracy"] = cls.get("accuracy")
        metrics["cv_auc"] = cls.get("auc")
        metrics["cv_folds"] = len(model["cv"].get("folds") or [])
    if model.get("permutation"):
        metrics["perm_n"] = model["permutation"]["n_perms"]
        metrics["perm_p_r2y"] = model["permutation"]["p_r2y"]
        metrics["perm_p_q2"] = model["permutation"]["p_q2"]
        metrics["perm_r2y_intercept"] = model["permutation"]["r2y_intercept"]
        metrics["perm_q2_intercept"] = model["permutation"]["q2_intercept"]
    model["metrics"] = metrics


def _vip_frame(prep: dict, vip: np.ndarray, threshold: float | None = None) -> pd.DataFrame:
    """VIP 表：rank / feature / name / mz / rt / vip（按 VIP 降序）。"""
    frame = prep["feature_meta"].copy()
    frame["vip"] = np.asarray(vip, dtype=float)
    frame = frame.sort_values("vip", ascending=False, kind="mergesort")
    out = pd.DataFrame({
        "rank": np.arange(1, len(frame) + 1),
        "feature": frame.index.astype(str),
        "name": frame["name"].to_numpy(),
        "mz": frame["mz"].to_numpy(),
        "rt": frame["rt"].to_numpy(),
        "vip": frame["vip"].to_numpy(),
    })
    if threshold is not None:
        out["above_threshold"] = out["vip"] >= float(threshold)
    return out.reset_index(drop=True)


def _splot_frame(prep: dict, fit: dict) -> pd.DataFrame:
    """S-plot 表：p1（预测载荷）与 p(corr)（与预测得分的相关载荷），供 Bylesjö 2006 语义使用。"""
    frame = prep["feature_meta"].copy()
    frame["loading_p1"] = np.asarray(fit["p_pred"], dtype=float)
    frame["p_corr"] = np.asarray(fit["p_corr"], dtype=float)
    frame["vip"] = np.asarray(fit["vip"], dtype=float)
    out = pd.DataFrame({
        "feature": frame.index.astype(str),
        "name": frame["name"].to_numpy(),
        "mz": frame["mz"].to_numpy(),
        "rt": frame["rt"].to_numpy(),
        "loading_p1": frame["loading_p1"].to_numpy(),
        "p_corr": frame["p_corr"].to_numpy(),
        "vip": frame["vip"].to_numpy(),
    })
    out["abs_p_corr"] = out["p_corr"].abs()
    out["abs_p1"] = out["loading_p1"].abs()
    return out.sort_values(["abs_p_corr", "abs_p1"], ascending=False).reset_index(drop=True)


def _scores_frame_pls(prep: dict, fit: dict) -> pd.DataFrame:
    frame = pd.DataFrame(fit["T"], index=pd.Index(prep["samples"], name="sample"))
    frame.columns = [f"T{j + 1}" for j in range(fit["T"].shape[1])]
    frame["group"] = prep["labels"]
    return frame


# ---------------------------------------------------------------------------
# 结果表 / 缓存
# ---------------------------------------------------------------------------


VIP_COLUMNS = [
    {"key": "rank", "label": "排名", "type": "number"},
    {"key": "feature", "label": "特征", "type": "string"},
    {"key": "name", "label": "名称", "type": "string"},
    {"key": "mz", "label": "m/z", "type": "number"},
    {"key": "rt", "label": "RT (min)", "type": "number"},
    {"key": "vip", "label": "VIP", "type": "number"},
]

SPLOT_COLUMNS = [
    {"key": "feature", "label": "特征", "type": "string"},
    {"key": "name", "label": "名称", "type": "string"},
    {"key": "mz", "label": "m/z", "type": "number"},
    {"key": "rt", "label": "RT (min)", "type": "number"},
    {"key": "loading_p1", "label": "p[1]", "type": "number"},
    {"key": "p_corr", "label": "p(corr)[1]", "type": "number"},
    {"key": "vip", "label": "VIP", "type": "number"},
]


def _perm_frame(perm: dict) -> pd.DataFrame:
    r2 = perm.get("r2y_null") or []
    q2 = perm.get("q2_null") or []
    corr = perm.get("corr") or []
    n = max(len(r2), len(q2))
    return pd.DataFrame({
        "perm": np.arange(1, n + 1),
        "r2y_null": r2[:n],
        "q2_null": q2[:n],
        "corr_with_original": corr[:n],
    })


def _cached_store(ctx) -> dict | None:
    results = ctx.get("results")
    if not isinstance(results, dict):
        return None
    store = results.get("plsda")
    return store if isinstance(store, dict) and store.get("method") else None


def _cached_model(ctx, params: dict) -> dict | None:
    """从 dataset.results["plsda"] 复原一个轻量 model（scores/vip/splot/permutation）。"""
    store = _cached_store(ctx)
    if not store:
        return None
    samples = [str(s) for s in (ctx.get("samples") or [])]
    scores = store.get("scores")
    vip = store.get("vip")
    if not isinstance(vip, pd.DataFrame):
        vip = store.get("table")
    labels = store.get("labels")
    if not isinstance(scores, pd.DataFrame) or not isinstance(vip, pd.DataFrame):
        return None
    if labels is None or len(labels) != len(samples):
        return None
    wanted = params.get("method")
    if wanted:
        try:
            wanted_method = _resolve_method(params)
        except EngineError:
            wanted_method = None
        if wanted_method and wanted_method != store.get("method"):
            return None
    model = {
        "method": store.get("method") or "plsda",
        "scores": scores,
        "vip": vip,
        "splot": store.get("splot") if isinstance(store.get("splot"), pd.DataFrame) else None,
        "permutation": store.get("permutation") if isinstance(store.get("permutation"), dict) else None,
        "cv": store.get("cv") if isinstance(store.get("cv"), dict) else None,
        "labels": np.asarray([str(x) for x in labels], dtype=object),
        "classes": list(store.get("classes") or []),
        "metrics": dict(store.get("metrics") or {}),
        "n_comp": int(store.get("n_comp") or 0),
        "n_orth": int(store.get("n_orth") or 0),
        "center_scale": store.get("center_scale") or "auto",
        "params": dict(params or {}),
        "warnings": [],
        "from_cache": True,
        # 缓存复用也要保留"真实实现"标记，否则再次出图/分析会误标成内置实现
        "impl_label": str(store.get("implementation") or ""),
        "implementation": ("ropls"
                           if str(store.get("implementation") or "").startswith("ropls")
                           else "builtin"),
        "ropls": store.get("ropls"),
        "loadings": store.get("loadings") if isinstance(store.get("loadings"), pd.DataFrame) else None,
    }
    model["prep"] = {"groups": model["classes"], "features": list(vip["feature"]),
                     "samples": samples, "n_samples": len(samples),
                     "n_features": int(len(vip)), "feature_meta": pd.DataFrame()}
    if not model["metrics"]:
        _attach_metrics(model)
    return model


def _model_for_render(ctx, params: dict, *, need: str = "scores") -> dict:
    """绘图/分析需要模型时：先尝试缓存的步骤结果，必要时重新拟合。"""
    cached = _cached_model(ctx, params)
    if cached is not None:
        if need == "scores" and isinstance(cached["scores"], pd.DataFrame):
            return cached
        if need == "vip" and isinstance(cached["vip"], pd.DataFrame):
            return cached
        if need == "splot" and isinstance(cached["splot"], pd.DataFrame):
            return cached
        if need == "permutation" and isinstance(cached["permutation"], dict):
            return cached
        if need == "cv" and isinstance(cached["cv"], dict):
            return cached
    if need == "permutation":
        return _fit_model(ctx, params, with_cv=True, with_permutation=True,
                          n_perms=params.get("n_perms") or params.get("permutations") or 200)
    if need == "cv":
        return _fit_model(ctx, params, with_cv=True,
                          cv_folds=params.get("folds") or params.get("cv_folds"))
    if need == "splot":
        forced = _pdict(params)
        forced["method"] = "oplsda"
        return _fit_model(ctx, forced, with_cv=False)
    return _fit_model(ctx, params, with_cv=False)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------


def _dpi(params: dict) -> int:
    return _int_param(params, "dpi", DEFAULT_DPI, 60, 400)


def _save(fig, outdir: str, name: str, params: dict, **extra) -> dict:
    fig_obj = save_fig(fig, outdir, name, title=None, dpi=_dpi(params))
    for key, value in extra.items():
        fig_obj[key] = value
    return fig_obj


def _hotelling_ellipse(points, level: float = 0.95):
    """组内 95% 置信椭圆（Hotelling T² 均值置信域）：返回 (m×2) 边界点或 None。"""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    m = pts.shape[0]
    if m < 3:
        return None
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    if not np.all(np.isfinite(cov)) or float(np.linalg.det(cov)) <= 1e-18:
        return None
    try:
        crit = float(stats.f.ppf(level, 2, m - 2))
        radius2 = (2.0 * (m - 1) / (m - 2)) * crit
    except Exception:  # noqa: BLE001
        radius2 = float(stats.chi2.ppf(level, 2))
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    vals = np.clip(vals, 0.0, None)
    radii = np.sqrt(vals * radius2)
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    circle = np.column_stack([np.cos(theta), np.sin(theta)]) * radii[None, :]
    return circle @ vecs.T + center


def _scores_axes(model: dict):
    """挑选 score 图的两个轴：PLS → T1/T2；OPLS → T_pred/T_orth1。"""
    scores = model["scores"]
    columns = [c for c in scores.columns if str(c) != "group"]
    if not columns:
        return None
    xcol = columns[0]
    ycol = columns[1] if len(columns) > 1 else None
    x = pd.to_numeric(scores[xcol], errors="coerce").to_numpy(dtype=float)
    if ycol is None:
        y = np.arange(len(scores), dtype=float)
        ylabel = "sample order"
    else:
        y = pd.to_numeric(scores[ycol], errors="coerce").to_numpy(dtype=float)
        ylabel = f"{ycol} score"
    return x, y, f"{xcol} score", ylabel


def _render_scores(model: dict, params: dict, outdir: str) -> dict:
    axes = _scores_axes(model)
    if axes is None:
        raise prerequisite("没有可用的得分向量")
    x, y, xlabel, ylabel = axes
    labels = np.asarray(model["labels"], dtype=object)
    groups = list(model["classes"]) or list(dict.fromkeys(map(str, labels)))
    colors = palette(max(len(groups), 1))
    metrics = model.get("metrics") or {}
    is_opls = model.get("method") == "oplsda"

    fig, ax = new_fig(6.8, 5.2)
    for i, group in enumerate(groups):
        mask = labels == str(group)
        if not mask.any():
            continue
        ax.scatter(x[mask], y[mask], s=36, color=colors[i % len(colors)],
                   edgecolor="white", linewidth=0.6, alpha=0.92,
                   label=f"{group} (n={int(mask.sum())})", zorder=3)
        ellipse = _hotelling_ellipse(np.column_stack([x[mask], y[mask]]))
        if ellipse is not None:
            ax.plot(ellipse[:, 0], ellipse[:, 1], color=colors[i % len(colors)],
                    linewidth=1.1, alpha=0.75, zorder=2)
    ax.axhline(0.0, color="#c8ccd4", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#c8ccd4", linewidth=0.8, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = "OPLS-DA score plot" if is_opls else "PLS-DA score plot"
    r2x = metrics.get("r2x")
    r2y = metrics.get("r2y")
    q2 = metrics.get("q2")
    extra = []
    if r2x is not None:
        extra.append(f"R2X={r2x:.3f}")
    if r2y is not None:
        extra.append(f"R2Y={r2y:.3f}")
    if q2 is not None:
        extra.append(f"Q2={q2:.3f}")
    subtitle = ("  ".join(extra) + "; ellipses = 95% Hotelling") if extra else "ellipses = 95% Hotelling"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    name = "plsda-scores-opls" if is_opls else "plsda-scores"
    return _save(fig, outdir, name, params, n_samples=int(len(labels)),
                 n_groups=int(len(groups)), points=int(len(labels)))


def _render_vip(model: dict, params: dict, outdir: str) -> dict:
    vip = model.get("vip")
    if not isinstance(vip, pd.DataFrame) or vip.empty:
        raise prerequisite("还没有 VIP 结果：先跑「PLS-DA / OPLS-DA」步骤")
    top_n = _int_param(params, "top_n", 30, 1, 200)
    threshold = _float_param(params, "threshold", 1.0)
    sub = vip.head(top_n).iloc[::-1]
    labels = [_short(str(v), 24) for v in sub["feature"]]
    values = pd.to_numeric(sub["vip"], errors="coerce").to_numpy(dtype=float)
    colors = ["#e15759" if value >= threshold else "#9aa4b2" for value in values]
    fig, ax = new_fig(7.4, max(3.4, 0.26 * len(sub) + 1.5))
    ax.barh(np.arange(len(sub)), values, color=colors, height=0.74, zorder=3)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(threshold, color="#4e79a7", linewidth=1.1, linestyle="--", zorder=4,
               label=f"threshold = {threshold:g}")
    ax.set_xlabel("VIP score")
    ax.set_title(f"Variable importance in projection (top {len(sub)})", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    n_above = int(np.sum(pd.to_numeric(vip["vip"], errors="coerce").to_numpy() >= threshold))
    return _save(fig, outdir, "vip-bar", params, n_above=n_above, n_features=int(len(vip)))


def _render_permutation(model: dict, params: dict, outdir: str) -> dict:
    perm = model.get("permutation")
    if not isinstance(perm, dict) or not perm.get("r2y_null"):
        raise prerequisite("还没有置换检验结果：先跑「PLS-DA / OPLS-DA」步骤或「PLS-DA 置换检验」分析")
    r2 = np.asarray(perm.get("r2y_null") or [], dtype=float)
    q2 = np.asarray(perm.get("q2_null") or [], dtype=float)
    r2 = r2[np.isfinite(r2)]
    q2 = q2[np.isfinite(q2)]
    fig, axes = new_fig(10.8, 4.4, ncols=2)
    axes = np.atleast_1d(axes)
    panels = (
        (axes[0], r2, float(perm.get("r2y_obs") or 0.0), float(perm.get("p_r2y") or 1.0),
         "R2Y", "#4e79a7"),
        (axes[1], q2, float(perm.get("q2_obs") or 0.0), float(perm.get("p_q2") or 1.0),
         "Q2", "#e15759"),
    )
    for ax, values, observed, p_value, label, color in panels:
        if values.size:
            ax.hist(values, bins=min(24, max(6, values.size // 4)), color=color,
                    alpha=0.55, edgecolor="white", zorder=2)
        ax.axvline(observed, color="#2b2f36", linewidth=1.6, linestyle="-", zorder=4,
                   label=f"observed = {observed:.3f}")
        ax.set_xlabel(f"permuted {label}")
        ax.set_ylabel("count")
        ax.set_title(f"{label} null distribution (p = {p_value:.4g})", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, outdir, "plsda-permutation", params,
                 n_perms=int(perm.get("n_perms") or len(r2)),
                 p_r2y=float(perm.get("p_r2y") or 1.0), p_q2=float(perm.get("p_q2") or 1.0))


def _render_splot(model: dict, params: dict, outdir: str) -> dict:
    splot = model.get("splot")
    if not isinstance(splot, pd.DataFrame) or splot.empty:
        raise prerequisite("S-plot 需要 OPLS-DA：请先用 method=oplsda 跑「PLS-DA / OPLS-DA」步骤")
    top_n = _int_param(params, "top_n", 20, 1, 100)
    x = pd.to_numeric(splot["loading_p1"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(splot["p_corr"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    features = splot["feature"].to_numpy()[finite]
    scale = np.abs(x)
    top = np.max(scale) if scale.size else 1.0
    fig, ax = new_fig(6.9, 5.4)
    scatter = ax.scatter(x, y, c=scale, cmap="RdYlBu_r", s=26, alpha=0.9,
                         edgecolor="white", linewidth=0.3,
                         vmin=0.0, vmax=top if top > 0 else 1.0, zorder=3)
    order = np.argsort(-np.abs(y))[:top_n]
    for idx in order:
        ax.annotate(_short(str(features[idx]), 18), (x[idx], y[idx]), fontsize=6.5,
                    color="#3a3f47", xytext=(3, 3), textcoords="offset points", zorder=4)
    ax.axhline(0.0, color="#c8ccd4", linewidth=0.8)
    ax.axvline(0.0, color="#c8ccd4", linewidth=0.8)
    ax.set_xlabel("p[1] (predictive loading)")
    ax.set_ylabel("p(corr)[1] (correlation loading)")
    ax.set_title("OPLS-DA S-plot (Bylesjo 2006)", fontsize=10)
    bar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("|p[1]|", fontsize=8)
    return _save(fig, outdir, "oplsda-splot", params,
                 n_features=int(len(x)), labeled=int(len(order)))


def _render_cv(model: dict, params: dict, outdir: str) -> dict:
    cv = model.get("cv")
    if not isinstance(cv, dict):
        raise prerequisite("还没有交叉验证结果：先跑「PLS-DA / OPLS-DA」步骤或「交叉验证」分析")
    cls = cv.get("classification") or {}
    confusion = np.asarray(cls.get("confusion") or [], dtype=float)
    classes = list(cls.get("classes") or model.get("classes") or [])
    if confusion.size == 0:
        raise prerequisite("交叉验证没有产生有效预测（样本或分组太少）")
    accuracy = float(cls.get("accuracy") or 0.0)
    auc = cls.get("auc")
    fig, ax = new_fig(5.6, 5.0)
    image = ax.imshow(confusion, cmap="Blues", aspect="equal")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion[i, j]
            ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=10,
                    color="white" if value > confusion.max() * 0.6 else "#2b2f36")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([_short(c, 12) for c in classes], fontsize=8)
    ax.set_yticklabels([_short(c, 12) for c in classes], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    title = f"Cross-validated confusion matrix (accuracy={accuracy:.3f}"
    title += f", AUC={auc:.3f})" if isinstance(auc, (int, float)) else ")"
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    return _save(fig, outdir, "plsda-cv-confusion", params,
                 accuracy=accuracy, auc=auc, n_folds=int(len(cv.get("folds") or [])))


# ---------------------------------------------------------------------------
# PLOTS 注册函数（kind -> fn(ctx, params, outdir) -> figure dict）
# ---------------------------------------------------------------------------


def plot_plsda_scores(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    model = _model_for_render(ctx, params, need="scores")
    return _render_scores(model, params, outdir)


def plot_vip_bar(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    model = _model_for_render(ctx, params, need="vip")
    return _render_vip(model, params, outdir)


def plot_permutation_test(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    if "n_perms" not in params and "permutations" not in params:
        params["n_perms"] = 200
    model = _model_for_render(ctx, params, need="permutation")
    return _render_permutation(model, params, outdir)


def plot_oplsda_splot(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    model = _model_for_render(ctx, params, need="splot")
    return _render_splot(model, params, outdir)


PLOTS = {
    "plsda_scores": plot_plsda_scores,
    "oplsda_splot": plot_oplsda_splot,
    "vip_bar": plot_vip_bar,
    "permutation_test": plot_permutation_test,
}


# ---------------------------------------------------------------------------
# STEP plsda
# ---------------------------------------------------------------------------


def _summary_bullets(model: dict) -> list[str]:
    prep = model["prep"]
    metrics = model.get("metrics") or {}
    method = "OPLS-DA" if model["method"] == "oplsda" else "PLS-DA"
    bullets = [
        f"实现：{_label_method(model)}",
        f"{method}：{prep['n_samples']} 样本 × {prep['n_features']} 特征，"
        f"{model.get('n_comp')} 个成分（{model.get('scale_label')}，分组列 {prep.get('group_key') or '—'}）",
        f"R²X(cum)={metrics.get('r2x', 0.0):.3f}，R²Y(cum)={metrics.get('r2y', 0.0):.3f}，"
        f"Q²(cum)={metrics.get('q2') if metrics.get('q2') is not None else 'NA'}",
    ]
    if isinstance(metrics.get("q2"), (int, float)):
        bullets[-1] = (f"R²X(cum)={metrics.get('r2x', 0.0):.3f}，R²Y(cum)={metrics.get('r2y', 0.0):.3f}，"
                       f"Q²(cum)={metrics['q2']:.3f}（{metrics.get('cv_folds', '—')} 折交叉验证）")
    if model["method"] == "oplsda":
        bullets.append(f"OPLS-DA：1 个预测成分 + {model.get('n_orth')} 个正交成分"
                       f"（R²X 预测部分 {metrics.get('r2x_pred', 0.0):.3f}）")
    perm = model.get("permutation")
    if isinstance(perm, dict):
        bullets.append(
            f"置换检验 {perm['n_perms']} 次：R²Y p={perm['p_r2y']:.4g}，Q² p={perm['p_q2']:.4g}"
            f"（R²Y 截距 {perm.get('r2y_intercept') if perm.get('r2y_intercept') is None else round(perm['r2y_intercept'], 3)}，"
            f"Q² 截距 {perm.get('q2_intercept') if perm.get('q2_intercept') is None else round(perm['q2_intercept'], 3)}）"
        )
        if perm.get("note"):
            model["warnings"].append(perm["note"])
    if model.get("cv"):
        cls = model["cv"].get("classification") or {}
        acc = cls.get("accuracy")
        auc = cls.get("auc")
        text = f"交叉验证准确率 {acc:.3f}" if isinstance(acc, (int, float)) else "交叉验证未产生有效预测"
        if isinstance(auc, (int, float)):
            text += f"，AUC {auc:.3f}"
        bullets.append(text)
    vip = model.get("vip")
    if isinstance(vip, pd.DataFrame) and not vip.empty:
        n_above = int(np.sum(pd.to_numeric(vip["vip"], errors="coerce").to_numpy() >= 1.0))
        bullets.append(f"VIP ≥ 1 的特征 {n_above} 个（共 {len(vip)} 个）")
    return bullets


def step_plsda(ctx, params=None, outdir=None):
    """步骤：PLS-DA / OPLS-DA（含 VIP、置换检验、交叉验证、S-plot）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    model = _fit_model(
        ctx, params, with_cv=True,
        cv_folds=params.get("cv_folds") or 7,
        with_permutation=True,
        n_perms=params.get("permutations") or 200,
    )
    prep = model["prep"]
    metrics = dict(model["metrics"])
    if model.get("cv"):
        metrics["cv_confusion"] = (model["cv"].get("classification") or {}).get("confusion")
        metrics["cv_pred_labels"] = (model["cv"].get("classification") or {}).get("pred_labels")
    warnings = list(dict.fromkeys(model.get("warnings") or []))

    vip_frame = model["vip"].copy()
    vip_frame["above_threshold"] = vip_frame["vip"] >= 1.0
    tables = [
        table("plsda_metrics", "模型指标",
              columns=[{"key": "metric", "label": "指标", "type": "string"},
                       {"key": "value", "label": "数值", "type": "string"}],
              rows=[{"metric": k, "value": v} for k, v in metrics.items()
                    if isinstance(v, (int, float, str, bool)) or v is None]),
        table("plsda_vip", "VIP（全部特征按 VIP 降序）", VIP_COLUMNS,
              rows=vip_frame.head(500).to_dict(orient="records"),
              note="VIP ≥ 1 通常认为对判别有贡献（Wold 1995）；完整表见 data_out"),
        table("plsda_r2q2", "每个成分的 R²X / R²Y / Q²",
              columns=[{"key": "component", "label": "成分", "type": "number"},
                       {"key": "r2x_cum", "label": "R²X(cum)", "type": "number"},
                       {"key": "r2y_cum", "label": "R²Y(cum)", "type": "number"},
                       {"key": "q2_cum", "label": "Q²(cum)", "type": "number"}],
              rows=_component_rows(model)),
    ]
    perm = model.get("permutation")
    if isinstance(perm, dict):
        tables.append(table("plsda_permutation", "置换检验零分布", [
            {"key": "perm", "label": "置换序号", "type": "number"},
            {"key": "r2y_null", "label": "R²Y(null)", "type": "number"},
            {"key": "q2_null", "label": "Q²(null)", "type": "number"},
            {"key": "corr_with_original", "label": "与原始 Y 的相关", "type": "number"},
        ], rows=_perm_frame(perm).to_dict(orient="records"),
            note=f"经验 p：R²Y={perm['p_r2y']:.4g}，Q²={perm['p_q2']:.4g}（Westerhuis 2008）"))

    figures = [_render_scores(model, params, outdir), _render_vip(model, params, outdir)]
    if isinstance(perm, dict):
        figures.append(_render_permutation(model, params, outdir))
    if model["method"] == "oplsda" and isinstance(model.get("splot"), pd.DataFrame):
        figures.append(_render_splot(model, params, outdir))
        tables.append(table("plsda_splot", "OPLS-DA S-plot（按 |p(corr)| 排序）", SPLOT_COLUMNS,
                            rows=model["splot"].head(200).to_dict(orient="records")))

    data_out = []
    try:
        path = export_csv(vip_frame, outdir, "plsda_vip", index=False)
        data_out.append(data_file("csv", path, "VIP 全表"))
        path = export_csv(model["scores"], outdir, "plsda_scores", index=True)
        data_out.append(data_file("csv", path, "得分表"))
        if isinstance(model.get("splot"), pd.DataFrame):
            path = export_csv(model["splot"], outdir, "plsda_splot", index=False)
            data_out.append(data_file("csv", path, "S-plot 全表"))
    except Exception as error:  # noqa: BLE001 - 导出失败不该阻断步骤
        warnings.append(f"结果 CSV 导出失败：{error}")

    store = {
        "plsda": {
            "method": model["method"],
            "n_comp": int(model.get("n_comp") or 0),
            "n_orth": int(model.get("n_orth") or 0),
            "center_scale": model["center_scale"],
            "scale_label": model["scale_label"],
            "group_key": prep.get("group_key"),
            "classes": list(model["classes"]),
            "labels": [str(x) for x in model["labels"]],
            "n_samples": int(prep["n_samples"]),
            "n_features": int(prep["n_features"]),
            "metrics": metrics,
            "r2x": model.get("r2x"),
            "r2y": model.get("r2y"),
            "q2": model.get("q2"),
            "vip": vip_frame,
            "table": vip_frame,
            "scores": model["scores"],
            "splot": model.get("splot"),
            "permutation": perm,
            "cv": model.get("cv"),
            # 结果结构新增（只增不改）：loadings 表 / 置换 p 值 / 真实实现标记
            "loadings": model.get("loadings"),
            "perm_p": ({"r2y": perm.get("p_r2y"), "q2": perm.get("p_q2"),
                        "n_perms": perm.get("n_perms")} if isinstance(perm, dict) else None),
            "implementation": _label_method(model),
            "ropls": model.get("ropls"),
            "opls": ({"n_orth": model.get("n_orth"), "r2x_pred": model.get("r2x_pred"),
                      "r2x_orth": model.get("r2x_orth"), "r2y": model.get("r2y_cum"),
                      "p_corr": model.get("p_corr"), "p1": model.get("p_pred")}
                     if model["method"] == "oplsda" else None),
            "params": params,
        },
        # results["vip"]：registry 的 _find_store/export 会对 store 值做 `if not stored`，
        # 裸 DataFrame 会触发 "truth value ambiguous"，所以这里存 dict（含 table/vip 两个键）；
        # DataFrame 本身仍在 results["plsda"]["vip"]，下游分析可直接取。
        "vip": {
            "table": vip_frame,
            "vip": vip_frame,
            "method": model["method"],
            "n_features": int(len(vip_frame)),
            "n_above": int((vip_frame["vip"] >= 1.0).sum()),
            "threshold": 1.0,
        },
    }
    if isinstance(model.get("splot"), pd.DataFrame):
        store["splot"] = model["splot"]
    if isinstance(perm, dict):
        store["permutation"] = perm
    if model.get("cv"):
        store["cv"] = model["cv"]

    payload = result(
        title="PLS-DA / OPLS-DA",
        bullets=_summary_bullets(model),
        metrics=metrics,
        tables=tables,
        figures=figures,
        warnings=warnings,
        notes=[f"用时 {timer.ms} ms；实现：{_label_method(model)}（method 字段写明真实实现，"
               f"ropls 不可用时才回退内置 NIPALS）"],
        data_out=data_out,
    )
    # store 在 result() 之后挂上：DataFrame 必须留在内存里（不能经过 jsonable），
    # registry._apply_result 会把它写进 dataset.results，下游模块与分析直接复用。
    payload["store"] = store
    _with_method(payload, model)
    return payload


def _component_rows(model: dict) -> list[dict]:
    """逐成分的累计 R²X/R²Y/Q²（PLS 来自 NIPALS；OPLS 只有 1 个预测成分 + 正交成分）。"""
    rows: list[dict] = []
    if model["method"] == "oplsda":
        r2x_pred = float(model.get("r2x_pred") or 0.0)
        orth = model.get("r2x_orth") or []
        r2y = float(model.get("r2y_cum") or model.get("r2y") or 0.0)
        q2 = model.get("q2_cum")
        rows.append({"component": 1, "r2x_cum": round(r2x_pred, 5),
                     "r2y_cum": round(r2y, 5),
                     "q2_cum": None if q2 is None else round(float(q2), 5)})
        acc = r2x_pred
        for j, value in enumerate(orth):
            acc += float(value)
            rows.append({"component": j + 2, "r2x_cum": round(acc, 5), "r2y_cum": None,
                         "q2_cum": None})
        return rows
    r2x = model.get("r2x") or []
    r2y = model.get("r2y") or []
    q2 = model.get("q2") or []
    for i in range(max(len(r2x), len(r2y))):
        rows.append({
            "component": i + 1,
            "r2x_cum": round(float(r2x[i]), 5) if i < len(r2x) else None,
            "r2y_cum": round(float(r2y[i]), 5) if i < len(r2y) else None,
            "q2_cum": round(float(q2[i]), 5) if i < len(q2) else None,
        })
    return rows


# ---------------------------------------------------------------------------
# ANALYSES
# ---------------------------------------------------------------------------


def analysis_plsda_vip(ctx, params=None, outdir=None):
    """VIP 表 + 条形图（Wold 1995）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    top_n = _int_param(params, "top_n", 30, 1, 200)
    threshold = _float_param(params, "threshold", 1.0)
    model = _model_for_render(ctx, params, need="vip")
    vip = model["vip"].copy()
    vip["above_threshold"] = vip["vip"] >= threshold
    n_above = int(vip["above_threshold"].sum())
    figure = _render_vip(model, {"top_n": top_n, "threshold": threshold}, outdir)
    top = vip.head(top_n)
    payload = result(
        title="变量投影重要度（VIP）",
        bullets=[f"实现：{_label_method(model)}",
                 f"VIP 阈值 {threshold:g}：{n_above} 个特征达到阈值（共 {len(vip)} 个）",
                 f"VIP 最高的 5 个特征：{'、'.join(map(str, top['feature'].head(5)))}",
                 "VIP 由 PLS-DA/OPLS-DA 模型计算（Wold 1995），VIP 高不等于统计显著，需结合差异分析"],
        metrics={"n_features": int(len(vip)), "threshold": threshold, "n_above": n_above,
                 "model_type": model.get("method"),
                 "implementation": _label_method(model)},
        tables=[table("vip", f"VIP Top {len(top)}", VIP_COLUMNS,
                      rows=top.to_dict(orient="records"))],
        figures=[figure],
        warnings=list(model.get("warnings") or []),
        notes=[f"用时 {timer.ms} ms"],
    )
    return _with_method(payload, model)


def analysis_oplsda_splot(ctx, params=None, outdir=None):
    """OPLS-DA S-plot（p[1] vs p(corr)[1]；Trygg & Wold 2002；Bylesjö 2006）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    forced = _pdict(params)
    forced["method"] = "oplsda"
    model = _fit_model(ctx, forced, with_cv=False)
    splot = model["splot"].copy()
    top_n = _int_param(params, "top_n", 20, 1, 100)
    figure = _render_splot(model, {"top_n": top_n}, outdir)
    top = splot.head(top_n)
    bullets = [
        f"实现：{_label_method(model)}",
        f"OPLS-DA：1 个预测成分 + {model.get('n_orth')} 个正交成分；"
        f"R²X 预测部分 {model.get('r2x_pred', 0.0):.3f}，R²Y {model.get('r2y_cum', 0.0):.3f}",
        f"p(corr) 绝对值最高的特征：{'、'.join(map(str, top['feature'].head(5)))}",
        "S-plot 右上/左下（高 |p1| + 高 |p(corr)|）是候选判别标志物（Bylesjö 2006）",
    ]
    payload = result(
        title="OPLS-DA S-plot",
        bullets=bullets,
        metrics={"n_features": int(len(splot)), "n_orth": int(model.get("n_orth") or 0),
                 "r2x_pred": model.get("r2x_pred"), "r2y": model.get("r2y_cum"),
                 "implementation": _label_method(model)},
        tables=[table("splot", f"S-plot Top {len(top)}", SPLOT_COLUMNS,
                      rows=top.to_dict(orient="records"))],
        figures=[figure],
        warnings=list(model.get("warnings") or []),
        notes=[f"用时 {timer.ms} ms"],
    )
    return _with_method(payload, model)


def analysis_plsda_permutation(ctx, params=None, outdir=None):
    """置换检验：R²Y/Q² 零分布 + 经验 p（Westerhuis 2008）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    n_perms = _int_param(params, "n_perms", 200, 10, 2000)
    model = _fit_model(ctx, params, with_cv=True, with_permutation=True, n_perms=n_perms)
    perm = model.get("permutation") or {}
    frame = _perm_frame(perm)
    figure = _render_permutation(model, params, outdir)
    bullets = [
        f"实现：{_label_method(model)}",
        f"置换 {perm.get('n_perms')} 次（请求 {perm.get('requested')} 次）："
        f"R²Y 经验 p={perm.get('p_r2y', 1.0):.4g}，Q² 经验 p={perm.get('p_q2', 1.0):.4g}",
        f"置换图回归截距：R²Y={_fmt(perm.get('r2y_intercept'))}，Q²={_fmt(perm.get('q2_intercept'))}"
        f"（截距越低说明模型越可靠；Westerhuis 2008）",
        "p < 0.05 表示模型区分能力优于随机标签",
    ]
    warnings = list(model.get("warnings") or [])
    if perm.get("note"):
        warnings.append(perm["note"])
    payload = result(
        title="PLS-DA 置换检验",
        bullets=bullets,
        metrics={**{k: perm.get(k) for k in ("n_perms", "requested", "r2y_obs", "q2_obs",
                                             "p_r2y", "p_q2", "r2y_intercept", "q2_intercept")},
                 "implementation": _label_method(model),
                 "perm_source": perm.get("source") or "builtin"},
        tables=[table("permutation", "置换检验零分布",
                      columns=[{"key": "perm", "label": "置换序号", "type": "number"},
                               {"key": "r2y_null", "label": "R²Y(null)", "type": "number"},
                               {"key": "q2_null", "label": "Q²(null)", "type": "number"},
                               {"key": "corr_with_original", "label": "与原始 Y 的相关", "type": "number"}],
                      rows=frame.to_dict(orient="records"))],
        figures=[figure],
        warnings=list(dict.fromkeys(warnings)),
        notes=[f"用时 {timer.ms} ms"],
    )
    return _with_method(payload, model)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def analysis_plsda_cv(ctx, params=None, outdir=None):
    """k 折交叉验证：混淆矩阵 + 准确率 + AUC（Brindle 2002 判别思路）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    folds = _int_param(params, "folds", 7, 2, 50)
    model = _fit_model(ctx, params, with_cv=True, cv_folds=folds)
    cv = model.get("cv") or {}
    cls = cv.get("classification") or {}
    classes = list(cls.get("classes") or model["classes"])
    confusion = cls.get("confusion") or []
    rows = []
    for i, cls_name in enumerate(classes):
        row = {"actual": cls_name}
        for j, pred_name in enumerate(classes):
            try:
                row[str(pred_name)] = int(confusion[i][j])
            except (IndexError, TypeError):
                row[str(pred_name)] = 0
        rows.append(row)
    columns = [{"key": "actual", "label": "实际", "type": "string"}] + [
        {"key": str(c), "label": str(c), "type": "number"} for c in classes]
    figure = _render_cv(model, params, outdir)
    accuracy = cls.get("accuracy")
    auc = cls.get("auc")
    bullets = [
        f"实现：{_label_method(model)}",
        f"{folds} 折交叉验证：准确率 {_fmt(accuracy)}" + (f"，AUC {_fmt(auc)}" if auc is not None else ""),
        f"模型 Q²(cum)={_fmt(model.get('q2_cum'))}（ropls crossvalI 与 Python 分层 k 折各自给出；"
        f"混淆矩阵来自 Python 分层 k 折）",
        "混淆矩阵对角线为正确判别数；准确率明显高于 1/组数 才说明判别有效",
    ]
    payload = result(
        title="交叉验证与混淆矩阵",
        bullets=bullets,
        metrics={"folds": folds, "accuracy": accuracy, "auc": auc,
                 "q2": model.get("q2_cum"), "n_samples": int(model["prep"]["n_samples"])},
        tables=[
            table("cv_confusion", "交叉验证混淆矩阵", columns, rows=rows),
            table("cv_per_fold", "每折 Q² / PRESS",
                  columns=[{"key": "component", "label": "成分", "type": "number"},
                           {"key": "q2_cum", "label": "Q²(cum)", "type": "number"},
                           {"key": "press", "label": "PRESS", "type": "number"}],
                  rows=[{"component": i + 1,
                         "q2_cum": round(float(q), 5),
                         "press": round(float(p), 6)}
                        for i, (q, p) in enumerate(zip(cv.get("q2") or [], cv.get("press") or []))]),
        ],
        figures=[figure],
        warnings=list(model.get("warnings") or []),
        notes=[f"用时 {timer.ms} ms"],
    )
    return _with_method(payload, model)


def analysis_plsda_multiclass(ctx, params=None, outdir=None):
    """多组判别：多类 PLS-DA（one-hot PLS2）+ 一类对其余 VIP 与 AUC。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    n_comp = _int_param(params, "n_comp", 3, 1, 20)
    base = _pdict(params)
    base["method"] = "plsda"
    base["n_comp"] = n_comp
    model = _fit_model(ctx, base, with_cv=True, cv_folds=params.get("cv_folds") or 7)
    prep = model["prep"]
    groups = list(model["classes"])
    warnings = list(model.get("warnings") or [])
    if len(groups) < 2:
        raise prerequisite("多组判别至少需要 2 个分组")
    if len(groups) == 2:
        warnings.append("当前只有 2 个分组，多类判别退化为二分类 PLS-DA")

    # 一类对其余（OVR）：每个类别单独拟合 PLS1，取 VIP 与 CV AUC
    ovr_rows = []
    vip_matrix = np.zeros((prep["n_features"], len(groups)))
    classes_idx = [str(g) for g in groups]
    folds = _cv_folds(prep["labels"], int(params.get("cv_folds") or 7), np.random.default_rng(SEED))
    for ci, cls_name in enumerate(classes_idx):
        y_bin = np.array([1.0 if str(lab) == cls_name else 0.0 for lab in prep["labels"]])
        y_c = y_bin - y_bin.mean()
        Y = y_c.reshape(-1, 1)
        try:
            fit = _pls2_nipals(prep["Xp"], Y, n_comp)
            vip_c = _vip(fit["W"], fit["T"], fit["Q"])
            vip_matrix[:, ci] = vip_c
            # OVR 交叉验证得分
            oof = np.full(prep["n_samples"], np.nan)
            n = prep["n_samples"]
            for test in folds:
                train_mask = np.ones(n, dtype=bool)
                train_mask[test] = False
                train = np.flatnonzero(train_mask)
                if train.size < 3:
                    continue
                Xtr, xmean, xscale, _ = _preprocess(prep["X"][train], prep["center_scale"])
                sub = _pls2_nipals(Xtr, Y[train], fit["n_comp"])
                coef = _pls_coef(sub["W"], sub["P"], sub["Q"])
                oof[test] = ((prep["X"][test] - xmean) / xscale) @ coef[:, 0]
            auc = None
            try:
                from sklearn.metrics import roc_auc_score

                ok = np.isfinite(oof)
                if ok.any() and len(np.unique(y_bin[ok])) == 2:
                    auc = float(roc_auc_score(y_bin[ok], oof[ok]))
            except Exception:  # noqa: BLE001
                auc = None
            ovr_rows.append({
                "class": cls_name,
                "n": int(np.sum(prep["labels"] == cls_name)),
                "n_comp": int(fit["n_comp"]),
                "r2y": round(float(fit["r2y"][-1]), 5),
                "auc_ovr": None if auc is None else round(auc, 5),
                "top_vip_feature": str(prep["features"][int(np.argmax(vip_c))]),
                "top_vip": round(float(np.max(vip_c)), 5),
            })
        except EngineError as error:
            warnings.append(f"类别「{cls_name}」的一对其余拟合失败：{error.message}")

    combined = np.sqrt(np.mean(vip_matrix ** 2, axis=1)) if vip_matrix.size else np.zeros(prep["n_features"])
    combined_frame = prep["feature_meta"].copy()
    combined_frame["vip_combined"] = combined
    for ci, cls_name in enumerate(classes_idx):
        combined_frame[f"vip_{cls_name}"] = vip_matrix[:, ci]
    combined_frame = combined_frame.sort_values("vip_combined", ascending=False)
    top = pd.DataFrame({
        "rank": np.arange(1, len(combined_frame) + 1),
        "feature": combined_frame.index.astype(str),
        "name": combined_frame["name"].to_numpy(),
        "mz": combined_frame["mz"].to_numpy(),
        "rt": combined_frame["rt"].to_numpy(),
        "vip_combined": combined_frame["vip_combined"].to_numpy(),
    }).head(200)

    cv = model.get("cv") or {}
    cls = cv.get("classification") or {}
    figure = _render_scores(model, params, outdir)
    vip_figure = _render_vip(model, params, outdir)
    bullets = [
        f"实现：{_label_method(model)}",
        f"多类 PLS-DA（one-hot PLS2，{n_comp} 成分）：{len(groups)} 个分组（{'、'.join(classes_idx)}）",
        f"交叉验证准确率 {_fmt(cls.get('accuracy'))}，多类 AUC(macro OVR) {_fmt(cls.get('auc'))}",
        f"综合 VIP（各类别 VIP 的均方根）最高的特征：{'、'.join(map(str, top['feature'].head(5)))}",
    ]
    payload = result(
        title="多组判别（多类 PLS-DA）",
        bullets=bullets,
        metrics={"n_groups": len(groups), "n_comp": n_comp,
                 "accuracy": cls.get("accuracy"), "auc_macro": cls.get("auc"),
                 "confusion": cls.get("confusion")},
        tables=[
            table("multiclass_ovr", "一类对其余（OVR）判别指标",
                  columns=[{"key": "class", "label": "类别", "type": "string"},
                           {"key": "n", "label": "样本数", "type": "number"},
                           {"key": "n_comp", "label": "成分", "type": "number"},
                           {"key": "r2y", "label": "R²Y", "type": "number"},
                           {"key": "auc_ovr", "label": "AUC(OVR)", "type": "number"},
                           {"key": "top_vip_feature", "label": "该类别 VIP 最高特征", "type": "string"},
                           {"key": "top_vip", "label": "最高 VIP", "type": "number"}],
                  rows=ovr_rows),
            table("multiclass_vip", "综合 VIP（各类别均方根）", VIP_COLUMNS[:5] + [
                {"key": "vip", "label": "VIP", "type": "number"}],
                rows=[{"rank": int(r["rank"]), "feature": r["feature"], "name": r["name"],
                       "mz": r["mz"], "rt": r["rt"], "vip": r["vip_combined"]}
                      for r in top.to_dict(orient="records")]),
        ],
        figures=[figure, vip_figure],
        warnings=list(dict.fromkeys(warnings)),
        notes=[f"用时 {timer.ms} ms"],
    )
    return _with_method(payload, model)


def _pca_svd(prep: dict, pc: int) -> dict:
    """内置 PCA（SVD）：返回 {loadings, ratio, label}。"""
    try:
        _u, s, vt = np.linalg.svd(prep["Xp"], full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise EngineError("internal", f"PCA（SVD）失败：{error}")
    if s.size == 0:
        raise EngineError("internal", "PCA 没有提取到主成分")
    var = (s ** 2) / max(prep["n_samples"] - 1, 1)
    ratio = var / max(float(var.sum()), 1e-18)
    if pc > vt.shape[0]:
        raise prerequisite(f"只有 {vt.shape[0]} 个主成分，无法取第 {pc} 个（请减小 pc）")
    return {"loadings": np.asarray(vt[pc - 1], dtype=float), "ratio": ratio,
            "label": "内置 SVD 实现（ropls 不可用）"}


def _pca_ropls(ctx, params: dict, prep: dict, pc: int) -> dict:
    """ropls 官方 PCA（ropls::opls，y=NULL）：返回 {loadings, ratio, label}。

    ropls 不可用抛 mb_rbridge.RUnavailable；运行期错误按 runtime 上抛（不静默换算法）。
    """
    n_comp = int(max(int(pc), 5))
    payload = {
        "expr": prep["X"],
        "features": prep["features"],
        "samples": prep["samples"],
        "info": _r_info_frame(ctx, prep),
        "params": {
            "mode": "pca",
            "n_comp": n_comp,
            "orth_comp": 0,
            "center_scale": str(prep["center_scale"]),
            "cv_folds": 7,
            "permutations": 0,
            "seed": int(SEED),
            "group_col": "_mb_group",
        },
    }
    rres = rb.run_r(R_SCRIPT, payload, outdir=_r_outdir(ctx, "pca"), timeout=R_TIMEOUT)
    r_payload = rres.get("result") or {}
    tables = rres.get("tables") or {}
    loadings_r = tables.get("loadings")
    pca_var = r_payload.get("pca_var") or []
    total = _num(r_payload.get("pca_total_var"))
    if loadings_r is None or not pca_var or not total:
        raise EngineError("internal", "ropls 的 PCA 返回缺少 pca_var/loadings")
    ratio = np.asarray([_num(v, 0.0) for v in pca_var], dtype=float) / max(float(total), 1e-18)
    if pc > len(ratio):
        raise prerequisite(f"只有 {len(ratio)} 个主成分，无法取第 {pc} 个（请减小 pc）")
    column = f"loading_p{pc}"
    if column not in loadings_r.columns:
        raise EngineError("internal", f"ropls 的 PCA 返回缺少 {column} 列")
    aligned = loadings_r.reindex(pd.Index([str(f) for f in prep["features"]]))
    loadings = pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=float)
    loadings = np.where(np.isfinite(loadings), loadings, 0.0)
    version = str(r_payload.get("ropls_version") or "").strip()
    label = f"ropls::opls（PCA 模式，predI={int(r_payload.get('predI') or n_comp)}，ropls {version}）"
    return {"loadings": loadings, "ratio": ratio, "label": label,
            "ropls_version": r_payload.get("ropls_version")}


def analysis_pca_loadings(ctx, params=None, outdir=None):
    """PCA 载荷与贡献：优先 ropls 官方 PCA，回退内置 SVD（Wold 1994）。

    载荷表与贡献率两种实现同构：contribution_pct = 载荷² / Σ载荷² × 100，
    explained_variance = 该主成分方差 / 全部主成分方差。
    """
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    pc = _int_param(params, "pc", 1, 1, 10000)
    top_n = _int_param(params, "top_n", 20, 1, 200)
    prep = _prepare(ctx, center_scale=str(params.get("center_scale") or "auto"),
                    require_groups=False)
    impl = _impl_param(params)
    result_block = None
    model: dict = {}
    if impl in ("auto", "ropls"):
        try:
            result_block = _pca_ropls(ctx, params, prep, pc)
            model = {"impl_label": str(result_block.get("label") or "ropls::opls（PCA 模式）")}
        except rb.RUnavailable as error:
            if impl == "ropls":
                raise EngineError("unsupported",
                                  f"ropls 官方实现不可用（implementation=ropls）：{error.message}")
            model = {"impl_label": f"内置 SVD 实现（ropls 不可用）—{error.message}"}
    if result_block is None:
        result_block = _pca_svd(prep, pc)
        if not model:
            model = {"impl_label": str(result_block.get("label") or "内置 SVD 实现")}
    loadings = np.asarray(result_block["loadings"], dtype=float)
    ratio = np.asarray(result_block["ratio"], dtype=float)
    contribution = (loadings ** 2) / max(float(np.sum(loadings ** 2)), 1e-18) * 100.0
    frame = prep["feature_meta"].copy()
    frame["loading"] = loadings
    frame["contribution_pct"] = contribution
    frame["abs_loading"] = np.abs(loadings)
    frame = frame.sort_values("abs_loading", ascending=False)
    out = pd.DataFrame({
        "rank": np.arange(1, len(frame) + 1),
        "feature": frame.index.astype(str),
        "name": frame["name"].to_numpy(),
        "mz": frame["mz"].to_numpy(),
        "rt": frame["rt"].to_numpy(),
        "loading": frame["loading"].to_numpy(),
        "contribution_pct": frame["contribution_pct"].to_numpy(),
    })
    head = out.head(top_n).iloc[::-1]
    fig, ax = new_fig(7.2, max(3.4, 0.26 * len(head) + 1.5))
    values = head["loading"].to_numpy(dtype=float)
    ax.barh(np.arange(len(head)), values,
            color=["#e15759" if value >= 0 else "#4e79a7" for value in values],
            height=0.74, zorder=3)
    ax.set_yticks(np.arange(len(head)))
    ax.set_yticklabels([_short(v, 24) for v in head["feature"]], fontsize=7)
    ax.axvline(0.0, color="#c8ccd4", linewidth=0.8)
    ax.set_xlabel(f"PC{pc} loading")
    ax.set_title(f"PCA loadings PC{pc} (explained variance {ratio[pc - 1]:.1%})", fontsize=10)
    ax.grid(axis="y", visible=False)
    figure = _save(fig, outdir, "pca-loadings", params, pc=pc,
                   explained_variance=float(ratio[pc - 1]))
    payload = result(
        title=f"PCA 载荷与贡献（PC{pc}）",
        bullets=[f"PC{pc} 解释方差 {ratio[pc - 1]:.1%}（前 {min(5, len(ratio))} 个主成分累计 {ratio[:5].sum():.1%}）",
                 f"|载荷| 最大的 5 个特征：{'、'.join(map(str, out['feature'].head(5)))}",
                 "载荷绝对值大说明该特征对当前主成分贡献大（Wold 1994）；载荷方向与得分方向一致"],
        metrics={"pc": pc, "explained_variance": float(ratio[pc - 1]),
                 "explained_variance_5": float(ratio[:5].sum()),
                 "n_features": int(len(out)), "scaling": prep["scale_label"]},
        tables=[table(f"pca_loadings_pc{pc}", f"PC{pc} 载荷 Top {len(head)}",
                      columns=[{"key": "rank", "label": "排名", "type": "number"},
                               {"key": "feature", "label": "特征", "type": "string"},
                               {"key": "name", "label": "名称", "type": "string"},
                               {"key": "mz", "label": "m/z", "type": "number"},
                               {"key": "rt", "label": "RT (min)", "type": "number"},
                               {"key": "loading", "label": "载荷", "type": "number"},
                               {"key": "contribution_pct", "label": "贡献(%)", "type": "number"}],
                      rows=head.iloc[::-1].to_dict(orient="records"))],
        figures=[figure],
        warnings=list(prep["warnings"]),
        notes=[f"用时 {timer.ms} ms；实现：{_label_method(model)}"],
    )
    return _with_method(payload, model)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

STEP_PARAMS = [
    {"key": "method", "label": "方法", "type": "select", "default": "plsda",
     "options": [{"value": "plsda", "label": "PLS-DA"},
                 {"value": "oplsda", "label": "OPLS-DA"}]},
    {"key": "n_comp", "label": "成分数", "type": "number", "default": 3},
    {"key": "cv_folds", "label": "交叉验证折数", "type": "number", "default": 7},
    {"key": "permutations", "label": "置换次数", "type": "number", "default": 200},
    {"key": "center_scale", "label": "标度", "type": "select", "default": "auto",
     "options": [{"value": "auto", "label": "auto（单位方差）"},
                 {"value": "pareto", "label": "Pareto"},
                 {"value": "none", "label": "仅中心化"}]},
    {"key": "group_col", "label": "分组列", "type": "text", "default": None},
]

_METHOD_PARAM = {"key": "method", "label": "方法（留空=沿用步骤结果）", "type": "select",
                 "default": None,
                 "options": [{"value": "plsda", "label": "PLS-DA"},
                             {"value": "oplsda", "label": "OPLS-DA"}]}
_NCOMP_PARAM = {"key": "n_comp", "label": "成分数（留空=沿用）", "type": "number", "default": None}

STEPS = [
    {
        "id": "plsda",
        "title": "PLS-DA / OPLS-DA",
        "requires": ["transform"],
        "params": STEP_PARAMS,
        "run": step_plsda,
    },
]

ANALYSES = [
    {
        "id": "plsda_vip",
        "title": "变量投影重要度（VIP）",
        "group": "多变量判别",
        "method": "VIP（Wold 1995；ropls::opls 的 vipVn；ropls 不可用时回退内置 NIPALS）",
        "needs": ["plsda"],
        "params": [
            {"key": "top_n", "label": "Top N", "type": "number", "default": 30},
            {"key": "threshold", "label": "VIP 阈值", "type": "number", "default": 1},
            _METHOD_PARAM, _NCOMP_PARAM,
        ],
        "run": analysis_plsda_vip,
    },
    {
        "id": "oplsda_splot",
        "title": "OPLS-DA S-plot",
        "group": "多变量判别",
        "method": "OPLS-DA S-plot（Trygg & Wold 2002；Bylesjö 2006；ropls::opls 的 p1/p(corr)，不可用时回退内置）",
        "needs": ["plsda"],
        "params": [
            {"key": "top_n", "label": "标注特征数", "type": "number", "default": 20},
            {"key": "n_comp", "label": "总成分数（1 预测 + 其余正交）", "type": "number", "default": 3},
        ],
        "run": analysis_oplsda_splot,
    },
    {
        "id": "plsda_permutation",
        "title": "PLS-DA 置换检验",
        "group": "多变量判别",
        "method": "标签置换 + R²Y/Q² 零分布（Westerhuis 2008；ropls permI 的 pR2Y/pQ2 与零分布，不可用时回退内置）",
        "needs": ["plsda"],
        "params": [
            {"key": "n_perms", "label": "置换次数", "type": "number", "default": 200},
            _NCOMP_PARAM,
        ],
        "run": analysis_plsda_permutation,
    },
    {
        "id": "plsda_cv",
        "title": "交叉验证与混淆矩阵",
        "group": "多变量判别",
        "method": "分层 k 折 + 混淆矩阵 + AUC（Brindle 2002；Python 分层 k 折，模型 Q² 来自 ropls crossvalI）",
        "needs": ["plsda"],
        "params": [
            {"key": "folds", "label": "折数", "type": "number", "default": 7},
            _NCOMP_PARAM,
        ],
        "run": analysis_plsda_cv,
    },
    {
        "id": "plsda_multiclass",
        "title": "多组判别",
        "group": "多变量判别",
        "method": "多类 PLS-DA（ropls::opls 多类 PLS2；一类对其余 VIP/AUC）",
        "needs": ["groups"],
        "params": [
            {"key": "n_comp", "label": "成分数", "type": "number", "default": 3},
            {"key": "cv_folds", "label": "交叉验证折数", "type": "number", "default": 7},
            {"key": "group_col", "label": "分组列", "type": "text", "default": None},
        ],
        "run": analysis_plsda_multiclass,
    },
    {
        "id": "pca_loadings",
        "title": "PCA 载荷与贡献",
        "group": "多变量判别",
        "method": "PCA 载荷与贡献（Wold 1994；ropls::opls PCA 优先，不可用时回退内置 SVD）",
        "needs": ["transform"],
        "params": [
            {"key": "pc", "label": "主成分", "type": "number", "default": 1},
            {"key": "top_n", "label": "Top N", "type": "number", "default": 20},
            {"key": "center_scale", "label": "标度", "type": "select", "default": "auto",
             "options": [{"value": "auto", "label": "auto（单位方差）"},
                         {"value": "pareto", "label": "Pareto"},
                         {"value": "none", "label": "仅中心化"}]},
        ],
        "run": analysis_pca_loadings,
    },
]

__all__ = [
    "STEPS", "ANALYSES", "PLOTS",
    "step_plsda", "analysis_plsda_vip", "analysis_oplsda_splot", "analysis_plsda_permutation",
    "analysis_plsda_cv", "analysis_plsda_multiclass", "analysis_pca_loadings",
    "plot_plsda_scores", "plot_oplsda_splot", "plot_vip_bar", "plot_permutation_test",
    "_pls2_nipals", "_opls_fit", "_vip", "_hotelling_ellipse", "_fit_model",
]
