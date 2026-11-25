#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — P0-2 回归测试：全样本预筛泄漏 vs 训练折内筛选

对应 BIO_P0_BACKLOG.md §4 P0-2：
  * 新的 `mb_biomarker.nested_cv_auc` 必须做「训练折内填补 + 标度 + Cohen's d 筛选 + 模型 +
    阈值」的嵌套交叉验证；纯噪声（60×6000）上 AUC 应回到 ~0.5–0.6，而不是全样本预筛的
    ~0.99；
  * 按患者/受试者（subject/patient/donor）分组划分，避免同一个体的重复样本跨折；
  * `step_biomarker` 的 metrics 拆分 apparent_auc / nested_cv_auc / external_auc（未提供外部
    验证集时明确注明）；
  * `analysis_model_validation` 的自助法按 subject 重抽并重做整条流程。

用法：cd plugins/dsh-metabolomics && MPLBACKEND=Agg python3 test/test_p0_leakage.py
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

import mb_biomarker as B  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="mb-p0-leak-")
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
# 纯噪声数据
# ---------------------------------------------------------------------------


def noise_xy(seed: int = 0, n: int = 60, p: int = 6000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.asarray([0] * (n // 2) + [1] * (n - n // 2), dtype=int)
    return X, y


def leaky_whole_batch_auc(X: np.ndarray, y: np.ndarray, top: int = 20, folds: int = 5) -> float:
    """旧路径复现：全样本 Cohen's d 预筛 → 在预筛池上做 CV（泄漏对照）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pos = X[y == 1]
    neg = X[y == 0]
    mean_pos = pos.mean(axis=0)
    mean_neg = neg.mean(axis=0)
    pooled = np.sqrt((pos.var(axis=0, ddof=1) + neg.var(axis=0, ddof=1)) / 2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(pooled > 0, np.abs(mean_pos - mean_neg) / pooled, 0.0)
    picked = np.argsort(-np.asarray(score, dtype=float))[:top]
    model = Pipeline([("scaler", StandardScaler()),
                      ("logit", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced",
                                                   random_state=0))])
    proba = cross_val_predict(model, X[:, picked], y, cv=StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=0), method="predict_proba")
    ranks = pd.Series(proba[:, 1]).rank(method="average").to_numpy()
    pos_ranks = ranks[y == 1]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# ctx 构造（step_biomarker / model_validation 用）
# ---------------------------------------------------------------------------


def make_biomarker_ctx(X, y, subjects=None, features=None, X_kind="lognorm"):
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    features = list(features) if features is not None else [f"F{j + 1:05d}" for j in range(p)]
    samples = [f"S{i + 1:03d}" for i in range(n)]
    group = np.where(np.asarray(y, dtype=int) == 1, "case", "control")
    info = pd.DataFrame({"group": group}, index=pd.Index(samples, name="sample"))
    if subjects is not None:
        info["subject"] = [str(s) for s in subjects]
    feature_info = pd.DataFrame(
        {"name": features, "mz": np.nan, "rt": np.nan},
        index=pd.Index(features, name="feature"),
    )
    rng = np.random.default_rng(99)
    de_table = pd.DataFrame({
        "feature": features,
        "log2FC": rng.normal(0.0, 0.2, p),
        "p": rng.uniform(0, 1, p),
        "p_adj": rng.uniform(0, 1, p),
        "mean_case": np.nan,
        "mean_control": np.nan,
    })
    return {
        "dataset_id": "ds_p0_leak",
        "name": "p0-leakage",
        "features": features,
        "samples": samples,
        "X": np.asarray(X, dtype=np.float32),
        "X_kind": X_kind,
        "raw": None,
        "feature_info": feature_info,
        "sample_info": info,
        "group_key": "group",
        "groups": ["case", "control"],
        "matrix_order": "samples_by_features",
        "species": "human",
        "results": {"de": {"table": de_table, "case": "case", "control": "control",
                           "log2fc_basis": "log2-mean-difference", "paired": False}},
        "library": None,
        "rng": np.random.default_rng(0),
        "workdir": OUTDIR,
        "outdir": OUTDIR,
        "engine": {"python": sys.executable, "mode": "builtin"},
        "notes": [],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# 1) nested_cv_auc 本体
# ---------------------------------------------------------------------------


def test_nested_cv_noise() -> None:
    print("\n[1] 纯噪声 60×6000：折内筛选后 AUC 应回到 ~0.5–0.6")
    fn = getattr(B, "nested_cv_auc", None)
    check("mb_biomarker.nested_cv_auc 存在", callable(fn), str(type(fn)))
    if not callable(fn):
        return
    X, y = noise_xy(seed=0, n=60, p=6000)
    try:
        leaky = leaky_whole_batch_auc(X, y)
    except Exception:  # noqa: BLE001
        leaky = float("nan")
    print(f"    （对照：全样本 Cohen's d 预筛后 CV AUC = {leaky:.4f}）")
    check("对照：全样本监督预筛在纯噪声上 AUC≈0.99（泄漏复现）",
          np.isfinite(leaky) and leaky > 0.9, f"{leaky:.4f}")
    try:
        res = fn(X, y, folds=5, seed=0)
        auc = float(res["auc"])
        check("纯噪声 nested AUC 落在 [0.3, 0.75]", 0.3 <= auc <= 0.75, f"{auc:.4f}")
        check("nested AUC 显著低于整批预筛（差 > 0.2）",
              np.isfinite(leaky) and leaky - auc > 0.2, f"{leaky:.3f} - {auc:.3f}")
        check("返回每折 AUC 与折数", len(res.get("fold_aucs") or []) == res.get("folds"))
    except Exception:  # noqa: BLE001
        check("纯噪声 nested CV 可跑", False, traceback.format_exc(limit=3))

    # 多随机种子：不允许出现某次 0.9+ 的假信号
    aucs = []
    for seed in (1, 2, 3):
        Xs, ys = noise_xy(seed=seed, n=60, p=2000)
        try:
            aucs.append(float(fn(Xs, ys, folds=5, seed=seed)["auc"]))
        except Exception:  # noqa: BLE001
            aucs.append(float("nan"))
    finite = [v for v in aucs if np.isfinite(v)]
    check("3 个噪声种子的 nested AUC 全部 ≤ 0.8",
          len(finite) == 3 and all(v <= 0.8 for v in finite),
          ", ".join(f"{v:.3f}" for v in aucs))
    check("3 个噪声种子的 nested AUC 均值 ∈ [0.4, 0.68]",
          bool(finite) and 0.4 <= float(np.mean(finite)) <= 0.68,
          f"mean={np.mean(finite):.3f}" if finite else "无有效值")


def test_group_aware_split() -> None:
    print("\n[2] 按受试者分组划分：同一个体的重复样本不得跨折")
    fn = getattr(B, "nested_cv_auc", None)
    if not callable(fn):
        check("nested_cv_auc 存在（分组检查）", False, "缺少函数")
        return
    rng = np.random.default_rng(5)
    n_subjects, n_features = 20, 600
    per_subject = 3   # 每个体 3 次重复：样本级划分时同个体样本几乎必然跨折（泄漏演示更明确）
    subject_offset = rng.normal(0.0, 3.0, (n_subjects, n_features))
    y_subject = np.asarray([0, 1] * (n_subjects // 2), dtype=int)
    rng.shuffle(y_subject)
    X = np.vstack([subject_offset[s] for s in range(n_subjects) for _ in range(per_subject)])
    y = np.repeat(y_subject, per_subject)
    subjects = [f"P{s + 1:02d}" for s in range(n_subjects) for _ in range(per_subject)]
    # 无分组：同个体既在训练又在测试 → 可以记住个体效应 → AUC 高
    sample_auc = float(fn(X, y, folds=5, seed=0)["auc"])
    # 有分组：个体不跨折 → 应回到随机水平
    grouped_auc = float(fn(X, y, groups=np.asarray(subjects), folds=5, seed=0)["auc"])
    check("样本级划分会泄漏个体效应（AUC > 0.85）", sample_auc > 0.85, f"{sample_auc:.3f}")
    check("按 subject 分组划分后 AUC 回落到 ≤ 0.7", grouped_auc <= 0.7, f"{grouped_auc:.3f}")
    check("分组划分的 AUC 明显低于样本级划分", sample_auc - grouped_auc > 0.2,
          f"{sample_auc:.3f} - {grouped_auc:.3f}")


# ---------------------------------------------------------------------------
# 3) step_biomarker：metrics 拆分
# ---------------------------------------------------------------------------


def test_step_biomarker_metrics() -> None:
    print("\n[3] step_biomarker：apparent / nested / external 三套指标")
    X, y = noise_xy(seed=0, n=60, p=800)
    subjects = [f"P{i // 2 + 1:02d}" for i in range(60)]
    ctx = make_biomarker_ctx(X, y, subjects=subjects)
    try:
        payload = B.step_biomarker(ctx, {"model": "rf", "n_features": 10, "cv_folds": 5,
                                         "n_trees": 60, "pool": "all"}, out("step-biomarker"))
    except Exception:  # noqa: BLE001
        check("纯噪声 step_biomarker 可跑", False, traceback.format_exc(limit=3))
        return
    metrics = payload.get("summary", {}).get("metrics") or {}
    print(f"    metrics keys = {sorted(metrics.keys())}")
    check("metrics 含 nested_cv_auc", "nested_cv_auc" in metrics, str(sorted(metrics.keys())))
    nested = float(metrics.get("nested_cv_auc") or float("nan"))
    check("纯噪声 nested_cv_auc ≤ 0.75", np.isfinite(nested) and nested <= 0.75, f"{nested:.4f}")
    check("metrics 含 apparent_auc", "apparent_auc" in metrics, str(sorted(metrics.keys())))
    apparent = float(metrics.get("apparent_auc") or float("nan"))
    check("表观 AUC 明显高于嵌套 CV（乐观偏差被暴露）",
          np.isfinite(apparent) and apparent - nested > 0.15, f"{apparent:.3f} vs {nested:.3f}")
    check("metrics 明确 external_auc 未提供",
          ("external_auc" not in metrics) or metrics.get("external_auc") in (None, ""),
          str(metrics.get("external_auc")))
    joined = " ".join(str(v) for v in payload.get("notes") or [])
    check("notes 说明没有外部验证集 / 表观不等于验证", "外部" in joined or "apparent" in joined.lower(),
          joined[:120])
    check("n_samples = 60（无 QC 可剔除）", int(metrics.get("n_samples") or 0) == 60,
          str(metrics.get("n_samples")))


# ---------------------------------------------------------------------------
# 4) analysis_model_validation：自助法按 subject 重抽 + 重做流程
# ---------------------------------------------------------------------------


def test_model_validation_bootstrap() -> None:
    print("\n[4] analysis_model_validation：自助法按 subject 重抽并重做流程")
    rng = np.random.default_rng(1)
    n_subjects, per_subject, p = 20, 3, 400
    subjects = [f"P{s + 1:02d}" for s in range(n_subjects) for _ in range(per_subject)]
    y_subject = np.asarray([0, 1] * (n_subjects // 2), dtype=int)
    y = np.repeat(y_subject, per_subject)
    X = rng.normal(size=(n_subjects * per_subject, p))
    X[y == 1, :5] += 1.2  # 注入可检测的真实信号
    ctx = make_biomarker_ctx(X, y, subjects=subjects)
    try:
        payload = B.analysis_model_validation(ctx, {"n_boot": 20, "folds": 3, "pool": "all"},
                                               out("model-validation"))
    except Exception:  # noqa: BLE001
        check("model_validation 可跑（含 subject 分组）", False, traceback.format_exc(limit=3))
        return
    metrics = payload.get("summary", {}).get("metrics") or {}
    for key in ("apparent_auc", "optimism", "corrected_auc", "cv_auc", "nested_cv_auc"):
        check(f"metrics 含 {key}", key in metrics, str(sorted(metrics.keys())))
    check("自助法记录按 subject 分组（grouped=True）",
          bool(metrics.get("bootstrap_grouped")), str(metrics.get("bootstrap_grouped")))
    check("记录自助重抽的个体数（n_subjects=20）",
          int(metrics.get("n_subjects") or 0) == n_subjects, str(metrics.get("n_subjects")))
    check("自助法有有效重抽样", int(metrics.get("n_boot_valid") or 0) >= 10,
          str(metrics.get("n_boot_valid")))
    check("nested_cv_auc ≤ apparent_auc",
          float(metrics.get("nested_cv_auc") or 0.0) <= float(metrics.get("apparent_auc") or 0.0) + 1e-9,
          f"{metrics.get('nested_cv_auc')} vs {metrics.get('apparent_auc')}")


# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("P0-2 回归：训练折内预处理/筛选 + 按受试者划分 + 指标拆分")
    print("=" * 78)
    test_nested_cv_noise()
    test_group_aware_split()
    test_step_biomarker_metrics()
    test_model_validation_bootstrap()
    print("\n" + "=" * 78)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
