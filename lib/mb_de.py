"""
dsh-metabolomics — 单变量差异 / 多组比较 / 趋势 / 剂量响应（mb-stat）

本模块提供：
  STEPS    : de（requires=["transform"]）
  ANALYSES : univariate_de / volcano_de / fc_bar / heatmap_top / multi_group_de /
             trend_course / dose_response
  PLOTS    : volcano_de / fc_bar / de_heatmap_top / trend_plot / dose_response

方法（全部 numpy/scipy 自实现；本机无 limma/MetaboAnalystR，不冒充官方包）：
  * 两组差异：Welch t（Welch 1947；不假设等方差，MetaboAnalyst 默认）/
    Mann-Whitney U（Wilcoxon 秩和；正态近似、并列值按 scipy 校正）/
    配对 t / 配对 Wilcoxon（符号秩）。
  * limma 风格经验贝叶斯 moderated t（Smyth 2004）：先按两组汇总方差 s²（自由度 d=N−2），
    再用 fitFDist 的矩估计近似反推先验自由度 d0 与先验方差 s0²：
      s²_post = (d0·s0² + d·s²)/(d0+d)，t = Δ/(√s²_post·√(1/n1+1/n2))，df_post = d+d0。
    这不是 limma 的完整基因级 EB 实现（本机无 R/limma 依赖），参数估计与文献一致但不保证逐位相同。
  * 多重检验：Benjamini-Hochberg FDR（`mb_common.bh_fdr`）。
  * 效应量：log2FC。若当前矩阵是 log 尺度（X_kind=log2/lognorm）取组均值差；
    否则取 log2(均值比)（含伪计数避免除零）；标度后的矩阵只能给"标度单位差"并在 warnings 说明。
  * 多组：单因素 ANOVA（Fisher）/ Welch ANOVA（Welch 1951）/ Kruskal-Wallis（Kruskal & Wallis 1952），
    事后 Tukey-Kramer（Tukey 1949，studentized range）/ Games-Howell（Games & Howell 1976）/
    Dunn（Dunn 1964，秩和 z 检验）。
  * 时间/剂量趋势：按时间分箱的均值 ± SE + 线性回归 + Spearman 秩相关（Fiehn 2008 代谢轨迹语义）。
  * 剂量响应：4 参数 logistic（4PL，R 的 drc 语义）：
      y = d + (a − d)/(1 + (x/c)^b)，curve_fit 在 log10(x) 上参数化；EC50=c，Hill 斜率=b；
      置信区间用协方差矩阵一阶传播（近似正态），曲线 CI 用参数后验抽样。
  * 热图：行 z-score（`mb_common.zscore_rows`）+ 层次聚类（scipy Ward/相关距离）。

约定（DESIGN.md §2.3/§2.5）：
  * 只读 ctx；结果写回走返回的 store（在 result() 之后挂上，DataFrame 必须留在内存里）。
  * 缺前置 → prerequisite（中文）；参数非法 → bad_request；不吞异常。
  * 图上文字一律英文；返回给模型/UI 的文字是中文；确定性（SEED=0）。

与 mb-base 的图归属：`heatmap_top` 这个 kind 归 mb-base 的 `mb_plots.py`（DESIGN §5 基础图归属），
本模块的差异热图以 `de_heatmap_top` 注册，避免同一 kind 双 owner；分析 id 仍叫 `heatmap_top`
（DESIGN §4 的分析清单不变）。
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy import special, stats

from mb_common import (
    PALETTE,
    EngineError,
    Timer,
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
    welch_de,
    zscore_rows,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SEED = 0
DEFAULT_DPI = 140
LOG2_FC_MIN = 1e-9

CONTROL_HINTS = ("control", "ctrl", "ctl", "con", "normal", "healthy", "nc", "wt",
                 "vehicle", "sham", "baseline", "untreated", "placebo", "mock", "blank", "0")

PAIR_COL_HINTS = ("subject", "patient", "donor", "pair", "individual", "subject_id",
                  "patient_id", "donor_id", "pair_id", "mouse", "animal", "id")

TIME_COL_HINTS = ("time", "hour", "h", "day", "week", "month", "minute", "tp", "timepoint",
                  "time_point", "duration", "dose", "concentration", "conc", "level")

DE_COLUMNS = [
    {"key": "feature", "label": "特征", "type": "string"},
    {"key": "name", "label": "名称", "type": "string"},
    {"key": "mz", "label": "m/z", "type": "number"},
    {"key": "rt", "label": "RT (min)", "type": "number"},
    {"key": "log2FC", "label": "log2FC", "type": "number"},
    {"key": "mean_case", "label": "均值(实验组)", "type": "number"},
    {"key": "mean_control", "label": "均值(对照组)", "type": "number"},
    {"key": "p", "label": "P 值", "type": "number"},
    {"key": "p_adj", "label": "FDR (BH)", "type": "number"},
    {"key": "regulation", "label": "方向", "type": "string"},
    {"key": "fc_hit", "label": "达 FC 阈值", "type": "bool"},
    {"key": "detected_frac", "label": "检出比例", "type": "number"},
]

METHOD_ALIASES = {
    "welch": "welch", "welch_t": "welch", "t": "welch", "ttest": "welch", "t_test": "welch",
    "welch's t": "welch", "welch-t": "welch",
    "wilcoxon": "wilcoxon", "mannwhitney": "wilcoxon", "mann-whitney": "wilcoxon",
    "mann_whitney": "wilcoxon", "mwu": "wilcoxon", "wilcox": "wilcoxon", "rank": "wilcoxon",
    "limma": "limma", "moderated": "limma", "ebayes": "limma",
}

MULTI_METHOD_ALIASES = {
    "anova": "anova", "oneway": "anova", "f": "anova", "fisher": "anova",
    "welch": "welch", "welch_anova": "welch", "welch-anova": "welch",
    "kruskal": "kruskal", "kruskalwallis": "kruskal", "kruskal-wallis": "kruskal",
    "kw": "kruskal",
}

POSTHOC_ALIASES = {
    "tukey": "tukey", "tukey-kramer": "tukey", "tukeyhsd": "tukey",
    "games_howell": "games_howell", "games-howell": "games_howell", "gh": "games_howell",
    "dunn": "dunn", "dunns": "dunn", "none": "none", "": "none",
}


# ---------------------------------------------------------------------------
# 参数 / ctx 小工具
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
    if not math.isfinite(value):
        raise bad_request(f"参数「{key}」必须是有限数字")
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _bool_param(params: dict, key: str, default: bool = False) -> bool:
    raw = params.get(key)
    if raw is None or raw == "":
        return bool(default)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "y", "on", "是", "对"):
        return True
    if text in ("0", "false", "no", "n", "off", "否", "不"):
        return False
    raise bad_request(f"参数「{key}」必须是布尔值，收到：{raw!r}")


def _short(text, limit: int = 26) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


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
            f"特征/样本名数量与矩阵形状不一致：{len(samples)}×{len(features)} vs "
            f"{arr.shape[0]}×{arr.shape[1]}",
        )
    return arr, features, samples


def _sample_info_frame(ctx, samples: list[str]) -> pd.DataFrame:
    info = ctx.get("sample_info")
    index = pd.Index([str(s) for s in samples], name="sample")
    if isinstance(info, pd.DataFrame):
        if list(map(str, info.index)) == [str(s) for s in samples]:
            return info
        try:
            return info.reindex(index)
        except Exception:  # noqa: BLE001
            return pd.DataFrame(index=index)
    return pd.DataFrame(index=index)


def _feature_meta_frame(ctx, features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.Index([str(f) for f in features], name="feature"))
    info = ctx.get("feature_info")
    for col in ("name", "mz", "rt"):
        if isinstance(info, pd.DataFrame) and col in info.columns:
            try:
                series = info[col].reindex(frame.index)
            except Exception:  # noqa: BLE001
                series = pd.Series(np.nan, index=frame.index)
            if col in ("mz", "rt"):
                frame[col] = pd.to_numeric(series, errors="coerce")
            else:
                frame[col] = series.where(series.notna(), "").astype(object)
        else:
            frame[col] = np.nan if col in ("mz", "rt") else ""
    names = frame["name"].astype(str)
    frame["label"] = np.where(names.str.len().to_numpy() > 0, names.to_numpy(), frame.index.astype(str))
    return frame


def _guess_column(frame: pd.DataFrame, hints) -> str | None:
    """按关键词猜一列（先精确匹配，再包含匹配）。"""
    if not isinstance(frame, pd.DataFrame):
        return None
    columns = [str(c) for c in frame.columns]
    lowered = {c.lower(): c for c in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for hint in hints:
        for col in columns:
            if hint in col.lower():
                return col
    return None


def _group_column(ctx, frame: pd.DataFrame, params: dict) -> str | None:
    key = str(params.get("group_col") or ctx.get("group_key") or "").strip()
    if key:
        return key
    hints = ("group", "groups", "condition", "treatment", "class", "type", "phenotype",
             "disease", "status", "arm", "cohort", "label", "category")
    return _guess_column(frame, hints)


def _value_labels(frame: pd.DataFrame, key: str, samples: list[str]) -> np.ndarray:
    values = frame[key].tolist() if key in frame.columns else [None] * len(samples)
    out = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out.append("NA")
        else:
            out.append(str(value))
    return np.array(out, dtype=object)


# ---------------------------------------------------------------------------
# 比较组解析（case / control / 多组）
# ---------------------------------------------------------------------------


QC_VALUE_HINTS = ("qc", "pool", "pooled", "quality control", "qc sample", "qcsample",
                  "quality-control", "blank", "pool_qc")


def _qc_mask(info: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """识别 QC 样本：优先看 qc/is_qc/qc_flag/sample_type 列，再退回分组值关键词（QC/pool）。"""
    n = len(labels)
    mask = np.zeros(n, dtype=bool)
    if isinstance(info, pd.DataFrame):
        for col in ("qc", "is_qc", "qc_flag", "isqc", "sample_type", "sampletype", "type_qc"):
            if col not in info.columns:
                continue
            try:
                values = info[col].astype(str).str.strip().str.lower().tolist()
            except Exception:  # noqa: BLE001
                continue
            if col in ("sample_type", "sampletype", "type_qc"):
                # 这类列只看 QC 关键词（"1"/"true" 不代表 QC）
                hit = np.array([str(v) in QC_VALUE_HINTS for v in values], dtype=bool)
            else:
                hit = np.array([str(v) in ("1", "1.0", "true", "yes", "y", "qc", "pool",
                                           "pooled", "quality control", "qc sample")
                                for v in values], dtype=bool)
            mask |= hit
    if not mask.any():
        lowered = np.array([str(v).strip().lower() for v in labels], dtype=object)
        mask = np.array([value in QC_VALUE_HINTS for value in lowered], dtype=bool)
    return mask


def _resolve_comparison(ctx, params: dict, require_pair: bool = False) -> dict:
    """解析实验组/对照组。

    两组时可自动推断（对照组关键词优先、否则按字母序）；未显式指定 case/control 时自动排除
    QC 样本（qc=1 / group=QC / pool 等），只用剩下两组比较，因此 demo（case/control/QC）可直接跑。
    """
    params = _pdict(params)
    X, features, samples = _matrix(ctx)
    info = _sample_info_frame(ctx, samples)
    key = _group_column(ctx, info, params)
    if not key:
        raise prerequisite("还没有分组列：先在「样本」里设置分组列（group_col），差异分析需要两组标签")
    if key not in info.columns:
        raise prerequisite(f"样本信息里没有分组列「{key}」；可选列：{'、'.join(map(str, info.columns)) or '无'}")
    labels = _value_labels(info, key, samples)
    warnings: list[str] = []
    valid = labels != "NA"
    if not valid.all():
        warnings.append(f"有 {int((~valid).sum())} 个样本的分组为空（NA），已从差异分析里剔除")
    case = params.get("case")
    control = params.get("control")
    case = str(case).strip() if case is not None else None
    control = str(control).strip() if control is not None else None
    if case == "":
        case = None
    if control == "":
        control = None
    explicit = bool(case or control)

    # 未显式指定比较组时自动剔除 QC 样本（QC 与生物样本的差异没有生物学意义）
    usable = valid.copy()
    qc_excluded = 0
    if not explicit:
        qc = _qc_mask(info, labels) & valid
        if qc.any() and int((valid & ~qc).sum()) >= 4:
            usable = valid & ~qc
            qc_excluded = int(qc.sum())
    groups: list[str] = []
    for value in labels[usable]:
        if value not in groups:
            groups.append(value)
    if len(groups) < 2:
        extra = f"（已自动排除 {qc_excluded} 个 QC 样本）" if qc_excluded else ""
        raise prerequisite(f"分组列「{key}」只有 {len(groups)} 个有效分组（{'、'.join(groups) or '无'}）{extra}，"
                           f"差异分析至少需要两组")
    if case and case not in groups:
        raise bad_request(f"实验组「{case}」不在分组「{key}」里；可选：{'、'.join(groups)}")
    if control and control not in groups:
        raise bad_request(f"对照组「{control}」不在分组「{key}」里；可选：{'、'.join(groups)}")
    if case and control:
        if case == control:
            raise bad_request(f"实验组与对照组不能相同（都是「{case}」）")
    elif case:
        candidates = [g for g in groups if g != case]
        if len(candidates) > 1:
            hinted = [g for g in candidates if any(h in g.lower() for h in CONTROL_HINTS)]
            if len(hinted) != 1:
                raise prerequisite(f"分组有 {len(groups)} 组（{'、'.join(groups)}），只指定了 case="
                                   f"{case}，请同时指定 control")
            control = hinted[0]
        else:
            control = candidates[0]
        warnings.append(f"未指定 contrast 参数中参与比较的对照，已按关键词推断 control={control}")
    elif control:
        candidates = [g for g in groups if g != control]
        if len(candidates) > 1:
            raise prerequisite(f"分组有 {len(groups)} 组（{'、'.join(groups)}），只指定了 control="
                               f"{control}，请同时指定 case")
        case = candidates[0]
    else:
        if len(groups) > 2:
            extra = (f"（已自动排除 {qc_excluded} 个 QC 样本）" if qc_excluded else "")
            raise prerequisite(
                f"分组有 {len(groups)} 组（{'、'.join(groups)}）{extra}，单变量差异只做两组比较："
                f"请指定 case 与 control，或用「多组 ANOVA」分析"
            )
        hinted = [g for g in groups if any(h in g.lower() for h in CONTROL_HINTS)]
        if len(hinted) == 1:
            control = hinted[0]
            case = [g for g in groups if g != control][0]
        else:
            ordered = sorted(groups)
            control, case = ordered[0], ordered[1]
            warnings.append(f"未指定 case/control：按字母序把「{control}」当对照组、「{case}」当实验组"
                            f"（可用 case/control 参数指定，方向影响 log2FC 正负）")
    if qc_excluded:
        warnings.append(
            f"未指定 case/control：已自动排除 {qc_excluded} 个 QC 样本（group=QC 或 qc=1），"
            f"只比较「{case}」与「{control}」；如需把 QC 当一组参与比较，请显式传 case/control"
        )
    mask_case = labels == case
    mask_control = labels == control
    n_case = int(mask_case.sum())
    n_control = int(mask_control.sum())
    if n_case < 2 or n_control < 2:
        raise prerequisite(f"分组「{case}」({n_case}) 或「{control}」({n_control}) 样本不足 2 个，"
                           f"无法做差异检验")
    return {
        "group_key": key,
        "labels": labels,
        "groups": groups,
        "case": case,
        "control": control,
        "mask_case": mask_case,
        "mask_control": mask_control,
        "n_case": n_case,
        "n_control": n_control,
        "samples": samples,
        "features": features,
        "X": X,
        "sample_info": info,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# log2FC 效应量
# ---------------------------------------------------------------------------


def _fc_basis(ctx, warnings: list[str]) -> str:
    """判断当前矩阵的值尺度，决定 log2FC 的算法。"""
    kind = str(ctx.get("X_kind") or "").lower()
    if kind in ("log2", "lognorm", "log", "log1p"):
        return "log"
    if kind in ("scaled", "zscore", "uv", "pareto", "standard"):
        warnings.append(
            "当前矩阵已标度（X_kind=scaled）：log2FC 是「标度单位均值差」，不等于真实倍数变化；"
            "如需真实 FC，请用未标度的归一化矩阵重跑差异分析"
        )
        return "scaled"
    return "ratio"


_LOG_METHODS = ("log2", "log10", "ln")


def _layer_arrays(ctx) -> dict | None:
    """P0-1：从 ``ctx["results"]["transform"]["layers"]`` 取三层并按当前 features/samples 对齐。

    返回 ``{"linear","log","model","transform","scale","features","samples"}`` 或 None（没有
    可用的层快照 / 与当前矩阵对不上）。对齐规则：优先按快照的 features/samples 顺序重排；
    特征或样本集合不一致（例如 transform 之后又做了特征过滤）时返回 None，由调用方回退并
    在 warnings 里说明 —— 绝不把错位的层当未标度层用。
    """
    results = ctx.get("results") if hasattr(ctx, "get") else None
    if not isinstance(results, dict):
        return None
    store = results.get("transform")
    if not isinstance(store, dict):
        return None
    layers = store.get("layers")
    if not isinstance(layers, dict):
        return None
    values = layers.get("values")
    if not isinstance(values, dict):
        return None
    features = [str(f) for f in (ctx.get("features") or [])]
    samples = [str(s) for s in (ctx.get("samples") or [])]
    layer_features = [str(f) for f in (layers.get("features") or [])]
    layer_samples = [str(s) for s in (layers.get("samples") or [])]
    if not features or not samples or not layer_features or not layer_samples:
        return None

    def _positions(current: list[str], snapshot: list[str]) -> list[int] | None:
        if current == snapshot:
            return list(range(len(current)))
        if len(current) != len(snapshot) or len(set(snapshot)) != len(snapshot):
            return None
        position = {key: i for i, key in enumerate(snapshot)}
        if any(key not in position for key in current):
            return None
        return [position[key] for key in current]

    row_pos = _positions(samples, layer_samples)
    col_pos = _positions(features, layer_features)
    if row_pos is None or col_pos is None:
        return None
    n, p = len(samples), len(features)
    aligned: dict[str, np.ndarray | None] = {}
    for key in ("linear", "log", "model"):
        arr = values.get(key)
        if arr is None:
            aligned[key] = None
            continue
        try:
            arr = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if arr.ndim != 2:
            return None
        if arr.shape == (len(layer_samples), len(layer_features)):
            pass
        elif arr.shape == (len(layer_features), len(layer_samples)):
            arr = arr.T
        else:
            return None
        if row_pos != list(range(len(samples))) or col_pos != list(range(len(features))):
            arr = arr[np.ix_(row_pos, col_pos)]
        if arr.shape != (n, p):
            return None
        aligned[key] = arr
    if aligned.get("log") is None and aligned.get("model") is None:
        return None
    center = layers.get("center")
    denom = layers.get("denom")
    try:
        center = None if center is None else np.asarray(center, dtype=np.float64).ravel()
        denom = None if denom is None else np.asarray(denom, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        center = denom = None
    if (center is not None and center.size != p) or (denom is not None and denom.size != p):
        center = denom = None
    return {
        "linear": aligned.get("linear"),
        "log": aligned.get("log"),
        "model": aligned.get("model"),
        "center": center,
        "denom": denom,
        "transform": dict(layers.get("transform") or {}),
        "scale": dict(layers.get("scale") or {}),
        "features": features,
        "samples": samples,
    }


def _layer_value(layers: dict, key: str) -> np.ndarray | None:
    """取某一层；预算截断导致该层没存时，从 model 层反演（model → log → linear）。

    model = (transform(x) − center) / denom，因此 log（未标度变换层）= model*denom + center，
    linear 再按 transform.method 逆变换。反演失败返回 None（调用方回退并在 warnings 说明）。
    """
    if not isinstance(layers, dict):
        return None
    arr = layers.get(key)
    if arr is not None:
        return np.asarray(arr, dtype=np.float64)
    model = layers.get("model")
    center = layers.get("center")
    denom = layers.get("denom")
    if model is None or center is None or denom is None:
        return None
    model = np.asarray(model, dtype=np.float64)
    unscaled = model * np.asarray(denom, dtype=np.float64) + np.asarray(center, dtype=np.float64)
    if key == "log":
        return unscaled
    method = str((layers.get("transform") or {}).get("method") or "none").strip().lower()
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "log2":
            return np.power(2.0, unscaled)
        if method == "log10":
            return np.power(10.0, unscaled)
        if method == "ln":
            return np.exp(unscaled)
        if method == "cuberoot":
            return np.power(unscaled, 3.0)
    return unscaled  # method=none：log 层就是 linear 层


def _log2fc(mean_case: np.ndarray, mean_control: np.ndarray, basis: str) -> np.ndarray:
    if basis == "log":
        return mean_case - mean_control
    if basis == "scaled":
        return mean_case - mean_control
    # ratio：先平移成正数再加伪计数，避免除零/负值
    shift = 0.0
    stack = np.concatenate([mean_case.ravel(), mean_control.ravel()])
    finite = stack[np.isfinite(stack)]
    if finite.size and float(finite.min()) <= 0:
        shift = -float(finite.min()) + abs(float(finite.min())) * 0.01 + 1e-9
    a = np.clip(mean_case + shift, LOG2_FC_MIN, None)
    b = np.clip(mean_control + shift, LOG2_FC_MIN, None)
    return np.log2(a / b)


# ---------------------------------------------------------------------------
# 检验实现
# ---------------------------------------------------------------------------


def _nan_stats(mat: np.ndarray):
    """按列（特征）算 nan-aware 的 n / mean / var（ddof=1）。"""
    arr = np.asarray(mat, dtype=float)
    finite = np.isfinite(arr)
    n = finite.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(arr, axis=0) / np.maximum(n, 1.0), np.nan)
        dev = arr - mean
        var = np.nansum(dev * dev, axis=0) / np.maximum(n - 1, 1.0)
    var = np.where(n >= 2, var, np.nan)
    return n, mean, var


def _t_p(t: np.ndarray, df) -> np.ndarray:
    """双侧 t 检验 p 值（向量化；df 可以是标量或数组）。"""
    t = np.asarray(t, dtype=float)
    df = np.asarray(df, dtype=float)
    with np.errstate(invalid="ignore"):
        p = 2.0 * stats.t.sf(np.abs(t), df)
    return np.where(np.isfinite(p), p, np.nan)


def _welch_core(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """nan-aware Welch t 检验（Welch 1947）：返回 (t, p, 平均自由度)。"""
    n_a, mean_a, var_a = _nan_stats(A)
    n_b, mean_b, var_b = _nan_stats(B)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(var_a / np.maximum(n_a, 1.0) + var_b / np.maximum(n_b, 1.0))
        delta = mean_a - mean_b
        t = np.where(se > 1e-15, delta / np.where(se > 1e-15, se, 1.0), 0.0)
        va = var_a / np.maximum(n_a, 1.0)
        vb = var_b / np.maximum(n_b, 1.0)
        df = (va + vb) ** 2 / (
            np.where(n_a > 1, va * va / np.maximum(n_a - 1, 1.0), 0.0)
            + np.where(n_b > 1, vb * vb / np.maximum(n_b - 1, 1.0), 0.0)
        )
    valid = (n_a >= 2) & (n_b >= 2)
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    p = _t_p(t, df)
    p = np.where(valid, p, np.nan)
    t = np.where(valid, t, np.nan)
    mean_df = float(np.nanmean(df[valid])) if valid.any() else 1.0
    return t, p, mean_df


def _welch_p(A: np.ndarray, B: np.ndarray, case: str, control: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Welch t：无缺失时用共享实现 `mb_common.welch_de`，含缺失时用本模块 nan-aware 回退。"""
    has_nan = bool(np.isnan(A).any() or np.isnan(B).any())
    if not has_nan:
        try:
            labels = [case] * A.shape[0] + [control] * B.shape[0]
            genes = [f"F{i}" for i in range(A.shape[1])]
            frame = welch_de(np.vstack([A, B]), labels, case, control, genes=genes,
                             lfc=0.0, padj_cut=0.05)
            frame = frame.set_index("gene")
            order = [f"F{i}" for i in range(A.shape[1])]
            t = pd.to_numeric(frame["t"], errors="coerce").reindex(order).to_numpy(dtype=float)
            p = pd.to_numeric(frame["P.Value"], errors="coerce").reindex(order).to_numpy(dtype=float)
            return t, p, "welch (mb_common.welch_de)"
        except EngineError:
            raise
        except Exception:  # noqa: BLE001 - 共享实现异常时回退到 nan-aware 版本
            pass
    t, p, _df = _welch_core(A, B)
    return t, p, "welch (nan-aware)"


def _mannwhitney(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney U（Wilcoxon 秩和）：返回有符号 z 与双侧 p（逐特征，nan-aware）。"""
    p_count = A.shape[1]
    z = np.full(p_count, np.nan)
    p = np.full(p_count, np.nan)
    for j in range(p_count):
        a = A[:, j][np.isfinite(A[:, j])]
        b = B[:, j][np.isfinite(B[:, j])]
        if a.size < 2 or b.size < 2:
            continue
        if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
            z[j] = 0.0
            p[j] = 1.0
            continue
        try:
            res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        except ValueError:
            continue
        u = float(res.statistic)
        n = a.size + b.size
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = a.size * b.size / 2.0
            sigma = math.sqrt(a.size * b.size * (n + 1) / 12.0)
            zj = (u - mu) / sigma if sigma > 0 else 0.0
        p[j] = float(res.pvalue)
        z[j] = zj
    return z, p


def _paired_core(A: np.ndarray, B: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """配对检验：逐特征算差值 d = A − B，再做配对 t / 配对 Wilcoxon。"""
    d = A - B
    p_count = d.shape[1]
    stat = np.full(p_count, np.nan)
    p = np.full(p_count, np.nan)
    for j in range(p_count):
        dj = d[:, j][np.isfinite(d[:, j])]
        n = dj.size
        if n < 2:
            continue
        if method == "wilcoxon":
            if np.all(dj == 0):
                stat[j] = 0.0
                p[j] = 1.0
                continue
            try:
                res = stats.wilcoxon(dj, alternative="two-sided")
            except ValueError:
                continue
            stat[j] = float(res.statistic)
            p[j] = float(res.pvalue)
        else:
            mean = float(dj.mean())
            sd = float(dj.std(ddof=1))
            if sd <= 0:
                stat[j] = 0.0
                p[j] = 1.0
                continue
            t = mean / (sd / math.sqrt(n))
            stat[j] = t
            p[j] = float(2.0 * stats.t.sf(abs(t), n - 1))
    return stat, p


def _trigamma_inverse(y: float) -> float:
    """解 trigamma(x)=y（x>0，trigamma 单调递减），用于 fitFDist 的矩估计。"""
    try:
        from scipy.optimize import brentq

        lo, hi = 1e-8, 1e8
        if y >= float(special.polygamma(1, lo)):
            return lo
        if y <= float(special.polygamma(1, hi)):
            return hi
        return float(brentq(lambda x: float(special.polygamma(1, x)) - y, lo, hi, maxiter=200))
    except Exception:  # noqa: BLE001
        return 1.0


def _fit_fdist(s2: np.ndarray, d: float) -> tuple[float, float]:
    """fitFDist 的矩估计近似（Smyth 2004 §3）：返回 (d0, s0²)。"""
    s2 = np.asarray(s2, dtype=float)
    s2 = s2[np.isfinite(s2) & (s2 > 0)]
    if s2.size < 3 or d <= 0:
        return 0.0, 1.0
    z = np.log(s2)
    e = z - float(special.digamma(d / 2.0)) + math.log(d / 2.0)
    var_e = float(np.var(e, ddof=1))
    target = var_e - float(special.polygamma(1, d / 2.0))
    if not math.isfinite(target) or target <= 1e-10:
        d0 = 1e6
    else:
        d0 = 2.0 * _trigamma_inverse(target)
    d0 = float(min(max(d0, 1e-4), 1e6))
    s0sq = math.exp(float(np.mean(e)) + float(special.digamma(d0 / 2.0)) - math.log(d0 / 2.0))
    if not math.isfinite(s0sq) or s0sq <= 0:
        s0sq = float(np.median(s2))
    return d0, s0sq


def _limma(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """limma 风格经验贝叶斯 moderated t（Smyth 2004；内置矩估计近似）。"""
    n_a, mean_a, var_a = _nan_stats(A)
    n_b, mean_b, var_b = _nan_stats(B)
    n_a0 = float(n_a.max()) if n_a.size else 0.0
    n_b0 = float(n_b.max()) if n_b.size else 0.0
    d = n_a0 + n_b0 - 2.0
    if d < 1:
        raise prerequisite(f"样本太少（每组 {n_a0:g} / {n_b0:g}），经验贝叶斯 moderated t 需要每组 ≥2")
    with np.errstate(invalid="ignore", divide="ignore"):
        ss = (np.maximum(n_a - 1, 0) * np.where(np.isfinite(var_a), var_a, np.nan)
              + np.maximum(n_b - 1, 0) * np.where(np.isfinite(var_b), var_b, np.nan))
        s2 = ss / d
    ok = np.isfinite(s2) & (s2 > 0)
    d0, s0sq = _fit_fdist(s2[ok], d)
    floor = float(np.nanmin(s2[ok])) * 1e-3 if ok.any() else 1e-12
    floor = max(floor, 1e-12)
    s2_use = np.where(np.isfinite(s2) & (s2 > 0), s2, floor)
    s2_post = (d0 * s0sq + d * s2_use) / (d0 + d)
    se = np.sqrt(s2_post * (1.0 / max(n_a0, 1.0) + 1.0 / max(n_b0, 1.0)))
    delta = mean_a - mean_b
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(se > 0, delta / np.where(se > 0, se, 1.0), 0.0)
    df_post = d + d0
    p = _t_p(t, df_post)
    return t, p, {"d0": float(d0), "s0_sq": float(s0sq), "df_residual": float(d), "df_post": float(df_post),
                  "n_case": n_a0, "n_control": n_b0}


# ---------------------------------------------------------------------------
# 配对设计
# ---------------------------------------------------------------------------


def _pair_columns(info: pd.DataFrame, params: dict) -> tuple[list[str], str | None]:
    """解析配对列：``pair_col`` 支持单列名、``["subject","visit"]``、``"subject+visit"`` 复合键。

    返回 (列名列表, 猜出来的列名或 None)。没给 pair_col 时按 PAIR_COL_HINTS 猜一列。
    """
    raw = params.get("pair_col")
    columns: list[str] = []
    if isinstance(raw, (list, tuple)):
        columns = [str(v).strip() for v in raw if str(v).strip()]
    elif raw is not None and str(raw).strip():
        text = str(raw).strip()
        for sep in ("+", "&", "|", ","):
            if sep in text:
                columns = [part.strip() for part in text.split(sep) if part.strip()]
                break
        if not columns:
            columns = [text]
    if columns:
        missing = [col for col in columns if col not in info.columns]
        if missing:
            raise bad_request(
                f"配对列「{'、'.join(missing)}」不在样本信息里；可选列：{'、'.join(map(str, info.columns))}"
            )
        return columns, "+".join(columns)
    guessed = _guess_column(info, PAIR_COL_HINTS)
    return ([guessed] if guessed else []), guessed


def _pair_keys(info: pd.DataFrame, columns: list[str], indices: np.ndarray, group_label: str) -> list:
    """按 pair_col（可复合）生成每行的配对键，并校验键非空、组内唯一。

    重复键会让"配对"退化成"取第一条"（旧 bug），因此直接 bad_request 而不是猜。
    """
    keys: list = []
    for index in np.asarray(indices, dtype=int):
        parts = []
        for column in columns:
            try:
                value = info.iloc[int(index)][column]
            except (KeyError, IndexError) as error:  # pragma: no cover - 列已校验
                raise bad_request(f"读取配对列「{column}」失败：{error}") from error
            text = "" if value is None else str(value).strip()
            if text == "" or text.lower() in ("nan", "none", "na", "null"):
                raise prerequisite(
                    f"配对列「{column}」在第 {int(index) + 1} 个样本（{info.index[int(index)]}）为空；"
                    "配对需要显式的唯一 ID"
                )
            parts.append(text)
        keys.append(tuple(parts) if len(parts) > 1 else parts[0])
    seen: dict = {}
    duplicates: list[str] = []
    for key, index in zip(keys, np.asarray(indices, dtype=int)):
        if key in seen:
            label = " + ".join(key) if isinstance(key, tuple) else str(key)
            duplicates.append(f"{label}（样本 {seen[key]} 与 {info.index[int(index)]}）")
        else:
            seen[key] = str(info.index[int(index)])
    if duplicates:
        raise bad_request(
            f"配对列「{' + '.join(columns)}」在{group_label}内有重复键：{'；'.join(duplicates[:5])}"
            f"{'…' if len(duplicates) > 5 else ''}。配对键必须唯一（重复测量请用复合键，"
            f"如 pair_col=[\"subject\", \"visit\"] 或 \"subject+visit\"）"
        )
    return keys


def _paired_indices(comp: dict, params: dict, warnings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """构造配对下标：按显式唯一 ID（pair_col，可复合）一一配对（BIO_P0_BACKLOG §4 P0-3）。

    设计原则：
      * 没有配对列 → prerequisite，**不再**按行序悄悄配对（行序不是设计信息）；
      * 配对列在任一组内重复 → bad_request（重复键会让配对退化成"取第一条"）；
      * 两组各有一个同名键 → 配成一对；未配对的样本不参与配对检验并给出 warning。
    """
    info = comp["sample_info"]
    case_idx = np.flatnonzero(comp["mask_case"])
    control_idx = np.flatnonzero(comp["mask_control"])
    columns, hint = _pair_columns(info, params)
    if not columns:
        raise prerequisite(
            "配对分析需要显式的配对 ID 列（如 subject/patient/pair）：请在样本表里提供唯一配对列，"
            "或设置参数 pair_col。没有配对信息时按行序配对会把不相关的样本当成一对"
        )
    keys_case = _pair_keys(info, columns, case_idx, "实验组")
    keys_control = _pair_keys(info, columns, control_idx, "对照组")
    control_lookup: dict = {}
    for key, index in zip(keys_control, control_idx):
        control_lookup.setdefault(key, []).append(int(index))
    pairs_case: list[int] = []
    pairs_control: list[int] = []
    unmatched_case: list[str] = []
    for key, index in zip(keys_case, case_idx):
        targets = control_lookup.get(key) or []
        if not targets:
            unmatched_case.append(" + ".join(key) if isinstance(key, tuple) else str(key))
            continue
        pairs_case.append(int(index))
        pairs_control.append(int(targets.pop(0)))
    matched_controls = set(pairs_control)
    unmatched_control = [
        " + ".join(key) if isinstance(key, tuple) else str(key)
        for key, index in zip(keys_control, control_idx) if int(index) not in matched_controls
    ]
    if len(pairs_case) < 2:
        raise prerequisite(
            f"配对列「{' + '.join(columns)}」只匹配到 {len(pairs_case)} 对（至少 2 对）："
            f"实验组有 {len(case_idx)} 个、对照组有 {len(control_idx)} 个样本"
        )
    if unmatched_case or unmatched_control:
        detail = []
        if unmatched_case:
            detail.append(f"实验组未配对 {len(unmatched_case)} 个（如 {unmatched_case[0]}）")
        if unmatched_control:
            detail.append(f"对照组未配对 {len(unmatched_control)} 个（如 {unmatched_control[0]}）")
        warnings.append(
            f"按配对列「{' + '.join(columns)}」匹配到 {len(pairs_case)} 对（实验组 "
            f"{len(case_idx)} / 对照组 {len(control_idx)} 个样本）；{'；'.join(detail)}，"
            f"这些样本未参与配对检验"
        )
    return np.array(pairs_case, dtype=int), np.array(pairs_control, dtype=int)


# ---------------------------------------------------------------------------
# 差异计算核心
# ---------------------------------------------------------------------------


def _normalize_method(text, aliases: dict, label: str) -> str:
    raw = str(text or "").strip().lower()
    key = raw.replace(" ", "").replace("_", "_")
    if key in aliases:
        return aliases[key]
    if raw in aliases:
        return aliases[raw]
    raise bad_request(f"未知的{label}「{text}」（可选：{', '.join(sorted(set(aliases.values())))}）")


def _de_compute(ctx, params: dict) -> dict:
    """单变量差异核心：解析分组 → 收集矩阵 → 检验 → BH → 表。"""
    params = _pdict(params)
    comp = _resolve_comparison(ctx, params)
    warnings = list(comp["warnings"])
    method = _normalize_method(params.get("method") or "welch", METHOD_ALIASES, "差异方法")
    paired = _bool_param(params, "paired", False)
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    fc = _float_param(params, "fc", 1.5, 1e-6, None)
    X = comp["X"]
    features = comp["features"]
    samples = [str(s) for s in (comp.get("samples") or ctx.get("samples") or [])]
    if paired:
        case_idx, control_idx = _paired_indices(comp, params, warnings)
    else:
        case_idx = np.flatnonzero(comp["mask_case"])
        control_idx = np.flatnonzero(comp["mask_control"])
    A = X[case_idx]
    B = X[control_idx]

    missing = int(np.isnan(A).sum() + np.isnan(B).sum())
    if missing:
        warnings.append(
            f"比较矩阵含 {missing} 个缺失值：检验使用成对完整观测（每组至少 2 个有效值），"
            f"建议先跑「缺失值填补」"
        )

    n_a, mean_a, var_a = _nan_stats(A)
    n_b, mean_b, var_b = _nan_stats(B)
    label = "welch"
    if paired:
        stat, p = _paired_core(A, B, "wilcoxon" if method == "wilcoxon" else "t")
        label = f"paired {('wilcoxon' if method == 'wilcoxon' else 't')}"
        if method == "limma":
            warnings.append("paired=true 时 limma 不支持配对设计，已改为配对 t 检验")
    elif method == "welch":
        stat, p, label = _welch_p(A, B, comp["case"], comp["control"])
    elif method == "wilcoxon":
        stat, p = _mannwhitney(A, B)
        label = "mann-whitney"
    else:
        stat, p, info = _limma(A, B)
        label = "limma-moderated-t"
        warnings.append(
            f"limma 经验贝叶斯为内置矩估计近似（d0={info['d0']:.3g}，s0²={info['s0_sq']:.4g}，"
            f"df_post={info['df_post']:.3g}）；本机未安装 limma，结果不保证与 R limma 逐位相同"
        )

    # P0-1：效应量（log2FC / 均值）来自「未标度层」，统计量仍在当前矩阵上算。
    #   有 log 变换 → 未标度 log 层均值差（2 倍信号严格 = 1）；
    #   只有 cuberoot / 无变换 → 未标度线性层均值比；
    #   没有层快照（旧数据/跨会话丢失）→ 回退旧口径并在 warnings 说明。
    layers = _layer_arrays(ctx)
    if layers is not None:
        layer_method = str((layers.get("transform") or {}).get("method") or "none").strip().lower()
        layer_log = _layer_value(layers, "log")
        layer_linear = _layer_value(layers, "linear") if layer_method not in _LOG_METHODS else None
        if layer_method in _LOG_METHODS and layer_log is not None:
            mean_a = np.nanmean(layer_log[case_idx], axis=0)
            mean_b = np.nanmean(layer_log[control_idx], axis=0)
            if layer_method == "log10":
                mean_a = mean_a * math.log2(10.0)
                mean_b = mean_b * math.log2(10.0)
            elif layer_method == "ln":
                mean_a = mean_a / math.log(2.0)
                mean_b = mean_b / math.log(2.0)
            log2fc = mean_a - mean_b
            basis = "log"
            log2fc_basis = "log2-mean-difference (未标度 log 层)"
        elif layer_linear is not None:
            linear_a = np.nanmean(layer_linear[case_idx], axis=0)
            linear_b = np.nanmean(layer_linear[control_idx], axis=0)
            log2fc = _log2fc(linear_a, linear_b, "ratio")
            mean_a, mean_b, basis = linear_a, linear_b, "ratio"
            log2fc_basis = "log2-ratio-of-means (未标度线性层)"
        else:
            layers = None
    if layers is None:
        basis = _fc_basis(ctx, warnings)
        log2fc = _log2fc(mean_a, mean_b, basis)
        log2fc_basis = {"log": "log2-mean-difference", "ratio": "log2-ratio-of-means",
                        "scaled": "scaled-difference"}[basis]

    # 检出比例：用 ctx["raw"]（原始未填补强度）的缺失掩码，按本次比较的样本索引统计；
    # raw 不存在/形状不符时回退当前矩阵（并保持旧的 1.0 语义，不伪造检出率）。
    detected_frac = None
    raw = ctx.get("raw") if hasattr(ctx, "get") else None
    if raw is not None:
        try:
            raw_arr = np.asarray(raw, dtype=np.float64)
            if raw_arr.ndim == 2 and raw_arr.shape == (len(samples), len(features)):
                compare_idx = np.unique(np.concatenate([case_idx, control_idx]))
                detected = np.isfinite(raw_arr[compare_idx, :]).sum(axis=0)
                detected_frac = detected.astype(float) / float(len(compare_idx))
            else:
                warnings.append(
                    f"ctx['raw'] 形状 {getattr(raw_arr, 'shape', None)} 与当前矩阵 "
                    f"({len(samples)}, {len(features)}) 不一致：detected_frac 回退当前矩阵（可能高估检出率）"
                )
        except (TypeError, ValueError):
            detected_frac = None
    if detected_frac is None:
        detected = np.isfinite(A).sum(axis=0) + np.isfinite(B).sum(axis=0)
        detected_frac = detected.astype(float) / float(A.shape[0] + B.shape[0])
    finite_p = np.isfinite(p)
    p_adj = bh_fdr(np.where(finite_p, p, np.nan))

    log2_fc_cut = math.log2(max(fc, 1e-6))
    with np.errstate(invalid="ignore"):
        fc_hit = np.abs(log2fc) >= log2_fc_cut
    regulation = np.full(len(features), "ns", dtype=object)
    ok_up = (p_adj <= fdr) & fc_hit & (log2fc > 0)
    ok_down = (p_adj <= fdr) & fc_hit & (log2fc < 0)
    regulation[ok_up] = "up"
    regulation[ok_down] = "down"

    meta = _feature_meta_frame(ctx, features)
    frame = pd.DataFrame({
        "feature": [str(f) for f in features],
        "name": meta["name"].to_numpy(),
        "mz": meta["mz"].to_numpy(),
        "rt": meta["rt"].to_numpy(),
        "log2FC": log2fc,
        "mean_case": mean_a,
        "mean_control": mean_b,
        "p": p,
        "p_adj": p_adj,
        "regulation": regulation,
        "fc_hit": fc_hit,
        "detected_frac": detected_frac,
        "t": stat,
    })
    frame = frame.sort_values(["p_adj", "p"], ascending=True, na_position="last").reset_index(drop=True)

    n_up = int((frame["regulation"] == "up").sum())
    n_down = int((frame["regulation"] == "down").sum())
    n_tested = int(np.isfinite(frame["p"]).sum())
    up = frame.loc[frame["regulation"] == "up", "feature"].astype(str).tolist()
    down = frame.loc[frame["regulation"] == "down", "feature"].astype(str).tolist()
    return {
        "frame": frame,
        "comp": comp,
        "method": label,
        "method_requested": method,
        "paired": paired,
        "contrast": {"case": comp["case"], "control": comp["control"]},
        "fdr": fdr,
        "fc": fc,
        "log2fc_basis": str(log2fc_basis),
        "n_up": n_up,
        "n_down": n_down,
        "n_tested": n_tested,
        "up": up,
        "down": down,
        "case_idx": case_idx.tolist(),
        "control_idx": control_idx.tolist(),
        "warnings": warnings,
        "group_key": comp["group_key"],
    }


def _de_store(computed: dict, params: dict) -> dict:
    frame = computed["frame"]
    return {
        "table": frame,
        "de_table": frame,
        "method": computed["method"],
        "method_requested": computed["method_requested"],
        "paired": bool(computed["paired"]),
        "group_key": computed["group_key"],
        "case": computed["contrast"]["case"],
        "control": computed["contrast"]["control"],
        "comparison": dict(computed["contrast"]),
        "fdr": computed["fdr"],
        "fc": computed["fc"],
        "log2fc_basis": computed["log2fc_basis"],
        "n_tested": computed["n_tested"],
        "n_up": computed["n_up"],
        "n_down": computed["n_down"],
        "up": computed["up"],
        "down": computed["down"],
        "params": dict(params or {}),
    }


def _de_metrics(computed: dict) -> dict:
    frame = computed["frame"]
    return {
        "method": computed["method"],
        "contrast": f"{computed['contrast']['case']} vs {computed['contrast']['control']}",
        "n_case": int(computed["comp"]["n_case"]),
        "n_control": int(computed["comp"]["n_control"]),
        "n_tested": computed["n_tested"],
        "n_up": computed["n_up"],
        "n_down": computed["n_down"],
        "fdr": computed["fdr"],
        "fc": computed["fc"],
        "log2fc_basis": computed["log2fc_basis"],
        "n_features": int(len(frame)),
        "min_p_adj": None if frame["p_adj"].isna().all() else float(frame["p_adj"].min()),
    }


def _de_bullets(computed: dict) -> list[str]:
    frame = computed["frame"]
    bullets = [
        f"比较：{computed['contrast']['case']}（n={computed['comp']['n_case']}） vs "
        f"{computed['contrast']['control']}（n={computed['comp']['n_control']}）；"
        f"方法 {computed['method']}，log2FC 口径 {computed['log2fc_basis']}",
        f"FDR≤{computed['fdr']:g} 且 |FC|≥{computed['fc']:g}：上调 {computed['n_up']} 个、"
        f"下调 {computed['n_down']} 个（有效检验 {computed['n_tested']}/{len(frame)} 个特征）",
    ]
    top = frame.head(5)
    if not top.empty:
        names = top["name"].astype(str).where(top["name"].astype(str).str.len() > 0, top["feature"].astype(str))
        bullets.append("最显著的 5 个特征：" + "、".join(map(str, names)))
        bullets.append(f"最显著特征的 FDR = {float(top['p_adj'].iloc[0]):.4g}，"
                       f"log2FC = {float(top['log2FC'].iloc[0]):.3f}")
    return bullets


# ---------------------------------------------------------------------------
# 缓存（dataset.results["de"]）
# ---------------------------------------------------------------------------


def _coerce_de_frame(stored) -> pd.DataFrame | None:
    """把 results["de"] 的各种形态统一成 DataFrame，并把列名归一化。"""
    frame = None
    if isinstance(stored, pd.DataFrame):
        frame = stored
    elif isinstance(stored, dict):
        for key in ("table", "de_table", "data", "rows"):
            value = stored.get(key)
            if isinstance(value, pd.DataFrame):
                frame = value
                break
            if isinstance(value, list) and value and isinstance(value[0], dict):
                frame = pd.DataFrame(value)
                break
    elif isinstance(stored, list) and stored and isinstance(stored[0], dict):
        frame = pd.DataFrame(stored)
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    rename = {}
    if "feature" not in frame.columns:
        for candidate in ("gene", "id", "feature_id", "metabolite"):
            if candidate in frame.columns:
                rename[candidate] = "feature"
                break
        else:
            frame["feature"] = [str(i) for i in frame.index]
    for src, dst in (("gene", "feature"), ("logFC", "log2FC"), ("P.Value", "p"),
                     ("adj.P.Val", "p_adj"), ("padj", "p_adj"), ("pvalue", "p"),
                     ("regulation", "regulation")):
        if src in frame.columns and dst not in frame.columns:
            rename[src] = dst
    if rename:
        frame = frame.rename(columns=rename)
    if "p" not in frame.columns or "log2FC" not in frame.columns:
        return None
    frame["feature"] = frame["feature"].astype(str)
    for col in ("log2FC", "p", "p_adj", "mean_case", "mean_control", "mz", "rt", "detected_frac"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "p_adj" not in frame.columns:
        frame["p_adj"] = bh_fdr(frame["p"].to_numpy())
    if "regulation" not in frame.columns:
        frame["regulation"] = "ns"
    if "name" not in frame.columns:
        frame["name"] = ""
    return frame


def _cached_de(ctx) -> pd.DataFrame | None:
    results = ctx.get("results")
    if not isinstance(results, dict):
        return None
    return _coerce_de_frame(results.get("de"))


def _apply_thresholds(frame: pd.DataFrame, fdr: float, fc: float) -> pd.DataFrame:
    """按分析参数重新标注 regulation/fc_hit（不重跑检验）。"""
    out = frame.copy()
    log2fc = pd.to_numeric(out["log2FC"], errors="coerce").to_numpy(dtype=float)
    p_adj = pd.to_numeric(out["p_adj"], errors="coerce").to_numpy(dtype=float)
    log2_cut = math.log2(max(fc, 1e-6))
    fc_hit = np.abs(log2fc) >= log2_cut
    regulation = np.full(len(out), "ns", dtype=object)
    regulation[(p_adj <= fdr) & fc_hit & (log2fc > 0)] = "up"
    regulation[(p_adj <= fdr) & fc_hit & (log2fc < 0)] = "down"
    out["fc_hit"] = fc_hit
    out["regulation"] = regulation
    return out


def _ensure_de(ctx, params: dict) -> tuple[pd.DataFrame, dict]:
    """取 DE 表：优先用缓存的步骤结果，缺失时现场计算。返回 (frame, meta)。"""
    params = _pdict(params)
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    fc = _float_param(params, "fc", 1.5, 1e-6, None)
    cached = _cached_de(ctx)
    if cached is not None:
        frame = _apply_thresholds(cached, fdr, fc)
        store = (ctx.get("results") or {}).get("de")
        meta = {}
        if isinstance(store, dict):
            meta = {k: store.get(k) for k in ("method", "case", "control", "group_key",
                                              "contrast", "comparison", "log2fc_basis", "paired")}
        meta.setdefault("method", "cached")
        return frame, meta
    computed = _de_compute(ctx, params)
    return computed["frame"], {
        "method": computed["method"], "case": computed["contrast"]["case"],
        "control": computed["contrast"]["control"], "group_key": computed["group_key"],
        "log2fc_basis": computed["log2fc_basis"], "paired": computed["paired"],
        "_computed": computed,
    }


def _top_features(frame: pd.DataFrame, top_n: int, by: str = "p") -> pd.DataFrame:
    """按显著性/效应量取 top 特征（跳过 p 为 NaN 的行，但若全 NaN 则退回|log2FC|）。"""
    sub = frame.copy()
    if "p_adj" in sub.columns:
        p = pd.to_numeric(sub["p_adj"], errors="coerce")
    else:
        p = pd.to_numeric(sub["p"], errors="coerce")
    fc = pd.to_numeric(sub["log2FC"], errors="coerce").abs()
    if by == "fc":
        key = fc.fillna(0.0)
        sub = sub.assign(_sort=-key)
    else:
        key = p
        sub = sub.assign(_sort=key.fillna(np.inf))
    sub = sub.sort_values(["_sort"], kind="mergesort").drop(columns=["_sort"])
    return sub.head(int(top_n)).reset_index(drop=True)


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


def _feature_label(row, limit: int = 22) -> str:
    name = str(row.get("name") or "").strip()
    if name:
        return _short(name, limit)
    return _short(str(row.get("feature") or ""), limit)


def _volcano_arrays(frame: pd.DataFrame, fdr: float, fc: float):
    x = pd.to_numeric(frame["log2FC"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(frame["p"], errors="coerce").to_numpy(dtype=float)
    if "p_adj" in frame.columns:
        p_adj = pd.to_numeric(frame["p_adj"], errors="coerce").to_numpy(dtype=float)
    else:
        # 容错：没有 p_adj 列时退回未校正 p（不崩，只是着色按 p 判定）
        p_adj = np.array(p, dtype=float, copy=True)
    with np.errstate(invalid="ignore"):
        y = -np.log10(np.clip(p, 1e-300, 1.0))
    log2_cut = math.log2(max(fc, 1e-6))
    up = np.isfinite(x) & np.isfinite(y) & (p_adj <= fdr) & (x >= log2_cut)
    down = np.isfinite(x) & np.isfinite(y) & (p_adj <= fdr) & (x <= -log2_cut)
    ns = np.isfinite(x) & np.isfinite(y) & ~up & ~down
    return x, y, up, down, ns, log2_cut


def _render_volcano(frame: pd.DataFrame, params: dict, outdir: str) -> dict:
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    fc = _float_param(params, "fc", 1.5, 1e-6, None)
    n_labels = _int_param(params, "n_labels", 15, 0, 100)
    x, y, up, down, ns, log2_cut = _volcano_arrays(frame, fdr, fc)
    # 标注用：局部取 p_adj（_volcano_arrays 内部变量不外泄）
    p_adj = pd.to_numeric(frame["p_adj"], errors="coerce").to_numpy(dtype=float) \
        if "p_adj" in frame.columns else np.full(len(frame), np.nan)
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = new_fig(7.4, 5.4)
    ax.scatter(x[ns], y[ns], s=11, color="#b7bdc7", alpha=0.65, linewidth=0, zorder=2,
               label=f"ns ({int(ns.sum())})")
    ax.scatter(x[up], y[up], s=17, color="#e15759", alpha=0.85, linewidth=0, zorder=3,
               label=f"up ({int(up.sum())})")
    ax.scatter(x[down], y[down], s=17, color="#4e79a7", alpha=0.85, linewidth=0, zorder=3,
               label=f"down ({int(down.sum())})")
    ax.axhline(-math.log10(max(fdr, 1e-300)), color="#9aa4b2", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(log2_cut, color="#9aa4b2", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(-log2_cut, color="#9aa4b2", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 P value")
    ax.set_title(f"Volcano plot (FC cut = {fc:g}, FDR = {fdr:g}, points = {int(finite.sum())})",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    # 标注最显著 / 效应量最大的点
    if n_labels > 0 and finite.any():
        order = np.argsort(np.where(np.isfinite(p_adj), p_adj, np.inf))[:n_labels]
        for idx in order:
            if not finite[idx]:
                continue
            row = frame.iloc[int(idx)]
            ax.annotate(_feature_label(row, 18), (x[idx], y[idx]), fontsize=6.2, color="#3a3f47",
                        xytext=(3, 2), textcoords="offset points", zorder=5)
    return _save(fig, outdir, "volcano-de", params, n_points=int(finite.sum()),
                 n_up=int(up.sum()), n_down=int(down.sum()))


def _render_fc_bar(frame: pd.DataFrame, params: dict, outdir: str) -> dict:
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    fc = _float_param(params, "fc", 1.5, 1e-6, None)
    top_n = _int_param(params, "top_n", 20, 1, 100)
    tagged = _apply_thresholds(frame, fdr, fc)
    sig = tagged[tagged["regulation"] != "ns"]
    note = None
    pool = sig if not sig.empty else tagged
    if sig.empty:
        note = "没有特征达到 FDR/FC 阈值，退回展示 |log2FC| 最大的特征"
    sub = pool.reindex(pd.to_numeric(pool["log2FC"], errors="coerce").abs()
                       .sort_values(ascending=False).index).head(top_n).iloc[::-1]
    values = pd.to_numeric(sub["log2FC"], errors="coerce").to_numpy(dtype=float)
    labels = [_feature_label(row) for _, row in sub.iterrows()]
    colors = ["#e15759" if v > 0 else "#4e79a7" for v in values]
    fig, ax = new_fig(7.6, max(3.4, 0.27 * len(sub) + 1.5))
    ax.barh(np.arange(len(sub)), values, color=colors, height=0.72, zorder=3)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.0, color="#c8ccd4", linewidth=0.8, zorder=2)
    ax.axvline(math.log2(max(fc, 1e-6)), color="#9aa4b2", linewidth=0.9, linestyle="--", zorder=2)
    ax.axvline(-math.log2(max(fc, 1e-6)), color="#9aa4b2", linewidth=0.9, linestyle="--", zorder=2)
    ax.set_xlabel("log2 fold change")
    ax.set_title(f"Fold change (top {len(sub)}; dashed = +/-log2({fc:g}))", fontsize=10)
    ax.grid(axis="y", visible=False)
    out = _save(fig, outdir, "fc-bar", params, n_features=int(len(sub)))
    if note:
        out["note"] = note
    return out


def _heatmap_payload(ctx, frame: pd.DataFrame, params: dict) -> dict:
    """取热图用的特征 z-score 矩阵与样本顺序（按分组排序）。"""
    top_n = _int_param(params, "top_n", 50, 2, 300)
    sub = _top_features(frame, top_n, by="p")
    X, features, samples = _matrix(ctx)
    feature_index = {f: i for i, f in enumerate(features)}
    keep = [f for f in sub["feature"].astype(str) if f in feature_index]
    if len(keep) < 2:
        raise prerequisite("DE 结果里没有可用于热图的特征（特征名与矩阵不匹配）")
    idx = [feature_index[f] for f in keep]
    mat = X[:, idx].T                                 # 特征 × 样本
    scaled = zscore_rows(mat)                         # 按行 z-score
    labels = None
    info = _sample_info_frame(ctx, samples)
    key = str(params.get("group_col") or ctx.get("group_key") or "").strip()
    if key in info.columns:
        labels = _value_labels(info, key, samples)
    order = np.arange(len(samples))
    if labels is not None:
        group_order: list[str] = []
        for value in labels:
            if value not in group_order:
                group_order.append(value)
        ranks = {g: i for i, g in enumerate(group_order)}
        order = np.array(sorted(order, key=lambda i: (ranks.get(labels[i], 999), i)))
    scaled = scaled[:, order]
    cluster = _bool_param(params, "cluster", True)
    row_order = np.arange(len(keep))
    if cluster and len(keep) > 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import pdist

            dist = pdist(scaled, metric="correlation")
            dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
            row_order = leaves_list(linkage(dist, method="average"))
        except Exception:  # noqa: BLE001 - 距离全 NaN 等退化情形
            row_order = np.arange(len(keep))
    return {
        "matrix": scaled[row_order],
        "features": [keep[i] for i in row_order],
        "samples": [samples[i] for i in order],
        "labels": None if labels is None else np.asarray(labels, dtype=object)[order],
        "n_features": int(len(keep)),
    }


def _render_heatmap(ctx, frame: pd.DataFrame, params: dict, outdir: str) -> dict:
    payload = _heatmap_payload(ctx, frame, params)
    mat = payload["matrix"]
    n_features, n_samples = mat.shape
    fig, ax = new_fig(max(6.0, min(14.0, 0.18 * n_samples + 3.2)),
                      max(3.6, 0.20 * n_features + 1.6))
    image = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    meta = _feature_meta_frame(ctx, payload["features"])
    if n_features <= 60:
        names = []
        for feature in payload["features"]:
            row = meta.loc[feature] if feature in meta.index else {}
            names.append(_short(str(row.get("name") or feature), 20))
        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(names, fontsize=6)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{n_features} features (row z-score)")
    if payload["labels"] is not None:
        groups = list(dict.fromkeys(map(str, payload["labels"])))
        centers = [float(np.mean(np.flatnonzero(payload["labels"] == g))) for g in groups if np.any(payload["labels"] == g)]
        names = [g for g in groups if np.any(payload["labels"] == g)]
        ax.set_xticks(centers)
        ax.set_xticklabels(names, fontsize=8)
    elif n_samples <= 40:
        ax.set_xticks(np.arange(n_samples))
        ax.set_xticklabels([_short(s, 12) for s in payload["samples"]], fontsize=6, rotation=90)
    else:
        ax.set_xticks([])
    ax.set_xlabel("sample (ordered by group)")
    ax.set_title(f"Top {n_features} differential features (row z-score)", fontsize=10)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("z-score", fontsize=8)
    return _save(fig, outdir, "heatmap-top", params, n_features=int(n_features),
                 n_samples=int(n_samples))


def _trend_payload(ctx, params: dict) -> dict:
    """时间/剂量趋势数据：分箱均值 ± SE + 线性回归 + Spearman。"""
    params = _pdict(params)
    X, features, samples = _matrix(ctx)
    info = _sample_info_frame(ctx, samples)
    time_col = params.get("time_col")
    time_col = str(time_col).strip() if time_col else ""
    if not time_col:
        time_col = _guess_column(info, TIME_COL_HINTS) or ""
    if not time_col or time_col not in info.columns:
        raise prerequisite(
            f"请指定时间/剂量列（time_col）；样本信息里没有识别到时间列（可选列："
            f"{'、'.join(map(str, info.columns)) or '无'}）"
        )
    raw = pd.to_numeric(info[time_col], errors="coerce")
    valid_time = raw.notna().to_numpy()
    if valid_time.sum() < 3:
        raise prerequisite(f"时间列「{time_col}」只有 {int(valid_time.sum())} 个有效数值，无法做趋势分析")
    x_all = raw.to_numpy(dtype=float)[valid_time]
    if len(np.unique(x_all)) < 2:
        raise prerequisite(f"时间列「{time_col}」只有一个取值（{np.unique(x_all)}），无法做趋势/剂量分析")
    sub_idx = np.flatnonzero(valid_time)

    feature_index = {f: i for i, f in enumerate(features)}
    wanted = params.get("features")
    if isinstance(wanted, str):
        wanted = [item.strip() for item in wanted.replace(";", ",").split(",") if item.strip()]
    wanted = [str(f) for f in (wanted or []) if str(f) in feature_index]
    picked: list[str] = wanted
    if not picked:
        cached = _cached_de(ctx)
        if cached is not None and not cached.empty:
            cached = _apply_thresholds(cached, _float_param(params, "fdr", 0.05),
                                       _float_param(params, "fc", 1.5))
            picked = [f for f in cached["feature"].astype(str).head(8) if f in feature_index]
        if not picked:
            # 与时间相关性最强的特征
            corr = []
            for feature in features:
                col = X[sub_idx, feature_index[feature]]
                ok = np.isfinite(col)
                if ok.sum() < 3 or len(np.unique(col[ok])) < 2:
                    corr.append(0.0)
                    continue
                try:
                    rho = float(stats.spearmanr(x_all[ok], col[ok]).statistic)
                except Exception:  # noqa: BLE001
                    rho = 0.0
                corr.append(0.0 if not math.isfinite(rho) else abs(rho))
            order = np.argsort(-np.asarray(corr))
            picked = [features[i] for i in order[: min(8, len(features))]]
    if not picked:
        raise prerequisite("没有可用于趋势分析的特征")
    picked = picked[: _int_param(params, "top_n", 8, 1, 30)]

    meta = _feature_meta_frame(ctx, picked)
    rows = []
    series = []
    unique_x = np.unique(x_all)
    for feature in picked:
        y_all = X[sub_idx, feature_index[feature]]
        ok = np.isfinite(y_all)
        if ok.sum() < 3:
            continue
        xv = x_all[ok]
        yv = y_all[ok]
        row = {"feature": feature, "name": meta.loc[feature, "name"],
               "mz": meta.loc[feature, "mz"], "rt": meta.loc[feature, "rt"]}
        row["n"] = int(xv.size)
        try:
            fit = stats.linregress(xv, yv)
            slope, intercept, r_value, p_slope = float(fit.slope), float(fit.intercept), \
                float(fit.rvalue), float(fit.pvalue)
        except ValueError:
            slope = intercept = p_slope = float("nan")
            r_value = float("nan")
        row["slope"] = slope
        row["p_slope"] = p_slope
        try:
            rho_res = stats.spearmanr(xv, yv)
            rho, p_rho = float(rho_res.statistic), float(rho_res.pvalue)
        except Exception:  # noqa: BLE001
            rho = p_rho = float("nan")
        row["rho_spearman"] = rho
        row["p_spearman"] = p_rho
        trend = "flat"
        if math.isfinite(p_slope) and p_slope <= 0.05:
            trend = "increasing" if slope > 0 else "decreasing"
        row["trend"] = trend
        # 分箱均值 ± SE：时间点 ≤8 个时按原值分组，否则按分位数分 6 箱
        if unique_x.size <= 8:
            binned = [(float(value), yv[np.isclose(xv, value)]) for value in unique_x]
        else:
            edges = np.unique(np.quantile(xv, np.linspace(0.0, 1.0, 7)))
            if edges.size < 3:
                binned = [(float(value), yv[np.isclose(xv, value)]) for value in np.unique(xv)]
            else:
                inner = edges[1:-1]
                assign = np.clip(np.digitize(xv, inner), 0, len(edges) - 2)
                binned = []
                for b in range(len(edges) - 1):
                    sel = assign == b
                    if sel.any():
                        binned.append((float(xv[sel].mean()), yv[sel]))
        xs, means, ses = [], [], []
        for center, ys in binned:
            ys = ys[np.isfinite(ys)]
            if ys.size == 0:
                continue
            xs.append(float(center))
            means.append(float(ys.mean()))
            ses.append(float(ys.std(ddof=1) / math.sqrt(ys.size)) if ys.size > 1 else 0.0)
        series.append({"feature": feature, "label": _feature_label(row, 24),
                       "x": xs, "mean": means, "se": ses})
        first, last = (means[0], means[-1]) if means else (float("nan"), float("nan"))
        row["span_change"] = (last - first) if means else float("nan")
        rows.append(row)
    if not rows:
        raise prerequisite("趋势分析没有可用的特征（都缺少有效观测）")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["p_slope"], kind="mergesort", na_position="last").reset_index(drop=True)
    return {
        "frame": frame,
        "series": series,
        "time_col": str(time_col),
        "x": x_all,
        "sub_idx": sub_idx,
        "warnings": [],
    }


def _render_trend(payload: dict, params: dict, outdir: str) -> dict:
    series = payload.get("series") or []
    if not series:
        raise prerequisite("没有可画的时间趋势特征")
    fig, ax = new_fig(7.2, 4.8)
    for i, item in enumerate(series[:6]):
        color = PALETTE[i % len(PALETTE)]
        xs = np.asarray(item["x"], dtype=float)
        means = np.asarray(item["mean"], dtype=float)
        ses = np.asarray(item["se"], dtype=float)
        ax.errorbar(xs, means, yerr=ses, marker="o", markersize=4, linewidth=1.3,
                    color=color, capsize=2.5, elinewidth=0.9, label=item["label"], zorder=3)
    ax.set_xlabel(f"{payload.get('time_col') or 'time'}")
    ax.set_ylabel("mean intensity (mean +/- SE)")
    ax.set_title(f"Metabolite trajectories over {payload.get('time_col') or 'time'}", fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2 if len(series) > 4 else 1)
    return _save(fig, outdir, "trend-plot", params, n_features=int(len(series)))


# ---------------------------------------------------------------------------
# 4PL 剂量响应
# ---------------------------------------------------------------------------


def _four_pl(t, bottom, top, log_ec50, hill):
    """4 参数 logistic 在 log10 剂量 t 上的形式：bottom + (top-bottom)/(1+10^(hill*(log_ec50-t)))。

    指数做 ±300 限幅：hill 与 log_ec50 在拟合/后验抽样里可能取到极端值，未限幅的
    `10.0 ** x` 对标量输入会抛 OverflowError（数组输入退化为 inf），限幅后始终有限。
    """
    exponent = np.clip(hill * (log_ec50 - t), -300.0, 300.0)
    return bottom + (top - bottom) / (1.0 + 10.0 ** exponent)


def _fit_4pl(x: np.ndarray, y: np.ndarray) -> dict:
    """4PL 拟合（curve_fit，多初值重试）；不收敛/数据不足时抛可读错误。"""
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size < 5:
        raise prerequisite(f"有效剂量点只有 {x.size} 个（需要至少 5 个，且剂量必须 > 0）")
    if np.unique(x).size < 3:
        raise prerequisite(f"剂量列只有 {np.unique(x).size} 个不同取值，无法拟合 4 参数 logistic")
    logx = np.log10(x)
    span = float(np.ptp(y))
    if not math.isfinite(span) or span <= 0:
        raise prerequisite("该特征在样本间没有变化（响应范围 0），无法拟合剂量响应")
    # 初值：低/高剂量两端 + 两种方向（降曲线靠交换 bottom/top 表达）
    top0 = float(np.max(y))
    bottom0 = float(np.min(y))
    logc0 = float(np.median(logx))
    inits = [
        [bottom0, top0, logc0, 1.0],
        [top0, bottom0, logc0, 1.0],
        [bottom0, top0, float(np.percentile(logx, 25)), 2.0],
        [top0, bottom0, float(np.percentile(logx, 75)), 2.0],
    ]
    # 边界收紧：Hill 斜率 0.1–10，EC50 限制在实测剂量范围内（不外推），避免无关特征"跑飞"
    lower = [-np.inf, -np.inf, float(logx.min()), 0.1]
    upper = [np.inf, np.inf, float(logx.max()), 10.0]
    best = None
    errors: list[str] = []
    for p0 in inits:
        try:
            popt, pcov = curve_fit(_four_pl, logx, y, p0=p0, maxfev=200000,
                                   bounds=(lower, upper), method="trf")
        except (RuntimeError, ValueError, TypeError, OverflowError, FloatingPointError) as error:
            errors.append(str(error)[:120])
            continue
        try:
            pred = _four_pl(logx, *popt)
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:  # noqa: BLE001
            r2 = float("nan")
        cand = {"popt": np.asarray(popt, dtype=float), "pcov": np.asarray(pcov, dtype=float),
                "r2": float(r2), "n": int(x.size), "log10_x": logx, "y": y}
        if best is None or (math.isfinite(cand["r2"]) and cand["r2"] > best["r2"]):
            best = cand
    if best is None:
        detail = "；".join(errors[:2]) if errors else "未找到收敛解"
        raise prerequisite(f"4 参数 logistic（4PL）拟合不收敛：{detail}。"
                           f"请检查剂量列是否单调、特征与剂量是否相关")
    # 质量门槛：R² < 0.05 视为"不呈剂量响应"（否则会把无关特征的垃圾拟合当 EC50 返回）；
    # 0.05–0.2 仍给出结果但标注拟合质量低，交给人判断。
    if not all(np.isfinite(best["popt"])) or not math.isfinite(best["r2"]) or best["r2"] < 0.05:
        raise prerequisite(
            f"4 参数 logistic（4PL）拟合不收敛或不呈剂量响应（R²={best['r2']:.3f} < 0.05）；"
            f"请检查剂量列范围是否覆盖响应变化、该特征是否与剂量相关"
        )
    bottom, top, log_ec50, hill = (float(v) for v in best["popt"])
    if not math.isfinite(hill) or abs(hill) < 1e-9:
        raise prerequisite("4PL 拟合得到的 Hill 斜率接近 0（响应与剂量无关），无法给出 EC50")
    warning = None
    if best["r2"] < 0.2:
        warning = (f"4PL 拟合质量较低（R²={best['r2']:.3f}）：该特征与剂量的关系弱，"
                   f"EC50 仅供参考，建议换特征或确认剂量设计")
    return {
        "bottom": bottom,
        "top": top,
        "log_ec50": log_ec50,
        "ec50": _pow10(log_ec50),
        "hill": hill,
        "r2": float(best["r2"]),
        "n": int(best["n"]),
        "pcov": best["pcov"],
        "log10_x": best["log10_x"],
        "y": best["y"],
        "direction": "increasing" if hill * (top - bottom) > 0 else "decreasing",
        "warning": warning,
    }


def _pow10(value) -> float | None:
    """10^value 的安全版本：非有限/溢出的置信区间端点返回 None（避免 OverflowError）。"""
    if value is None or not math.isfinite(float(value)):
        return None
    try:
        out = float(10.0 ** max(min(float(value), 250.0), -250.0))
    except (OverflowError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _corr_with(x: np.ndarray, Y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """nan-aware 的逐特征 Pearson 相关（自动挑"与剂量最相关"的特征时做预筛，向量化）。"""
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    ok = np.isfinite(Y) & valid[:, None] & np.isfinite(x)[:, None]
    n = ok.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        xs = np.where(ok, x[:, None], 0.0)
        ys = np.where(ok, Y, 0.0)
        mx = xs.sum(axis=0) / np.maximum(n, 1.0)
        my = ys.sum(axis=0) / np.maximum(n, 1.0)
        cov = (xs * ys).sum(axis=0) / np.maximum(n, 1.0) - mx * my
        vx = (xs * xs).sum(axis=0) / np.maximum(n, 1.0) - mx * mx
        vy = (ys * ys).sum(axis=0) / np.maximum(n, 1.0) - my * my
        denom = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        r = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), np.nan)
    return r


def _dose_payload(ctx, params: dict) -> dict:
    """剂量响应数据：选特征 → 取剂量列 → 4PL 拟合 → 表 + 参数 CI。"""
    params = _pdict(params)
    X, features, samples = _matrix(ctx)
    info = _sample_info_frame(ctx, samples)
    dose_col = params.get("dose_col")
    dose_col = str(dose_col).strip() if dose_col else ""
    if not dose_col:
        dose_col = _guess_column(info, ("dose", "concentration", "conc", "level", "dilution")) or ""
    if not dose_col or dose_col not in info.columns:
        raise prerequisite(f"请指定剂量列（dose_col）；样本信息里没有识别到剂量列（可选列："
                           f"{'、'.join(map(str, info.columns)) or '无'}）")
    dose = pd.to_numeric(info[dose_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(dose)
    warnings: list[str] = []
    if (dose[ok] <= 0).any():
        warnings.append(f"剂量列「{dose_col}」含非正值（{int((dose[ok] <= 0).sum())} 个样本），"
                        f"已从 4PL 拟合中剔除（log 剂量要求 > 0）")
    feature_index = {f: i for i, f in enumerate(features)}
    feature = str(params.get("feature") or "").strip()
    if feature and feature not in feature_index:
        raise bad_request(f"特征「{feature}」不在当前矩阵里；先用 preview 查看特征 id")
    fit = None
    good = None
    if feature:
        y = X[:, feature_index[feature]]
        good = ok & np.isfinite(y) & (dose > 0)
        if good.sum() < 5:
            raise prerequisite(f"特征「{feature}」在有效剂量样本里只有 {int(good.sum())} 个观测，"
                               f"不足以拟合 4PL")
        fit = _fit_4pl(dose[good], y[good])
    else:
        # 自动选：按 |Pearson(剂量, 特征)| 预筛 → 前 5 个候选逐个试 4PL（拟合成功即用）
        corr = _corr_with(dose, X, ok & (dose > 0))
        order = np.argsort(-np.abs(np.nan_to_num(corr, nan=0.0)))
        candidates = [features[i] for i in order[:5]]
        errors: list[str] = []
        best_abs_r = 0.0
        for cand in candidates:
            y_c = X[:, feature_index[cand]]
            good_c = ok & np.isfinite(y_c) & (dose > 0)
            if good_c.sum() < 5 or len(np.unique(y_c[good_c])) < 2:
                continue
            try:
                fit_c = _fit_4pl(dose[good_c], y_c[good_c])
            except EngineError as error:
                errors.append(f"{cand}: {error.message[:70]}")
                continue
            feature, y, good, fit = cand, y_c, good_c, fit_c
            best_abs_r = abs(float(corr[feature_index[cand]]))
            break
        if fit is None:
            detail = "；".join(errors[:2]) or "前 5 个候选特征都不可用"
            raise prerequisite(
                f"没有找到与剂量列「{dose_col}」呈剂量响应的特征（自动扫描 |r| 最高的 5 个特征都失败："
                f"{detail}）。请显式指定 feature，或确认剂量列是否合理"
            )
        warnings.append(f"未指定 feature，已自动选择与剂量相关性最强且 4PL 可拟合的特征「{feature}」"
                        f"（|Pearson r|={best_abs_r:.3f}）")
    if fit.get("warning"):
        warnings.append(str(fit["warning"]))
    ci = _float_param(params, "ci", 0.95, 0.5, 0.999)
    z = float(stats.norm.ppf(0.5 + ci / 2.0))
    pcov = fit["pcov"]
    se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    se_logc = float(se[2]) if se.size >= 3 else float("nan")

    def _interval(name: str, value: float, index: int) -> tuple[float | None, float | None]:
        if not np.isfinite(value) or se.size <= index or not np.isfinite(se[index]):
            return None, None
        return float(value - z * se[index]), float(value + z * se[index])

    ec50_low = _pow10(fit["log_ec50"] - z * se_logc) if math.isfinite(se_logc) else None
    ec50_high = _pow10(fit["log_ec50"] + z * se_logc) if math.isfinite(se_logc) else None
    b_lo, b_hi = _interval("bottom", fit["bottom"], 0)
    t_lo, t_hi = _interval("top", fit["top"], 1)
    h_lo, h_hi = _interval("hill", fit["hill"], 3)
    meta = _feature_meta_frame(ctx, [feature])
    rows = [
        {"parameter": "bottom", "estimate": fit["bottom"], "ci_low": b_lo, "ci_high": b_hi},
        {"parameter": "top", "estimate": fit["top"], "ci_low": t_lo, "ci_high": t_hi},
        {"parameter": "EC50", "estimate": fit["ec50"], "ci_low": ec50_low, "ci_high": ec50_high},
        {"parameter": "Hill slope", "estimate": fit["hill"], "ci_low": h_lo, "ci_high": h_hi},
        {"parameter": "R2", "estimate": fit["r2"], "ci_low": None, "ci_high": None},
    ]
    return {
        "feature": feature,
        "name": str(meta.loc[feature, "name"]) if feature in meta.index else "",
        "mz": float(meta.loc[feature, "mz"]) if feature in meta.index else float("nan"),
        "rt": float(meta.loc[feature, "rt"]) if feature in meta.index else float("nan"),
        "dose_col": str(dose_col),
        "dose": dose[good],
        "y": y[good],
        "fit": fit,
        "ci": ci,
        "rows": rows,
        "warnings": warnings,
    }


def _render_dose(payload: dict, params: dict, outdir: str) -> dict:
    fit = payload["fit"]
    row = {"name": payload.get("name"), "feature": payload["feature"]}
    label = _feature_label(row, 30)
    dose = np.asarray(payload["dose"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    fig, ax = new_fig(6.8, 4.9)
    ax.scatter(dose, y, s=24, color="#4e79a7", alpha=0.75, edgecolor="white", linewidth=0.4,
               zorder=3, label="samples")
    order = np.argsort(dose)
    grid = np.logspace(np.log10(dose.min()), np.log10(dose.max()), 200)
    curve = _four_pl(np.log10(grid), fit["bottom"], fit["top"], fit["log_ec50"], fit["hill"])
    ax.plot(grid, curve, color="#e15759", linewidth=1.8, zorder=4, label="4PL fit")
    # 曲线 CI：参数协方差矩阵多元正态抽样（确定性 seed）
    rng = np.random.default_rng(SEED)
    pcov = np.asarray(fit["pcov"], dtype=float)
    try:
        draws = rng.multivariate_normal(fit["popt"] if "popt" in fit else
                                        np.array([fit["bottom"], fit["top"], fit["log_ec50"], fit["hill"]]),
                                        pcov, size=300)
    except Exception:  # noqa: BLE001
        draws = np.array([[fit["bottom"], fit["top"], fit["log_ec50"], fit["hill"]]])
    for sample in draws:
        try:
            ax.plot(grid, _four_pl(np.log10(grid), *sample), color="#f28e2b", linewidth=0.4,
                    alpha=0.05, zorder=2)
        except Exception:  # noqa: BLE001
            continue
    ec50 = float(fit["ec50"])
    ax.axvline(ec50, color="#59a14f", linewidth=1.1, linestyle="--", zorder=5,
               label=f"EC50 = {ec50:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel(f"dose ({payload.get('dose_col')}, log scale)")
    ax.set_ylabel(f"intensity ({label})")
    ax.set_title(f"Dose-response (4PL): {label}  R2={fit['r2']:.3f}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    return _save(fig, outdir, "dose-response", params, ec50=ec50, r2=float(fit["r2"]),
                 n_points=int(dose.size))


# ---------------------------------------------------------------------------
# PLOTS 注册函数（kind -> fn(ctx, params, outdir) -> figure dict）
# ---------------------------------------------------------------------------


def plot_volcano_de(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    frame, _meta = _ensure_de(ctx, params)
    return _render_volcano(frame, params, outdir)


def plot_fc_bar(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    frame, _meta = _ensure_de(ctx, params)
    return _render_fc_bar(frame, params, outdir)


def plot_heatmap_top(ctx, params=None, outdir=None):
    """差异特征热图（行 z-score + 层次聚类）。

    PLOTS 键用 `de_heatmap_top`：`heatmap_top` 这个 kind 归 mb-base 的 mb_plots.py
    （DESIGN §5 基础图归属），本模块不再抢占该 kind（见 lead 的分工裁定）。
    """
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    frame, _meta = _ensure_de(ctx, params)
    return _render_heatmap(ctx, frame, params, outdir)


def plot_trend_plot(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    payload = _trend_payload(ctx, params)
    return _render_trend(payload, params, outdir)


def plot_dose_response(ctx, params=None, outdir=None):
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    payload = _dose_payload(ctx, params)
    return _render_dose(payload, params, outdir)


PLOTS = {
    "volcano_de": plot_volcano_de,
    "fc_bar": plot_fc_bar,
    "de_heatmap_top": plot_heatmap_top,
    "trend_plot": plot_trend_plot,
    "dose_response": plot_dose_response,
}


# ---------------------------------------------------------------------------
# STEP de
# ---------------------------------------------------------------------------


STEP_PARAMS = [
    {"key": "group_col", "label": "分组列", "type": "text", "default": None},
    {"key": "case", "label": "实验组", "type": "text", "default": None},
    {"key": "control", "label": "对照组", "type": "text", "default": None},
    {"key": "method", "label": "方法", "type": "select", "default": "welch",
     "options": [{"value": "welch", "label": "Welch t 检验"},
                 {"value": "wilcoxon", "label": "Mann-Whitney U"},
                 {"value": "limma", "label": "limma 经验贝叶斯（内置近似）"}]},
    {"key": "paired", "label": "配对", "type": "bool", "default": False},
    {"key": "pair_col", "label": "配对列（可选）", "type": "text", "default": None},
    {"key": "fdr", "label": "FDR 阈值", "type": "number", "default": 0.05},
    {"key": "fc", "label": "倍数阈值", "type": "number", "default": 1.5},
]


def step_de(ctx, params=None, outdir=None):
    """步骤：单变量差异（Welch / Mann-Whitney / limma + BH）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    computed = _de_compute(ctx, params)
    frame = computed["frame"]
    warnings = list(dict.fromkeys(computed["warnings"]))
    store = _de_store(computed, params)
    metrics = _de_metrics(computed)
    sig = frame[frame["regulation"] != "ns"]

    tables = [
        table("de", "差异特征表（按 FDR 升序）", DE_COLUMNS, rows=frame,
              note=f"共 {len(frame)} 个特征；regulation 已内含 FDR≤{computed['fdr']:g} 与 "
                   f"|FC|≥{computed['fc']:g} 双阈值；完整表见 data_out"),
        table("de_sig", "显著差异特征（Top 200）", DE_COLUMNS, rows=sig.head(200),
              note=f"显著特征 {len(sig)} 个（上调 {computed['n_up']} / 下调 {computed['n_down']}）"),
    ]
    figures = [
        _render_volcano(frame, params, outdir),
        _render_fc_bar(frame, params, outdir),
    ]
    try:
        figures.append(_render_heatmap(ctx, frame, {**params, "top_n": 30}, outdir))
    except EngineError as error:
        warnings.append(f"热图未生成：{error.message}")

    data_out = []
    try:
        path = export_csv(frame, outdir, "de_all", index=False)
        data_out.append(data_file("csv", path, "差异全表"))
        path = export_csv(sig, outdir, "de_sig", index=False)
        data_out.append(data_file("csv", path, "显著差异表"))
    except Exception as error:  # noqa: BLE001 - 导出失败不该阻断步骤
        warnings.append(f"差异表导出失败：{error}")

    payload = result(
        title="单变量差异",
        bullets=_de_bullets(computed),
        metrics=metrics,
        tables=tables,
        figures=figures,
        warnings=warnings,
        notes=[f"用时 {timer.ms} ms；log2FC 口径 {computed['log2fc_basis']}；"
               f"FDR 用 Benjamini-Hochberg；t/limma 均为本机内置实现（无 R limma 依赖）"],
        data_out=data_out,
    )
    payload["store"] = {"de": store}
    return payload


# ---------------------------------------------------------------------------
# ANALYSES
# ---------------------------------------------------------------------------


def _analysis_meta(meta: dict) -> dict:
    return {k: v for k, v in (meta or {}).items() if k != "_computed"}


def analysis_univariate_de(ctx, params=None, outdir=None):
    """单变量差异（Welch / Mann-Whitney / limma + BH）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    computed = _de_compute(ctx, params)
    frame = computed["frame"]
    warnings = list(dict.fromkeys(computed["warnings"]))
    figures = [_render_volcano(frame, params, outdir), _render_fc_bar(frame, params, outdir)]
    payload = result(
        title="单变量差异",
        bullets=_de_bullets(computed),
        metrics=_de_metrics(computed),
        tables=[table("de", "差异特征表（按 FDR 升序）", DE_COLUMNS, rows=frame,
                      note="完整表见 data_out / 步骤结果缓存")],
        figures=figures,
        warnings=warnings,
        notes=[f"用时 {timer.ms} ms；方法 {computed['method']}，log2FC 口径 {computed['log2fc_basis']}"],
    )
    payload["store"] = {"de": _de_store(computed, params)}
    return payload


def analysis_volcano_de(ctx, params=None, outdir=None):
    """火山图：log2FC vs −log10 p，按 FDR/FC 着色并标注最显著特征。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    fc = _float_param(params, "fc", 1.5, 1e-6, None)
    n_labels = _int_param(params, "n_labels", 15, 0, 100)
    frame, meta = _ensure_de(ctx, params)
    x, y, up, down, ns, _cut = _volcano_arrays(frame, fdr, fc)
    finite = np.isfinite(x) & np.isfinite(y)
    figure = _render_volcano(frame, {**params, "n_labels": n_labels, "fdr": fdr, "fc": fc}, outdir)
    tagged = _apply_thresholds(frame, fdr, fc)
    sig = tagged[tagged["regulation"] != "ns"].copy()
    sig["neg_log10_p"] = -np.log10(np.clip(pd.to_numeric(sig["p"], errors="coerce"), 1e-300, 1.0))
    labeled = _top_features(sig if not sig.empty else tagged, n_labels, by="p")
    columns = [{"key": "feature", "label": "特征", "type": "string"},
               {"key": "name", "label": "名称", "type": "string"},
               {"key": "log2FC", "label": "log2FC", "type": "number"},
               {"key": "p", "label": "P 值", "type": "number"},
               {"key": "p_adj", "label": "FDR", "type": "number"},
               {"key": "regulation", "label": "方向", "type": "string"},
               {"key": "neg_log10_p", "label": "-log10 P", "type": "number"}]
    rows = labeled[[c for c in ("feature", "name", "log2FC", "p", "p_adj", "regulation") if c in labeled.columns]].copy()
    rows["neg_log10_p"] = -np.log10(np.clip(pd.to_numeric(rows["p"], errors="coerce"), 1e-300, 1.0))
    return result(
        title="火山图",
        bullets=[f"共 {int(finite.sum())} 个特征进入火山图：上调 {int(up.sum())}、下调 {int(down.sum())}、"
                 f"不显著 {int(ns.sum())}",
                 f"阈值：FDR≤{fdr:g}，|log2FC|≥{math.log2(max(fc, 1e-6)):.3f}（FC={fc:g}）",
                 f"标注最显著的 {len(labeled)} 个特征（表内为标注点）"],
        metrics={"n_points": int(finite.sum()), "n_up": int(up.sum()), "n_down": int(down.sum()),
                 "n_ns": int(ns.sum()), "fdr": fdr, "fc": fc,
                 "method": meta.get("method"), "contrast": meta.get("contrast") or
                 {"case": meta.get("case"), "control": meta.get("control")}},
        tables=[table("volcano", f"火山图标注点（{len(labeled)}）",
                      columns=columns, rows=rows.to_dict(orient="records"))],
        figures=[figure],
        notes=[f"用时 {timer.ms} ms；y 轴为未校正 P 值，颜色按 BH-FDR + FC 双阈值"],
    )


def analysis_fc_bar(ctx, params=None, outdir=None):
    """倍数变化条形图：显著特征里 |log2FC| 最大的 top/bottom。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    fc_threshold = _float_param(params, "fc", 1.5, 1e-6, None)
    top_n = _int_param(params, "top_n", 20, 1, 100)
    frame, _meta = _ensure_de(ctx, params)
    tagged = _apply_thresholds(frame, fdr, fc_threshold)
    sig = tagged[tagged["regulation"] != "ns"]
    pool = sig if not sig.empty else tagged
    order = pd.to_numeric(pool["log2FC"], errors="coerce").abs().sort_values(ascending=False).index
    top = pool.reindex(order).head(top_n)
    figure = _render_fc_bar(frame, {**params, "fdr": fdr, "fc": fc_threshold, "top_n": top_n}, outdir)
    rows = top[[c for c in ("feature", "name", "log2FC", "p", "p_adj", "regulation") if c in top.columns]].copy()
    rows["fold_change"] = np.power(2.0, pd.to_numeric(rows["log2FC"], errors="coerce"))
    warnings = []
    if sig.empty:
        warnings.append("没有特征达到 FDR/FC 阈值，展示的是 |log2FC| 最大的特征（不代表显著）")
    return result(
        title="倍数变化条形图",
        bullets=[f"展示 {len(rows)} 个 |log2FC| 最大的特征"
                 + ("（显著集内）" if not sig.empty else "（无显著特征，按全表）"),
                 f"阈值：FDR≤{fdr:g}，FC={fc_threshold:g}；显著特征 {len(sig)} 个"],
        metrics={"n_features": int(len(rows)), "n_sig": int(len(sig)),
                 "max_log2FC": None if rows.empty else float(rows["log2FC"].max()),
                 "min_log2FC": None if rows.empty else float(rows["log2FC"].min())},
        tables=[table("fc_bar", f"|log2FC| Top {len(rows)}",
                      columns=[{"key": "feature", "label": "特征", "type": "string"},
                               {"key": "name", "label": "名称", "type": "string"},
                               {"key": "log2FC", "label": "log2FC", "type": "number"},
                               {"key": "fold_change", "label": "倍数变化", "type": "number"},
                               {"key": "p_adj", "label": "FDR", "type": "number"},
                               {"key": "regulation", "label": "方向", "type": "string"}],
                      rows=rows.to_dict(orient="records"))],
        figures=[figure],
        warnings=warnings,
        notes=[f"用时 {timer.ms} ms"],
    )


def analysis_heatmap_top(ctx, params=None, outdir=None):
    """差异特征热图：top_n 差异特征 × 样本，行 z-score + 层次聚类。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    fc = _float_param(params, "fc", 1.5, 1e-6, None)
    top_n = _int_param(params, "top_n", 50, 2, 300)
    frame, _meta = _ensure_de(ctx, params)
    tagged = _apply_thresholds(frame, fdr, fc)
    sig = tagged[tagged["regulation"] != "ns"]
    pool = sig if not sig.empty else tagged
    payload = _heatmap_payload(ctx, pool, {**params, "top_n": top_n})
    figure = _render_heatmap(ctx, pool, {**params, "top_n": top_n}, outdir)
    meta = _feature_meta_frame(ctx, payload["features"])
    rows = []
    for feature in payload["features"]:
        if feature in meta.index:
            rows.append({"feature": feature, "name": meta.loc[feature, "name"],
                         "mz": meta.loc[feature, "mz"], "rt": meta.loc[feature, "rt"]})
    warnings = []
    if sig.empty:
        warnings.append("没有特征达到 FDR/FC 阈值，热图展示的是最显著的 top 特征（不代表显著）")
    return result(
        title="差异特征热图",
        bullets=[f"热图：{payload['n_features']} 个特征 × {len(payload['samples'])} 个样本，"
                 f"行 z-score{'（层次聚类排序）' if _bool_param(params, 'cluster', True) else ''}",
                 f"特征来自{'显著差异集' if not sig.empty else '全表最显著特征'}（FDR≤{fdr:g}, FC={fc:g}）"],
        metrics={"n_features": int(payload["n_features"]), "n_samples": int(len(payload["samples"])),
                 "n_sig": int(len(sig))},
        tables=[table("heatmap_features", "热图特征列表",
                      columns=[{"key": "feature", "label": "特征", "type": "string"},
                               {"key": "name", "label": "名称", "type": "string"},
                               {"key": "mz", "label": "m/z", "type": "number"},
                               {"key": "rt", "label": "RT (min)", "type": "number"}],
                      rows=rows)],
        figures=[figure],
        warnings=warnings,
        notes=[f"用时 {timer.ms} ms"],
    )


def _multi_means(X: np.ndarray, labels: np.ndarray, groups: list[str]):
    """按组算 nan-aware 的 n / 均值 / 方差（k×p）。"""
    k = len(groups)
    p = X.shape[1]
    counts = np.zeros((k, p))
    means = np.full((k, p), np.nan)
    variances = np.full((k, p), np.nan)
    sums = np.zeros((k, p))
    for i, group in enumerate(groups):
        sub = X[labels == group]
        n_g, mean_g, var_g = _nan_stats(sub)
        counts[i] = n_g
        means[i] = mean_g
        variances[i] = var_g
        sums[i] = np.nansum(sub, axis=0)
    return counts, sums, means, variances


def _anova_multi(X, labels, groups, method: str) -> dict:
    """多组比较：one-way ANOVA / Welch ANOVA / Kruskal-Wallis（逐特征）。"""
    k = len(groups)
    counts, sums, means, variances = _multi_means(X, labels, groups)
    p_count = X.shape[1]
    statistic = np.full(p_count, np.nan)
    pvalues = np.full(p_count, np.nan)
    df1 = np.full(p_count, float(k - 1))
    df2 = np.full(p_count, np.nan)
    if method in ("anova", "welch"):
        N = counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            grand = sums.sum(axis=0) / np.maximum(N, 1.0)
            if method == "anova":
                ssb = np.nansum(counts * (means - grand) ** 2, axis=0)
                ssw = np.nansum(np.maximum(counts - 1, 0) * variances, axis=0)
                df_err = N - k
                with np.errstate(invalid="ignore", divide="ignore"):
                    msb = ssb / max(k - 1, 1)
                    msw = ssw / np.maximum(df_err, 1e-12)
                    F = np.where(msw > 0, msb / np.where(msw > 0, msw, 1.0), np.nan)
                df2 = df_err
                valid = (df_err > 0) & np.isfinite(F)
                pvals = np.where(valid, stats.f.sf(np.where(valid, F, 0.0), k - 1,
                                                   np.maximum(df_err, 1e-12)), np.nan)
            else:
                w = np.where((variances > 0) & np.isfinite(variances),
                             counts / np.where(variances > 0, variances, 1.0), 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    V = np.nansum(w, axis=0)
                    wmean = np.nansum(w * means, axis=0) / np.where(V > 0, V, 1.0)
                    A = np.nansum(w * (means - wmean) ** 2, axis=0) / max(k - 1, 1)
                    term = np.nansum((1.0 - w / np.where(V > 0, V, 1.0)) ** 2 /
                                     np.maximum(counts - 1, 1.0), axis=0)
                    B = 1.0 + 2.0 * (k - 2) / max(k * k - 1, 1) * term
                    F = np.where(B > 0, A / np.where(B > 0, B, 1.0), np.nan)
                    df_w = np.where(term > 0, (k * k - 1) / (3.0 * np.maximum(term, 1e-12)), np.nan)
                df2 = df_w
                valid = np.isfinite(F) & np.isfinite(df_w) & (df_w > 0)
                pvals = np.where(valid, stats.f.sf(np.where(valid, F, 0.0), k - 1,
                                                   np.where(valid, df_w, 1.0)), np.nan)
        statistic, pvalues = F, pvals
    else:
        for j in range(p_count):
            samples = [X[labels == group, j] for group in groups]
            samples = [s[np.isfinite(s)] for s in samples]
            samples = [s for s in samples if s.size >= 1]
            if len(samples) < 2 or sum(s.size for s in samples) < 4:
                continue
            try:
                res = stats.kruskal(*samples)
            except ValueError:
                continue
            statistic[j] = float(res.statistic)
            pvalues[j] = float(res.pvalue)
        df2 = np.full(p_count, np.nan)
    return {"statistic": statistic, "p": pvalues, "df1": df1, "df2": df2,
            "counts": counts, "means": means, "variances": variances}


def _posthoc_one(values: list[np.ndarray], groups: list[str], method: str,
                 posthoc: str) -> list[dict]:
    """单个特征的事后两两比较（Tukey-Kramer / Games-Howell / Dunn）。"""
    valid = [(g, v) for g, v in zip(groups, values) if v.size >= 2]
    if len(valid) < 2:
        return []
    names = [g for g, _ in valid]
    arrs = [v for _, v in valid]
    means = np.array([float(v.mean()) for v in arrs])
    variances = np.array([float(v.var(ddof=1)) for v in arrs])
    counts = np.array([float(v.size) for v in arrs])
    rows: list[dict] = []
    if posthoc == "tukey":
        df_err = float(counts.sum() - len(arrs))
        mse = float(np.sum((counts - 1) * variances) / max(df_err, 1.0))
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                diff = float(means[i] - means[j])
                with np.errstate(invalid="ignore", divide="ignore"):
                    denom = math.sqrt(max(mse / 2.0, 0.0) * (1.0 / counts[i] + 1.0 / counts[j]))
                    q = abs(diff) / denom if denom > 0 else 0.0
                try:
                    p = float(stats.studentized_range.sf(q, len(arrs), max(df_err, 1.0)))
                except Exception:  # noqa: BLE001
                    p = float("nan")
                rows.append({"group_a": names[i], "group_b": names[j], "diff": diff, "p": p})
    elif posthoc == "games_howell":
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                diff = float(means[i] - means[j])
                with np.errstate(invalid="ignore", divide="ignore"):
                    var_i = variances[i] / counts[i]
                    var_j = variances[j] / counts[j]
                    se = math.sqrt(max(var_i + var_j, 0.0))
                    q = abs(diff) / (se / math.sqrt(2.0)) if se > 0 else 0.0
                    denom = (var_i ** 2 / max(counts[i] - 1, 1.0)
                             + var_j ** 2 / max(counts[j] - 1, 1.0))
                    df = (var_i + var_j) ** 2 / denom if denom > 0 else float(counts.sum() - len(arrs))
                try:
                    p = float(stats.studentized_range.sf(q, len(arrs), max(df, 1.0)))
                except Exception:  # noqa: BLE001
                    p = float("nan")
                rows.append({"group_a": names[i], "group_b": names[j], "diff": diff, "p": p})
    else:  # dunn
        pooled = np.concatenate(arrs)
        ranks = stats.rankdata(pooled)
        offsets = np.cumsum([0] + [v.size for v in arrs[:-1]])
        mean_ranks = np.array([float(ranks[int(off) : int(off + v.size)].mean())
                               for off, v in zip(offsets, arrs)])
        n_total = float(pooled.size)
        _, tie_counts = np.unique(pooled, return_counts=True)
        ties = float(np.sum(tie_counts ** 3 - tie_counts))
        sigma2 = n_total * (n_total + 1) / 12.0 - ties / (12.0 * max(n_total - 1.0, 1.0))
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                with np.errstate(invalid="ignore", divide="ignore"):
                    denom = math.sqrt(max(sigma2, 0.0) * (1.0 / counts[i] + 1.0 / counts[j]))
                    z = (mean_ranks[i] - mean_ranks[j]) / denom if denom > 0 else 0.0
                p = float(2.0 * stats.norm.sf(abs(z)))
                rows.append({"group_a": names[i], "group_b": names[j],
                             "diff": float(means[i] - means[j]), "p": p})
    if rows and posthoc == "dunn":
        adjusted = bh_fdr([row["p"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["p_adj"] = float(value)
    else:
        for row in rows:
            row["p_adj"] = row["p"]
    return rows


def analysis_multi_group_de(ctx, params=None, outdir=None):
    """多组比较：ANOVA / Welch ANOVA / Kruskal-Wallis + 事后检验（Tukey / Games-Howell / Dunn）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    method = _normalize_method(params.get("method") or "anova", MULTI_METHOD_ALIASES, "多组检验方法")
    posthoc = _normalize_method(params.get("posthoc") or
                                ("tukey" if method == "anova" else
                                 "games_howell" if method == "welch" else "dunn"),
                                POSTHOC_ALIASES, "事后检验")
    fdr = _float_param(params, "fdr", 0.05, 1e-6, 1.0)
    X, features, samples = _matrix(ctx)
    info = _sample_info_frame(ctx, samples)
    key = _group_column(ctx, info, params)
    if not key:
        raise prerequisite("还没有分组列：先在「样本」里设置分组列（group_col），多组比较需要分组")
    if key not in info.columns:
        raise prerequisite(f"样本信息里没有分组列「{key}」")
    labels = _value_labels(info, key, samples)
    groups: list[str] = []
    for value in labels:
        if value != "NA" and value not in groups:
            groups.append(value)
    if len(groups) < 2:
        raise prerequisite(f"分组列「{key}」只有 {len(groups)} 个有效分组，多组比较至少需要 2 组")
    missing = int(np.isnan(X).sum())
    warnings: list[str] = []
    if missing:
        warnings.append(f"矩阵含 {missing} 个缺失值：各组统计量用成对完整观测，建议先填补")
    if len(groups) == 2:
        warnings.append("只有 2 个分组：多组 ANOVA 等价于两组比较，建议直接用「单变量差异」")

    res = _anova_multi(X, labels, groups, method)
    statistic = res["statistic"]
    pvalues = res["p"]
    p_adj = bh_fdr(pvalues)
    means = res["means"]
    meta = _feature_meta_frame(ctx, features)
    best = np.nanargmax(np.where(np.isfinite(means), means, -np.inf), axis=0)
    worst = np.nanargmin(np.where(np.isfinite(means), means, np.inf), axis=0)
    rows = []
    for j, feature in enumerate(features):
        hi = best[j] if np.isfinite(means[best[j], j]) else None
        lo = worst[j] if np.isfinite(means[worst[j], j]) else None
        rows.append({
            "feature": str(feature),
            "name": meta["name"].iloc[j],
            "mz": meta["mz"].iloc[j],
            "rt": meta["rt"].iloc[j],
            "statistic": float(statistic[j]) if np.isfinite(statistic[j]) else None,
            "df1": float(res["df1"][j]) if np.isfinite(res["df1"][j]) else None,
            "df2": float(res["df2"][j]) if np.isfinite(res["df2"][j]) else None,
            "p": float(pvalues[j]) if np.isfinite(pvalues[j]) else None,
            "p_adj": float(p_adj[j]) if np.isfinite(p_adj[j]) else None,
            "max_group": groups[hi] if hi is not None else None,
            "min_group": groups[lo] if lo is not None else None,
            "max_diff": float(means[hi, j] - means[lo, j])
            if hi is not None and lo is not None else None,
        })
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["p_adj", "p"], ascending=True, na_position="last").reset_index(drop=True)
    n_sig = int((pd.to_numeric(frame["p_adj"], errors="coerce") <= fdr).sum())

    # 事后检验：显著特征（最多 30 个）逐两两比较
    posthoc_rows: list[dict] = []
    if posthoc != "none":
        selected = frame[pd.to_numeric(frame["p_adj"], errors="coerce") <= fdr].head(30)
        if selected.empty:
            selected = frame.head(10)
        feature_index = {f: i for i, f in enumerate(features)}
        for _, row in selected.iterrows():
            j = feature_index.get(str(row["feature"]))
            if j is None:
                continue
            values = [X[labels == group, j] for group in groups]
            for item in _posthoc_one(values, groups, method, posthoc):
                posthoc_rows.append({"feature": row["feature"], "name": row["name"],
                                     **item})
    posthoc_frame = pd.DataFrame(posthoc_rows)

    # 图：显著特征的组均值 ± SE 条形图
    figure = None
    try:
        show = frame.head(9)
        if not show.empty:
            n_cols = min(3, len(show))
            n_rows = int(math.ceil(len(show) / n_cols))
            fig, axes = new_fig(3.4 * n_cols + 0.8, 2.7 * n_rows + 0.6, nrows=n_rows, ncols=n_cols)
            axes = np.atleast_1d(axes).ravel()
            for ax, (_, row) in zip(axes, show.iterrows()):
                j = _feature_index_of(features, row["feature"])
                heights, errors, colors = [], [], []
                for i, group in enumerate(groups):
                    sub = X[labels == group, j] if j is not None else np.array([])
                    sub = sub[np.isfinite(sub)]
                    heights.append(float(np.mean(sub)) if sub.size else 0.0)
                    errors.append(float(np.std(sub, ddof=1) / math.sqrt(sub.size)) if sub.size > 1 else 0.0)
                    colors.append(PALETTE[i % len(PALETTE)])
                ax.bar(np.arange(len(groups)), heights, yerr=errors, color=colors,
                       capsize=3, width=0.68, zorder=3)
                ax.set_xticks(np.arange(len(groups)))
                ax.set_xticklabels([_short(g, 10) for g in groups], fontsize=7, rotation=45)
                ax.set_ylabel("mean +/- SE", fontsize=8)
                ax.set_title(_short(str(row["name"] or row["feature"]), 22), fontsize=8)
                if row["p_adj"] is not None:
                    ax.text(0.98, 0.96, f"FDR={float(row['p_adj']):.3g}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=7, color="#5a6068")
            for ax in axes[len(show):]:
                ax.axis("off")
            fig.tight_layout()
            figure = _save(fig, outdir, "multi-group-de", params, n_features=int(len(show)))
    except Exception as error:  # noqa: BLE001 - 图失败不该阻断分析
        warnings.append(f"多组均值图未生成：{error}")

    tables = [
        table("multi_group_de", f"多组比较（{method}{'' if posthoc == 'none' else ' + ' + posthoc}）",
              columns=[{"key": "feature", "label": "特征", "type": "string"},
                       {"key": "name", "label": "名称", "type": "string"},
                       {"key": "mz", "label": "m/z", "type": "number"},
                       {"key": "rt", "label": "RT (min)", "type": "number"},
                       {"key": "statistic", "label": "统计量", "type": "number"},
                       {"key": "df1", "label": "df1", "type": "number"},
                       {"key": "df2", "label": "df2", "type": "number"},
                       {"key": "p", "label": "P 值", "type": "number"},
                       {"key": "p_adj", "label": "FDR", "type": "number"},
                       {"key": "max_group", "label": "均值最高组", "type": "string"},
                       {"key": "min_group", "label": "均值最低组", "type": "string"},
                       {"key": "max_diff", "label": "最大组间差", "type": "number"}],
              rows=frame, note=f"共 {len(frame)} 个特征；FDR≤{fdr:g} 的显著特征 {n_sig} 个"),
    ]
    if not posthoc_frame.empty:
        tables.append(table("multi_group_posthoc", f"事后两两比较（{posthoc}）",
                            columns=[{"key": "feature", "label": "特征", "type": "string"},
                                     {"key": "name", "label": "名称", "type": "string"},
                                     {"key": "group_a", "label": "组 A", "type": "string"},
                                     {"key": "group_b", "label": "组 B", "type": "string"},
                                     {"key": "diff", "label": "均值差 (A-B)", "type": "number"},
                                     {"key": "p", "label": "P 值", "type": "number"},
                                     {"key": "p_adj", "label": "校正 P", "type": "number"}],
                            rows=posthoc_frame))
    bullets = [
        f"{len(groups)} 组（{'、'.join(groups)}）用 {method} 比较，FDR≤{fdr:g} 的显著特征 {n_sig} 个",
        f"分组列「{key}」；最大组间差最大的特征："
        + ("、".join(map(str, frame["feature"].head(3))) if len(frame) else "无"),
    ]
    if not posthoc_frame.empty:
        bullets.append(f"事后检验（{posthoc}）共 {len(posthoc_frame)} 个两两比较，"
                       f"其中校正后 P<0.05 的 {int((posthoc_frame['p_adj'] < 0.05).sum())} 个")
    return result(
        title="多组 ANOVA",
        bullets=bullets,
        metrics={"n_groups": len(groups), "method": method, "posthoc": posthoc,
                 "n_features": int(len(frame)), "n_sig": n_sig, "fdr": fdr,
                 "n_posthoc": int(len(posthoc_frame))},
        tables=tables,
        figures=[figure] if figure else [],
        warnings=warnings,
        notes=[f"用时 {timer.ms} ms；ANOVA 为等方差 F 检验，Welch ANOVA 用 Welch-Satterthwaite 近似，"
               f"Kruskal-Wallis 为秩检验；事后检验已做多重比较校正"],
    )


def _feature_index_of(features: list[str], feature) -> int | None:
    try:
        return features.index(str(feature))
    except ValueError:
        return None


def analysis_trend_course(ctx, params=None, outdir=None):
    """时间/剂量趋势：分箱均值 ± SE + 线性/Spearman 趋势（Fiehn 2008 代谢轨迹语义）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    payload = _trend_payload(ctx, params)
    frame = payload["frame"]
    figure = _render_trend(payload, params, outdir)
    n_up = int((frame["trend"] == "increasing").sum())
    n_down = int((frame["trend"] == "decreasing").sum())
    return result(
        title="时间/剂量趋势",
        bullets=[f"时间列「{payload['time_col']}」，{int(len(payload['x']))} 个样本参与分箱；"
                 f"检验 {len(frame)} 个特征",
                 f"显著上升 {n_up} 个、显著下降 {n_down} 个（线性回归 P≤0.05）",
                 f"趋势最强的特征：{'、'.join(map(str, frame['feature'].head(3)))}"],
        metrics={"time_col": payload["time_col"], "n_samples": int(len(payload["x"])),
                 "n_features": int(len(frame)), "n_increasing": n_up, "n_decreasing": n_down,
                 "n_timepoints": int(len(np.unique(payload["x"])))},
        tables=[table("trend", "特征时间趋势",
                      columns=[{"key": "feature", "label": "特征", "type": "string"},
                               {"key": "name", "label": "名称", "type": "string"},
                               {"key": "mz", "label": "m/z", "type": "number"},
                               {"key": "rt", "label": "RT (min)", "type": "number"},
                               {"key": "n", "label": "样本数", "type": "number"},
                               {"key": "slope", "label": "线性斜率", "type": "number"},
                               {"key": "p_slope", "label": "斜率 P", "type": "number"},
                               {"key": "rho_spearman", "label": "Spearman ρ", "type": "number"},
                               {"key": "p_spearman", "label": "Spearman P", "type": "number"},
                               {"key": "trend", "label": "趋势", "type": "string"},
                               {"key": "span_change", "label": "首末箱变化", "type": "number"}],
                      rows=frame)],
        figures=[figure],
        notes=[f"用时 {timer.ms} ms；分箱：时间点 ≤8 个时按原值分组，否则按分位数分 6 箱"],
    )


def analysis_dose_response(ctx, params=None, outdir=None):
    """剂量响应（EC50）：4 参数 logistic 拟合（内置实现；drc/drc 语义）。"""
    timer = Timer()
    params = _pdict(params)
    outdir = outdir or ctx.get("outdir") or os.getcwd()
    payload = _dose_payload(ctx, params)
    fit = payload["fit"]
    figure = _render_dose(payload, params, outdir)
    ec50_text = f"{fit['ec50']:.4g}"
    bullets = [
        f"特征「{payload['feature']}」在剂量列「{payload['dose_col']}」上的 4PL 拟合："
        f"EC50={ec50_text}，Hill 斜率={fit['hill']:.3f}，R²={fit['r2']:.3f}",
        f"渐近线：bottom={fit['bottom']:.4g}，top={fit['top']:.4g}；"
        f"响应方向 {fit['direction']}",
        f"共 {fit['n']} 个有效剂量点（剂量 > 0 且响应非缺失）",
    ]
    return result(
        title="剂量响应（EC50）",
        bullets=bullets,
        metrics={"feature": payload["feature"], "dose_col": payload["dose_col"],
                 "ec50": float(fit["ec50"]), "hill": float(fit["hill"]), "r2": float(fit["r2"]),
                 "bottom": float(fit["bottom"]), "top": float(fit["top"]),
                 "n_points": int(fit["n"]), "ci": payload["ci"]},
        tables=[table("dose_response", "4PL 参数估计",
                      columns=[{"key": "parameter", "label": "参数", "type": "string"},
                               {"key": "estimate", "label": "估计值", "type": "number"},
                               {"key": "ci_low", "label": "CI 下限", "type": "number"},
                               {"key": "ci_high", "label": "CI 上限", "type": "number"}],
                      rows=payload["rows"],
                      note=f"CI 为 {payload['ci']:.0%} 近似区间（协方差矩阵一阶传播）；"
                           f"EC50 = 10^log(EC50)，与曲线拐点一致")],
        figures=[figure],
        warnings=list(payload["warnings"]),
        notes=[f"用时 {timer.ms} ms；4PL 用 scipy curve_fit 在 log10(剂量) 上参数化"],
    )


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

ANALYSES = [
    {
        "id": "univariate_de",
        "title": "单变量差异",
        "group": "差异与趋势",
        "method": "Welch t / Mann-Whitney / limma 经验贝叶斯 + BH（Xia 2015 MetaboAnalyst；内置实现）",
        "needs": ["groups"],
        "params": [
            {"key": "method", "label": "方法", "type": "select", "default": "welch",
             "options": [{"value": "welch", "label": "Welch t 检验"},
                         {"value": "wilcoxon", "label": "Mann-Whitney U"},
                         {"value": "limma", "label": "limma 经验贝叶斯（内置近似）"}]},
            {"key": "paired", "label": "配对", "type": "bool", "default": False},
            {"key": "pair_col", "label": "配对列（可选）", "type": "text", "default": None},
            {"key": "fdr", "label": "FDR 阈值", "type": "number", "default": 0.05},
            {"key": "fc", "label": "倍数阈值", "type": "number", "default": 1.5},
            {"key": "group_col", "label": "分组列", "type": "text", "default": None},
            {"key": "case", "label": "实验组", "type": "text", "default": None},
            {"key": "control", "label": "对照组", "type": "text", "default": None},
        ],
        "run": analysis_univariate_de,
    },
    {
        "id": "volcano_de",
        "title": "火山图",
        "group": "差异与趋势",
        "method": "log2FC vs −log10 P（火山图语义；内置实现）",
        "needs": ["de"],
        "params": [
            {"key": "n_labels", "label": "标注数", "type": "number", "default": 15},
            {"key": "fc", "label": "倍数阈值", "type": "number", "default": 1.5},
            {"key": "fdr", "label": "FDR 阈值", "type": "number", "default": 0.05},
        ],
        "run": analysis_volcano_de,
    },
    {
        "id": "fc_bar",
        "title": "倍数变化条形图",
        "group": "差异与趋势",
        "method": "top/bottom |log2FC|（内置实现）",
        "needs": ["de"],
        "params": [
            {"key": "top_n", "label": "Top N", "type": "number", "default": 20},
            {"key": "fc", "label": "倍数阈值", "type": "number", "default": 1.5},
            {"key": "fdr", "label": "FDR 阈值", "type": "number", "default": 0.05},
        ],
        "run": analysis_fc_bar,
    },
    {
        "id": "heatmap_top",
        "title": "差异特征热图",
        "group": "差异与趋势",
        "method": "行 z-score + 层次聚类（内置实现）",
        "needs": ["de"],
        "params": [
            {"key": "top_n", "label": "Top N", "type": "number", "default": 50},
            {"key": "cluster", "label": "聚类", "type": "bool", "default": True},
            {"key": "fc", "label": "倍数阈值", "type": "number", "default": 1.5},
            {"key": "fdr", "label": "FDR 阈值", "type": "number", "default": 0.05},
        ],
        "run": analysis_heatmap_top,
    },
    {
        "id": "multi_group_de",
        "title": "多组 ANOVA",
        "group": "差异与趋势",
        "method": "ANOVA / Welch ANOVA / Kruskal-Wallis + 事后检验（Tukey / Games-Howell / Dunn）",
        "needs": ["groups"],
        "params": [
            {"key": "method", "label": "方法", "type": "select", "default": "anova",
             "options": [{"value": "anova", "label": "单因素 ANOVA"},
                         {"value": "welch", "label": "Welch ANOVA"},
                         {"value": "kruskal", "label": "Kruskal-Wallis"}]},
            {"key": "posthoc", "label": "事后检验", "type": "select", "default": "tukey",
             "options": [{"value": "tukey", "label": "Tukey-Kramer"},
                         {"value": "games_howell", "label": "Games-Howell"},
                         {"value": "dunn", "label": "Dunn"},
                         {"value": "none", "label": "不做事后检验"}]},
            {"key": "fdr", "label": "FDR 阈值", "type": "number", "default": 0.05},
            {"key": "group_col", "label": "分组列", "type": "text", "default": None},
        ],
        "run": analysis_multi_group_de,
    },
    {
        "id": "trend_course",
        "title": "时间/剂量趋势",
        "group": "差异与趋势",
        "method": "分箱均值 ± SE + 线性/Spearman 趋势（Fiehn 2008 代谢轨迹；内置实现）",
        "needs": ["groups"],
        "params": [
            {"key": "time_col", "label": "时间/剂量列", "type": "text", "default": None},
            {"key": "features", "label": "特征列表（留空=自动选）", "type": "text", "default": []},
            {"key": "top_n", "label": "最多特征数", "type": "number", "default": 8},
        ],
        "run": analysis_trend_course,
    },
    {
        "id": "dose_response",
        "title": "剂量响应（EC50）",
        "group": "差异与趋势",
        "method": "4 参数 logistic 拟合（内置实现；R drc 语义）",
        "needs": ["groups"],
        "params": [
            {"key": "dose_col", "label": "剂量列", "type": "text", "default": None},
            {"key": "feature", "label": "特征（留空=自动选）", "type": "text", "default": ""},
            {"key": "ci", "label": "置信区间", "type": "number", "default": 0.95},
        ],
        "run": analysis_dose_response,
    },
]

STEPS = [
    {
        "id": "de",
        "title": "单变量差异",
        "requires": ["transform"],
        "params": STEP_PARAMS,
        "run": step_de,
    },
]

__all__ = [
    "STEPS", "ANALYSES", "PLOTS",
    "step_de", "analysis_univariate_de", "analysis_volcano_de", "analysis_fc_bar",
    "analysis_heatmap_top", "analysis_multi_group_de", "analysis_trend_course",
    "analysis_dose_response", "plot_volcano_de", "plot_fc_bar", "plot_heatmap_top",
    "plot_trend_plot", "plot_dose_response",
    "_de_compute", "_fit_4pl", "_limma", "_welch_core", "_fit_fdist",
]
