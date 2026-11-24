"""
dsh-metabolomics — 数据集状态层（DESIGN §2.3 / §2.4 / §9）

职责：
  * Dataset：一个 LC-MS/GC-MS 代谢组数据集的全部内存状态
      - 强度矩阵 X / raw（(n_samples, n_features) float32，缺失一律 np.nan）
      - 特征注释 feature_info（index=features；mz/rt/name/formula/adduct/hmdb/kegg/class/polarity/confidence）
      - 样本表 sample_info（index=samples；group/batch/qc/age/bmi/os_time/os_event 等）
      - 步骤状态 steps/current、history、analyses_run、results（跨分析缓存）、library（MS2 谱库）
  * State：多数据集注册表（new/get/close/reset/find_by_path/summaries，id 规则 ds1/ds2/…）
  * build_feature_info() / build_sample_info()：导入层与 demo 共用的表构造
  * make_demo()：DESIGN §9 内置示例（72 样本 × 1600 特征 × 120 真差异 × 5 通路 × 40 条 MS2 谱图）

约定：
  * X / raw 一律 float32、形状 (n_samples, n_features)：行=样本、列=特征（DESIGN §2.3 matrix_order
    归一化后统一为 "samples_by_features"）；
  * 缺失值一律 np.nan（不补 0、不补均值）；本文件不改 X 的数值，只负责承载与记录；
  * feature_info / sample_info 的 index 必须分别等于 features / samples（顺序一致）；
  * 只依赖标准库 + numpy/pandas + mb_common（不 import 其它 mb_* 模块，避免循环依赖）。
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mb_common import (
    ADDUCTS,
    ELEMENT_MASS,
    EngineError,
    formula_str,
    jsonable,
    neutral_mass,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 12 步流程（顺序即左栏卡片顺序；每项 {"id","title","requires"}，与 DESIGN §3 / registry 一致）
STEP_ORDER = [
    {"id": "qc", "title": "样本质控", "requires": []},
    {"id": "filter", "title": "特征过滤", "requires": ["qc"]},
    {"id": "impute", "title": "缺失值填补", "requires": ["filter"]},
    {"id": "normalize", "title": "归一化", "requires": ["impute"]},
    {"id": "transform", "title": "变换/标度", "requires": ["normalize"]},
    {"id": "pca", "title": "主成分分析", "requires": ["transform"]},
    {"id": "plsda", "title": "PLS-DA / OPLS-DA", "requires": ["transform"]},
    {"id": "de", "title": "单变量差异", "requires": ["transform"]},
    {"id": "batch", "title": "批次校正", "requires": ["normalize"]},
    {"id": "enrich", "title": "通路富集", "requires": ["de"]},
    {"id": "biomarker", "title": "标志物筛选", "requires": ["de"]},
    {"id": "annotate", "title": "代谢物鉴定", "requires": ["transform"]},
]

STEP_IDS = [item["id"] for item in STEP_ORDER]
STEP_INDEX = {item["id"]: index for index, item in enumerate(STEP_ORDER)}

#: 矩阵值类型（DESIGN §2.3 X_kind）：原始强度 / 对数归一化 / 标准化
X_KINDS = ("raw", "lognorm", "scaled")

#: feature_info 的标准列（DESIGN §2.3；缺的列一律补 NaN，多出来的列保留）
FEATURE_COLUMNS = (
    "mz", "rt", "name", "formula", "adduct", "hmdb", "kegg", "class", "polarity", "confidence",
)

#: 样本信息里的时间/事件列名关键字（column_kind 用；与 bg_state/registry 保持同一套）
TIME_PATTERNS = (
    "time", "times", "os_time", "os.time", "dfs_time", "rfs_time", "pfs_time", "survival_time",
    "month", "months", "days", "follow", "duration", "timepoint", "time_point",
)
EVENT_PATTERNS = ("event", "events", "status", "dead", "death", "censor", "vital", "outcome")

#: history / analyses_run 上限（DESIGN §2.4：内存最多 50/100 条，摘要返回截断到 20 条）
MAX_HISTORY = 50
MAX_ANALYSES_RUN = 100
SUMMARY_HISTORY_LIMIT = 20

#: demo 规模（DESIGN §9）
DEMO_SEED = 0
DEMO_N_SAMPLES = 72
DEMO_N_CASE = 30
DEMO_N_CONTROL = 30
DEMO_N_QC = 12
DEMO_N_FEATURES = 1600
DEMO_N_BATCHES = 2
DEMO_N_DIFF = 120
DEMO_N_ANNOTATED = 336
DEMO_N_PATHWAYS = 12
DEMO_N_ACTIVE_PATHWAYS = 5
DEMO_PATHWAY_SIZE = 28
DEMO_DIFF_PER_PATHWAY = 24
DEMO_N_SPECTRA = 40
DEMO_N_MATCHED_SPECTRA = 20
DEMO_N_RISK_FEATURES = 60
DEMO_N_BATCH_FEATURES = 240

#: demo 化学/信号参数
DEMO_MZ_MIN = 100.0
DEMO_MZ_MAX = 900.0
DEMO_RT_MIN = 0.3
DEMO_RT_MAX = 18.0
DEMO_NEUTRAL_MASS_MIN = 105.0
DEMO_NEUTRAL_MASS_MAX = 885.0
DEMO_MZ_TOL_PPM = 3.0          # 反解分子式与 2 位小数 m/z 的理论质量误差上限
DEMO_LOG_ABUNDANCE_MEAN = 11.2  # 中位强度 ≈ e^11.2 ≈ 7.3e4（LC-MS 峰面积的常见量级）
DEMO_LOG_ABUNDANCE_SD = 1.25
DEMO_ENTRY_NOISE = 0.12
DEMO_BIOLOGICAL_SD = 0.22
DEMO_QC_SD = 0.05
DEMO_BATCH_LOG_SHIFT = 0.30
DEMO_MISSING_BASE = 0.02
DEMO_MISSING_SLOPE = 0.13
DEMO_N_POOR_FEATURES = 24  # 低丰度/高缺失特征：给「特征过滤」留出可剔除的对象
DEMO_RISK_LOADING = 0.50
DEMO_HAZARD_BASE = 0.018
DEMO_HAZARD_BETA = 0.95
DEMO_CENSOR_TIME = 60.0

#: 8 类化学分类（HMDB class 语义；DESIGN §9）
DEMO_CLASS_NAMES = (
    "Amino acid", "Lipid", "Bile acid", "Carbohydrate",
    "Nucleotide", "Organic acid", "Amine", "Other",
)
DEMO_CLASS_WEIGHTS = (0.18, 0.20, 0.05, 0.10, 0.10, 0.15, 0.10, 0.12)

#: 12 条内置 KEGG 通路（通路名, 归属化学分类）；前 5 条是差异特征集中的"活跃通路"
DEMO_PATHWAYS = (
    ("Aminoacyl-tRNA biosynthesis", "Amino acid"),
    ("Bile acid biosynthesis", "Bile acid"),
    ("Purine metabolism", "Nucleotide"),
    ("Pyrimidine metabolism", "Nucleotide"),
    ("Glycolysis / Gluconeogenesis", "Carbohydrate"),
    ("Citrate cycle (TCA cycle)", "Organic acid"),
    ("Fatty acid metabolism", "Lipid"),
    ("Glycerophospholipid metabolism", "Lipid"),
    ("Sphingolipid metabolism", "Lipid"),
    ("Arginine and proline metabolism", "Amino acid"),
    ("Tryptophan metabolism", "Amine"),
    ("Steroid hormone biosynthesis", "Other"),
)

#: demo 的加合物混合比例（正离子为主，负离子占三成；adduct 列同时体现 pos/neg）
DEMO_POSITIVE_ADDUCT_MIX = (("[M+H]+", 0.70), ("[M+Na]+", 0.16), ("[M+NH4]+", 0.09), ("[M+K]+", 0.05))
DEMO_NEGATIVE_ADDUCT_MIX = (("[M-H]-", 0.72), ("[M+HCOO]-", 0.18), ("[M+Cl]-", 0.10))
DEMO_POSITIVE_FRACTION = 0.70

#: 样本信息里优先当分组列的列名 / 不能当分组列的列名（与 registry.guess_group 保持同向）
_GROUP_COLUMN_PREFERENCE = (
    "group", "groups", "Group", "condition", "Condition", "treatment", "class", "Class",
    "type", "Type", "phenotype", "disease", "Disease", "status", "arm", "cohort", "label",
    "sample_group", "category",
)
_GROUP_COLUMN_AVOID = (
    "batch", "plate", "run", "qc", "sample", "id", "time", "age", "bmi", "dose", "event",
    "sex", "gender", "patient", "subject",
)

#: 特征 id 里内嵌的 mz/rt（mb_io 的命名规则 mz{:.4f}_rt{:.2f}），用于缺注释时自动回填
_FEATURE_ID_PATTERN = re.compile(
    r"mz[_\-\s]*([0-9]+(?:\.[0-9]+)?)[_\-]?rt[_\-\s]*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

#: CHNO 分子式反解用的候选网格（C 4–45 / N 0–4 / O 0–12，共 2730 个组合，模块加载时算一次）
_FORMULA_C = np.repeat(np.arange(4, 46, dtype=np.int32), 5 * 13)
_FORMULA_N = np.tile(np.repeat(np.arange(0, 5, dtype=np.int32), 13), 42)
_FORMULA_O = np.tile(np.arange(0, 13, dtype=np.int32), 42 * 5)
_FORMULA_H_MASS = float(ELEMENT_MASS["H"])
_FORMULA_OFFSET = (
    float(ELEMENT_MASS["C"]) * _FORMULA_C
    + float(ELEMENT_MASS["N"]) * _FORMULA_N
    + float(ELEMENT_MASS["O"]) * _FORMULA_O
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """UTC ISO8601（毫秒）时间戳，供 history/analyses_run 使用。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _dedupe(names, prefix: str = "feature") -> list[str]:
    """去空/去引号/同名加后缀（保持顺序），保证 features/samples 唯一。"""
    seen: dict[str, int] = {}
    out: list[str] = []
    for index, raw in enumerate(names or []):
        name = str(raw).strip().strip('"').strip("'")
        if not name:
            name = f"{prefix}{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _float(value):
    """None / NaN / Inf 安全转 float。"""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def polarity_from_adduct(adduct) -> str | None:
    """由加合物名推断离子极性：'[M+H]+' → pos，'[M-H]-' → neg。"""
    text = str(adduct or "").strip()
    if not text:
        return None
    if text.rstrip().endswith("+"):
        return "pos"
    if text.rstrip().endswith("-"):
        return "neg"
    return None


def normalize_polarity(value) -> str | None:
    """把导入层的极性写法统一成 pos / neg / both（无法判定返回 None）。"""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.startswith("pos") or text in ("p", "esi+", "+", "positive"):
        return "pos"
    if text.startswith("neg") or text in ("n", "esi-", "-", "negative"):
        return "neg"
    if text in ("both", "mixed", "pos+neg", "pos_neg", "dual"):
        return "both"
    return text


def _present_mask(series: pd.Series) -> np.ndarray:
    """列里"真正有值"的位置：非 NA 且不是空串/whitespace（注释覆盖率统计用）。"""
    mask = series.notna().to_numpy()
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        try:
            texts = series.astype(str).str.strip().to_numpy()
            mask &= texts != ""
            mask &= texts != "nan"
            mask &= texts != "None"
        except Exception:  # noqa: BLE001 - 统计用途，失败就按 notna 算
            pass
    return mask


# ---------------------------------------------------------------------------
# 列种类（DESIGN §2.4 sample_columns / §2.2 columns）
# ---------------------------------------------------------------------------

def column_kind(series) -> str:
    """判断样本信息列的种类：categorical / numeric / time / event（UI 展示与临床识别用）。

    规则（先名字后数值，避免 os_event 这种 0/1 数值列被当成分组）：
      * 名字含 event/status/dead/censor/… 且是 0/1（或数值）→ event；
      * 名字含 time/month/days/follow/… 且是数值 → time；
      * 数值且唯一值 > 12 → numeric；
      * 其余（含文本、布尔、≤12 个取值的小整数）→ categorical。
    """
    try:
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
    except Exception:  # noqa: BLE001
        return "categorical"
    key = str(getattr(series, "name", "") or "").lower()
    if series.dropna().empty:
        return "categorical"
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    try:
        texts = series.dropna().astype(str).str.strip().str.lower()
        binary = set(texts.unique()) <= {"0", "1", "0.0", "1.0", "true", "false", "yes", "no"}
    except Exception:  # noqa: BLE001
        binary = False
    if binary and any(pattern in key for pattern in EVENT_PATTERNS):
        return "event"
    if is_numeric and any(pattern in key for pattern in EVENT_PATTERNS):
        return "event"
    if is_numeric and any(pattern in key for pattern in TIME_PATTERNS):
        return "time"
    if is_numeric and int(series.nunique(dropna=True)) > 12:
        return "numeric"
    return "categorical"


# ---------------------------------------------------------------------------
# 表构造（导入层 / demo 共用）
# ---------------------------------------------------------------------------

def _align_frame(frame: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame:
    """把外部 DataFrame 对齐到 index（先按索引名，失败再按位置），列名统一成 str。"""
    out = pd.DataFrame(index=index)
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        return out
    src = frame.copy()
    src.columns = [str(column) for column in src.columns]
    src.index = [str(item) for item in src.index]
    try:
        aligned = src.reindex(index)
        if aligned.isna().all(axis=None) and len(src) == len(index):
            # 索引名完全不同（导入层常见）：按位置对齐
            aligned = src.copy()
            aligned.index = index
        out = out.join(aligned)
    except Exception:  # noqa: BLE001 - 对不上就当没有样本/特征信息
        return pd.DataFrame(index=index)
    return out


def build_feature_info(features, extra=None) -> pd.DataFrame:
    """把导入层/demo 给的特征注释整理成规整 DataFrame（index=features，按 features 顺序对齐）。

    * extra 可以是 DataFrame（按索引对齐）或 dict（按列名 + 与 features 等长的序列）；
    * 统一补齐 DESIGN §2.3 的标准列（mz/rt/name/formula/adduct/hmdb/kegg/class/polarity/confidence），
      缺的列填 NaN——下游模块可以无脑取列；
    * 若特征 id 是 mb_io 的 "mz203.0526_rt1.23" 形式且 mz/rt 缺失，自动回填（保证鉴定类分析可用）；
    * 若 adduct 有值而 polarity 缺失，按加合物推断极性。
    """
    names = [str(item) for item in (features if features is not None else [])]
    index = pd.Index(names, name="feature")
    frame = pd.DataFrame(index=index)
    if isinstance(extra, pd.DataFrame):
        frame = _align_frame(extra, index)
    elif isinstance(extra, dict):
        for key, values in extra.items():
            if isinstance(values, pd.Series):
                values = values.reindex(index) if values.index.equals(index) else values.tolist()
            if isinstance(values, (list, tuple, np.ndarray)) and len(values) == len(index):
                frame[str(key)] = list(values)
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    _backfill_mz_rt(frame)
    if "polarity" in frame.columns and "adduct" in frame.columns:
        try:
            missing = frame["polarity"].isna().to_numpy()
            if missing.any():
                inferred = frame["adduct"].map(polarity_from_adduct).to_numpy(dtype=object)
                values = frame["polarity"].astype(object).to_numpy()
                values[missing] = inferred[missing]
                frame["polarity"] = pd.Series(values, index=frame.index, dtype=object)
        except Exception:  # noqa: BLE001
            pass
    ordered = ([column for column in FEATURE_COLUMNS if column in frame.columns]
               + [column for column in frame.columns if column not in FEATURE_COLUMNS])
    frame = frame.loc[:, ordered]
    frame.index = index
    return frame


def _backfill_mz_rt(frame: pd.DataFrame) -> None:
    """特征 id 形如 mz203.0526_rt1.23 时，回填缺失的 mz/rt（原地修改）。"""
    if frame.empty or "mz" not in frame.columns or "rt" not in frame.columns:
        return
    mz_missing = frame["mz"].isna()
    rt_missing = frame["rt"].isna()
    if not (mz_missing.any() or rt_missing.any()):
        return
    mz_values: list[float] = []
    rt_values: list[float] = []
    hits = 0
    for name in frame.index.astype(str):
        match = _FEATURE_ID_PATTERN.search(name)
        if match is None:
            mz_values.append(np.nan)
            rt_values.append(np.nan)
            continue
        hits += 1
        mz_values.append(float(match.group(1)))
        rt_values.append(float(match.group(2)))
    if not hits:
        return
    mz_array = np.asarray(mz_values, dtype=float)
    rt_array = np.asarray(rt_values, dtype=float)
    if mz_missing.any():
        values = pd.to_numeric(frame["mz"], errors="coerce").to_numpy(dtype=float)
        values[mz_missing.to_numpy()] = mz_array[mz_missing.to_numpy()]
        frame["mz"] = values
    if rt_missing.any():
        values = pd.to_numeric(frame["rt"], errors="coerce").to_numpy(dtype=float)
        values[rt_missing.to_numpy()] = rt_array[rt_missing.to_numpy()]
        frame["rt"] = values


def build_sample_info(samples, info=None, extra=None) -> pd.DataFrame:
    """把导入层/demo 给的样本信息整理成规整 DataFrame（index=samples，按 samples 顺序对齐）。

    * info 可以是 DataFrame（按索引对齐）或 dict（列名 → 与 samples 等长的序列）；
    * extra 是补充列（dict，已存在的列不覆盖）；对齐失败时退化为"只有索引"的空表；
    * index 一律 name="sample"，索引值统一成 str——下游按样本名 join 分组/临床表时更稳。
    """
    names = [str(item) for item in (samples if samples is not None else [])]
    index = pd.Index(names, name="sample")
    frame = pd.DataFrame(index=index)
    if isinstance(info, pd.DataFrame):
        frame = _align_frame(info, index)
    elif isinstance(info, dict):
        for key, values in info.items():
            if isinstance(values, pd.Series):
                values = values.reindex(index) if values.index.equals(index) else values.tolist()
            if isinstance(values, (list, tuple, np.ndarray, pd.Index)) and len(values) == len(index):
                frame[str(key)] = list(values)
    if extra:
        for key, values in extra.items():
            key = str(key)
            if key in frame.columns:
                continue
            if isinstance(values, pd.Series):
                values = values.reindex(index) if values.index.equals(index) else values.tolist()
            if isinstance(values, (list, tuple, np.ndarray, pd.Index)) and len(values) == len(index):
                frame[key] = list(values)
    frame.index = index
    return frame


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset:
    """一个代谢组数据集：强度矩阵 + 特征注释 + 样本表 + 流程状态。

    属性（DESIGN §2.3 / §2.4）：
      id, name, path, source_format, X, raw, X_kind, features, samples,
      feature_info(pd.DataFrame, index=features), sample_info(pd.DataFrame, index=samples),
      group_key, groups, species, polarity, results, library,
      steps, current, history, analyses_run, notes, warnings
    """

    def __init__(
        self,
        ds_id: str | None = None,
        *,
        name: str = "metabo",
        X,
        raw=None,
        X_kind: str | None = None,
        features=None,
        samples=None,
        feature_info: pd.DataFrame | None = None,
        sample_info: pd.DataFrame | None = None,
        group_key: str | None = None,
        species: str = "unknown",
        polarity: str | None = None,
        path: str | None = None,
        source_path: str | None = None,
        source_format: str = "unknown",
        matrix_order: str = "samples_by_features",
        library=None,
        notes=None,
        warnings=None,
        genes=None,
        value_kind: str | None = None,
        **extra,
    ):
        if X is None:
            raise EngineError("bad_request", "缺少强度矩阵（X）：请先导入特征表或载入内置示例（demo）")
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise EngineError("bad_request", "强度矩阵必须是二维（样本 × 特征）")
        n_samples, n_features = int(arr.shape[0]), int(arr.shape[1])
        if n_samples == 0 or n_features == 0:
            raise EngineError("bad_request", "强度矩阵为空（0 个样本或 0 个特征）")

        self.id = str(ds_id or "")
        self.name = str(name or "metabo")
        self.path = None if path is None and source_path is None else str(path or source_path)
        self.source_format = str(source_format or "unknown")
        self.X = arr
        self.raw = self._as_intensities(raw, "raw")

        feature_list = features if features is not None else genes
        self.features = _dedupe(
            feature_list if feature_list is not None else [f"F{i + 1:04d}" for i in range(n_features)],
            "feature",
        )
        if len(self.features) != n_features:
            raise EngineError(
                "bad_request",
                f"特征数（{len(self.features)}）与矩阵列数（{n_features}）不一致",
            )
        self.samples = _dedupe(
            samples if samples is not None else [f"sample{i + 1}" for i in range(n_samples)],
            "sample",
        )
        if len(self.samples) != n_samples:
            raise EngineError(
                "bad_request",
                f"样本数（{len(self.samples)}）与矩阵行数（{n_samples}）不一致",
            )

        self.X_kind = self._resolve_x_kind(X_kind if X_kind is not None else value_kind)
        self.matrix_order = str(matrix_order or "samples_by_features")
        self.feature_info = build_feature_info(self.features, feature_info)
        self.sample_info = build_sample_info(self.samples, sample_info)

        self.species = str(species or "unknown").strip().lower() or "unknown"
        self.polarity = normalize_polarity(polarity)
        self.library = self._normalize_library(library)

        self.results: dict = {}
        self.steps: list[str] = []
        self.current: str | None = None
        self.history: list[dict] = []
        self.analyses_run: list[dict] = []
        self.notes = [str(item) for item in (notes or []) if str(item).strip()]
        self.warnings = [str(item) for item in (warnings or []) if str(item).strip()]
        self.created_at = now_iso()
        self._group_levels: list[str] = []
        self.group_key: str | None = None

        if self.matrix_order != "samples_by_features":
            self.warnings.append(
                f"导入层给出的矩阵方向为 {self.matrix_order}；本工作台统一按样本×特征存放，"
                "请确认导入结果已转置"
            )
            self.matrix_order = "samples_by_features"

        key = str(group_key or "").strip()
        if key:
            if key in self.sample_info.columns:
                self.set_group(key)
            else:
                self.warnings.append(f"指定的分组列「{key}」不在样本信息里，已改为自动识别")
                guessed = self._guess_group()
                if guessed:
                    self.group_key = guessed
        else:
            guessed = self._guess_group()
            if guessed:
                self.group_key = guessed

        if extra:
            known = "、".join(sorted(str(item) for item in extra))[:200]
            self.notes.append(f"忽略未识别的数据集参数：{known}")

    # -- 构造期辅助 -------------------------------------------------------
    def _as_intensities(self, values, label: str):
        """把 raw 之类的附加强度矩阵转成 float32 并校验形状。"""
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2:
            raise EngineError("bad_request", f"{label} 必须是二维（样本 × 特征）")
        if arr.shape != self.X.shape:
            raise EngineError(
                "bad_request",
                f"{label} 形状 {arr.shape} 与强度矩阵形状 {self.X.shape} 不一致",
            )
        return arr

    @staticmethod
    def _resolve_x_kind(kind: str | None) -> str:
        """矩阵值类型归一（DESIGN §2.3：raw / lognorm / scaled；未知写法保留小写原值）。"""
        key = str(kind or "").strip().lower()
        if not key or key in ("auto", "unknown", "none"):
            return "raw"
        if key in ("raw", "intensity", "intensities", "area", "peak_area", "counts", "count"):
            return "raw"
        if key in ("lognorm", "log", "log2", "lognormalized", "normalized", "norm"):
            return "lognorm"
        if key in ("scaled", "scale", "zscore", "z_score", "autoscaled", "pareto", "uv"):
            return "scaled"
        return key

    @staticmethod
    def _normalize_library(library) -> dict | None:
        """MS2 谱库：{"spectra":[{"id","precursor_mz","peaks":[[mz,rel_int],...]}],"path":None}。"""
        if not library:
            return None
        if isinstance(library, (list, tuple)):
            library = {"spectra": list(library), "path": None}
        if not isinstance(library, dict):
            return None
        spectra = []
        for index, item in enumerate(library.get("spectra") or []):
            if not isinstance(item, dict):
                continue
            peaks = item.get("peaks") or []
            spectra.append({
                "id": str(item.get("id") or f"MS2-{index + 1:03d}"),
                "precursor_mz": _float(item.get("precursor_mz")),
                "peaks": peaks,
                "feature": item.get("feature"),
                "name": item.get("name"),
            })
        if not spectra:
            return None
        return {"spectra": spectra, "path": library.get("path")}

    # -- 基本属性 ---------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def groups(self) -> list[str] | None:
        """当前分组列的取值列表（无分组 → None；顺序取样本表里首次出现的顺序）。"""
        key = self.group_key
        if key and isinstance(self.sample_info, pd.DataFrame) and key in self.sample_info.columns:
            try:
                values = [str(item) for item in self.sample_info[key].astype(str).tolist()]
            except Exception:  # noqa: BLE001
                values = []
            levels = [item for item in dict.fromkeys(values) if item not in ("nan", "None", "", "<NA>")]
            if levels:
                self._group_levels = levels
                return list(levels)
        return list(self._group_levels) if self._group_levels else None

    @groups.setter
    def groups(self, values) -> None:
        # registry 会直接 setattr(dataset, "groups", [...])；这里只做缓存，真值仍以 sample_info 为准
        try:
            self._group_levels = [str(item) for item in (values or [])]
        except TypeError:
            self._group_levels = []

    @property
    def group_levels(self) -> list[str]:
        """分组取值（groups 的别名，便于模块里读）。"""
        return list(self.groups or [])

    def group_labels(self) -> np.ndarray:
        """每个样本的分组标签（无分组时全为 "all"）。"""
        if not self.group_key or self.group_key not in self.sample_info.columns:
            return np.array(["all"] * self.n_samples)
        return self.sample_info[self.group_key].astype(str).values

    def _guess_group(self) -> str | None:
        """自动挑一个像分组的列（2–20 组、非批次/QC/时间/临床列）。"""
        if self.sample_info is None or self.sample_info.empty:
            return None
        scored: list[tuple[int, str]] = []
        for column in self.sample_info.columns:
            key = str(column).lower()
            try:
                n_unique = int(self.sample_info[column].nunique(dropna=True))
            except Exception:  # noqa: BLE001
                continue
            if not 1 < n_unique <= 20:
                continue
            score = 0
            if key in _GROUP_COLUMN_PREFERENCE:
                score += 10
            elif any(pattern in key for pattern in _GROUP_COLUMN_PREFERENCE):
                score += 5
            if any(pattern in key for pattern in _GROUP_COLUMN_AVOID):
                score -= 8
            if column_kind(self.sample_info[column]) != "categorical":
                score -= 4
            if score > 0:
                scored.append((score, str(column)))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], list(self.sample_info.columns).index(item[1])))
        return scored[0][1]

    def set_group(self, key: str) -> None:
        """把 sample_info 的某列设为分组列（不改变 X）。"""
        key = str(key or "").strip()
        if not key or key not in self.sample_info.columns:
            available = ", ".join(str(column) for column in self.sample_info.columns) or "无"
            raise EngineError("bad_request", f"样本信息里没有列「{key}」；可用列：{available}")
        self.group_key = key
        values = [str(item) for item in self.sample_info[key].astype(str).tolist()]
        self._group_levels = [
            item for item in dict.fromkeys(values) if item not in ("nan", "None", "", "<NA>")
        ]

    def sample_columns(self) -> list[dict]:
        """样本信息列清单（DESIGN §2.4 sample_columns）。"""
        out: list[dict] = []
        for column in self.sample_info.columns:
            series = self.sample_info[column]
            try:
                n_unique = int(series.nunique(dropna=True))
            except Exception:  # noqa: BLE001
                n_unique = 0
            out.append({"key": str(column), "kind": column_kind(series), "n_unique": n_unique})
        return out

    # -- 特征注释 ---------------------------------------------------------
    def annotation_mask(self) -> np.ndarray:
        """有数据库注释（kegg/hmdb/pathway）的特征掩码；一个都没有时退回"有名字/分子式"。"""
        info = self.feature_info
        if not isinstance(info, pd.DataFrame) or len(info) == 0:
            return np.zeros(0, dtype=bool)
        mask = np.zeros(len(info), dtype=bool)
        database_columns = [c for c in ("kegg", "hmdb", "hmdb_id", "pathway") if c in info.columns]
        for column in database_columns:
            mask |= _present_mask(info[column])
        if not mask.any():
            for column in ("name", "class", "formula"):
                if column in info.columns:
                    mask |= _present_mask(info[column])
        return mask

    def n_annotated(self) -> int:
        """有注释的特征数（DESIGN §2.4 feature_meta.n_annotated）。"""
        try:
            return int(self.annotation_mask().sum())
        except Exception:  # noqa: BLE001
            return 0

    def feature_meta(self) -> dict:
        """DESIGN §2.4 feature_meta：{has_mz, has_rt, n_annotated}。"""
        info = self.feature_info
        has_mz = bool("mz" in info.columns and len(info) and info["mz"].notna().any())
        has_rt = bool("rt" in info.columns and len(info) and info["rt"].notna().any())
        return {"has_mz": has_mz, "has_rt": has_rt, "n_annotated": self.n_annotated()}

    def feature_ids(self) -> list[str]:
        """特征 id 列表（features 的副本，避免调用方误改内部列表）。"""
        return list(self.features)

    def sample_ids(self) -> list[str]:
        """样本 id 列表（samples 的副本）。"""
        return list(self.samples)

    # -- 摘要 / 预览 ------------------------------------------------------
    def summary(self, history_limit: int = SUMMARY_HISTORY_LIMIT) -> dict:
        """DESIGN §2.4 dataset 摘要（键名逐字一致；history/analyses_run 各截断到 20 条）。"""
        limit = max(0, int(history_limit))
        out = {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "source_format": self.source_format,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "species": self.species,
            "polarity": self.polarity,
            "feature_meta": self.feature_meta(),
            "sample_columns": self.sample_columns(),
            "steps": list(self.steps),
            "current": self.current,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "history": list(self.history[-limit:]) if limit else [],
            "analyses_run": list(self.analyses_run[-limit:]) if limit else [],
        }
        return jsonable(out)

    def preview(self) -> dict:
        """DESIGN §2.2 preview（键名逐字一致；只给前 50 特征 / 前 20 样本，不返回矩阵）。"""
        meta = self.feature_meta()
        out = {
            "features": list(self.features[:50]),
            "samples": list(self.samples[:20]),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "columns": self.sample_columns(),
            "steps": list(self.steps),
            "current": self.current,
            "feature_meta": {
                "has_mz": meta["has_mz"],
                "has_rt": meta["has_rt"],
                "annotated": int(meta["n_annotated"]),
            },
        }
        return jsonable(out)

    # -- 历史 -------------------------------------------------------------
    def add_history(self, step: str, params: dict, ok: bool, ms: int, summary: str = "") -> None:
        """记录一次步骤执行（DESIGN §2.4 history 条目）。"""
        self.history.append({
            "at": now_iso(),
            "step": str(step),
            "params": jsonable(params or {}),
            "ok": bool(ok),
            "ms": int(ms),
            "summary": str(summary or "")[:400],
        })
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    def add_analysis_run(self, analysis_id: str, ok: bool, ms: int, summary: str = "") -> None:
        """记录一次分析执行（DESIGN §2.4 analyses_run 条目）。"""
        self.analyses_run.append({
            "at": now_iso(),
            "id": str(analysis_id),
            "ok": bool(ok),
            "ms": int(ms),
            "summary": str(summary or "")[:400],
        })
        if len(self.analyses_run) > MAX_ANALYSES_RUN:
            self.analyses_run = self.analyses_run[-MAX_ANALYSES_RUN:]

    # -- 步骤状态 ---------------------------------------------------------
    def mark_step(self, step: str) -> None:
        """标记某步完成：清空其下游 steps 与 results 缓存（history 保留）。"""
        step = str(step)
        if step in STEP_INDEX:
            downstream = set(STEP_IDS[STEP_INDEX[step] + 1:])
            self.steps = [item for item in self.steps if item not in downstream and item != step]
            self.steps.append(step)
            for key in list(self.results.keys()):
                if key in downstream:
                    self.results.pop(key, None)
        elif step not in self.steps:
            self.steps.append(step)
        self.current = step
        self.reorder_steps()

    def reorder_steps(self) -> None:
        """按 DESIGN §3 的顺序排列 steps（未知步骤排最后）。"""
        self.steps = ([item for item in STEP_IDS if item in self.steps]
                      + [item for item in self.steps if item not in STEP_INDEX])

    # -- 写回（registry 消费步骤/分析返回值时调用）------------------------
    def store_result(self, key: str, value) -> None:
        """写入 results[key]（跨分析缓存；例如 de/vip/enrich/annotation）。"""
        key = str(key or "").strip()
        if not key:
            return
        self.results[key] = value

    def apply_replace(self, replace: dict | None) -> None:
        """步骤 replace：替换矩阵/特征/样本/raw/值类型/谱库（形状必须自洽）。

        replace = {"X", "features", "samples", "raw", "value_kind"(或 X_kind),
                   "feature_info", "sample_info", "library"}
        """
        if not replace or not isinstance(replace, dict):
            return
        if replace.get("X") is not None:
            arr = np.asarray(replace["X"], dtype=np.float32)
            if arr.ndim != 2:
                raise EngineError("internal", "replace.X 必须是二维（样本 × 特征）")
            new_features = self.features
            new_samples = self.samples
            if replace.get("features") is not None:
                new_features = _dedupe([str(item) for item in replace["features"]], "feature")
                if len(new_features) != arr.shape[1]:
                    raise EngineError("internal", "replace.features 与矩阵列数不一致")
            if replace.get("samples") is not None:
                new_samples = _dedupe([str(item) for item in replace["samples"]], "sample")
                if len(new_samples) != arr.shape[0]:
                    raise EngineError("internal", "replace.samples 与矩阵行数不一致")
            if (len(new_features), len(new_samples)) != (arr.shape[1], arr.shape[0]):
                raise EngineError(
                    "internal",
                    f"replace 矩阵形状 {arr.shape} 与特征/样本数（{len(new_features)}, "
                    f"{len(new_samples)}）不一致",
                )
            features_changed = new_features != self.features
            samples_changed = new_samples != self.samples
            self.X = arr
            if features_changed:
                self.features = new_features
                self.feature_info = self._reindex_feature_info(new_features)
            if samples_changed:
                self.samples = new_samples
                self.sample_info = self._reindex_sample_info(new_samples)
        if "raw" in replace:
            self.raw = self._as_intensities(replace.get("raw"), "replace.raw")
        kind = replace.get("value_kind") or replace.get("X_kind")
        if kind:
            self.X_kind = self._resolve_x_kind(kind)
        if isinstance(replace.get("feature_info"), pd.DataFrame):
            self.feature_info = build_feature_info(self.features, replace["feature_info"])
        if isinstance(replace.get("sample_info"), pd.DataFrame):
            self.sample_info = build_sample_info(self.samples, replace["sample_info"])
        if "library" in replace:
            self.library = self._normalize_library(replace.get("library"))
        if replace.get("features") is not None and replace.get("X") is None:
            self.features = _dedupe([str(item) for item in replace["features"]], "feature")
            self.feature_info = self._reindex_feature_info(self.features)
        if replace.get("samples") is not None and replace.get("X") is None:
            self.samples = _dedupe([str(item) for item in replace["samples"]], "sample")
            self.sample_info = self._reindex_sample_info(self.samples)

    def _reindex_feature_info(self, features: list[str]) -> pd.DataFrame:
        """按新的 features 顺序重组 feature_info（保留仍存在的行，新增行填 NaN）。"""
        target = pd.Index(features, name="feature")
        info = self.feature_info
        if not isinstance(info, pd.DataFrame) or info.empty:
            return build_feature_info(features)
        try:
            info = info.copy()
            info.index = [str(item) for item in info.index]
            keep = [item for item in features if item in set(info.index)]
            out = info.loc[keep].reindex(target)
            for column in FEATURE_COLUMNS:
                if column not in out.columns:
                    out[column] = np.nan
            return out
        except Exception:  # noqa: BLE001 - 重组失败就退化成纯空表
            return build_feature_info(features)

    def _reindex_sample_info(self, samples: list[str]) -> pd.DataFrame:
        """按新的 samples 顺序重组 sample_info（保留仍存在的行，新增行填 NaN）。"""
        target = pd.Index(samples, name="sample")
        info = self.sample_info
        if not isinstance(info, pd.DataFrame) or info.empty:
            return pd.DataFrame(index=target)
        try:
            info = info.copy()
            info.index = [str(item) for item in info.index]
            keep = [item for item in samples if item in set(info.index)]
            return info.loc[keep].reindex(target)
        except Exception:  # noqa: BLE001
            return pd.DataFrame(index=target)

    def apply_sample_columns(self, columns, values) -> None:
        """分析/步骤返回样本级列（sample_columns + sample_values）写回 sample_info。"""
        if not columns or not isinstance(values, dict):
            return
        if isinstance(columns, dict):
            columns = [columns]
        for item in columns:
            key = str(item.get("key") if isinstance(item, dict) else item or "").strip()
            if not key or key not in values:
                continue
            vals = values[key]
            if vals is None:
                continue
            try:
                if len(vals) != self.n_samples:
                    continue
            except TypeError:
                continue
            try:
                self.sample_info[key] = list(vals)
            except Exception:  # noqa: BLE001 - 写不进去不影响步骤本身
                continue

    # -- 杂项 -------------------------------------------------------------
    def describe(self) -> str:
        """一句话描述（日志/提示用）。"""
        parts = [f"{self.id or '(未编号)'} {self.name}", f"{self.n_samples} 样本 × {self.n_features} 特征"]
        if self.group_key:
            groups = self.groups or []
            parts.append(f"分组 {self.group_key}（{'/'.join(groups[:6])}）")
        if self.X_kind:
            parts.append(f"值类型 {self.X_kind}")
        return f"{parts[0]}（{'，'.join(parts[1:])}）"

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"<Dataset {self.describe()}>"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State:
    """多数据集注册表（DESIGN §2.2 list/select/close/reset；id 规则 ds1/ds2/…，与 sp_state 一致）。"""

    def __init__(self):
        self.datasets: dict[str, Dataset] = {}
        self.active: str | None = None
        self.counter = 0

    def new(self, **kwargs) -> Dataset:
        """新建数据集：自动分配 id（ds1/ds2/…），无 active 时设为 active。"""
        self.counter += 1
        ds_id = str(kwargs.pop("ds_id", None) or kwargs.pop("id", None) or f"ds{self.counter}")
        while ds_id in self.datasets:
            self.counter += 1
            ds_id = f"ds{self.counter}"
        dataset = Dataset(ds_id, **kwargs)
        dataset.id = ds_id
        self.datasets[ds_id] = dataset
        if not self.active or self.active not in self.datasets:
            self.active = ds_id
        return dataset

    def add(self, dataset: Dataset) -> Dataset:
        """把已构造好的 Dataset 注册进来（id 为空/冲突时重新编号）。"""
        if not isinstance(dataset, Dataset):
            raise EngineError("internal", "只能注册 Dataset 实例")
        if not dataset.id or dataset.id in self.datasets:
            self.counter += 1
            new_id = f"ds{self.counter}"
            while new_id in self.datasets:
                self.counter += 1
                new_id = f"ds{self.counter}"
            dataset.id = new_id
        self.datasets[dataset.id] = dataset
        if not self.active or self.active not in self.datasets:
            self.active = dataset.id
        return dataset

    def get(self, dataset_id: str | None = None) -> Dataset:
        """取数据集：None → active；不存在 → EngineError('not_found', 中文)。"""
        target = str(dataset_id).strip() if dataset_id else self.active
        if not target:
            raise EngineError(
                "not_found",
                "还没有打开任何数据集：请先导入特征表，或载入内置示例（demo）",
            )
        dataset = self.datasets.get(target)
        if dataset is None:
            available = "、".join(self.datasets) or "无"
            raise EngineError(
                "not_found",
                f"数据集 {target} 不存在（可能已关闭）；当前可用：{available}",
            )
        return dataset

    def select(self, dataset_id: str | None = None) -> Dataset:
        """切换 active 数据集（等价 get + 设置 active）。"""
        dataset = self.get(dataset_id)
        self.active = dataset.id
        return dataset

    def summaries(self) -> list[dict]:
        """所有数据集的摘要（DESIGN §2.2 list，顺序=创建顺序）。"""
        out = []
        for dataset in self.datasets.values():
            try:
                out.append(dataset.summary())
            except Exception as error:  # noqa: BLE001 - 单个数据集摘要失败不影响列表
                out.append({"id": dataset.id, "name": dataset.name, "error": str(error)})
        return out

    def close(self, dataset_id: str | None) -> None:
        """关闭并移除数据集；关的是 active 时自动切到下一个。"""
        target = str(dataset_id or "").strip() or (self.active or "")
        if not target or target not in self.datasets:
            raise EngineError("not_found", f"数据集 {target or '(空)'} 不存在（可能已关闭）")
        self.datasets.pop(target, None)
        if self.active == target:
            self.active = next(iter(self.datasets), None)

    def find_by_path(self, path) -> Dataset | None:
        """按导入路径查重（同一路径不重复打开）。"""
        if not path:
            return None
        try:
            target = os.path.abspath(os.path.expanduser(str(path)))
        except Exception:  # noqa: BLE001
            return None
        for dataset in self.datasets.values():
            if not dataset.path:
                continue
            try:
                if os.path.abspath(os.path.expanduser(str(dataset.path))) == target:
                    return dataset
            except Exception:  # noqa: BLE001
                continue
        return None

    def reset(self) -> None:
        """清空全部数据集（编号也从 ds1 重新开始）。"""
        self.datasets.clear()
        self.active = None
        self.counter = 0

    def __len__(self) -> int:  # pragma: no cover - 便于日志
        return len(self.datasets)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"<State datasets={list(self.datasets)} active={self.active}>"


# ---------------------------------------------------------------------------
# demo 数据（DESIGN §9）
# ---------------------------------------------------------------------------

def _demo_pick_adduct(rng) -> str:
    """按正/负离子比例挑一个加合物（正离子为主）。"""
    if float(rng.random()) < DEMO_POSITIVE_FRACTION:
        mix = DEMO_POSITIVE_ADDUCT_MIX
    else:
        mix = DEMO_NEGATIVE_ADDUCT_MIX
    names = np.array([item[0] for item in mix], dtype=object)
    probs = np.array([item[1] for item in mix], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _demo_formula_for_mass(neutral: float, tol_ppm: float = DEMO_MZ_TOL_PPM) -> dict | None:
    """给一个中性精确质量反解最简单的 CHNO 分子式（误差 ≤ tol_ppm；无解返回 None）。

    做法：枚举 C(4–45)/N(0–4)/O(0–12) 共 2730 个组合，H 由质量残差四舍五入得到，
    再用饱和度（H ≤ 2C+2+2N）与 RDBE(0–18) 过滤，取质量误差最小的组合。
    这样 demo 里 m/z（2 位小数）与 formula 的理论质量自洽，质量精度/分子式推断能命中。
    """
    neutral = float(neutral)
    if not math.isfinite(neutral):
        return None
    if not (DEMO_NEUTRAL_MASS_MIN <= neutral <= DEMO_NEUTRAL_MASS_MAX):
        return None
    h = np.rint((neutral - _FORMULA_OFFSET) / _FORMULA_H_MASS)
    valid = (h >= 2) & (h <= 2 * _FORMULA_C + 2 + 2 * _FORMULA_N)
    if not valid.any():
        return None
    masses = _FORMULA_OFFSET + h * _FORMULA_H_MASS
    error_ppm = np.abs(masses - neutral) / neutral * 1e6
    valid &= error_ppm <= float(tol_ppm)
    if not valid.any():
        return None
    rdbe = 1.0 + _FORMULA_C - h / 2.0 + _FORMULA_N / 2.0
    valid &= (rdbe >= 0.0) & (rdbe <= 18.0)
    if not valid.any():
        return None
    best = int(np.argmin(np.where(valid, error_ppm, np.inf)))
    counts = {
        "C": int(_FORMULA_C[best]),
        "H": int(round(float(h[best]))),
        "O": int(_FORMULA_O[best]),
    }
    if int(_FORMULA_N[best]) > 0:
        counts["N"] = int(_FORMULA_N[best])
    return counts


def _demo_chemicals(rng, n_features: int) -> tuple[list[str], np.ndarray, list[str]]:
    """为每个特征造 (adduct, m/z, formula)：m/z 取 2 位小数，分子式按该 m/z 反解（≤3 ppm）。"""
    adducts: list[str] = []
    mzs: list[float] = []
    formulas: list[str] = []
    used_mz: set[float] = set()  # 一个特征=一个化合物：m/z 唯一（同分异构体留给真实数据）
    guard = 0
    limit = max(2000, n_features * 60)
    while len(mzs) < n_features:
        guard += 1
        if guard > limit:
            raise EngineError("internal", "demo 分子式反解失败（候选网格无解），请检查 DEMO_* 参数")
        adduct = _demo_pick_adduct(rng)
        shift, charge = ADDUCTS.get(adduct, (0.0, 1))
        # 让 m/z 落在 [100,900] 且反推的中性质量仍在 CHNO 网格可解区间
        low = max(DEMO_MZ_MIN, (DEMO_NEUTRAL_MASS_MIN + float(shift)) / int(charge))
        high = min(DEMO_MZ_MAX, (DEMO_NEUTRAL_MASS_MAX + float(shift)) / int(charge))
        if high <= low:
            continue
        mz = round(float(rng.uniform(low, high)), 2)
        if mz in used_mz:
            continue
        neutral = neutral_mass(mz, adduct)
        counts = _demo_formula_for_mass(neutral)
        if counts is None:
            continue
        used_mz.add(float(mz))
        adducts.append(adduct)
        mzs.append(float(mz))
        formulas.append(formula_str(counts))
    return adducts, np.asarray(mzs, dtype=float), formulas


def _demo_annotations(rng, n_features: int) -> dict:
    """化学分类 + 12 条 KEGG 通路注释（336 个特征带 kegg/hmdb/pathway；其余只有化学分类）。"""
    classes = np.empty(n_features, dtype=object)
    pathways = np.full(n_features, np.nan, dtype=object)
    kegg = np.full(n_features, np.nan, dtype=object)
    hmdb = np.full(n_features, np.nan, dtype=object)
    confidence = np.full(n_features, np.nan, dtype=float)

    weights = np.asarray(DEMO_CLASS_WEIGHTS, dtype=float)
    weights = weights / weights.sum()
    default_class = rng.choice(len(DEMO_CLASS_NAMES), size=n_features, p=weights)
    for index in range(n_features):
        classes[index] = DEMO_CLASS_NAMES[int(default_class[index])]

    n_annotated = min(DEMO_N_PATHWAYS * DEMO_PATHWAY_SIZE, n_features)
    annotated = np.sort(rng.choice(n_features, n_annotated, replace=False)).astype(int)
    for path_index, (pathway_name, pathway_class) in enumerate(DEMO_PATHWAYS):
        members = annotated[path_index * DEMO_PATHWAY_SIZE:(path_index + 1) * DEMO_PATHWAY_SIZE]
        for offset, feature_index in enumerate(members):
            feature_index = int(feature_index)
            classes[feature_index] = pathway_class
            pathways[feature_index] = pathway_name
            serial = path_index * DEMO_PATHWAY_SIZE + offset
            kegg[feature_index] = f"C{(serial * 7 + 3) % 9999 + 1:05d}"
            hmdb[feature_index] = f"HMDB{((serial * 7 + 3) * 13) % 9999 + 1:07d}"
            confidence[feature_index] = float(rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2]))

    names: list[str] = []
    counters: dict[str, int] = {}
    for cls in classes:
        counters[cls] = counters.get(cls, 0) + 1
        names.append(f"{cls} {counters[cls]:03d}")

    return {
        "classes": classes,
        "names": np.asarray(names, dtype=object),
        "pathways": pathways,
        "kegg": kegg,
        "hmdb": hmdb,
        "confidence": confidence,
        "annotated_index": annotated,
    }


def _demo_diff_index(rng, annotated: np.ndarray) -> tuple[list[int], dict[int, float]]:
    """真差异特征：前 5 条通路各 24 个（12 上调 / 12 下调），|log2FC| 0.8–2.0。"""
    diff: list[int] = []
    log2fc: dict[int, float] = {}
    for path_index in range(DEMO_N_ACTIVE_PATHWAYS):
        start = path_index * DEMO_PATHWAY_SIZE
        members = annotated[start:start + DEMO_DIFF_PER_PATHWAY]
        magnitudes = rng.uniform(0.8, 2.0, len(members))
        signs = np.empty(len(members), dtype=float)
        half = len(members) // 2
        signs[:half] = 1.0
        signs[half:] = -1.0
        rng.shuffle(signs)
        for feature_index, magnitude, sign in zip(members, magnitudes, signs):
            feature_index = int(feature_index)
            log2fc[feature_index] = float(sign * magnitude)
            diff.append(feature_index)
    return diff, log2fc


def _demo_ms2_library(rng, n_spectra: int, matched_index: list[int],
                      mz_of: np.ndarray, feature_ids: list[str]) -> dict:
    """40 条 MS2 谱图：前 20 条的 precursor_mz 与特征 m/z 精确对应，其余为无关谱图。"""
    sorted_mz = np.sort(np.asarray(mz_of, dtype=float))
    spectra: list[dict] = []
    for position in range(n_spectra):
        feature_id = None
        if position < len(matched_index):
            feature_index = int(matched_index[position])
            precursor = float(mz_of[feature_index])
            feature_id = feature_ids[feature_index]
        else:
            precursor = float(rng.uniform(DEMO_MZ_MIN, DEMO_MZ_MAX))
            for _ in range(200):  # 与所有特征 m/z 至少差 0.02 Da，避免"假命中"
                cursor = int(np.searchsorted(sorted_mz, precursor))
                near = np.inf
                for offset in (cursor - 1, cursor):
                    if 0 <= offset < sorted_mz.size:
                        near = min(near, abs(float(sorted_mz[offset]) - precursor))
                if near > 0.02:
                    break
                precursor = float(rng.uniform(DEMO_MZ_MIN, DEMO_MZ_MAX))
            precursor = round(precursor, 4)

        n_peaks = int(rng.integers(8, 19))
        if precursor > 150.0:
            fragment_mz = np.sort(precursor * rng.uniform(0.20, 0.92, n_peaks))
        else:
            fragment_mz = np.sort(rng.uniform(20.0, max(precursor - 5.0, 30.0), n_peaks))
        rel_int = rng.uniform(5.0, 100.0, n_peaks)
        rel_int = rel_int / rel_int.max() * 100.0
        peaks = [[round(float(mz), 4), round(float(intensity), 1)]
                 for mz, intensity in zip(fragment_mz, rel_int)]
        spectrum = {
            "id": f"MS2-{position + 1:03d}",
            "precursor_mz": round(float(precursor), 4),
            "peaks": peaks,
        }
        if feature_id is not None:
            spectrum["feature"] = feature_id
        spectra.append(spectrum)
    return {"spectra": spectra, "path": None}


def make_demo(state: State, name: str | None = None) -> Dataset:
    """内置示例数据（DESIGN §9）。

    合成一份"病例-对照 + 合并 QC"的 LC-MS 代谢组特征表：
      * 72 样本 = case 30 / control 30 / QC 12，2 个批次（每批 6 个 QC）；
      * 1600 个特征，m/z 100–900（2 位小数，CHNO 分子式反解使理论质量自洽 ≤3 ppm）、RT 0.3–18 min；
      * 120 个真差异特征（|log2FC| 0.8–2.0，case 上调/下调各半），集中在 5 条 KEGG 通路；
      * 336 个特征带 kegg/hmdb/class/pathway 注释，覆盖 12 条内置通路（供 ORA/Mummichog 演示）；
      * 按丰度相关机制注入约 7% 缺失值；批次效应、QC 的低 RSD、高危评分驱动的生存时间都已注入；
      * 内置 40 条 MS2 谱图（20 条 precursor 与特征 m/z 对应）。

    确定性：seed=0（每次调用都重新播种，两次生成的矩阵逐元素相同）；生成时间 < 2 秒。
    """
    rng = np.random.default_rng(DEMO_SEED)
    n_features = int(DEMO_N_FEATURES)
    n_samples = int(DEMO_N_SAMPLES)

    # ---- 1) 特征理化性质：加合物 / m/z / 分子式 / 保留时间 -----------------
    adducts, mzs, formulas = _demo_chemicals(rng, n_features)
    polarities = np.asarray(
        ["pos" if str(item).rstrip().endswith("+") else "neg" for item in adducts], dtype=object
    )
    rts = np.round(DEMO_RT_MIN + (DEMO_RT_MAX - DEMO_RT_MIN) * rng.beta(1.5, 2.0, n_features), 2)
    feature_ids = _dedupe(
        [f"mz{mz:.4f}_rt{rt:.2f}" for mz, rt in zip(mzs, rts)], "feature"
    )

    # ---- 2) 化学分类 / KEGG 通路注释 --------------------------------------
    annotations = _demo_annotations(rng, n_features)
    annotated_index = annotations["annotated_index"]

    # ---- 3) 真差异特征（5 条通路 × 24 个，上调/下调各半）------------------
    diff_index, log2fc = _demo_diff_index(rng, annotated_index)
    diff_set = set(diff_index)

    # ---- 4) 高危评分特征 / 批次效应特征（都与差异特征互不重叠）------------
    remaining = np.asarray([i for i in range(n_features) if i not in diff_set], dtype=int)
    risk_index = np.sort(rng.choice(remaining, min(DEMO_N_RISK_FEATURES, remaining.size), replace=False))
    risk_set = set(int(i) for i in risk_index)
    pool = np.asarray([i for i in remaining if int(i) not in risk_set], dtype=int)
    batch_index = np.sort(rng.choice(pool, min(DEMO_N_BATCH_FEATURES, pool.size), replace=False))

    # ---- 5) 样本设计：分组 / 批次 / QC 标记 -------------------------------
    labels = np.asarray(
        ["case"] * DEMO_N_CASE + ["control"] * DEMO_N_CONTROL + ["QC"] * DEMO_N_QC, dtype=object
    )
    sample_names = (
        [f"C{i:02d}" for i in range(1, DEMO_N_CASE + 1)]
        + [f"N{i:02d}" for i in range(1, DEMO_N_CONTROL + 1)]
        + [f"QC{i:02d}" for i in range(1, DEMO_N_QC + 1)]
    )
    n_batches = DEMO_N_BATCHES
    batch = np.asarray(
        (["B1", "B2"] * ((DEMO_N_CASE + 1) // n_batches))[:DEMO_N_CASE]
        + (["B1", "B2"] * ((DEMO_N_CONTROL + 1) // n_batches))[:DEMO_N_CONTROL]
        + ["B1"] * (DEMO_N_QC // 2) + ["B2"] * (DEMO_N_QC - DEMO_N_QC // 2),
        dtype=object,
    )
    is_qc = labels == "QC"
    qc_flag = is_qc.astype(int)
    case_rows = np.flatnonzero(labels == "case")
    control_rows = np.flatnonzero(labels == "control")

    # ---- 6) 强度矩阵：基础丰度 × 样本因子 × 噪声 × 批次 × 分组 × 高危评分 ----
    base = rng.lognormal(mean=DEMO_LOG_ABUNDANCE_MEAN, sigma=DEMO_LOG_ABUNDANCE_SD, size=n_features)
    sample_sd = np.where(is_qc, DEMO_QC_SD, DEMO_BIOLOGICAL_SD)
    log_sample_factor = rng.normal(0.0, sample_sd)
    # 抹掉随机抽到的"组间文库大小漂移"（真实数据里这正是 PQN/内标归一化要处理的东西；
    # demo 若不抹掉，会凭空给上千个噪声特征注入假差异，把真信号淹掉）
    for rows in (case_rows, control_rows):
        if rows.size:
            log_sample_factor[rows] -= float(log_sample_factor[rows].mean())
    sample_factor = np.exp(log_sample_factor)
    X = base[None, :] * sample_factor[:, None]
    X *= np.exp(rng.normal(0.0, DEMO_ENTRY_NOISE, size=(n_samples, n_features)))

    # 批次效应：批次 B2 在 240 个特征上整体上移 ~35%（与分组正交）
    batch2_rows = np.flatnonzero(batch == "B2")
    if batch2_rows.size and len(batch_index):
        X[np.ix_(batch2_rows, batch_index)] *= np.exp(
            rng.normal(DEMO_BATCH_LOG_SHIFT, 0.06, (batch2_rows.size, len(batch_index)))
        )

    # 分组效应：case 组在 120 个真差异特征上按 log2FC 缩放（正=上调，负=下调）
    up_index = np.asarray([i for i in diff_index if log2fc[i] > 0], dtype=int)
    down_index = np.asarray([i for i in diff_index if log2fc[i] < 0], dtype=int)
    if up_index.size:
        X[np.ix_(case_rows, up_index)] *= (2.0 ** np.asarray([log2fc[i] for i in up_index]))[None, :]
    if down_index.size:
        X[np.ix_(case_rows, down_index)] *= (2.0 ** np.asarray([log2fc[i] for i in down_index]))[None, :]

    # 高危评分（潜变量）：驱动 60 个特征的丰度，也驱动生存时间（QC 取 0 不影响）
    risk_latent = rng.normal(0.0, 1.0, n_samples)
    risk_latent[is_qc] = 0.0
    X[:, risk_index] *= np.exp(DEMO_RISK_LOADING * risk_latent[:, None])

    # ---- 7) 缺失值：按丰度相关机制注入（低丰度更容易缺失），整体约 7% ------
    log_abundance = np.log(np.maximum(base, 1e-9))
    span = float(np.ptp(log_abundance))
    abundance_norm = (log_abundance - log_abundance.min()) / (span if span > 0 else 1.0)
    miss_prob = np.clip(
        DEMO_MISSING_BASE + DEMO_MISSING_SLOPE * (1.0 - abundance_norm) ** 1.5, 0.0, 0.6
    )
    # 最弱的 24 个特征再叠一层高缺失（真实数据里没检出/信号差的特征），让「特征过滤」有东西可剔
    poor = np.argsort(log_abundance)[:min(DEMO_N_POOR_FEATURES, n_features)]
    miss_prob[poor] = np.clip(miss_prob[poor] + rng.uniform(0.35, 0.60, poor.size), 0.0, 0.85)
    missing_mask = rng.random((n_samples, n_features)) < miss_prob[None, :]
    X[missing_mask] = np.nan
    missing_rate = float(missing_mask.mean())

    # ---- 8) 临床列：age/bmi/sex + 生存（高危评分越高，生存时间越短）--------
    age = np.clip(np.round(rng.normal(58.0, 9.0, n_samples)), 32, 86).astype(float)
    bmi = np.round(np.clip(rng.normal(25.6, 3.9, n_samples), 17.5, 40.0), 1)
    sex = np.where(rng.random(n_samples) < 0.52, "female", "male").astype(object)

    with np.errstate(divide="ignore", invalid="ignore"):
        risk_expr = np.nanmean(np.log2(np.clip(X[:, risk_index], 1e-9, None)), axis=1)
    risk_expr = np.nan_to_num(risk_expr, nan=0.0)
    biologic = ~is_qc
    center = float(risk_expr[biologic].mean()) if biologic.any() else 0.0
    spread = float(risk_expr[biologic].std(ddof=0)) if biologic.any() else 1.0
    risk_score = np.full(n_samples, np.nan, dtype=float)
    risk_score[biologic] = (risk_expr[biologic] - center) / (spread if spread > 0 else 1.0)

    hazard = DEMO_HAZARD_BASE * np.exp(DEMO_HAZARD_BETA * risk_latent)
    surv_time = -np.log(rng.uniform(0.02, 0.98, n_samples)) / np.maximum(hazard, 1e-9)
    os_time = np.round(np.minimum(surv_time, DEMO_CENSOR_TIME), 1)
    os_event = (surv_time <= DEMO_CENSOR_TIME).astype(float)

    # QC 是合并样本：没有临床信息（填 NaN，生存分析应自行丢弃）
    age[is_qc] = np.nan
    bmi[is_qc] = np.nan
    sex[is_qc] = np.nan
    os_time[is_qc] = np.nan
    os_event[is_qc] = np.nan
    risk_score[is_qc] = np.nan

    # ---- 9) 打乱样本与特征顺序（避免分组/通路在矩阵里成块排列）-------------
    sample_order = rng.permutation(n_samples)
    X = np.ascontiguousarray(X[sample_order], dtype=np.float32)
    labels = labels[sample_order]
    batch = batch[sample_order]
    qc_flag = qc_flag[sample_order]
    age = age[sample_order]
    bmi = bmi[sample_order]
    sex = sex[sample_order]
    os_time = os_time[sample_order]
    os_event = os_event[sample_order]
    risk_score = risk_score[sample_order]
    sample_names = [sample_names[i] for i in sample_order]

    feature_order = rng.permutation(n_features)
    X = np.ascontiguousarray(X[:, feature_order], dtype=np.float32)
    raw = X.copy()
    ordered_feature_ids = [feature_ids[int(i)] for i in feature_order]

    # dataset.features 的顺序必须与 X 的列顺序逐列一致
    feature_info = pd.DataFrame(
        {
            "mz": mzs[feature_order],
            "rt": rts[feature_order],
            "name": annotations["names"][feature_order],
            "formula": np.asarray(formulas, dtype=object)[feature_order],
            "adduct": np.asarray(adducts, dtype=object)[feature_order],
            "hmdb": annotations["hmdb"][feature_order],
            "kegg": annotations["kegg"][feature_order],
            "class": annotations["classes"][feature_order],
            "polarity": polarities[feature_order],
            "confidence": annotations["confidence"][feature_order],
            "pathway": annotations["pathways"][feature_order],
        },
        index=pd.Index(ordered_feature_ids, name="feature"),
    )

    sample_info = pd.DataFrame(
        {
            "group": [str(item) for item in labels],
            "batch": [str(item) for item in batch],
            "qc": qc_flag.astype(int),
            "age": age,
            "bmi": bmi,
            "sex": sex,
            "os_time": os_time,
            "os_event": os_event,
        },
        index=pd.Index(sample_names, name="sample"),
    )

    # ---- 10) MS2 谱库：20 条与特征 m/z 对应 + 20 条无关谱图 ----------------
    step = max(1, int(annotated_index.size) // DEMO_N_MATCHED_SPECTRA)
    matched_index = [int(i) for i in annotated_index[::step][:DEMO_N_MATCHED_SPECTRA]]
    library = _demo_ms2_library(rng, DEMO_N_SPECTRA, matched_index, mzs, feature_ids)

    notes = [
        "内置示例数据：72 样本（case 30 / control 30 / QC 12）× 1600 特征；2 个批次（每批 6 个 QC）",
        "120 个注入差异特征（|log2FC| 0.8–2.0，case 上调/下调各半），集中在 5 条 KEGG 通路",
        f"{int(annotated_index.size)} 个特征带 KEGG/HMDB 注释（覆盖 {DEMO_N_PATHWAYS} 条内置通路），"
        "其余特征只有化学分类预测",
        f"m/z 取 2 位小数（100–900 Da），用 CHNO 分子式反解保证理论质量与 m/z 自洽（≤{DEMO_MZ_TOL_PPM:g} ppm）",
        f"缺失值按丰度相关机制注入（实测 {missing_rate * 100:.1f}%），另 {DEMO_N_POOR_FEATURES} 个低丰度特征"
        "高缺失（供「特征过滤」演示），QC 样本的临床字段为空",
        f"内置 {DEMO_N_SPECTRA} 条 MS2 谱图，其中 {DEMO_N_MATCHED_SPECTRA} 条的 precursor_mz 与特征 m/z 精确对应",
    ]
    dataset = state.new(
        name=name or "demo-metabo",
        X=X,
        raw=raw,
        X_kind="raw",
        features=ordered_feature_ids,
        samples=sample_names,
        feature_info=feature_info,
        sample_info=sample_info,
        group_key="group",
        species="human",
        polarity="both",
        path=None,
        source_format="demo",
        library=library,
        notes=notes,
    )

    # 真值（供测试/演示对照；不在协议里，registry 不会消费）
    dataset.results["demo_truth"] = jsonable({
        "seed": DEMO_SEED,
        "groups": [str(item) for item in labels],
        "qc": [int(item) for item in qc_flag],
        "batch": [str(item) for item in batch],
        "missing_rate": missing_rate,
        "diff_features": [feature_ids[i] for i in diff_index],
        "log2fc": {feature_ids[i]: float(value) for i, value in log2fc.items()},
        "pathway_features": {
            pathway_name: [feature_ids[int(i)] for i in
                           annotated_index[p * DEMO_PATHWAY_SIZE:(p + 1) * DEMO_PATHWAY_SIZE]]
            for p, (pathway_name, _) in enumerate(DEMO_PATHWAYS)
        },
        "pathway_kegg": {
            pathway_name: [str(annotations["kegg"][int(i)])
                           for i in annotated_index[p * DEMO_PATHWAY_SIZE:(p + 1) * DEMO_PATHWAY_SIZE]]
            for p, (pathway_name, _) in enumerate(DEMO_PATHWAYS)
        },
        "active_pathways": [item[0] for item in DEMO_PATHWAYS[:DEMO_N_ACTIVE_PATHWAYS]],
        "risk_features": [feature_ids[int(i)] for i in risk_index],
        "risk_score": [None if not np.isfinite(item) else round(float(item), 4) for item in risk_score],
        "batch_features": [feature_ids[int(i)] for i in batch_index],
        "library_matched_features": [feature_ids[int(i)] for i in matched_index],
    })
    return dataset


__all__ = [
    "Dataset", "State", "STEP_ORDER", "STEP_IDS", "STEP_INDEX", "X_KINDS", "FEATURE_COLUMNS",
    "build_feature_info", "build_sample_info", "column_kind", "make_demo", "now_iso",
    "polarity_from_adduct", "normalize_polarity",
    "DEMO_SEED", "DEMO_N_SAMPLES", "DEMO_N_FEATURES", "DEMO_N_DIFF", "DEMO_N_ANNOTATED",
    "DEMO_CLASS_NAMES", "DEMO_PATHWAYS",
    "TIME_PATTERNS", "EVENT_PATTERNS",
]
