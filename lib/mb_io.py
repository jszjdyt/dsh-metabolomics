"""
dsh-metabolomics — 数据导入层（DESIGN.md §6）

对外 API（逐字固定，registry / engine 按此调用）::

    SUPPORTED_FORMATS: list[str]
    FORMAT_INFO: list[dict]          # [{"id","label","hint"}]
    load_source(path, workdir=None, options=None) -> dict
    inspect_source(path, workdir=None, options=None) -> dict

``load_source`` 统一返回::

    {"format", "X", "features", "samples", "feature_info", "sample_info",
     "notes", "warnings", "meta": {"polarity","has_mz","has_rt","library"?}}

* ``X`` 形状固定 **样本 × 特征**（float32，缺失为 ``np.nan``，不做填补）。
* ``feature_info`` 索引 = features，列至少 ``mz,rt,name,formula,adduct,hmdb,kegg,class``。
* 特征 id 生成规则：有 mz+rt → ``mz{:.4f}_rt{:.2f}``；只有名字 → 名字；重名加序号。

支持的格式（DESIGN §6）：matrix_text / xlsx / geo_series / archive / mztab /
isa_tab（MetaboLights）/ msms_library（msp/mgf）/ mzml_tic（mzML/mzXML）/
rds / rdata（R 桥）。

实现约定：
* 只依赖标准库 + numpy/pandas/openpyxl（openpyxl 用到时才 import）。
* 网络：仅在 URL 输入时用标准库 urllib；不做 EBI/NCBI 检索（那是 mb_geo）。
* 解压防路径穿越（绝对路径 / ``..`` / 盘符一律丢弃并 warning）。
* 谱库与 mzML 不做定量矩阵：矩阵是 1×1 占位（``np.nan``），真正内容在
  ``meta["library"]`` / ``meta["tic"]``；占位原因见 ``_library_payload``。
"""

from __future__ import annotations

import base64
import bz2
import csv
import gzip
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from mb_common import EngineError, hmdb_class_of  # noqa: F401 - hmdb_class_of 用于 class 回填

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = [
    "auto",
    "matrix_text",
    "xlsx",
    "geo_series",
    "archive",
    "mztab",
    "isa_tab",
    "msms_library",
    "mzml_tic",
    "rds",
    "rdata",
]

FORMAT_INFO = [
    {"id": "matrix_text", "label": "特征表（CSV/TSV/TXT，可 gz/bz2）",
     "hint": "样本×特征或特征×样本自动判方向；含 mz/rt 列会写进特征注释；分隔符自动嗅探"},
    {"id": "xlsx", "label": "Excel（.xlsx/.xlsm/.xls）",
     "hint": "可指定 sheet；多个可用 sheet 会要求选择；规则同特征表"},
    {"id": "geo_series", "label": "GEO series matrix",
     "hint": "含 !series_matrix_table_begin 的矩阵；!Sample_characteristics_ch1 转成样本表"},
    {"id": "archive", "label": "压缩包（zip/tar/tar.gz）",
     "hint": "解压后自动找特征表/mzTab/ISA-Tab/msp；多候选时用 member 指定"},
    {"id": "mztab", "label": "mzTab",
     "hint": "MTD 元数据 + PRH/PEH/SMH 段；按 abundance_assay 列建矩阵"},
    {"id": "isa_tab", "label": "ISA-Tab（MetaboLights）",
     "hint": "m_*.txt 代谢物矩阵 + s_*/a_* 表型；整目录或单个 m_ 文件都可"},
    {"id": "msms_library", "label": "MS2 谱库（msp/mgf）",
     "hint": "挂成谱库（前体 m/z + 峰表），不当作定量矩阵"},
    {"id": "mzml_tic", "label": "mzML/mzXML",
     "hint": "轻量 XML 解析 TIC 与 MS2 谱图，不做峰检测"},
    {"id": "rds", "label": "R 对象（.rds）",
     "hint": "有 Rscript 时桥接读取 matrix/data.frame/ExpressionSet"},
    {"id": "rdata", "label": "R 工作空间（.RData/.rda）",
     "hint": "有 Rscript 时桥接读取；可用 object 参数指定对象"},
    {"id": "auto", "label": "自动识别",
     "hint": "按扩展名与文件内容自动判断格式"},
]

#: data_format 输入别名 → 规范 id
FORMAT_ALIASES = {
    "csv": "matrix_text", "tsv": "matrix_text", "txt": "matrix_text",
    "text": "matrix_text", "matrix": "matrix_text", "table": "matrix_text",
    "gz": "matrix_text", "gzip": "matrix_text", "metaboanalyst": "matrix_text",
    "excel": "xlsx", "xls": "xlsx", "xlsm": "xlsx",
    "geo": "geo_series", "series_matrix": "geo_series", "geoseries": "geo_series",
    "zip": "archive", "tar": "archive", "tar.gz": "archive", "tgz": "archive",
    "compressed": "archive", "archive": "archive",
    "mztab-m": "mztab", "mztabm": "mztab",
    "isa": "isa_tab", "isatab": "isa_tab", "metabolights": "isa_tab",
    "msp": "msms_library", "mgf": "msms_library", "library": "msms_library",
    "spectra": "msms_library", "ms2": "msms_library",
    "mzml": "mzml_tic", "mzxml": "mzml_tic", "tic": "mzml_tic", "raw_tic": "mzml_tic",
    "r": "rds", "rds": "rds", "rdata": "rdata", "rda": "rdata",
}

TEXT_SUFFIXES = (".csv", ".tsv", ".txt", ".tab", ".tsv2", ".matrix", ".counts",
                 ".maf", ".csv.gz", ".tsv.gz", ".txt.gz", ".tab.gz")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2",
                    ".tar.xz", ".txz")
XLSX_SUFFIXES = (".xlsx", ".xlsm", ".xls")
MZML_SUFFIXES = (".mzml", ".mzxml", ".mzml.gz")
LIBRARY_SUFFIXES = (".msp", ".mgf", ".msp.gz", ".mgf.gz")
R_SUFFIXES = (".rds", ".rdata", ".rda")

MAX_UNPACK_BYTES = 20 * 1024 ** 3        # 解压总大小上限 20GB
MAX_DOWNLOAD_BYTES = 8 * 1024 ** 3       # 单文件下载上限 8GB
MAX_TEXT_BYTES_FOR_FULL = 2 * 1024 ** 3  # >2GB 文本矩阵直接拒绝
MAX_INSPECT_STREAM_BYTES = 1024 ** 3     # inspect 超过 1GB 不再流式统计行数
MAX_MZML_SPECTRA = 500                   # mzML 里最多解析的 MS2 谱图数
MAX_TIC_POINTS = 20000                   # TIC 曲线最多保留点数（超出等距抽稀）
R_TIMEOUT = 300
DOWNLOAD_TIMEOUT = 120
_UA = "dsh-metabolomics/0.1 (metabolomics workbench; local desktop app)"

#: 缺失值记号（大小写不敏感，去空格后比较）；空串也表示缺失
MISSING_TOKENS = frozenset({
    "", "na", "n/a", "n.a.", "nan", "null", "none", "-", "--", ".", "..", "nd", "n.d.",
    "n.d", "missing", "not detected", "notdetected", "<lod", "< loq", "0.#",
    "inf", "-inf", "infinity", "#n/a",
})

#: feature_info 的标准列（DESIGN §2.3）
FEATURE_COLUMNS = ["mz", "rt", "name", "formula", "adduct", "hmdb", "kegg", "class",
                   "polarity", "confidence"]

#: 各种格式里"特征注释列"的别名 → 标准列
_META_ALIASES: dict[str, tuple[str, ...]] = {
    "mz": ("mz", "m/z", "m_z", "mzmed", "mz_mean", "mzmean", "mzmin", "mzmax", "mass",
           "exact_mass", "exactmass", "neutral_mass", "precursor_mz", "precursormz",
           "precursor_mz_value", "mass_to_charge", "mz_expected", "expected_mz",
           "theoretical_mz", "observed_mz", "mz_value", "mzvalue", "mz_ion",
           "precursor_mass", "precursormass", "accurate_mass", "target_mz", "mz(da)"),
    "rt": ("rt", "r.t.", "retention_time", "retentiontime", "retention time", "rt_s",
           "rt(s)", "rt_sec", "rt(sec)", "rt_min", "rt(min)", "rt(minutes)", "retention_time_min",
           "retention_time_s", "rtmed", "rt_mean", "rtmean", "rtmin", "rt_sec_", "tr",
           "tr(min)", "scan_time", "time_min", "retention_time_(min)", "retention index"),
    "name": ("name", "metabolite", "metabolite_name", "metabolite_identification",
             "compound", "compound_name", "feature_name", "feature.name", "annotation",
             "identity", "metabolite name", "biochemical", "lipid_name", "brief name",
             "metabolite_id_name", "compound_id_name"),
    "formula": ("formula", "molecular_formula", "chemical_formula", "empirical_formula",
                "molecular formula", "chem_formula", "formula_", "sum_formula"),
    "adduct": ("adduct", "adducts", "adduct_ion", "ion_type", "ion_species", "adduct ion"),
    "hmdb": ("hmdb", "hmdb_id", "hmdbid", "hmdb id", "database_identifier", "hmdb_id_"),
    "kegg": ("kegg", "kegg_id", "keggid", "kegg compound", "kegg_compound", "kegg id"),
    "class": ("class", "hmdb_class", "chemical_class", "superclass", "metabolite_class",
              "chemical_taxonomy", "class_", "ontology_class", "compound_class", "lipid_class"),
    "polarity": ("polarity", "ion_mode_polarity", "polarity_"),
    "confidence": ("confidence", "annotation_level", "msi_level", "ident_level",
                   "identification_level", "reliability", "level"),
}
_META_INDEX: dict[str, str] = {}
for _kind, _aliases in _META_ALIASES.items():
    for _alias in _aliases:
        _META_INDEX.setdefault(_alias, _kind)

#: 首列像"特征 id 列"的表头名（→ 设成行索引）
_INDEX_COLUMN_NAMES = frozenset({
    "id", "feature", "features", "feature_id", "featureid", "feature_name",
    "metabolite", "metabolites", "compound", "compound_id", "name", "names",
    "bin", "peak", "peak_id", "peakid", "probe", "probe_id", "probeid",
    "row", "index", "x", "variable", "variables",
    "sample", "sample_id", "sampleid", "samples", "sample_name", "subject",
    "patient", "title", "geo_accession", "gsm", "spectrum", "spectrum_id",
})

#: 特征 id 形态（mz..._rt... / HMDB / KEGG / XCMS / 纯 m/z）
_FEATURE_LABEL_RE = re.compile(
    r"^(?:"
    r"mz[_\-\s]?\d{2,}(?:\.\d+)?(?:[_\-\s]?rt[_\-\s]?\d+(?:\.\d+)?)?"
    r"|(?:hmdb)\d+"
    r"|(?:c|d|g)\d{4,5}"
    r"|(?:hmdb|kegg|chebi|pubchem|cas)[:_\-]\w+"
    r"|[mfx][_\-]?\d+"
    r"|m\d+\.\d+(?:t\d+(?:\.\d+)?)?"
    r"|\d{2,4}\.\d{2,6}"
    r"|\d+(?:\.\d+)?[_\-\s](?:da|m/z|mz|s|min)"
    r")$",
    re.IGNORECASE,
)

#: 样本 id 形态（GSM/S1/QC/ctrl/case/repeat…）
_SAMPLE_LABEL_RE = re.compile(
    r"^(?:gsm\d+|mtbls\d+|s\d{1,4}|sample[_\-.\s]?\d+|smp\d+|ctrl\w*|control\w*|"
    r"case\w*|treat\w*|normal\w*|tumou?r\w*|qc\d*|pool\w*|blank\w*|rep\d+|replicate\w*|"
    r"wt\d*|ko\d*|sh\w+|oe\w*|dose\w*|time\w*\d+|p\d+[_\-\w]*|subject\w*|patient\w*|"
    r"[\w.\-]*?(?:pre|post|day|week|hour)[_\-]?\d+)$",
    re.IGNORECASE,
)

#: 特征 id 里内嵌的 mz/rt（与 mb_state._FEATURE_ID_PATTERN 同义）
_MZRT_LABEL_RE = re.compile(
    r"mz[_\-\s]*([0-9]+(?:\.[0-9]+)?)[_\-]?rt[_\-\s]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_WS_RE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _err(code: str, message: str, **details) -> EngineError:
    return EngineError(code, message, **details)


def _workdirs(workdir=None) -> dict:
    """保证 <workdir>/{unpack,tmp,downloads} 存在；workdir 缺省用临时目录。"""
    base = Path(workdir).expanduser() if workdir else Path(tempfile.mkdtemp(prefix="mb-io-"))
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _err("bad_request", f"无法创建工作目录：{base}（{error}）")
    dirs = {"base": base, "unpack": base / "unpack", "tmp": base / "tmp",
            "downloads": base / "downloads"}
    for key in ("unpack", "tmp", "downloads"):
        try:
            dirs[key].mkdir(parents=True, exist_ok=True)
        except OSError:  # noqa: PERF203 - 目录不可写时退回临时目录
            dirs[key] = Path(tempfile.mkdtemp(prefix=f"mb-{key}-"))
    return dirs


def _looks_like_url(source) -> bool:
    return bool(re.match(r"^(?:https?|ftp)://", str(source or "").strip(), re.I))


def _norm_format(value) -> str:
    """data_format 参数规范化（别名 → 规范的 SUPPORTED_FORMATS id）。"""
    text = str(value or "auto").strip().lower()
    if not text or text == "auto":
        return "auto"
    if text in SUPPORTED_FORMATS:
        return text
    return FORMAT_ALIASES.get(text, text)


def _norm_key(text) -> str:
    """列名归一化：小写、去首尾空白、非字母数字压成下划线。"""
    raw = str(text or "").strip().lower()
    raw = raw.replace("（", "(").replace("）", ")").replace("／", "/")
    raw = re.sub(r"[^0-9a-z./+\-]+", "_", raw)
    return re.sub(r"_+", "_", raw).strip("_")


def _open_text(path):
    """按 .gz/.bz2 打开文本句柄（统一 utf-8 + errors=replace）。"""
    lower = str(path).lower()
    if lower.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if lower.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _read_head(path, size: int = 2 * 1024 * 1024) -> str:
    """读文件头（自动解压 gz/bz2），返回文本。"""
    path = Path(path)
    try:
        lower = path.name.lower()
        if lower.endswith(".gz"):
            with gzip.open(path, "rb") as handle:
                raw = handle.read(size)
        elif lower.endswith(".bz2"):
            with bz2.open(path, "rb") as handle:
                raw = handle.read(size)
        else:
            with open(path, "rb") as handle:
                raw = handle.read(size)
    except (OSError, EOFError) as error:
        raise _err("bad_request", f"无法读取文件（{path.name}）：{error}")
    return raw.decode("utf-8", "replace")


def _iter_lines(path):
    """流式迭代文本行（不整表载入，供 inspect 计数用）。"""
    try:
        with _open_text(path) as handle:
            for line in handle:
                yield line.rstrip("\r\n")
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise _err("bad_request", f"无法读取文件（{Path(path).name}）：{error}")


def _sniff_delimiter(lines) -> str:
    """分隔符嗅探：看首个非空行的候选计数，返回 ','/'\\t'/';'/'|'/'whitespace'。"""
    sample = [line for line in (lines or [])[:50] if line.strip()]
    if not sample:
        return ","
    first = sample[0]
    best, best_count = ",", 0
    for candidate in ("\t", ",", ";", "|"):
        count = first.count(candidate)
        if count > best_count:
            best, best_count = candidate, count
    if best_count > 0:
        return best
    if len(sample) >= 2 and all(len(_split_line(line, "whitespace")) >= 2 for line in sample[:5]):
        return "whitespace"
    return ","


def _split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return re.split(r"\s+", str(line).strip())
    try:
        return next(csv.reader([line], delimiter=delimiter))
    except (csv.Error, StopIteration):
        return str(line).split(delimiter)


def _is_numeric_text(value) -> bool:
    text = str(value).strip().replace(",", "").replace(" ", "")
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def _to_float(value) -> float:
    """单元格 → float；缺失/无法解析一律 nan。"""
    text = str(value).strip()
    if text.strip().lower() in MISSING_TOKENS:
        return float("nan")
    try:
        return float(text.replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def _numeric_series(series) -> pd.Series:
    text = series.astype(str).str.strip()
    missing = text.str.lower().isin(MISSING_TOKENS) | series.isna()
    values = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    values = values.mask(missing)
    return values.astype(float)


def _dedupe(names, prefix: str = "feature") -> list[str]:
    """去空/同名加序号（保持顺序），保证唯一。"""
    seen: dict[str, int] = {}
    out: list[str] = []
    for index, raw in enumerate(names or []):
        name = str(raw).strip().strip('"').strip("'")
        if not name or name.lower() in ("nan", "none", "null"):
            name = f"{prefix}{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name] + 1}"
            while name in seen:
                seen[name] += 1
                name = f"{name.rsplit('_', 1)[0]}_{seen[name] + 1}"
        seen[name] = 0
        out.append(name)
    return out


def _ratio(labels, pattern) -> float:
    labels = [str(item).strip() for item in (labels or []) if str(item).strip()]
    if not labels:
        return 0.0
    hits = sum(1 for label in labels if pattern.match(label))
    return hits / len(labels)


def _parse_feature_id_mz_rt(label) -> tuple[float, float]:
    match = _MZRT_LABEL_RE.search(str(label or ""))
    if not match:
        return float("nan"), float("nan")
    try:
        return float(match.group(1)), float(match.group(2))
    except (TypeError, ValueError):
        return float("nan"), float("nan")


def _meta_kind(name) -> str | None:
    """列名 → 特征注释标准列名（mz/rt/name/...）；不是注释列返回 None。"""
    key = _norm_key(name)
    if key in _META_INDEX:
        return _META_INDEX[key]
    # 带单位的变体：RT(min) / mz(Da) / retention_time_min
    stripped = re.sub(r"\(.*?\)", "", key).strip("_-")
    if stripped in _META_INDEX:
        return _META_INDEX[stripped]
    for suffix in ("_min", "_sec", "_s", "_da", "_value", "_id"):
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in _META_INDEX and _META_INDEX[base] in ("rt", "mz"):
                return _META_INDEX[base]
    return None


def _resolve_column(frame: pd.DataFrame, spec):
    """把列名 / 列序号描述解析成 frame 的真实列名。"""
    if spec is None or spec == "":
        return None
    columns = list(frame.columns)
    if isinstance(spec, (int, np.integer)):
        index = int(spec)
        return columns[index] if 0 <= index < len(columns) else None
    text = str(spec).strip()
    if text.isdigit() and int(text) < len(columns):
        return columns[int(text)]
    for column in columns:
        if str(column).strip().lower() == text.lower():
            return column
    for column in columns:
        if _norm_key(column) == _norm_key(text):
            return column
    return None


# ---------------------------------------------------------------------------
# 矩阵装配（统一转成 样本 × 特征）
# ---------------------------------------------------------------------------


def _detect_header(cells) -> bool:
    """首行是否表头：只要有一个非数值单元格就当表头。"""
    values = [str(cell) for cell in (cells or [])]
    if not values:
        return False
    numeric = sum(1 for cell in values if _is_numeric_text(cell))
    return numeric < len(values)


def _detect_index(lines, delimiter: str, header: bool) -> bool:
    """首列是否行名（值非数值，或表头名是 id/feature 类）。"""
    if not lines:
        return False
    first_cells = _split_line(lines[0], delimiter) if lines else [""]
    data_lines = [line for line in lines[1:41] if line.strip()]
    if data_lines:
        values = []
        for line in data_lines:
            cells = _split_line(line, delimiter)
            values.append(cells[0] if cells else "")
        non_numeric = sum(1 for cell in values
                          if str(cell).strip() and not _is_numeric_text(cell))
        if non_numeric >= max(1, int(0.6 * len(values))):
            return True
    if header and first_cells:
        name = _norm_key(first_cells[0])
        if name in _INDEX_COLUMN_NAMES:
            return True
    return False


def _infer_orientation(row_labels, col_labels, shape, notes: list[str]) -> bool:
    """判断是否需要转置（True=行是特征，需要转成 样本×特征）。"""
    sample_rows = _ratio(row_labels, _SAMPLE_LABEL_RE)
    sample_cols = _ratio(col_labels, _SAMPLE_LABEL_RE)
    feature_rows = _ratio(row_labels, _FEATURE_LABEL_RE)
    feature_cols = _ratio(col_labels, _FEATURE_LABEL_RE)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if feature_cols >= 0.6 and feature_rows < 0.4:
        notes.append("列名像特征 id（mz/RT/HMDB 等）→ 行是样本（保持方向）")
        return False
    if feature_rows >= 0.6 and feature_cols < 0.4:
        notes.append("行名像特征 id（mz/RT/HMDB 等）→ 行是特征（已转置成 样本×特征）")
        return True
    if sample_rows >= 0.6 and sample_cols < 0.4:
        notes.append("行名像样本名 → 行是样本（保持方向）")
        return False
    if sample_cols >= 0.6 and sample_rows < 0.4:
        notes.append("列名像样本名 → 列是样本（已转置成 样本×特征）")
        return True
    if n_cols > n_rows:
        notes.append(f"名称无法判断方向，按列数（{n_cols}）多于行数（{n_rows}）推断列是特征 → 行是样本（保持）")
        return False
    notes.append(f"名称无法判断方向，按行数（{n_rows}）多于列数（{n_cols}）推断行是特征 → 已转置成 样本×特征")
    return True


def _align_frame(frame, index_values, index_name: str) -> pd.DataFrame:
    """按 index_values 对齐 DataFrame（先精确、再大小写不敏感、最后 NaN 行）。"""
    expected = pd.Index([str(item) for item in (index_values or [])], name=index_name)
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(index=expected)
    out = frame.copy()
    out.index = [str(item) for item in out.index]
    if list(out.index) == list(expected):
        out.index = expected
        return out
    mapping: dict[str, str] = {}
    for label in out.index:
        mapping.setdefault(str(label).strip().lower(), label)
    rows = []
    for key in expected:
        hit = mapping.get(str(key).strip().lower())
        rows.append(out.loc[hit] if hit is not None else pd.Series(dtype=object))
    aligned = pd.DataFrame(rows, index=expected)
    for column in out.columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned.reindex(columns=list(out.columns))


def _empty_feature_frame(features) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.Index([str(item) for item in (features or [])], name="feature"))
    for column in FEATURE_COLUMNS:
        frame[column] = np.nan
    return frame


def _feature_frame(features, columns: dict[str, list]) -> pd.DataFrame:
    """按 features 顺序拼 feature_info（缺列填 NaN，空串当缺失）。"""
    count = len(features)
    frame = pd.DataFrame(index=pd.Index([str(item) for item in features], name="feature"))
    for column in FEATURE_COLUMNS:
        values = columns.get(column)
        if values is None:
            frame[column] = np.nan
            continue
        values = list(values)[:count] + [None] * max(0, count - len(values))
        if column in ("mz", "rt", "confidence"):
            frame[column] = [float("nan") if _is_missing(value) else _to_float(value)
                             for value in values]
        else:
            cleaned = []
            for value in values:
                text = "" if value is None else str(value).strip()
                cleaned.append(np.nan if _is_missing(text) else text)
            frame[column] = pd.Series(cleaned, index=frame.index, dtype=object)
    return frame


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if value is not value:  # noqa: PLR0124 - numpy NaN
            return True
    except Exception:  # noqa: BLE001
        pass
    text = str(value).strip()
    return text.lower() in MISSING_TOKENS


def _payload(fmt: str, source, X, features, samples, feature_info=None, sample_info=None,
             notes=None, warnings=None, meta=None, name=None) -> dict:
    """组装 §6 的统一导入结果（样本×特征、float32、缺失 np.nan）。"""
    if X is None:
        matrix = np.zeros((0, 0), dtype=np.float32)
    else:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise _err("internal", "导入结果矩阵维度异常（应为 样本×特征 二维）")
    feature_list = _dedupe([str(item) for item in (features or [])], "feature")
    sample_list = _dedupe([str(item) for item in (samples or [])], "sample")
    if matrix.size:
        if len(feature_list) != matrix.shape[1]:
            raise _err("internal", f"特征数（{len(feature_list)}）与矩阵列数（{matrix.shape[1]}）不一致")
        if len(sample_list) != matrix.shape[0]:
            raise _err("internal", f"样本数（{len(sample_list)}）与矩阵行数（{matrix.shape[0]}）不一致")
    info = feature_info if isinstance(feature_info, pd.DataFrame) else _empty_feature_frame(feature_list)
    info = _align_frame(info, feature_list, "feature")
    if isinstance(sample_info, pd.DataFrame) and sample_list:
        sinfo = _align_frame(sample_info, sample_list, "sample")
    else:
        sinfo = pd.DataFrame(index=pd.Index(sample_list, name="sample"))
    out_meta = dict(meta or {})
    out_meta.setdefault("n_samples", int(matrix.shape[0]))
    out_meta.setdefault("n_features", int(matrix.shape[1]))
    out_meta.setdefault("has_mz", bool(len(info) and "mz" in info.columns and info["mz"].notna().any()))
    out_meta.setdefault("has_rt", bool(len(info) and "rt" in info.columns and info["rt"].notna().any()))
    out_meta.setdefault("polarity", None)
    base_name = str(name or Path(urllib.parse.urlparse(str(source)).path or str(source)).name or "dataset")
    out_notes = [str(item) for item in (notes or []) if str(item).strip()]
    out_warnings = [str(item) for item in (warnings or []) if str(item).strip()]
    return {
        "format": str(fmt),
        "name": base_name,
        "X": matrix,
        "features": feature_list,
        "samples": sample_list,
        "feature_info": info,
        "sample_info": sinfo,
        "library": out_meta.get("library"),
        "notes": out_notes,
        "warnings": out_warnings,
        "meta": out_meta,
    }


def _matrix_from_frame(frame: pd.DataFrame, fmt: str, source, options: dict,
                       notes: list[str], warnings: list[str],
                       sample_info=None, meta_extra=None, id_values=None) -> dict:
    """把任意分隔表格转成 样本×特征 矩阵并组装 payload。

    方向判定顺序（DESIGN §6）：
      1. options.transpose 显式指定；
      2. 表里有 mz/rt（或 ≥2 个 name/formula/adduct/hmdb/kegg 注释列）→ 行是特征，转置；
      3. 行/列名形态打分（样本名 vs 特征 id）；
      4. 兜底按"特征数 > 样本数"的常规形态推断。
    """
    options = options or {}
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise _err("bad_request", f"表格没有可用的行列：{Path(str(source)).name}")
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame.index = [str(item).strip() for item in frame.index]

    # 1) 显式 feature_col：该列的值就是特征 id
    feature_col = _resolve_column(frame, options.get("feature_col"))
    if feature_col is not None:
        id_values = [str(value).strip() for value in frame[feature_col].tolist()]
        frame = frame.drop(columns=[feature_col])
        notes.append(f"按 feature_col=「{feature_col}」取特征 id")

    # 2) 识别特征注释列
    meta_cols: dict[str, str] = {}
    for column in frame.columns:
        kind = _meta_kind(column)
        if kind and kind not in meta_cols:
            meta_cols[kind] = column
    for option_key, kind in (("mz_col", "mz"), ("rt_col", "rt")):
        resolved = _resolve_column(frame, options.get(option_key))
        if resolved is not None:
            meta_cols[kind] = resolved
    for kind in list(meta_cols):
        if meta_cols[kind] not in frame.columns:
            del meta_cols[kind]
    strong_meta = bool(
        "mz" in meta_cols or "rt" in meta_cols
        or len(set(meta_cols) & {"name", "formula", "adduct", "hmdb", "kegg"}) >= 2
    )
    meta_col_set = set(meta_cols.values())
    data_cols = [column for column in frame.columns if column not in meta_col_set]
    if not data_cols:
        raise _err("bad_request",
                   "表里只有注释列、没有数值列（需要每个样本一列强度值）")

    numbers = pd.DataFrame({column: _numeric_series(frame[column]) for column in data_cols},
                           index=frame.index)
    total_cells = max(1, numbers.shape[0] * numbers.shape[1])

    def _bad_cells(column) -> int:
        raw = frame[column]
        as_text = raw.astype(str).str.strip().str.lower()
        empty = raw.isna() | as_text.isin(MISSING_TOKENS)
        return int(((~empty) & numbers[column].isna()).sum())

    bad_count = sum(_bad_cells(column) for column in data_cols)
    if bad_count:
        warnings.append(f"有 {bad_count} 个单元格不是数值（已按缺失 np.nan 处理，占 {bad_count / total_cells:.2%}）")

    all_nan_rows = numbers.isna().all(axis=1)
    all_nan_cols = numbers.isna().all(axis=0)
    drop_rows = int(all_nan_rows.sum())
    drop_cols = int(all_nan_cols.sum())
    if not len(numbers) or not len(numbers.columns) or all_nan_rows.all() or all_nan_cols.all():
        raise _err("bad_request",
                   f"表格里没有可解析的数值（需要每个样本一列强度值）：{Path(str(source)).name}")
    keep_rows = ~all_nan_rows
    keep_cols = ~all_nan_cols
    numbers = numbers.loc[keep_rows, keep_cols]
    if numbers.empty or numbers.shape[0] == 0 or numbers.shape[1] == 0:
        raise _err("bad_request", f"表格里没有可解析的数值：{Path(str(source)).name}")
    if drop_rows:
        warnings.append(f"丢弃 {drop_rows} 行全空/全非数值行")
    if drop_cols:
        warnings.append(f"丢弃 {drop_cols} 个全缺失列")
    row_labels = [str(item) for item in numbers.index]
    col_labels = [str(item) for item in numbers.columns]
    # 被整列丢弃的文本列（全非数值）：按方向补成样本注释或特征注释
    dropped_text_cols = [column for column in data_cols if column not in numbers.columns]

    # feature_col 的 id 需要跟着保留下来的行一起过滤
    if id_values is not None:
        positions = [i for i, keep in enumerate(keep_rows.tolist()) if keep]
        id_values = [id_values[i] for i in positions if i < len(id_values)]

    # 3) 方向
    transpose_opt = options.get("transpose")
    if transpose_opt is not None:
        transpose = bool(transpose_opt)
        notes.append(f"按用户指定 transpose={transpose} 处理")
    elif strong_meta:
        transpose = True
        notes.append("表里有特征注释列（" + "、".join(sorted(meta_cols)) +
                     "）→ 判定为「特征×样本」，已转置成 样本×特征")
    else:
        transpose = _infer_orientation(row_labels, col_labels, numbers.shape, notes)

    values = numbers.to_numpy(dtype=np.float64)
    sample_annotations: dict[str, list] = {}
    feature_annotations: dict[str, list] = {}

    if transpose:
        values = values.T
        samples = [str(column) for column in numbers.columns]
        features_raw = [str(item) for item in numbers.index]
        index_labels = list(numbers.index)
        aligned_meta = {kind: frame[column].reindex(index_labels).tolist()
                        for kind, column in meta_cols.items()}
        for column in dropped_text_cols:
            series = frame[column].reindex(index_labels)
            if series.notna().any():
                feature_annotations[str(column)] = series.tolist()
    else:
        samples = [str(item) for item in numbers.index]
        features_raw = [str(column) for column in numbers.columns]
        index_labels = list(numbers.index)
        aligned_meta = {}
        # 结构上不属于特征的行级/列级文本列 → 样本注释（如 group/class/batch）
        for column in dropped_text_cols:
            series = frame[column].reindex(index_labels)
            if series.notna().any():
                sample_annotations[str(column)] = series.tolist()
        for kind, column in meta_cols.items():
            series = frame[column].reindex(index_labels)
            if series.notna().any():
                sample_annotations.setdefault(str(column), series.tolist())
        if dropped_text_cols:
            warnings.append("列 " + "、".join(str(c) for c in dropped_text_cols[:5]) +
                            " 不是数值，已作为样本注释放进 sample_info")

    n_features = int(values.shape[1])
    # 从标签里回填 mz/rt（MetaboAnalyst 风格 mz203.0526_rt1.23）
    label_mz = [float("nan")] * n_features
    label_rt = [float("nan")] * n_features
    for index, label in enumerate(features_raw):
        mz, rt = _parse_feature_id_mz_rt(label)
        label_mz[index] = mz
        label_rt[index] = rt
    mz_values = _numeric_list(aligned_meta.get("mz"), n_features)
    rt_values = _numeric_list(aligned_meta.get("rt"), n_features)
    for index in range(n_features):
        if math.isnan(mz_values[index]) and not math.isnan(label_mz[index]):
            mz_values[index] = label_mz[index]
        if math.isnan(rt_values[index]) and not math.isnan(label_rt[index]):
            rt_values[index] = label_rt[index]

    name_values = _text_list(aligned_meta.get("name"), n_features)
    formula_values = _text_list(aligned_meta.get("formula"), n_features)
    adduct_values = _text_list(aligned_meta.get("adduct"), n_features)
    hmdb_values = _text_list(aligned_meta.get("hmdb"), n_features)
    kegg_values = _text_list(aligned_meta.get("kegg"), n_features)
    class_values = _text_list(aligned_meta.get("class"), n_features)
    polarity_values = _text_list(aligned_meta.get("polarity"), n_features)
    confidence_values = _text_list(aligned_meta.get("confidence"), n_features)

    # 4) 特征 id
    raw_ids = id_values if id_values is not None else None
    ids: list[str] = []
    for index in range(n_features):
        mz, rt = mz_values[index], rt_values[index]
        name = name_values[index]
        fallback = raw_ids[index] if raw_ids is not None and index < len(raw_ids) else ""
        if not math.isnan(mz) and not math.isnan(rt):
            ids.append(f"mz{mz:.4f}_rt{rt:.2f}")
        elif name:
            ids.append(str(name))
        elif str(fallback).strip():
            ids.append(str(fallback).strip())
        else:
            ids.append(features_raw[index] if features_raw[index] else f"F{index + 1}")
    features = _dedupe(ids, "feature")
    dup = sum(1 for a, b in zip(ids, features) if str(a) != str(b))
    if dup:
        warnings.append(f"{dup} 个特征 id 重复，已加序号去重")

    feature_info = _feature_frame(features, {
        "mz": mz_values, "rt": rt_values, "name": name_values, "formula": formula_values,
        "adduct": adduct_values, "hmdb": hmdb_values, "kegg": kegg_values,
        "class": class_values, "polarity": polarity_values, "confidence": confidence_values,
    })
    if transpose and feature_annotations:
        for column, values_list in feature_annotations.items():
            if column not in feature_info.columns:
                feature_info[column] = list(values_list)[:n_features] + [np.nan] * max(
                    0, n_features - len(values_list))
    _backfill_class(feature_info, warnings)

    # 5) 缺失 / 0
    zero_as_missing = bool(options.get("zero_as_missing"))
    n_missing = int(np.isnan(values).sum())
    if zero_as_missing:
        zeros = int((values == 0).sum())
        if zeros:
            values = np.where(values == 0, np.nan, values)
            n_missing += zeros
            warnings.append(f"按 zero_as_missing=true 把 {zeros} 个 0 视为缺失（np.nan）")
    if n_missing:
        warnings.append(f"有 {n_missing} 个缺失单元格（{n_missing / max(1, values.size):.2%}），"
                        "已保留为 np.nan，未做填补")

    # 6) 样本表
    sinfo = _merge_sample_info(sample_info, samples, sample_annotations)
    meta = {
        "has_mz": bool(np.isfinite(mz_values).any()),
        "has_rt": bool(np.isfinite(rt_values).any()),
        "matrix_order_original": "features_by_samples" if transpose else "samples_by_features",
        "missing_values": {"n_missing": int(n_missing), "imputed_value": None},
        "feature_columns": sorted(meta_cols),
        "n_non_numeric_cells": int(bad_count),
    }
    polarity = _polarity_of(feature_info)
    if polarity:
        meta["polarity"] = polarity
    meta.update(meta_extra or {})
    return _payload(fmt, source, values, features, samples, feature_info=feature_info,
                    sample_info=sinfo, notes=notes, warnings=warnings, meta=meta)


def _numeric_list(values, count: int) -> list[float]:
    out = [float("nan")] * count
    if values is None:
        return out
    for index in range(min(count, len(values))):
        out[index] = _to_float(values[index])
    return out


def _text_list(values, count: int) -> list[str]:
    out = [""] * count
    if values is None:
        return out
    for index in range(min(count, len(values))):
        value = values[index]
        if _is_missing(value):
            continue
        out[index] = str(value).strip()
    return out


def _backfill_class(feature_info: pd.DataFrame, warnings: list[str]) -> None:
    """class 列缺注释时用内置 HMDB 化学分类词典按名字回填（Wishart 2022）。"""
    if feature_info.empty or "class" not in feature_info.columns or "name" not in feature_info.columns:
        return
    # _align_frame 可能把"目前全空"的列推断成 float，先强制成 object 再回填
    if feature_info["class"].dtype != object:
        feature_info["class"] = feature_info["class"].astype(object)
    missing = feature_info["class"].isna()
    if not missing.any():
        return
    filled = 0
    for index in feature_info.index[missing]:
        name = feature_info.at[index, "name"]
        if not name:
            continue
        category = None
        try:
            category = hmdb_class_of(str(name))
        except Exception:  # noqa: BLE001 - 词典失败不影响导入
            category = None
        if category:
            feature_info.at[index, "class"] = category
            filled += 1
    if filled:
        warnings.append(f"按内置 HMDB 分类词典给 {filled} 个代谢物补了 class（化学分类）")


def _polarity_of(feature_info: pd.DataFrame) -> str | None:
    if feature_info.empty:
        return None
    for column in ("polarity", "adduct"):
        if column not in feature_info.columns:
            continue
        values = [str(value).strip() for value in feature_info[column].dropna().tolist()]
        if not values:
            continue
        modes = set()
        for value in values:
            lowered = value.lower()
            if "pos" in lowered or value.rstrip().endswith("+"):
                modes.add("pos")
            elif "neg" in lowered or value.rstrip().endswith("-"):
                modes.add("neg")
        if modes == {"pos"}:
            return "pos"
        if modes == {"neg"}:
            return "neg"
        if modes:
            return "both"
    return None


def _merge_sample_info(sample_info, samples, annotations: dict[str, list]) -> pd.DataFrame:
    """样本表 = 导入自带 sample_info + 表内样本注释；长度按 samples 对齐。"""
    base = sample_info if isinstance(sample_info, pd.DataFrame) else None
    if base is not None and len(base):
        out = _align_frame(base, samples, "sample")
    else:
        out = pd.DataFrame(index=pd.Index([str(item) for item in samples], name="sample"))
    for column, values in (annotations or {}).items():
        name = str(column)
        if name in out.columns:
            name = f"{name}_sheet"
            suffix = 2
            while name in out.columns:
                name = f"{column}_sheet{suffix}"
                suffix += 1
        cleaned = [np.nan if _is_missing(value) else str(value).strip() for value in values]
        if len(cleaned) < len(out):
            cleaned += [np.nan] * (len(out) - len(cleaned))
        out[name] = cleaned[:len(out)]
    return out


# ---------------------------------------------------------------------------
# 文本矩阵读取
# ---------------------------------------------------------------------------


def _read_matrix_frame(path, options: dict, notes: list[str], warnings: list[str]) -> pd.DataFrame:
    """读取任意分隔文本 → DataFrame（尚未判方向）。"""
    options = options or {}
    head = _read_head(path, 4 * 1024 * 1024)
    if not head.strip():
        raise _err("bad_request", f"文件是空的：{Path(path).name}")
    delimiter = options.get("delimiter") or _sniff_delimiter(head.splitlines())
    delimiter = _normalize_delimiter(delimiter)
    lines = head.splitlines()
    first_cells = _split_line(lines[0], delimiter) if lines else [""]
    header = _detect_header(first_cells)
    has_index = _detect_index(lines, delimiter, header)
    notes.append("分隔符：" + ("TAB" if delimiter == "\t" else
                              ("空白" if delimiter == "whitespace" else delimiter)))
    if not header:
        notes.append("未检测到表头，列名自动生成")
    if not has_index:
        notes.append("未检测到行名，行名自动生成")
    kwargs = {"header": 0 if header else None, "low_memory": False}
    if has_index:
        kwargs["index_col"] = 0
    sep = r"\s+" if delimiter == "whitespace" else delimiter
    kwargs["sep"] = sep
    try:
        lower = str(path).lower()
        if delimiter == "whitespace":
            kwargs["engine"] = "python"
            kwargs.pop("low_memory", None)
        if lower.endswith((".gz", ".bz2")):
            with _open_text(path) as handle:
                frame = pd.read_csv(handle, **kwargs)
        else:
            frame = pd.read_csv(str(path), **kwargs)
    except (UnicodeDecodeError, LookupError):
        raise _err("unsupported", "文件不是可读文本（可能二进制或非 UTF-8 编码）；可用 data_format 指定格式")
    except pd.errors.ParserError as error:
        raise _err("bad_request", f"无法解析表格（{Path(path).name}）：{error}")
    except EngineError:
        raise
    except Exception as error:  # noqa: BLE001
        raise _err("bad_request", f"读取表格失败（{Path(path).name}）：{error}")
    return frame


def _normalize_delimiter(value) -> str:
    text = str(value or "")
    mapping = {"tab": "\t", "\\t": "\t", "comma": ",", "semicolon": ";", "pipe": "|",
               "space": "whitespace", "blank": "whitespace", "whitespace": "whitespace",
               " ": "whitespace"}
    if text.lower() in mapping:
        return mapping[text.lower()]
    if text in (",", "\t", ";", "|"):
        return text
    if len(text) > 1:
        raise _err("bad_request", f"delimiter 参数不合法：{value!r}（应为逗号/TAB/分号/空白）")
    return text or ","


# ---------------------------------------------------------------------------
# GEO series matrix
# ---------------------------------------------------------------------------


def _parse_geo_series(path, source, options) -> dict:
    """解析 GEO series matrix：table 段 → X（转置）、!Sample_* → sample_info。"""
    notes = ["从 !series_matrix_table_begin 段解析（特征×样本 → 已转置成 样本×特征）",
             "!Sample_* 元数据已转成样本表"]
    warnings: list[str] = []
    size = Path(path).stat().st_size if Path(path).exists() else 0
    if size > MAX_TEXT_BYTES_FOR_FULL:
        raise _err("bad_request",
                   f"series matrix 过大（>{MAX_TEXT_BYTES_FOR_FULL // 1024 ** 3}GB），请先裁剪样本/特征再导入")
    sample_lines: list[str] = []
    header_cells = None
    rows: list[list[str]] = []
    in_table = False
    try:
        for line in _iter_lines(path):
            stripped = line.strip()
            if stripped.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if stripped.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                if not stripped:
                    continue
                cells = next(csv.reader([line], delimiter="\t"))
                if header_cells is None:
                    header_cells = cells
                else:
                    rows.append(cells)
            elif stripped.startswith("!Sample_"):
                sample_lines.append(line)
    except EngineError:
        raise
    except Exception as error:  # noqa: BLE001
        raise _err("bad_request", f"series matrix 读取失败（{Path(path).name}）：{error}")
    if header_cells is None or not rows:
        raise _err("bad_request", f"series matrix 里没有表格段：{Path(path).name}")
    table_samples = _dedupe([str(cell).strip() for cell in header_cells[1:]], "sample")
    n_samples = len(table_samples)
    feature_ids: list[str] = []
    values_list: list[list[float]] = []
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        feature_ids.append(str(row[0]).strip().strip('"'))
        values_list.append([_to_float(cell) for cell in row[1:1 + n_samples]])
    if not feature_ids:
        raise _err("bad_request", f"series matrix 里没有特征行：{Path(path).name}")
    values = np.full((len(feature_ids), n_samples), np.nan, dtype=np.float64)
    for row_index, row_values in enumerate(values_list):
        for column_index, value in enumerate(row_values[:n_samples]):
            values[row_index, column_index] = value
    if np.isnan(values).all():
        raise _err("bad_request", f"series matrix 表格里没有可解析的数值：{Path(path).name}")

    # ---- !Sample_* 元数据 ----
    records: list[tuple[str, list[str]]] = []
    for line in sample_lines:
        try:
            parts = next(csv.reader([line], delimiter="\t"))
        except Exception:  # noqa: BLE001
            continue
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        if key.startswith("!Sample_"):
            key = key[len("!Sample_"):]
        vals = [str(item).strip().strip('"') for item in parts[1:]]
        if len(vals) < n_samples:
            vals += [""] * (n_samples - len(vals))
        records.append((key, vals[:n_samples]))
    info_data: dict[str, list] = {}
    used: set[str] = set()

    def _unique(base: str) -> str:
        name = base
        counter = 1
        while name in used:
            counter += 1
            name = f"{base}_{counter}"
        used.add(name)
        return name

    organism = ""
    for key, vals in records:
        if key == "geo_accession":
            info_data[_unique("geo_accession")] = vals
            continue
        if key == "title":
            info_data[_unique("title")] = vals
            continue
        if key == "characteristics_ch1":
            prefix = ""
            if vals and ":" in vals[0]:
                prefix = vals[0].split(":", 1)[0].strip()
            if prefix and all((":" in value) or value == "" for value in vals):
                info_data[_unique(_norm_key(prefix) or "characteristics")] = [
                    value.split(":", 1)[1].strip() if ":" in value else "" for value in vals]
            else:
                info_data[_unique("characteristics_ch1")] = vals
            continue
        if key in ("organism_ch1", "organism_ch2"):
            organism = organism or next((value for value in vals if value), "")
        if key in ("source_name_ch1", "molecule_ch1", "platform_id", "data_processing"):
            info_data[_unique(key)] = vals
    sample_info = pd.DataFrame(index=pd.Index(table_samples, name="sample"))
    for column, vals in info_data.items():
        sample_info[column] = (vals + [""] * n_samples)[:n_samples]

    feature_info = _empty_feature_frame(feature_ids)
    note_ids = [str(item) for item in feature_ids[:5]]
    notes.append("特征 id 直接取 series matrix 的第一列（通常为探针/基因/代谢物 ID）："
                 + "、".join(note_ids) + ("…" if len(feature_ids) > 5 else ""))
    if not any(_is_numeric_text(item) and float(item) > 20 for item in note_ids):
        warnings.append("series matrix 不含 m/z 与 RT 列，加合物/分子式/MS2 鉴定类分析不可用")
    meta = {
        "species": organism or "unknown",
        "series": {"n_samples": n_samples, "n_features": len(feature_ids)},
        "polarity": None,
    }
    return _payload("geo_series", source, values.T, feature_ids, table_samples,
                    feature_info=feature_info, sample_info=sample_info,
                    notes=notes, warnings=warnings, meta=meta)


# ---------------------------------------------------------------------------
# xlsx
# ---------------------------------------------------------------------------


def _open_workbook(path):
    try:
        import openpyxl
    except Exception:  # noqa: BLE001
        raise _err("unsupported", "缺少 openpyxl，无法读取 Excel；请另存为 CSV/TSV 后导入")
    try:
        return openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as error:  # noqa: BLE001
        raise _err("bad_request", f"无法读取 Excel（{Path(path).name}）：{error}")


def _grid_to_frame(rows: list[list], notes: list[str], warnings: list[str]) -> pd.DataFrame:
    """把二维单元格网格转成 DataFrame（表头/行名自动识别）。"""
    rows = [list(row) for row in rows if any(str(cell).strip() for cell in row)]
    if len(rows) < 2:
        raise _err("bad_request", "Excel/表格里没有至少「表头 + 1 行数据」")
    width = max(len(row) for row in rows)
    padded = [list(row) + [""] * (width - len(row)) for row in rows]
    header = _detect_header([str(cell) for cell in padded[0]])
    first = [str(cell).strip() for cell in padded[0]]
    body = padded[1:] if header else padded
    has_index = False
    if body:
        non_numeric = sum(1 for row in body[:20] if row and str(row[0]).strip()
                          and not _is_numeric_text(row[0]))
        has_index = non_numeric >= max(1, int(0.6 * min(len(body), 20)))
    if not has_index and header and _norm_key(first[0]) in _INDEX_COLUMN_NAMES:
        has_index = True
    if header:
        columns = _dedupe(first[:width], "col")
    else:
        columns = [f"col_{i + 1}" for i in range(width)]
    frame = pd.DataFrame(body, columns=columns, dtype=object)
    if has_index:
        frame = frame.set_index(frame.columns[0])
    frame.index = [str(item) for item in frame.index]
    frame.columns = [str(item) for item in frame.columns]
    return frame


def _usable_sheet_rows(workbook, name: str) -> list[list]:
    worksheet = workbook[name]
    rows: list[list] = []
    for row in worksheet.iter_rows(values_only=True):
        if row is None:
            continue
        if all(cell is None or str(cell).strip() == "" for cell in row):
            continue
        rows.append(["" if cell is None else cell for cell in row])
    return rows


def _read_xlsx_frame(path, options: dict, notes: list[str], warnings: list[str]) -> pd.DataFrame:
    """读取 Excel；多个可用 sheet 且未指定 sheet → ambiguous + candidates。"""
    options = options or {}
    workbook = _open_workbook(path)
    try:
        names = list(workbook.sheetnames)
        if not names:
            raise _err("bad_request", f"Excel 里没有工作表：{Path(path).name}")
        requested = options.get("sheet")
        if requested is not None and str(requested) != "":
            chosen = None
            text = str(requested).strip()
            if isinstance(requested, (int, np.integer)) or (text.isdigit() and text not in names):
                index = int(requested)
                if not 0 <= index < len(names):
                    raise _err("not_found", f"sheet 序号越界：{index}（共 {len(names)} 个）")
                chosen = names[index]
            elif text in names:
                chosen = text
            else:
                lowered = {name.lower(): name for name in names}
                chosen = lowered.get(text.lower())
            if chosen is None:
                raise _err("not_found", f"工作表不存在：{requested}（可用：{', '.join(names[:10])}）")
            rows = _usable_sheet_rows(workbook, chosen)
            notes.append(f"Excel：{Path(path).name}，读取 sheet「{chosen}」")
            return _grid_to_frame(rows, notes, warnings)
        candidates: list[dict] = []
        for name in names:
            rows = _usable_sheet_rows(workbook, name)
            if len(rows) < 2 or len(rows[0]) < 2:
                continue
            numeric = sum(1 for row in rows[1:6] for cell in row[1:6] if _is_numeric_text(cell))
            if numeric < 2:
                continue
            candidates.append({"name": name, "rows": len(rows), "cols": len(rows[0]), "kind": "sheet"})
        if not candidates:
            raise _err("bad_request", f"Excel 里没有可用的二维数值表：{Path(path).name}")
        if len(candidates) > 1:
            raise _err(
                "ambiguous",
                f"Excel 里有 {len(candidates)} 个可用 sheet，请用 sheet 参数指定："
                + "、".join(item["name"] for item in candidates[:8]),
                candidates=candidates,
            )
        chosen = candidates[0]["name"]
        rows = _usable_sheet_rows(workbook, chosen)
        notes.append(f"Excel：{Path(path).name}，读取 sheet「{chosen}」")
        return _grid_to_frame(rows, notes, warnings)
    finally:
        try:
            workbook.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# mzTab
# ---------------------------------------------------------------------------

_ABUNDANCE_RE = re.compile(r"^(?:protein_|peptide_|small_molecule_|smf_)?abundance_(?:assay|study_variable)"
                           r"\[(\d+)\]$", re.IGNORECASE)


def _parse_mztab(path, source, options) -> dict:
    """解析 mzTab：MTD 元数据 + 数据段（PRT/PEP/SML/SEP）→ 样本×特征。"""
    notes = ["mzTab：解析 MTD 元数据与数据段（按 abundance_assay 列建矩阵）"]
    warnings: list[str] = []
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    try:
        for line in _iter_lines(path):
            if not line.strip():
                continue
            cells = line.split("\t")
            tag = cells[0].strip()
            if tag == "MTD" and len(cells) >= 3:
                metadata[cells[1].strip()] = cells[2].strip()
            elif tag in ("PRH", "PEH", "PSH", "SMH", "SEH", "OTH"):
                if header is None:
                    header = [cell.strip() for cell in cells[1:]]
            elif tag in ("PRT", "PEP", "PSM", "SML", "SEP"):
                if header is not None:
                    rows.append([cell.strip() for cell in cells[1:]])
    except EngineError:
        raise
    except Exception as error:  # noqa: BLE001
        raise _err("bad_request", f"mzTab 读取失败（{Path(path).name}）：{error}")
    if header is None or not rows:
        raise _err("bad_request", f"mzTab 里没有数据段（需要 PRH/PEH/SMH/SEH 头 + 数据行）：{Path(path).name}")
    abundance = [(index, match.group(1)) for index, column in enumerate(header)
                 for match in [_ABUNDANCE_RE.match(column)] if match]
    if not abundance:
        raise _err("unsupported",
                   "mzTab 里没有 abundance_assay[..] 定量列（可能是鉴定-only 文件）；"
                   "请用带定量列的 mzTab 或改用特征表导入")
    header_index = {_norm_key(column): index for index, column in enumerate(header)}

    def _column(*aliases) -> int | None:
        for alias in aliases:
            key = _norm_key(alias)
            if key in header_index:
                return header_index[key]
        return None

    mz_col = _column("mass_to_charge", "mz", "exp_mass_to_charge", "calc_mass_to_charge")
    rt_col = _column("retention_time", "rt")
    name_col = _column("name", "description", "chemical_name")
    formula_col = _column("chemical_formula", "formula", "empirical_formula")
    adduct_col = _column("adduct_ion", "adduct", "charge")
    db_col = _column("database_identifier", "identifier", "accession")
    id_col = _column("smf_id", "sml_id", "prt_id", "psm_id", "pep_id", "protein_accession")

    # 样本名：优先 study_variable/assay 描述，否则 assay_i
    sample_names: list[str] = []
    for _, assay_number in abundance:
        name = metadata.get(f"assay[{assay_number}]-name") or metadata.get(f"sample[{assay_number}]-name")
        if not name:
            name = f"assay_{assay_number}"
        sample_names.append(name)
    n_features = len(rows)
    values = np.full((n_features, len(abundance)), np.nan, dtype=np.float64)
    feature_ids: list[str] = []
    mz_values: list[float] = []
    rt_values: list[float] = []
    name_values: list[str] = []
    formula_values: list[str] = []
    adduct_values: list[str] = []
    hmdb_values: list[str] = []
    for row_index, row in enumerate(rows):
        def cell(index):
            return row[index].strip() if index is not None and index < len(row) else ""
        mz_values.append(_to_float(cell(mz_col)))
        rt_values.append(_to_float(cell(rt_col)))
        name_values.append(cell(name_col) or cell(db_col))
        formula_values.append(cell(formula_col))
        adduct_values.append(cell(adduct_col))
        hmdb_values.append(cell(db_col))
        fallback = cell(id_col) or f"feature_{row_index + 1}"
        feature_ids.append(fallback)
        for j, (column_index, _) in enumerate(abundance):
            values[row_index, j] = _to_float(row[column_index]) if column_index < len(row) else float("nan")
    if np.isnan(values).all():
        raise _err("bad_request", f"mzTab 定量列里没有可解析数值：{Path(path).name}")
    # 与矩阵导入一致的特征 id 规则：mz+rt → mz{:.4f}_rt{:.2f}；否则 name；否则原 id
    for index in range(n_features):
        mz, rt, name = mz_values[index], rt_values[index], name_values[index]
        if not math.isnan(mz) and not math.isnan(rt):
            feature_ids[index] = f"mz{mz:.4f}_rt{rt:.2f}"
        elif name:
            feature_ids[index] = str(name)
    features = _dedupe(feature_ids, "feature")
    feature_info = _feature_frame(features, {
        "mz": mz_values, "rt": rt_values, "name": name_values, "formula": formula_values,
        "adduct": adduct_values, "hmdb": hmdb_values,
    })
    _backfill_class(feature_info, warnings)
    species = metadata.get("species[1]-name") or metadata.get("species[1]-taxid") or "unknown"
    meta = {
        "species": species,
        "mztab_version": metadata.get("mzTab-version", ""),
        "mztab_mode": metadata.get("mzTab-mode", ""),
        "mztab_type": metadata.get("mzTab-type", ""),
        "polarity": _polarity_of(feature_info),
        "n_metadata": len(metadata),
    }
    notes.append(f"mzTab：{len(metadata)} 条 MTD 元数据；{len(abundance)} 个定量 assay；"
                 f"{n_features} 个特征")
    return _payload("mztab", source, values.T, features, sample_names,
                    feature_info=feature_info, notes=notes, warnings=warnings, meta=meta)


# ---------------------------------------------------------------------------
# ISA-Tab（MetaboLights）
# ---------------------------------------------------------------------------

#: ISA-Tab 里只描述代谢物（不是样本强度）的列，导入时忽略
_ISA_ONLY_COLUMNS = frozenset({"smiles", "inchi", "uri", "taxid", "species", "fragmentation",
                               "reliability", "charge", "modifications", "database_identifier"})

_ISA_META_ALIASES = {
    "database_identifier": "hmdb", "metabolite_identification": "name",
    "chemical_formula": "formula", "empirical_formula": "formula",
    "mass_to_charge": "mz", "retention_time": "rt",
}


def _isa_find(folder: Path, prefix: str) -> list[Path]:
    found: list[Path] = []
    for pattern in (f"{prefix}_*.txt", f"{prefix}_*.tsv", f"{prefix}_*.csv"):
        found.extend(sorted(folder.glob(pattern)))
    return [item for item in found if item.is_file()]


def _read_isa_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """读 ISA-Tab 的制表符表格 → (header, rows)。"""
    header: list[str] = []
    rows: list[list[str]] = []
    with _open_text(path) as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            cells = [cell.strip().strip('"') for cell in line.split("\t")]
            if not header:
                header = cells
            else:
                rows.append(cells)
    return header, rows


def _isa_sample_info(sources: list[Path], samples: list[str]) -> pd.DataFrame:
    """把 s_*.txt / a_*.txt 的表型合并成样本表（按 Sample Name 对齐）。"""
    frame = pd.DataFrame(index=pd.Index([str(item) for item in samples], name="sample"))
    sample_lower = {str(item).strip().lower(): str(item) for item in samples}
    for path in sources:
        try:
            header, rows = _read_isa_table(path)
        except EngineError:
            continue
        if not header or not rows:
            continue
        name_index = None
        for index, column in enumerate(header):
            key = _norm_key(column)
            if key in ("sample_name", "sample name", "sample", "sample_id"):
                name_index = index
                break
        if name_index is None:
            continue
        prefix = path.name.split("_", 1)[0].lower()
        for column_index, column in enumerate(header):
            if column_index == name_index:
                continue
            key = _norm_key(column)
            if not key or key in ("source_name", "raw_spectral_data_file", "derived_spectral_data_file",
                                  "raw_data_file", "derived_data_file", "parameter_file_name"):
                continue
            target = key
            if target in frame.columns:
                target = f"{key}_{prefix}"
                suffix = 2
                while target in frame.columns:
                    target = f"{key}_{prefix}{suffix}"
                    suffix += 1
            values: list = [np.nan] * len(frame)
            for row in rows:
                if column_index >= len(row):
                    continue
                sample_key = str(row[name_index]).strip().lower() if name_index < len(row) else ""
                hit = sample_lower.get(sample_key)
                if hit is None:
                    continue
                value = row[column_index].strip()
                if value:
                    values[list(frame.index).index(hit)] = value
            frame[target] = values
    return frame


def _parse_isa_tab(path, source, options) -> dict:
    """解析 MetaboLights ISA-Tab：m_* 矩阵 + s_*/a_* 表型。"""
    notes: list[str] = []
    warnings: list[str] = []
    path = Path(path)
    folder = path.parent if path.is_file() else path
    if path.is_file() and not path.name.lower().startswith("m_"):
        # 允许直接给 s_/a_/i_ 文件：自动找同目录的 m_ 矩阵
        candidates_m = _isa_find(folder, "m")
        if not candidates_m:
            raise _err("bad_request",
                       f"ISA-Tab 需要 m_*.txt 代谢物矩阵，但目录里没有找到（{folder}）")
        path = candidates_m[0]
        notes.append(f"传入的是 {Path(source).name}，自动改用同目录的 {path.name}")
    matrix_candidates = [path] if path.name.lower().startswith("m_") else _isa_find(folder, "m")
    if not matrix_candidates:
        raise _err("bad_request", f"没有找到 ISA-Tab 的 m_*.txt 矩阵文件：{folder}")
    if options.get("member"):
        wanted = str(options["member"])
        matched = [item for item in matrix_candidates
                   if item.name == wanted or item.name.endswith(wanted)]
        if matched:
            matrix_candidates = matched
    if len(matrix_candidates) > 1 and not options.get("member"):
        raise _err("ambiguous",
                   f"目录里有 {len(matrix_candidates)} 个 m_* 矩阵文件，请用 member 指定",
                   candidates=[{"name": item.name, "size": int(item.stat().st_size),
                                "kind": "isa_tab"} for item in matrix_candidates])
    matrix_path = matrix_candidates[0]
    header, rows = _read_isa_table(matrix_path)
    if not header or not rows:
        raise _err("bad_request", f"ISA-Tab 矩阵是空的：{matrix_path.name}")
    width = max(len(row) for row in rows)
    padded = [list(row) + [""] * (width - len(row)) for row in rows]
    columns = _dedupe(header[:width], "col")
    frame = pd.DataFrame(padded, columns=columns, dtype=object)
    frame.index = [f"metabolite_{index + 1}" for index in range(len(frame))]
    # ISA 注释列别名
    rename: dict[str, str] = {}
    for column in frame.columns:
        key = _norm_key(column)
        if key in _ISA_META_ALIASES:
            rename[column] = _ISA_META_ALIASES[key]
        elif key in ("smiles", "inchi", "uri", "taxid", "species", "fragmentation", "reliability"):
            rename[column] = key
    if rename:
        frame = frame.rename(columns=rename)
        notes.append("ISA-Tab 注释列已映射：" + "、".join(sorted(set(rename.values()))))
    sample_files = _isa_find(folder, "s") + _isa_find(folder, "a")
    # ISA 里"只描述代谢物、不是样本强度"的列：直接从矩阵里剔除，避免被当成样本
    isa_only = _ISA_ONLY_COLUMNS
    drop_columns = [column for column in frame.columns
                    if _norm_key(column) in isa_only and _meta_kind(column) is None]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)
        notes.append("ISA-Tab 里只描述代谢物的列已忽略：" + "、".join(str(c) for c in drop_columns))
    meta_kinds = {_meta_kind(column) for column in frame.columns}
    data_columns = [column for column in frame.columns if _meta_kind(column) is None]
    sample_names = [str(column) for column in data_columns]
    if not sample_names:
        raise _err("bad_request", f"ISA-Tab 矩阵里没有样本列：{matrix_path.name}")
    payload = _matrix_from_frame(frame, "isa_tab", source, options, notes, warnings,
                                 meta_extra={"isa_matrix": matrix_path.name,
                                             "isa_folder": str(folder)})
    # 补 s_/a_ 表型
    sinfo = _isa_sample_info(sample_files, payload["samples"])
    if len(sinfo.columns):
        merged = payload["sample_info"].join(sinfo, how="left", rsuffix="_isa") \
            if len(payload["sample_info"].columns) else sinfo
        payload["sample_info"] = merged
        notes.append(f"已合并 {len(sample_files)} 个 ISA 表型文件的 {len(sinfo.columns)} 列")
    payload["notes"] = list(payload.get("notes") or []) + notes
    return payload


# ---------------------------------------------------------------------------
# MS2 谱库（msp / mgf）
# ---------------------------------------------------------------------------

_MSP_PRECURSOR_KEYS = ("precursormz", "precursor_mz", "precursor mz", "precursormass",
                       "precursor_mass", "pepmass", "parent", "mw", "exactmass", "exact_mass")


def _peak_pair(line: str):
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?)[\s,;:]+"
                     r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?)", str(line).strip())
    if not match:
        return None
    try:
        return [float(match.group(1)), float(match.group(2))]
    except ValueError:
        return None


def _parse_msp_text(text: str, notes: list[str], warnings: list[str]) -> list[dict]:
    spectra: list[dict] = []
    current = {"id": "", "name": "", "precursor_mz": float("nan"), "peaks": []}
    assigned = False

    def flush():
        nonlocal current, assigned
        if current["peaks"] or current["id"] or current["name"]:
            spectrum = {
                "id": str(current["id"] or current["name"] or f"spectrum_{len(spectra) + 1}"),
                "name": str(current["name"] or ""),
                "precursor_mz": float(current["precursor_mz"]),
                "peaks": current["peaks"],
            }
            spectra.append(spectrum)
        current = {"id": "", "name": "", "precursor_mz": float("nan"), "peaks": []}
        assigned = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        lowered_key = _norm_key(line.split(":", 1)[0].split("=", 1)[0]) if (":" in line or "=" in line) else ""
        if lowered_key in ("name", "title", "compound", "compound_name"):
            value = re.split(r"[:=]", line, maxsplit=1)[-1].strip().strip('"')
            current["name"] = value
            if not current["id"]:
                current["id"] = value
            continue
        if lowered_key in ("num_peaks", "numpeaks", "num peaks", "peaks"):
            continue
        if lowered_key in _MSP_PRECURSOR_KEYS:
            value = re.split(r"[:=]", line, maxsplit=1)[-1].strip().split()
            if value:
                current["precursor_mz"] = _to_float(value[0].replace(",", ""))
            continue
        pair = _peak_pair(line)
        if pair is not None:
            current["peaks"].append(pair)
    flush()
    if not spectra:
        raise _err("bad_request", "msp 里没有解析到任何谱图（需要 Name:/PrecursorMZ: 与峰表）")
    return spectra


def _parse_mgf_text(text: str, notes: list[str], warnings: list[str]) -> list[dict]:
    spectra: list[dict] = []
    current = None

    def new_spectrum():
        return {"id": "", "name": "", "precursor_mz": float("nan"), "peaks": []}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("BEGIN IONS"):
            current = new_spectrum()
            continue
        if line.upper().startswith("END IONS"):
            if current is not None:
                if not current["id"]:
                    current["id"] = current["name"] or f"spectrum_{len(spectra) + 1}"
                spectra.append(current)
                current = None
            continue
        if current is None:
            current = new_spectrum()
        if "=" in line:
            key, value = line.split("=", 1)
            key = _norm_key(key)
            value = value.strip()
            if key in ("title", "name", "compound"):
                current["name"] = value
                if not current["id"]:
                    current["id"] = value
            elif key in ("pepmass", "precursormz", "precursor_mz", "precursor"):
                current["precursor_mz"] = _to_float(value.split()[0]) if value.split() else float("nan")
            continue
        pair = _peak_pair(line)
        if pair is not None:
            current["peaks"].append(pair)
    if current is not None and current["peaks"]:
        if not current["id"]:
            current["id"] = current["name"] or f"spectrum_{len(spectra) + 1}"
        spectra.append(current)
    if not spectra:
        raise _err("bad_request", "mgf 里没有解析到任何谱图（需要 BEGIN IONS/END IONS 与峰表）")
    return spectra


def _library_payload(fmt: str, source, spectra: list[dict], notes: list[str],
                     warnings: list[str], meta_extra=None, allow_empty: bool = False) -> dict:
    """谱库文件 → 统一 payload（无定量矩阵，用 1×1 NaN 占位）。

    为什么要占位矩阵：``mb_state.Dataset`` 拒绝 0 样本/0 特征的矩阵，而谱库文件
    （msp/mgf 或只有 TIC 的 mzML）本身没有任何定量矩阵。用 1×1 的 ``np.nan`` 占位，
    既能让导入链路通过，又能明确表达"这里没有数据"；真正的谱图在 ``library``。
    """
    notes = list(notes) + [
        "文件不含定量矩阵：矩阵按 1×1 占位（np.nan），谱图/TIC 挂在 dataset.library（meta.library）",
        "可用 ms2_library_match / ms2_network 分析使用该谱库",
    ]
    if not spectra and not allow_empty:
        raise _err("bad_request", f"谱库里没有谱图：{Path(str(source)).name}")
    library = {"spectra": spectra, "path": str(source)}
    features = ["library_placeholder"]
    samples = ["library"]
    X = np.full((1, 1), np.nan, dtype=np.float32)
    feature_info = _feature_frame(features, {})
    meta = {
        "library": library,
        "n_spectra": len(spectra),
        "polarity": _spectra_polarity(spectra),
        "has_mz": False,
        "has_rt": False,
    }
    meta.update(meta_extra or {})
    return _payload(fmt, source, X, features, samples, feature_info=feature_info,
                    notes=notes, warnings=warnings, meta=meta)


def _spectra_polarity(spectra: list[dict]) -> str | None:
    for spectrum in spectra[:50]:
        name = str(spectrum.get("id") or "") + " " + str(spectrum.get("name") or "")
        lowered = name.lower()
        if "pos" in lowered:
            return "pos"
        if "neg" in lowered:
            return "neg"
    return None


# ---------------------------------------------------------------------------
# mzML / mzXML（TIC + MS2 谱图，不做峰检测）
# ---------------------------------------------------------------------------

_MZML_CV = {
    "ms_level": "MS:1000511",
    "selected_mz": "MS:1000744",
    "isolation_target": "MS:1000827",
    "tic": "MS:1000285",
    "rt": "MS:1000016",
    "mz_array": "MS:1000514",
    "intensity_array": "MS:1000515",
    "time_array": "MS:1000595",
    "float32": "MS:1000521",
    "float64": "MS:1000523",
    "zlib": "MS:1000574",
    "no_compression": "MS:1000576",
    "tic_chrom": "MS:1000235",
}


def _iter_local(element, local_name: str):
    """按本地名遍历后代元素（mzML/mzXML 带命名空间，ElementTree 的 tag 是 {ns}tag）。"""
    for child in element.iter():
        if str(child.tag).split("}")[-1] == local_name:
            yield child


def _decode_binary(element, cv: dict[str, str]) -> np.ndarray:
    """解 mzML 的 binaryDataArray（base64 + 可选 zlib，32/64 bit float）。"""
    if element is None:
        return np.zeros(0, dtype=np.float64)
    text = "".join(element.itertext()).strip()
    if not text:
        return np.zeros(0, dtype=np.float64)
    attributes: dict[str, str] = {}
    for param in _iter_local(element, "cvParam"):
        accession = param.get("accession") or ""
        attributes[accession] = param.get("value") or ""
    try:
        raw = base64.b64decode(text)
    except Exception:  # noqa: BLE001
        return np.zeros(0, dtype=np.float64)
    if _MZML_CV["zlib"] in attributes or "zlib" in str(attributes).lower():
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            return np.zeros(0, dtype=np.float64)
    precision = "f" if _MZML_CV["float32"] in attributes else "d"
    dtype = np.dtype("<f4") if precision == "f" else np.dtype("<f8")
    if len(raw) % dtype.itemsize:
        return np.zeros(0, dtype=np.float64)
    count = len(raw) // dtype.itemsize
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    return np.frombuffer(raw, dtype=dtype, count=count).astype(np.float64)


def _downsample(values: np.ndarray, limit: int) -> list[float]:
    if values.size <= limit:
        return [float(item) for item in values.tolist()]
    positions = np.linspace(0, values.size - 1, limit).astype(int)
    return [float(item) for item in values[positions].tolist()]


def _parse_mzml(path, source, options) -> dict:
    """轻量解析 mzML：TIC 曲线（meta.tic）+ MS2 谱图（meta.library）。"""
    notes = ["mzML/mzXML 只做轻量 XML 解析：提取 TIC 曲线与 MS2 谱图，不做峰检测/定量"]
    warnings: list[str] = []
    lower = Path(path).name.lower()
    if lower.endswith((".mzxml", ".mzxml.gz")):
        tic, spectra = _parse_mzxml_records(path, notes, warnings)
    else:
        tic, spectra = _parse_mzml_records(path, notes, warnings)
    meta = {"tic": tic, "library": {"spectra": spectra, "path": str(source)},
            "n_spectra": len(spectra),
            "polarity": _spectra_polarity(spectra)}
    payload = _library_payload("mzml_tic", source, spectra, notes, warnings,
                               meta_extra=meta, allow_empty=True)
    if not spectra:
        payload["warnings"].append("没有 MS2 谱图（ms level 2），只提供 TIC 曲线（meta.tic）；"
                                   "可先看 TIC 做 QC，谱图分析需要带 MS2 的 mzML 或 msp/mgf 谱库")
    return payload


def _parse_mzml_records(path, notes: list[str], warnings: list[str]):
    tic_time: list[float] = []
    tic_intensity: list[float] = []
    spectra: list[dict] = []
    try:
        for _, element in ET.iterparse(str(path), events=("end",)):
            tag = element.tag.split("}")[-1]
            if tag == "chromatogram":
                params = {param.get("accession"): (param.get("value") or "")
                          for param in _iter_local(element, "cvParam")}
                if _MZML_CV["tic_chrom"] in params:
                    arrays = list(_iter_local(element, "binaryDataArray"))
                    time_values = np.zeros(0)
                    intensity_values = np.zeros(0)
                    for array in arrays:
                        accession = {param.get("accession") for param in _iter_local(array, "cvParam")}
                        decoded = _decode_binary(array, {})
                        if _MZML_CV["time_array"] in accession:
                            time_values = decoded
                        elif _MZML_CV["intensity_array"] in accession:
                            intensity_values = decoded
                    if time_values.size and intensity_values.size:
                        size = min(time_values.size, intensity_values.size)
                        tic_time.extend(_downsample(time_values[:size], MAX_TIC_POINTS))
                        tic_intensity.extend(_downsample(intensity_values[:size], MAX_TIC_POINTS))
                element.clear()
            elif tag == "spectrum" and len(spectra) < MAX_MZML_SPECTRA:
                params = {param.get("accession"): (param.get("value") or "")
                          for param in _iter_local(element, "cvParam")}
                try:
                    level = int(float(params.get(_MZML_CV["ms_level"]) or 1))
                except (TypeError, ValueError):
                    level = 1
                if level != 2:
                    element.clear()
                    continue
                precursor = params.get(_MZML_CV["selected_mz"]) or params.get(
                    _MZML_CV["isolation_target"]) or ""
                peaks: list[list[float]] = []
                mz_array = np.zeros(0)
                intensity_array = np.zeros(0)
                for array in _iter_local(element, "binaryDataArray"):
                    accession = {param.get("accession") for param in _iter_local(array, "cvParam")}
                    decoded = _decode_binary(array, {})
                    if _MZML_CV["mz_array"] in accession:
                        mz_array = decoded
                    elif _MZML_CV["intensity_array"] in accession:
                        intensity_array = decoded
                size = min(mz_array.size, intensity_array.size)
                for index in range(min(size, 2000)):
                    peaks.append([round(float(mz_array[index]), 4), round(float(intensity_array[index]), 2)])
                spectrum_id = element.get("id") or f"spectrum_{len(spectra) + 1}"
                spectra.append({
                    "id": str(spectrum_id),
                    "name": str(spectrum_id),
                    "precursor_mz": _to_float(precursor),
                    "peaks": peaks,
                })
                element.clear()
            elif tag in ("mzML",):
                pass
    except (ET.ParseError, OSError) as error:
        raise _err("bad_request", f"mzML 解析失败（{Path(path).name}）：{error}")
    tic = {"time": tic_time, "intensity": tic_intensity, "n": len(tic_time), "unit": "minute"}
    if not spectra:
        warnings.append("没有解析到 MS2 谱图（ms level 2）；若文件只有 MS1，请用于 QC/TIC 查看")
    return tic, spectra


def _mzxml_retention(value) -> float:
    match = re.search(r"PT([0-9.]+)S", str(value or ""), re.IGNORECASE)
    if not match:
        return float("nan")
    try:
        return float(match.group(1)) / 60.0
    except ValueError:
        return float("nan")


def _parse_mzxml_records(path, notes: list[str], warnings: list[str]):
    tic_time: list[float] = []
    tic_intensity: list[float] = []
    spectra: list[dict] = []
    compressed = 0
    try:
        for _, element in ET.iterparse(str(path), events=("end",)):
            tag = element.tag.split("}")[-1]
            if tag != "scan":
                continue
            retention = _mzxml_retention(element.get("retentionTime"))
            try:
                level = int(element.get("msLevel") or 1)
            except (TypeError, ValueError):
                level = 1
            peaks_element = element.find("peaks")
            mz_values = np.zeros(0)
            intensity_values = np.zeros(0)
            if peaks_element is not None:
                if peaks_element.get("compressedLen"):
                    compressed += 1
                else:
                    text = (peaks_element.text or "").strip()
                    if text:
                        try:
                            raw = base64.b64decode(text)
                            count = len(raw) // 8
                            pairs = np.frombuffer(raw[:count * 8], dtype=">f4").reshape(count, 2)
                            mz_values = pairs[:, 0].astype(np.float64)
                            intensity_values = pairs[:, 1].astype(np.float64)
                        except Exception:  # noqa: BLE001
                            mz_values = np.zeros(0)
                            intensity_values = np.zeros(0)
            if mz_values.size:
                tic_time.append(retention)
                tic_intensity.append(float(np.nansum(intensity_values)))
            precursor = element.find("precursorMz")
            precursor_mz = float("nan")
            if precursor is not None:
                precursor_mz = _to_float((precursor.text or "").strip())
            if level == 2 and len(spectra) < MAX_MZML_SPECTRA and mz_values.size:
                peaks = [[round(float(mz), 4), round(float(intensity), 2)]
                         for mz, intensity in zip(mz_values[:2000], intensity_values[:2000])]
                spectrum_id = element.get("num") or f"scan_{len(spectra) + 1}"
                spectra.append({"id": str(spectrum_id), "name": str(spectrum_id),
                                "precursor_mz": precursor_mz, "peaks": peaks})
            element.clear()
    except (ET.ParseError, OSError) as error:
        raise _err("bad_request", f"mzXML 解析失败（{Path(path).name}）：{error}")
    if compressed:
        warnings.append(f"有 {compressed} 个 scan 使用压缩峰表（compressedLen），已跳过（不做 zlib 峰解码）")
    tic = {"time": tic_time, "intensity": tic_intensity, "n": len(tic_time), "unit": "minute"}
    if not spectra:
        warnings.append("没有解析到 MS2 谱图（msLevel=2）；若文件只有 MS1，请用于 QC/TIC 查看")
    return tic, spectra


# ---------------------------------------------------------------------------
# R 桥（rds / rdata）
# ---------------------------------------------------------------------------

_R_CANDIDATES = ("/Library/Frameworks/R.framework/Resources/bin/Rscript",
                 "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript", "/usr/bin/Rscript")

_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
inp <- args[[1]]; out_csv <- args[[2]]; out_meta <- args[[3]]
want <- if (length(args) >= 4) args[[4]] else ""
to_matrix <- function(obj) {
  if (is.matrix(obj)) return(obj)
  if (is.data.frame(obj)) return(as.matrix(obj))
  if (inherits(obj, "ExpressionSet")) {
    if (requireNamespace("Biobase", quietly = TRUE)) return(as.matrix(Biobase::exprs(obj)))
    return(NULL)
  }
  if (inherits(obj, "SummarizedExperiment")) {
    if (requireNamespace("SummarizedExperiment", quietly = TRUE)) {
      return(as.matrix(SummarizedExperiment::assay(obj)))
    }
    return(NULL)
  }
  if (is.list(obj)) {
    for (candidate in c("counts", "exprs", "data", "X", "abundance")) {
      if (!is.null(obj[[candidate]]) && (is.matrix(obj[[candidate]]) || is.data.frame(obj[[candidate]]))) {
        return(as.matrix(obj[[candidate]]))
      }
    }
  }
  if (is.numeric(obj) && is.null(dim(obj))) return(NULL)
  NULL
}
if (grepl("\\.rds$", inp, ignore.case = TRUE)) {
  obj <- readRDS(inp)
  m <- to_matrix(obj)
  if (is.null(m) && want != "") { obj <- readRDS(inp); m <- to_matrix(obj[[want]]) }
  obj_name <- if (is.null(attr(obj, "class"))) "rds" else paste(class(obj), collapse = "/")
} else {
  env <- new.env(parent = baseenv())
  load(inp, envir = env)
  objs <- ls(env)
  if (want != "" && want %in% objs) {
    m <- to_matrix(get(want, envir = env)); obj_name <- want
  } else {
    m <- NULL; obj_name <- ""
    for (name in objs) {
      candidate <- to_matrix(get(name, envir = env))
      if (!is.null(candidate)) {
        m <- candidate; obj_name <- name
        if (name == want) break
      }
    }
    if (is.null(m)) { cat("NO_MATRIX\n"); quit(status = 2) }
  }
}
if (is.null(m)) { cat("NO_MATRIX\n"); quit(status = 2) }
if (nrow(m) > 50000L) { cat("TOO_MANY_ROWS\n"); quit(status = 3) }
write.csv(m, out_csv, row.names = TRUE)
writeLines(c(sprintf("object=%s", obj_name),
             sprintf("n_rows=%d", nrow(m)),
             sprintf("n_cols=%d", ncol(m))), out_meta)
"""


def _read_r_meta(path: Path) -> dict:
    """读 R 桥写的 key=value 元信息（对象名/维度）。"""
    meta: dict = {}
    try:
        if not path.exists():
            return meta
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            text = value.strip()
            if key.strip() in ("n_rows", "n_cols"):
                try:
                    meta[key.strip()] = int(float(text))
                except ValueError:
                    continue
            else:
                meta[key.strip()] = text
    except OSError:
        return meta
    return meta


def _rscript() -> str | None:
    """找可用的 Rscript（环境变量 DSH_METAB_RSCRIPT 优先）。"""
    candidates = [os.environ.get("DSH_METAB_RSCRIPT") or ""] + list(_R_CANDIDATES)
    found = shutil.which("Rscript")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _load_r(path, options: dict, notes: list[str], warnings: list[str]) -> dict:
    """用 Rscript 桥接读取 .rds/.RData → CSV → 矩阵。"""
    rscript = _rscript()
    if not rscript:
        raise _err("unsupported",
                   "本机没有找到 Rscript，无法读取 RDS/RData：请在 R 里用 saveRDS/write.csv 转成 CSV，"
                   "或安装 R（https://cran.r-project.org）后重试")
    workdir = Path(tempfile.mkdtemp(prefix="mb-r-"))
    script_path = workdir / "mb_io.R"
    out_csv = workdir / "matrix.csv"
    out_meta = workdir / "meta.json"
    script_path.write_text(_R_SCRIPT, encoding="utf-8")
    command = [rscript, "--vanilla", str(script_path), str(path), str(out_csv), str(out_meta),
               str(options.get("object") or "")]
    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=R_TIMEOUT,
                                  check=False)
    except subprocess.TimeoutExpired:
        raise _err("timeout", f"R 读取超时（>{R_TIMEOUT}s）：{Path(path).name}")
    except OSError as error:
        raise _err("unsupported", f"无法调用 Rscript：{error}")
    stderr = (finished.stderr or "").strip()
    if finished.returncode != 0 or not out_csv.exists():
        if "NO_MATRIX" in (finished.stdout or "") or finished.returncode == 2:
            raise _err("unsupported",
                       f"R 对象里没有可用的二维矩阵/数据框：{Path(path).name}"
                       "（支持 matrix / data.frame / ExpressionSet / SummarizedExperiment）")
        raise _err("unsupported", f"R 读取失败（{Path(path).name}）：{stderr[-400:] or '未知原因'}")
    meta = _read_r_meta(out_meta)
    try:
        frame = pd.read_csv(out_csv, index_col=0)
    except Exception as error:  # noqa: BLE001
        raise _err("bad_request", f"R 导出的矩阵无法解析：{error}")
    if frame.empty:
        raise _err("bad_request", f"R 读出的矩阵是空的：{Path(path).name}")
    notes.append("R 桥：" + f"读取对象「{meta.get('object') or options.get('object') or '?'}」"
                 f"（{frame.shape[0]} 行 × {frame.shape[1]} 列）")
    if stderr:
        warnings.append("R 警告：" + stderr.splitlines()[-1][:200])
    fmt = "rdata" if str(path).lower().endswith((".rdata", ".rda")) else "rds"
    return _matrix_from_frame(frame, fmt, path, options, notes, warnings,
                              meta_extra={"r_object": meta.get("object")})


# ---------------------------------------------------------------------------
# 压缩包
# ---------------------------------------------------------------------------


_DOC_NAME_RE = re.compile(
    r"^(?:read[-_ ]?me|licen[cs]e|licence|notice|changelog|changes|notes?|filelist|manifest|"
    r"version|citation|authors?|about|description|history)(?:[._\-].*)?\.(?:txt|csv|tsv|md)$")


def _classify_member(name: str) -> str | None:
    """压缩包成员 → 格式 id；文档/不认识的返回 None。"""
    lower = str(name).replace("\\", "/").lower()
    base = os.path.basename(lower)
    if not base or base.startswith("."):
        return None
    if _DOC_NAME_RE.match(base):
        return None
    inner = base
    while inner.endswith((".gz", ".bz2")):
        inner = inner[: -3]
    if inner.endswith(ARCHIVE_SUFFIXES):
        return None
    if inner.endswith(XLSX_SUFFIXES):
        return "xlsx"
    if inner.endswith(LIBRARY_SUFFIXES):
        return "msms_library"
    if inner.endswith(MZML_SUFFIXES):
        return "mzml_tic"
    if inner.endswith(R_SUFFIXES):
        return "rdata" if inner.endswith((".rdata", ".rda")) else "rds"
    if inner.startswith("m_") and inner.endswith((".txt", ".tsv")):
        return "isa_tab"
    if inner.startswith(("s_", "a_")) and inner.endswith((".txt", ".tsv")):
        return "isa_side"
    if inner.startswith("i_") and inner.endswith((".txt", ".tsv")):
        return "isa_side"
    if inner.endswith((".txt", ".tsv", ".csv", ".tab", ".tsv2")):
        if "series_matrix" in inner:
            return "geo_series"
        return "matrix_text"
    return None


def _safe_member_target(root: Path, name: str) -> Path | None:
    """防路径穿越：绝对路径、盘符、``..`` 一律拒绝，返回落在 root 内的目标。"""
    text = str(name or "").replace("\\", "/")
    if not text or text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return None
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None
    if not parts:
        return None
    target = root.joinpath(*parts).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def _extract_archive(path: Path, dirs: dict, warnings: list[str]) -> Path:
    """把 zip/tar 解到 <workdir>/unpack/<uuid>（防穿越，限制总大小）。"""
    path = Path(path)
    name = path.name.lower()
    root = Path(tempfile.mkdtemp(prefix=f"unpack-{uuid.uuid4().hex[:8]}-", dir=str(dirs["unpack"])))
    total = 0
    skipped = 0
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    target = _safe_member_target(root, info.filename)
                    if target is None:
                        skipped += 1
                        continue
                    total += int(info.file_size)
                    if total > MAX_UNPACK_BYTES:
                        raise _err("bad_request", "压缩包解压后超过 20GB 上限，请先裁剪后重新打包")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink)
        elif name.endswith(ARCHIVE_SUFFIXES) or tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or member.issym() or member.islnk():
                        if member.issym() or member.islnk():
                            skipped += 1
                        continue
                    target = _safe_member_target(root, member.name)
                    if target is None:
                        skipped += 1
                        continue
                    total += int(member.size)
                    if total > MAX_UNPACK_BYTES:
                        raise _err("bad_request", "压缩包解压后超过 20GB 上限，请先裁剪后重新打包")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    with handle as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink)
        else:
            raise _err("unsupported", f"不是支持的压缩格式：{path.name}")
    except EngineError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError) as error:
        raise _err("bad_request", f"压缩包损坏或格式不识别（{path.name}）：{error}")
    except Exception as error:  # noqa: BLE001
        raise _err("bad_request", f"解压失败（{path.name}）：{error}")
    if skipped:
        warnings.append(f"压缩包里有 {skipped} 个成员因路径不安全（绝对路径/../符号链接）被跳过")
    return root


def _archive_candidates(root: Path, options: dict) -> list[dict]:
    """扫描解压目录，按格式把候选文件分组（ISA-Tab 的 s_/a_/i_ 归到 m_ 一组）。"""
    candidates: list[dict] = []
    isa_side: list[Path] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = str(item.relative_to(root)).replace(os.sep, "/")
        kind = _classify_member(relative)
        if kind is None:
            continue
        if kind == "isa_side":
            isa_side.append(item)
            continue
        size = int(item.stat().st_size)
        candidates.append({"name": relative, "size": size, "kind": kind,
                           "path": str(item)})
    if isa_side and not any(item["kind"] == "isa_tab" for item in candidates):
        # 只有 s_/a_ 没有 m_：让用户显式指定，避免静默失败
        warnings_candidates = [{"name": str(item.relative_to(root)).replace(os.sep, "/"),
                                "size": int(item.stat().st_size), "kind": "isa_side"}
                               for item in isa_side]
        candidates.extend(warnings_candidates)
    # 文本候选必须真的含数值单元格（排除 readme/说明文档）
    validated: list[dict] = []
    for item in candidates:
        if item["kind"] not in ("matrix_text", "geo_series"):
            validated.append(item)
            continue
        try:
            head = _read_head(item["path"], 200 * 1024)
        except EngineError:
            validated.append(item)
            continue
        if item["kind"] == "geo_series" and "!series_matrix_table_begin" in head:
            validated.append(item)
            continue
        numeric = sum(1 for token in re.split(r"[\s,;\t]+", head[:40000])
                      if _is_numeric_text(token))
        if numeric >= 4:
            validated.append(item)
    return validated


def _load_archive(path, dirs, options) -> dict:
    """解压 → 选定候选文件 → 递归解析。"""
    warnings: list[str] = []
    root = _extract_archive(Path(path), dirs, warnings)
    candidates = _archive_candidates(root, options)
    if not candidates:
        raise _err("bad_request",
                   f"压缩包里没有可识别的数据文件（特征表/mzTab/ISA-Tab/msp）：{Path(path).name}")
    member = options.get("member")
    if member:
        text = str(member).replace("\\", "/").strip()
        matched = None
        for item in candidates:
            if item["name"] == text or item["name"].endswith("/" + text) or \
                    os.path.basename(item["name"]) == os.path.basename(text):
                matched = item
                break
        if matched is None:
            raise _err("not_found", f"member={member!r} 不在压缩包里",
                       candidates=candidates[:40])
        chosen = matched
    else:
        preferred = [item for item in candidates if item["kind"] in
                     ("isa_tab", "mztab", "geo_series")]
        if len(candidates) == 1:
            chosen = candidates[0]
        elif len(preferred) == 1:
            chosen = preferred[0]
        else:
            raise _err("ambiguous",
                       f"压缩包里有 {len(candidates)} 个候选数据文件，请用 member 指定（例如 {candidates[0]['name']}）",
                       candidates=[{"name": item["name"], "size": item["size"], "kind": item["kind"]}
                                   for item in candidates[:40]])
    notes = [f"压缩包：{Path(path).name}；解压后选中 {chosen['name']}"]
    inner_path = Path(chosen["path"])
    forced = "isa_tab" if chosen["kind"] in ("isa_tab", "isa_side") else chosen["kind"]
    payload = _load_path(inner_path, dirs, {**options, "member": None}, forced_format=forced)
    payload["notes"] = notes + list(payload.get("notes") or [])
    payload["warnings"] = warnings + list(payload.get("warnings") or [])
    payload["format"] = chosen["kind"] if chosen["kind"] != "matrix_text" else payload.get("format")
    payload["meta"] = dict(payload.get("meta") or {})
    payload["meta"]["archive"] = str(path)
    payload["meta"]["archive_member"] = chosen["name"]
    if "library" in payload["meta"]:
        payload["library"] = payload["meta"]["library"]
    return payload


# ---------------------------------------------------------------------------
# 格式探测与主入口
# ---------------------------------------------------------------------------


def _strip_compression(name: str) -> str:
    lower = str(name).lower()
    while lower.endswith((".gz", ".bz2")):
        lower = lower[: -3]
    return lower


def _detect_format(path: Path, options: dict) -> str:
    """按 data_format 参数 / 扩展名 / 内容判断格式。"""
    forced = _norm_format((options or {}).get("data_format") or (options or {}).get("format"))
    if forced and forced != "auto":
        return forced
    name = path.name.lower()
    if name.endswith(ARCHIVE_SUFFIXES):
        return "archive"
    inner = _strip_compression(name)
    if inner.endswith(XLSX_SUFFIXES):
        return "xlsx"
    if inner.endswith(LIBRARY_SUFFIXES):
        return "msms_library"
    if inner.endswith(MZML_SUFFIXES):
        return "mzml_tic"
    if inner.endswith(R_SUFFIXES):
        return "rdata" if inner.endswith((".rdata", ".rda")) else "rds"
    head = ""
    if path.is_file() and path.stat().st_size > 0:
        try:
            head = _read_head(path, 2 * 1024 * 1024)
        except EngineError:
            head = ""
    if "!series_matrix_table_begin" in head:
        return "geo_series"
    first_line = next((line for line in head.splitlines() if line.strip()), "")
    if first_line.startswith("MTD\t") or first_line.startswith(("MTD ",)):
        return "mztab"
    if path.name.lower().startswith(("m_", "a_", "s_", "i_")) or "ONTOLOGY SOURCE REFERENCE" in head:
        return "isa_tab"
    if inner.endswith((".txt", ".tsv", ".csv", ".tab", ".tsv2", ".matrix", ".counts", ".maf")):
        return "matrix_text"
    if head.strip():
        if "\t" in head or "," in head or ";" in head:
            return "matrix_text"
    raise _err("unsupported",
               f"无法识别文件格式：{path.name}；请用 data_format 参数显式指定"
               f"（支持：{', '.join(SUPPORTED_FORMATS)}）")


def _download_to(url: str, dest_dir: Path, filename: str | None = None) -> Path:
    """URL → 本地文件（≤8GB，流式写盘）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    name = filename or os.path.basename(urllib.parse.unquote(parsed.path or "")) or "download.bin"
    name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:160] or "download.bin"
    target = dest_dir / f"{int(time.time())}-{name}"
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            total = int(response.headers.get("Content-Length") or 0)
            if total and total > MAX_DOWNLOAD_BYTES:
                raise _err("unsupported", f"文件过大（{total / 1024 ** 3:.1f}GB），超过单文件 8GB 上限")
            written = 0
            with open(target, "wb") as sink:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        sink.close()
                        target.unlink(missing_ok=True)
                        raise _err("unsupported", "文件超过单文件 8GB 上限，已中止下载")
                    sink.write(chunk)
    except EngineError:
        raise
    except Exception as error:  # noqa: BLE001
        raise _err("unsupported", f"下载失败（网络不可达）：{error}")
    if not target.exists() or target.stat().st_size == 0:
        raise _err("unsupported", f"下载到的文件为空：{url}")
    return target


def _resolve_source(source, dirs: dict, options: dict) -> tuple[Path, list[str]]:
    """path/url → 本地文件路径（目录则找唯一数据文件，多候选 → ambiguous）。"""
    if source is None or str(source).strip() == "":
        raise _err("bad_request", "缺少数据源：请提供 path 或 url")
    text = str(source).strip()
    notes: list[str] = []
    if _looks_like_url(text):
        target = _download_to(text, dirs["downloads"])
        notes.append(f"已从 URL 下载到本地：{target.name}")
        return target, notes
    path = Path(text).expanduser()
    try:
        path = path.resolve()
    except Exception:  # noqa: BLE001
        pass
    if not path.exists():
        hint = "（~ 展开后仍不存在）" if text.startswith("~") else ""
        raise _err("not_found", f"路径不存在：{text}{hint}")
    if path.is_dir():
        aware = [item for item in sorted(path.iterdir())
                 if item.is_file() and not item.name.startswith(".")
                 and item.stat().st_size >= 8]
        recognisable = []
        for item in aware:
            kind = _classify_member(item.name)
            if kind is None and item.stat().st_size <= 64 * 1024 ** 2:
                try:
                    if "!series_matrix_table_begin" in _read_head(item, 256 * 1024):
                        kind = "geo_series"
                except EngineError:
                    kind = None
            if kind is not None:
                recognisable.append((item, kind))
        if len(recognisable) == 1:
            notes.append(f"目录里只有 {recognisable[0][0].name}，自动选中")
            return recognisable[0][0], notes
        # ISA-Tab 目录：多个成员但属于同一套（m_/s_/a_/i_）
        isa = [item for item, kind in recognisable if kind in ("isa_tab", "isa_side")]
        if isa:
            matrices = [item for item in isa if item.name.lower().startswith("m_")]
            if matrices:
                chosen = sorted(matrices)[0]
                notes.append(f"目录识别为 ISA-Tab（{len(isa)} 个相关文件），用 {chosen.name} 作为矩阵")
                return chosen, notes
        if recognisable:
            raise _err("ambiguous",
                       f"目录里有 {len(recognisable)} 个可识别数据文件，请指定具体文件",
                       candidates=[{"name": item.name, "size": int(item.stat().st_size),
                                    "kind": kind} for item, kind in recognisable[:40]])
        raise _err("bad_request", f"目录里没有可识别的数据文件：{path}")
    return path, notes


def _load_path(path: Path, dirs: dict, options: dict, forced_format: str | None = None) -> dict:
    """按格式分派到具体解析函数。"""
    fmt = forced_format or _detect_format(path, options)
    if fmt in ("matrix_text", "geo_series"):
        notes: list[str] = []
        warnings: list[str] = []
        head = _read_head(path, 2 * 1024 * 1024)
        if not head.strip():
            raise _err("bad_request", f"文件是空的：{Path(path).name}")
        if "!series_matrix_table_begin" in head:
            return _parse_geo_series(path, path, options)
        frame = _read_matrix_frame(path, options, notes, warnings)
        return _matrix_from_frame(frame, "matrix_text", path, options, notes, warnings)
    if fmt == "xlsx":
        notes, warnings = [], []
        frame = _read_xlsx_frame(path, options, notes, warnings)
        return _matrix_from_frame(frame, "xlsx", path, options, notes, warnings)
    if fmt == "archive":
        return _load_archive(path, dirs, options)
    if fmt == "mztab":
        return _parse_mztab(path, path, options)
    if fmt == "isa_tab":
        return _parse_isa_tab(path, path, options)
    if fmt == "msms_library":
        return _load_library_file(path, options)
    if fmt == "mzml_tic":
        return _parse_mzml(path, path, options)
    if fmt in ("rds", "rdata"):
        notes, warnings = [], []
        return _load_r(path, options, notes, warnings)
    raise _err("unsupported", f"不支持的导入格式：{fmt}")


def _load_library_file(path, options) -> dict:
    notes: list[str] = []
    warnings: list[str] = []
    text = _read_head(path, 32 * 1024 * 1024)
    if not text.strip():
        raise _err("bad_request", f"谱库文件是空的：{Path(path).name}")
    inner = _strip_compression(Path(path).name.lower())
    if inner.endswith(".mgf"):
        spectra = _parse_mgf_text(text, notes, warnings)
        notes.append(f"mgf：解析到 {len(spectra)} 张谱图")
    else:
        spectra = _parse_msp_text(text, notes, warnings)
        notes.append(f"msp：解析到 {len(spectra)} 张谱图")
    return _library_payload("msms_library", path, spectra, notes, warnings)


def load_source(path, workdir=None, options=None) -> dict:
    """全格式导入主入口（DESIGN §6）：返回 样本×特征 矩阵 + 特征/样本注释。

    参数：
      * path：本地文件/目录，或 http(s)/ftp 直链（下载到 workdir/downloads 后导入）；
      * workdir：工作目录（解压/下载临时文件放这里）；
      * options：data_format / member / object / transpose / delimiter / sheet /
        sample_sheet / clinical / feature_col / mz_col / rt_col / zero_as_missing。
    """
    merged = dict(options or {})
    dirs = _workdirs(workdir)
    resolved, resolve_notes = _resolve_source(path, dirs, merged)
    payload = _load_path(resolved, dirs, merged)
    payload["notes"] = resolve_notes + list(payload.get("notes") or [])
    payload = _finalize_payload(payload, resolved, merged)
    return payload


def _finalize_payload(payload: dict, resolved: Path, options: dict) -> dict:
    """补 name/样本表 join/格式字段，保证 registry.build_dataset 可直接消费。"""
    if not isinstance(payload, dict):
        raise _err("internal", "导入结果格式错误")
    payload.setdefault("format", "unknown")
    payload["name"] = str(options.get("name") or (Path(payload.get("name") or resolved.name).stem
                                                 if resolved.name else resolved.name))
    payload.setdefault("warnings", [])
    payload.setdefault("notes", [])
    payload.setdefault("meta", {})
    warnings = list(payload.get("warnings") or [])
    notes = list(payload.get("notes") or [])
    for key, label in (("sample_sheet", "样本信息表"), ("clinical", "临床表")):
        extra_path = options.get(key)
        if not extra_path:
            continue
        try:
            frame = _read_info_table(extra_path, options)
            frame, matched, total = _join_info_table(frame, payload.get("samples") or [], label)
            payload["sample_info"] = _merge_sample_info(payload.get("sample_info"),
                                                        payload.get("samples") or [], {})
            current = payload["sample_info"]
            for column in frame.columns:
                target = str(column)
                if target in current.columns:
                    target = f"{target}_{key}"
                current[target] = frame[column].tolist()
            payload["sample_info"] = current
            payload["meta"][f"{key}_match"] = {"matched": int(matched), "total": int(total)}
            notes.append(f"已 join {label}：{Path(str(extra_path)).name}"
                         f"（匹配 {matched}/{total} 个样本）")
            if matched < total:
                warnings.append(f"{label}只匹配上 {matched}/{total} 个样本（按样本名大小写不敏感匹配）")
            if matched == 0:
                warnings.append(f"{label}与样本名完全匹配不上（已忽略该表）")
        except EngineError as error:
            warnings.append(f"{label}读取失败（{error.message}）")
    payload["notes"] = notes
    payload["warnings"] = warnings
    payload["meta"].setdefault("polarity", None)
    payload["meta"].setdefault("has_mz", False)
    payload["meta"].setdefault("has_rt", False)
    if payload.get("library"):
        payload["meta"]["library"] = payload["library"]
    elif payload["meta"].get("library"):
        payload["library"] = payload["meta"]["library"]
    try:
        payload["meta"]["size"] = int(resolved.stat().st_size)
    except OSError:
        payload["meta"]["size"] = None
    payload["meta"]["source"] = str(resolved)
    return payload


def _read_info_table(path, options: dict) -> pd.DataFrame:
    """读样本信息表/临床表（csv/tsv/xlsx，键列自动识别）。"""
    path = Path(str(path)).expanduser()
    if not path.exists():
        raise _err("not_found", f"样本信息表不存在：{path}")
    lower = path.name.lower()
    if lower.endswith(XLSX_SUFFIXES):
        workbook = _open_workbook(path)
        try:
            names = list(workbook.sheetnames)
            if not names:
                raise _err("bad_request", f"Excel 没有工作表：{path.name}")
            rows = _usable_sheet_rows(workbook, names[0])
        finally:
            try:
                workbook.close()
            except Exception:  # noqa: BLE001
                pass
        frame = _grid_to_frame(rows, [], [])
        return frame.reset_index()
    head = _read_head(path, 1024 * 1024)
    delimiter = (options or {}).get("delimiter") or _sniff_delimiter(head.splitlines())
    delimiter = _normalize_delimiter(delimiter)
    try:
        frame = pd.read_csv(str(path), sep=delimiter, low_memory=False) if delimiter != "whitespace" \
            else pd.read_csv(str(path), sep=r"\s+", engine="python")
    except Exception as error:  # noqa: BLE001
        raise _err("bad_request", f"无法读取样本信息表（{path.name}）：{error}")
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _join_info_table(frame: pd.DataFrame, samples, label: str):
    """把信息表按样本名 join 到样本顺序上（大小写不敏感、去空白）。"""
    samples = [str(item) for item in (samples or [])]
    out = pd.DataFrame(index=pd.Index(samples, name="sample"))
    lower_samples = {str(item).strip().lower(): str(item) for item in samples}
    key_column = None
    best_ratio = 0.0
    for column in frame.columns:
        values = [str(value).strip().lower() for value in frame[column].tolist()]
        if not values:
            continue
        ratio = sum(1 for value in values if value in lower_samples) / len(values)
        if ratio > best_ratio:
            best_ratio, key_column = ratio, column
    if key_column is None or best_ratio < 0.2:
        # 兜底：用行索引名匹配
        index_values = [str(item).strip().lower() for item in frame.index]
        ratio = sum(1 for value in index_values if value in lower_samples) / max(1, len(index_values))
        if ratio >= 0.2:
            for column in frame.columns:
                out[column] = [
                    frame.at[frame.index[index_values.index(str(sample).strip().lower())], column]
                    if str(sample).strip().lower() in index_values else np.nan
                    for sample in samples]
            return out, sum(1 for sample in samples if str(sample).strip().lower() in index_values), len(samples)
        return out, 0, len(samples)
    values = [str(value).strip().lower() for value in frame[key_column].tolist()]
    mapping = {}
    for position, value in enumerate(values):
        mapping.setdefault(value, position)
    matched = 0
    for column in frame.columns:
        if column == key_column:
            continue
        column_values = []
        for sample in samples:
            position = mapping.get(str(sample).strip().lower())
            if position is None:
                column_values.append(np.nan)
            else:
                column_values.append(frame.iloc[position][column])
        out[column] = column_values
    for sample in samples:
        if str(sample).strip().lower() in mapping:
            matched += 1
    return out, matched, len(samples)


# ---------------------------------------------------------------------------
# inspect（不加载全表）
# ---------------------------------------------------------------------------


def _inspect_text(path: Path, options: dict) -> dict:
    """流式统计文本矩阵维度（绝不调用 _read_matrix_frame）。"""
    notes: list[str] = []
    warnings: list[str] = []
    head = _read_head(path, 4 * 1024 * 1024)
    if not head.strip():
        raise _err("bad_request", f"文件是空的：{path.name}")
    if "!series_matrix_table_begin" in head:
        return _inspect_geo(path, head, notes, warnings)
    delimiter = (options or {}).get("delimiter") or _sniff_delimiter(head.splitlines())
    delimiter = _normalize_delimiter(delimiter)
    lines = head.splitlines()
    first_cells = _split_line(lines[0], delimiter) if lines else [""]
    header = _detect_header(first_cells)
    has_index = _detect_index(lines, delimiter, header)
    size = int(path.stat().st_size)
    row_labels: list[str] = []
    for line in (lines[1:201] if header else lines[:200]):
        cells = _split_line(line, delimiter)
        if cells and cells[0].strip():
            row_labels.append(cells[0])
    if header:
        col_labels = [str(cell) for cell in first_cells[(1 if has_index else 0):]]
    else:
        width = max(0, len(first_cells) - (1 if has_index else 0))
        col_labels = [f"col_{index + 1}" for index in range(width)]
    # 非空数值列（前若干行里至少有 1 个数字）才算数据列，注释/文本列不算
    numeric_flags = [False] * len(col_labels)
    for line in (lines[1:41] if header else lines[:40]):
        if not line.strip():
            continue
        cells = _split_line(line, delimiter)
        if has_index:
            cells = cells[1:]
        for index, cell in enumerate(cells[:len(col_labels)]):
            if _is_numeric_text(cell):
                numeric_flags[index] = True
    meta_kinds = {_meta_kind(column) for column in col_labels}
    meta_kinds.discard(None)
    data_col_count = sum(1 for index, column in enumerate(col_labels)
                         if numeric_flags[index] and _meta_kind(column) is None)
    # 流式统计"真的有数字"的数据行（排除 Note/说明之类非数值行）
    data_rows = 0
    total_lines = 0
    if size <= MAX_INSPECT_STREAM_BYTES:
        seen_first = False
        for line in _iter_lines(path):
            total_lines += 1
            if not line.strip():
                continue
            if header and not seen_first:
                seen_first = True
                continue
            seen_first = True
            cells = _split_line(line, delimiter)
            if any(_is_numeric_text(cell) for cell in cells[:80]):
                data_rows += 1
    else:
        warnings.append("文件过大（>1GB），未统计行数，n_samples/n_features 为估算值")
    if meta_kinds:
        strong = ("mz" in meta_kinds or "rt" in meta_kinds
                  or len(meta_kinds & {"name", "formula", "adduct", "hmdb", "kegg"}) >= 2)
        if strong:
            n_samples, n_features = data_col_count, data_rows
            notes.append(f"表里有特征注释列（{'、'.join(sorted(meta_kinds))}）"
                         "→ 判定为「特征×样本」（导入时会转置成 样本×特征）")
        else:
            n_samples, n_features = data_rows, data_col_count
    elif has_index and row_labels:
        transpose = _infer_orientation(row_labels, col_labels, (data_rows, data_col_count), [])
        n_samples, n_features = (data_col_count, data_rows) if transpose else (data_rows, data_col_count)
    elif col_labels and _ratio(col_labels, _FEATURE_LABEL_RE) >= 0.75:
        n_samples, n_features = data_rows, data_col_count
    else:
        n_samples, n_features = ((data_rows, data_col_count) if data_col_count >= data_rows
                                 else (data_col_count, data_rows))
    embedded_mz = any(not math.isnan(_parse_feature_id_mz_rt(label)[0])
                      for label in (col_labels or row_labels)[:200])
    has_mz = "mz" in meta_kinds or embedded_mz
    has_rt = "rt" in meta_kinds or embedded_mz
    notes.append("inspect 只读文件头 + 流式计数行数，未构建矩阵")
    notes.append(f"分隔符：{'TAB' if delimiter == chr(9) else ('空白' if delimiter == 'whitespace' else delimiter)}")
    if header:
        notes.append(f"表头列数：{len(first_cells)}（{'含' if has_index else '不含'}行名列）；"
                     f"其中数值数据列 {data_col_count}")
    return {
        "format": "matrix_text",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "has_mz": bool(has_mz),
        "has_rt": bool(has_rt),
        "size": size,
        "delimiter": delimiter,
        "candidates": None,
        "notes": notes,
        "warnings": warnings,
        "suggested_import": "matrix_text",
    }


def _inspect_geo(path: Path, head: str, notes: list[str], warnings: list[str]) -> dict:
    n_samples = n_features = 0
    header_seen = False
    in_table = False
    ids: list[str] = []
    try:
        for line in _iter_lines(path):
            stripped = line.strip()
            if stripped.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if stripped.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table and stripped:
                cells = _split_line(line, "\t")
                if not header_seen:
                    n_samples = max(0, len(cells) - 1)
                    header_seen = True
                else:
                    n_features += 1
                    if len(ids) < 50:
                        ids.append(str(cells[0]).strip().strip('"'))
    except EngineError:
        raise
    except Exception:  # noqa: BLE001
        pass
    has_mz = any(_is_numeric_text(item) and float(item) > 20 for item in ids)
    if not has_mz:
        warnings.append("series matrix 通常没有 m/z/RT 列；鉴定类分析需要另导带 mz 的特征表")
    notes.append("GEO series matrix：inspect 只流式统计行列，未构建矩阵")
    return {
        "format": "geo_series",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "has_mz": bool(has_mz),
        "has_rt": False,
        "size": int(path.stat().st_size),
        "candidates": None,
        "notes": notes,
        "warnings": warnings,
        "suggested_import": "geo_series",
    }


def _inspect_xlsx(path: Path, options: dict) -> dict:
    notes: list[str] = []
    warnings: list[str] = []
    workbook = _open_workbook(path)
    try:
        names = list(workbook.sheetnames)
        sheets = []
        for name in names:
            worksheet = workbook[name]
            sheets.append({"name": name, "rows": int(worksheet.max_row or 0),
                           "cols": int(worksheet.max_column or 0), "kind": "sheet", "size": None})
        usable = [item for item in sheets if item["rows"] >= 2 and item["cols"] >= 2]
        requested = (options or {}).get("sheet")
        chosen = None
        if requested not in (None, ""):
            text = str(requested).strip()
            for item in sheets:
                if item["name"] == text or (text.isdigit() and int(text) == sheets.index(item)):
                    chosen = item
            if chosen is None:
                raise _err("not_found", f"工作表不存在：{requested}（可用：{', '.join(names[:10])}）")
        elif len(usable) == 1:
            chosen = usable[0]
        elif usable:
            chosen = usable[0]
            warnings.append(f"有 {len(usable)} 个可用 sheet（{'、'.join(item['name'] for item in usable[:6])}），"
                            "导入时需要用 sheet 参数指定")
        else:
            raise _err("bad_request", f"Excel 里没有可用的二维数值表：{path.name}")
        all_rows = _usable_sheet_rows(workbook, chosen["name"])
        rows = all_rows[:20]
        header = _detect_header([str(cell) for cell in rows[0]]) if rows else True
        has_index = False
        if len(rows) > 1:
            non_numeric = sum(1 for row in rows[1:] if row and str(row[0]).strip()
                              and not _is_numeric_text(row[0]))
            has_index = non_numeric >= max(1, int(0.6 * (len(rows) - 1)))
        cols = [str(cell) for cell in (rows[0] if rows else [])]
        if has_index:
            cols = cols[1:]
        body = all_rows[1:] if header else all_rows
        numeric_flags = [False] * len(cols)
        for row in body[:40]:
            cells = [str(cell) for cell in row]
            if has_index:
                cells = cells[1:]
            for index, cell in enumerate(cells[:len(cols)]):
                if _is_numeric_text(cell):
                    numeric_flags[index] = True
        numeric_rows = 0
        for row in body:
            cells = [str(cell) for cell in row]
            if any(_is_numeric_text(cell) for cell in cells):
                numeric_rows += 1
        kinds = {_meta_kind(column) for column in cols}
        kinds.discard(None)
        has_mz = "mz" in kinds
        has_rt = "rt" in kinds
        strong = has_mz or has_rt or len(kinds & {"name", "formula", "adduct", "hmdb", "kegg"}) >= 2
        data_col_count = sum(1 for index, column in enumerate(cols)
                             if numeric_flags[index] and _meta_kind(column) is None)
        if strong:
            n_samples, n_features = data_col_count, numeric_rows
        else:
            n_samples, n_features = ((numeric_rows, data_col_count)
                                     if data_col_count >= numeric_rows
                                     else (data_col_count, numeric_rows))
        notes.append(f"Excel：{path.name}，sheet「{chosen['name']}」；inspect 只读维度与前 20 行")
        if len(usable) > 1:
            notes.append(f"共 {len(usable)} 个可用 sheet，导入时用 sheet 指定")
        return {
            "format": "xlsx",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "has_mz": bool(has_mz),
            "has_rt": bool(has_rt),
            "size": int(path.stat().st_size),
            "sheet": chosen["name"],
            "sheets": [item["name"] for item in sheets],
            "candidates": usable if len(usable) > 1 else None,
            "notes": notes,
            "warnings": warnings,
            "suggested_import": "xlsx",
        }
    finally:
        try:
            workbook.close()
        except Exception:  # noqa: BLE001
            pass


def _inspect_archive(path: Path) -> dict:
    name = path.name.lower()
    entries: list[dict] = []
    total = 0
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    kind = _classify_member(info.filename)
                    if kind is None:
                        continue
                    total += int(info.file_size)
                    entries.append({"name": info.filename, "size": int(info.file_size), "kind": kind})
        elif name.endswith(ARCHIVE_SUFFIXES) or tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    kind = _classify_member(member.name)
                    if kind is None:
                        continue
                    total += int(member.size)
                    entries.append({"name": member.name, "size": int(member.size), "kind": kind})
        else:
            raise _err("unsupported", f"不是支持的压缩格式：{path.name}")
    except EngineError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError) as error:
        raise _err("bad_request", f"压缩包损坏或格式不识别（{path.name}）：{error}")
    if not entries:
        raise _err("bad_request", f"压缩包里没有可识别的数据文件：{path.name}")
    order = {"isa_tab": 0, "mztab": 1, "geo_series": 2, "matrix_text": 3, "xlsx": 4,
             "msms_library": 5, "mzml_tic": 6, "rds": 7, "rdata": 8, "isa_side": 9}
    entries.sort(key=lambda item: (order.get(item["kind"], 99), item["name"]))
    notes = [f"压缩包：{path.name}，{len(entries)} 个候选数据文件（未解压）",
             "导入时会解压到 <workdir>/unpack/ 并自动选择；多候选时用 member 指定"]
    return {
        "format": "archive",
        "n_samples": None,
        "n_features": None,
        "has_mz": None,
        "has_rt": None,
        "size": int(path.stat().st_size),
        "candidates": entries[:50],
        "notes": notes,
        "warnings": [],
        "suggested_import": "archive",
    }


def _inspect_mztab(path: Path, head: str) -> dict:
    notes: list[str] = []
    warnings: list[str] = []
    n_rows = 0
    header: list[str] | None = None
    metadata = 0
    try:
        for line in _iter_lines(path):
            if not line.strip():
                continue
            cells = line.split("\t")
            tag = cells[0].strip()
            if tag == "MTD":
                metadata += 1
            elif tag in ("PRH", "PEH", "PSH", "SMH", "SEH"):
                header = [cell.strip() for cell in cells[1:]]
            elif tag in ("PRT", "PEP", "PSM", "SML", "SEP"):
                n_rows += 1
    except EngineError:
        raise
    except Exception:  # noqa: BLE001
        pass
    if header is None:
        raise _err("bad_request", f"mzTab 里没有数据段头（PRH/PEH/SMH/SEH）：{path.name}")
    abundance = [column for column in header if _ABUNDANCE_RE.match(column)]
    n_samples = len(abundance)
    kinds = {_meta_kind(column) for column in header}
    kinds.discard(None)
    notes.append(f"mzTab：{metadata} 条 MTD；inspect 只流式统计行数")
    if not abundance:
        warnings.append("没有 abundance_assay 列（可能是鉴定-only mzTab）")
    return {
        "format": "mztab",
        "n_samples": int(n_samples),
        "n_features": int(n_rows),
        "has_mz": bool("mz" in kinds or "mass_to_charge" in [c.lower() for c in header]),
        "has_rt": bool("rt" in kinds or any("retention_time" in c.lower() for c in header)),
        "size": int(path.stat().st_size),
        "candidates": None,
        "notes": notes,
        "warnings": warnings,
        "suggested_import": "mztab",
    }


def _inspect_isa(path: Path) -> dict:
    notes: list[str] = []
    warnings: list[str] = []
    folder = path.parent if path.is_file() else path
    matrix = path if path.name.lower().startswith("m_") else None
    if matrix is None:
        found = _isa_find(folder, "m")
        if not found:
            raise _err("bad_request", f"ISA-Tab 目录里没有 m_*.txt 矩阵：{folder}")
        matrix = found[0]
    header, rows = _read_isa_table(matrix)
    cols = [column for column in header
            if _meta_kind(column) is None and _norm_key(column) not in _ISA_ONLY_COLUMNS]
    kinds = {_meta_kind(column) for column in header}
    kinds.discard(None)
    notes.append(f"ISA-Tab：矩阵 {matrix.name}；inspect 只读表头与前若干行")
    return {
        "format": "isa_tab",
        "n_samples": len(cols),
        "n_features": len(rows),
        "has_mz": bool("mz" in kinds or any("mass_to_charge" in _norm_key(c) for c in header)),
        "has_rt": bool("rt" in kinds or any("retention" in _norm_key(c) for c in header)),
        "size": int(path.stat().st_size),
        "candidates": None,
        "notes": notes,
        "warnings": warnings,
        "suggested_import": "isa_tab",
    }


def _inspect_library(path: Path) -> dict:
    notes: list[str] = []
    warnings: list[str] = []
    blocks = 0
    begins = 0
    for line in _iter_lines(path):
        stripped = line.strip()
        if not stripped:
            if blocks == 0:
                continue
            blocks += 1
            continue
        if stripped.upper().startswith("BEGIN IONS"):
            begins += 1
    n_spectra = begins if begins else max(1, blocks + 1)
    notes.append(f"谱库：{path.name}，约 {n_spectra} 张谱图（inspect 只数块，不解析峰表）")
    return {
        "format": "msms_library",
        "n_samples": 1,
        "n_features": 0,
        "has_mz": False,
        "has_rt": False,
        "size": int(path.stat().st_size),
        "candidates": None,
        "notes": notes,
        "warnings": warnings,
        "n_spectra": int(n_spectra),
        "suggested_import": "msms_library",
    }


def _inspect_mzml(path: Path) -> dict:
    notes: list[str] = []
    warnings: list[str] = []
    spectra = 0
    chromatograms = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                spectra += chunk.count(b"<spectrum ")
                chromatograms += chunk.count(b"<chromatogram ")
    except OSError as error:
        raise _err("bad_request", f"无法读取文件（{path.name}）：{error}")
    size = int(path.stat().st_size)
    if size > MAX_UNPACK_BYTES:
        warnings.append("文件很大（>20GB），解析会较慢")
    notes.append(f"mzML/mzXML：约 {spectra} 个 spectrum、{chromatograms} 个 chromatogram；"
                 "只提取 TIC 与 MS2 谱图，不做峰检测")
    return {
        "format": "mzml_tic",
        "n_samples": 1,
        "n_features": 0,
        "has_mz": True,
        "has_rt": chromatograms > 0,
        "size": size,
        "candidates": None,
        "notes": notes,
        "warnings": warnings,
        "n_spectra": int(spectra),
        "suggested_import": "mzml_tic",
    }


def _inspect_r(path: Path) -> dict:
    rscript = _rscript()
    if not rscript:
        raise _err("unsupported",
                   "本机没有找到 Rscript，无法探测 RDS/RData：请先在 R 里转成 CSV，或安装 R 后重试")
    workdir = Path(tempfile.mkdtemp(prefix="mb-r-ins-"))
    script_path = workdir / "mb_io.R"
    out_csv = workdir / "matrix.csv"
    out_meta = workdir / "meta.json"
    script_path.write_text(_R_SCRIPT, encoding="utf-8")
    try:
        finished = subprocess.run([rscript, "--vanilla", str(script_path), str(path), str(out_csv),
                                   str(out_meta), ""],
                                  capture_output=True, text=True, timeout=R_TIMEOUT, check=False)
    except (subprocess.TimeoutExpired, OSError) as error:
        raise _err("unsupported", f"R 探测失败：{error}")
    if finished.returncode != 0 or not out_csv.exists():
        raise _err("unsupported",
                   f"R 对象里没有可用的二维矩阵/数据框：{path.name}（{((finished.stderr or '')).strip()[:200]}）")
    frame = pd.read_csv(out_csv, index_col=0)
    notes = [f"R 桥：{path.name}（{frame.shape[0]} 行 × {frame.shape[1]} 列，inspect 已只读一遍）"]
    return {
        "format": "rdata" if path.name.lower().endswith((".rdata", ".rda")) else "rds",
        "n_samples": int(frame.shape[0]),
        "n_features": int(frame.shape[1]),
        "has_mz": False,
        "has_rt": False,
        "size": int(path.stat().st_size),
        "candidates": None,
        "notes": notes,
        "warnings": [],
        "suggested_import": "rdata" if path.name.lower().endswith((".rdata", ".rda")) else "rds",
    }


def _inspect_url(url: str, dirs: dict, options: dict) -> dict:
    """URL inspect：文本类用 Range 只读头部；二进制类先下载再探测。"""
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(urllib.parse.unquote(parsed.path or "")) or "download"
    lower = name.lower()
    inner = _strip_compression(lower)
    if (any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)
            or inner.endswith(XLSX_SUFFIXES + LIBRARY_SUFFIXES + MZML_SUFFIXES + R_SUFFIXES)):
        target = _download_to(url, dirs["downloads"], filename=name)
        info = inspect_source(str(target), workdir=str(dirs["base"]),
                              options={**options, "data_format": "auto"})
        info["source"] = url
        info["notes"] = [f"URL 文件已下载到临时目录（{name}）后 inspect"] + list(info.get("notes") or [])
        return info
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _UA,
                                                       "Range": "bytes=0-2097151"})
        with urllib.request.urlopen(request, timeout=60) as response:
            partial = response.read(2 * 1024 * 1024)
            total = int(response.headers.get("Content-Length") or 0)
    except Exception as error:  # noqa: BLE001
        raise _err("unsupported", f"URL 不可达：{error}")
    suffix = ".txt"
    for candidate in (".tsv", ".csv", ".txt", ".gz"):
        if lower.endswith(candidate):
            suffix = candidate
            break
    temp = dirs["tmp"] / f"url-head-{uuid.uuid4().hex[:8]}{suffix}"
    try:
        temp.write_bytes(partial)
        if suffix == ".gz":
            # 截断的 gz 只能探测到内容片段：改成按文本处理
            temp = dirs["tmp"] / f"url-head-{uuid.uuid4().hex[:8]}.txt"
            raw = gzip.decompress(partial) if partial[:2] == b"\x1f\x8b" else partial
            temp.write_bytes(raw)
        info = inspect_source(str(temp), workdir=str(dirs["base"]),
                              options={**options, "data_format": "auto"})
    except EngineError:
        raise
    finally:
        pass
    info["source"] = url
    info["size"] = int(total or len(partial))
    info["notes"] = [f"URL 只读取头部 {len(partial) / 1024:.0f}KB 预检（行数可能不完整）"] + \
                    list(info.get("notes") or [])
    return info


def inspect_source(path, workdir=None, options=None) -> dict:
    """只探测格式与维度，不构建矩阵（DESIGN §2.2 inspect）。"""
    merged = dict(options or {})
    dirs = _workdirs(workdir)
    if _looks_like_url(path):
        return _inspect_url(str(path).strip(), dirs, merged)
    resolved, resolve_notes = _resolve_source(path, dirs, merged)
    fmt = _detect_format(resolved, merged)
    if fmt == "archive":
        info = _inspect_archive(resolved)
    elif fmt == "xlsx":
        info = _inspect_xlsx(resolved, merged)
    elif fmt == "geo_series":
        info = _inspect_geo(resolved, "", [], [])
    elif fmt == "matrix_text":
        info = _inspect_text(resolved, merged)
    elif fmt == "mztab":
        info = _inspect_mztab(resolved, "")
    elif fmt == "isa_tab":
        info = _inspect_isa(resolved)
    elif fmt == "msms_library":
        info = _inspect_library(resolved)
    elif fmt == "mzml_tic":
        info = _inspect_mzml(resolved)
    elif fmt in ("rds", "rdata"):
        info = _inspect_r(resolved)
    else:
        info = {"format": fmt}
    info["format"] = info.get("format") or fmt
    info["source"] = str(resolved)
    info.setdefault("candidates", None)
    info.setdefault("has_mz", None)
    info.setdefault("has_rt", None)
    info["notes"] = list(resolve_notes) + list(info.get("notes") or [])
    info.setdefault("warnings", [])
    try:
        info.setdefault("size", int(resolved.stat().st_size))
    except OSError:
        info.setdefault("size", None)
    info.setdefault("suggested_import", info["format"])
    return info


# ---------------------------------------------------------------------------
# 自检（只用于测试/手工验证）
# ---------------------------------------------------------------------------


def supported_format_ids() -> list[str]:
    return [item["id"] for item in FORMAT_INFO]


__all__ = ["SUPPORTED_FORMATS", "FORMAT_INFO", "FORMAT_ALIASES", "load_source",
           "inspect_source", "supported_format_ids", "FEATURE_COLUMNS"]
