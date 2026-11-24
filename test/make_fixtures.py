#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — 导入层测试 fixture 生成器（DESIGN §11）

生成 16 组 fixture 到 ``test/fixtures/``，并写 ``truth.json`` 记录每组 fixture 的
期望结果（形状 / 样本名 / 特征 id / 数值矩阵 / has_mz / has_rt），供
``test/py_io_test.py`` 交叉校验。

用法::

    cd plugins/dsh-metabolomics
    python3 test/make_fixtures.py               # 生成到 test/fixtures/
    python3 test/make_fixtures.py /tmp/xxx      # 生成到指定目录

fixture 清单：
  1. samples_x_features.csv      样本×特征（含缺失 NA、0、非数值行）
  2. features_x_samples.tsv      特征×样本 + mz/rt/name/formula/adduct/hmdb 注释列
  3. metabo.xlsx                 同 2 的 Excel 版（单可用 sheet）
  4. multisheet.xlsx             两个可用 sheet → 需要 sheet 参数
  5. samples_x_features.tsv.gz   同 1 的 gzip 版
  6. feature_table_mz_rt.tsv     带 mz/rt + 文本列 + 0 值（测 zero_as_missing）
  7. features.zip                单候选压缩包（data/feature_table.csv）
  8. multi_candidate.zip         两个候选表 → ambiguous；member 可指定
  9. study.mztab                 mzTab（MTD + SMH + SML，2 个 assay）
 10. isa/                        ISA-Tab 三件套（i_/s_/a_/m_）
 11. library.msp                 MS2 谱库（3 张谱图）
 12. tiny.mzML                   mzML（1 条 TIC 曲线 + 1 张 MS2）
 13. geo_series.txt              GEO series matrix（含 characteristics 表型）
 14. matrix_no_header.tsv        无表头特征×样本矩阵
 15. bad/                        坏文件（空 / 二进制 / 坏 zip / 坏 xlsx）
 16. sample_sheet.csv + clinical.csv  样本表/临床表 join 用
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "fixtures"


def _digest(values) -> str:
    """NaN 感知的矩阵指纹（测试里交叉校验用）。"""
    import hashlib

    arr = np.asarray(values, dtype=np.float64)
    filled = np.where(np.isnan(arr), -999.0, np.round(arr, 6))
    return hashlib.sha256(filled.astype(np.float32).tobytes()).hexdigest()[:16]


def _entry(path, X, samples, features, fmt, **extra) -> dict:
    arr = np.asarray(X, dtype=np.float64)
    assert arr.shape == (len(samples), len(features)), (
        f"{path}: shape {arr.shape} != ({len(samples)}, {len(features)})")
    data = {
        "file": str(path),
        "format": fmt,
        "n_samples": int(arr.shape[0]),
        "n_features": int(arr.shape[1]),
        "samples": [str(item) for item in samples],
        "features": [str(item) for item in features],
        "digest": _digest(arr),
        "n_missing": int(np.isnan(arr).sum()),
        "sum": round(float(np.nansum(arr)), 4),
        "matrix": [[None if np.isnan(value) else float(value) for value in row] for row in arr],
    }
    data.update(extra)
    return data


# ---------------------------------------------------------------------------
# 1. 样本 × 特征 CSV
# ---------------------------------------------------------------------------

SAMPLES_8 = [f"S{i}" for i in range(1, 9)]
FEATURES_LABEL = ["mz203.0526_rt1.23", "mz180.0634_rt2.10", "mz118.0863_rt0.90",
                  "mz300.1234_rt5.50", "mz450.7890_rt7.25", "mz512.3456_rt9.10"]


def _samples_by_features_matrix() -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.integers(150, 4200, size=(len(SAMPLES_8), len(FEATURES_LABEL))).astype(float)
    values[2, 3] = np.nan
    values[5, 1] = 0.0
    values[6, 4] = np.nan
    return values


def write_samples_by_features(outdir: Path) -> dict:
    values = _samples_by_features_matrix()
    frame = pd.DataFrame(values, index=SAMPLES_8, columns=FEATURES_LABEL)
    frame.index.name = "sample"
    # 末尾加一行非数值行：导入时应被丢弃并 warning
    note_row = pd.DataFrame([["text"] * len(FEATURES_LABEL)], index=["Note"],
                            columns=FEATURES_LABEL)
    frame = pd.concat([frame, note_row])
    path = outdir / "samples_x_features.csv"
    frame.to_csv(path, na_rep="NA")
    return _entry(path, values, SAMPLES_8, FEATURES_LABEL, "matrix_text",
                  has_mz=True, has_rt=True, trailing_rows_dropped=1)


# ---------------------------------------------------------------------------
# 2. 特征 × 样本 TSV（带注释列）
# ---------------------------------------------------------------------------

FEATURE_META_ROWS = [
    {"mz": 203.0526, "rt": 1.23, "name": "Glucose", "formula": "C6H12O6",
     "adduct": "[M+H]+", "hmdb": "HMDB0000122", "kegg": "C00031", "class": "Carbohydrate"},
    {"mz": 180.0634, "rt": 2.10, "name": "Fructose", "formula": "C6H12O6",
     "adduct": "[M-H]-", "hmdb": "HMDB0000660", "kegg": "C00095", "class": "Carbohydrate"},
    {"mz": 118.0863, "rt": 0.90, "name": "Valine", "formula": "C5H11NO2",
     "adduct": "[M+H]+", "hmdb": "HMDB0000883", "kegg": "C00183", "class": "Amino acid"},
    {"mz": 146.1176, "rt": 1.05, "name": "Glutamine", "formula": "C5H10N2O3",
     "adduct": "[M+H]+", "hmdb": "HMDB0000641", "kegg": "C00064", "class": "Amino acid"},
    {"mz": 408.2876, "rt": 12.40, "name": "Chenodeoxycholic acid", "formula": "C24H40O4",
     "adduct": "[M-H]-", "hmdb": "HMDB0000518", "kegg": "C02528", "class": "Bile acid"},
]


def _features_by_samples_matrix() -> np.ndarray:
    rng = np.random.default_rng(11)
    values = rng.integers(200, 5200, size=(len(FEATURE_META_ROWS), len(SAMPLES_8))).astype(float)
    values[1, 2] = np.nan
    return values


def expected_mz_rt_ids(meta_rows) -> list[str]:
    return [f"mz{float(row['mz']):.4f}_rt{float(row['rt']):.2f}" for row in meta_rows]


def write_features_by_samples(outdir: Path) -> dict:
    values = _features_by_samples_matrix()
    frame = pd.DataFrame(values, columns=SAMPLES_8)
    for key in ("mz", "rt", "name", "formula", "adduct", "hmdb", "kegg", "class"):
        frame.insert(0, key, [row[key] for row in FEATURE_META_ROWS])
    path = outdir / "features_x_samples.tsv"
    frame.to_csv(path, sep="\t", index=False, na_rep="NA")
    return _entry(path, values.T, SAMPLES_8, expected_mz_rt_ids(FEATURE_META_ROWS),
                  "matrix_text", has_mz=True, has_rt=True,
                  feature_info={"name": [row["name"] for row in FEATURE_META_ROWS],
                                "formula": [row["formula"] for row in FEATURE_META_ROWS],
                                "hmdb": [row["hmdb"] for row in FEATURE_META_ROWS],
                                "mz": [row["mz"] for row in FEATURE_META_ROWS],
                                "rt": [row["rt"] for row in FEATURE_META_ROWS]})


# ---------------------------------------------------------------------------
# 3/4. Excel
# ---------------------------------------------------------------------------


def write_xlsx(outdir: Path) -> list[dict]:
    values = _features_by_samples_matrix()
    frame = pd.DataFrame(values, columns=SAMPLES_8)
    for key in ("mz", "rt", "name", "formula", "adduct"):
        frame.insert(0, key, [row[key] for row in FEATURE_META_ROWS])
    path = outdir / "metabo.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="data", index=False)
        pd.DataFrame({"note": ["fixture", "second sheet is not usable"]}).to_excel(
            writer, sheet_name="notes", index=False)
    entries = [_entry(path, values.T, SAMPLES_8, expected_mz_rt_ids(FEATURE_META_ROWS),
                      "xlsx", has_mz=True, has_rt=True, sheet="data")]

    # 两个可用 sheet → ambiguous
    multi = outdir / "multisheet.xlsx"
    small_a = pd.DataFrame({"mz": [100.1, 200.2], "rt": [1.0, 2.0], "S1": [10, 20], "S2": [30, 40]})
    small_b = pd.DataFrame({"mz": [300.3, 400.4], "rt": [3.0, 4.0], "S1": [50, 60], "S2": [70, 80]})
    with pd.ExcelWriter(multi, engine="openpyxl") as writer:
        small_a.to_excel(writer, sheet_name="run1", index=False)
        small_b.to_excel(writer, sheet_name="run2", index=False)
    entries.append({
        "file": str(multi), "format": "xlsx", "ambiguous": True,
        "candidates": ["run1", "run2"],
        "expected_shape": (2, 2),
    })
    return entries


# ---------------------------------------------------------------------------
# 5. gzip
# ---------------------------------------------------------------------------


def write_gzip(outdir: Path) -> dict:
    values = _samples_by_features_matrix()
    frame = pd.DataFrame(values, index=SAMPLES_8, columns=FEATURES_LABEL)
    frame.index.name = "sample"
    path = outdir / "samples_x_features.tsv.gz"
    text = frame.to_csv(sep="\t", na_rep="NA")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return _entry(path, values, SAMPLES_8, FEATURES_LABEL, "matrix_text",
                  has_mz=True, has_rt=True)


# ---------------------------------------------------------------------------
# 6. 带 mz/rt 与 0 值的特征表
# ---------------------------------------------------------------------------

ZERO_FEATURES = ["mz100.0500_rt0.50", "mz150.0750_rt1.10", "mz250.1250_rt2.20",
                 "mz350.1750_rt3.30"]


def _zero_matrix() -> np.ndarray:
    rng = np.random.default_rng(23)
    values = rng.integers(500, 5000, size=(len(SAMPLES_8[:4]), len(ZERO_FEATURES))).astype(float)
    values[2, 0] = 0.0      # 转置后落在 样本0 × 特征2（测试 0 值语义）
    values[3, 0] = np.nan
    return values


def write_zero_table(outdir: Path) -> dict:
    values = _zero_matrix()
    samples = SAMPLES_8[:4]
    frame = pd.DataFrame(values, columns=samples)
    frame.insert(0, "mz", [100.05, 150.075, 250.125, 350.175])
    frame.insert(1, "rt", [0.50, 1.10, 2.20, 3.30])
    frame.insert(2, "name", ["A", "B", "C", "D"])
    frame.insert(3, "notes", ["first", "second", "third", "fourth"])  # 文本列 → 特征注释
    path = outdir / "feature_table_mz_rt.tsv"
    frame.to_csv(path, sep="\t", index=False, na_rep="NA")
    return _entry(path, values.T, samples, ZERO_FEATURES, "matrix_text",
                  has_mz=True, has_rt=True, zero_cell=(0, 2))


# ---------------------------------------------------------------------------
# 7/8. 压缩包
# ---------------------------------------------------------------------------


def write_archives(outdir: Path) -> list[dict]:
    values = _samples_by_features_matrix()[:, :3]
    features = FEATURES_LABEL[:3]
    frame = pd.DataFrame(values, index=SAMPLES_8, columns=features)
    frame.index.name = "sample"

    single = outdir / "features.zip"
    with zipfile.ZipFile(single, "w") as archive:
        archive.writestr("data/feature_table.csv", frame.to_csv(na_rep="NA"))
        archive.writestr("data/readme.txt", "fixture archive\n")
    entries = [_entry(single, values, SAMPLES_8, features, "matrix_text",
                      has_mz=True, has_rt=True, member="data/feature_table.csv")]

    multi = outdir / "multi_candidate.zip"
    frame_b = frame.iloc[:, :2].copy()
    frame_b.columns = ["mz111.1111_rt1.11", "mz222.2222_rt2.22"]
    with zipfile.ZipFile(multi, "w") as archive:
        archive.writestr("a_table.csv", frame.to_csv(na_rep="NA"))
        archive.writestr("b_table.csv", frame_b.to_csv(na_rep="NA"))
    entries.append({
        "file": str(multi), "format": "archive", "ambiguous": True,
        "candidates": ["a_table.csv", "b_table.csv"],
        "member": "b_table.csv",
        "member_features": list(frame_b.columns),
        "member_samples": list(SAMPLES_8),
    })
    return entries


# ---------------------------------------------------------------------------
# 9. mzTab
# ---------------------------------------------------------------------------

MZTAB_FEATURES = ["mz180.0634_rt1.20", "mz203.0526_rt2.00", "mz118.0863_rt0.60"]


def write_mztab(outdir: Path) -> dict:
    values = np.array([[1000.0, 2000.0], [1500.0, 2500.0], [500.0, 750.0]])
    lines = [
        "MTD\tmzTab-version\t1.0.0",
        "MTD\tmzTab-mode\tSummary",
        "MTD\tmzTab-type\tQuantification",
        "MTD\tdescription\tmetabolomics fixture",
        "MTD\tspecies[1]-name\tHomo sapiens",
        "MTD\tspecies[1]-taxid\t9606",
        "MTD\tassay[1]-name\tSample_A",
        "MTD\tassay[2]-name\tSample_B",
        "SMH\tSMF_ID\tdatabase_identifier\tchemical_formula\tmass_to_charge\tretention_time\t"
        "name\tabundance_assay[1]\tabundance_assay[2]",
        "SML\t1\tHMDB0000660\tC6H12O6\t180.0634\t1.2\tFructose\t1000\t2000",
        "SML\t2\tHMDB0000122\tC6H12O6\t203.0526\t2.0\tGlucose\t1500\t2500",
        "SML\t3\tHMDB0000883\tC5H11NO2\t118.0863\t0.6\tValine\t500\t750",
    ]
    path = outdir / "study.mztab"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _entry(path, values.T, ["Sample_A", "Sample_B"], MZTAB_FEATURES, "mztab",
                  has_mz=True, has_rt=True,
                  feature_info={"name": ["Fructose", "Glucose", "Valine"],
                                "formula": ["C6H12O6", "C6H12O6", "C5H11NO2"],
                                "hmdb": ["HMDB0000660", "HMDB0000122", "HMDB0000883"]})


# ---------------------------------------------------------------------------
# 10. ISA-Tab
# ---------------------------------------------------------------------------

ISA_FEATURES = ["mz180.0634_rt1.20", "mz203.0526_rt2.00", "mz118.0863_rt0.60",
                "mz146.1176_rt1.10"]


def write_isa(outdir: Path) -> dict:
    folder = outdir / "isa"
    folder.mkdir(parents=True, exist_ok=True)
    samples = [f"Sample_{i}" for i in range(1, 6)]
    values = np.array([
        [1200, 1500, 1800, 2100, 2400],
        [1300, 1600, 1900, 2200, 2500],
        [800, 900, 1000, 1100, 1200],
        [700, 800, 900, 1000, 1100],
    ], dtype=float)
    investigation = "\n".join([
        "ONTOLOGY SOURCE REFERENCE",
        "Term Source Name\tBTO\tCHEBI\tEFO\tMS\tMTBLS\tNCBITaxon\tOBI",
        "Term Source File\thttp://purl.obolibrary.org/obo/bto.owl\t"
        "http://purl.obolibrary.org/obo/chebi.owl\thttp://www.ebi.ac.uk/efo/efo.owl\t"
        "http://purl.obolibrary.org/obo/ms.owl\t"
        "https://github.com/EBI-Metabolights/MtblsWS-Py\t"
        "http://purl.obolibrary.org/obo/ncbitaxon.owl\t"
        "http://purl.obolibrary.org/obo/obi.owl",
        "INVESTIGATION",
        "Study Identifier\tMTBLS_DEMO",
        "Study Title\tMetabolomics fixture study",
        "Study Description\tSynthetic ISA-Tab study for import tests.",
        "Study Submission Date\t2024-01-01",
        "Study Public Release Date\t2024-01-01",
        "Study Assay Technology Type\tmass spectrometry",
        "STUDY ASSAYS",
        "Study Assay File Name\ta_DEMO.txt",
    ]) + "\n"
    (folder / "i_Investigation.txt").write_text(investigation, encoding="utf-8")

    sample_header = ["Source Name", "Sample Name", "Characteristics[Organism]",
                     "Characteristics[Gender]", "Characteristics[Age]",
                     "Factor Value[Group]"]
    sample_rows = []
    for index, sample in enumerate(samples):
        sample_rows.append([f"source_{index + 1}", sample, "Homo sapiens",
                            "Male" if index % 2 == 0 else "Female", str(30 + index),
                            "Case" if index < 3 else "Control"])
    (folder / "s_DEMO.txt").write_text(
        "\n".join(["\t".join(sample_header)] + ["\t".join(row) for row in sample_rows]) + "\n",
        encoding="utf-8")

    assay_header = ["Sample Name", "Raw Spectral Data File", "Factor Value[Group]"]
    assay_rows = [[sample, f"{sample}.mzML", "Case" if index < 3 else "Control"]
                  for index, sample in enumerate(samples)]
    (folder / "a_DEMO.txt").write_text(
        "\n".join(["\t".join(assay_header)] + ["\t".join(row) for row in assay_rows]) + "\n",
        encoding="utf-8")

    maf_header = ["database_identifier", "chemical_formula", "metabolite_identification",
                  "mass_to_charge", "retention_time", "charge", "taxid", "species"] + samples
    maf_rows = [
        ["HMDB0000660", "C6H12O6", "Fructose", "180.0634", "1.2", "1", "9606", "Homo sapiens"],
        ["HMDB0000122", "C6H12O6", "Glucose", "203.0526", "2.0", "1", "9606", "Homo sapiens"],
        ["HMDB0000883", "C5H11NO2", "Valine", "118.0863", "0.6", "1", "9606", "Homo sapiens"],
        ["HMDB0000641", "C5H10N2O3", "Glutamine", "146.1176", "1.1", "1", "9606", "Homo sapiens"],
    ]
    maf_lines = ["\t".join(maf_header)]
    for row_index, row in enumerate(maf_rows):
        values_row = [str(int(values[row_index, j])) for j in range(len(samples))]
        maf_lines.append("\t".join(row + values_row))
    maf_path = folder / "m_DEMO_metabolite_profiling_v2_maf.tsv"
    maf_path.write_text("\n".join(maf_lines) + "\n", encoding="utf-8")
    return _entry(maf_path, values.T, samples, ISA_FEATURES, "isa_tab",
                  has_mz=True, has_rt=True,
                  folder=str(folder),
                  sample_columns=["characteristics_organism", "characteristics_gender",
                                  "characteristics_age", "factor_value_group"])


# ---------------------------------------------------------------------------
# 11. MS2 谱库（msp）
# ---------------------------------------------------------------------------

MSP_SPECTRA = [
    {"name": "Glucose", "precursor_mz": 203.0526,
     "peaks": [(100.05, 500.0), (120.08, 320.0), (203.05, 1000.0)]},
    {"name": "Fructose", "precursor_mz": 203.0526,
     "peaks": [(101.06, 480.0), (121.09, 300.0), (203.05, 900.0)]},
    {"name": "Valine", "precursor_mz": 118.0863,
     "peaks": [(72.08, 800.0), (118.08, 600.0)]},
]


def write_msp(outdir: Path) -> dict:
    blocks = []
    for index, spectrum in enumerate(MSP_SPECTRA, start=1):
        lines = [f"Name: {spectrum['name']}",
                 f"PrecursorMZ: {spectrum['precursor_mz']}",
                 f"Num Peaks: {len(spectrum['peaks'])}"]
        for mz, intensity in spectrum["peaks"]:
            lines.append(f"{mz} {intensity}")
        lines.append("")
        blocks.append("\n".join(lines))
    path = outdir / "library.msp"
    path.write_text("\n".join(blocks), encoding="utf-8")
    return {
        "file": str(path),
        "format": "msms_library",
        "n_spectra": len(MSP_SPECTRA),
        "ids": [item["name"] for item in MSP_SPECTRA],
        "precursor_mz": [item["precursor_mz"] for item in MSP_SPECTRA],
        "n_peaks": [len(item["peaks"]) for item in MSP_SPECTRA],
    }


# ---------------------------------------------------------------------------
# 12. mzML（TIC + MS2）
# ---------------------------------------------------------------------------


def _encode_array(values) -> str:
    raw = zlib.compress(np.asarray(values, dtype="<f8").tobytes())
    return base64.b64encode(raw).decode("ascii")


def write_mzml(outdir: Path) -> dict:
    tic_time = [0.5, 1.5, 2.5, 3.5]
    tic_intensity = [1000.0, 2500.0, 1800.0, 900.0]
    ms2_mz = [72.0813, 118.0863]
    ms2_intensity = [800.0, 600.0]
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run id="run1">
    <spectrumList count="2">
      <spectrum id="scan=1" defaultArrayLength="3">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="40">
            <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <binary>{_encode_array([72.0813, 118.0863, 203.0526])}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="40">
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <binary>{_encode_array([100.0, 200.0, 300.0])}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>
      <spectrum id="scan=2" defaultArrayLength="2">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>
        <precursorList count="1">
          <precursor>
            <selectedIonList count="1">
              <selectedIon>
                <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="118.0863"/>
              </selectedIon>
            </selectedIonList>
          </precursor>
        </precursorList>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="32">
            <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <binary>{_encode_array(ms2_mz)}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="32">
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <binary>{_encode_array(ms2_intensity)}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>
    </spectrumList>
    <chromatogramList count="1">
      <chromatogram id="TIC" defaultArrayLength="4">
        <cvParam cvRef="MS" accession="MS:1000235" name="total ion current chromatogram" value=""/>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="48">
            <cvParam cvRef="MS" accession="MS:1000595" name="time array" value=""/>
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <binary>{_encode_array(tic_time)}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="48">
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <binary>{_encode_array(tic_intensity)}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </chromatogram>
    </chromatogramList>
  </run>
</mzML>
"""
    path = outdir / "tiny.mzML"
    path.write_text(xml, encoding="utf-8")
    return {
        "file": str(path),
        "format": "mzml_tic",
        "tic_time": tic_time,
        "tic_intensity": tic_intensity,
        "n_spectra": 1,
        "ms2_precursor": 118.0863,
        "ms2_peaks": [[72.0813, 800.0], [118.0863, 600.0]],
    }


# ---------------------------------------------------------------------------
# 13. GEO series matrix
# ---------------------------------------------------------------------------

GEO_SAMPLES = ["GSM1000001", "GSM1000002", "GSM1000003"]
GEO_FEATURES = ["FEATURE_A", "FEATURE_B", "FEATURE_C", "FEATURE_D"]


def write_geo_series(outdir: Path) -> dict:
    values = np.array([
        [10.0, 20.0, np.nan],
        [30.0, 40.0, 50.0],
        [60.0, np.nan, 80.0],
        [90.0, 100.0, 110.0],
    ])
    lines = [
        '!Series_title\t"Fixture metabolomics series"',
        '!Series_geo_accession\t"GSE999999"',
        '!Series_summary\t"Synthetic series matrix for import tests."',
        '!Series_overall_design\t"3 samples, 4 features."',
        '!Sample_title\t"case_1"\t"control_1"\t"case_2"',
        '!Sample_geo_accession\t"GSM1000001"\t"GSM1000002"\t"GSM1000003"',
        '!Sample_organism_ch1\t"Homo sapiens"\t"Homo sapiens"\t"Homo sapiens"',
        '!Sample_characteristics_ch1\t"group: case"\t"group: control"\t"group: case"',
        '!Sample_characteristics_ch1\t"age: 45"\t"age: 52"\t"age: 61"',
        "!series_matrix_table_begin",
        '"ID_REF"\t"GSM1000001"\t"GSM1000002"\t"GSM1000003"',
    ]
    for index, feature in enumerate(GEO_FEATURES):
        row = [f'"{feature}"']
        for value in values[index]:
            row.append('"NA"' if np.isnan(value) else f"{value:g}")
        lines.append("\t".join(row))
    lines.append("!series_matrix_table_end")
    path = outdir / "geo_series.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _entry(path, values.T, GEO_SAMPLES, GEO_FEATURES, "geo_series",
                  has_mz=False, has_rt=False, organism="Homo sapiens",
                  sample_columns=["group", "age", "title", "geo_accession"])


# ---------------------------------------------------------------------------
# 14. 无表头矩阵
# ---------------------------------------------------------------------------


def write_no_header(outdir: Path) -> dict:
    values = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ])
    path = outdir / "matrix_no_header.tsv"
    path.write_text("\n".join("\t".join(f"{value:g}" for value in row) for row in values) + "\n",
                    encoding="utf-8")
    return _entry(path, values.T, ["0", "1", "2"], ["0", "1", "2", "3", "4"], "matrix_text",
                  has_mz=False, has_rt=False)


# ---------------------------------------------------------------------------
# 15. 坏文件
# ---------------------------------------------------------------------------


def write_bad(outdir: Path) -> dict:
    folder = outdir / "bad"
    folder.mkdir(parents=True, exist_ok=True)
    empty = folder / "empty.csv"
    empty.write_bytes(b"")
    binary = folder / "binary.bin"
    binary.write_bytes(bytes(range(256)) * 8)
    broken_zip = folder / "broken.zip"
    broken_zip.write_bytes(b"not a real zip file" * 20)
    broken_xlsx = folder / "broken.xlsx"
    broken_xlsx.write_bytes(b"PK\x03\x04this is not really an xlsx" * 5)
    text_only = folder / "text_only.csv"
    text_only.write_text("a,b,c\nx,y,z\nfoo,bar,baz\n", encoding="utf-8")
    return {
        "empty": str(empty),
        "binary": str(binary),
        "broken_zip": str(broken_zip),
        "broken_xlsx": str(broken_xlsx),
        "text_only": str(text_only),
    }


# ---------------------------------------------------------------------------
# 16. 样本表 / 临床表
# ---------------------------------------------------------------------------


def write_sample_tables(outdir: Path) -> dict:
    sheet = outdir / "sample_sheet.csv"
    pd.DataFrame({
        "Sample": SAMPLES_8,
        "Group": ["case"] * 4 + ["control"] * 4,
        "Batch": ["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2"],
    }).to_csv(sheet, index=False)
    clinical = outdir / "clinical.csv"
    pd.DataFrame({
        "sample": SAMPLES_8[:7],  # S8 故意缺失 → 测试 matched<total
        "age": [41, 52, 63, 44, 55, 66, 37],
        "os_time": [12, 24, 36, 18, 30, 42, 9],
        "os_event": [1, 0, 1, 1, 0, 1, 0],
    }).to_csv(clinical, index=False)
    return {"sample_sheet": str(sheet), "clinical": str(clinical)}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build(outdir=None) -> dict:
    """生成全部 fixture，返回 truth 字典（并写 truth.json）。"""
    target = Path(outdir) if outdir else DEFAULT_OUT
    target.mkdir(parents=True, exist_ok=True)
    truth: dict = {"root": str(target), "cases": {}}
    cases = truth["cases"]
    cases["samples_x_features"] = write_samples_by_features(target)
    cases["features_x_samples"] = write_features_by_samples(target)
    xlsx_entries = write_xlsx(target)
    cases["xlsx"] = xlsx_entries[0]
    cases["xlsx_ambiguous"] = xlsx_entries[1]
    cases["gzip"] = write_gzip(target)
    cases["zero_table"] = write_zero_table(target)
    archive_entries = write_archives(target)
    cases["archive_single"] = archive_entries[0]
    cases["archive_multi"] = archive_entries[1]
    cases["mztab"] = write_mztab(target)
    cases["isa_tab"] = write_isa(target)
    cases["msms_library"] = write_msp(target)
    cases["mzml_tic"] = write_mzml(target)
    cases["geo_series"] = write_geo_series(target)
    cases["no_header"] = write_no_header(target)
    cases["bad"] = write_bad(target)
    cases["info_tables"] = write_sample_tables(target)
    (target / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return truth


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    outdir = argv[0] if argv else str(DEFAULT_OUT)
    truth = build(outdir)
    print(f"fixtures 已生成：{truth['root']}")
    for name, case in truth["cases"].items():
        detail = ""
        if "n_samples" in case:
            detail = f"（{case['n_samples']} 样本 × {case['n_features']} 特征，{case['format']}）"
        elif case.get("ambiguous"):
            detail = "（多候选，应报 ambiguous）"
        print(f"  - {name}{detail}")
    print("PASS fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
