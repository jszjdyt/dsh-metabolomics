#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — 数据导入层回归测试（DESIGN §6 / §11）

覆盖：
  * 8+ 种 fixture 的 load_source：X 形状 / features / samples / feature_info 与真值一致；
  * GEO series matrix / mzTab / ISA-Tab / msp 谱库 / mzML（TIC + MS2）解析；
  * 方向判定（样本×特征 / 特征×样本 / 无表头）、transpose / delimiter / feature_col 选项；
  * 缺失值语义（NA→np.nan、0 默认保留、zero_as_missing）、重名去重、文本列降级；
  * inspect 不加载全表（把整表读取函数替换成炸弹后仍能 inspect）；
  * 坏文件给可读中文错误、目录/压缩包多候选 → ambiguous；
  * 样本表 / 临床表 join（matched/total）；
  * 谱库类返回能被 mb_registry.build_dataset 接受（1×1 占位 + dataset.library）。

不联网、不依赖 R/npm，跑完打印 PASS 并 exit 0。

用法：cd plugins/dsh-metabolomics && python3 test/py_io_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lib"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mb-mpl-"))

import numpy as np  # noqa: E402

import make_fixtures  # noqa: E402
import mb_io  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []


def check(condition, label: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if condition:
        return True
    FAILURES.append(f"{label}{('：' + detail) if detail else ''}")
    print(f"  FAIL {label}" + (f"：{detail}" if detail else ""))
    return False


def check_equal(got, want, label: str) -> bool:
    return check(got == want, label, f"got={got!r} want={want!r}")


def check_close(got, want, label: str, tol: float = 1e-6) -> bool:
    try:
        return check(abs(float(got) - float(want)) <= tol, label, f"got={got!r} want={want!r}")
    except (TypeError, ValueError):
        return check(False, label, f"无法比较 got={got!r} want={want!r}")


def check_raises(fn, codes, label: str):
    """断言抛 EngineError 且 code 在 codes 内（codes 可为 str），返回异常对象。"""
    global CHECKS
    CHECKS += 1
    allowed = {codes} if isinstance(codes, str) else set(codes)
    try:
        fn()
    except mb_io.EngineError as error:
        if error.code in allowed:
            if not str(error.message).strip():
                FAILURES.append(f"{label}：错误消息为空")
                print(f"  FAIL {label}：错误消息为空")
            return error
        FAILURES.append(f"{label}：错误码 {error.code} 不在 {sorted(allowed)}（{error.message}）")
        print(f"  FAIL {label}：错误码 {error.code} 不在 {sorted(allowed)}")
        return error
    except Exception as error:  # noqa: BLE001
        FAILURES.append(f"{label}：抛了 {type(error).__name__} 而不是 EngineError：{error}")
        print(f"  FAIL {label}：抛了 {type(error).__name__}")
        return None
    FAILURES.append(f"{label}：没有抛异常")
    print(f"  FAIL {label}：没有抛异常")
    return None


def _as_float_matrix(case) -> np.ndarray:
    rows = case["matrix"]
    return np.array([[np.nan if value is None else float(value) for value in row] for row in rows],
                    dtype=np.float64)


def _matrix_matches(got, want) -> bool:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return False
    mask = np.isnan(want)
    if not np.array_equal(np.isnan(got), mask):
        return False
    return bool(np.allclose(got[~mask], want[~mask], rtol=1e-5, atol=1e-4))


def _any_close(values, target, tol=1e-3) -> bool:
    for value in values:
        try:
            if abs(float(value) - float(target)) <= tol:
                return True
        except (TypeError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# 逐 fixture 校验
# ---------------------------------------------------------------------------


def _check_case(case: dict, label: str, options=None) -> dict:
    payload = mb_io.load_source(case["file"], workdir=WORKDIR, options=options or {})
    print(f"[load] {label}")
    check_equal(payload["format"], case["format"], f"{label}: format")
    check_equal(len(payload["samples"]), case["n_samples"], f"{label}: n_samples")
    check_equal(len(payload["features"]), case["n_features"], f"{label}: n_features")
    check_equal(payload["X"].shape, (case["n_samples"], case["n_features"]), f"{label}: X 形状")
    check_equal(payload["X"].dtype, np.dtype("float32"), f"{label}: X dtype=float32")
    check_equal(payload["samples"], case["samples"], f"{label}: samples")
    check_equal(payload["features"], case["features"], f"{label}: features")
    check(_matrix_matches(payload["X"], _as_float_matrix(case)), f"{label}: X 数值一致",
          f"digest={case['digest']}")
    check_equal(int(np.isnan(payload["X"]).sum()), case["n_missing"], f"{label}: 缺失数")
    for column in ("mz", "rt", "name", "formula", "adduct", "hmdb", "kegg", "class"):
        check(column in payload["feature_info"].columns, f"{label}: feature_info 含 {column}")
    check_equal(list(map(str, payload["feature_info"].index)), case["features"],
                f"{label}: feature_info 索引与 features 对齐")
    check_equal(bool(payload["meta"].get("has_mz")), bool(case.get("has_mz", False)),
                f"{label}: has_mz")
    check_equal(bool(payload["meta"].get("has_rt")), bool(case.get("has_rt", False)),
                f"{label}: has_rt")
    return payload


def test_matrix_and_archives(truth: dict) -> None:
    print("\n== 1. matrix_text / gzip / xlsx / archive")
    cases = truth["cases"]
    payload = _check_case(cases["samples_x_features"], "samples_x_features")
    check(any("丢弃" in warning or "缺失" in warning for warning in payload["warnings"]),
          "samples_x_features: 缺失/丢行有 warning", str(payload["warnings"]))
    check_equal(payload["meta"].get("matrix_order_original"), "samples_by_features",
                "方向：样本×特征不需要转置")

    _check_case(cases["gzip"], "gzip")
    payload = _check_case(cases["features_x_samples"], "features_x_samples")
    check_equal(payload["meta"].get("matrix_order_original"), "features_by_samples",
                "方向：特征×样本已转置")
    mz = payload["feature_info"]["mz"].tolist()
    rt = payload["feature_info"]["rt"].tolist()
    check_close(mz[0], 203.0526, "features_x_samples: mz 注释")
    check_close(rt[0], 1.23, "features_x_samples: rt 注释")
    check_equal(payload["feature_info"]["name"].tolist()[0], "Glucose", "name 注释")
    check_equal(payload["feature_info"]["formula"].tolist()[0], "C6H12O6", "formula 注释")
    check_equal(payload["feature_info"]["hmdb"].tolist()[0], "HMDB0000122", "hmdb 注释")

    payload = _check_case(cases["xlsx"], "xlsx")
    check_equal(payload["meta"].get("sheet") if payload["meta"].get("sheet") else "data",
                "data", "xlsx: 读取 data sheet")

    # 多 sheet 未指定 → ambiguous + candidates
    error = check_raises(lambda: mb_io.load_source(cases["xlsx_ambiguous"]["file"],
                                                   workdir=WORKDIR),
                         "ambiguous", "xlsx 多可用 sheet → ambiguous")
    if error is not None:
        names = [item.get("name") for item in (error.details.get("candidates") or [])]
        check(set(names) >= {"run1", "run2"}, "ambiguous 带 candidates", str(names))
    payload = mb_io.load_source(cases["xlsx_ambiguous"]["file"], workdir=WORKDIR,
                                options={"sheet": "run2"})
    check_equal(payload["features"], ["mz300.3000_rt3.00", "mz400.4000_rt4.00"],
                "xlsx: 指定 sheet 后读到 run2")

    payload = _check_case(cases["archive_single"], "archive_single")
    check("压缩包" in " ".join(payload["notes"]), "archive: 有解压说明", str(payload["notes"]))

    # 多候选 → ambiguous；member 指定后可读
    error = check_raises(lambda: mb_io.load_source(cases["archive_multi"]["file"],
                                                   workdir=WORKDIR),
                         "ambiguous", "archive 多候选 → ambiguous")
    if error is not None:
        names = [item.get("name") for item in (error.details.get("candidates") or [])]
        check(any("a_table.csv" in str(name) for name in names), "archive candidates 含 a_table.csv",
              str(names))
    payload = mb_io.load_source(cases["archive_multi"]["file"], workdir=WORKDIR,
                                options={"member": "b_table.csv"})
    check_equal(payload["features"], cases["archive_multi"]["member_features"],
                "archive: member 指定后 features")
    check_equal(payload["samples"], cases["archive_multi"]["member_samples"], "archive: member 样本")


def test_option_switches(truth: dict) -> None:
    print("\n== 2. 选项：transpose / delimiter / feature_col / zero_as_missing")
    cases = truth["cases"]
    flipped = mb_io.load_source(cases["samples_x_features"]["file"], workdir=WORKDIR,
                               options={"transpose": True})
    check_equal(flipped["X"].shape, (6, 8), "transpose=True 后形状翻转")
    check_equal(flipped["samples"], cases["samples_x_features"]["features"], "transpose 后 samples=原特征")

    # 分号分隔 + 显式 delimiter
    text = "sample;mz100.0000_rt1.00;mz200.0000_rt2.00\nA;1;2\nB;3;4\nC;5;6\n"
    path = os.path.join(TMPDIR, "semicolon.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    payload = mb_io.load_source(path, workdir=WORKDIR)
    check_equal(payload["X"].shape, (3, 2), "分号分隔自动嗅探")
    check_close(payload["X"][2, 1], 6, "分号分隔数值")
    # 空白分隔
    path2 = os.path.join(TMPDIR, "whitespace.txt")
    with open(path2, "w", encoding="utf-8") as handle:
        handle.write("sample f1 f2\nA 1 2\nB 3 4\nC 5 6\n")
    payload = mb_io.load_source(path2, workdir=WORKDIR, options={"delimiter": "whitespace"})
    check_equal(payload["X"].shape, (3, 2), "空白分隔（显式 delimiter）")

    # feature_col：首列是目标 id
    path3 = os.path.join(TMPDIR, "feature_col.csv")
    with open(path3, "w", encoding="utf-8") as handle:
        handle.write("bin,S1,S2,S3\nf1,10,20,30\nf2,40,50,60\n")
    payload = mb_io.load_source(path3, workdir=WORKDIR, options={"feature_col": "bin"})
    check_equal(payload["features"], ["f1", "f2"], "feature_col 指定特征 id")
    check_equal(payload["samples"], ["S1", "S2", "S3"], "feature_col 行是特征 → 转置")

    # zero_as_missing
    zero_case = cases["zero_table"]
    payload = mb_io.load_source(zero_case["file"], workdir=WORKDIR)
    check_close(payload["X"][0, 2], 0.0, "0 默认保留（不当缺失）")
    payload = mb_io.load_source(zero_case["file"], workdir=WORKDIR,
                                options={"zero_as_missing": True})
    check(np.isnan(payload["X"][0, 2]), "zero_as_missing=true → 0 变 NaN")
    check_equal(payload["notes"] and True, True, "zero_as_missing 有 note")
    payload = mb_io.load_source(zero_case["file"], workdir=WORKDIR)
    check("notes" in payload["feature_info"].columns, "特征表里的文本列进 feature_info")


def test_structured_formats(truth: dict) -> None:
    print("\n== 3. mzTab / ISA-Tab / GEO series matrix")
    cases = truth["cases"]
    payload = _check_case(cases["mztab"], "mztab")
    check_equal(payload["samples"], ["Sample_A", "Sample_B"], "mztab: assay 名当样本名")
    check_equal(payload["feature_info"]["name"].tolist()[0], "Fructose", "mztab: name 注释")
    check_equal(payload["feature_info"]["formula"].tolist()[0], "C6H12O6", "mztab: formula 注释")
    check_equal(payload["meta"].get("mztab_version"), "1.0.0", "mztab: MTD 版本")
    check_equal(payload["meta"].get("species"), "Homo sapiens", "mztab: 物种")

    payload = _check_case(cases["isa_tab"], "isa_tab")
    check_equal(payload["samples"], [f"Sample_{i}" for i in range(1, 6)], "isa: 样本名")
    columns = list(payload["sample_info"].columns)
    check(any("group" in str(column) for column in columns), "isa: 表型里合并了 Group", str(columns))
    check(any("gender" in str(column) for column in columns), "isa: 合并了 Gender", str(columns))
    check_equal(payload["feature_info"]["name"].tolist()[0], "Fructose", "isa: 名称注释")
    check_equal(payload["feature_info"]["formula"].tolist()[0], "C6H12O6", "isa: 分子式")
    check_equal(payload["feature_info"]["hmdb"].tolist()[0], "HMDB0000660", "isa: 数据库 ID")

    payload = _check_case(cases["geo_series"], "geo_series")
    columns = list(payload["sample_info"].columns)
    check(set(columns) >= {"group", "age"}, "geo: characteristics 拆成列", str(columns))
    check_equal(payload["sample_info"]["group"].tolist(), ["case", "control", "case"],
                "geo: group 值")
    check_equal(payload["sample_info"]["title"].tolist(), ["case_1", "control_1", "case_2"],
                "geo: title 列")
    check_equal(payload["meta"].get("species"), "Homo sapiens", "geo: 物种来自 organism_ch1")

    payload = _check_case(cases["no_header"], "no_header")
    check_equal(payload["samples"], ["0", "1", "2"], "无表头：按列推断样本")


def test_library_and_mzml(truth: dict) -> None:
    print("\n== 4. MS2 谱库 / mzML（TIC + 谱图）")
    cases = truth["cases"]
    payload = mb_io.load_source(cases["msms_library"]["file"], workdir=WORKDIR)
    check_equal(payload["format"], "msms_library", "msp: format")
    spectra = (payload["meta"].get("library") or {}).get("spectra") or []
    check_equal(len(spectra), cases["msms_library"]["n_spectra"], "msp: 谱图数")
    check_equal([item["id"] for item in spectra], cases["msms_library"]["ids"], "msp: 谱图 id=Name")
    check_equal([item["precursor_mz"] for item in spectra], cases["msms_library"]["precursor_mz"],
                "msp: 前体 m/z")
    check_equal([len(item["peaks"]) for item in spectra], cases["msms_library"]["n_peaks"],
                "msp: 峰数")
    check_equal(payload["X"].shape, (1, 1), "msp: 1×1 占位矩阵")
    check_equal(payload["library"], payload["meta"].get("library"), "msp: library 顶层与 meta 一致")
    check(any("不具备定量矩阵" in note or "不含定量矩阵" in note for note in payload["notes"]),
          "msp: note 说明没有定量矩阵", str(payload["notes"]))

    payload = mb_io.load_source(cases["mzml_tic"]["file"], workdir=WORKDIR)
    check_equal(payload["format"], "mzml_tic", "mzML: format")
    tic = payload["meta"].get("tic") or {}
    check_equal(tic.get("n"), len(cases["mzml_tic"]["tic_time"]), "mzML: TIC 点数")
    check(np.allclose(tic.get("time"), cases["mzml_tic"]["tic_time"]), "mzML: TIC 时间轴")
    check(np.allclose(tic.get("intensity"), cases["mzml_tic"]["tic_intensity"]),
          "mzML: TIC 强度轴")
    spectra = (payload["meta"].get("library") or {}).get("spectra") or []
    check_equal(len(spectra), cases["mzml_tic"]["n_spectra"], "mzML: MS2 谱图数")
    if spectra:
        check_close(spectra[0].get("precursor_mz"), cases["mzml_tic"]["ms2_precursor"],
                    "mzML: 前体 m/z")
        check_equal(spectra[0].get("peaks"), cases["mzml_tic"]["ms2_peaks"], "mzML: 峰表")

    # registry 能接受谱库 payload（X 占位 1×1 + dataset.library）
    try:
        import mb_registry  # noqa: PLC0415
        import mb_state  # noqa: PLC0415

        state = mb_state.State()
        dataset = mb_registry.build_dataset(state, payload, name="lib", source=cases["mzml_tic"]["file"])
        check_equal(len(dataset.samples), 1, "registry: 谱库 dataset 只有 1 个占位样本")
        check_equal(mb_registry.n_spectra(dataset), 1, "registry: n_spectra 读到 MS2 数量")
        check(mb_registry.feature_meta(dataset).get("has_mz") is False,
              "registry: 谱库 dataset 的 has_mz=false")
    except Exception as error:  # noqa: BLE001
        check(False, "registry.build_dataset 接受谱库 payload",
              f"{type(error).__name__}: {error}")


def test_sample_joins(truth: dict) -> None:
    print("\n== 5. 样本表 / 临床表 join")
    cases = truth["cases"]
    tables = cases["info_tables"]
    payload = mb_io.load_source(cases["samples_x_features"]["file"], workdir=WORKDIR,
                                options={"sample_sheet": tables["sample_sheet"],
                                         "clinical": tables["clinical"]})
    info = payload["sample_info"]
    check("Group" in info.columns or "group" in info.columns, "join: 样本表列进入 sample_info",
          str(list(info.columns)))
    group_key = "Group" if "Group" in info.columns else "group"
    check_equal(info[group_key].tolist(),
                ["case"] * 4 + ["control"] * 4, "join: 分组值按样本对齐")
    check_equal(payload["meta"]["sample_sheet_match"]["matched"], 8, "join: 样本表 matched")
    check_equal(payload["meta"]["sample_sheet_match"]["total"], 8, "join: 样本表 total")
    check_equal(payload["meta"]["clinical_match"]["matched"], 7, "join: 临床表 matched（S8 缺失）")
    check(any("只匹配上" in warning for warning in payload["warnings"]),
          "join: 部分匹配有 warning", str(payload["warnings"]))
    check("os_time" in info.columns and "os_event" in info.columns, "join: 生存列进入 sample_info")

    # 大小写/空白不敏感
    path = os.path.join(TMPDIR, "sheet_case.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Sample,group\n  s1 ,Case\nS2,Control\n")
    payload = mb_io.load_source(cases["samples_x_features"]["file"], workdir=WORKDIR,
                                options={"sample_sheet": path})
    check_equal(payload["meta"]["sample_sheet_match"]["matched"], 2,
                "join: 样本名大小写/空白不敏感")
    check_equal(payload["sample_info"]["group"].tolist()[:2], ["Case", "Control"], "join: 值保留原样")


def test_inspect_without_loading(truth: dict) -> None:
    print("\n== 6. inspect 不加载全表")
    cases = truth["cases"]
    original_reader = mb_io._read_matrix_frame
    original_xlsx = mb_io._read_xlsx_frame
    original_extract = mb_io._extract_archive
    original_parse_geo = mb_io._parse_geo_series

    def _bomb(*args, **kwargs):
        raise AssertionError("inspect 不应该调用整表读取函数")

    mb_io._read_matrix_frame = _bomb  # type: ignore[assignment]
    mb_io._read_xlsx_frame = _bomb  # type: ignore[assignment]
    mb_io._extract_archive = _bomb  # type: ignore[assignment]
    mb_io._parse_geo_series = _bomb  # type: ignore[assignment]
    try:
        for name in ("samples_x_features", "features_x_samples", "gzip", "zero_table",
                     "no_header", "xlsx", "geo_series", "archive_single", "mztab"):
            case = cases[name]
            info = mb_io.inspect_source(case["file"], workdir=WORKDIR)
            label = f"inspect {name}"
            if str(case["file"]).lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                # 压缩包 inspect 只列候选（不解压、不算维度）
                check_equal(info["format"], "archive", f"{label}: format")
                check(len(info.get("candidates") or []) >= 1, f"{label}: 列出候选文件")
            else:
                check_equal(info["format"], case["format"], f"{label}: format")
                if case.get("n_samples") is not None and not case.get("ambiguous"):
                    check_equal(info.get("n_samples"), case["n_samples"], f"{label}: n_samples")
                    check_equal(info.get("n_features"), case["n_features"], f"{label}: n_features")
            check(isinstance(info.get("notes"), list) and info["notes"], f"{label}: 有 notes")
            check(isinstance(info.get("size"), int) and info["size"] > 0, f"{label}: size")
        info = mb_io.inspect_source(cases["xlsx_ambiguous"]["file"], workdir=WORKDIR)
        check_equal(len(info.get("candidates") or []), 2, "inspect xlsx: 列出多个可用 sheet")
        info = mb_io.inspect_source(cases["archive_single"]["file"], workdir=WORKDIR)
        check(len(info.get("candidates") or []) >= 1, "inspect archive: 列出候选文件")
        info = mb_io.inspect_source(cases["msms_library"]["file"], workdir=WORKDIR)
        check_equal(info.get("format"), "msms_library", "inspect msp: format")
        check(info.get("n_spectra", 0) >= 1, "inspect msp: 谱图数")
        info = mb_io.inspect_source(cases["mzml_tic"]["file"], workdir=WORKDIR)
        check_equal(info.get("format"), "mzml_tic", "inspect mzML: format")
        info = mb_io.inspect_source(cases["isa_tab"]["file"], workdir=WORKDIR)
        check_equal(info.get("format"), "isa_tab", "inspect ISA: format")
        check_equal(info.get("n_samples"), 5, "inspect ISA: n_samples")
    finally:
        mb_io._read_matrix_frame = original_reader  # type: ignore[assignment]
        mb_io._read_xlsx_frame = original_xlsx  # type: ignore[assignment]
        mb_io._extract_archive = original_extract  # type: ignore[assignment]
        mb_io._parse_geo_series = original_parse_geo  # type: ignore[assignment]


def test_bad_files(truth: dict) -> None:
    print("\n== 7. 坏文件 / 错误码")
    bad = truth["cases"]["bad"]
    error = check_raises(lambda: mb_io.load_source(bad["empty"], workdir=WORKDIR), "bad_request",
                         "空文件 → bad_request")
    if error is not None:
        check("空" in error.message, "空文件错误消息可读", error.message)
    check_raises(lambda: mb_io.load_source(bad["binary"], workdir=WORKDIR),
                 ("unsupported", "bad_request"), "二进制文件 → 可读错误")
    error = check_raises(lambda: mb_io.load_source(bad["broken_zip"], workdir=WORKDIR),
                         ("bad_request", "unsupported"), "坏 zip → 可读错误")
    if error is not None:
        check("压缩" in error.message or "zip" in error.message.lower(),
              "坏 zip 错误消息可读", error.message)
    check_raises(lambda: mb_io.load_source(bad["broken_xlsx"], workdir=WORKDIR),
                 ("bad_request", "unsupported"), "坏 xlsx → 可读错误")
    error = check_raises(lambda: mb_io.load_source(bad["text_only"], workdir=WORKDIR),
                         "bad_request", "全文本无数字 → bad_request")
    if error is not None:
        check("数值" in error.message, "全文本错误消息可读", error.message)
    check_raises(lambda: mb_io.load_source(os.path.join(TMPDIR, "no-such-file.csv"),
                                           workdir=WORKDIR),
                 "not_found", "不存在的文件 → not_found")
    check_raises(lambda: mb_io.load_source("", workdir=WORKDIR), "bad_request",
                 "空路径 → bad_request")
    check_raises(lambda: mb_io.inspect_source(bad["empty"], workdir=WORKDIR), "bad_request",
                 "inspect 空文件 → bad_request")

    # R 桥缺失 → unsupported（中文）
    import mb_state  # noqa: F401  - 只为确认模块可导入

    rds = os.path.join(TMPDIR, "dummy.rds")
    with open(rds, "wb") as handle:
        handle.write(b"X\n")
    original = mb_io._rscript
    mb_io._rscript = lambda: None  # type: ignore[assignment]
    try:
        error = check_raises(lambda: mb_io.load_source(rds, workdir=WORKDIR), "unsupported",
                             "无 Rscript 读 rds → unsupported")
        if error is not None:
            check("Rscript" in error.message and "CSV" in error.message,
                  "无 R 的错误消息给替代方案", error.message)
    finally:
        mb_io._rscript = original  # type: ignore[assignment]


def test_catalog_and_payload_contract(truth: dict) -> None:
    print("\n== 8. 目录契约 / payload 结构")
    check_equal(mb_io.SUPPORTED_FORMATS,
                ["auto", "matrix_text", "xlsx", "geo_series", "archive", "mztab", "isa_tab",
                 "msms_library", "mzml_tic", "rds", "rdata"], "SUPPORTED_FORMATS 逐字一致")
    ids = [item.get("id") for item in mb_io.FORMAT_INFO]
    check(set(mb_io.SUPPORTED_FORMATS) <= set(ids), "FORMAT_INFO 覆盖全部格式", str(ids))
    for item in mb_io.FORMAT_INFO:
        check(set(item) >= {"id", "label", "hint"}, f"FORMAT_INFO 元素键完整：{item.get('id')}")

    # registry.formats_catalog 能吃 FORMAT_INFO
    try:
        import mb_registry  # noqa: PLC0415

        catalog = mb_registry.formats_catalog()
        check_equal(catalog["ids"], mb_io.SUPPORTED_FORMATS, "registry.formats_catalog 用 mb_io 清单")
        check(len(catalog["formats"]) == len(mb_io.FORMAT_INFO), "registry 格式条目数一致")
    except Exception as error:  # noqa: BLE001
        check(False, "registry.formats_catalog 消费 FORMAT_INFO", f"{type(error).__name__}: {error}")

    case = truth["cases"]["samples_x_features"]
    payload = mb_io.load_source(case["file"], workdir=WORKDIR)
    for key in ("format", "X", "features", "samples", "feature_info", "sample_info",
                "notes", "warnings", "meta"):
        check(key in payload, f"payload 含键 {key}")
    check(isinstance(payload["notes"], list) and isinstance(payload["warnings"], list),
          "payload notes/warnings 是 list")
    check(isinstance(payload["meta"], dict), "payload meta 是 dict")
    check_equal(payload["meta"].get("n_samples"), 8, "meta.n_samples")
    check_equal(payload["meta"].get("n_features"), 6, "meta.n_features")
    check(payload["meta"].get("source", "").endswith(".csv"), "meta.source 指向源文件")

    # 特征 id 命名规则：mz{:.4f}_rt{:.2f}
    check(all(feature.startswith("mz") and "_rt" in feature for feature in payload["features"]),
          "特征 id 规则 mz{:.4f}_rt{:.2f}", str(payload["features"][:3]))

    # 重名去重
    path = os.path.join(TMPDIR, "dup_names.tsv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("feature,S1,S2\nGlucose,1,2\nGlucose,3,4\nGlucose,5,6\n")
    payload = mb_io.load_source(path, workdir=WORKDIR)
    check_equal(len(set(payload["features"])), 3, "重名特征 id 去重后唯一")
    check(any("重复" in warning for warning in payload["warnings"]), "重名有 warning",
          str(payload["warnings"]))


def main() -> int:
    print("dsh-metabolomics 数据导入层测试")
    print(f"fixtures 目录：{make_fixtures.DEFAULT_OUT}")
    truth = make_fixtures.build(make_fixtures.DEFAULT_OUT)
    global WORKDIR, TMPDIR
    WORKDIR = tempfile.mkdtemp(prefix="mb-io-work-")
    TMPDIR = tempfile.mkdtemp(prefix="mb-io-tmp-")
    try:
        test_matrix_and_archives(truth)
        test_option_switches(truth)
        test_structured_formats(truth)
        test_library_and_mzml(truth)
        test_sample_joins(truth)
        test_inspect_without_loading(truth)
        test_bad_files(truth)
        test_catalog_and_payload_contract(truth)
    finally:
        shutil.rmtree(WORKDIR, ignore_errors=True)
        shutil.rmtree(TMPDIR, ignore_errors=True)
    if FAILURES:
        print(f"\nFAIL {len(FAILURES)} / {CHECKS} checks")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print(f"\nPASS {CHECKS} checks")
    return 0


WORKDIR = tempfile.mkdtemp(prefix="mb-io-work-")
TMPDIR = tempfile.mkdtemp(prefix="mb-io-tmp-")

if __name__ == "__main__":
    sys.exit(main())
