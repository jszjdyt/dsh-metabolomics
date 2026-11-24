#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — GEO + MetaboLights 数据源离线测试（DESIGN §7 / §11）

不联网：把 `mb_geo` 的网络单点 `_urlopen(request, timeout)` 换成 FakeUrlOpen，
返回固定的 E-utilities JSON / GEO FTP 目录 HTML / series matrix 头部 /
MetaboLights WS JSON / ISA-Tab 文本 / 文件字节流，断言：

  * geo_search：解析、中文化（title_zh / tags / disease）、method 客户端过滤、
    has_clinical 过滤、逐级放宽（relaxed / used_term / relax_steps）、not_found；
  * 中文反查：`血清 代谢组学` → `serum metabolomics`（query_en 与真实检索式）；
  * geo_detail：matrix/suppl 目录解析、suggested、clinical 字段、contacts；
  * geo_download：落盘内容、progress、已存在跳过、overwrite、路径校验、
    `.part` 清理与旧文件不被破坏、持续失败 → unsupported；
  * ml_search / ml_detail / ml_download：EBI Search 编号检索 + ISA-Tab 元数据补充、
    详情（title/description/publications/contacts/files）、默认下载 ISA 元数据；
  * 网络全失败重试后 → unsupported（中文可读）。

实现假设（DESIGN 契约，不锁私有函数）：
  * `mb_geo._urlopen(request, timeout)` 可整体替换；
  * `geo_search(term, species=..., method=..., min_samples=..., has_clinical=...,
    limit=..., page=..., sort=...)`、`geo_detail(gse)`、
    `geo_download(gse, files, dest_dir, overwrite=..., progress=...)`、
    `ml_search(query=..., limit=..., page=..., assay_type=...)`、
    `ml_detail(accession)`、`ml_download(accession, files, dest_dir, ...)`；
  * 重试退避常量若存在（`RETRY_SLEEPS`）会被置 0，同时 patch `time.sleep`，测试不睡。

用法：cd plugins/dsh-metabolomics && python3 test/py_geo_test.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "lib"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mb-mpl-"))

import mb_geo  # noqa: E402

try:  # 引擎错误类型是契约的一部分（mb_common 固定）
    from mb_common import EngineError  # noqa: E402
except Exception:  # noqa: BLE001 - 兜底：用模块里导出的同名类
    EngineError = getattr(mb_geo, "EngineError")

CHECKS = 0
FAILURES: list[str] = []
SOFT_WARNINGS: list[str] = []


def _soft_warn(message: str) -> None:
    """不判失败的观察（例如实现选择了另一种合法行为），最后统一打印。"""
    if message not in SOFT_WARNINGS:
        SOFT_WARNINGS.append(message)


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


def check_in(needle, haystack, label: str, detail: str = "") -> bool:
    extra = f"；{detail}" if detail else ""
    return check(needle in haystack, label,
                 f"{needle!r} 不在 {str(haystack)[:200]!r}{extra}")


def check_has_chinese(text, label: str) -> bool:
    return check(bool(re.search(r"[\u4e00-\u9fff]", str(text or ""))), label,
                 f"没有中文：{str(text)[:160]!r}")


def check_raises(fn, codes, label: str):
    global CHECKS
    CHECKS += 1
    allowed = {codes} if isinstance(codes, str) else set(codes)
    try:
        fn()
    except EngineError as error:
        if getattr(error, "code", None) in allowed:
            check_has_chinese(getattr(error, "message", ""), f"{label}：错误消息中文可读")
            return error
        FAILURES.append(f"{label}：错误码 {getattr(error, 'code', None)} 不在 {sorted(allowed)}"
                        f"（{getattr(error, 'message', '')}）")
        print(f"  FAIL {label}：错误码 {getattr(error, 'code', None)} 不在 {sorted(allowed)}")
        return error
    except Exception as error:  # noqa: BLE001
        FAILURES.append(f"{label}：抛了 {type(error).__name__} 而不是 EngineError：{error}")
        print(f"  FAIL {label}：抛了 {type(error).__name__}")
        return None
    FAILURES.append(f"{label}：没有抛异常")
    print(f"  FAIL {label}：没有抛异常")
    return None


# ---------------------------------------------------------------------------
# 固定假响应
# ---------------------------------------------------------------------------

ESEARCH_JSON = {
    "header": {"type": "esearch"},
    "esearchresult": {
        "count": "37",
        "retmax": "2",
        "retstart": "0",
        "idlist": ["200013861", "200099999"],
        "querytranslation": 'serum metabolomics AND GSE[ETYP] AND "Homo sapiens"[Organism]',
    },
}

ESUMMARY_JSON = {
    "header": {"type": "esummary"},
    "result": {
        "uids": ["200013861", "200099999"],
        "200013861": {
            "uid": "200013861", "accession": "GSE13861", "gse": "13861",
            "title": "Serum metabolomics of type 2 diabetes by LC-MS",
            "summary": "LC-MS profiling of serum from patients with type 2 diabetes and healthy "
                       "controls; clinical data available.",
            "gpl": "6884", "taxon": "Homo sapiens", "entrytype": "GSE",
            "gdstype": "High throughput sequencing", "pdat": "2019/05/21",
            "suppfile": "TXT, XLSX", "n_samples": 120,
            "pubmedids": ["31000000"],
            "samples": [{"accession": "GSM348407"}],
            "ftplink": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13861/",
        },
        "200099999": {
            "uid": "200099999", "accession": "GSE99999", "gse": "99999",
            "title": "Urine NMR metabolomics of obesity",
            "summary": "1H NMR spectroscopy of urine samples from obese subjects.",
            "gpl": "24676", "taxon": "Homo sapiens", "entrytype": "GSE",
            "gdstype": "High throughput sequencing", "pdat": "2021/11/02",
            "suppfile": "", "n_samples": 45, "pubmedids": [], "samples": [], "ftplink": "",
        },
    },
}

MATRIX_DIR_HTML = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html><head><title>Index of /geo/series/GSE13nnn/GSE13861/matrix</title></head><body>
<h1>Index of /geo/series/GSE13nnn/GSE13861/matrix</h1><pre>Name                           Last modified      Size  <hr>
<a href="/geo/series/GSE13nnn/GSE13861/">Parent Directory</a>                             -
<a href="GSE13861_series_matrix.txt.gz">GSE13861_series_matrix.txt.gz</a>  2026-10-06 07:05   14M
<hr></pre></body></html>"""

SUPPL_DIR_HTML = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html><head><title>Index of /geo/series/GSE13nnn/GSE13861/suppl</title></head><body>
<h1>Index of /geo/series/GSE13nnn/GSE13861/suppl</h1><pre>Name                                                             Last modified      Size  <hr>
<a href="/geo/series/GSE13nnn/GSE13861/">Parent Directory</a>                                                                      -
<a href="GSE13861_ClinicalInformation.xlsx.gz">GSE13861_ClinicalInformation.xlsx.gz</a> 2011-01-04 11:41  9.4K
<a href="filelist.txt">filelist.txt</a>                                                     2013-01-17 14:47  172
<a href="GSE13861_metabolites.csv.gz">GSE13861_metabolites.csv.gz</a>                                      2019-05-21 10:00   2.5M
<a href="GSE13861_RAW.tar">GSE13861_RAW.tar</a>                                                 2013-01-17 14:47  6.3M
<hr></pre></body></html>"""

MATRIX_HEAD_TEXT = """!Series_title\t"Serum metabolomics of type 2 diabetes by LC-MS"
!Series_geo_accession\t"GSE13861"
!Series_status\t"Public on May 21 2019"
!Series_pubmed_id\t"31000000"
!Series_overall_design\t"Serum from 60 type 2 diabetes patients and 60 controls were profiled by LC-MS."
!Series_contact_name\t"Sangbae,,Kim"
!Series_contact_email\t"sangbkim22@gmail.com"
!Series_contact_institute\t"HGSC"
!Sample_title\t"case_1"\t"control_1"
!Sample_geo_accession\t"GSM348407"\t"GSM348370"
!Sample_organism_ch1\t"Homo sapiens"\t"Homo sapiens"
!Sample_characteristics_ch1\t"group: case"\t"group: control"
!Sample_characteristics_ch1\t"age: 62"\t"age: 51"
!series_matrix_table_begin
"ID_REF"\t"GSM348407"\t"GSM348370"
"""

FILE_BYTES = b"hello metabolomics download\n"

# ---- MetaboLights（响应格式按 2026-11-24 实测的真实 WS JSON 构造）----
ML_EBISEARCH_JSON = {
    "hitCount": 2,
    "entries": [{"id": "MTBLS1", "source": "metabolights"},
                {"id": "MTBLS2", "source": "metabolights"}],
    "facets": [],
}

#: /ws/studies/public/study/<acc> 实测结构（title/description/organism/users/
#: publications/assays/factors/studySize；注意日期字段是 epoch 毫秒）
ML_PUBLIC_STUDY_MTBLS1 = {
    "content": {
        "ObjectType": "Study",
        "studyIdentifier": "MTBLS1",
        "title": "A metabolomic study of urinary changes in type 2 diabetes in human "
                 "compared to the control group",
        "description": "Type 2 diabetes mellitus is the result of a combination of impaired "
                       "insulin secretion with reduced insulin sensitivity of target tissues. "
                       "In this study, NMR-based metabolomic analysis of urine was applied.",
        "studyDescription": None,
        "studyStatus": "PUBLIC",
        "studySubmissionDate": 1329177600000,
        "studyPublicReleaseDate": 1329177600000,
        "studySize": 226631680,
        "organism": [{"organismName": "Homo sapiens", "organismPart": "urine"}],
        "users": [{"firstName": "Reza", "lastName": "Salek", "email": "rms72@cam.ac.uk",
                   "affiliation": "University of Cambridge"},
                  {"firstName": "Jules", "lastName": "Griffin", "email": "jlg40@cam.ac.uk"}],
        "publications": [{
            "title": "A metabolomic comparison of urinary changes in type 2 diabetes in "
                     "mouse, rat, and human.",
            "doi": "10.1152/physiolgenomics.00194.2006",
            "pubmedId": "17190852",
            "authorList": "Salek RM,Maguire ML,Griffin JL",
        }],
        "assays": [{
            "measurement": "untargeted analysis",
            "technology": "NMR spectroscopy assay",
            "platform": "Nuclear Magnetic Resonance (NMR) - - - Bruker",
            "fileName": "a_MTBLS1_NMR_metabolite_profiling_NMR_spectroscopy.txt",
            "assayNumber": 1,
        }],
        "factors": [{"name": "Gender"}, {"name": "Metabolic syndrome"}],
    }
}

ML_PUBLIC_STUDY_GENERIC = {
    "content": {
        "studyIdentifier": "MTBLS2",
        "title": "Generic MetaboLights study",
        "description": "Generic description for offline tests.",
        "studyStatus": "PUBLIC",
        "studySubmissionDate": 1329177600000,
        "studyPublicReleaseDate": 1329177600000,
        "organism": [{"organismName": "Mus musculus", "organismPart": "liver"}],
        "users": [{"firstName": "Test", "lastName": "User", "email": "test@example.org"}],
        "publications": [],
        "assays": [],
    }
}

ML_STUDY_RECORD = {
    "mtblsStudy": {
        "studyStatus": "Public",
        "studyCategory": "nmr",
        "firstPublicDate": "2012-02-14T00:00:00",
        "studyHttpUrl":
            "http://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1",
        "datasetLicense": "EMBL-EBI Terms of Use",
    }
}

ML_INVESTIGATION_TEXT = """ONTOLOGY SOURCE REFERENCE
Term Source Name\tBTO\tCHEBI\tEFO\tMS\tMTBLS
INVESTIGATION
Investigation Identifier\tMTBLS1
STUDY
Study Identifier\tMTBLS1
Study Title\tA metabolomic study of urinary changes in type 2 diabetes in human compared to the control group
Study Description\tType 2 diabetes mellitus urine was profiled by NMR spectroscopy.
Study Submission Date\t2012-02-14
Study Public Release Date\t2012-02-14
STUDY ASSAYS
Study Assay Technology Type\tNMR spectroscopy assay
Study Assay Technology Platform\tNuclear Magnetic Resonance (NMR) - - - Bruker
STUDY PUBLICATIONS
Study Publication DOI\t10.1152/physiolgenomics.00194.2006
Study Publication Title\tA metabolomic comparison of urinary changes in type 2 diabetes
STUDY CONTACTS
Study Person Last Name\tSalek
Study Person Email\trms72@cam.ac.uk
"""

ML_FILES_JSON = {
    "latest": [],
    "obfuscationCode": "",
    "private": [],
    "study": [
        {"file": "a_MTBLS1_NMR_metabolite_profiling_NMR_spectroscopy.txt", "directory": False,
         "status": "active", "type": "metadata_assay", "size": 4096},
        {"file": "i_Investigation.txt", "directory": False, "status": "active",
         "type": "metadata_investigation", "size": 12000},
        {"file": "m_MTBLS1_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv", "directory": False,
         "status": "active", "type": "metadata_maf", "size": 244211},
        {"file": "s_MTBLS1.txt", "directory": False, "status": "active",
         "type": "metadata_sample", "size": 26206},
    ],
    "uploadPath": "",
}

ML_SAMPLE_TEXT = (
    "Source Name\tSample Name\tCharacteristics[Organism]\tCharacteristics[Gender]\n"
    "source_1\tSample_1\tHomo sapiens\tMale\n"
    "source_2\tSample_2\tHomo sapiens\tFemale\n"
)


class FakeResponse:
    """最小 urlopen 响应：read(limit) / headers / context manager。"""

    def __init__(self, data: bytes, status: int = 200, headers: dict | None = None):
        self._data = data or b""
        self._pos = 0
        self.status = status
        self.headers = headers if headers is not None else {
            "Content-Length": str(len(self._data))}

    def read(self, size: int | None = None) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        return None


class FakeUrlOpen:
    """按 URL 子串路由的假 urlopen；可设置失败次数 / 全失败 / 覆盖响应。"""

    def __init__(self):
        self.calls: list[str] = []
        self.fail_times: dict[str, int] = {}
        self.fail_code = 503
        self.override: dict[str, bytes] = {}
        self.fail_all = False
        self.length_bump = 0          # >0 时 Content-Length 故意多算（完整性校验失败）

    def fail(self, needle: str, times: int = 1) -> None:
        self.fail_times[needle] = times

    def _json(self, payload) -> bytes:
        return json.dumps(payload).encode("utf-8")

    def _payload(self, url: str) -> tuple[bytes, dict] | None:
        if "eutils.ncbi.nlm.nih.gov" in url:
            if "esearch.fcgi" in url:
                term = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("term", [""])[0]
                if "zzzz" in term.lower():
                    empty = json.loads(json.dumps(ESEARCH_JSON))
                    empty["esearchresult"]["count"] = "0"
                    empty["esearchresult"]["idlist"] = []
                    return self._json(empty), {}
                if "profiling" in term.lower() and "dataset" in term.lower():
                    # 首次（过严的检索式）无结果 → 触发逐级放宽
                    empty = json.loads(json.dumps(ESEARCH_JSON))
                    empty["esearchresult"]["count"] = "0"
                    empty["esearchresult"]["idlist"] = []
                    return self._json(empty), {}
                return self._json(ESEARCH_JSON), {}
            if "esummary.fcgi" in url:
                return self._json(ESUMMARY_JSON), {}
            return self._json({"error": "unknown eutils endpoint"}), {}
        if "ftp.ncbi.nlm.nih.gov" in url:
            if url.rstrip("/").endswith("/matrix"):
                return MATRIX_DIR_HTML.encode(), {}
            if url.rstrip("/").endswith("/suppl"):
                return SUPPL_DIR_HTML.encode(), {}
            if "series_matrix.txt.gz" in url:
                return MATRIX_HEAD_TEXT.encode(), {}
            if "filelist.txt" in url:
                return FILE_BYTES, {}
            if "metabolites.csv.gz" in url:
                return b"col1,col2\n1,2\n", {}
            return b"fake geo payload\n", {}
        if "ebisearch/ws/rest/metabolights" in url:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("query", [""])[0]
            if "nomatch" in query.lower():
                return self._json({"hitCount": 0, "entries": [], "facets": []}), {}
            return self._json(ML_EBISEARCH_JSON), {}
        if "metabolights/ws/studies" in url:
            parsed = urllib.parse.urlparse(url)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            if path.rstrip("/").endswith("/ws/studies"):
                # 浏览/回退接口：searchString 有值且明确查不到时返回空
                search_text = " ".join(query.get("searchString", [])).lower()
                if "nomatch" in search_text:
                    return self._json({"content": [], "studies": 0}), {}
                return self._json({"content": ["MTBLS1", "MTBLS2", "MTBLS3"],
                                   "studies": 3}), {}
            accession = "MTBLS1" if "mtbls1" in path.lower() else "MTBLS2"
            if "/studies/public/study/" in path:
                # 实测端点：/ws/studies/public/study/<acc> → {"content": {...}}
                payload = (ML_PUBLIC_STUDY_MTBLS1 if accession == "MTBLS1"
                           else ML_PUBLIC_STUDY_GENERIC)
                return self._json(payload), {}
            if path.endswith("/investigation"):
                return ML_INVESTIGATION_TEXT.encode("utf-8"), {}
            if path.endswith("/files"):
                if accession == "MTBLS1":
                    return self._json(ML_FILES_JSON), {}
                payload = {"study": [
                    {"file": "i_Investigation.txt", "directory": False, "status": "active",
                     "type": "metadata_investigation"},
                    {"file": "s_MTBLS2.txt", "directory": False, "status": "active",
                     "type": "metadata_sample"},
                ], "latest": [], "private": []}
                return self._json(payload), {}
            if path.endswith("/title"):
                return self._json({"title": "A metabolomic study of urinary changes in type "
                                            "2 diabetes in human"}), {}
            if path.endswith("/description"):
                return self._json({"description": "Type 2 diabetes mellitus urine was profiled "
                                                  "by NMR spectroscopy."}), {}
            if path.endswith("/publications"):
                return self._json({"publications": [{
                    "title": "A metabolomic comparison of urinary changes",
                    "doi": "10.1152/physiolgenomics.00194.2006",
                    "authorList": "Salek RM"}]}), {}
            if path.endswith("/contacts"):
                return self._json({"contacts": [{
                    "firstName": "Reza", "lastName": "Salek", "email": "rms72@cam.ac.uk",
                    "affiliation": "University of Cambridge"}]}), {}
            # 兜底：任何研究子端点都返回"大而全"的 JSON，兼容不同实现的读法
            merged = dict(ML_STUDY_RECORD)
            merged.update({
                "title": "A metabolomic study of urinary changes in type 2 diabetes",
                "description": "Type 2 diabetes mellitus urine was profiled by NMR spectroscopy.",
                "publications": [{"title": "A metabolomic comparison", "doi": "10.1152/x"}],
                "contacts": [{"firstName": "Reza", "lastName": "Salek",
                              "email": "rms72@cam.ac.uk"}],
            })
            merged.update(ML_FILES_JSON)
            return self._json(merged), {}
        if "ftp.ebi.ac.uk" in url and "/metabolights/" in url:
            if "s_MTBLS1.txt" in url:
                return ML_SAMPLE_TEXT.encode("utf-8"), {}
            return b"fake metabolights file\n", {}
        return None

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        self.calls.append(url)
        if self.fail_all:
            raise urllib.error.URLError("fake network down")
        for needle, remaining in list(self.fail_times.items()):
            if needle in url and remaining > 0:
                self.fail_times[needle] = remaining - 1
                raise urllib.error.HTTPError(url, self.fail_code, "fake failure", None, None)
        method = "GET"
        try:
            method = request.get_method()
        except Exception:  # noqa: BLE001
            method = "GET"
        info = self._payload(url)
        if info is None:
            raise urllib.error.HTTPError(url, 404, "fake not found", None, None)
        payload, _headers = info
        headers = {"Content-Length": str(len(payload) + (self.length_bump if method != "HEAD" else 0))}
        if method == "HEAD":
            return FakeResponse(b"", headers={"Content-Length": str(len(payload))})
        if url in self.override:
            payload = self.override[url]
            headers = {"Content-Length": str(len(payload))}
        return FakeResponse(payload, headers=headers)


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------


def test_dictionary() -> None:
    print("[1] ZH_DICT / 中文反查")
    size = len(getattr(mb_geo, "ZH_DICT", {}) or {})
    check(size >= 200, "ZH_DICT ≥200 条", str(size))
    dict_ = mb_geo.ZH_DICT
    check_equal(dict_.get("gastric cancer"), "胃癌", "gastric cancer→胃癌")
    check_equal(dict_.get("homo sapiens"), "人", "homo sapiens→人")
    for key, zh in (("metabolomics", "代谢组学"), ("lipidomics", "脂质组学"),
                    ("diabetes", "糖尿病"), ("urine", "尿液"), ("serum", "血清"),
                    ("bile acid", "胆汁酸"), ("nmr", "核磁共振"), ("lc-ms", "液相色谱-质谱")):
        check_equal(dict_.get(key), zh, f"代谢组学词 {key}→{zh}")
    if hasattr(mb_geo, "translate_query_zh"):
        translated, used, _left = mb_geo.translate_query_zh("血清 代谢组学")
        check_in("serum", translated.lower(), "中文反查：血清→serum")
        check_in("metabolomics", translated.lower(), "中文反查：代谢组学→metabolomics")
        check(bool(used), "中文反查返回命中列表", str(used))


def test_geo_search(fake: FakeUrlOpen) -> None:
    print("[2] geo_search")
    result = mb_geo.geo_search("serum metabolomics", species="human", limit=20, page=0)
    check_equal(result.get("page"), 0, "page")
    check(result.get("total", 0) >= 2, "total", str(result.get("total")))
    items = result.get("items") or []
    check_equal(len(items), 2, "items 条数")
    if not items:
        return
    item = items[0]
    check_equal(item.get("gse"), "GSE13861", "gse")
    check_equal(item.get("n_samples"), 120, "n_samples")
    check_equal(item.get("method"), "LC-MS", "method 判定（标题 LC-MS）")
    check_equal(item.get("species"), "Homo sapiens", "species")
    check_equal(item.get("release"), "2019-05-21", "release 日期规范化")
    check("GPL6884" in str(item.get("platform")), "platform 补 GPL 前缀", str(item.get("platform")))
    check_equal(item.get("disease"), "type 2 diabetes", "disease 识别")
    check(bool(item.get("has_matrix")) is True, "has_matrix")
    check(bool(item.get("has_clinical")) is True, "has_clinical（patients/clinical 关键词）")
    check_has_chinese(item.get("title_zh"), "title_zh 中文化")
    check_in("血清", str(item.get("title_zh")), "title_zh 含血清")
    check_in("糖尿病", str(item.get("title_zh")), "title_zh 含糖尿病")
    check(bool(item.get("tags")), "tags 非空", str(item.get("tags")))
    check_in("人", item.get("tags") or [], "tags 含物种中文")
    second = items[1]
    check_equal(second.get("method"), "NMR", "第二条 method=NMR（1H NMR）")
    # 检索式确实带 (GSE[ETYP])
    search_urls = [url for url in fake.calls if "esearch.fcgi" in url]
    check(bool(search_urls), "调用了 esearch")
    if search_urls:
        term = urllib.parse.parse_qs(urllib.parse.urlparse(search_urls[-1]).query).get("term", [""])[0]
        check_in("GSE[ETYP]", term, "检索式含 (GSE[ETYP])")
        check_in("Homo sapiens", term, "species=human → Organism 过滤")

    # method 客户端过滤
    nmr_only = mb_geo.geo_search("serum metabolomics", method="nmr", limit=20)
    gses = [item.get("gse") for item in nmr_only.get("items") or []]
    check_equal(gses, ["GSE99999"], "method=nmr 只留 NMR")
    lcms_only = mb_geo.geo_search("serum metabolomics", method="ms", limit=20)
    gses = [item.get("gse") for item in lcms_only.get("items") or []]
    check("GSE13861" in gses, "method=ms 兼容 LC-MS", str(gses))
    if "method_filter" in nmr_only:
        check_equal(nmr_only.get("method_filter"), "nmr", "method_filter 标注")

    # has_clinical 过滤
    clinical = mb_geo.geo_search("serum metabolomics", has_clinical=True, limit=20)
    check_equal([item.get("gse") for item in clinical.get("items") or []], ["GSE13861"],
                "has_clinical=true 只留带临床信息的")

    # 排序
    by_samples = mb_geo.geo_search("serum metabolomics", sort="samples", limit=20)
    check_equal([item.get("gse") for item in by_samples.get("items") or []],
                ["GSE13861", "GSE99999"], "sort=samples 按样本数降序")
    by_date = mb_geo.geo_search("serum metabolomics", sort="date", limit=20)
    check_equal([item.get("gse") for item in by_date.get("items") or []],
                ["GSE99999", "GSE13861"], "sort=date 按发布日期降序")

    # 逐级放宽
    relaxed = mb_geo.geo_search("serum metabolomics profiling dataset", limit=20)
    check(bool(relaxed.get("relaxed")) is True, "结果少时 relaxed=true", str(relaxed.get("relaxed")))
    check(bool(relaxed.get("used_term")), "返回 used_term", str(relaxed.get("used_term")))
    check(len(relaxed.get("relax_steps") or []) >= 2, "返回 relax_steps",
          str(relaxed.get("relax_steps")))

    # 无结果 → not_found
    check_raises(lambda: mb_geo.geo_search("zzzz-nothing", limit=5), "not_found",
                 "无结果 → not_found")
    check_raises(lambda: mb_geo.geo_search("   "), "bad_request", "空关键词 → bad_request")


def test_chinese_query(fake: FakeUrlOpen) -> None:
    print("[3] 中文关键词反查")
    result = mb_geo.geo_search("胃癌", species="human", limit=10)
    check_equal(result.get("query_zh"), "胃癌", "query_zh 保留原中文")
    check("gastric cancer" in str(result.get("query_en", "")).lower(),
          "query_en 翻成 gastric cancer", str(result.get("query_en")))
    check(bool(result.get("translated")), "translated 列表非空", str(result.get("translated")))
    search_urls = [url for url in fake.calls if "esearch.fcgi" in url]
    if search_urls:
        term = urllib.parse.parse_qs(urllib.parse.urlparse(search_urls[-1]).query).get("term", [""])[0]
        check_in("gastric cancer", term, "真实检索式里是英文")
    result2 = mb_geo.geo_search("血清 代谢组学", limit=10)
    check("serum" in str(result2.get("query_en", "")).lower()
          and "metabolomics" in str(result2.get("query_en", "")).lower(),
          "血清 代谢组学 → serum metabolomics", str(result2.get("query_en")))


def test_geo_detail(fake: FakeUrlOpen) -> None:
    print("[4] geo_detail")
    detail = mb_geo.geo_detail("gse13861")
    check_equal(detail.get("gse"), "GSE13861", "GSE 编号规范化")
    check_equal(detail.get("species"), "Homo sapiens", "species")
    check_equal(detail.get("n_samples"), 120, "n_samples")
    check("GPL6884" in str(detail.get("platforms")), "platforms", str(detail.get("platforms")))
    check_in("31000000", str(detail.get("pubmed")), "pubmed id")
    check_equal(detail.get("release"), "2019-05-21", "release")
    check_in("type 2 diabetes", str(detail.get("design")), "design 来自 series matrix 头部")
    check_in("Sangbae,Kim", str(detail.get("contacts")), "contacts（,, 还原成 ,）")
    clinical = detail.get("clinical") or {}
    check(clinical.get("from_series_matrix") is True, "clinical.from_series_matrix")
    field_keys = [field.get("key") for field in clinical.get("fields") or []]
    check_equal(field_keys, ["group:ch1", "age:ch1"], "clinical 字段 key")
    names = [item.get("name") for item in detail.get("files") or []]
    check_in("GSE13861_series_matrix.txt.gz", names, "series matrix 文件")
    check_in("GSE13861_RAW.tar", names, "RAW.tar 文件")
    check_in("GSE13861_metabolites.csv.gz", names, "suppl 代谢物表")
    matrix = next((item for item in detail.get("files") or []
                   if item.get("kind") == "series_matrix"), {})
    check_in("/matrix/GSE13861_series_matrix.txt.gz", str(matrix.get("url")),
             "series matrix 绝对 URL")
    check_equal(matrix.get("size"), 14 * 1024 ** 2, "14M 大小解析")
    check_equal(detail.get("suggested"), "GSE13861_series_matrix.txt.gz",
                "suggested 优先 series_matrix")
    check_equal(detail.get("method"), "LC-MS", "detail method")
    check_raises(lambda: mb_geo.geo_detail("SRP000001"), "bad_request",
                 "非 GSE 编号 → bad_request")


def test_geo_download(fake: FakeUrlOpen) -> None:
    print("[5] geo_download")
    dest = tempfile.mkdtemp(prefix="mb-geo-dl-")
    progress: list[tuple] = []
    try:
        result = mb_geo.geo_download("GSE13861", ["filelist.txt"], dest,
                                     progress=lambda message, pct=None: progress.append((message, pct)))
        check_equal(os.path.abspath(dest), result.get("dir"), "dir 为绝对路径")
        entries = result.get("files") or []
        check_equal(len(entries), 1, "下载文件数")
        if entries:
            entry = entries[0]
            check(bool(entry.get("ok")), "ok=true")
            check_equal(entry.get("bytes"), len(FILE_BYTES), "bytes")
            with open(entry["path"], "rb") as handle:
                check_equal(handle.read(), FILE_BYTES, "落盘内容")
        check(isinstance(result.get("ms"), int), "ms 字段")
        check(len(progress) >= 2, "进度回调被调用", str(len(progress)))
        leftovers = [name for name in os.listdir(dest) if name.endswith(".part")]
        check_equal(leftovers, [], "下载成功后没有 .part 残留")

        again = mb_geo.geo_download("GSE13861", ["filelist.txt"], dest)
        check(bool((again.get("files") or [{}])[0].get("skipped")) is True,
              "overwrite=False 已存在 → skipped")
        overwritten = mb_geo.geo_download("GSE13861", ["filelist.txt"], dest, overwrite=True)
        check(not (overwritten.get("files") or [{}])[0].get("skipped"),
              "overwrite=True 不跳过")

        # URL 直链
        url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13861/suppl/filelist.txt")
        from_url = mb_geo.geo_download("GSE13861", [url], dest)
        check_equal((from_url.get("files") or [{}])[0].get("name"), "filelist.txt",
                    "URL 下载文件名取 basename")

        # 路径 / 文件名校验
        relative_target = os.path.abspath(f"rel-geo-{os.getpid()}")
        try:
            resolved = mb_geo.geo_download("GSE13861", ["filelist.txt"], relative_target)
            check(os.path.isabs(resolved.get("dir", "")), "绝对路径 dest_dir 可用",
                  str(resolved.get("dir")))
        except EngineError as error:  # 允许实现直接拒绝（也是合法策略）
            check(error.code == "bad_request", "绝对路径 dest_dir 报错码", str(error.code))
        finally:
            shutil.rmtree(relative_target, ignore_errors=True)
        check_raises(lambda: mb_geo.geo_download("GSE13861", ["filelist.txt"], "/"),
                     "bad_request", "根目录 → bad_request")
        check_raises(lambda: mb_geo.geo_download("GSE13861", ["missing.txt"], dest),
                     "not_found", "不存在的文件 → not_found")

        # 完整性校验失败：旧文件保持原样 + 清理 .part
        target = os.path.join(dest, "filelist.txt")
        with open(target, "wb") as handle:
            handle.write(b"OLD-CONTENT")
        fake.length_bump = 10
        try:
            check_raises(lambda: mb_geo.geo_download("GSE13861", ["filelist.txt"], dest,
                                                     overwrite=True),
                         "unsupported", "内容长度不符 → unsupported")
        finally:
            fake.length_bump = 0
        with open(target, "rb") as handle:
            check_equal(handle.read(), b"OLD-CONTENT", "校验失败时旧文件未被破坏")
        leftovers = [name for name in os.listdir(dest) if name.endswith(".part")]
        check_equal(leftovers, [], "校验失败后没有 .part 残留")
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_metabolights(fake: FakeUrlOpen) -> None:
    print("[6] MetaboLights 检索 / 详情 / 下载")
    search = mb_geo.ml_search(query="diabetes", limit=10)
    check_equal(search.get("source"), "metabolights", "source=metabolights")
    check_equal(search.get("total"), 2, "total=hitCount（EBI Search）")
    items = search.get("items") or []
    check(len(items) >= 1, "items 非空", str(len(items)))
    if items:
        item = items[0]
        check_equal(item.get("accession"), "MTBLS1", "accession")
        check_in("metabol", str(item.get("title", "")).lower(), "title 来自研究元数据")
        check_has_chinese(item.get("title_zh"), "ml title_zh 中文化")
        check_in("糖尿病", str(item.get("title_zh")), "title_zh 含糖尿病")
        check_equal(item.get("method"), "NMR", "ml method（NMR spectroscopy）")
        check(str(item.get("url", "")).endswith("MTBLS1"), "url 指向研究页",
              str(item.get("url")))
    # EBI Search 请求里必须带 query
    search_urls = [url for url in fake.calls if "ebisearch" in url]
    check(bool(search_urls), "调用了 EBI Search")
    if search_urls:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(search_urls[-1]).query).get(
            "query", [""])[0]
        check_in("diabetes", query, "EBI Search query 参数")

    # assay_type 过滤
    nmr_only = mb_geo.ml_search(query="diabetes", limit=10, assay_type="nmr")
    accessions = [item.get("accession") for item in nmr_only.get("items") or []]
    check_equal(accessions, ["MTBLS1"], "assay_type=nmr 只留 NMR 研究")
    try:
        ms_only = mb_geo.ml_search(query="diabetes", limit=10, assay_type="ms")
        ms_accessions = [item.get("accession") for item in ms_only.get("items") or []]
        # 允许实现把 unknown 也算作 MS（信息不足时更宽松）；但 NMR 必须被排除
        check("MTBLS1" not in ms_accessions, "assay_type=ms 排除 NMR 研究", str(ms_accessions))
    except EngineError as error:
        check(error.code == "not_found", "assay_type=ms 全被过滤 → not_found",
              str(getattr(error, "code", None)))

    # 中文 → 英文检索
    cn = mb_geo.ml_search(query="糖尿病", limit=5)
    check_equal((cn.get("query_en") or "").lower(), "diabetes", "糖尿病→diabetes")
    check(bool(cn.get("items")), "中文检索有结果")
    check_raises(lambda: mb_geo.ml_search(query="nomatch-xyz", limit=5), "not_found",
                 "ml 无结果 → not_found")

    # 详情
    detail = mb_geo.ml_detail("mtbls1")
    check_equal(detail.get("accession"), "MTBLS1", "ml detail accession 规范化")
    check_in("metabol", str(detail.get("title", "")).lower(), "ml detail title")
    description = str(detail.get("description") or detail.get("summary") or "")
    check_in("diabetes", description.lower(), "ml detail description")
    submission = str(detail.get("submission_date") or detail.get("release") or "")
    check(bool(submission), "ml detail 有提交/发布日期", submission)
    if submission.isdigit():
        _soft_warn("ml detail 日期是原始 epoch 毫秒（%s）：真实接口 studySubmissionDate 是 ms，"
                   "建议除以 1000 再格式化" % submission)
    else:
        check_in("2012", submission, "ml detail 提交日期 2012-02-14", submission)
    publications = detail.get("publications") or []
    check(bool(publications), "ml detail publications 非空", str(publications)[:120])
    check(any("metabolomic" in str(item).lower() or "10.1152" in str(item)
              for item in publications), "publications 含真实文献信息", str(publications)[:160])
    contacts = detail.get("contacts") or []
    check(bool(contacts), "ml detail contacts 非空", str(contacts)[:120])
    contacts_text = str(contacts)
    check(("Salek" in contacts_text or "Reza" in contacts_text or "@" in contacts_text),
          "contacts 含研究者姓名或邮箱", contacts_text[:160])
    if "@" in contacts_text and "Salek" not in contacts_text:
        _soft_warn("ml detail contacts 只有邮箱：真实 /ws/studies/public/study/<acc> 的 users[] "
                   "字段是 firstName/lastName/affiliation（没有 fullName），"
                   "建议拼成「姓名 <邮箱>」")
    check_equal(detail.get("method"), "NMR", "ml detail method")
    files = detail.get("files") or []
    check(len(files) >= 4, "ml detail files 数量", str(len(files)))
    names = [item.get("name") for item in files]
    check_in("i_Investigation.txt", names, "files 含 i_Investigation.txt")
    check_in("m_MTBLS1_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv", names,
             "files 含 m_ 代谢物表")
    for entry in files:
        check_in("ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1",
                 str(entry.get("url")), f"文件下载 URL：{entry.get('name')}")
    if detail.get("n_assays") is not None:
        check_equal(detail.get("n_assays"), 1, "ml detail n_assays")
    check_raises(lambda: mb_geo.ml_detail("NOTML"), "bad_request",
                 "非法 accession → bad_request")

    # 下载（默认 = 建议文件 / ISA 元数据）
    dest = tempfile.mkdtemp(prefix="mb-ml-dl-")
    try:
        result = mb_geo.ml_download("MTBLS1", [], dest)
        check_equal(os.path.abspath(dest), result.get("dir"), "ml dir 为绝对路径")
        entries = result.get("files") or []
        check(len(entries) >= 1, "ml_download 至少 1 个文件", str(len(entries)))
        check(all(item.get("ok") for item in entries), "ml_download 全部 ok",
              str([(item.get("name"), item.get("ok")) for item in entries]))
        downloaded = [str(item.get("name")) for item in entries]
        check(any(name.lower().startswith(("i_", "s_", "a_", "m_")) for name in downloaded),
              "默认下载 ISA 元数据文件", str(downloaded))
        for item in entries:
            check(os.path.exists(item.get("path") or ""), f"ml 落盘：{item.get('name')}")
        leftovers = [name for name in os.listdir(dest) if name.endswith(".part")]
        check_equal(leftovers, [], "ml 下载后没有 .part 残留")
        again = mb_geo.ml_download("MTBLS1", [], dest)
        check(all(item.get("skipped") for item in again.get("files") or []),
              "ml overwrite=False 时全部 skipped")
        check_raises(lambda: mb_geo.ml_download("MTBLS1", ["nope.txt"], dest), "not_found",
                     "ml 未知文件 → not_found")
        relative_target = os.path.abspath(f"rel-ml-{os.getpid()}")
        try:
            resolved = mb_geo.ml_download("MTBLS1", [], relative_target)
            check(os.path.isabs(resolved.get("dir", "")), "ml 绝对路径 dest_dir 可用",
                  str(resolved.get("dir")))
        except EngineError as error:
            check(error.code == "bad_request", "ml 绝对路径 dest_dir 报错码", str(error.code))
        finally:
            shutil.rmtree(relative_target, ignore_errors=True)
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_network_failure(fake: FakeUrlOpen) -> None:
    print("[7] 网络失败重试 → unsupported")
    before = len(fake.calls)
    fake.fail_all = True
    try:
        error = check_raises(lambda: mb_geo.geo_search("serum metabolomics", limit=5),
                             "unsupported", "esearch 持续失败 → unsupported")
        if error is not None:
            check_in("不可达", str(getattr(error, "message", "")), "错误消息说明不可达")
        attempts = len(fake.calls) - before
        check(attempts >= 2, "失败后至少重试 1 次（总请求 ≥2）", str(attempts))
        check_raises(lambda: mb_geo.geo_detail("GSE13861"), "unsupported",
                     "detail 网络失败 → unsupported")
        check_raises(lambda: mb_geo.ml_search(query="diabetes", limit=5), "unsupported",
                     "ml_search 网络失败 → unsupported")
    finally:
        fake.fail_all = False
    # 恢复后仍可用（软 404：suppl 目录不存在时只剩 matrix 文件）
    fake.fail("suppl", times=99)
    fake.fail_code = 404
    try:
        detail = mb_geo.geo_detail("GSE13861")
        kinds = [item.get("kind") for item in detail.get("files") or []]
        check_equal(kinds, ["series_matrix"], "suppl 404 时只剩 series matrix")
        check_equal(detail.get("suggested"), "GSE13861_series_matrix.txt.gz",
                    "suppl 404 时 suggested 仍可用")
    finally:
        fake.fail_times.pop("suppl", None)
        fake.fail_code = 503


def test_dir_html_parser() -> None:
    print("[8] 目录 HTML 解析边界（若实现暴露 _parse_dir_html）")
    parser = getattr(mb_geo, "_parse_dir_html", None)
    if parser is None:
        print("  （实现没有暴露 _parse_dir_html，跳过边界解析断言）")
        return
    files = parser(MATRIX_DIR_HTML, "matrix")
    check_equal(len(files), 1, "Parent Directory 被忽略")
    check_in("/matrix/GSE13861_series_matrix.txt.gz", files[0].get("url"),
             "matrix 目录 URL 拼接正确")
    files2 = parser(SUPPL_DIR_HTML, "suppl")
    check_equal([item.get("name") for item in files2],
                ["GSE13861_ClinicalInformation.xlsx.gz", "filelist.txt",
                 "GSE13861_metabolites.csv.gz", "GSE13861_RAW.tar"], "suppl 文件顺序")
    check_equal(files2[0].get("size"), int(9.4 * 1024), "9.4K 大小解析")
    check_equal(parser("", "matrix"), [], "空 HTML → []")


def main() -> int:
    print("dsh-metabolomics 数据源（GEO / MetaboLights）离线测试")
    fake = FakeUrlOpen()
    original_urlopen = getattr(mb_geo, "_urlopen", None)
    original_sleeps = getattr(mb_geo, "RETRY_SLEEPS", None)
    original_sleep = getattr(mb_geo, "time", None)
    if original_urlopen is None:
        print("FAIL：mb_geo 没有模块级 _urlopen，无法 mock")
        return 1
    mb_geo._urlopen = fake  # type: ignore[assignment]
    if original_sleeps is not None:
        mb_geo.RETRY_SLEEPS = (0.0, 0.0)  # type: ignore[assignment]
    # 即使实现不读 RETRY_SLEEPS，也把 time.sleep 打成空操作，保证测试不睡
    if original_sleep is not None and hasattr(original_sleep, "sleep"):
        original_sleep_sleep = original_sleep.sleep
        original_sleep.sleep = lambda *_args, **_kwargs: None
    try:
        test_dictionary()
        test_geo_search(fake)
        test_chinese_query(fake)
        test_geo_detail(fake)
        test_geo_download(fake)
        test_metabolights(fake)
        test_network_failure(fake)
        test_dir_html_parser()
    finally:
        mb_geo._urlopen = original_urlopen  # type: ignore[assignment]
        if original_sleeps is not None:
            mb_geo.RETRY_SLEEPS = original_sleeps  # type: ignore[assignment]
        if original_sleep is not None and hasattr(original_sleep, "sleep"):
            original_sleep.sleep = original_sleep_sleep
    if SOFT_WARNINGS:
        print("\n观察（不影响通过）：")
        for item in SOFT_WARNINGS:
            print(f"  - {item}")
    if FAILURES:
        print(f"\nFAIL {len(FAILURES)} / {CHECKS} checks")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print(f"\nPASS {CHECKS} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
