"""
dsh-metabolomics — 数据源：GEO 检索/详情/下载 + EBI MetaboLights 检索/详情/下载

对外函数（mb_engine / mb_registry 逐字调用）：
  geo_search(term, species="all", method="all", min_samples=0, has_clinical=False,
             limit=20, page=0, sort="relevance", **kw) -> dict
  geo_detail(gse) -> dict
  geo_download(gse, files, dest_dir, overwrite=False, progress=None) -> dict
  ml_search(query, limit=20, page=0, assay_type=None) -> dict
  ml_detail(accession) -> dict
  ml_download(accession, files, dest_dir, overwrite=False, progress=None) -> dict

实现要点（与 dsh-geobulk / dsh-spatial 同一套约定）：
  * 只用标准库 urllib（不依赖 requests）；SSL 用 certifi CA（系统 openssl 可能缺根证书）。
  * 网络失败重试 ≤2 次（退避 1s/2s），仍失败抛 EngineError("unsupported", "…暂时不可达：<原因>")。
  * GEO：esearch(db=gds) + esummary(db=gds, UID = 200000000 + GSE 号)；目录解析
    `<GSE>/matrix/` 与 `<GSE>/suppl/`；系列矩阵头用 HTTP Range 只取 96KB 解析临床字段。
  * MetaboLights（已实测的 EBI 公开 API）：
      GET /ws/studies?searchString=<q>&page=<n>&size=<k>        → {"content": ["MTBLS1", ...]}
      GET /ws/studies/public/study/<ACC>                        → {"content": {title, description, organism[], assays[]}}
      GET /ws/studies/<ACC>/files                               → {"study": [{file, size, directory}], ...}
      FTP 目录：https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/<ACC>/
  * 下载：dest_dir 必须绝对路径且不是 `/`；单文件 ≤8GB；先写 `<name>.part`，完整性校验
    （写入字节 == Content-Length，缺失时按扩展名做 gzip/bz2/tar 可读性兜底）通过后原子替换；
    失败/中断保留旧文件并清理 `.part`；只支持失败重试，不做断点续传。
  * 测试通过替换模块级 `_urlopen(request, timeout)` 来 mock 网络。
"""

from __future__ import annotations

import bz2
import gzip
import html
import json
import os
import re
import socket
import ssl
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

from mb_common import EngineError

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"
NCBI_ORIGIN = "https://ftp.ncbi.nlm.nih.gov"
GEO_BASE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
ML_API = "https://www.ebi.ac.uk/metabolights/ws"
ML_FTP = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"

USER_AGENT = "dsh-metabolomics/1.0 (metabolomics workbench; local desktop app)"
FTP_UA_PREFIX = "curl/8.7.1 "  # ftp.ncbi.nlm.nih.gov 的 WAF 实测只放行 curl/ 前缀
FTP_USER_AGENT = FTP_UA_PREFIX + "(dsh-metabolomics/1.0; metabolomics workbench)"
HTTP_TIMEOUT = 120
RETRY_SLEEPS = (1.0, 2.0)
ESUMMARY_BATCH = 100
MAX_ESUMMARY_CANDIDATES = 300
MAX_DOWNLOAD_BYTES = 8 * 1024 ** 3
MATRIX_HEAD_BYTES = 96 * 1024
MATRIX_HEAD_TIMEOUT = 30
DOWNLOAD_CHUNK = 512 * 1024
PART_SUFFIX = ".part"
VALIDATE_GZIP_FULL_MAX = 512 * 1024 ** 2
SERIES_MATRIX_KIND = "series_matrix"

_SPECIES_ALIAS = {
    "human": "Homo sapiens", "homo sapiens": "Homo sapiens", "人": "Homo sapiens", "人类": "Homo sapiens",
    "mouse": "Mus musculus", "mus musculus": "Mus musculus", "小鼠": "Mus musculus",
    "rat": "Rattus norvegicus", "大鼠": "Rattus norvegicus",
    "zebrafish": "Danio rerio", "斑马鱼": "Danio rerio",
    "yeast": "Saccharomyces cerevisiae", "酵母": "Saccharomyces cerevisiae",
    "plant": "Arabidopsis thaliana", "拟南芥": "Arabidopsis thaliana",
}

_STUDY_TYPE_ALIAS = {
    "all": None, "any": None, "": None, "全部": None,
    "sequencing": "high throughput sequencing",
    "high throughput sequencing": "high throughput sequencing",
    "测序": "high throughput sequencing",
    "array": "expression profiling by array",
    "芯片": "expression profiling by array",
    "expression profiling by array": "expression profiling by array",
    "methylation": "methylation profiling by array",
}

#: 技术判定（顺序即优先级）：先看更具体的
_METHOD_RULES = [
    ("LC-MS", ("lc-ms", "lc/ms", "lcms", "uplc", "hplc-ms", "liquid chromatography", "c18", "reversed phase",
               "液相色谱", "液质")),
    ("GC-MS", ("gc-ms", "gc/ms", "gcms", "gas chromatography", "气相色谱", "气质")),
    ("CE-MS", ("ce-ms", "capillary electrophoresis", "毛细管电泳")),
    ("NMR", ("nmr", "nuclear magnetic resonance", "核磁")),
    ("MS", ("mass spectrometry", "mass-spectrometry", "ms/ms", "ms2", "orbitrap", "q-tof", "qtof",
            "quadrupole", "time-of-flight", "tof", "质谱", "非靶向", "untargeted", "targeted", "靶向")),
]

_CLINICAL_KEYWORDS = (
    "clinical", "patient", "survival", "prognos", "follow-up", "follow up", "outcome", "cohort",
    "recurrence", "relapse", "mortality", "hazard ratio", "risk score", "cox regression",
    "case-control", "cross-sectional", "bmi", "age", "临床", "患者", "预后", "生存", "队列", "病例",
)

# ---------------------------------------------------------------------------
# 中英词典（≥200 条；键统一小写）
# 疾病 / 体液组织 / 技术 / 代谢物类别 / 物种 / 常用词
# ---------------------------------------------------------------------------

ZH_DICT = {
    # ---- 疾病与表型（54） ----
    "type 2 diabetes": "2型糖尿病", "diabetes": "糖尿病", "diabetes mellitus": "糖尿病",
    "obesity": "肥胖", "metabolic syndrome": "代谢综合征", "insulin resistance": "胰岛素抵抗",
    "cardiovascular disease": "心血管疾病", "hypertension": "高血压", "heart failure": "心力衰竭",
    "coronary artery disease": "冠心病", "atherosclerosis": "动脉粥样硬化", "stroke": "卒中",
    "fatty liver": "脂肪肝", "non-alcoholic fatty liver disease": "非酒精性脂肪肝", "nafld": "非酒精性脂肪肝",
    "liver cirrhosis": "肝硬化", "hepatocellular carcinoma": "肝细胞癌", "liver cancer": "肝癌",
    "gastric cancer": "胃癌", "colorectal cancer": "结直肠癌", "colon cancer": "结肠癌",
    "lung cancer": "肺癌", "breast cancer": "乳腺癌", "pancreatic cancer": "胰腺癌",
    "prostate cancer": "前列腺癌", "ovarian cancer": "卵巢癌", "esophageal cancer": "食管癌",
    "renal cell carcinoma": "肾细胞癌", "kidney disease": "肾病", "chronic kidney disease": "慢性肾病",
    "alzheimer's disease": "阿尔茨海默病", "parkinson's disease": "帕金森病",
    "multiple sclerosis": "多发性硬化", "depression": "抑郁", "autism": "孤独症",
    "inflammatory bowel disease": "炎症性肠病", "crohn's disease": "克罗恩病", "colitis": "结肠炎",
    "rheumatoid arthritis": "类风湿关节炎", "psoriasis": "银屑病", "asthma": "哮喘",
    "chronic obstructive pulmonary disease": "慢阻肺", "copd": "慢阻肺", "tuberculosis": "结核病",
    "sepsis": "脓毒症", "covid-19": "新型冠状病毒肺炎", "sars-cov-2": "新型冠状病毒",
    "hepatitis": "肝炎", "thyroid": "甲状腺", "polycystic ovary syndrome": "多囊卵巢综合征",
    "pcos": "多囊卵巢综合征", "endometriosis": "子宫内膜异位症", "pregnancy": "妊娠",
    "preterm birth": "早产", "aging": "衰老", "sarcopenia": "肌少症", "cachexia": "恶病质",
    "drug resistance": "耐药", "chemoresistance": "化疗耐药", "toxicity": "毒性",
    # ---- 体液与组织（34） ----
    "serum": "血清", "plasma": "血浆", "urine": "尿液", "urinary": "尿液", "faeces": "粪便", "feces": "粪便",
    "stool": "粪便", "faecal": "粪便", "saliva": "唾液", "sputum": "痰液", "bile": "胆汁",
    "cerebrospinal fluid": "脑脊液", "csf": "脑脊液", "breast milk": "母乳", "milk": "乳汁",
    "sweat": "汗液", "tear": "泪液", "seminal": "精液", "amniotic fluid": "羊水",
    "tissue": "组织", "liver tissue": "肝组织", "kidney tissue": "肾组织", "brain tissue": "脑组织",
    "heart tissue": "心脏组织", "muscle": "肌肉", "adipose tissue": "脂肪组织", "fat": "脂肪",
    "gut": "肠道", "intestine": "肠道", "microbiome": "微生物组", "cell culture": "细胞培养",
    "cell line": "细胞系", "plant": "植物", "leaf": "叶片", "root": "根",
    # ---- 技术与方法（30） ----
    "metabolomics": "代谢组学", "metabonomics": "代谢组学", "lipidomics": "脂质组学",
    "untargeted metabolomics": "非靶向代谢组学", "targeted metabolomics": "靶向代谢组学",
    "lc-ms": "液相色谱-质谱", "lc/ms": "液相色谱-质谱", "liquid chromatography": "液相色谱",
    "gc-ms": "气相色谱-质谱", "gas chromatography": "气相色谱", "nmr": "核磁共振",
    "nuclear magnetic resonance": "核磁共振", "mass spectrometry": "质谱", "ms/ms": "串联质谱",
    "capillary electrophoresis": "毛细管电泳", "ce-ms": "毛细管电泳-质谱",
    "quadrupole time-of-flight": "四极杆飞行时间", "q-tof": "四极杆飞行时间", "orbitrap": "轨道阱",
    "hplc": "高效液相色谱", "uplc": "超高效液相色谱", "gc": "气相色谱", "lc": "液相色谱",
    "derivatization": "衍生化", "ionization": "离子化", "esi": "电喷雾", "apci": "大气压化学电离",
    "transcriptomics": "转录组学", "proteomics": "蛋白组学", "metagenomics": "宏基因组学",
    "multi-omics": "多组学", "isotope": "同位素", "flux": "通量",
    # ---- 代谢物类别（28） ----
    "amino acid": "氨基酸", "amino acids": "氨基酸", "lipid": "脂质", "lipids": "脂质",
    "fatty acid": "脂肪酸", "bile acid": "胆汁酸", "bile acids": "胆汁酸", "steroid": "类固醇",
    "sphingolipid": "鞘脂", "phospholipid": "磷脂", "triglyceride": "甘油三酯", "cholesterol": "胆固醇",
    "carbohydrate": "碳水化合物", "sugar": "糖", "glucose": "葡萄糖", "nucleotide": "核苷酸",
    "organic acid": "有机酸", "short-chain fatty acid": "短链脂肪酸", "scfa": "短链脂肪酸",
    "amine": "胺", "polyamine": "多胺", "vitamin": "维生素", "polyphenol": "多酚",
    "flavonoid": "黄酮", "alkaloid": "生物碱", "terpenoid": "萜类", "peptide": "肽",
    "metabolite": "代谢物", "metabolites": "代谢物", "biomarker": "生物标志物", "biomarkers": "生物标志物",
    "signature": "特征签名", "pathway": "通路", "pathways": "通路",
    # ---- 物种（12） ----
    "homo sapiens": "人", "human": "人", "mus musculus": "小鼠", "mouse": "小鼠",
    "rattus norvegicus": "大鼠", "rat": "大鼠", "danio rerio": "斑马鱼", "zebrafish": "斑马鱼",
    "saccharomyces cerevisiae": "酿酒酵母", "yeast": "酵母", "escherichia coli": "大肠杆菌",
    "arabidopsis thaliana": "拟南芥",
    # ---- 常用生物学词（40） ----
    "metabolic": "代谢", "metabolism": "代谢", "response": "反应", "treatment": "治疗",
    "therapy": "治疗", "drug": "药物", "exposure": "暴露", "diet": "饮食", "nutrition": "营养",
    "exercise": "运动", "fasting": "禁食", "gut microbiota": "肠道菌群", "inflammation": "炎症",
    "oxidative stress": "氧化应激", "apoptosis": "凋亡", "proliferation": "增殖",
    "differentiation": "分化", "development": "发育", "infection": "感染", "vaccine": "疫苗",
    "mutation": "突变", "expression": "表达", "profiling": "谱分析", "screening": "筛选",
    "prediction": "预测", "diagnosis": "诊断", "prognosis": "预后", "survival": "生存",
    "risk": "风险", "severity": "严重程度", "outcome": "结局", "recovery": "恢复",
    "placebo": "安慰剂", "cohort": "队列", "longitudinal": "纵向", "cross-sectional": "横断面",
    "randomized": "随机", "case-control": "病例对照", "obese": "肥胖", "healthy": "健康",
}

# 疾病关键词（用于 item.disease 与标签）
DISEASE_KEYS = [
    "type 2 diabetes", "diabetes", "obesity", "metabolic syndrome", "insulin resistance",
    "cardiovascular disease", "hypertension", "heart failure", "coronary artery disease",
    "atherosclerosis", "fatty liver", "non-alcoholic fatty liver disease", "liver cirrhosis",
    "hepatocellular carcinoma", "liver cancer", "gastric cancer", "colorectal cancer",
    "colon cancer", "lung cancer", "breast cancer", "pancreatic cancer", "prostate cancer",
    "ovarian cancer", "esophageal cancer", "renal cell carcinoma", "chronic kidney disease",
    "alzheimer's disease", "parkinson's disease", "multiple sclerosis", "depression", "autism",
    "inflammatory bowel disease", "crohn's disease", "colitis", "rheumatoid arthritis",
    "psoriasis", "asthma", "copd", "tuberculosis", "sepsis", "covid-19", "hepatitis",
    "polycystic ovary syndrome", "endometriosis", "pregnancy", "preterm birth", "cachexia",
]

_ZH_SORTED = sorted(ZH_DICT.items(), key=lambda kv: len(kv[0]), reverse=True)
_ZH_PATTERNS = [
    (re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE), zh)
    for term, zh in _ZH_SORTED
]
_DISEASE_PATTERNS = [
    (re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE), term)
    for term in sorted(DISEASE_KEYS, key=len, reverse=True)
]
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ZH_TO_EN = sorted(((zh, en) for en, zh in ZH_DICT.items() if isinstance(zh, str) and zh),
                   key=lambda item: len(item[0]), reverse=True)
_GENERIC_QUERY_RE = re.compile(
    r"\b(metabolom\w*|metabonom\w*|lipidom\w*|profiling|profile|expression|data|dataset|datasets|study|studies)\b",
    re.I,
)
_SPECIES_TAGS = {"Homo sapiens": "人", "Mus musculus": "小鼠", "Rattus norvegicus": "大鼠"}
_TECH_TAGS = {"high throughput sequencing": "高通量测序", "expression profiling by array": "芯片",
              "methylation profiling by array": "甲基化"}
_ML_ASSAY_TAGS = {"nmr": "NMR", "mass": "MS", "lc": "LC-MS", "gc": "GC-MS"}


# ---------------------------------------------------------------------------
# 网络层（测试替换 _urlopen）
# ---------------------------------------------------------------------------

_SSL_CTX = None


def _ssl_context():
    global _SSL_CTX
    if _SSL_CTX is None:
        try:
            import certifi

            _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001 - 没有 certifi 时退回默认 CA
            _SSL_CTX = ssl.create_default_context()
    return _SSL_CTX


def _urlopen(request, timeout):  # pragma: no cover - 真实网络路径
    """单点 URL 打开（测试 monkeypatch 这一层）。"""
    try:
        return urllib.request.urlopen(request, timeout=timeout, context=_ssl_context())
    except TypeError:  # pragma: no cover
        return urllib.request.urlopen(request, timeout=timeout)


def _reason(error) -> str:
    if isinstance(error, urllib.error.HTTPError):
        code = int(getattr(error, "code", 0) or 0)
        if code == 404:
            return "资源不存在（HTTP 404）"
        if code in (429, 503):
            return f"服务限流或维护（HTTP {code}）"
        if code >= 500:
            return f"服务端错误（HTTP {code}）"
        return f"HTTP {code}"
    if isinstance(error, urllib.error.URLError):
        detail = getattr(error, "reason", error)
        if isinstance(detail, (socket.timeout, TimeoutError)):
            return "连接超时"
        if isinstance(detail, ssl.SSLError):
            return "TLS 握手失败"
        if isinstance(detail, socket.gaierror):
            return "域名解析失败（可能无网络）"
        return str(detail)
    if isinstance(error, (socket.timeout, TimeoutError)):
        return "连接超时"
    if isinstance(error, ssl.SSLError):
        return "TLS 握手失败"
    if isinstance(error, json.JSONDecodeError):
        return "返回内容不是合法 JSON"
    return str(error) or type(error).__name__


def _network_fail(error, host: str = "NCBI") -> EngineError:
    return EngineError("unsupported", f"{host} 暂时不可达：{_reason(error)}")


def _request(url: str, headers: dict | None = None):
    if "ftp.ncbi.nlm.nih.gov" in url:
        agent = FTP_USER_AGENT
    elif "ftp.ebi.ac.uk" in url:
        agent = FTP_UA_PREFIX + "(dsh-metabolomics/1.0; MetaboLights)"
    else:
        agent = USER_AGENT
    merged = {"User-Agent": agent, "Accept": "*/*"}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged)


def _host_of(url: str) -> str:
    return "MetaboLights（EBI）" if "ebi.ac.uk" in url else "NCBI"


def _fetch_bytes(url: str, *, timeout: int = HTTP_TIMEOUT, retries: int = 2,
                 headers: dict | None = None, limit: int | None = None,
                 soft_404: bool = False):
    last_error = None
    attempts = max(0, int(retries)) + 1
    for attempt in range(attempts):
        try:
            with _urlopen(_request(url, headers), timeout) as response:
                data = response.read() if limit is None else response.read(limit)
            return data or b""
        except urllib.error.HTTPError as error:
            if soft_404 and int(getattr(error, "code", 0) or 0) == 404:
                return None
            last_error = error
        except (urllib.error.URLError, socket.timeout, TimeoutError, ssl.SSLError, OSError) as error:
            last_error = error
        except Exception as error:  # noqa: BLE001
            last_error = error
        if attempt < attempts - 1:
            delay = RETRY_SLEEPS[min(attempt, len(RETRY_SLEEPS) - 1)]
            if delay > 0:
                time.sleep(delay)
    raise _network_fail(last_error, _host_of(url))


def _fetch_json(url: str, *, retries: int = 2, soft_404: bool = False, host: str | None = None,
                timeout: int = HTTP_TIMEOUT):
    raw = _fetch_bytes(url, retries=retries, soft_404=soft_404, timeout=timeout)
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _network_fail(error, host or _host_of(url))
    if isinstance(data, dict) and data.get("error"):
        raise EngineError("bad_request", f"检索式无法解析：{str(data.get('error'))[:300]}")
    return data


def _fetch_text(url: str, *, retries: int = 2, soft_404: bool = False, limit: int | None = None):
    raw = _fetch_bytes(url, retries=retries, soft_404=soft_404, limit=limit)
    if raw is None:
        return None
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _norm_gse(gse) -> str:
    text = str(gse or "").strip().upper().replace(" ", "")
    match = re.match(r"^(?:GSE)?(\d{1,9})$", text)
    if not match:
        raise EngineError("bad_request", f"GSE 编号格式不正确：{gse!r}（示例：GSE13861）")
    return f"GSE{int(match.group(1))}"


def _gse_number(gse: str) -> int:
    return int(re.sub(r"\D", "", gse))


def _ftp_prefix(gse: str) -> str:
    number = str(_gse_number(gse))
    base = number[:-3] if len(number) > 3 else "0"
    return f"GSE{base}nnn"


def _series_root(gse: str) -> str:
    return f"{GEO_FTP}/{_ftp_prefix(gse)}/{gse}"


def _uid_for(gse: str) -> str:
    return str(200000000 + _gse_number(gse))


def _norm_ml(accession) -> str:
    text = str(accession or "").strip().upper().replace(" ", "")
    if text.startswith("MTBLS"):
        text = "MTBLS" + text[5:]
    match = re.match(r"^MTBLS(\d{1,7})$", text)
    if not match:
        raise EngineError("bad_request", f"MetaboLights 编号格式不正确：{accession!r}（示例：MTBLS1）")
    return f"MTBLS{int(match.group(1))}"


def _norm_date(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # EBI 常返回 epoch 毫秒/秒（如 1769904000000）
    if re.fullmatch(r"\d{10,13}", text):
        try:
            stamp = int(text)
            if stamp > 10 ** 11:
                stamp //= 1000
            return time.strftime("%Y-%m-%d", time.gmtime(stamp))
        except (ValueError, OSError):
            return text
    match = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", text)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match = re.match(r"^([A-Z][a-z]{2})\s+(\d{1,2}),?\s+(\d{4})$", text)
    if match:
        months = {m: i + 1 for i, m in enumerate(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
        month = months.get(match.group(1))
        if month:
            return f"{match.group(3)}-{month:02d}-{int(match.group(2)):02d}"
    return text


def _platforms(value) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    out = []
    for token in re.split(r"[,;\s]+", text):
        token = token.strip().upper()
        if not token:
            continue
        if token.isdigit():
            token = f"GPL{token}"
        if token not in out:
            out.append(token)
    return out


def _parse_size(token: str) -> int:
    text = str(token or "").strip().upper()
    if not text or text in ("-", "N/A"):
        return 0
    match = re.match(r"^([0-9.]+)\s*([KMGTP]?)", text)
    if not match:
        return 0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0
    factor = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4, "P": 1024 ** 5}[match.group(2)]
    return int(value * factor)


def _clean_quotes(value: str) -> str:
    text = str(value or "").strip().strip('"').strip()
    if ",," in text:
        text = text.replace(",,", ",")
    return text


def _translate(text: str) -> str:
    source = str(text or "")
    if not source:
        return ""
    result = source
    hits = 0
    for pattern, zh in _ZH_PATTERNS:
        def _repl(match, _zh=zh):
            nonlocal hits
            hits += 1
            return _zh
        result, _count = pattern.subn(_repl, result)
    if not hits:
        return ""
    return re.sub(r"\s{2,}", " ", result).strip()


def _matched_chinese(text: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    source = str(text or "")
    for pattern, zh in _ZH_PATTERNS:
        if len(out) >= limit:
            break
        if pattern.search(source) and zh not in out:
            out.append(zh)
    return out


def _detect_disease(title: str, summary: str) -> str:
    for text in (title, summary):
        for pattern, term in _DISEASE_PATTERNS:
            if pattern.search(str(text or "")):
                return term
    return ""


def _looks_clinical(text: str) -> bool:
    lower = str(text or "").lower()
    return any(keyword in lower for keyword in _CLINICAL_KEYWORDS)


def method_hint(title: str, summary: str, platform: str = "") -> str:
    """从标题/摘要/平台判定技术：LC-MS / GC-MS / CE-MS / NMR / MS / unknown。"""
    blob = f"{title} {summary} {platform}".lower()
    for label, keys in _METHOD_RULES:
        if any(key in blob for key in keys):
            return label
    return "unknown"


def _method_matches(wanted: str, label: str) -> bool:
    wanted = str(wanted or "all").strip().lower()
    if wanted in ("", "all", "any", "全部"):
        return True
    if wanted == "ms":
        return label in ("MS", "LC-MS", "GC-MS", "CE-MS")
    if wanted in ("lc-ms", "lcms", "lc"):
        return label == "LC-MS"
    if wanted in ("gc-ms", "gcms", "gc"):
        return label == "GC-MS"
    if wanted == "nmr":
        return label == "NMR"
    return label.lower() == wanted


def _make_tags(title: str, summary: str, species: str, study_type: str, disease: str) -> list[str]:
    tags: list[str] = []
    disease_zh = ZH_DICT.get(disease.lower()) if disease else None
    if disease_zh:
        tags.append(disease_zh)
    for item in _matched_chinese(title, limit=6) + _matched_chinese(summary, limit=4):
        if item not in tags:
            tags.append(item)
    species_zh = _SPECIES_TAGS.get(str(species or "").strip())
    if species_zh and species_zh not in tags:
        tags.append(species_zh)
    study_zh = _TECH_TAGS.get(str(study_type or "").strip().lower())
    if study_zh and study_zh not in tags:
        tags.append(study_zh)
    return tags[:10] or ["GEO 数据集"]


# ---------------------------------------------------------------------------
# 检索式
# ---------------------------------------------------------------------------


_METHOD_QUERY_TERMS = {
    "lc-ms": '("LC-MS"[All Fields] OR "liquid chromatography"[All Fields] OR "LC-MS/MS"[All Fields] OR "UPLC"[All Fields] OR "HPLC"[All Fields])',
    "gc-ms": '("GC-MS"[All Fields] OR "gas chromatography"[All Fields])',
    "nmr": '("NMR"[All Fields] OR "nuclear magnetic resonance"[All Fields])',
    "ms": '("mass spectrometry"[All Fields] OR "LC-MS"[All Fields] OR "GC-MS"[All Fields] OR "metabolomics"[All Fields])',
}


def _method_clause(method: str) -> str:
    return _METHOD_QUERY_TERMS.get(str(method or "all").strip().lower(), "")


def _build_query(term: str, species: str, study_type: str, min_samples: int) -> str:
    clauses = [f"({term})"]
    if "etyp" not in term.lower():
        clauses.append("(GSE[ETYP])")
    species_key = str(species or "all").strip().lower()
    if species_key not in ("all", "", "any", "全部"):
        organism = _SPECIES_ALIAS.get(species_key) or str(species).strip()
        clauses.append(f'("{organism}"[Organism])')
    study_key = str(study_type or "all").strip().lower()
    mapped = _STUDY_TYPE_ALIAS.get(study_key, study_type)
    if mapped and str(mapped).lower() not in ("all", ""):
        clauses.append(f'("{str(mapped).strip()}"[DataSet Type])')
    try:
        min_n = int(min_samples or 0)
    except (TypeError, ValueError):
        min_n = 0
    if min_n > 0:
        clauses.append(f"({min_n}:10000000[Number of Samples])")
    return " AND ".join(clauses)


def _cascade_terms(term: str) -> list[str]:
    base = re.sub(r"\s{2,}", " ", str(term or "")).strip()
    candidates = [base]
    stripped = re.sub(r"\s{2,}", " ", _GENERIC_QUERY_RE.sub(" ", base)).strip()
    if stripped and stripped != base:
        candidates.append(stripped)
    words = [w for w in re.split(r"\s+", stripped or base) if len(w) > 1]
    if len(words) > 3:
        candidates.append(" ".join(words[:3]))
    if len(words) > 2:
        candidates.append(" ".join(words[:2]))
    out, seen = [], set()
    for item in candidates:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out or [base]


def translate_query_zh(term: str) -> tuple[str, list[str], list[str]]:
    """中文关键词 → 英文（内置词典反查）。返回 (新查询, 命中列表, 未识别中文)。"""
    source = str(term or "")
    if not _CJK_RE.search(source):
        return source, [], []
    out = source
    used: list[str] = []
    for zh, en in _ZH_TO_EN:
        if zh and zh in out:
            out = out.replace(zh, en)
            used.append(f"{zh}→{en}")
    leftover = _CJK_RE.findall(out)
    out = re.sub(r"\s{2,}", " ", _CJK_RE.sub(" ", out)).strip()
    return out, used, sorted(set(leftover))


def _esearch(query: str, *, retmax: int, retstart: int = 0):
    params = urllib.parse.urlencode({"db": "gds", "term": query, "retmode": "json",
                                    "retmax": int(retmax), "retstart": int(retstart)})
    data = _fetch_json(f"{EUTILS}/esearch.fcgi?{params}") or {}
    payload = data.get("esearchresult") or {}
    count = int(payload.get("count") or 0)
    ids = [str(item) for item in (payload.get("idlist") or [])]
    return ids, count, str(payload.get("querytranslation") or "")


def _esummary(uids: list[str]) -> list[dict]:
    records: list[dict] = []
    for start in range(0, len(uids), ESUMMARY_BATCH):
        batch = uids[start:start + ESUMMARY_BATCH]
        if not batch:
            continue
        params = urllib.parse.urlencode({"db": "gds", "id": ",".join(batch), "retmode": "json"})
        data = _fetch_json(f"{EUTILS}/esummary.fcgi?{params}") or {}
        result = data.get("result") or {}
        ordered = [str(item) for item in (result.get("uids") or batch)]
        by_uid = {}
        for uid in ordered:
            record = result.get(uid)
            if isinstance(record, dict) and not record.get("error"):
                by_uid[uid] = record
        for uid in batch:
            record = by_uid.get(uid)
            if record:
                records.append(record)
    return records


def _norm_gse_safe(value) -> str:
    try:
        return _norm_gse(value)
    except EngineError:
        text = str(value or "").strip()
        return f"GSE{text}" if text.isdigit() else text


def _item_from_summary(record: dict) -> dict:
    accession = str(record.get("accession") or "")
    gse = accession.upper() if accession.upper().startswith("GSE") else _norm_gse_safe(record.get("gse"))
    title = str(record.get("title") or "")
    summary = str(record.get("summary") or "")
    species = str(record.get("taxon") or "")
    try:
        n_samples = int(record.get("n_samples") or len(record.get("samples") or []))
    except (TypeError, ValueError):
        n_samples = 0
    platforms = _platforms(record.get("gpl"))
    study_type = str(record.get("gdstype") or "").strip()
    disease = _detect_disease(title, summary)
    label = method_hint(title, summary, " ".join(platforms))
    return {
        "gse": gse,
        "title": title,
        "title_zh": _translate(title),
        "summary": summary[:1200],
        "species": species,
        "n_samples": n_samples,
        "platform": ", ".join(platforms),
        "study_type": study_type.lower(),
        "release": _norm_date(record.get("pdat")),
        "has_matrix": bool(n_samples > 0),
        "has_clinical": _looks_clinical(f"{title} {summary}"),
        "method": label,
        "disease": disease,
        "tags": _make_tags(title, summary, species, study_type, disease),
    }


def _sort_items(items: list[dict], sort: str) -> list[dict]:
    mode = str(sort or "relevance").strip().lower()
    if mode == "date":
        return sorted(items, key=lambda it: (it.get("release") or ""), reverse=True)
    if mode in ("samples", "sample", "n_samples"):
        return sorted(items, key=lambda it: int(it.get("n_samples") or 0), reverse=True)
    return items


def geo_search(term, species="all", method="all", min_samples=0, has_clinical=False,
               limit=20, page=0, sort="relevance", **kw) -> dict:
    """GEO 系列检索（E-utilities esearch + esummary，中文化字段，按技术筛选）。"""
    query_term = str(term or "").strip()
    if not query_term:
        raise EngineError("bad_request", "请输入检索关键词（支持中文，如「血清 代谢组学 LC-MS」）")
    try:
        limit = max(1, min(100, int(limit or 20)))
    except (TypeError, ValueError):
        limit = 20
    try:
        page = max(0, int(page or 0))
    except (TypeError, ValueError):
        page = 0
    try:
        min_samples = max(0, int(min_samples or 0))
    except (TypeError, ValueError):
        min_samples = 0
    sort_mode = str(sort or "relevance").strip().lower()
    if sort_mode not in ("relevance", "date", "samples", "sample", "n_samples"):
        sort_mode = "relevance"
    study_type = str(kw.get("study_type") or "all")

    translated = {}
    if _CJK_RE.search(query_term):
        query_term, used_terms, leftover = translate_query_zh(query_term)
        translated = {"query_zh": str(term or ""), "query_en": query_term, "translated": used_terms}
        if leftover:
            translated["translate_warning"] = "这些中文词没有词典映射（已忽略）：" + "、".join(leftover[:6])
        if not query_term:
            raise EngineError("bad_request",
                              "关键词里的中文没能翻译成英文，请改用英文关键词或补充疾病名（如：糖尿病 → diabetes）")

    needs_filter = bool(has_clinical) or sort_mode != "relevance" or str(method or "all").lower() not in ("all", "", "any")
    # 选了技术（LC-MS/GC-MS/NMR）就把技术词并进检索式：比只做事后过滤召回更全，
    # 也不会因为候选池只有 60 条而误报「只有 3 条」。
    if str(method or "all").lower() not in ("all", "", "any"):
        clause = _method_clause(method)
        if clause:
            query_term = f"({query_term}) AND {clause}"
            translated["query_en"] = query_term
    cascade = _cascade_terms(query_term)
    selected_term = cascade[0]
    ids: list[str] = []
    total = 0
    translation = ""
    relaxed_used: list[str] = []
    if not needs_filter:
        for index, candidate in enumerate(cascade):
            query = _build_query(candidate, species, study_type, min_samples)
            ids, total, translation = _esearch(query, retmax=limit, retstart=page * limit)
            selected_term = candidate
            if total > 0:
                relaxed_used = cascade[:index + 1]
                if total >= max(6, limit // 2) or index == len(cascade) - 1:
                    break
        items = [_item_from_summary(record) for record in _esummary(ids)]
    else:
        want = (page + 1) * limit
        candidate_n = min(MAX_ESUMMARY_CANDIDATES, max(want * 3, 60))
        for index, candidate in enumerate(cascade):
            query = _build_query(candidate, species, study_type, min_samples)
            ids, total, translation = _esearch(query, retmax=candidate_n, retstart=0)
            selected_term = candidate
            if total > 0:
                relaxed_used = cascade[:index + 1]
                if total >= max(6, candidate_n // 3) or index == len(cascade) - 1:
                    break
        items = [_item_from_summary(record) for record in _esummary(ids)]
        if has_clinical:
            items = [item for item in items if item.get("has_clinical")]
        if str(method or "all").lower() not in ("all", "", "any"):
            # 严格按技术筛选：选了 LC-MS 就只给 LC-MS 的系列；不命中返回空 + 提示，
            # 而不是拿一堆不相关数据充数（用户会被"明明选了仪器还全是别的"搞懵）。
            filtered = [item for item in items if _method_matches(method, item.get("method", "unknown"))]
            items = filtered
            if not filtered:
                translated.setdefault("notes", []).append(
                    "没有命中该技术的系列：可去掉技术筛选，或改用更通用的关键词（如只写代谢组学/metabolomics）")
        if sort_mode != "relevance":
            items = _sort_items(items, sort_mode)
        start = page * limit
        total = len(items)          # 过滤后总数（别把未过滤的命中数当结果数）
        items = items[start:start + limit]

    if not items and not ids:
        raise EngineError("not_found",
                          "没有检索到匹配的 GEO 系列：试试更宽泛的关键词（比如只写疾病名或 metabolomics）或去掉技术筛选")

    result = {"total": int(total), "page": int(page), "items": items, "source": "geo"}
    if translation:
        result["query"] = translation
    result.update({key: value for key, value in translated.items() if key != "notes"})
    if translated.get("notes"):
        result["notes"] = translated["notes"]
    if needs_filter:
        result["filtered"] = bool(has_clinical)
    if len(relaxed_used) > 1:
        result["relaxed"] = True
        result["used_term"] = selected_term
        result["relax_steps"] = relaxed_used
    return result


# ---------------------------------------------------------------------------
# 目录解析 / 系列矩阵头
# ---------------------------------------------------------------------------

_HREF_RE = re.compile(r'<a\s+href="(?P<href>[^"]*)"[^>]*>(?P<name>[^<]*)</a>(?P<rest>[^<]*)', re.IGNORECASE)
_SIZE_TOKEN_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?[KMGTP]?$", re.IGNORECASE)


def _classify_file(name: str, source_dir: str) -> str:
    lower = name.lower()
    if "series_matrix" in lower:
        return SERIES_MATRIX_KIND
    if re.search(r"_raw\.tar(\.gz)?$", lower):
        return "raw"
    if source_dir == "matrix":
        return "other"
    return "supplementary"


def _parse_size_from_rest(rest: str) -> int:
    tokens = [token for token in re.split(r"\s+", str(rest or "").strip()) if token]
    for token in reversed(tokens):
        if token == "-":
            return 0
        if _SIZE_TOKEN_RE.match(token):
            return _parse_size(token)
    return 0


def _parse_dir_html(text: str, source_dir: str, base_override: str | None = None) -> list[dict]:
    if not text:
        return []
    base = base_override or ""
    if not base:
        match = re.search(r"Index of\s+([^<\s]+)", text, re.IGNORECASE)
        if match:
            base = match.group(1).rstrip("/")
    files: list[dict] = []
    for match in _HREF_RE.finditer(text):
        href = html.unescape(match.group("href") or "").strip()
        raw_name = html.unescape(match.group("name") or "").strip()
        if not href or not raw_name or href.startswith(("/", "http://", "https://", "ftp://", "?")):
            continue
        if href.endswith("/"):
            continue
        if href.startswith("./"):
            href = href[2:]
        path = urllib.parse.unquote(href)
        name = os.path.basename(path.rstrip("/")) or raw_name
        size = _parse_size_from_rest(match.group("rest"))
        if base.startswith("/"):
            url = f"{NCBI_ORIGIN}{base}/{urllib.parse.quote(href)}"
        elif base:
            url = f"{base}/{urllib.parse.quote(href)}"
        else:
            url = urllib.parse.urljoin(GEO_FTP + "/", href)
        files.append({"name": name, "size": int(size), "url": url,
                      "kind": _classify_file(name, source_dir), "source": source_dir})
    out, seen = [], set()
    for item in files:
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        out.append(item)
    return out


def _list_series_files(gse: str) -> list[dict]:
    root = _series_root(gse)
    files: list[dict] = []
    matrix_html = _fetch_text(f"{root}/matrix/", soft_404=True)
    if matrix_html:
        files.extend(_parse_dir_html(matrix_html, "matrix", base_override=f"{root}/matrix"))
    suppl_html = _fetch_text(f"{root}/suppl/", soft_404=True)
    if suppl_html:
        files.extend(_parse_dir_html(suppl_html, "suppl", base_override=f"{root}/suppl"))
    order = {SERIES_MATRIX_KIND: 0, "supplementary": 1, "raw": 2, "other": 3}
    files.sort(key=lambda item: (order.get(item["kind"], 9), item["name"]))
    return files


def _parse_matrix_header(text: str) -> dict:
    design = ""
    contacts: list[str] = []
    contributors: list[str] = []
    characteristics: list[list[str]] = []
    organism = ""
    n_samples = 0
    for line in str(text or "").splitlines():
        if not line.startswith("!"):
            if line.strip():
                break
            continue
        parts = line.rstrip("\r\n").split("\t")
        key = parts[0].strip()
        values = [_clean_quotes(value) for value in parts[1:]]
        if key == "!Series_overall_design" and values:
            design = design or values[0]
        elif key in ("!Series_contact_name", "!Series_contact_email", "!Series_contact_institute",
                     "!Series_contact_department", "!Series_contact_address"):
            contacts.extend(value for value in values if value)
        elif key == "!Series_contributor":
            contributors.extend(value for value in values if value)
        elif key == "!Sample_characteristics_ch1":
            characteristics.append(values)
        elif key == "!Sample_organism_ch1" and values:
            organism = organism or values[0]
        elif key in ("!Sample_geo_accession", "!Sample_title"):
            n_samples = max(n_samples, len([value for value in values if value]))
    fields: list[dict] = []
    for index, values in enumerate(characteristics):
        non_empty = [value for value in values if value and value.lower() not in ("", "na", "nan")]
        if not non_empty:
            continue
        first = non_empty[0]
        if ":" in first:
            prefix = first.split(":", 1)[0].strip()
            key = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", prefix).strip("_").lower() + ":ch1"
            examples = [(value.split(":", 1)[1].strip() if ":" in value else value) for value in non_empty]
            example = examples[0] if examples else first.split(":", 1)[1].strip()
        else:
            key = f"characteristics_ch1_{index + 1}"
            example = first
        base_key, suffix = key, 2
        while any(field["key"] == key for field in fields):
            key = f"{base_key}#{suffix}"
            suffix += 1
        fields.append({"key": key, "n": len(non_empty), "example": example})
    contact_text = ""
    if contacts or contributors:
        names = contacts[0] if contacts else ""
        emails = next((value for value in contacts if "@" in value), "")
        pieces = [names]
        if emails:
            pieces.append(f"<{emails}>")
        if contributors:
            pieces.append("; ".join(contributors[:6]))
        contact_text = " ".join(piece for piece in pieces if piece)
    return {"design": design, "contacts": contact_text, "clinical_fields": fields,
            "organism": organism, "n_samples": n_samples}


def _fetch_matrix_head(files: list[dict]) -> dict:
    matrix_file = next((item for item in files if item.get("kind") == SERIES_MATRIX_KIND), None)
    if not matrix_file:
        return {}
    url = matrix_file.get("url") or ""
    if not url:
        return {}
    try:
        raw = _fetch_bytes(url, limit=MATRIX_HEAD_BYTES, retries=1, timeout=MATRIX_HEAD_TIMEOUT,
                          headers={"Range": f"bytes=0-{MATRIX_HEAD_BYTES - 1}"})
    except EngineError:
        return {}
    if not raw:
        return {}
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = zlib.decompressobj(31).decompress(raw)
        except zlib.error:
            raw = b""
    return _parse_matrix_header(raw.decode("utf-8", "replace"))


def _summary_for_gse(gse: str) -> dict:
    records = _esummary([_uid_for(gse)])
    if records:
        return records[0]
    ids, _, _ = _esearch(f"{gse}[ACCN]", retmax=10, retstart=0)
    candidates = [item for item in ids if item.startswith("2")]
    if candidates:
        records = _esummary(candidates[:1])
        if records:
            return records[0]
    raise EngineError("not_found", f"GEO 系列 {gse} 不存在或 NCBI 没有返回摘要")


def geo_detail(gse) -> dict:
    """GEO 系列详情（摘要 + 目录文件列表 + 系列矩阵头部抽取的临床字段）。"""
    gse = _norm_gse(gse)
    record = _summary_for_gse(gse)
    title = str(record.get("title") or "")
    summary = str(record.get("summary") or "")
    taxon = str(record.get("taxon") or "")
    try:
        n_samples = int(record.get("n_samples") or len(record.get("samples") or []))
    except (TypeError, ValueError):
        n_samples = 0
    pubmed_ids = [str(item) for item in (record.get("pubmedids") or []) if str(item).strip()]
    files = _list_series_files(gse)
    header = _fetch_matrix_head(files) if files else {}
    platforms = _platforms(record.get("gpl"))
    species = taxon or header.get("organism") or ""
    clinical_fields = header.get("clinical_fields") or []
    disease = _detect_disease(title, summary)
    suggested = ""
    for item in files:
        if item.get("kind") == SERIES_MATRIX_KIND:
            suggested = item["name"]
            break
    if not suggested and files:
        suggested = files[0]["name"]
    return {
        "gse": gse,
        "title": title,
        "title_zh": _translate(title),
        "summary": summary,
        "design": header.get("design") or "",
        "species": species,
        "n_samples": int(n_samples or header.get("n_samples") or 0),
        "platforms": platforms,
        "pubmed": ", ".join(pubmed_ids),
        "release": _norm_date(record.get("pdat")),
        "contacts": header.get("contacts") or "",
        "method": method_hint(title, summary, " ".join(platforms)),
        "clinical": {"from_series_matrix": bool(clinical_fields), "fields": clinical_fields},
        "files": [{"name": item["name"], "size": int(item["size"]), "url": item["url"],
                   "kind": item["kind"], "source": item.get("source") or ""} for item in files],
        "suggested": suggested,
        "has_matrix": bool(suggested),
        "has_clinical": bool(clinical_fields) or _looks_clinical(f"{title} {summary}"),
        "disease": disease,
        "tags": _make_tags(title, summary, species, str(record.get("gdstype") or ""), disease),
        "url": f"{GEO_BASE}?acc={gse}",
    }


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------


def _check_dest_dir(dest_dir) -> str:
    text = str(dest_dir or "").strip()
    if not text:
        raise EngineError("bad_request", "请指定下载目录（dest_dir 必须是绝对路径）")
    path = os.path.abspath(os.path.expanduser(text))
    if not os.path.isabs(path):
        raise EngineError("bad_request", f"dest_dir 必须是绝对路径：{dest_dir!r}（示例：/Users/you/METABO）")
    if path == os.path.sep:
        raise EngineError("bad_request", "不能直接下载到文件系统根目录 /，请指定子目录")
    if os.path.exists(path) and not os.path.isdir(path):
        raise EngineError("bad_request", f"dest_dir 已经是文件而不是目录：{path}")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        raise EngineError("bad_request", f"无法创建下载目录：{path}（{error}）")
    if not os.access(path, os.W_OK):
        raise EngineError("bad_request", f"下载目录不可写：{path}")
    return path


def _safe_filename(url: str, fallback: str) -> str:
    name = os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(url).path or "")) or fallback
    name = os.path.basename(name.replace("\\", "/"))
    if not name or name in (".", "..") or "/" in name or "\x00" in name:
        name = re.sub(r"[^0-9A-Za-z._-]+", "_", fallback or "download.bin") or "download.bin"
    return name


def _emit(progress, message: str, pct: float | None = None) -> None:
    if not progress:
        return
    try:
        progress(message, pct)
    except TypeError:
        try:
            progress(message)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


class _DownloadIntegrityError(Exception):
    pass


def _content_length(headers):
    if headers is None:
        return None
    try:
        text = str(headers.get("Content-Length") or "").strip()
    except Exception:  # noqa: BLE001
        return None
    if not text:
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _remove_quiet(path: str) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _check_tar_members(archive: tarfile.TarFile, *, check_size: bool, path: str) -> None:
    members = archive.getmembers()
    if not check_size or not members:
        return
    expected_end = max((int(item.offset_data) + int(item.size) for item in members), default=0)
    actual = os.path.getsize(path)
    if actual < expected_end:
        raise _DownloadIntegrityError(f"tar 数据被截断：实际 {actual} 字节 < 成员所需 {expected_end} 字节")


def _validate_download(path: str, expected, target: str) -> None:
    try:
        size = os.path.getsize(path)
    except OSError as error:
        raise _DownloadIntegrityError(f"无法读取临时文件：{error}")
    if expected is not None:
        if size != expected:
            raise _DownloadIntegrityError(
                f"写入 {size} 字节，与 Content-Length={expected} 不一致（{os.path.basename(target)}）")
        return
    name = os.path.basename(target).lower()
    try:
        if name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:gz") as archive:
                _check_tar_members(archive, check_size=False, path=path)
        elif name.endswith((".tar.bz2", ".tbz", ".tbz2")):
            with tarfile.open(path, "r:bz2") as archive:
                _check_tar_members(archive, check_size=False, path=path)
        elif name.endswith(".tar"):
            with tarfile.open(path, "r:") as archive:
                _check_tar_members(archive, check_size=True, path=path)
        elif name.endswith(".gz"):
            with gzip.open(path, "rb") as handle:
                if size <= VALIDATE_GZIP_FULL_MAX:
                    while handle.read(DOWNLOAD_CHUNK):
                        pass
                else:
                    handle.read(DOWNLOAD_CHUNK)
        elif name.endswith(".bz2"):
            with bz2.open(path, "rb") as handle:
                if size <= VALIDATE_GZIP_FULL_MAX:
                    while handle.read(DOWNLOAD_CHUNK):
                        pass
                else:
                    handle.read(DOWNLOAD_CHUNK)
    except (OSError, EOFError, tarfile.TarError, zlib.error) as error:
        raise _DownloadIntegrityError(f"{os.path.basename(target)} 无法通过容器可读性校验：{error}")


def _download_one(url: str, target: str, *, progress=None, label: str = "") -> int:
    part = f"{target}{PART_SUFFIX}"
    attempts = len(RETRY_SLEEPS) + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        _remove_quiet(part)
        written = 0
        try:
            with _urlopen(_request(url), HTTP_TIMEOUT) as response:
                total = _content_length(getattr(response, "headers", None))
                if total is not None and total > MAX_DOWNLOAD_BYTES:
                    raise EngineError("unsupported",
                                      f"文件过大（{total / 1024 ** 3:.1f} GB），超过单文件 8GB 上限：{label or url}")
                with open(part, "wb") as handle:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_DOWNLOAD_BYTES:
                            raise EngineError("unsupported", f"文件超过单文件 8GB 上限，已中止：{label or url}")
                        handle.write(chunk)
                        if progress and total:
                            _emit(progress, f"下载 {label or url}：{written / 1024 ** 2:.1f}/"
                                            f"{total / 1024 ** 2:.1f} MB", min(99.0, written / total * 100.0))
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
                _validate_download(part, total, target)
            os.replace(part, target)
            return written
        except EngineError:
            _remove_quiet(part)
            raise
        except _DownloadIntegrityError as error:
            last_error = error
            _remove_quiet(part)
        except Exception as error:  # noqa: BLE001
            last_error = error
            _remove_quiet(part)
        if attempt < attempts - 1:
            delay = RETRY_SLEEPS[min(attempt, len(RETRY_SLEEPS) - 1)]
            if delay > 0:
                time.sleep(delay)
    if isinstance(last_error, _DownloadIntegrityError):
        raise EngineError("unsupported", f"下载校验失败：{last_error}（已重试 {attempts} 次，旧文件未改动）")
    raise _network_fail(last_error)


def geo_download(gse, files, dest_dir, overwrite=False, progress=None) -> dict:
    """把 GEO 指定文件下载到 dest_dir（绝对路径、不可为 /）。"""
    started = time.time()
    gse = _norm_gse(gse)
    dest = _check_dest_dir(dest_dir)
    if isinstance(files, str):
        requested = [files]
    elif isinstance(files, (list, tuple, set)):
        requested = [str(item) for item in files if str(item).strip()]
    else:
        requested = []
    _emit(progress, f"正在获取 {gse} 的文件列表", 1)
    detail_files: list[dict] = []
    detail_error = None
    try:
        detail_files = geo_detail(gse).get("files") or []
    except EngineError as error:
        if error.code == "not_found":
            raise
        detail_error = error
    by_name = {item["name"]: item for item in detail_files}
    if not requested:
        suggested = next((item["name"] for item in detail_files
                          if item.get("kind") == SERIES_MATRIX_KIND), "")
        if not suggested and detail_files:
            suggested = detail_files[0]["name"]
        if not suggested:
            raise EngineError("not_found", f"{gse} 在 GEO FTP 上没有可下载文件（matrix/ 与 suppl/ 都为空）")
        requested = [suggested]
    plan, unknown = [], []
    for item in requested:
        if re.match(r"^(?:https?|ftp)://", item, re.IGNORECASE):
            plan.append({"name": _safe_filename(item, f"{gse}.txt"), "url": item, "size": 0})
            continue
        entry = by_name.get(item) or by_name.get(os.path.basename(item))
        if entry is None:
            if detail_files:
                unknown.append(item)
                continue
            subdir = "matrix" if SERIES_MATRIX_KIND in item.lower() else "suppl"
            plan.append({"name": os.path.basename(item),
                         "url": f"{_series_root(gse)}/{subdir}/{urllib.parse.quote(os.path.basename(item))}",
                         "size": 0})
            continue
        plan.append({"name": entry["name"], "url": entry["url"], "size": int(entry.get("size") or 0)})
    if unknown:
        raise EngineError("not_found", f"文件不在 {gse} 的可下载列表里：{', '.join(unknown[:5])}",
                          candidates=[{"name": item["name"], "size": item["size"], "kind": item["kind"]}
                                      for item in detail_files[:30]])
    if detail_error is not None and not plan:
        raise detail_error
    results: list[dict] = []
    total = len(plan)
    failed = 0
    for index, entry in enumerate(plan, start=1):
        target = os.path.join(dest, entry["name"])
        base = (index - 1) / max(total, 1) * 100.0
        if os.path.exists(target) and not overwrite:
            results.append({"name": entry["name"], "path": target, "bytes": os.path.getsize(target),
                            "ok": True, "skipped": True})
            _emit(progress, f"已存在，跳过：{entry['name']}", min(100.0, base + 100.0 / total))
            continue
        _emit(progress, f"正在下载 {entry['name']}（{index}/{total}）", base)
        try:
            written = _download_one(entry["url"], target, progress=progress, label=entry["name"])
            results.append({"name": entry["name"], "path": target, "bytes": int(written), "ok": True})
        except EngineError as error:
            if error.code == "bad_request":
                raise
            failed += 1
            results.append({"name": entry["name"], "path": target, "bytes": 0, "ok": False,
                            "error": error.message})
    ok_count = sum(1 for item in results if item.get("ok"))
    if ok_count == 0 and results:
        first_error = next((item.get("error") for item in results if item.get("error")), "")
        raise EngineError("unsupported", f"下载失败：{first_error or 'NCBI 暂时不可达'}")
    _emit(progress, f"下载完成：{ok_count}/{len(results)} 个文件", 100)
    result = {"dir": dest, "gse": gse, "files": results, "ms": int((time.time() - started) * 1000),
              "resume": False, "notes": ["下载只支持失败重试（每次从头下载），不支持断点续传"]}
    if failed:
        result["warnings"] = [f"{failed} 个文件下载失败"]
    return result


# ---------------------------------------------------------------------------
# MetaboLights（EBI）
# ---------------------------------------------------------------------------


def _ml_technique(assays) -> str:
    blob = " ".join(
        " ".join(str(assay.get(key) or "") for key in ("technology", "platform", "measurement"))
        for assay in (assays or []) if isinstance(assay, dict)
    ).lower()
    if not blob:
        return "unknown"
    if "nmr" in blob or "nuclear magnetic" in blob:
        return "NMR"
    if "gc" in blob and ("mass" in blob or "ms" in blob):
        return "GC-MS"
    if "lc" in blob or "liquid chromatography" in blob:
        return "LC-MS"
    if "mass" in blob or "ms" in blob:
        return "MS"
    return "unknown"


def ml_search(query, limit=20, page=0, assay_type=None, **kw) -> dict:
    """MetaboLights 研究检索（EBI 公开 API：/ws/studies?searchString=&page=&size=）。"""
    term = str(query or "").strip()
    if not term:
        raise EngineError("bad_request", "请输入 MetaboLights 检索关键词（如 diabetes / urine NMR）")
    try:
        limit = max(1, min(100, int(limit or 20)))
    except (TypeError, ValueError):
        limit = 20
    try:
        page = max(1, int(page or 0) + 1)  # EBI 分页从 1 开始
    except (TypeError, ValueError):
        page = 1
    translated = {}
    if _CJK_RE.search(term):
        term, used, leftover = translate_query_zh(term)
        translated = {"query_zh": str(query or ""), "query_en": term, "translated": used}
        if leftover:
            translated["translate_warning"] = "这些中文词没有词典映射（已忽略）：" + "、".join(leftover[:6])
        if not term:
            raise EngineError("bad_request", "MetaboLights 只支持英文检索：请把中文换成英文关键词（如 糖尿病 → diabetes）")
    # 优先 ebisearch（有 hitCount、size≥50 稳定），失败回退 /ws/studies?searchString=
    hit_count = None
    accessions: list[str] = []
    page_size = 50
    start = (page - 1) * limit
    ebisearch_failed = False
    try:
        params = urllib.parse.urlencode({"query": term, "format": "json", "size": page_size,
                                         "start": (start // page_size) * page_size})
        data = _fetch_json(f"https://www.ebi.ac.uk/ebisearch/ws/rest/metabolights?{params}",
                           host="MetaboLights（EBI）", timeout=12, retries=0) or {}
        hit_count = int(data.get("hitCount") or 0) or None
        accessions = [str(item.get("id")) for item in (data.get("entries") or [])
                      if isinstance(item, dict) and str(item.get("id") or "").startswith("MTBLS")]
        offset = start - (start // page_size) * page_size
        accessions = accessions[offset:offset + limit]
    except EngineError:
        # 只有“请求失败”才回退；/ws/studies?searchString= 在零命中时会忽略参数返回全部 3712 条，
        # 拿它做回退会把“查不到”变成“浏览全部”。
        ebisearch_failed = True
        accessions = []
    if ebisearch_failed and not accessions:
        params = urllib.parse.urlencode({"searchString": term, "page": page, "size": limit})
        data = _fetch_json(f"{ML_API}/studies?{params}", host="MetaboLights（EBI）", timeout=12, retries=0) or {}
        accessions = [str(item) for item in (data.get("content") or []) if str(item).strip()][:limit]
        hit_count = None
    if not accessions and not ebisearch_failed:
        raise EngineError("not_found",
                          f"MetaboLights 没有检索到「{term}」的研究：换个更宽的关键词（如 diabetes、urine、lipidomics）")
    items = []
    targets = accessions[:limit]
    metas: dict[str, dict] = {}

    def _one(acc: str):
        try:
            return acc, ml_detail(acc, with_files=False)
        except EngineError:
            return acc, None

    # 元信息预算：EBI 的 public/study 每个约 3-15s 且抖动大，limit=20 全量取会撞 120s 引擎超时。
    # 只对前 enrich_max 条取元信息（滑动窗口并发 + 预算），其余只回编号（needs_detail=true）。
    budget = 12.0
    enrich_max = 6
    if isinstance(kw, dict):
        try:
            budget = float(kw.get("budget_s") or kw.get("budget") or 12.0)
        except (TypeError, ValueError):
            budget = 12.0
        try:
            enrich_max = int(kw.get("enrich_max") or 6)
        except (TypeError, ValueError):
            enrich_max = 6
    enrich_max = max(0, min(enrich_max, len(targets)))
    if enrich_max:
        from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait as _wait
        import time as _time

        started_at = _time.time()
        pending: dict = {}
        queue = list(targets[:enrich_max])
        pool = ThreadPoolExecutor(max_workers=min(4, enrich_max))
        try:
            while queue or pending:
                while queue and len(pending) < pool._max_workers and (_time.time() - started_at) < budget:
                    acc = queue.pop(0)
                    pending[pool.submit(_one, acc)] = acc
                if not pending:
                    break
                done, _not_done = _wait(list(pending), timeout=max(0.2, budget - (_time.time() - started_at)),
                                        return_when=FIRST_COMPLETED)
                for future in done:
                    acc = pending.pop(future, None)
                    try:
                        _acc, meta = future.result()
                    except Exception:  # noqa: BLE001
                        meta = None
                    if meta is not None and acc:
                        metas[acc] = meta
                if (_time.time() - started_at) >= budget:
                    break
        finally:
            for future in list(pending):
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
    truncated = 0
    for accession in targets:
        meta = metas.get(accession)
        if meta is None:
            truncated += 1
            items.append({"accession": accession, "gse": accession, "title": "",
                          "title_zh": "", "summary": "", "description": "", "species": "",
                          "organism_part": "", "n_assays": None, "n_samples": None,
                          "method": "unknown", "study_type": "metabolomics", "release": "",
                          "submission_date": "", "public_date": "", "contacts": "",
                          "publications": [], "has_matrix": None, "has_clinical": None,
                          "disease": "", "tags": [], "files": [], "suggested": "",
                          "needs_detail": True,
                          "url": f"https://www.ebi.ac.uk/metabolights/{accession}"})
            continue
        label = meta.get("method", "unknown")
        if assay_type and str(assay_type).strip().lower() not in ("", "all"):
            wanted = str(assay_type).lower()
            if wanted in ("nmr",) and label != "NMR":
                continue
            if wanted in ("ms", "mass", "mass spectrometry") and label == "NMR":
                continue
            if wanted in ("lc", "lc-ms") and label != "LC-MS":
                continue
            if wanted in ("gc", "gc-ms") and label != "GC-MS":
                continue
        items.append(meta)
    if not items:
        raise EngineError("not_found",
                          f"MetaboLights 没有检索到「{term}」的研究：换个更宽的关键词（如 diabetes、urine、lipidomics）")
    result = {"total": int(hit_count or len(items)), "page": page - 1, "items": items,
              "source": "metabolights", "query_en": term}
    if truncated:
        result["notes"] = [f"元信息请求已到 {budget:.0f}s 预算：{truncated} 条只返回编号，"
                           f"点开详情（ml_detail）再取标题/物种/文件"]
    if translated:
        result.update(translated)
    return result


def _ml_head_size(url: str, timeout: int = 8, max_files: int = 12):
    """HEAD 取文件大小（软失败返回 None；只对前 max_files 个文件发请求）。"""
    try:
        request = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": FTP_USER_AGENT, "Accept": "*/*"})
        with _urlopen(request, timeout) as response:
            value = _content_length(getattr(response, "headers", None))
            return int(value) if value is not None else None
    except Exception:  # noqa: BLE001
        return None


def ml_detail(accession, with_files: bool = True, with_sizes: bool = False) -> dict:
    """MetaboLights 研究详情（标题/描述/物种/技术/提交日期/文件列表）。

    with_files=False 时跳过 `/files` 请求（供 ml_search 批量取元信息，避免逐条两次往返）。
    """
    acc = _norm_ml(accession)
    meta_raw = _fetch_json(f"{ML_API}/studies/public/study/{acc}", soft_404=True,
                           host="MetaboLights（EBI）", timeout=15, retries=1)
    meta = (meta_raw or {}).get("content") if isinstance(meta_raw, dict) else None
    if not isinstance(meta, dict):
        raise EngineError("not_found", f"MetaboLights 研究 {acc} 不存在或未公开")
    files = []
    if with_files:
        files_raw = _fetch_json(f"{ML_API}/studies/{acc}/files", soft_404=True,
                                host="MetaboLights（EBI）", timeout=15, retries=1)
        for entry in ((files_raw or {}).get("study") or []) if isinstance(files_raw, dict) else []:
            if not isinstance(entry, dict) or entry.get("directory"):
                continue
            name = str(entry.get("file") or "").strip()
            if not name:
                continue
            url = f"{ML_FTP}/{acc}/{urllib.parse.quote(name)}"
            size = int(entry.get("size") or 0) or None
            if size is None and with_sizes and len(files) < 12:
                size = _ml_head_size(url)
            files.append({"name": name, "size": size, "url": url,
                          "kind": _ml_file_kind(name), "source": "metabolights"})
    organisms = meta.get("organism") or []
    species = ", ".join(sorted({str(item.get("organismName") or "") for item in organisms
                                if isinstance(item, dict) and item.get("organismName")})) or ""
    parts = ", ".join(sorted({str(item.get("organismPart") or "") for item in organisms
                              if isinstance(item, dict) and item.get("organismPart")} - {""}))
    title = str(meta.get("title") or "")
    description = str(meta.get("description") or meta.get("studyDescription") or "")
    label = _ml_technique(meta.get("assays"))
    suggested = next((item["name"] for item in files if item["kind"] == "data"), "")
    if not suggested and files:
        suggested = files[0]["name"]
    return {
        "accession": acc,
        "gse": acc,
        "title": title,
        "title_zh": _translate(title),
        "summary": description,
        "description": description,
        "species": species,
        "organism_part": parts,
        "n_assays": len(meta.get("assays") or []),
        # studySize 是字节数（不是样本数）；样本数从 sampleTable.data 行数取
        "n_samples": _ml_sample_count(meta),
        "study_size_bytes": int(meta.get("studySize") or 0) or None,
        "method": label,
        "study_type": "metabolomics",
        "release": _norm_date(meta.get("studySubmissionDate") or meta.get("studyPublicReleaseDate")),
        "submission_date": _norm_date(meta.get("studySubmissionDate")),
        "public_date": _norm_date(meta.get("studyPublicReleaseDate")),
        "contacts": _ml_contacts(meta.get("users")),
        "publications": [str((item.get("title") or item.get("doi") or ""))
                         for item in (meta.get("publications") or []) if isinstance(item, dict)][:10],
        "has_matrix": bool(files),
        "has_clinical": _looks_clinical(f"{title} {description}"),
        "disease": _detect_disease(title, description),
        "tags": _make_tags(title, description, species, "metabolomics", _detect_disease(title, description)),
        "files": files,
        "suggested": suggested,
        "url": f"https://www.ebi.ac.uk/metabolights/{acc}",
    }


def _ml_contacts(users) -> str:
    """拼联系人：真实 users[] 只有 firstName/lastName/email/affiliation（没有 fullName）。"""
    out: list[str] = []
    for item in users or []:
        if not isinstance(item, dict):
            continue
        name = " ".join(part for part in (str(item.get("firstName") or "").strip(),
                                          str(item.get("lastName") or "").strip()) if part).strip()
        email = str(item.get("email") or "").strip()
        label = f"{name} <{email}>" if name and email else (name or email)
        if label and label not in out:
            out.append(label)
    return ", ".join(out[:6])[:400]


def _ml_sample_count(meta: dict):
    """MetaboLights 样本数：sampleTable.data 的行数（studySize 是字节数，不能用）。"""
    table = meta.get("sampleTable") if isinstance(meta, dict) else None
    if isinstance(table, dict):
        rows = table.get("data")
        if isinstance(rows, list) and rows:
            return len(rows)
    samples = meta.get("samples") if isinstance(meta, dict) else None
    if isinstance(samples, list) and samples:
        return len(samples)
    return None


def _ml_file_kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith((".tsv", ".txt", ".csv", ".xlsx")) and lower.startswith(("m_", "a_", "s_", "i_")):
        return "isa_tab"
    if lower.endswith((".tsv", ".txt", ".csv", ".xlsx", ".mzml", ".raw", ".cdf", ".zip", ".mzxml")):
        return "data"
    if lower.endswith((".pdf", ".docx")):
        return "document"
    return "other"


def ml_download(accession, files, dest_dir, overwrite=False, progress=None) -> dict:
    """下载 MetaboLights 文件（EBI FTP/HTTPS）。"""
    started = time.time()
    acc = _norm_ml(accession)
    dest = _check_dest_dir(dest_dir)
    if isinstance(files, str):
        requested = [files]
    elif isinstance(files, (list, tuple, set)):
        requested = [str(item) for item in files if str(item).strip()]
    else:
        requested = []
    _emit(progress, f"正在获取 {acc} 的文件列表", 1)
    try:
        detail = ml_detail(acc)
    except EngineError:
        raise
    by_name = {item["name"]: item for item in detail.get("files") or []}
    if not requested:
        if not by_name:
            raise EngineError("not_found", f"{acc} 没有可下载文件")
        requested = [detail.get("suggested") or next(iter(by_name))]
    plan, unknown = [], []
    for item in requested:
        if re.match(r"^(?:https?|ftp)://", item, re.IGNORECASE):
            plan.append({"name": _safe_filename(item, f"{acc}.txt"), "url": item})
            continue
        entry = by_name.get(item) or by_name.get(os.path.basename(item))
        if entry is None:
            unknown.append(item)
            continue
        plan.append({"name": entry["name"], "url": entry["url"]})
    if unknown:
        raise EngineError("not_found", f"文件不在 {acc} 的可下载列表里：{', '.join(unknown[:5])}",
                          candidates=[{"name": item["name"], "size": item["size"], "kind": item["kind"]}
                                      for item in list(by_name.values())[:30]])
    results, failed = [], 0
    total = len(plan)
    for index, entry in enumerate(plan, start=1):
        target = os.path.join(dest, entry["name"])
        base = (index - 1) / max(total, 1) * 100.0
        if os.path.exists(target) and not overwrite:
            results.append({"name": entry["name"], "path": target, "bytes": os.path.getsize(target),
                            "ok": True, "skipped": True})
            _emit(progress, f"已存在，跳过：{entry['name']}", min(100.0, base + 100.0 / total))
            continue
        _emit(progress, f"正在下载 {entry['name']}（{index}/{total}）", base)
        try:
            written = _download_one(entry["url"], target, progress=progress, label=entry["name"])
            results.append({"name": entry["name"], "path": target, "bytes": int(written), "ok": True})
        except EngineError as error:
            if error.code == "bad_request":
                raise
            failed += 1
            results.append({"name": entry["name"], "path": target, "bytes": 0, "ok": False,
                            "error": error.message})
    ok_count = sum(1 for item in results if item.get("ok"))
    if ok_count == 0 and results:
        first_error = next((item.get("error") for item in results if item.get("error")), "")
        raise EngineError("unsupported", f"下载失败：{first_error or 'EBI 暂时不可达'}")
    _emit(progress, f"下载完成：{ok_count}/{len(results)} 个文件", 100)
    out = {"dir": dest, "accession": acc, "files": results,
           "ms": int((time.time() - started) * 1000), "resume": False,
           "notes": ["EBI 文件较大时下载较慢；只支持失败重试，不支持断点续传"]}
    if failed:
        out["warnings"] = [f"{failed} 个文件下载失败"]
    return out


__all__ = ["geo_search", "geo_detail", "geo_download",
           "ml_search", "ml_detail", "ml_download",
           "ZH_DICT", "DISEASE_KEYS", "method_hint", "translate_query_zh"]
