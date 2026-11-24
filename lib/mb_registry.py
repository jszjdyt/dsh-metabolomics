"""
dsh-metabolomics — 模块注册表与调度层（Host 侧 Python worker 的调度核心）

职责（DESIGN.md §2.5 / §3 / §4 / §5）：
  * 动态导入全部步骤/分析模块（缺文件、导入失败只跳过并记录，不拖垮整体）
  * 汇总 STEPS / ANALYSES / PLOTS 目录（PLOTS 合并成全局 kind → fn 表）
  * 构造 ctx（§2.3）、执行步骤与分析、消费 replace / store / sample_columns+sample_values 写回
  * 样本表操作、预览、自检（audit）、建议（suggest）、预设流程（recipe）、导出（export）

约定：
  * 模块只依赖 mb_common（及 mb_state 的纯数据类），**不 import 本文件**（避免循环导入）。
  * 本文件由 Host 组维护；模块作者不要改。
  * 所有错误统一用 mb_common.EngineError（code 决定 HTTP 状态）。
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mb_common import (
    EngineError,
    data_file,
    ensure_dir,
    export_csv,
    export_json,
    jsonable,
    slug,
)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PLUGIN_NAME = "dsh-metabolomics"
PLUGIN_VERSION = "0.1.0"
SEED = 0

# 8 个步骤/分析模块（缺失只跳过；DESIGN §1）
MODULES = [
    "mb_prep",
    "mb_plsda",
    "mb_de",
    "mb_enrich",
    "mb_biomarker",
    "mb_batch",
    "mb_annotate",
    "mb_network",
]

# 可选的基础图模块（DESIGN §1 里由 mb-base 维护）
PLOT_MODULES = ["mb_plots"]

# ---------------------------------------------------------------------------
# 目录（静态契约：模块没落盘时 UI 也能渲染出完整骨架）
# ---------------------------------------------------------------------------


def _p(key: str, label: str, ptype: str = "text", default=None) -> dict:
    """构造 §2.5 的 params 元素。"""
    return {"key": key, "label": label, "type": ptype, "default": default}


# 12 个流程步骤：顺序即 UI 左栏卡片顺序（DESIGN §3）
STEP_ORDER = [
    {
        "id": "qc",
        "title": "样本质控",
        "requires": [],
        "params": [_p("group_col", "分组列", "text", None), _p("qc_label", "QC 样本标签", "text", ""),
                   _p("min_features", "每样本最少特征", "number", 0)],
    },
    {
        "id": "filter",
        "title": "特征过滤",
        "requires": ["qc"],
        "params": [_p("min_sample_frac", "最少样本比例", "number", 0.5), _p("min_intensity", "最小强度", "number", 0),
                   _p("rsd_max", "QC RSD 上限(%)", "number", None)],
    },
    {
        "id": "impute",
        "title": "缺失值填补",
        "requires": ["filter"],
        "params": [_p("method", "方法", "select", "half_min"), _p("k", "KNN 邻居数", "number", 5)],
    },
    {
        "id": "normalize",
        "title": "归一化",
        "requires": ["impute"],
        "params": [_p("method", "方法", "select", "pqn")],
    },
    {
        "id": "transform",
        "title": "变换/标度",
        "requires": ["normalize"],
        "params": [_p("transform", "变换", "select", "log2"), _p("scale", "标度", "select", "auto")],
    },
    {
        "id": "pca",
        "title": "主成分分析",
        "requires": ["transform"],
        "params": [_p("n_comp", "主成分数", "number", 3), _p("color_by", "着色", "select", "group")],
    },
    {
        "id": "plsda",
        "title": "PLS-DA / OPLS-DA",
        "requires": ["transform"],
        "params": [_p("method", "方法", "select", "plsda"), _p("n_comp", "成分数", "number", 3),
                   _p("cv_folds", "交叉验证折数", "number", 7), _p("permutations", "置换次数", "number", 200)],
    },
    {
        "id": "de",
        "title": "单变量差异",
        "requires": ["transform"],
        "params": [_p("group_col", "分组列", "text", None), _p("case", "实验组", "text", None),
                   _p("control", "对照组", "text", None), _p("method", "方法", "select", "welch"),
                   _p("paired", "配对", "bool", False), _p("fdr", "FDR 阈值", "number", 0.05),
                   _p("fc", "倍数阈值", "number", 1.5)],
    },
    {
        "id": "batch",
        "title": "批次校正",
        "requires": ["normalize"],
        "params": [_p("batch_key", "批次列", "text", None), _p("method", "方法", "select", "rlsc"),
                   _p("qc_label", "QC 样本标签", "text", "")],
    },
    {
        "id": "enrich",
        "title": "通路富集",
        "requires": ["de"],
        "params": [_p("method", "方法", "select", "mummichog"), _p("db", "数据库", "select", "kegg"),
                   _p("p_cut", "P 阈值", "number", 0.05), _p("n_perm", "置换次数", "number", 1000)],
    },
    {
        "id": "biomarker",
        "title": "标志物筛选",
        "requires": ["de"],
        "params": [_p("model", "模型", "select", "rf"), _p("n_features", "特征数", "number", 10),
                   _p("cv_folds", "交叉验证折数", "number", 5), _p("time_col", "生存时间列", "text", None),
                   _p("event_col", "生存事件列", "text", None)],
    },
    {
        "id": "annotate",
        "title": "代谢物鉴定",
        "requires": ["transform"],
        "params": [_p("adduct", "加合物", "text", "[M+H]+"), _p("ppm", "质量精度(ppm)", "number", 10),
                   _p("formula", "推断分子式", "bool", True), _p("library", "MS2 谱库路径", "text", "")],
    },
]
STEP_INDEX = {item["id"]: i for i, item in enumerate(STEP_ORDER)}
STEP_TITLES = {item["id"]: item["title"] for item in STEP_ORDER}

# 步骤 → 可能写入 dataset.results 的键（重跑清空下游时一起清）
STEP_RESULT_KEYS = {
    "qc": ["qc"],
    "filter": ["filter"],
    "impute": ["impute"],
    "normalize": ["normalize"],
    "transform": ["transform"],
    "pca": ["pca"],
    "plsda": ["plsda", "vip", "splot", "permutation", "cv"],
    "de": ["de"],
    "batch": ["batch"],
    "enrich": ["enrich"],
    "biomarker": ["biomarker"],
    "annotate": ["annotation"],
}

# 36 项分析（DESIGN §4；顺序 = UI 分组内的展示顺序）
ANALYSIS_ORDER = [
    # ---- 质控与基础 ----
    {"id": "qc_summary", "title": "质控汇总", "group": "质控与基础",
     "method": "汇总表 + TIC/RSD 图（Dunn 2011 代谢组学 QC 原则）", "needs": ["qc"], "params": []},
    {"id": "rsd_qc", "title": "QC 样本 RSD", "group": "质控与基础",
     "method": "RSD = SD/mean；QC 与生物样本对比（Want 2010；Dunn 2011）", "needs": ["qc"],
     "params": [_p("qc_label", "QC 样本标签", "text", ""), _p("threshold", "RSD 阈值(%)", "number", 30)]},
    {"id": "missing_pattern", "title": "缺失模式", "group": "质控与基础",
     "method": "缺失率分布 + 样本/特征缺失热图（内置实现）", "needs": ["qc"],
     "params": [_p("max_features", "最多特征数", "number", 500)]},
    {"id": "sample_outlier", "title": "样本离群", "group": "质控与基础",
     "method": "PCA + Hotelling T² + DModX（Wold 1994；SIMCA 语义）", "needs": ["transform"],
     "params": [_p("n_comp", "主成分数", "number", 3), _p("alpha", "置信度", "number", 0.95)]},
    {"id": "sample_corr", "title": "样本相关性", "group": "质控与基础",
     "method": "Pearson/Spearman 相关 + 层次聚类（内置实现）", "needs": ["transform"],
     "params": [_p("method", "相关系数", "select", "pearson")]},
    # ---- 多变量判别 ----
    {"id": "plsda_vip", "title": "变量投影重要度（VIP）", "group": "多变量判别",
     "method": "VIP（Wold 1995；MetaboAnalyst 语义）", "needs": ["plsda"],
     "params": [_p("top_n", "Top N", "number", 30), _p("threshold", "VIP 阈值", "number", 1)]},
    {"id": "oplsda_splot", "title": "OPLS-DA S-plot", "group": "多变量判别",
     "method": "OPLS（Trygg & Wold 2002；Bylesjö 2006 S-plot）", "needs": ["plsda"],
     "params": [_p("top_n", "Top N", "number", 20)]},
    {"id": "plsda_permutation", "title": "PLS-DA 置换检验", "group": "多变量判别",
     "method": "标签置换 + R²Y/Q² 零分布（Westerhuis 2008）", "needs": ["plsda"],
     "params": [_p("n_perms", "置换次数", "number", 200)]},
    {"id": "plsda_cv", "title": "交叉验证与混淆矩阵", "group": "多变量判别",
     "method": "k-fold / 留一交叉验证（Brindle 2002 判别思路）", "needs": ["plsda"],
     "params": [_p("folds", "折数", "number", 7)]},
    {"id": "plsda_multiclass", "title": "多组判别", "group": "多变量判别",
     "method": "多类 PLS-DA（一对其余 + 综合 VIP）", "needs": ["groups"],
     "params": [_p("n_comp", "成分数", "number", 3)]},
    {"id": "pca_loadings", "title": "PCA 载荷与贡献", "group": "多变量判别",
     "method": "PCA 载荷表 + 贡献条形图（Wold 1994）", "needs": ["transform"],
     "params": [_p("pc", "主成分", "number", 1), _p("top_n", "Top N", "number", 20)]},
    # ---- 差异与趋势 ----
    {"id": "univariate_de", "title": "单变量差异", "group": "差异与趋势",
     "method": "Welch t / Mann-Whitney / limma + BH（Xia 2015 MetaboAnalyst）", "needs": ["groups"],
     "params": [_p("method", "方法", "select", "welch"), _p("paired", "配对", "bool", False),
                _p("fdr", "FDR 阈值", "number", 0.05), _p("fc", "倍数阈值", "number", 1.5)]},
    {"id": "volcano_de", "title": "火山图", "group": "差异与趋势",
     "method": "log2FC vs −log10 p（火山图语义）", "needs": ["de"],
     "params": [_p("n_labels", "标注数", "number", 15), _p("fc", "倍数阈值", "number", 1.5),
                _p("fdr", "FDR 阈值", "number", 0.05)]},
    {"id": "fc_bar", "title": "倍数变化条形图", "group": "差异与趋势",
     "method": "top/bottom 倍数变化（内置实现）", "needs": ["de"],
     "params": [_p("top_n", "Top N", "number", 20)]},
    {"id": "heatmap_top", "title": "差异特征热图", "group": "差异与趋势",
     "method": "z-score + 层次聚类（内置实现）", "needs": ["de"],
     "params": [_p("top_n", "Top N", "number", 50), _p("cluster", "聚类", "bool", True)]},
    {"id": "multi_group_de", "title": "多组 ANOVA", "group": "差异与趋势",
     "method": "Welch ANOVA / Kruskal-Wallis + 事后检验（Tukey / Dunn）", "needs": ["groups"],
     "params": [_p("method", "方法", "select", "anova"), _p("posthoc", "事后检验", "select", "tukey")]},
    {"id": "trend_course", "title": "时间/剂量趋势", "group": "差异与趋势",
     "method": "分箱均值±SE + 线性/交换拟合（Fiehn 2008 代谢轨迹）", "needs": ["groups"],
     "params": [_p("time_col", "时间列", "text", None), _p("features", "特征列表", "text", [])]},
    {"id": "dose_response", "title": "剂量响应（EC50）", "group": "差异与趋势",
     "method": "4 参数 logistic 拟合（内置实现；R 的 drc 语义）", "needs": ["groups"],
     "params": [_p("dose_col", "剂量列", "text", None), _p("feature", "特征", "text", ""),
                _p("ci", "置信区间", "number", 0.95)]},
    # ---- 通路与化学分类 ----
    {"id": "mummichog_pathway", "title": "Mummichog 式通路富集", "group": "通路与化学分类",
     "method": "质量映射 + 置换检验（Li 2013 mummichog；Uppal 2017）", "needs": ["de"],
     "params": [_p("db", "数据库", "select", "kegg"), _p("n_perm", "置换次数", "number", 1000),
                _p("ppm", "质量精度(ppm)", "number", 10), _p("p_cut", "P 阈值", "number", 0.05)]},
    {"id": "ora_kegg", "title": "KEGG 通路 ORA", "group": "通路与化学分类",
     "method": "超几何检验 + BH（Kanehisa 2023 KEGG）", "needs": ["annotation"],
     "params": [_p("p_cut", "P 阈值", "number", 0.05), _p("min_size", "最小通路大小", "number", 3)]},
    {"id": "ora_hmdb_class", "title": "HMDB 化学分类富集", "group": "通路与化学分类",
     "method": "超几何检验（Wishart 2022 HMDB 5.0）", "needs": ["annotation"],
     "params": [_p("p_cut", "P 阈值", "number", 0.05)]},
    {"id": "metabolite_set_enrich", "title": "代谢物集富集（MSEA）", "group": "通路与化学分类",
     "method": "秩基 GSEA（Subramanian 2005；Xia 2015 MSEA）", "needs": ["de"],
     "params": [_p("n_perm", "置换次数", "number", 1000), _p("min_size", "最小集合大小", "number", 5)]},
    # ---- 标志物与建模 ----
    {"id": "rf_importance", "title": "随机森林重要度", "group": "标志物与建模",
     "method": "RF + 置换重要度（Breiman 2001）", "needs": ["groups"],
     "params": [_p("n_trees", "树数", "number", 300), _p("top_n", "Top N", "number", 20)]},
    {"id": "lasso_panel", "title": "LASSO 代谢物组合", "group": "标志物与建模",
     "method": "L1 logistic/回归 + 交叉验证选 λ（Tibshirani 1996）", "needs": ["groups"],
     "params": [_p("folds", "折数", "number", 5), _p("alpha", "λ 选择", "text", "cv")]},
    {"id": "roc_biomarker", "title": "单标志物 ROC", "group": "标志物与建模",
     "method": "ROC/AUC + DeLong 置信区间（DeLong 1988）", "needs": ["groups"],
     "params": [_p("feature", "特征", "text", ""), _p("ci", "置信区间", "bool", True)]},
    {"id": "biomarker_score", "title": "组合评分", "group": "标志物与建模",
     "method": "多标志物线性评分 + ROC + 分组（内置实现）", "needs": ["groups"],
     "params": [_p("features", "特征列表", "text", []), _p("weights", "权重", "select", "model")]},
    {"id": "model_validation", "title": "模型验证（自助法）", "group": "标志物与建模",
     "method": "bootstrap/置换 + 乐观度校正（Harrell 1996）", "needs": ["groups"],
     "params": [_p("n_boot", "自助次数", "number", 200), _p("folds", "折数", "number", 5)]},
    {"id": "survival_metabolite", "title": "代谢物分组生存", "group": "标志物与建模",
     "method": "中位/最优 cutoff + logrank + Cox（lifelines；Therneau 2000）", "needs": ["clinical"],
     "params": [_p("feature", "特征", "text", ""), _p("time_col", "时间列", "text", None),
                _p("event_col", "事件列", "text", None), _p("cutoff", "分组方式", "select", "median")]},
    # ---- 批次与网络 ----
    {"id": "rlsc_batch", "title": "QC-RLSC 批次校正", "group": "批次与网络",
     "method": "QC 样本 LOESS 信号校正（Dunn 2011）", "needs": ["qc"],
     "params": [_p("batch_key", "批次列", "text", None), _p("qc_label", "QC 样本标签", "text", ""),
                _p("span", "LOESS 跨度", "number", 0.75)]},
    {"id": "combat_batch", "title": "ComBat 批次校正", "group": "批次与网络",
     "method": "经验贝叶斯（Johnson 2007；内置实现）", "needs": ["qc"],
     "params": [_p("batch_key", "批次列", "text", None), _p("group_key", "分组列", "text", None)]},
    {"id": "correlation_network", "title": "共表达网络", "group": "批次与网络",
     "method": "|r| 阈值 + 模块（WGCNA 思想，Langfelder 2008；networkx）", "needs": ["transform"],
     "params": [_p("threshold", "相关阈值", "number", 0.6), _p("top_n", "最多特征", "number", 800)]},
    {"id": "metabolite_modules", "title": "代谢物模块", "group": "批次与网络",
     "method": "层次聚类/相似度切分 + 模块特征值（Langfelder 2008）", "needs": ["transform"],
     "params": [_p("n_modules", "模块数", "text", "auto"), _p("min_size", "最小模块", "number", 10)]},
    # ---- 代谢物鉴定 ----
    {"id": "adduct_annot", "title": "加合物推断", "group": "代谢物鉴定",
     "method": "常见加合物质量差匹配（正/负离子；MS-DIAL 语义）", "needs": ["mz"],
     "params": [_p("polarity", "极性", "select", "pos"), _p("ppm", "质量精度(ppm)", "number", 10),
                _p("adducts", "加合物列表", "text", [])]},
    {"id": "formula_candidates", "title": "分子式候选", "group": "代谢物鉴定",
     "method": "CHNOPS 组合 + 质量精度 + RDBE 过滤（七黄金规则，Kind & Fiehn 2007）", "needs": ["mz"],
     "params": [_p("ppm", "质量精度(ppm)", "number", 5), _p("max_c", "最大碳数", "number", 50),
                _p("hetero_atoms", "杂原子上限", "number", 6)]},
    {"id": "ms2_library_match", "title": "MS2 库匹配", "group": "代谢物鉴定",
     "method": "余弦相似度 + 前体匹配（MassBank/GNPS 语义；Stein 1994）", "needs": ["ms2"],
     "params": [_p("ppm", "前体精度(ppm)", "number", 20), _p("min_score", "最低得分", "number", 0.5),
                _p("top_n", "Top N", "number", 5)]},
    {"id": "ms2_network", "title": "分子网络", "group": "代谢物鉴定",
     "method": "余弦网络 + 连通分量（GNPS 分子网络；Wang 2016）", "needs": ["ms2"],
     "params": [_p("min_score", "最低相似度", "number", 0.6), _p("max_nodes", "最多节点", "number", 200)]},
]
ANALYSIS_INDEX = {item["id"]: item for item in ANALYSIS_ORDER}
ANALYSIS_GROUPS = [
    "质控与基础", "多变量判别", "差异与趋势", "通路与化学分类", "标志物与建模", "批次与网络", "代谢物鉴定",
]

# 图类型白名单（DESIGN §5，与 impl.mjs 的 PLOT_KINDS 保持一致）
PLOT_KINDS = [
    "qc_tic", "qc_rsd", "qc_missing", "sample_corr", "sample_dendro", "pca_scores", "pca_variance",
    "hotelling_t2", "plsda_scores", "oplsda_splot", "vip_bar", "permutation_test", "volcano_de",
    "fc_bar", "heatmap_top", "enrich_bar", "enrich_dot", "roc_curve", "rf_importance", "lasso_path",
    "trend_plot", "dose_response", "mass_accuracy", "mass_network", "batch_pca",
    "de_heatmap_top", "batch_effect", "batch_corrected_tic", "corr_heatmap", "corr_network",
    "km_metabolite", "calibration_curve",
]

# 数据格式兜底清单（mb_io 未就绪时给 UI 用；DESIGN §6）
FORMAT_FALLBACK = [
    {"id": "matrix_text", "label": "特征表（csv/tsv/txt/gz）", "hint": "样本×特征或特征×样本，含 mz/rt 列更好", "available": True},
    {"id": "xlsx", "label": "Excel（.xlsx/.xls）", "hint": "可指定 sheet；规则同特征表", "available": True},
    {"id": "geo_series", "label": "GEO series matrix", "hint": "含 !series_matrix_table_begin 的矩阵文件", "available": True},
    {"id": "archive", "label": "压缩包（zip/tar.gz）", "hint": "自动找特征表/ISA-Tab/mzTab/msp", "available": True},
    {"id": "mztab", "label": "mzTab", "hint": "MTD/PRH 开头的 mzTab 文件", "available": True},
    {"id": "isa_tab", "label": "ISA-Tab（MetaboLights）", "hint": "m_*.txt 矩阵 + s_*/a_* 表型", "available": True},
    {"id": "msms_library", "label": "MS2 谱库（msp/mgf）", "hint": "作为谱库挂载，不当定量矩阵", "available": True},
    {"id": "mzml_tic", "label": "mzML/mzXML", "hint": "轻量解析 TIC 与 MS2 谱图（不做峰检测）", "available": True},
    {"id": "rds", "label": "RDS", "hint": "R 桥可用时读取 matrix/data.frame", "available": True},
    {"id": "rdata", "label": "RData", "hint": "R 桥可用时读取", "available": True},
]

# 预设流程（DESIGN §2.2 recipe）
RECIPES = [
    {
        "id": "standard",
        "title": "标准代谢组学流程",
        "detail": "质控→过滤→填补→归一化→变换→PCA→PLS-DA→差异→富集→标志物",
        "steps": [
            {"step": "qc", "params": {}},
            {"step": "filter", "params": {}},
            {"step": "impute", "params": {"method": "half_min"}},
            {"step": "normalize", "params": {"method": "pqn"}},
            {"step": "transform", "params": {"transform": "log2", "scale": "auto"}},
            {"step": "pca", "params": {"n_comp": 3}},
            {"step": "plsda", "params": {"n_comp": 3, "cv_folds": 7, "permutations": 200}},
            {"step": "de", "params": {"method": "welch", "fdr": 0.05, "fc": 1.5}},
            {"step": "enrich", "params": {"method": "mummichog", "db": "kegg", "n_perm": 1000}},
            {"step": "biomarker", "params": {"model": "rf", "n_features": 10}},
        ],
    },
    {
        "id": "quick",
        "title": "快速差异",
        "detail": "质控→过滤→填补→归一化→变换→差异",
        "steps": [
            {"step": "qc", "params": {}},
            {"step": "filter", "params": {}},
            {"step": "impute", "params": {}},
            {"step": "normalize", "params": {"method": "pqn"}},
            {"step": "transform", "params": {}},
            {"step": "de", "params": {"method": "welch"}},
        ],
    },
    {
        "id": "batch",
        "title": "批次校正流程",
        "detail": "质控→过滤→填补→归一化→批次校正→变换→PCA→差异",
        "steps": [
            {"step": "qc", "params": {}},
            {"step": "filter", "params": {}},
            {"step": "impute", "params": {}},
            {"step": "normalize", "params": {"method": "pqn"}},
            {"step": "batch", "params": {"method": "rlsc"}},
            {"step": "transform", "params": {}},
            {"step": "pca", "params": {"n_comp": 3}},
            {"step": "de", "params": {"method": "welch"}},
        ],
    },
    {
        "id": "annotate",
        "title": "代谢物鉴定",
        "detail": "变换→加合物/分子式推断→（有 MS2 时）库匹配与分子网络",
        "steps": [
            {"step": "qc", "params": {}},
            {"step": "filter", "params": {}},
            {"step": "impute", "params": {}},
            {"step": "normalize", "params": {}},
            {"step": "transform", "params": {}},
            {"step": "annotate", "params": {"ppm": 10, "formula": True}},
        ],
    },
]

_RUNTIME: dict = {
    "modules": {},
    "plots": {},
    "plot_owner": {},
    "errors": {},
    "conflicts": [],
    "loaded_at": 0.0,
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_set(obj, name: str, value) -> None:
    try:
        setattr(obj, name, value)
    except Exception:  # noqa: BLE001 - 只读 property 等
        pass


def _safe_get(obj, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001
        return default


def column_kind(series) -> str:
    """样本信息列种类：categorical / numeric / time / event。"""
    try:
        name = str(getattr(series, "name", "") or "").lower()
        if series.dropna().empty:
            return "categorical"
        as_str = series.astype(str).str.lower().str.strip()
        uniq = set(as_str.dropna().unique())
        if uniq and uniq <= {"0", "1", "0.0", "1.0"} and any(p in name for p in ("event", "status", "dead", "censor")):
            return "event"
        if any(p in name for p in ("event", "status", "dead", "censor")) and pd.api.types.is_numeric_dtype(series):
            return "event"
        if any(p in name for p in ("time", "month", "day", "follow", "age", "dose", "hour")) and pd.api.types.is_numeric_dtype(series):
            return "time"
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 12:
            return "numeric"
        return "categorical"
    except Exception:  # noqa: BLE001
        return "categorical"


def _norm_params(params) -> list[dict]:
    """把模块导出的 params 统一成 [{key,label,type,default}]。

    支持三种写法（模块作者可以用最省事的那种）：
      * list[dict]：标准写法（DESIGN §2.5）
      * dict：{key: default} 简写
      * list[str]：只有 key
    """
    out: list[dict] = []
    if isinstance(params, dict):
        for key, default in params.items():
            out.append(_p(str(key), str(key), _guess_type(default), default))
        return out
    for item in params or []:
        if isinstance(item, str):
            out.append(_p(item, item))
            continue
        if not isinstance(item, dict):
            continue
        param = dict(item)
        key = str(param.get("key") or param.get("name") or "")
        if not key:
            continue
        param["key"] = key
        param.setdefault("label", param.get("title") or key)
        param.setdefault("type", _guess_type(param.get("default")))
        param.setdefault("default", None)
        out.append(param)
    return out


def _guess_type(default) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        return "number"
    if isinstance(default, (list, tuple)):
        return "list"
    return "text"


def _param_keys(item) -> set[str]:
    return {str(p.get("key")) for p in (item.get("params") or []) if isinstance(p, dict)}


# ---------------------------------------------------------------------------
# 模块加载
# ---------------------------------------------------------------------------


def load_modules(force: bool = False) -> dict:
    """导入全部步骤/分析/绘图模块，收集 STEPS / ANALYSES / PLOTS。

    单个模块缺失或 import 失败**只记录不抛出**：增量交付时插件必须整体可用。
    """
    if not force and _RUNTIME["modules"]:
        return _RUNTIME
    modules: dict[str, dict] = {}
    plots: dict[str, object] = {}
    plot_owner: dict[str, str] = {}
    errors: dict[str, str] = {}
    conflicts: list[str] = []
    for name in list(MODULES) + list(PLOT_MODULES):
        try:
            mod = importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - 缺文件/语法错误都只跳过
            modules[name] = {"steps": [], "analyses": [], "plots": [], "ok": False,
                             "error": f"{type(error).__name__}: {error}"}
            errors[name] = modules[name]["error"]
            continue
        steps = []
        for item in getattr(mod, "STEPS", []) or []:
            if not isinstance(item, dict) or "id" not in item or not callable(item.get("run")):
                continue
            steps.append({
                "id": str(item["id"]),
                "title": str(item.get("title") or item["id"]),
                "requires": [str(x) for x in (item.get("requires") or [])],
                "params": _norm_params(item.get("params")),
                "run": item["run"],
                "module": name,
            })
        analyses = []
        for item in getattr(mod, "ANALYSES", []) or []:
            if not isinstance(item, dict) or "id" not in item or not callable(item.get("run")):
                continue
            analyses.append({
                "id": str(item["id"]),
                "title": str(item.get("title") or item["id"]),
                "group": str(item.get("group") or "其它"),
                "method": str(item.get("method") or ""),
                "needs": [str(x) for x in (item.get("needs") or [])],
                "params": _norm_params(item.get("params")),
                "run": item["run"],
                "module": name,
            })
        mine: list[str] = []
        for kind, fn in (getattr(mod, "PLOTS", {}) or {}).items():
            kind = str(kind)
            if not callable(fn):
                continue
            if kind in plots:
                conflicts.append(f"{kind}: {plot_owner[kind]} 与 {name} 重复（保留前者）")
                continue
            plots[kind] = fn
            plot_owner[kind] = name
            mine.append(kind)
        modules[name] = {"steps": steps, "analyses": analyses, "plots": mine, "ok": True, "error": None}
    _RUNTIME["modules"] = modules
    _RUNTIME["plots"] = plots
    _RUNTIME["plot_owner"] = plot_owner
    _RUNTIME["errors"] = errors
    _RUNTIME["conflicts"] = conflicts
    _RUNTIME["loaded_at"] = time.time()
    return _RUNTIME


def module_status() -> dict:
    """每个模块的就绪状态（/health、/steps 与 ping 用）。"""
    data = load_modules()
    return {
        name: {
            "ok": bool(info.get("ok")),
            "error": info.get("error"),
            "steps": [s["id"] for s in info.get("steps", [])],
            "analyses": [a["id"] for a in info.get("analyses", [])],
            "plots": list(info.get("plots") or []),
        }
        for name, info in data["modules"].items()
    }


def missing_modules() -> list[str]:
    return [name for name, info in module_status().items() if not info.get("ok")]


def _module_steps() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for info in load_modules()["modules"].values():
        for step in info.get("steps", []):
            found.setdefault(step["id"], step)
    return found


def _module_analyses() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for info in load_modules()["modules"].values():
        for analysis in info.get("analyses", []):
            found.setdefault(analysis["id"], analysis)
    return found


def all_plots() -> dict:
    return dict(load_modules()["plots"])


# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------


def steps_catalog() -> list[dict]:
    """UI 左栏流程卡：12 项固定顺序 + 模块提供的参数与就绪状态。"""
    implemented = _module_steps()
    out: list[dict] = []
    for base in STEP_ORDER:
        item = {
            "id": base["id"],
            "title": base["title"],
            "requires": list(base["requires"]),
            "params": [dict(p) for p in base["params"]],
        }
        impl = implemented.get(base["id"])
        if impl:
            item["params"] = impl["params"] or item["params"]
            item["requires"] = impl["requires"] or item["requires"]
            item["available"] = True
            item["module"] = impl["module"]
        else:
            item["available"] = False
            item["module"] = None
            item["reason"] = "该步骤的模块尚未就绪"
        out.append(item)
    return out


def plots_catalog() -> list[dict]:
    data = load_modules()
    owner = data["plot_owner"]
    kinds = list(PLOT_KINDS) + [k for k in sorted(owner) if k not in PLOT_KINDS]
    return [
        {"id": kind, "owner": owner.get(kind), "available": kind in owner}
        for kind in kinds
    ]


def available_plots() -> list[str]:
    return sorted(all_plots().keys())


def analyses_catalog(dataset=None) -> dict:
    """全部分析 + 可用性。

    静态目录（DESIGN §4 的 36 项）永远出现在结果里：模块没就绪时 available=false
    并给出可读原因，UI 因此不会白屏或跳变。
    """
    impls = _module_analyses()
    items: list[dict] = []
    groups: list[str] = []

    def push(analysis_id: str, base: dict | None, impl: dict | None):
        needs = list((impl or {}).get("needs") or (base or {}).get("needs") or [])
        group = str((impl or {}).get("group") or (base or {}).get("group") or "其它")
        title = str((base or {}).get("title") or (impl or {}).get("title") or analysis_id)
        method = str((impl or {}).get("method") or (base or {}).get("method") or "")
        params = (impl or {}).get("params") or _norm_params((base or {}).get("params") or [])
        if impl is None:
            ok, reason = False, "该分析的模块尚未就绪"
        else:
            ok, reason = needs_ok(dataset, needs)
        item = {
            "id": analysis_id,
            "title": title,
            "group": group,
            "method": method,
            "needs": needs,
            "params": params,
            "available": bool(ok),
            "reason": None if ok else reason,
            "module": (impl or {}).get("module"),
        }
        if group not in groups:
            groups.append(group)
        items.append(item)

    seen: set[str] = set()
    for base in ANALYSIS_ORDER:
        analysis_id = base["id"]
        seen.add(analysis_id)
        push(analysis_id, base, impls.get(analysis_id))
    for analysis_id, impl in impls.items():
        if analysis_id in seen:
            continue
        push(analysis_id, None, impl)

    def sort_key(item):
        group = item["group"]
        gidx = ANALYSIS_GROUPS.index(group) if group in ANALYSIS_GROUPS else len(ANALYSIS_GROUPS) + groups.index(group)
        order = {base["id"]: i for i, base in enumerate(ANALYSIS_ORDER)}
        return (gidx, order.get(item["id"], 999), item["title"])

    groups.sort(key=lambda g: (ANALYSIS_GROUPS.index(g) if g in ANALYSIS_GROUPS else len(ANALYSIS_GROUPS) + groups.index(g), g))
    items.sort(key=sort_key)
    return {"analyses": items, "groups": groups}


def formats_catalog() -> dict:
    """数据格式目录（优先用 mb_io 的 FORMAT_INFO）。"""
    formats = []
    ids = []
    try:
        import mb_io  # noqa: PLC0415 - 懒导入：mb_io 未就绪也不能影响其它命令

        info = getattr(mb_io, "FORMAT_INFO", None)
        ids = [str(x) for x in (getattr(mb_io, "SUPPORTED_FORMATS", []) or [])]
        if isinstance(info, dict):
            for key, value in info.items():
                if isinstance(value, dict):
                    item = {"id": str(key), **{k: v for k, v in value.items() if k != "id"}}
                else:
                    item = {"id": str(key), "label": str(value)}
                formats.append(item)
        elif isinstance(info, (list, tuple)):
            for value in info:
                if isinstance(value, dict):
                    formats.append(dict(value))
                else:
                    formats.append({"id": str(value), "label": str(value)})
    except Exception:  # noqa: BLE001 - mb_io 缺失/导入失败
        formats = []
    if not formats:
        formats = [dict(item) for item in FORMAT_FALLBACK]
        ids = [item["id"] for item in formats]
    for item in formats:
        item.setdefault("label", item.get("id"))
        item.setdefault("hint", "")
        item.setdefault("available", True)
    if not ids:
        ids = [item.get("id") for item in formats if item.get("id")]
    return {"formats": formats, "ids": ids}


# ---------------------------------------------------------------------------
# 数据集字段访问（mb_state 的 Dataset；缺字段时回退，不崩）
# ---------------------------------------------------------------------------


def ensure_fields(dataset) -> None:
    """补齐 registry 依赖的可变容器字段（mb_state 版本差异兜底）。"""
    if dataset is None:
        return
    if not isinstance(_safe_get(dataset, "steps"), list):
        _safe_set(dataset, "steps", [])
    if not isinstance(_safe_get(dataset, "history"), list):
        _safe_set(dataset, "history", [])
    if not isinstance(_safe_get(dataset, "analyses_run"), list):
        _safe_set(dataset, "analyses_run", [])
    if not isinstance(_safe_get(dataset, "notes"), list):
        _safe_set(dataset, "notes", [])
    if not isinstance(_safe_get(dataset, "warnings"), list):
        _safe_set(dataset, "warnings", [])
    if not isinstance(_safe_get(dataset, "results"), dict):
        _safe_set(dataset, "results", {})
    if not _safe_get(dataset, "X_kind"):
        _safe_set(dataset, "X_kind", "raw")
    if _safe_get(dataset, "current", None) is None and _safe_get(dataset, "steps"):
        _safe_set(dataset, "current", dataset.steps[-1])


def sample_info(dataset) -> pd.DataFrame:
    frame = _safe_get(dataset, "sample_info")
    samples = [str(s) for s in (_safe_get(dataset, "samples") or [])]
    if isinstance(frame, pd.DataFrame):
        if list(map(str, frame.index)) != samples:
            try:
                frame = frame.reindex(pd.Index(samples, name=getattr(frame.index, "name", None) or "sample"))
            except Exception:  # noqa: BLE001
                frame = pd.DataFrame(index=pd.Index(samples, name="sample"))
        return frame
    return pd.DataFrame(index=pd.Index(samples, name="sample"))


def feature_info(dataset) -> pd.DataFrame:
    frame = _safe_get(dataset, "feature_info")
    features = [str(f) for f in (_safe_get(dataset, "features") or [])]
    if isinstance(frame, pd.DataFrame):
        if list(map(str, frame.index)) != features:
            try:
                frame = frame.reindex(pd.Index(features, name=getattr(frame.index, "name", None) or "feature"))
            except Exception:  # noqa: BLE001
                frame = pd.DataFrame(index=pd.Index(features, name="feature"))
        return frame
    return pd.DataFrame(index=pd.Index(features, name="feature"))


def feature_meta(dataset) -> dict:
    """DESIGN §2.4 的 feature_meta：has_mz / has_rt / n_annotated。"""
    info = feature_info(dataset)
    has_mz = bool("mz" in info.columns and info["mz"].notna().any()) if len(info) else False
    has_rt = bool("rt" in info.columns and info["rt"].notna().any()) if len(info) else False
    n_annotated = 0
    for col in ("name", "hmdb", "formula", "class", "kegg", "hmdb_id"):
        if col in info.columns:
            n_annotated = max(n_annotated, int(info[col].notna().sum()))
    return {"has_mz": has_mz, "has_rt": has_rt, "n_annotated": int(n_annotated)}


def library_of(dataset):
    """MS2 谱库（懒加载结构；DESIGN §2.3 library）。"""
    lib = _safe_get(dataset, "library")
    return lib if isinstance(lib, dict) and lib.get("spectra") else None


def n_spectra(dataset) -> int:
    lib = library_of(dataset)
    return len(lib.get("spectra") or []) if lib else 0


def clinical_columns(dataset) -> dict:
    """找生存时间/事件列（供 needs=clinical 与 survival 用）。"""
    info = sample_info(dataset)
    time_pat = ("time", "os_time", "dfs_time", "rfs_time", "pfs_time", "survival", "month", "days", "follow")
    event_pat = ("event", "status", "dead", "death", "censor", "vital")
    time_col = event_col = None
    for col in info.columns:
        key = str(col).lower()
        if time_col is None and any(p in key for p in time_pat) and pd.api.types.is_numeric_dtype(info[col]):
            time_col = str(col)
        if event_col is None and any(p in key for p in event_pat):
            event_col = str(col)
    return {"time": time_col, "event": event_col}


def _has_nan(dataset) -> bool:
    X = _safe_get(dataset, "X")
    if X is None:
        return False
    try:
        arr = np.asarray(X)
        if arr.size > 5e7:
            sample = arr.reshape(-1)[:: max(1, arr.size // 1000000)]
            return bool(np.isnan(sample.astype(np.float64, copy=False)).any())
        return bool(np.isnan(arr.astype(np.float64, copy=False)).any())
    except Exception:  # noqa: BLE001
        return False


def needs_ok(dataset, needs) -> tuple[bool, str | None]:
    """语义前置判断（DESIGN §2.5 的 needs 取值）。"""
    normalized = [str(n).strip() for n in (needs or []) if str(n).strip()]
    if not normalized:
        return True, None
    if dataset is None:
        return False, "还没有载入数据集"
    ensure_fields(dataset)
    steps = set(_safe_get(dataset, "steps") or [])
    results = _safe_get(dataset, "results") or {}
    x_kind = str(_safe_get(dataset, "X_kind") or "").lower()
    for need in normalized:
        if need in ("groups", "group"):
            key = _safe_get(dataset, "group_key")
            if not key:
                return False, "需要先指定分组列（样本 → 设置分组）"
            info = sample_info(dataset)
            if key not in info.columns:
                return False, f"分组列「{key}」不在样本信息里"
            n_groups = int(info[key].nunique(dropna=True))
            if n_groups < 2:
                return False, f"分组列「{key}」只有 {n_groups} 组，至少需要 2 组"
            continue
        if need in ("transform", "normalize", "scaled", "lognorm"):
            if need in steps or x_kind in ("lognorm", "scaled", "normalized", "log2"):
                continue
            return False, "需要先完成「变换/标度」步骤"
        if need == "impute":
            if "impute" in steps or not _has_nan(dataset):
                continue
            return False, "矩阵里还有缺失值，需要先跑「缺失值填补」"
        if need in ("qc", "filter", "pca", "batch", "biomarker", "annotate"):
            if need in steps:
                continue
            return False, f"需要先完成「{STEP_TITLES.get(need, need)}」步骤"
        if need == "plsda":
            if "plsda" in steps or "plsda" in results or "vip" in results:
                continue
            return False, "需要先跑「PLS-DA / OPLS-DA」"
        if need == "de":
            if "de" in steps or "de" in results:
                continue
            return False, "需要先跑「单变量差异」"
        if need == "vip":
            if "vip" in results or "plsda" in results or "plsda" in steps:
                continue
            return False, "需要先跑「PLS-DA / OPLS-DA」得到 VIP"
        if need == "enrich":
            if "enrich" in steps or "enrich" in results:
                continue
            return False, "需要先跑「通路富集」"
        if need == "annotation":
            if feature_meta(dataset).get("n_annotated", 0) > 0:
                continue
            return False, "特征表里没有代谢物注释（name/hmdb/formula/class），先导入带注释的特征表或跑「代谢物鉴定」"
        if need == "clinical":
            cols = clinical_columns(dataset)
            if cols.get("time") and cols.get("event"):
                continue
            return False, "样本信息里缺少生存时间/事件列，请先导入临床表"
        if need == "ms2":
            if n_spectra(dataset) > 0:
                continue
            return False, "还没有 MS2 谱库，请导入 msp/mgf 或带谱图的 mzML"
        if need == "mz":
            if feature_meta(dataset).get("has_mz"):
                continue
            return False, "特征表里没有 m/z（需要带 mz/rt 列的特征表）才能做加合物/分子式推断"
        # 未知 need：不阻塞（向前兼容）
        continue
    return True, None


def group_labels(dataset):
    """当前分组标签数组（无分组时返回 None）。"""
    key = _safe_get(dataset, "group_key")
    if not key:
        return None
    info = sample_info(dataset)
    if key not in info.columns:
        return None
    return info[key].astype(str).to_numpy()


def set_group(dataset, column: str) -> None:
    """把 sample_info 的某列设为分组列。"""
    column = str(column or "")
    info = sample_info(dataset)
    if column not in info.columns:
        raise EngineError("bad_request", f"样本信息里没有列「{column}」（可选：{', '.join(map(str, info.columns)) or '无'}）")
    values = [str(v) for v in info[column].tolist()]
    groups: list[str] = []
    for value in values:
        if value not in groups:
            groups.append(value)
    _safe_set(dataset, "group_key", column)
    _safe_set(dataset, "groups", groups)


def guess_group(dataset) -> str | None:
    """自动挑一个像分组的列（组数 2–20、非批次/QC/时间列）。"""
    info = sample_info(dataset)
    prefer = ("group", "groups", "condition", "treatment", "class", "type", "phenotype",
              "disease", "status", "arm", "cohort", "label", "sample_group", "category")
    avoid = ("batch", "plate", "run", "qc", "sample", "id", "time", "age", "bmi", "dose", "event")
    best = None
    best_score = -1
    for col in info.columns:
        key = str(col).lower()
        try:
            nunique = int(info[col].nunique(dropna=True))
        except Exception:  # noqa: BLE001
            continue
        if not (1 < nunique <= 20):
            continue
        score = 0
        if key in prefer:
            score += 10
        elif any(p in key for p in prefer):
            score += 6
        if any(p in key for p in avoid):
            score -= 8
        if score > best_score:
            best_score = score
            best = str(col)
    return best if best_score > 0 else None


# ---------------------------------------------------------------------------
# 结果应用（replace / store / sample_columns）与步骤记录
# ---------------------------------------------------------------------------


def _normalize_store(value):
    """把模块返回的缓存值统一成可长期保存的结构（DataFrame 留在内存）。"""
    if not isinstance(value, dict):
        return value
    out = dict(value)
    for key in ("table", "tables", "de_table", "vip_table", "enrich_table", "biomarker_table", "rows"):
        frame = out.get(key)
        if isinstance(frame, pd.DataFrame) or frame is None:
            continue
        if isinstance(frame, list) and frame and isinstance(frame[0], dict):
            try:
                out[key] = pd.DataFrame(frame)
            except Exception:  # noqa: BLE001
                continue
    return out


def _apply_replace(dataset, replace: dict) -> None:
    """应用模块返回的 replace（§2.5：{X, features, samples, raw, value_kind}）。"""
    if not isinstance(replace, dict) or not replace:
        return
    ensure_fields(dataset)
    samples = [str(s) for s in (_safe_get(dataset, "samples") or [])]
    features = [str(f) for f in (_safe_get(dataset, "features") or [])]
    new_X = replace.get("X")
    if new_X is not None:
        arr = np.asarray(new_X, dtype=np.float32)
        if arr.ndim != 2:
            raise EngineError("internal", "replace.X 必须是二维矩阵")
        new_samples = [str(s) for s in (replace.get("samples") or samples)]
        new_features = [str(f) for f in (replace.get("features") or features)]
        if arr.shape != (len(new_samples), len(new_features)):
            raise EngineError(
                "internal",
                f"replace 矩阵形状 {arr.shape} 与样本/特征数（{len(new_samples)}, {len(new_features)}）不一致",
            )
        _safe_set(dataset, "X", arr)
        if replace.get("samples") is not None:
            _safe_set(dataset, "samples", new_samples)
        if replace.get("features") is not None:
            _safe_set(dataset, "features", new_features)
        old_info = _safe_get(dataset, "feature_info")
        if isinstance(old_info, pd.DataFrame) and replace.get("features") is not None:
            try:
                keep = [f for f in new_features if f in set(map(str, old_info.index))]
                rebuilt = old_info.loc[keep].reindex(pd.Index(new_features))
                _safe_set(dataset, "feature_info", rebuilt)
            except Exception:  # noqa: BLE001
                pass
        old_samples_info = _safe_get(dataset, "sample_info")
        if isinstance(old_samples_info, pd.DataFrame) and replace.get("samples") is not None:
            try:
                _safe_set(dataset, "sample_info",
                          old_samples_info.reindex(pd.Index(new_samples, name=getattr(old_samples_info.index, "name", None) or "sample")))
            except Exception:  # noqa: BLE001
                pass
    for attr in ("raw",):
        value = replace.get(attr)
        if value is not None:
            try:
                _safe_set(dataset, attr, np.asarray(value, dtype=np.float32))
            except Exception:  # noqa: BLE001
                pass
    kind = replace.get("value_kind") or replace.get("X_kind")
    if kind:
        _safe_set(dataset, "X_kind", str(kind))


def _apply_sample_columns(dataset, columns, values) -> None:
    """把模块返回的 sample_columns + sample_values 写进 sample_info。"""
    if not columns or not isinstance(values, dict):
        return
    names: list[str] = []
    for item in columns:
        if isinstance(item, dict):
            key = item.get("key") or item.get("name")
        else:
            key = item
        if key:
            names.append(str(key))
    info = sample_info(dataset)
    changed = []
    for name in names:
        series = values.get(name)
        if series is None:
            continue
        try:
            arr = list(series)
        except TypeError:
            continue
        if len(arr) != len(info):
            continue
        try:
            info[name] = arr
            changed.append(name)
        except Exception:  # noqa: BLE001
            continue
    if changed:
        _safe_set(dataset, "sample_info", info)
    if "group" in changed and _safe_get(dataset, "group_key") is None:
        try:
            set_group(dataset, "group")
        except EngineError:
            pass


def _apply_result(dataset, payload: dict) -> dict:
    """应用结果副作用（replace / store / sample_columns）并返回干净的结果信封。"""
    if not isinstance(payload, dict):
        return {"summary": {"title": "完成", "bullets": [], "metrics": {}},
                "tables": [], "figures": [], "warnings": [], "notes": []}
    outgoing = dict(payload)
    ensure_fields(dataset)
    _apply_replace(dataset, outgoing.pop("replace", None) or outgoing.pop("_replace", None) or {})
    store: dict = {}
    for key in ("de_store", "result_store", "store", "_store"):
        value = outgoing.pop(key, None)
        if isinstance(value, dict):
            if key == "de_store":
                store["de"] = value
            else:
                store.update(value)
    results = _safe_get(dataset, "results") or {}
    for store_key, value in store.items():
        try:
            results[str(store_key)] = _normalize_store(value)
        except Exception:  # noqa: BLE001
            continue
    _safe_set(dataset, "results", results)
    _apply_sample_columns(dataset, outgoing.get("sample_columns"), outgoing.get("sample_values"))
    summary = outgoing.get("summary")
    if not isinstance(summary, dict):
        summary = {"title": "", "bullets": [], "metrics": {}}
    summary.setdefault("title", "")
    summary.setdefault("bullets", [])
    summary.setdefault("metrics", {})
    outgoing["summary"] = summary
    for key in ("tables", "figures", "warnings", "notes"):
        if not isinstance(outgoing.get(key), list):
            outgoing[key] = []
    return jsonable(outgoing)


def _add_history(dataset, step: str, params: dict, ok: bool, ms: int, summary: str = "") -> None:
    """追加 history（§2.4 结构）；mb_state 有 add_history 时优先用它。"""
    item = {
        "at": now_iso(),
        "step": str(step),
        "params": jsonable(dict(params or {})),
        "ok": bool(ok),
        "ms": int(ms),
        "summary": str(summary or "")[:400],
    }
    add = _safe_get(dataset, "add_history")
    if callable(add):
        try:
            add(step, dict(params or {}), bool(ok), int(ms), summary=str(summary or ""))
            return
        except TypeError:
            pass
        except Exception:  # noqa: BLE001
            return
    history = _safe_get(dataset, "history")
    if not isinstance(history, list):
        history = []
        _safe_set(dataset, "history", history)
    history.append(item)


def _add_analysis_run(dataset, analysis_id: str, ok: bool, ms: int, summary: str = "") -> None:
    item = {"at": now_iso(), "id": str(analysis_id), "ok": bool(ok), "ms": int(ms),
            "summary": str(summary or "")[:400]}
    add = _safe_get(dataset, "add_analysis_run")
    if callable(add):
        try:
            add(analysis_id, bool(ok), int(ms), str(summary or ""))
            return
        except TypeError:
            pass
        except Exception:  # noqa: BLE001
            return
    runs = _safe_get(dataset, "analyses_run")
    if not isinstance(runs, list):
        runs = []
        _safe_set(dataset, "analyses_run", runs)
    runs.append(item)


def _mark_step(dataset, step_id: str) -> None:
    steps = [s for s in (_safe_get(dataset, "steps") or []) if str(s) != step_id]
    steps.append(step_id)
    try:
        steps.sort(key=lambda s: STEP_INDEX.get(str(s), 999))
    except Exception:  # noqa: BLE001
        pass
    _safe_set(dataset, "steps", steps)
    _safe_set(dataset, "current", step_id)


def _downstream_steps(step_id: str) -> list[str]:
    """require 关系里的传递下游（重跑某步时清空）。"""
    children: dict[str, list[str]] = {}
    for base in STEP_ORDER:
        for req in base["requires"]:
            children.setdefault(req, []).append(base["id"])
    out: list[str] = []
    stack = list(children.get(step_id, []))
    while stack:
        current = stack.pop()
        if current in out:
            continue
        out.append(current)
        stack.extend(children.get(current, []))
    return out


def _clear_downstream(dataset, step_id: str) -> list[str]:
    """重跑某步：清空它的下游步骤与对应结果缓存（history 保留；DESIGN §3）。"""
    downstream = set(_downstream_steps(step_id))
    if not downstream:
        return []
    steps = [s for s in (_safe_get(dataset, "steps") or []) if s not in downstream]
    _safe_set(dataset, "steps", steps)
    results = _safe_get(dataset, "results") or {}
    dropped = []
    for item in downstream:
        for key in STEP_RESULT_KEYS.get(item, [item]):
            if key in results:
                results.pop(key, None)
                dropped.append(key)
    _safe_set(dataset, "results", results)
    return sorted(downstream)


# ---------------------------------------------------------------------------
# ctx（DESIGN §2.3）
# ---------------------------------------------------------------------------


def build_ctx(dataset, outdir: str, engine_info: dict | None = None) -> dict:
    """构造传给模块的 ctx。模块**只读**，写回走 replace / store / sample_columns。"""
    ensure_fields(dataset)
    ensure_dir(outdir)
    workdir = os.path.dirname(os.path.dirname(os.path.abspath(outdir))) or outdir
    info = engine_info or {}
    return {
        "dataset_id": _safe_get(dataset, "id"),
        "name": _safe_get(dataset, "name"),
        "features": list(_safe_get(dataset, "features") or []),
        "samples": list(_safe_get(dataset, "samples") or []),
        "X": _safe_get(dataset, "X"),
        "X_kind": str(_safe_get(dataset, "X_kind") or "raw"),
        "raw": _safe_get(dataset, "raw"),
        "feature_info": feature_info(dataset),
        "sample_info": sample_info(dataset),
        "group_key": _safe_get(dataset, "group_key"),
        "groups": list(_safe_get(dataset, "groups") or []) or None,
        "matrix_order": "samples_by_features",
        "species": str(_safe_get(dataset, "species") or "unknown"),
        "polarity": _safe_get(dataset, "polarity"),
        "results": _safe_get(dataset, "results") or {},
        "library": library_of(dataset),
        "rng": np.random.default_rng(SEED),
        "workdir": workdir,
        "outdir": outdir,
        "engine": {"python": str(info.get("python") or ""), "mode": str(info.get("mode") or "builtin")},
        "notes": list(_safe_get(dataset, "notes") or []),
        "warnings": list(_safe_get(dataset, "warnings") or []),
    }


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------


def _inject_group_col(dataset, item: dict, params: dict) -> dict:
    keys = _param_keys(item)
    group_key = _safe_get(dataset, "group_key")
    if group_key and "group_col" in keys and not params.get("group_col"):
        params["group_col"] = group_key
    return params


def run_step(dataset, step_id: str, params: dict, workdir: str, engine_info: dict | None = None) -> dict:
    """执行一个流程步骤（DESIGN §3）。"""
    if dataset is None:
        raise EngineError("not_found", "还没有载入数据集")
    impl = _module_steps().get(step_id)
    if impl is None:
        if step_id in STEP_INDEX:
            raise EngineError("unsupported", f"步骤「{STEP_TITLES.get(step_id, step_id)}」的模块尚未就绪")
        raise EngineError("bad_request", f"未知步骤：{step_id}（可选：{', '.join(STEP_INDEX)}）")
    ensure_fields(dataset)
    requires = impl["requires"] or next((b["requires"] for b in STEP_ORDER if b["id"] == step_id), [])
    missing = [r for r in requires if r not in (_safe_get(dataset, "steps") or [])]
    if missing:
        raise EngineError("prerequisite", f"需要先完成：{'、'.join(STEP_TITLES.get(m, m) for m in missing)}")
    outdir = ensure_dir(os.path.join(workdir, "analyses", str(_safe_get(dataset, "id") or "ds"), f"step-{step_id}"))
    ctx = build_ctx(dataset, outdir, engine_info)
    clean_params = {k: v for k, v in dict(params or {}).items() if v is not None}
    clean_params = _inject_group_col(dataset, impl, clean_params)
    started = time.time()
    try:
        payload = impl["run"](ctx, clean_params, outdir)
    except EngineError as error:
        elapsed = int((time.time() - started) * 1000)
        _add_history(dataset, step_id, clean_params, False, elapsed, summary=error.message)
        raise
    elapsed = int((time.time() - started) * 1000)
    clean = _apply_result(dataset, payload)
    cleared = _clear_downstream(dataset, step_id)
    _mark_step(dataset, step_id)
    bullets = (clean.get("summary") or {}).get("bullets") or []
    _add_history(dataset, step_id, clean_params, True, elapsed, summary=str(bullets[0]) if bullets else "")
    out = dict(clean)
    out.update({
        "step": step_id,
        "steps": list(_safe_get(dataset, "steps") or []),
        "current": _safe_get(dataset, "current"),
        "dataset": dataset_summary(dataset),
        "cleared": cleared,
        "ms": elapsed,
    })
    return out


def run_analysis(dataset, analysis_id: str, params: dict, workdir: str, engine_info: dict | None = None) -> dict:
    """执行一项高级分析（DESIGN §4）。"""
    if dataset is None:
        raise EngineError("not_found", "还没有载入数据集")
    impl = _module_analyses().get(analysis_id)
    if impl is None:
        base = ANALYSIS_INDEX.get(analysis_id)
        if base is not None:
            raise EngineError("unsupported", f"分析「{base['title']}」的模块尚未就绪")
        raise EngineError("not_found", f"分析「{analysis_id}」不存在")
    ok, reason = needs_ok(dataset, impl["needs"])
    if not ok:
        raise EngineError("prerequisite", reason or "缺少前置数据")
    outdir = ensure_dir(os.path.join(workdir, "analyses", str(_safe_get(dataset, "id") or "ds"), f"an-{analysis_id}"))
    ctx = build_ctx(dataset, outdir, engine_info)
    clean_params = {k: v for k, v in dict(params or {}).items() if v is not None}
    clean_params = _inject_group_col(dataset, impl, clean_params)
    if "time_col" in _param_keys(impl) and not clean_params.get("time_col"):
        cols = clinical_columns(dataset)
        if cols.get("time"):
            clean_params["time_col"] = cols["time"]
    if "event_col" in _param_keys(impl) and not clean_params.get("event_col"):
        cols = clinical_columns(dataset)
        if cols.get("event"):
            clean_params["event_col"] = cols["event"]
    started = time.time()
    try:
        payload = impl["run"](ctx, clean_params, outdir)
    except EngineError as error:
        _add_analysis_run(dataset, analysis_id, False, int((time.time() - started) * 1000), error.message)
        raise
    elapsed = int((time.time() - started) * 1000)
    clean = _apply_result(dataset, payload)
    bullets = (clean.get("summary") or {}).get("bullets") or []
    _add_analysis_run(dataset, analysis_id, True, elapsed, str(bullets[0]) if bullets else "")
    out = dict(clean)
    out.update({
        "analysis_id": analysis_id,
        "title": impl["title"],
        "group": impl["group"],
        "method": impl["method"],
        "dataset": dataset_summary(dataset),
        "ms": elapsed,
    })
    return out


def run_plot(dataset, kind: str, params: dict, workdir: str, engine_info: dict | None = None) -> dict:
    """按 kind 分发到模块导出的 PLOTS 处理函数（DESIGN §5）。"""
    if dataset is None:
        raise EngineError("not_found", "还没有载入数据集")
    data = load_modules()
    handler = data["plots"].get(kind)
    if handler is None:
        known = ", ".join(sorted(data["plots"])) or "无"
        raise EngineError("unsupported", f"图类型「{kind}」尚未实现（对应模块未就绪）；当前可用：{known}")
    outdir = ensure_dir(os.path.join(workdir, "analyses", str(_safe_get(dataset, "id") or "ds"), "plots"))
    ctx = build_ctx(dataset, outdir, engine_info)
    clean_params = {k: v for k, v in dict(params or {}).items() if v is not None}
    fig = handler(ctx, clean_params, outdir)
    if not isinstance(fig, dict):
        raise EngineError("internal", f"图类型「{kind}」的返回不是 figure dict")
    path = fig.get("file")
    if not path or not os.path.exists(path):
        raise EngineError("internal", f"图类型「{kind}」没有生成 PNG 文件")
    fig.setdefault("id", slug(kind))
    fig.setdefault("title", kind)
    fig["kind"] = kind
    try:
        fig["bytes"] = os.path.getsize(path)
    except OSError:
        fig["bytes"] = None
    return jsonable(fig)


# ---------------------------------------------------------------------------
# 数据集构造（mb_io → mb_state.Dataset）
# ---------------------------------------------------------------------------


def _state_new(state, payload: dict):
    """调用 state.new(**payload)，按签名过滤不接受的键（mb_state 版本差异兜底）。"""
    try:
        import inspect

        signature = inspect.signature(state.new)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        if not accepts_kwargs:
            payload = {k: v for k, v in payload.items() if k in signature.parameters}
    except Exception:  # noqa: BLE001
        pass
    return state.new(**payload)


def _frame_or_empty(frame, index: list[str], name: str) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        out = frame.copy()
        out.index = [str(i) for i in out.index]
        if len(out) and set(out.index) != set(index):
            try:
                out = out.reindex(pd.Index(index, name=getattr(out.index, "name", None) or name))
            except Exception:  # noqa: BLE001
                pass
        return out
    return pd.DataFrame(index=pd.Index(index, name=name))


def build_dataset(state, loaded: dict, name: str | None = None, source: str | None = None,
                  options: dict | None = None):
    """把 mb_io.load_source 的返回值变成 mb_state.Dataset（engine 与导出脚本共用）。"""
    options = dict(options or {})
    if not isinstance(loaded, dict):
        raise EngineError("internal", "导入结果格式错误")
    meta = dict(loaded.get("meta") or {})
    X = loaded.get("X")
    if X is None:
        raise EngineError("bad_request", "导入结果缺少强度矩阵（X）")
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise EngineError("bad_request", "强度矩阵必须是二维（样本 × 特征）")
    features = [str(f) for f in (loaded.get("features") or [])]
    samples = [str(s) for s in (loaded.get("samples") or [])]
    if len(features) != arr.shape[1]:
        raise EngineError("bad_request", f"特征数（{len(features)}）与矩阵列数（{arr.shape[1]}）不一致")
    if len(samples) != arr.shape[0]:
        raise EngineError("bad_request", f"样本数（{len(samples)}）与矩阵行数（{arr.shape[0]}）不一致")
    seen: dict[str, int] = {}
    deduped = []
    for sample in samples:
        if sample in seen:
            seen[sample] += 1
            deduped.append(f"{sample}_{seen[sample]}")
        else:
            seen[sample] = 0
            deduped.append(sample)
    samples = deduped
    raw = loaded.get("raw")
    raw_arr = np.asarray(raw, dtype=np.float32) if raw is not None else arr.copy()
    x_kind = str(loaded.get("X_kind") or meta.get("X_kind") or meta.get("value_kind") or "raw")
    if source:
        base = os.path.basename(str(source).rstrip(os.sep)) or "dataset"
    else:
        base = "dataset"
    ds_name = str(name or loaded.get("name") or base)
    feature_info = _frame_or_empty(loaded.get("feature_info"), features, "feature")
    sample_info = _frame_or_empty(loaded.get("sample_info"), samples, "sample")
    for col, default in (("mz", None), ("rt", None), ("name", None), ("formula", None),
                         ("adduct", None), ("hmdb", None), ("kegg", None), ("class", None),
                         ("polarity", None), ("confidence", None)):
        if col not in feature_info.columns:
            feature_info[col] = default
    payload = {
        "name": ds_name,
        "path": str(source) if source else None,
        "source_path": str(source) if source else None,
        "source_format": str(loaded.get("format") or meta.get("format") or "unknown"),
        "X": arr,
        "raw": raw_arr,
        "X_kind": x_kind,
        "value_kind": x_kind,
        "features": features,
        "genes": features,
        "samples": samples,
        "feature_info": feature_info,
        "sample_info": sample_info,
        "matrix_order": "samples_by_features",
        "species": str(meta.get("species") or "unknown"),
        "polarity": meta.get("polarity"),
        "library": loaded.get("library") or meta.get("library"),
        "notes": list(loaded.get("notes") or []),
        "warnings": list(loaded.get("warnings") or []),
    }
    dataset = _state_new(state, payload)
    ensure_fields(dataset)
    group_col = options.get("group_col")
    if group_col and str(group_col) in sample_info(dataset).columns:
        set_group(dataset, str(group_col))
    elif _safe_get(dataset, "group_key") is None:
        guess = guess_group(dataset)
        if guess:
            set_group(dataset, guess)
    return dataset


def dataset_summary(dataset) -> dict:
    """dataset 摘要（§2.4）：优先用 mb_state 的 summary()。"""
    if dataset is None:
        return {}
    fn = _safe_get(dataset, "summary")
    if callable(fn):
        try:
            out = fn()
            if isinstance(out, dict):
                out.setdefault("n_samples", len(_safe_get(dataset, "samples") or []))
                out.setdefault("n_features", len(_safe_get(dataset, "features") or []))
                out.setdefault("steps", list(_safe_get(dataset, "steps") or []))
                out.setdefault("current", _safe_get(dataset, "current"))
                out.setdefault("feature_meta", feature_meta(dataset))
                return jsonable(out)
        except Exception:  # noqa: BLE001
            pass
    info = sample_info(dataset)
    columns = [{"key": str(c), "kind": column_kind(info[c]), "n_unique": int(info[c].nunique(dropna=True))}
               for c in info.columns]
    return jsonable({
        "id": _safe_get(dataset, "id"),
        "name": _safe_get(dataset, "name"),
        "path": _safe_get(dataset, "path") or _safe_get(dataset, "source_path"),
        "source_format": _safe_get(dataset, "source_format") or "unknown",
        "n_samples": len(_safe_get(dataset, "samples") or []),
        "n_features": len(_safe_get(dataset, "features") or []),
        "species": _safe_get(dataset, "species") or "unknown",
        "polarity": _safe_get(dataset, "polarity"),
        "feature_meta": feature_meta(dataset),
        "sample_columns": columns,
        "steps": list(_safe_get(dataset, "steps") or []),
        "current": _safe_get(dataset, "current"),
        "warnings": list(_safe_get(dataset, "warnings") or []),
        "notes": list(_safe_get(dataset, "notes") or []),
        "history": list(_safe_get(dataset, "history") or [])[-20:],
        "analyses_run": list(_safe_get(dataset, "analyses_run") or [])[-20:],
    })


# ---------------------------------------------------------------------------
# 样本表操作（DESIGN §2.2 samples）
# ---------------------------------------------------------------------------


def _read_info_table(path: str) -> pd.DataFrame:
    path = str(path)
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(path, index_col=0)
    else:
        sep = "\t" if lower.endswith((".tsv", ".txt")) else ","
        try:
            frame = pd.read_csv(path, sep=sep, index_col=0)
        except Exception:  # noqa: BLE001
            frame = pd.read_csv(path, sep=None, engine="python", index_col=0)
    frame.columns = [str(c) for c in frame.columns]
    frame.index = [str(i) for i in frame.index]
    return frame


def _sample_info_table(dataset) -> list[dict]:
    frame = sample_info(dataset).copy()
    frame.insert(0, "sample", frame.index.astype(str))
    return jsonable(frame.reset_index(drop=True).to_dict(orient="records"))[:500]


def _samples_response(dataset, mapping=None, matched=None) -> dict:
    out = {
        "dataset": dataset_summary(dataset),
        "sample_info": _sample_info_table(dataset),
        "groups": list(_safe_get(dataset, "groups") or []) or None,
        "group_column": _safe_get(dataset, "group_key"),
        "total": len(_safe_get(dataset, "samples") or []),
    }
    if mapping is not None:
        out["mapping"] = mapping
    if matched is not None:
        out["matched"] = matched
    return jsonable(out)


def handle_samples(dataset, args: dict) -> dict:
    """样本表管理：list / update / auto_group / import_map / join_obs / derive / set_group / remove。"""
    if dataset is None:
        raise EngineError("not_found", "还没有载入数据集")
    ensure_fields(dataset)
    op = str(args.get("op") or "list")
    info = sample_info(dataset)
    _safe_set(dataset, "sample_info", info)

    if op == "list":
        return _samples_response(dataset)

    if op == "update":
        samples = args.get("samples") or []
        groups = args.get("groups") or []
        changed = 0
        if isinstance(samples, list):
            display = "display_name" not in info.columns
            if display:
                info["display_name"] = ""
            for entry in samples:
                if not isinstance(entry, dict):
                    continue
                ident = str(entry.get("id") or entry.get("sample") or "")
                if ident not in info.index:
                    continue
                if entry.get("name"):
                    info.loc[ident, "display_name"] = str(entry.get("name"))
                if entry.get("group"):
                    info.loc[ident, "group"] = str(entry.get("group"))
                changed += 1
        if groups:
            notes = [n for n in (_safe_get(dataset, "notes") or []) if not str(n).startswith("分组：")]
            notes.append("分组：" + "/".join(str(g) for g in groups))
            _safe_set(dataset, "notes", notes)
        _safe_set(dataset, "sample_info", info)
        if _safe_get(dataset, "group_key") is None and "group" in info.columns:
            set_group(dataset, "group")
        elif _safe_get(dataset, "group_key"):
            set_group(dataset, str(dataset.group_key))
        return _samples_response(dataset, matched=changed)

    if op == "auto_group":
        regex_text = str(args.get("group_regex") or args.get("pattern") or "")
        if not regex_text:
            raise EngineError("bad_request", "请提供 group_regex 正则（第 1 个捕获组=大组，如 (Case|Control)）")
        try:
            regex = re.compile(regex_text)
        except re.error as error:
            raise EngineError("bad_request", f"正则编译失败：{error}") from error
        values = []
        for name in info.index:
            match = regex.search(str(name))
            values.append(match.group(1) if match and match.groups() else "other")
        info["group"] = values
        _safe_set(dataset, "sample_info", info)
        set_group(dataset, "group")
        return _samples_response(dataset)

    if op == "derive":
        regex_text = str(args.get("group_regex") or args.get("source") or "")
        name = str(args.get("name") or args.get("column") or "derived")
        column = args.get("column")
        try:
            regex = re.compile(regex_text) if regex_text else None
        except re.error as error:
            raise EngineError("bad_request", f"正则编译失败：{error}") from error
        values = []
        for sample in info.index:
            if regex is not None:
                match = regex.search(str(sample))
                values.append(match.group(1) if match and match.groups() else "")
            elif column and str(column) in info.columns:
                values.append(info.loc[sample, str(column)])
            else:
                values.append("")
        info[name] = values
        _safe_set(dataset, "sample_info", info)
        return _samples_response(dataset)

    if op == "set_group":
        column = str(args.get("column") or "")
        if not column:
            raise EngineError("bad_request", "请提供 column（要作为分组的样本列）")
        set_group(dataset, column)
        return _samples_response(dataset)

    if op in ("import_map", "join_obs", "join"):
        path = str(args.get("path") or "")
        if not path:
            raise EngineError("bad_request", "请提供样本信息表路径（path）")
        if not os.path.exists(path):
            raise EngineError("not_found", f"文件不存在：{path}")
        frame = _read_info_table(path)
        key = args.get("key")
        if key and str(key) in frame.columns:
            frame = frame.set_index(str(key))
        frame.index = [str(i) for i in frame.index]
        matched = [i for i in info.index if i in frame.index]
        if not matched:
            lower = {str(i).strip().lower(): i for i in frame.index}
            remap = {}
            for ident in info.index:
                cand = lower.get(str(ident).strip().lower())
                if cand:
                    remap[cand] = ident
            if remap:
                frame = frame.rename(index=remap)
                matched = [i for i in info.index if i in frame.index]
        if not matched:
            raise EngineError(
                "bad_request",
                f"样本信息表没有样本能对上（表内示例：{list(frame.index[:3])}；数据集示例：{list(info.index[:3])}）",
            )
        mapping: dict = {}
        for col in frame.columns:
            target = str(col)
            if target in info.columns and target != "sample":
                target = f"{target}_joined"
            info[target] = frame[col].reindex(info.index)
            mapping[target] = str(col)
        notes = list(_safe_get(dataset, "notes") or [])
        notes.append(f"已合并样本信息表：{os.path.basename(path)}（匹配 {len(matched)}/{len(info)} 样本）")
        _safe_set(dataset, "notes", notes)
        if len(matched) < len(info):
            warnings = list(_safe_get(dataset, "warnings") or [])
            warnings.append(f"样本信息表只匹配到 {len(matched)}/{len(info)} 个样本")
            _safe_set(dataset, "warnings", warnings)
        _safe_set(dataset, "sample_info", info)
        if _safe_get(dataset, "group_key") is None:
            guess = guess_group(dataset)
            if guess:
                set_group(dataset, guess)
        return _samples_response(dataset, mapping=mapping, matched=len(matched))

    if op == "remove":
        names = {str(x) for x in (args.get("samples") or [])}
        if not names:
            raise EngineError("bad_request", "请提供要移除的样本（samples）")
        samples = [str(s) for s in (_safe_get(dataset, "samples") or [])]
        keep = [s for s in samples if s not in names]
        if not keep:
            raise EngineError("bad_request", "不能移除全部样本")
        mask = np.array([s in set(keep) for s in samples])
        X = _safe_get(dataset, "X")
        if X is not None:
            _safe_set(dataset, "X", np.asarray(X)[mask])
        raw = _safe_get(dataset, "raw")
        if raw is not None:
            try:
                _safe_set(dataset, "raw", np.asarray(raw)[mask])
            except Exception:  # noqa: BLE001
                pass
        _safe_set(dataset, "samples", keep)
        _safe_set(dataset, "sample_info", info.loc[[s for s in info.index if s in set(keep)]])
        _safe_set(dataset, "steps", [s for s in (_safe_get(dataset, "steps") or []) if s in ("qc", "filter")])
        _safe_set(dataset, "results", {})
        _safe_set(dataset, "current", None)
        return _samples_response(dataset, matched=len(names))

    raise EngineError("bad_request",
                      f"不支持的样本操作：{op}（可选 list/update/auto_group/import_map/join_obs/derive/set_group/remove）")


# ---------------------------------------------------------------------------
# 预览（DESIGN §2.2 preview）
# ---------------------------------------------------------------------------


def preview(dataset) -> dict:
    """数据集预览：优先用 mb_state.Dataset.preview()，再补 §2.2 必需键。"""
    if dataset is None:
        raise EngineError("not_found", "还没有载入数据集")
    ensure_fields(dataset)
    out = None
    fn = _safe_get(dataset, "preview")
    if callable(fn):
        try:
            value = fn()
            if isinstance(value, dict):
                out = dict(value)
        except Exception:  # noqa: BLE001
            out = None
    if out is None:
        out = {}
    base = _preview_build(dataset)
    for key, value in base.items():
        out.setdefault(key, value)
    out["n_samples"] = len(_safe_get(dataset, "samples") or [])
    out["n_features"] = len(_safe_get(dataset, "features") or [])
    return jsonable(out)


def _preview_build(dataset) -> dict:
    info = sample_info(dataset)
    X = _safe_get(dataset, "X")
    n_nan = None
    n_total = None
    if X is not None:
        try:
            arr = np.asarray(X)
            n_total = int(arr.size)
            if arr.size <= 5e7:
                n_nan = int(np.isnan(arr.astype(np.float64, copy=False)).sum())
        except Exception:  # noqa: BLE001
            n_nan = None
    return {
        "features": [str(f) for f in (_safe_get(dataset, "features") or [])[:50]],
        "samples": [str(s) for s in (_safe_get(dataset, "samples") or [])[:20]],
        "n_samples": len(_safe_get(dataset, "samples") or []),
        "n_features": len(_safe_get(dataset, "features") or []),
        "columns": [{"key": str(c), "kind": column_kind(info[c]), "n_unique": int(info[c].nunique(dropna=True))}
                    for c in info.columns],
        "steps": list(_safe_get(dataset, "steps") or []),
        "current": _safe_get(dataset, "current"),
        "feature_meta": feature_meta(dataset),
        "group_key": _safe_get(dataset, "group_key"),
        "groups": list(_safe_get(dataset, "groups") or []) or None,
        "value_kind": _safe_get(dataset, "X_kind"),
        "species": _safe_get(dataset, "species"),
        "polarity": _safe_get(dataset, "polarity"),
        "source_format": _safe_get(dataset, "source_format"),
        "n_missing": n_nan,
        "n_values": n_total,
        "clinical": clinical_columns(dataset),
        "library": {"loaded": n_spectra(dataset) > 0, "n_spectra": n_spectra(dataset)},
        "features_preview": [str(f) for f in (_safe_get(dataset, "features") or [])[:50]],
    }


# ---------------------------------------------------------------------------
# 建议 / 自检
# ---------------------------------------------------------------------------

AUDIT_IDS = [
    "qc_done", "missing_imputed", "normalized", "transformed", "group_defined", "group_balance",
    "batch_confounded", "low_samples", "low_features", "de_done", "enrich_done",
    "biomarker_done", "annotation_missing", "ms2_library",
]


def audit(dataset) -> dict:
    """确定性自检：预处理完整性 + 统计前提 + 下游覆盖（不评价生物学结论）。"""
    checks: list[dict] = []

    def add(cid, level, title, detail, hint=""):
        checks.append({"id": cid, "level": level, "title": title, "detail": detail, "hint": hint})

    if dataset is None:
        return {"score": 0, "level": "fail",
                "checks": [{"id": "dataset", "level": "fail", "title": "没有数据集",
                            "detail": "先导入数据或载入示例", "hint": "action=demo"}]}
    ensure_fields(dataset)
    steps = set(_safe_get(dataset, "steps") or [])
    results = _safe_get(dataset, "results") or {}
    info = sample_info(dataset)
    meta = feature_meta(dataset)

    add("qc_done", "ok" if "qc" in steps else "warn", "样本质控",
        "已完成" if "qc" in steps else "尚未做样本质控（TIC/RSD/缺失率）",
        "" if "qc" in steps else "运行「样本质控」")

    if _has_nan(dataset):
        add("missing_imputed", "fail" if "impute" not in steps else "warn", "缺失值",
            "矩阵里仍有缺失值" if "impute" not in steps else "已有缺失值，但填补步骤已跑过",
            "运行「缺失值填补」")
    else:
        add("missing_imputed", "ok", "缺失值", "矩阵没有缺失值")

    x_kind = str(_safe_get(dataset, "X_kind") or "").lower()
    if "normalize" in steps or x_kind in ("lognorm", "scaled", "normalized"):
        add("normalized", "ok", "归一化", f"当前值类型：{_safe_get(dataset, 'X_kind')}")
    else:
        add("normalized", "fail", "归一化", f"当前值类型：{_safe_get(dataset, 'X_kind')}，直接比较会有系统性偏差",
            "运行「归一化」")
    if "transform" in steps:
        add("transformed", "ok", "变换/标度", "已完成变换与标度")
    else:
        add("transformed", "warn", "变换/标度", "尚未做 log 变换/标度，多变量模型可能被高丰度特征主导",
            "运行「变换/标度」")

    group_key = _safe_get(dataset, "group_key")
    if group_key and group_key in info.columns:
        counts = {str(k): int(v) for k, v in info[group_key].astype(str).value_counts().to_dict().items()}
        small = [g for g, n in counts.items() if n < 3]
        if small:
            add("group_defined", "warn", "分组", f"分组列 {group_key}：{counts}", f"样本量 <3 的组：{small}")
        else:
            add("group_defined", "ok", "分组", f"分组列 {group_key}：{counts}")
        values = [counts[g] for g in counts]
        if len(values) > 1 and max(values) - min(values) >= max(3, int(0.5 * min(values))):
            add("group_balance", "warn", "组间平衡", f"组间样本数差异较大：{counts}", "注意统计效能")
        else:
            add("group_balance", "ok", "组间平衡", f"组间样本量接近：{counts}")
    else:
        add("group_defined", "fail", "分组", "还没有设置分组列", "在「样本」里设置分组列")
        add("group_balance", "warn", "组间平衡", "没有分组列，无法评估", "先设置分组")

    batch_col = next((str(c) for c in info.columns if str(c).lower() in ("batch", "batch_id", "plate", "run", "cohort")), None)
    if batch_col:
        if "batch" in steps:
            add("batch_confounded", "ok", "批次", f"批次列 {batch_col}，已做批次校正")
        else:
            confound = False
            if group_key and group_key in info.columns:
                try:
                    table = pd.crosstab(info[batch_col].astype(str), info[group_key].astype(str))
                    confound = bool(table.shape[0] > 1 and (table.values > 0).sum(axis=1).max() == 1)
                except Exception:  # noqa: BLE001
                    confound = False
            if confound:
                add("batch_confounded", "fail", "批次与分组混杂",
                    f"批次列 {batch_col} 与分组 {group_key} 一一对应", "无法区分批次效应与生物学差异")
            else:
                add("batch_confounded", "warn", "批次", f"检测到批次列 {batch_col}，尚未做批次校正", "运行「批次校正」")
    else:
        add("batch_confounded", "ok", "批次", "没有检测到批次列")

    n_samples = len(_safe_get(dataset, "samples") or [])
    add("low_samples", "warn" if n_samples < 6 else "ok", "样本量",
        f"当前 {n_samples} 个样本" + ("，统计效能有限" if n_samples < 6 else ""))
    n_features = len(_safe_get(dataset, "features") or [])
    add("low_features", "warn" if n_features < 50 else "ok", "特征数",
        f"当前 {n_features} 个特征" + ("，做多变量模型意义有限" if n_features < 50 else ""))

    add("de_done", "ok" if ("de" in steps or "de" in results) else "warn", "差异分析",
        "已完成" if ("de" in steps or "de" in results) else "尚未做单变量差异",
        "" if ("de" in steps or "de" in results) else "运行「单变量差异」")
    add("enrich_done", "ok" if ("enrich" in steps or "enrich" in results) else "warn", "通路富集",
        "已完成" if ("enrich" in steps or "enrich" in results) else "尚未做通路富集",
        "" if ("enrich" in steps or "enrich" in results) else "运行「通路富集」")
    add("biomarker_done", "ok" if ("biomarker" in steps or "biomarker" in results) else "warn", "标志物",
        "已完成" if ("biomarker" in steps or "biomarker" in results) else "尚未做标志物筛选",
        "" if ("biomarker" in steps or "biomarker" in results) else "运行「标志物筛选」")
    if meta.get("n_annotated", 0) > 0:
        add("annotation_missing", "ok", "代谢物注释", f"{meta['n_annotated']} 个特征有注释")
    else:
        add("annotation_missing", "warn", "代谢物注释", "特征表里没有 name/HMDB/KEGG/class 注释",
            "导入带注释的特征表，或运行「代谢物鉴定」")
    if n_spectra(dataset) > 0:
        add("ms2_library", "ok", "MS2 谱库", f"{n_spectra(dataset)} 张谱图可用")
    else:
        add("ms2_library", "warn", "MS2 谱库", "没有 MS2 谱库，无法做谱图匹配/分子网络",
            "导入 msp/mgf 或 mzML")

    checks = [c for c in checks if c["id"] in AUDIT_IDS]
    fails = sum(1 for c in checks if c["level"] == "fail")
    warns = sum(1 for c in checks if c["level"] == "warn")
    score = max(0, min(100, 100 - 30 * fails - 12 * warns))
    level = "fail" if fails else ("warn" if warns else "ok")
    return {"score": score, "level": level, "checks": checks}


def suggest(dataset) -> dict:
    """下一步建议：沿步骤 DAG + 可用分析生成（最多 8 条）。"""
    if dataset is None:
        return {"suggestions": [
            {"id": "dataset:demo", "action": "demo", "title": "载入示例数据",
             "detail": "72 样本 × 1600 特征的模拟代谢组学数据，可直接体验完整流程", "step": None, "params": {}},
            {"id": "dataset:open", "action": "open", "title": "导入本地特征表",
             "detail": "支持 csv/tsv/xlsx/gz/zip/mzTab/ISA-Tab；带 mz/rt 列更好", "step": None, "params": {}},
            {"id": "geo:search", "action": "search", "title": "检索 GEO 代谢组学数据",
             "detail": "例如「serum metabolomics LC-MS」；MetaboLights 用 action=ml_search", "step": None, "params": {}},
        ]}
    ensure_fields(dataset)
    steps = set(_safe_get(dataset, "steps") or [])
    results = _safe_get(dataset, "results") or {}
    suggestions: list[dict] = []

    def add(sid, title, detail, step=None, params=None, action="run"):
        suggestions.append({"id": sid, "action": action, "title": title, "detail": detail,
                            "step": step, "params": params or {}})

    if "qc" not in steps:
        add("run:qc", "样本质控", "先看每样本 TIC/检出特征/缺失率与 QC 样本 RSD", "qc")
    elif "filter" not in steps:
        add("run:filter", "特征过滤", "去掉低检出/低强度特征，必要时按 QC RSD 过滤", "filter")
    elif "impute" not in steps and _has_nan(dataset):
        add("run:impute", "缺失值填补", "用 half-min / KNN 填补，避免多变量分析丢样本", "impute")
    elif "normalize" not in steps:
        add("run:normalize", "归一化", "PQN / 内标 / TIC 归一化，消除稀释与进样量差异", "normalize")
    elif "transform" not in steps:
        add("run:transform", "变换/标度", "log2 变换 + Pareto/auto 标度，让高丰度特征不主导模型", "transform")
    elif "pca" not in steps:
        add("run:pca", "主成分分析", "看整体结构、离群样本与 Hotelling T²/DModX", "pca")
    elif _safe_get(dataset, "group_key") and "plsda" not in steps:
        add("run:plsda", "PLS-DA / OPLS-DA", "有监督判别 + VIP/置换检验/交叉验证", "plsda")
    elif _safe_get(dataset, "group_key") and "de" not in steps:
        add("run:de", "单变量差异", f"按「{_safe_get(dataset, 'group_key')}」做 Welch/Mann-Whitney + BH", "de")
    elif "de" in steps and "enrich" not in steps:
        add("run:enrich", "通路富集", "对差异特征做 mummichog 式通路富集（需要 m/z）", "enrich")
    elif "de" in steps and "biomarker" not in steps:
        add("run:biomarker", "标志物筛选", "RF/LASSO/ROC 组合评分与自助法验证", "biomarker")

    if not _safe_get(dataset, "group_key"):
        add("samples:set_group", "设置分组列", "差异/判别/标志物分析都需要分组列", None,
            action="samples")
    for item in analyses_catalog(dataset)["analyses"]:
        if len(suggestions) >= 8:
            break
        if not item["available"]:
            continue
        if any(str(r.get("id")) == item["id"] for r in (_safe_get(dataset, "analyses_run") or [])):
            continue
        if item["id"] in ("qc_summary", "missing_pattern"):
            continue
        add(f"analysis:{item['id']}", item["title"], item["method"] or "", None, action="analysis")
    return {"suggestions": suggestions[:8]}


# ---------------------------------------------------------------------------
# 预设流程
# ---------------------------------------------------------------------------


def recipe(recipe_id: str | None = None):
    if not recipe_id:
        return {"recipes": jsonable(RECIPES)}
    target = str(recipe_id)
    for item in RECIPES:
        if item["id"] == target:
            return jsonable(item)
    raise EngineError("not_found", f"预设流程不存在：{target}（可选：{', '.join(r['id'] for r in RECIPES)}）")


# ---------------------------------------------------------------------------
# 导出（DESIGN §0/§2.2 export）
# ---------------------------------------------------------------------------

EXPORT_KINDS = ["expr_csv", "de_csv", "vip_csv", "enrich_csv", "biomarker_csv", "sample_info_csv",
                "summary_json", "protocol", "methods", "report", "script"]

STORE_FOR_EXPORT = {
    "de_csv": ("de", "单变量差异"),
    "vip_csv": ("vip", "VIP"),
    "enrich_csv": ("enrich", "通路富集"),
    "biomarker_csv": ("biomarker", "标志物"),
}


def _store_to_frame(stored) -> pd.DataFrame:
    if isinstance(stored, pd.DataFrame):
        return stored
    if isinstance(stored, dict):
        for key in ("table", "de_table", "vip_table", "enrich_table", "biomarker_table", "rows", "data"):
            value = stored.get(key)
            if isinstance(value, pd.DataFrame):
                return value
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.DataFrame(value)
        try:
            return pd.DataFrame(stored)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    if isinstance(stored, list) and stored and isinstance(stored[0], dict):
        return pd.DataFrame(stored)
    return pd.DataFrame()


def _find_store(results: dict, key: str):
    if key in results:
        return results.get(key)
    if key == "vip":
        for alt in ("plsda", "splot"):
            value = results.get(alt)
            if isinstance(value, dict) and any(k in value for k in ("vip", "vip_table", "table")):
                return value.get("vip") or value.get("vip_table") or value.get("table")
    return None


def export_dataset(dataset, what: str, out_path: str | None, workdir: str,
                   engine_info: dict | None = None) -> dict:
    """导出结果文件；返回 {file, bytes, what}（script 额外带 manifest/可复现信息）。"""
    if dataset is None:
        raise EngineError("not_found", "还没有载入数据集")
    ensure_fields(dataset)
    what = str(what or "summary_json")
    if what not in EXPORT_KINDS:
        raise EngineError("bad_request", f"不支持的导出类型：{what}（可选：{', '.join(EXPORT_KINDS)}）")
    export_dir = ensure_dir(os.path.join(workdir, "exports"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{_safe_get(dataset, 'id') or 'ds'}-{what}-{stamp}"

    def target(ext: str) -> str:
        if out_path:
            path = str(out_path)
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                ensure_dir(parent)
            return path
        return os.path.join(export_dir, f"{base}.{ext}")

    def done(path: str, **extra) -> dict:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
        return {"file": path, "bytes": size, "what": what, **extra}

    if what == "expr_csv":
        path = target("csv")
        X = _safe_get(dataset, "X")
        frame = pd.DataFrame(X, index=[str(s) for s in (_safe_get(dataset, "samples") or [])],
                             columns=[str(f) for f in (_safe_get(dataset, "features") or [])])
        frame.index.name = "sample"
        frame.to_csv(path, encoding="utf-8")
        return done(path, rows=int(frame.shape[0]), columns=int(frame.shape[1]),
                    value_kind=_safe_get(dataset, "X_kind"))

    if what == "sample_info_csv":
        path = target("csv")
        frame = sample_info(dataset).copy()
        frame.insert(0, "sample", frame.index.astype(str))
        frame.to_csv(path, index=False, encoding="utf-8")
        return done(path, rows=int(frame.shape[0]))

    if what in STORE_FOR_EXPORT:
        store_key, label = STORE_FOR_EXPORT[what]
        stored = _find_store(_safe_get(dataset, "results") or {}, store_key)
        if not stored:
            raise EngineError("prerequisite", f"还没有{label}结果，先运行对应步骤/分析")
        frame = _store_to_frame(stored)
        if frame.empty:
            raise EngineError("prerequisite", f"{label}结果是空的，先重新运行对应步骤/分析")
        path = target("csv")
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return done(path, rows=int(frame.shape[0]), columns=int(frame.shape[1]))

    if what == "summary_json":
        path = target("json")
        payload = {
            "plugin": {"name": PLUGIN_NAME, "version": PLUGIN_VERSION},
            "dataset": dataset_summary(dataset),
            "steps": list(_safe_get(dataset, "steps") or []),
            "history": jsonable(_safe_get(dataset, "history") or []),
            "analyses_run": jsonable(_safe_get(dataset, "analyses_run") or []),
            "results_available": sorted((_safe_get(dataset, "results") or {}).keys()),
            "engine": jsonable(engine_info or {}),
            "seed": SEED,
            "exported_at": now_iso(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2)
        return done(path)

    if what == "protocol":
        path = target("json")
        payload = {
            "plugin": {"name": PLUGIN_NAME, "version": PLUGIN_VERSION},
            "dataset": dataset_summary(dataset),
            "source": {"path": _safe_get(dataset, "path") or _safe_get(dataset, "source_path"),
                       "format": _safe_get(dataset, "source_format")},
            "engine": jsonable(engine_info or {}),
            "seed": SEED,
            "steps": [
                {"step": item.get("step"), "params": item.get("params"), "ok": item.get("ok"), "ms": item.get("ms"),
                 "at": item.get("at")}
                for item in (_safe_get(dataset, "history") or [])
            ],
            "analyses": [
                {"id": item.get("id"), "ok": item.get("ok"), "ms": item.get("ms"), "at": item.get("at")}
                for item in (_safe_get(dataset, "analyses_run") or [])
            ],
            "modules": module_status(),
            "exported_at": now_iso(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2)
        return done(path)

    if what == "methods":
        path = target("md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_methods_markdown(dataset, engine_info))
        return done(path)

    if what == "report":
        path = target("md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_report_markdown(dataset, engine_info))
        return done(path)

    if what == "script":
        path = target("py")
        text = _script_python(dataset, engine_info, filename=os.path.basename(path))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return done(path, replayable=True, label="脚本可在本机 lib/ 目录下重放同一流程")

    raise EngineError("bad_request", f"不支持的导出类型：{what}")


def _engine_line(engine_info: dict | None) -> str:
    info = engine_info or {}
    packages = info.get("packages") or {}
    parts = [f"{name} {version}" for name, version in packages.items() if version]
    return (f"Python {info.get('python') or '?'}" + ("；" + "、".join(parts) if parts else ""))


def _fmt_value(value, digits: int = 4):
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return ""
    if abs(number) >= 1000 or (abs(number) < 1e-3 and number != 0):
        return f"{number:.{digits - 1}e}"
    return f"{number:.{digits}g}"


def _methods_markdown(dataset, engine_info: dict | None = None) -> str:
    """Methods（中文、可直接进论文方法学部分）。"""
    summary = dataset_summary(dataset)
    lines = ["# 方法（Methods）", "", "## 数据", ""]
    lines.append(f"- 数据集：{summary.get('name') or '—'}（{summary.get('n_samples')} 个样本 × "
                 f"{summary.get('n_features')} 个特征；来源格式 {summary.get('source_format') or '—'}）")
    lines.append(f"- 物种：{summary.get('species') or 'unknown'}；极性：{summary.get('polarity') or '未指定'}")
    if summary.get("path"):
        lines.append(f"- 原始文件：`{summary.get('path')}`")
    meta = summary.get("feature_meta") or {}
    lines.append(f"- 特征元数据：m/z {'有' if meta.get('has_mz') else '无'}，保留时间 "
                 f"{'有' if meta.get('has_rt') else '无'}，注释特征 {meta.get('n_annotated', 0)} 个")
    lines += ["", "## 预处理与分析步骤", ""]
    history = _safe_get(dataset, "history") or []
    if history:
        lines.append("| 步骤 | 参数 | 结果 |")
        lines.append("|---|---|---|")
        for item in history:
            params = json.dumps(item.get("params") or {}, ensure_ascii=False)
            lines.append(f"| {STEP_TITLES.get(str(item.get('step')), item.get('step'))} | `{params}` | "
                         f"{'成功' if item.get('ok') else '失败：' + str(item.get('summary') or '')} |")
    else:
        lines.append("- 尚未执行任何步骤。")
    analyses = _safe_get(dataset, "analyses_run") or []
    if analyses:
        titles = {item["id"]: item["title"] for item in ANALYSIS_ORDER}
        lines += ["", "## 分析", ""]
        for item in analyses:
            lines.append(f"- {titles.get(str(item.get('id')), item.get('id'))}（{item.get('id')}）"
                         f"{'成功' if item.get('ok') else '失败'}")
    lines += ["", "## 软件与统计", ""]
    lines.append(f"- 软件：{PLUGIN_NAME} {PLUGIN_VERSION}（DeepSeek Harness 插件，内置实现）；{_engine_line(engine_info)}")
    lines.append(f"- 随机种子：{SEED}（所有置换检验/交叉验证/自助法均可复现）")
    lines.append("- 差异分析：Welch t / Mann-Whitney / limma + Benjamini-Hochberg FDR")
    lines.append("- 多变量：PLS-DA/OPLS-DA（NIPALS 内置实现）+ VIP + 置换检验 + k 折交叉验证")
    lines.append("- 富集：Mummichog 式质量映射置换检验、KEGG/HMDB 超几何 ORA、秩基 MSEA")
    lines.append("- 批次：QC-RLSC（LOESS）与 ComBat（经验贝叶斯，内置实现）")
    lines.append("- 说明：本机未安装 xcms/MetaboAnalystR/pyopenms/matchms，上述方法均为独立复现实现，"
                 "算法参数与文献一致但不保证与官方包逐位相同。")
    lines.append("")
    return "\n".join(lines)


def _md_table(frame: pd.DataFrame, limit: int = 20) -> str:
    if frame is None or frame.empty:
        return "_（无结果）_"
    view = frame.head(limit)
    columns = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_fmt_value(row[c]) for c in view.columns) + " |")
    if len(frame) > limit:
        lines.append(f"| … | 共 {len(frame)} 行 |" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def _report_markdown(dataset, engine_info: dict | None = None) -> str:
    """分析报告（中文 Markdown：概览 + 步骤 + 主要结果表）。"""
    summary = dataset_summary(dataset)
    results = _safe_get(dataset, "results") or {}
    lines = ["# 代谢组学分析报告", ""]
    lines.append(f"- 数据集：{summary.get('name')}（{summary.get('n_samples')} 样本 × {summary.get('n_features')} 特征）")
    group_key = _safe_get(dataset, "group_key")
    if group_key:
        lines.append(f"- 分组列：{group_key}；分组：{'/'.join(map(str, _safe_get(dataset, 'groups') or []))}")
    lines.append(f"- 已完成步骤：{' → '.join(STEP_TITLES.get(s, s) for s in (_safe_get(dataset, 'steps') or [])) or '无'}")
    lines += ["", "## 质控与预处理", ""]
    history = _safe_get(dataset, "history") or []
    if history:
        for item in history[-12:]:
            lines.append(f"- **{STEP_TITLES.get(str(item.get('step')), item.get('step'))}**："
                         f"{item.get('summary') or ('成功' if item.get('ok') else '失败')}")
    else:
        lines.append("- 尚未执行步骤。")
    de_frame = _store_to_frame(results.get("de")) if results.get("de") else pd.DataFrame()
    if not de_frame.empty:
        lines += ["", "## 差异特征（Top 20）", "", _md_table(de_frame, 20)]
    vip_frame = _store_to_frame(_find_store(results, "vip")) if _find_store(results, "vip") else pd.DataFrame()
    if not vip_frame.empty:
        lines += ["", "## VIP（Top 20）", "", _md_table(vip_frame, 20)]
    enrich_frame = _store_to_frame(results.get("enrich")) if results.get("enrich") else pd.DataFrame()
    if not enrich_frame.empty:
        lines += ["", "## 通路富集（Top 15）", "", _md_table(enrich_frame, 15)]
    bio_frame = _store_to_frame(results.get("biomarker")) if results.get("biomarker") else pd.DataFrame()
    if not bio_frame.empty:
        lines += ["", "## 标志物（Top 15）", "", _md_table(bio_frame, 15)]
    audit_result = audit(dataset)
    lines += ["", "## 自检", "", f"- 得分：{audit_result['score']}/100（{audit_result['level']}）"]
    for check in audit_result["checks"]:
        if check["level"] != "ok":
            lines.append(f"- [{check['level']}] {check['title']}：{check['detail']}"
                         + (f"（{check['hint']}）" if check.get("hint") else ""))
    lines += ["", "## 复现", "",
              f"- 导出 `methods` 得到方法学描述，导出 `script` 得到可重放脚本；随机种子 {SEED}。", ""]
    return "\n".join(lines)


def _script_python(dataset, engine_info: dict | None = None, filename: str = "replay.py") -> str:
    """生成可重放脚本：重新导入原始数据并按记录的参数重跑全部步骤/分析。"""
    summary = dataset_summary(dataset)
    source = _safe_get(dataset, "path") or _safe_get(dataset, "source_path") or ""
    steps = [(item.get("step"), item.get("params") or {}) for item in (_safe_get(dataset, "history") or [])
             if item.get("ok")]
    analyses = [str(item.get("id")) for item in (_safe_get(dataset, "analyses_run") or []) if item.get("ok")]
    lib = HERE
    workdir = os.path.dirname(lib)
    header = f'''#!/usr/bin/env python3
"""dsh-metabolomics 可复现脚本（自动生成，文件名 {filename}）

数据集：{summary.get('name')}（{summary.get('n_samples')} 样本 × {summary.get('n_features')} 特征）
随机种子：{SEED}

用法：
    python3 {filename}
脚本会重新导入原始数据、按记录参数重跑步骤与分析，并把结果导出到 exports/。
"""
from __future__ import annotations

import json
import os
import sys

LIB = {lib!r}
WORKDIR = {workdir!r}
SOURCE = {source!r}
SEED = {SEED}
STEPS = {json.dumps(steps, ensure_ascii=False, default=str)}
ANALYSES = {json.dumps(analyses, ensure_ascii=False, default=str)}
EXPORTS = ["summary_json", "methods", "report"]

if LIB not in sys.path:
    sys.path.insert(0, LIB)

import numpy as np  # noqa: E402

import mb_io  # noqa: E402
import mb_registry  # noqa: E402
import mb_state  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")
os.makedirs(WORKDIR, exist_ok=True)
np.random.seed(SEED)


def main() -> int:
    state = mb_state.State()
    loaded = mb_io.load_source(SOURCE, workdir=WORKDIR, options={{"data_format": "auto"}})
    dataset = mb_registry.build_dataset(state, loaded, source=SOURCE)
    print(f"[data] {{dataset.name}}: {{len(dataset.samples)}} samples x {{len(dataset.features)}} features")

    for step, params in STEPS:
        try:
            result = mb_registry.run_step(dataset, step, params, WORKDIR)
            print(f"[step] {{step}}: {{result['summary'].get('bullets')}}")
        except Exception as error:  # noqa: BLE001
            print(f"[step] {{step}} 失败：{{error}}")
    for analysis_id in ANALYSES:
        try:
            result = mb_registry.run_analysis(dataset, analysis_id, {{}}, WORKDIR)
            print(f"[analysis] {{analysis_id}}: {{result['summary'].get('bullets')}}")
        except Exception as error:  # noqa: BLE001
            print(f"[analysis] {{analysis_id}} 失败：{{error}}")
    for what in EXPORTS:
        try:
            out = mb_registry.export_dataset(dataset, what, None, WORKDIR)
            print(f"[export] {{what}} -> {{out['file']}}")
        except Exception as error:  # noqa: BLE001
            print(f"[export] {{what}} 失败：{{error}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
    return header


# ---------------------------------------------------------------------------
# 步骤目录（/steps 与 ping 用）
# ---------------------------------------------------------------------------


def steps_payload(reload: bool = False) -> dict:
    if reload:
        load_modules(force=True)
    return {
        "steps": steps_catalog(),
        "plots": plots_catalog(),
        "analyses": [item["id"] for item in analyses_catalog(None)["analyses"]],
        "modules": module_status(),
        "missing_modules": missing_modules(),
        "conflicts": list(_RUNTIME.get("conflicts") or []),
    }
