# dsh-metabolomics — 代谢组学工作台

DeepSeek Harness 的**代谢组学可视化工作台**插件（Host 工具 `metabolomics` + 整页 UI），
与 `dsh-geobulk`（bulk）、`dsh-singlecell`（单细胞）、`dsh-spatial`（空间）同形态。

一句话：**搜 GEO / MetaboLights → 导入 LC-MS 特征表 → 质控/归一化 → PLS-DA/OPLS-DA → 差异/通路富集/标志物/代谢物鉴定**，页面上点着用，也能让 Agent 一句话跑完。

## 功能

**数据**
- GEO 检索（中文关键词用内置 ≥200 条词典反查英文；`method` 识别 LC-MS/GC-MS/NMR）
- MetaboLights（EBI）检索/详情/下载（第二数据源，GEO 的代谢组学数据太稀疏）
- 导入：特征表 csv/tsv/txt(.gz)/xlsx（自动判方向、识别 mz/rt 列）、GEO series matrix、zip/tar、mzTab、MetaboLights ISA-Tab（`m_*.txt`）、MS2 谱库（msp/mgf）、mzML（TIC + 谱图）、rds/RData（R 桥）

**常规流程**（12 步）
`qc` 样本质控 → `filter` 特征过滤 → `impute` 缺失填补 → `normalize` 归一化（PQN/TIC/median/quantile）→ `transform` 变换/标度（log2/log10/auto + auto/pareto/range）→ `pca`（含 Hotelling T²/DModX）→ `plsda`（PLS-DA/OPLS-DA，R²Y/Q²/VIP/置换）→ `de` 单变量差异 → `batch` 批次校正（QC-RLSC/ComBat）→ `enrich` 通路富集（Mummichog 式/ORA）→ `biomarker` 标志物（RF/LASSO/VIP/组合）→ `annotate` 代谢物鉴定（加合物/分子式/同位素/MS2）

**高级分析**（38 项，按组）
- 质控与基础：质控汇总、QC-RSD、缺失模式、样本离群（Hotelling T²/DModX）、样本相关性
- 多变量判别：VIP、OPLS-DA S-plot、置换检验、交叉验证/混淆矩阵、多组判别、PCA 载荷
- 差异与趋势：单变量差异（Welch/Mann-Whitney/limma）、火山图、FC 条形、差异热图、多组 ANOVA、时间趋势、剂量响应（4PL EC50）
- 通路与化学分类：Mummichog 式富集、KEGG ORA、HMDB 化学分类富集、代谢物集富集（秩基 GSEA）
- 标志物与建模：RF 重要度、LASSO 组合、单标志物 ROC（DeLong CI）、组合评分、自助法验证、代谢物分组生存
- 批次与网络：QC-RLSC、ComBat、共表达网络、代谢物模块
- 代谢物鉴定：加合物推断、分子式候选（CHNOPS + RDBE）、MS2 库匹配（余弦）、分子网络（GNPS 式）

**出图/导出**：25 种常规图（另有 7 种模块专用图 = 32 种）（TIC/RSD/缺失/PCA/PLS-DA/S-plot/VIP/置换/火山/热图/富集/ROC/重要度/趋势/剂量/质量精度/网络）；导出特征表、差异表、VIP、富集、标志物、汇总 JSON、Methods、报告、脚本。

**Agent 工具** `metabolomics`：`status/search/detail/download/ml_search/ml_detail/ml_download/open/demo/inspect/preview/run/plot/analyses/analysis/export/audit/suggest/recipe/samples/formats/reset/close`。

## 安装

```bash
python3 -c "import numpy,pandas,scipy,sklearn,matplotlib,statsmodels,openpyxl,lifelines,certifi"
# 设置 → 插件 → 添加本地插件 → 选 plugins/dsh-metabolomics
# 或 plugin_manager: action=install_bundle target=<绝对路径>
```

## 首次使用

1. 侧边栏「代谢工作台」→「载入示例数据」（72 样本 × 1600 特征，含 QC/批次/临床列）。
2. 左栏从上到下点「运行」：质控 → 过滤 → 填补 → 归一化 → 变换/标度 → PCA → PLS-DA → 差异 → 富集 → 标志物。
3. 真实数据：顶栏「GEO 检索」搜 `serum metabolomics LC-MS`，或「MetaboLights」搜研究编号/关键词；也可以直接填自己的特征表路径（必要时配样本表/临床表）。

## 说明与边界

- 本机没有 xcms/ropls/MetaboAnalystR/pyopenms/matchms，**所有多变量/富集/鉴定都是内置实现**（numpy/scipy/sklearn），依据写在每个分析的「方法」列（Wold 1995 VIP；Trygg & Wold 2002 OPLS；Westerhuis 2008 置换；Dieterle 2006 PQN；van den Berg 2006 标度；Dunn 2011 QC-RLSC；Johnson 2007 ComBat；Li 2013 mummichog；Subramanian 2005 GSEA；Breiman 2001 RF；Tibshirani 1996 LASSO；DeLong 1988；Harrell 1996；Stein 1994 谱库匹配；Wang 2016 分子网络）。
- Mummichog 式富集用内置通路-质量表 + 置换，**不做**真正的代谢网络传播；分子式候选只做质量/规则过滤，不等于结构鉴定。
- 特征注释依赖你表里的 `mz/rt/kegg/class` 列或内置词典；没有 m/z 时鉴定类分析不可用。
- 图内文字英文，中文说明在结果摘要里。
