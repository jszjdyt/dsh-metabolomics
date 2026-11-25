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

**计算实现（2026-11 更新：已装真包，默认走主流实现）**

- **PLS-DA / OPLS-DA → R `ropls` 1.47.0**（真包装在 R 4.4.2 库里；`method` 写 `ropls::opls（OPLS-DA，predI=1，orthoI=2，ropls 1.47.0）`）。
  demo 实测：PLS-DA R²Y=0.989 / Q²=0.554；OPLS-DA R²Y=1.000 / **Q²=0.915**，置换 p=0.02，CV 准确率 0.833，VIP 前 20 命中注入特征 18–20/20。
  PCA 也走 `ropls::opls`（PCA 模式）。
- **MS2 库匹配 / 分子网络 → `matchms` 1.0.0**（`CosineGreedy`，前体 ±ppm + 峰容差），`method` 写 `matchms.similarity.CosineGreedy(...)`。
- Python 侧环境 `~/.dsh/runtimes/bio312`（3.12）：matchms、rdkit、scanpy、lifelines、statsmodels 等。
- **ropls 安装说明**：`bioconductor.org` 与国内镜像在本机不可达，ropls 是从 `git.bioconductor.org` 源码装的（只去掉了 `MultiAssayExperiment`/`MultiDataSet` 两个**对象互操作**依赖，PCA/PLS/OPLS 核心算法未改）。详见 INSTALL.md。
- **仍是内置实现（已如实标注）**：加合物推断、分子式候选（CHNOPS + RDBE/七黄金规则，matchms/ropls 无此功能）；Mummichog 式通路富集（内置通路-质量表 + 置换，无 pip 版 mummichog）；QC-RLSC（Dunn 2011 自实现）；共表达网络/模块；DeLong CI；4PL 剂量响应。
- `ropls`/`matchms` 不在时自动回退内置实现，`method` 标"回退内置 NIPALS/内置余弦"并给原因。
- MetaboLights 检索按 12s 预算并发预取前几条元信息（EBI 端点 3–15s 抖动），其余点开详情再取。
- 图内文字英文 + 中文回落字体。
