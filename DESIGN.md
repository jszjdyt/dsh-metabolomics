# dsh-metabolomics 内部契约（代谢组学工作台）

> 本文件是并行开发者的**唯一事实来源**。实现与契约冲突时以契约为准；确需改协议，先改本文件再改代码。
> 参考实现（只读，不要改）：`../dsh-geobulk/`（同形态 + GEO + jobs + PLS/差异/富集/临床套路）、`../dsh-singlecell/`（同形态 UI）。
> 文献与产品基线见 §4；每一步/分析都要写清"用了什么方法、依据哪篇文献/哪个产品"。

## 0. 目标

在 DeepSeek Harness Web GUI 里提供一个**可视化、可操作、Agent 可驱动**的代谢组学工作台（LC-MS/GC-MS 特征表为核心，MS1/MS2 鉴定为辅）：

1. **GEO 检索**：中文/英文关键词检索代谢组学数据（NCBI E-utilities gds）；中文化字段；筛选（物种/方法/样本数/是否有人体临床）；看图选目录下载。**另加 MetaboLights（EBI）作为第二数据源**（`source=metabolights`），因为 GEO 的代谢组学数据稀疏。
2. **导入**：特征表（csv/tsv/txt/xlsx/gz；样本×特征或特征×样本；含 mz/rt 列）、GEO series matrix、zip/tar、mzTab、ISA-Tab（`m_*.txt`）、MetaboAnalyst 格式、MS2 谱库（msp/mzML）与 mzML（TIC/谱图）。
3. **常规流程**：样本质控 → 特征过滤 → 缺失值填补 → 归一化 → 变换/标度 → PCA → PLS-DA/OPLS-DA → 单变量差异 → 通路富集 → 标志物筛选 → 代谢物鉴定。
4. **高级分析**：VIP/S-plot/置换检验/交叉验证、多组 ANOVA + 事后、配对、时间趋势、剂量响应（4PL）、Mummichog 式通路富集、HMDB 化学分类富集、RF/LASSO/ROC/组合评分/自助法验证、KM 生存（代谢物分组）、QC-RLSC 与 ComBat 批次校正、共表达网络/WGCNA 式模块/模块-表型相关、加合物与分子式推断、同位素、MS2 库匹配与分子网络。
5. **出图与数据**：TIC/RSD/缺失/PCA/PLS-DA/S-plot/VIP/火山/热图/富集/ROC/重要度/趋势/剂量/质量精度/网络；导出特征表、差异表、VIP、富集表、标志物、汇总 JSON、Methods、报告、可复现脚本。
6. UI 风格对齐 singlecell/geobulk。

**硬性要求**同 dsh-spatial：Python 未就绪给可读原因；单步失败不影响整页；长任务有超时与进度；网络不可达给可读降级。

---

## 1. 目录与写权限（严格隔离）

```
plugins/dsh-metabolomics/
  DESIGN.md                      # Lead（本文件）
  README.md / INSTALL.md         # Lead
  package.json / cordis.patch.yml / icon.svg / locale/{zh,en}.json   # Lead
  index.js                       # Lead  Host 入口（/metabolomics 前缀路由 + metabolomics 工具 + prompt）
  lib/impl.mjs                   # Lead  唯一操作实现（HTTP + 工具共用）
  lib/engine.mjs                 # Lead  Python worker 管理
  lib/mb_engine.py               # Lead  worker 主循环 + 命令分发（协议 §2）
  lib/mb_registry.py             # Lead  模块汇总 + ctx + 调度 + 结果整形
  lib/mb_common.py               # mb-core  EngineError / table / save_fig / jsonable / 化学质量工具
  lib/mb_state.py                # mb-core  Dataset、样本表、demo、history
  lib/mb_io.py                   # mb-core  全格式导入
  lib/mb_geo.py                  # mb-core  GEO + MetaboLights 检索/详情/下载
  lib/mb_prep.py                 # mb-base  步骤 qc/filter/impute/normalize/transform/pca
  lib/mb_plots.py                # mb-base  基础图（qc_tic/qc_rsd/qc_missing/sample_corr/pca_scores/pca_variance/hotelling_t2/batch_pca/heatmap_top）
  lib/mb_plsda.py                # mb-stat  步骤 plsda + 判别分析（VIP/S-plot/置换/交叉验证/多类）
  lib/mb_de.py                   # mb-stat  步骤 de + 差异/趋势/剂量（火山/FC/热图/ANOVA/时间趋势/4PL）
  lib/mb_enrich.py               # mb-adv   步骤 enrich + 通路/化学分类富集（Mummichog / KEGG / HMDB class）
  lib/mb_biomarker.py            # mb-adv   步骤 biomarker + RF/LASSO/ROC/评分/验证/生存
  lib/mb_batch.py                # mb-adv   步骤 batch + RLSC/ComBat + 批次诊断
  lib/mb_annotate.py             # mb-adv   步骤 annotate + 加合物/分子式/同位素/MS2 库匹配/质量精度
  lib/mb_network.py              # mb-adv   共表达网络/WGCNA 式模块/模块-表型
  client.js                      # mb-ui    浏览器 UI
  test/ examples/ figures/
```

- 只写自己作用域；**写之前先 `read`**；模块间只允许 `import mb_common`（及 `mb_state` 的纯数据类）。

---

## 2. Python worker 协议

### 2.1 启动与传输（与 dsh-geobulk 完全一致）

- `python3 -u lib/mb_engine.py --workdir <abs_dir>`，`cwd=lib/`；stdin/stdout JSON-lines；
  成功 `{"id","ok":true,"result":{}}`；失败 `{"id","ok":false,"error":{"code","message","trace"?}}`；进度 `{"id","event":"progress","message","pct"}`。
- **stdout 只允许协议行**，日志走 stderr。`code` ∈ `bad_request|not_found|prerequisite|unsupported|ambiguous|internal|timeout|no_python|engine_exited|busy|disposed`。

### 2.2 命令表

| cmd | args | 返回 |
|---|---|---|
| `ping` | `{}` | `{python,platform,mode,workdir,packages:{numpy,pandas,scipy,sklearn,matplotlib,statsmodels,openpyxl,h5py},capabilities:{steps,analyses,plots,formats},geo:true,metabolights:bool}` |
| `open` | `{path?,url?,name?,data_format?,member?,object?,transpose?,delimiter?,sheet?,sample_sheet?,clinical?,feature_col?,mz_col?,rt_col?,group_col?,demo?}` | `{dataset,preview}` |
| `inspect` | 同上探测子集 | `{format,n_samples?,n_features?,has_mz?,has_rt?,candidates:[],notes:[],size}` |
| `demo` | `{name?}` | `{dataset,preview}` |
| `list`/`select`/`close`/`reset` | | 同 dsh-spatial |
| `preview` | `{dataset}` | `{features:[前 50 特征 id],samples:[前 20],n_samples,n_features,columns:[{key,kind,n_unique}],steps,current,feature_meta:{has_mz,has_rt,annotated}}` |
| `samples` | `{dataset,op:"list|update|auto_group|import_map|join_obs|derive|set_group|remove",...}` | `{dataset,sample_info,groups,group_column,mapping?,matched?,total?}` |
| `run` | `{dataset,step,params}` | `{step,summary,steps,current,dataset,tables?,figures?}` |
| `plot` | `{dataset,kind,params}` | `{file,width,height,bytes,kind}` |
| `analyses`/`analysis`/`export`/`audit`/`suggest`/`recipe`/`formats`/`steps` | 同 dsh-spatial | 同 |
| `geo_search`/`geo_detail`/`geo_download` | §7 | §7 |
| `ml_search`/`ml_detail`/`ml_download` | §7.4 MetaboLights | §7.4 |
| `shutdown` | `{}` | `{bye:true}` |

### 2.3 `ctx`

```python
ctx = {
  "dataset_id":"ds1","name":"demo-metabo",
  "features":["mz203.0526_rt1.23", ...],     # list[str] 长度 = X.shape[1]，唯一 id
  "samples":["S1", ...],                     # list[str] 长度 = X.shape[0]
  "X": np.ndarray float32,                   # (n_samples, n_features) 当前矩阵（含缺失 np.nan）
  "X_kind": "raw|lognorm|scaled",
  "raw": np.ndarray|None,                    # 原始强度（归一化前），用于 TIC/RSD/批次
  "feature_info": pd.DataFrame,              # index=features，列：mz, rt, name, formula, adduct, hmdb, kegg, class, polarity, confidence
  "sample_info": pd.DataFrame,               # index=samples，列：group, batch, qc(0/1), time, age, .../临床列
  "group_key":"group"|None,"groups":["case","control"]|None,
  "matrix_order":"samples_by_features",      # 或 features_by_samples（导入时归一化后统一为前者）
  "species":"human|mouse|unknown",
  "results":{},                              # 跨分析缓存 {"de":..,"plsda":..,"vip":..,"enrich":..,"annotation":..}
  "library": None|{"spectra":[{"id","precursor_mz","peaks":[[mz,int],...]}],"path"}  # MS2 库（懒加载）
  "rng": np.random.Generator,"workdir":str,"outdir":str,
  "engine":{"python":"..","mode":"builtin"},"notes":[],"warnings":[],
}
```

- **只读**：不得原地改 `ctx.X`/`feature_info`；写回由 registry 消费返回键（§2.5）。
- 缺失值一律 `np.nan`；未填补的分析必须自己处理缺失并在 warnings 说明。
- 大数据上限：n_samples ≤ 20000、n_features ≤ 200000；超过 20000 特征时多变量/网络分析内部先按方差/丰度筛到 20000 并 warning。

### 2.4 `dataset` 摘要

```json
{"id":"ds1","name":"MTBLS1","path":"/abs/x.tsv","source_format":"matrix_text",
 "n_samples":120,"n_features":1800,"species":"human","polarity":"pos",
 "feature_meta":{"has_mz":true,"has_rt":true,"n_annotated":312},
 "sample_columns":[{"key":"group","kind":"categorical","n_unique":3},{"key":"batch","kind":"categorical"}],
 "steps":["qc","filter","impute","normalize","transform","pca"],"current":"pca",
 "warnings":[],"notes":[],"history":[{"at":"ISO","step":"qc","params":{},"ok":true,"ms":812,"summary":".."}],
 "analyses_run":[{"at":"ISO","id":"plsda_vip","ok":true,"ms":120}]}
```

### 2.5 步骤 / 分析模块契约

结构同 dsh-spatial（`STEPS` / `ANALYSES` 列表；`{"id","title","group","method","needs","params","run"}`；
result 信封 `summary/tables/figures/sample_columns/sample_values/data_out/warnings/notes`）。
`needs` 取值：`groups | transform | plsda | de | vip | annotation | qc | clinical | ms2 | mz`。

**图的归属（重要）**：每个模块除 `STEPS`/`ANALYSES` 外还可以导出
```python
PLOTS = {"plsda_scores": fn, "vip_bar": fn}   # kind -> fn(ctx, params, outdir) -> figure dict
```
registry 会把所有模块的 `PLOTS` 合并成全局 `kind → fn` 表，`plot` 命令按 kind 分发（每个 kind 只能有一个 owner，
归属按 DESIGN.md §5 的分组：基础图在 `<prefix>_prep.py`/`<prefix>_plots.py`，其余各自模块）。

registry 额外消费：`"replace"`（`{"X","features","samples","raw","value_kind"}`）、`"store"`、`"sample_columns"+"sample_values"`。
图：PNG 白底 dpi≈140，**图上文字英文**；表 ≤500 行返回、完整写 `data_out`；缺前置 → `prerequisite`。

### 2.6 Host 层（Lead 固定）

前缀 `/metabolomics`，只接受回环；`GET /figure?id=`；长任务（下载、置换检验、交叉验证、网络）走 `POST` → `{job}` + `GET /jobs?id=`。

---

## 3. 步骤清单（顺序即左栏卡片顺序）

| step | 标题 | requires | 参数（默认） | 输出 |
|---|---|---|---|---|
| `qc` | 样本质控 | — | `group_col=null, qc_label="", min_features=0` | 每样本 TIC/检出特征/缺失率；QC 样本 RSD 分布；`qc_tic`+`qc_rsd`+`qc_missing` |
| `filter` | 特征过滤 | qc | `min_sample_frac=0.5, min_intensity=0, rsd_max=null` | 过滤前后特征数；RSD 表 |
| `impute` | 缺失值填补 | filter | `method=half_min, k=5` | 缺失率变化；方法说明 |
| `normalize` | 归一化 | impute | `method=pqn` | 归一化前后 TIC 分布；PQN 稀释因子表 |
| `transform` | 变换/标度 | normalize | `transform=log2, scale=auto` | 偏度/kurtosis 前后对比；标度方式说明 |
| `pca` | 主成分分析 | transform | `n_comp=3, color_by=group` | scores/loadings；碎石图；Hotelling T² 与 DModX |
| `plsda` | PLS-DA / OPLS-DA | transform | `method=plsda, n_comp=3, cv_folds=7, permutations=200` | R²X/R²Y/Q²、VIP、置换检验；scores + S-plot + VIP |
| `de` | 单变量差异 | transform | `group_col=null, case=null, control=null, method=welch, paired=false, fdr=0.05, fc=1.5` | 差异表 + 火山图 + FC 图 |
| `batch` | 批次校正 | normalize | `batch_key=null, method=rlsc, qc_label=""` | 校正前后 PCA 对比；批次混杂指标 |
| `enrich` | 通路富集 | de | `method=mummichog, db=kegg, p_cut=0.05, n_perm=1000` | 通路表 + barplot/dotplot |
| `biomarker` | 标志物筛选 | de | `model=rf, n_features=10, cv_folds=5, time_col=null, event_col=null` | 特征重要度/ROC/评分表 |
| `annotate` | 代谢物鉴定 | transform | `adduct=[M+H]+, ppm=10, formula=true, library=""` | 加合物/分子式候选表 + 质量精度图 |

- 重跑某步清空其下游（`history` 保留）。`de` 要求 `group_col` 有两个组（否则 `prerequisite`）。

---

## 4. 分析清单（≥34 项）

> 方法诚实：自实现写"内置实现（依据 X）"，不要冒充官方包（本机无 xcms/ropls/MetaboAnalystR/pyopenms/matchms）。

**质控与基础（group=质控与基础）**
| id | 标题 | method | needs | 参数 |
|---|---|---|---|---|
| `qc_summary` | 质控汇总 | 汇总表 + TIC/RSD 图（Dunn 2011 QC 原则） | qc | — |
| `rsd_qc` | QC 样本 RSD | RSD = SD/mean；QC vs 生物样本对比（Want 2010） | qc | `qc_label="", threshold=30` |
| `missing_pattern` | 缺失模式 | 缺失率分布 + 样本/特征缺失热图 | qc | `max_features=500` |
| `sample_outlier` | 样本离群 | PCA + Hotelling T² + DModX（Wold 1994） | transform | `n_comp=3, alpha=0.95` |
| `sample_corr` | 样本相关性 | Pearson/Spearman + 层次聚类 | transform | `method=pearson` |

**多变量判别（group=多变量判别）**
| id | 标题 | method | needs | 参数 |
|---|---|---|---|---|
| `plsda_vip` | 变量投影重要度（VIP） | VIP（Wold 1995；MetaboAnalyst） | plsda | `top_n=30, threshold=1` |
| `oplsda_splot` | OPLS-DA S-plot | OPLS（Trygg & Wold 2002；Bylesjö 2006） | plsda | `top_n=20` |
| `plsda_permutation` | PLS-DA 置换检验 | 标签置换 + R²Y/Q² 零分布（Westerhuis 2008） | plsda | `n_perms=200` |
| `plsda_cv` | 交叉验证与混淆矩阵 | k-fold / 留一（Brindle 2002 判别思路） | plsda | `folds=7` |
| `plsda_multiclass` | 多组判别 | 多类 PLS-DA（一对其余 + 综合 VIP） | groups | `n_comp=3` |
| `pca_loadings` | PCA 载荷与贡献 | 载荷表 + 贡献基因条形图 | transform | `pc=1, top_n=20` |

**差异与趋势（group=差异与趋势）**
| id | 标题 | method | needs | 参数 |
|---|---|---|---|---|
| `univariate_de` | 单变量差异 | Welch t / Mann-Whitney / limma + BH（Xia 2015 MetaboAnalyst） | groups | `method=welch, paired=false, fdr=0.05, fc=1.5` |
| `volcano_de` | 火山图 | log2FC vs −log10 p（Goxo 2019 火山图语义） | de | `n_labels=15, fc=1.5, fdr=0.05` |
| `fc_bar` | 倍数变化条形图 | top/bottom FC | de | `top_n=20` |
| `heatmap_top` | 差异特征热图 | z-score + 层次聚类 | de | `top_n=50, cluster=true` |
| `multi_group_de` | 多组 ANOVA | Welch ANOVA/Kruskal + 事后（Tukey/Dunn） | groups | `method=anova, posthoc=tukey` |
| `trend_course` | 时间/剂量趋势 | 分箱均值±SE + 线性/交换拟合（Fiehn 2008） | groups | `time_col, features=[]` |
| `dose_response` | 剂量响应（EC50） | 4 参数 logistic 拟合 | groups | `dose_col, feature, ci=0.95` |

**通路与化学分类（group=通路与化学分类）**
| id | 标题 | method | needs | 参数 |
|---|---|---|---|---|
| `mummichog_pathway` | Mummichog 式通路富集 | 质量映射 + 置换（Li 2013 mummichog；Uppal 2017） | de | `db=kegg, n_perm=1000, ppm=10, p_cut=0.05` |
| `ora_kegg` | KEGG 通路 ORA | 超几何 + BH（Kanehisa 2023 KEGG） | annotation | `p_cut=0.05, min_size=3` |
| `ora_hmdb_class` | HMDB 化学分类富集 | 超几何（Wishart 2022 HMDB 5.0） | annotation | `p_cut=0.05` |
| `metabolite_set_enrich` | 代谢物集富集（MSEA） | 秩基 GSEA（Subramanian 2005；Xia 2015 MSEA） | de | `n_perm=1000, min_size=5` |

**标志物与建模（group=标志物与建模）**
| id | 标题 | method | needs | 参数 |
|---|---|---|---|---|
| `rf_importance` | 随机森林重要度 | RF + 置换重要度（Breiman 2001） | groups | `n_trees=300, top_n=20` |
| `lasso_panel` | LASSO 代谢物组合 | L1 logistic/回归 + 交叉验证选 λ（Tibshirani 1996） | groups | `folds=5, alpha="cv"` |
| `roc_biomarker` | 单标志物 ROC | ROC/AUC + DeLong CI（DeLong 1988） | groups | `feature, ci=true` |
| `biomarker_score` | 组合评分 | 多标志物线性评分 + ROC + 分组 | groups | `features=[], weights="model"` |
| `model_validation` | 模型验证（自助法） | bootstrap/置换 + 乐观度校正（Harrell 1996） | groups | `n_boot=200, folds=5` |
| `survival_metabolite` | 代谢物分组生存 | 中位/最优 cutoff + logrank + Cox（lifelines） | clinical | `feature, time_col, event_col, cutoff=median` |

**批次与网络（group=批次与网络）**
| id | 标题 | method | needs | 参数 |
|---|---|---|---|---|
| `rlsc_batch` | QC-RLSC 批次校正 | QC 样本 LOESS 信号校正（Dunn 2011） | qc | `batch_key, qc_label, span=0.75` |
| `combat_batch` | ComBat 批次校正 | 经验贝叶斯（Johnson 2007；内置实现） | qc | `batch_key, group_key=null` |
| `correlation_network` | 共表达网络 | |r| 阈值 + 模块（WGCNA 思想 Langfelder 2008） | transform | `threshold=0.6, top_n=800` |
| `metabolite_modules` | 代谢物模块 | 层次聚类/相似度切分 + 模块特征值 | transform | `n_modules=auto, min_size=10` |

**代谢物鉴定（group=代谢物鉴定）**
| id | 标题 | method | needs | 参数 |
|---|---|---|---|---|
| `adduct_annot` | 加合物推断 | 常见加合物质量差匹配（正/负离子；MS-DIAL 语义） | mz | `polarity=pos, ppm=10, adducts=[]` |
| `formula_candidates` | 分子式候选 | CHNOPS 组合 + 质量精度 + RDBE 过滤（七黄金规则） | mz | `ppm=5, max_c=50, hetero_atoms=6` |
| `ms2_library_match` | MS2 库匹配 | 余弦相似度 + 前体匹配（MassBank/GNPS 语义；Stein 1994） | ms2 | `ppm=20, min_score=0.5, top_n=5` |
| `ms2_network` | 分子网络 | 余弦网络 + 连通分量（GNPS 分子网络；Wang 2016） | ms2 | `min_score=0.6, max_nodes=200` |

**验收要求**：以上 id 全部出现在 `/metabolomics/analyses`；缺前置返回中文 `prerequisite`。

---

## 5. 图（`kind` 白名单，与 impl.mjs 一致）

`qc_tic` `qc_rsd` `qc_missing` `sample_corr` `sample_dendro` `pca_scores` `pca_variance`
`hotelling_t2` `plsda_scores` `oplsda_splot` `vip_bar` `permutation_test` `volcano_de`
`fc_bar` `heatmap_top` `enrich_bar` `enrich_dot` `roc_curve` `rf_importance` `lasso_path`
`trend_plot` `dose_response` `mass_accuracy` `mass_network` `batch_pca`

- 通用参数：`{color_by:"group|batch|qc|none", top_n=20, dpi=140, figsize?}`；PCA/PLS 图必须画 95% 置信椭圆（Hotelling）。
- 颜色用 `mb_common.PALETTE`；分组色稳定（同一 group 值永远同色）。

## 6. 数据格式（`mb_io.py`）

| data_format | 识别 | 规则 |
|---|---|---|
| `matrix_text` | `.csv/.tsv/.txt(.gz)` | 分隔符嗅探；首列=特征 id（或首行=样本名）；自动判方向（行是样本→转置）；非数值行丢弃 + warning；从列名/表头提取 `mz`/`rt`（`mz`,`mass`,`m/z`；`rt`,`retention`,`RT(min)`） |
| `xlsx` | `.xlsx/.xls` | openpyxl；`sheet` 指定；同 matrix_text |
| `geo_series` | 含 `!series_matrix_table_begin` | 表段特征×样本 → 转置；`!Sample_characteristics_ch1` → sample_info |
| `archive` | `.zip/.tar/.tar.gz` | 解压到 `<workdir>/unpack/<uuid>`（防穿越）；找特征表/ISA-Tab/mzTab/msp；多候选 → `ambiguous`+candidates |
| `mztab` | `MTD`/`PRH` 开头 | 解析 `SEH/SEP` 或 `PEH/PEP` 段为矩阵 + `MTD` 元数据 |
| `isa_tab` | `m_*.txt`/`a_*.txt` | MetaboLights ISA-Tab：`m_*.txt` 样品×代谢物矩阵 + `s_*.txt`/`a_*.txt` 表型 |
| `msms_library` | `.msp/.mgf` | 解析谱图 → `library`（前体 m/z + 峰表），不当作定量矩阵 |
| `mzml_tic` | `.mzML/.mzXML` | 轻量 XML 解析 TIC 与 MS2 谱图（不做峰检测）；产出 QC 用 TIC 曲线或谱图库 |
| `rds`/`rdata` | R 桥 | 可用则读（matrix/data.frame/ExpressionSet） |

- 返回 `{"format","X","features","samples","feature_info","sample_info","notes","warnings","meta":{"polarity","has_mz","has_rt","matrix_order_original"}}`。
- 特征 id 生成：有 mz+rt → `mz{:.4f}_rt{:.2f}`；只有名字 → 名字；重名加序号。
- 缺失值：`NA/NaN/空/0`→`np.nan`（0 视为缺失仅在 `zero_as_missing=true` 时）。
- 样本表/临床表按样本名（大小写不敏感、去空白）join；匹配不上给 warning + matched/total。

## 7. 数据源

### 7.1 GEO `geo_search`
`{term, species:"human|mouse|all", method:"all|lc-ms|gc-ms|nmr|ms", min_samples, has_clinical:bool, limit=20, page=0, sort:"relevance|date|samples", source:"geo"}`
→ `{total,page,items:[{gse,title,title_zh,summary,species,n_samples,platform,study_type,release,has_matrix,has_clinical,method:"LC-MS|GC-MS|NMR|unknown",disease,tags}],query?,query_zh?,query_en?,translated?,relaxed?,used_term?}`
- 中文反查（`ZH_DICT` ≥200 条，含代谢组学词：代谢组学/脂质组学/非靶向/靶向/LC-MS/GC-MS/NMR/血清/血浆/尿液/粪便/胆汁/组织/糖尿病/肝癌…）。
- 检索式必须含 `(GSE[ETYP])`；`method` 过滤在客户端做（关键词 mass/LC-MS/GC-MS/NMR/spectrometry）。
- 少结果逐级放宽；无结果 → `not_found`；网络失败重试 ≤2 → `unsupported`。

### 7.2 GEO `geo_detail` / 7.3 `geo_download`
同 dsh-spatial（解析 `matrix/`+`suppl/`；`.part` 原子替换；单文件 ≤8GB；中文错误）。

### 7.4 MetaboLights（第二数据源，`source="metabolights"`）
- `ml_search {query, limit=20, page=0, assay_type?}` → `{total,page,items:[{accession:"MTBLS1",title,description,study_type,species,submission_date,n_assays,files_count,url}]}`
- `ml_detail {accession}` → `{accession,title,description,species,submission_date,contacts,publications,files:[{name,size,type,url}]}`
- `ml_download {accession, files:[], dest_dir, overwrite=false}` → 同 geo_download
- 端点以 EBI 公开 API 为准（`https://www.ebi.ac.uk/metabolights/ws/...`）。**必须先实测**（curl/python）再实现；接口不可用时不阻塞，返回 `unsupported` + 可读原因，并在 ping 里 `metabolights=false`。

## 8. Agent 工具 `metabolomics`

```
action: status | search | detail | download | ml_search | ml_detail | ml_download |
        open | demo | inspect | preview | run | plot | analyses | analysis |
        export | audit | suggest | recipe | samples | formats | reset | close
```
关键参数：`term, gse, accession, source, dest_dir, path, url, data_format, dataset, step, params, analysis, kind, what, op, out_path, recipe, actor`。
返回给模型：`summary.bullets` + 表前 20 行 + 图绝对路径 + 建议下一步。

## 9. demo 数据（`mb_state.py`）

- **72 样本**（case 30 / control 30 / QC 12）× **1600 特征**；`batch` 2 批（每批 QC 3 个）；`mz`/`rt` 列齐全；
  120 个真实差异特征（log2FC 0.8–2.0）；8 类代谢物（氨基酸/脂质/胆汁酸/糖/核苷酸/有机酸/胺/其它）带 HMDB class 与 KEGG 通路映射（内置字典 ≥120 条特征注释）；
  临床列 `age, bmi, os_time, os_event`（生存时间与一个"高危评分"相关，保证 KM 有差异）；**另置 40 个 MS2 谱图**（内嵌峰表）供 `ms2_library_match`/`ms2_network` 演示。
- 生成 < 2 秒；`test/make_demo.py` 复用。

## 10. 环境事实（禁止越界）

- Python 3.14.2：numpy 2.5.1 / pandas 2.3.3 / scipy 1.17.1 / sklearn 1.9.0 / matplotlib 3.11.0（仅 Agg）/ statsmodels 0.14.6 / openpyxl 3.1.5 / h5py 3.16 / networkx 3.6.1 / lifelines 0.30.3。
  **无 pyopenms、无 matchms、无 xcms、无 ropls、无 MetaboAnalystR、无 pyarrow、无 torch。**
  → PLS/OPLS/PLS-DA、VIP、置换、EC50、mummichog、RLSC、ComBat、HMDB 分类、分子网络**全部自己实现**（numpy/scipy/sklearn）；**禁止 pip/conda 安装**。
- R 4.4.2 可用（limma/preprocessCore/glmnet/randomForest/survival/ComplexHeatmap），**可选**加速；R 不在必须自动回退并标注 `lite`。
- `MPLBACKEND=Agg` + `MPLCONFIGDIR=<workdir>/mpl-cache`；`~/.matplotlib` 不可写。
- 网络：NCBI E-utilities/FTP 可达（用 `certifi` CA）；`rest.kegg.jp` 可达（通路库可在线取，缓存到 `<workdir>/genesets/`，失败用内置库）；HMDB 网站 403（**不要依赖**，用内置 HMDB class 词典）；MetaboLights REST 可达。
- 禁止：pip/conda 安装、改 profile、pnpm、写 `$DSH_HOME`、改 `../dsh-*` 其它插件。
- 提交前自检：`python3 -m py_compile lib/*.py`；`node --check client.js`。

## 11. 测试要求

- **mb-core**：`test/py_io_test.py`+`test/make_fixtures.py`：造 8 种 fixture（csv/tsv/xlsx/gz/转置矩阵/带 mz-rt 表/zip/mzTab/ISA-Tab/msp）断言 `X/features/samples/feature_info`；`inspect` 不加载全表；坏文件不崩。`test/py_geo_test.py`：mock `_urlopen` 断言 GEO 检索/详情/下载 + 中文翻译 + 放宽 + MetaboLights 解析（固定 JSON）。
- **mb-base**：`test/py_prep_test.py`：demo 上 qc→filter→impute→normalize→transform→pca 全通；缺失填补后无 NaN；PQN 后 TIC 变异下降；PCA 前两主成分能分开 case/control（AUC>0.6）。
- **mb-stat**：`test/py_plsda_test.py`：Q²>0.3、置换检验 p<0.05、VIP 前 20 命中注入差异特征 ≥50%；`test/py_de_test.py`：单变量差异 top100 命中率 ≥60%（Welch）、火山图点数一致、ANOVA/趋势/4PL 分支可用。
- **mb-adv**：`test/py_adv_test.py`：mummichog 返回 ≥1 通路且 p∈[0,1]；RF/ROC AUC 合理（注入强信号时 >0.8）；RLSC 后批次与分组混杂度下降；加合物/分子式在 ppm 内自洽；MS2 匹配自身谱图得分 ≈1。
- **mb-ui**：`test/client-logic.test.mjs`：纯函数 + 静态约束。
- 测试不依赖 npm 包。

## 12. 交付验收（Lead）

1. `plugin_manager install_bundle` 成功；`GET /metabolomics/health` 200 且 `engine.ready=true`。
2. demo 全流程 12 步全通，每步有 summary/图。
3. 真实 GEO 检索一次（"serum metabolomics LC-MS"）+ 真实下载一个文件成功（网络受限时 mock 证明链路）；MetaboLights 检索一次。
4. 导入真实特征表（工作区里若有，或用 fixture）→ 清洗 → PLS-DA → 差异 → 富集 → 标志物。
5. UI：侧边栏出现「代谢工作台」，整页可打开，主要 tab 不白屏。
