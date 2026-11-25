/**
 * dsh-metabolomics — 浏览器端整页 UI（Client 半边）
 *
 * 形态照 dsh-geobulk（同插件形态的参考实现）：window.__ModuleLoader__.load({ id:'dsh-metabolomics', factory(require){...} })，
 * 只 require('react')，不碰任何 @deepseek-ai Client 包（宿主版本一变就会炸）。
 *
 * 数据面：全部走同源相对路径 /metabolomics/*（见 DESIGN.md §2.6 / §7）；
 * 接口以 lib/impl.mjs 的实际路由为准，本文件不调用不存在的接口。
 *
 * 文件分两层：
 *   1) 纯函数层（本段到 MB_TEST）：STEPS 元数据 / 数值与时间格式化 / 分组取色 /
 *      二维散点→像素（质控小图）/ GEO 与 MetaboLights 搜索 payload / 建议→中文指令 /
 *      可注入 fetch 的 API 客户端 / 异步任务轮询状态机 / 服务端响应归一化。
 *      不碰 DOM、不 require react；node 环境（typeof window === 'undefined'）挂到
 *      globalThis.__MB_TEST__ 供 test/client-logic.test.mjs 直接 import。
 *   2) 浏览器层（factory 内）：React 组件 + 样式 + 槽注册。
 *
 * 约定：类名一律 mb- 前缀；颜色只用 --dsw-alias-* 主题变量（见 CSS 段），
 * 数据色（分组/连续值）用 JS 常量 PALETTE / CONTINUOUS_STOPS。
 */
;(function () {
  'use strict'

  var NS = 'dsh-metabolomics'
  var API = '/metabolomics'
  var POLL_MS = 1500
  var POLL_TIMEOUT_MS = 30 * 60 * 1000
  var DEFAULT_API_TIMEOUT_MS = 60000
  var HEAVY_API_TIMEOUT_MS = 600000
  var MAX_HISTORY_ROWS = 20
  var MAX_TABLE_RENDER_ROWS = 200
  var DEFAULT_GEO_DIR = '~/.dsh/metabolomics/GEO'
  var DEFAULT_ML_DIR = '~/.dsh/metabolomics/MetaboLights'

  /** 分组调色板（数据色，不参与主题；同一 group 值永远同色）。 */
  var PALETTE = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948',
    '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac', '#1f77b4', '#ff7f0e',
    '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
  ]

  /** 连续值色带（数据色）：低→高，用于 TIC / 缺失率映射。 */
  var CONTINUOUS_STOPS = ['#3b4cc0', '#6a9fdb', '#9ec6e8', '#e8d4a8', '#f0a35e', '#c0392b']

  /** 自检等级样式（由主题色派生）。 */
  var AUDIT_LEVEL_STYLE = {
    ok: { tone: 'ok', labelKey: 'auditOk' },
    warn: { tone: 'warn', labelKey: 'auditWarn' },
    fail: { tone: 'fail', labelKey: 'auditFail' },
  }

  /** 图 kind 白名单（与 DESIGN.md §5 / impl.mjs PLOT_KINDS 逐字一致）。 */
  var PLOT_KINDS = [
    'qc_tic', 'qc_rsd', 'qc_missing', 'sample_corr', 'sample_dendro', 'pca_scores', 'pca_variance',
    'hotelling_t2', 'plsda_scores', 'oplsda_splot', 'vip_bar', 'permutation_test', 'volcano_de',
    'fc_bar', 'heatmap_top', 'enrich_bar', 'enrich_dot', 'roc_curve', 'rf_importance', 'lasso_path',
    'trend_plot', 'dose_response', 'mass_accuracy', 'mass_network', 'batch_pca',
  ]

  /** 记录 tab 的导出按钮组（what 与 DESIGN.md §3 export / impl.mjs EXPORT_KINDS 一致）。 */
  var EXPORT_KINDS = [
    { id: 'expr_csv', labelKey: 'exportExprCsv', hint: 'CSV' },
    { id: 'de_csv', labelKey: 'exportDeCsv', hint: 'CSV' },
    { id: 'vip_csv', labelKey: 'exportVipCsv', hint: 'CSV' },
    { id: 'enrich_csv', labelKey: 'exportEnrichCsv', hint: 'CSV' },
    { id: 'biomarker_csv', labelKey: 'exportBiomarkerCsv', hint: 'CSV' },
    { id: 'sample_info_csv', labelKey: 'exportSampleCsv', hint: 'CSV' },
    { id: 'summary_json', labelKey: 'exportSummary', hint: 'JSON' },
    { id: 'protocol', labelKey: 'exportProtocol', hint: 'JSON' },
    { id: 'methods', labelKey: 'exportMethods', hint: 'Markdown' },
    { id: 'report', labelKey: 'exportReport', hint: 'Markdown' },
    { id: 'script', labelKey: 'exportScript', hint: 'Python' },
  ]

  /** 记录 tab 上方快速导出（只放最常用的三个）。 */
  var QUICK_EXPORTS = ['expr_csv', 'summary_json', 'report']

  /** 得分图着色维度（DESIGN.md §5 color_by 白名单：group|batch|qc|none）。 */
  var COLOR_BY = [
    { value: 'group', labelKey: 'colorGroup' },
    { value: 'batch', labelKey: 'colorBatch' },
    { value: 'qc', labelKey: 'colorQc' },
    { value: 'none', labelKey: 'colorNone' },
  ]

  /** GEO 检测方法筛选（DESIGN.md §7.1 method 取值；服务端在客户端侧过滤）。 */
  var GEO_METHODS = [
    { value: 'all', labelKey: 'geoMethodAll' },
    { value: 'lc-ms', label: 'LC-MS' },
    { value: 'gc-ms', label: 'GC-MS' },
    { value: 'nmr', label: 'NMR' },
    { value: 'ms', label: 'MS' },
  ]

  /** 代谢组学数据格式（服务端 /formats 不可用时的静态兜底，与 DESIGN.md §6 一致）。 */
  var DATA_FORMATS = [
    'auto', 'matrix_text', 'xlsx', 'geo_series', 'archive', 'mztab', 'isa_tab',
    'msms_library', 'mzml_tic', 'rds', 'rdata',
  ]

  /**
   * 各 tab 的分析白名单：只决定「这个 tab 展示哪些分析」，
   * 标题/方法/参数/可用性全部来自 GET /metabolomics/analyses（不硬编码 38 项分析的 schema）。
   */
  var TAB_ANALYSES = {
    qc: ['qc_summary', 'rsd_qc', 'missing_pattern', 'sample_outlier', 'sample_corr'],
    multivariate: ['plsda_vip', 'oplsda_splot', 'plsda_permutation', 'plsda_cv', 'plsda_multiclass', 'pca_loadings'],
    de: ['univariate_de', 'volcano_de', 'fc_bar', 'heatmap_top', 'multi_group_de', 'trend_course', 'dose_response'],
    pathway: ['mummichog_pathway', 'ora_kegg', 'ora_hmdb_class', 'metabolite_set_enrich'],
    biomarker: ['rf_importance', 'lasso_panel', 'roc_biomarker', 'biomarker_score', 'model_validation', 'survival_metabolite'],
    annotation: ['adduct_annot', 'formula_candidates', 'ms2_library_match', 'ms2_network'],
  }

  /**
   * 各 tab 的快捷出图（kind + 默认参数）。图统一走 POST /metabolomics/plot，
   * 由 PlotBlock 自动加载；没有前置时服务端返回可读 prerequisite，页面只显示提示。
   */
  var TAB_PLOTS = {
    qc: [{ kind: 'qc_rsd', titleKey: 'qcRsdTitle' }, { kind: 'qc_missing', titleKey: 'qcMissingTitle' }],
    multivariate: [
      { kind: 'plsda_scores', titleKey: 'mvScores', params: { color_by: 'group' } },
      { kind: 'vip_bar', titleKey: 'mvVip' },
      { kind: 'oplsda_splot', titleKey: 'mvSplot' },
      { kind: 'permutation_test', titleKey: 'mvPerm' },
    ],
    de: [
      { kind: 'volcano_de', titleKey: 'deVolcano' },
      { kind: 'heatmap_top', titleKey: 'deHeatmap' },
      { kind: 'fc_bar', titleKey: 'deFcBar' },
    ],
    pathway: [{ kind: 'enrich_bar', titleKey: 'pathwayBar' }, { kind: 'enrich_dot', titleKey: 'pathwayDot' }],
    biomarker: [
      { kind: 'roc_curve', titleKey: 'biomarkerRoc' },
      { kind: 'rf_importance', titleKey: 'biomarkerImportance' },
      { kind: 'lasso_path', titleKey: 'biomarkerLasso' },
    ],
    annotation: [
      { kind: 'mass_accuracy', titleKey: 'annotationMassAcc' },
      { kind: 'mass_network', titleKey: 'annotationNetwork' },
    ],
  }

  // ==========================================================================
  // 流程步骤：顺序即左栏卡片顺序；requires 与 DESIGN.md §3 / impl.mjs STEPS 逐字一致。
  // params 用通用渲染（number/select/boolean/column/text/genes/json）。
  // ==========================================================================
  var STEPS = [
    {
      id: 'qc',
      title: '样本质控',
      icon: 'qc',
      requires: [],
      params: [
        { key: 'group_col', label: '分组列', type: 'column', default: null },
        { key: 'qc_label', label: 'QC 样本标签', type: 'text', default: '', placeholder: '如 QC' },
        { key: 'min_features', label: '每样本最少检出特征', type: 'number', default: 0, min: 0, step: 10 },
      ],
    },
    {
      id: 'filter',
      title: '特征过滤',
      icon: 'filter',
      requires: ['qc'],
      params: [
        { key: 'min_sample_frac', label: '最低检出样本比例', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05 },
        { key: 'min_intensity', label: '最低强度', type: 'number', default: 0, min: 0, step: 100 },
        { key: 'rsd_max', label: 'RSD 上限（%）', type: 'number', default: null, min: 0, step: 5 },
      ],
    },
    {
      id: 'impute',
      title: '缺失值填补',
      icon: 'impute',
      requires: ['filter'],
      params: [
        {
          key: 'method', label: '方法', type: 'select', default: 'half_min',
          options: [
            { value: 'half_min', label: '最小值一半' },
            { value: 'knn', label: 'kNN' },
            { value: 'median', label: '中位数' },
            { value: 'mean', label: '均值' },
            { value: 'none', label: '不填补' },
          ],
        },
        { key: 'k', label: 'kNN 邻居数', type: 'number', default: 5, min: 1, step: 1 },
      ],
    },
    {
      id: 'normalize',
      title: '归一化',
      icon: 'normalize',
      requires: ['impute'],
      params: [
        {
          key: 'method', label: '方法', type: 'select', default: 'pqn',
          options: [
            { value: 'pqn', label: 'PQN' },
            { value: 'tic', label: 'TIC 归一' },
            { value: 'median', label: '中位数' },
            { value: 'mean', label: '均值' },
            { value: 'none', label: '不归一化' },
          ],
        },
      ],
    },
    {
      id: 'transform',
      title: '变换/标度',
      icon: 'transform',
      requires: ['normalize'],
      params: [
        {
          key: 'transform', label: '变换', type: 'select', default: 'log2',
          options: [
            { value: 'log2', label: 'log2' },
            { value: 'log10', label: 'log10' },
            { value: 'ln', label: 'ln' },
            { value: 'cube_root', label: '立方根' },
            { value: 'none', label: '不变换' },
          ],
        },
        {
          key: 'scale', label: '标度', type: 'select', default: 'auto',
          options: [
            { value: 'auto', label: '自动（Pareto/UV）' },
            { value: 'uv', label: 'UV（单位方差）' },
            { value: 'pareto', label: 'Pareto' },
            { value: 'mean_center', label: '仅中心化' },
            { value: 'none', label: '不标度' },
          ],
        },
      ],
    },
    {
      id: 'pca',
      title: '主成分分析',
      icon: 'pca',
      requires: ['transform'],
      params: [
        { key: 'n_comp', label: '主成分数', type: 'number', default: 3, min: 2, max: 20, step: 1 },
        {
          key: 'color_by', label: '着色', type: 'select', default: 'group',
          options: [
            { value: 'group', label: '分组' },
            { value: 'batch', label: '批次' },
            { value: 'qc', label: 'QC 样本' },
            { value: 'none', label: '单色' },
          ],
        },
      ],
    },
    {
      id: 'plsda',
      title: 'PLS-DA / OPLS-DA',
      icon: 'plsda',
      requires: ['transform'],
      params: [
        {
          key: 'method', label: '方法', type: 'select', default: 'plsda',
          options: [{ value: 'plsda', label: 'PLS-DA' }, { value: 'oplsda', label: 'OPLS-DA' }],
        },
        { key: 'n_comp', label: '成分数', type: 'number', default: 3, min: 1, max: 10, step: 1 },
        { key: 'cv_folds', label: '交叉验证折数', type: 'number', default: 7, min: 2, max: 20, step: 1 },
        { key: 'permutations', label: '置换次数', type: 'number', default: 200, min: 0, max: 2000, step: 50 },
      ],
    },
    {
      id: 'de',
      title: '单变量差异',
      icon: 'de',
      requires: ['transform'],
      params: [
        { key: 'group_col', label: '分组列', type: 'column', default: null },
        { key: 'case', label: '实验组', type: 'text', default: null, placeholder: '如 case' },
        { key: 'control', label: '对照组', type: 'text', default: null, placeholder: '如 control' },
        {
          key: 'method', label: '检验方法', type: 'select', default: 'welch',
          options: [
            { value: 'welch', label: 'Welch t' },
            { value: 'mannwhitney', label: 'Mann-Whitney' },
            { value: 'limma', label: 'limma' },
          ],
        },
        { key: 'paired', label: '配对检验', type: 'boolean', default: false },
        { key: 'fdr', label: 'FDR 阈值', type: 'number', default: 0.05, min: 0.001, max: 0.5, step: 0.005 },
        { key: 'fc', label: '倍数变化阈值', type: 'number', default: 1.5, min: 1, step: 0.1 },
      ],
    },
    {
      id: 'batch',
      title: '批次校正',
      icon: 'batch',
      requires: ['normalize'],
      params: [
        { key: 'batch_key', label: '批次列', type: 'column', default: null },
        {
          key: 'method', label: '方法', type: 'select', default: 'rlsc',
          options: [{ value: 'rlsc', label: 'QC-RLSC' }, { value: 'combat', label: 'ComBat' }, { value: 'none', label: '不校正' }],
        },
        { key: 'qc_label', label: 'QC 样本标签', type: 'text', default: '', placeholder: '如 QC' },
      ],
    },
    {
      id: 'enrich',
      title: '通路富集',
      icon: 'enrich',
      requires: ['de'],
      params: [
        {
          key: 'method', label: '方法', type: 'select', default: 'mummichog',
          options: [
            { value: 'mummichog', label: 'Mummichog 式' },
            { value: 'ora', label: 'ORA 超几何' },
            { value: 'msea', label: 'MSEA 秩基' },
          ],
        },
        { key: 'db', label: '数据库', type: 'select', default: 'kegg', options: [{ value: 'kegg', label: 'KEGG' }, { value: 'hmdb_class', label: 'HMDB 化学分类' }] },
        { key: 'p_cut', label: 'p 阈值', type: 'number', default: 0.05, min: 0.001, max: 0.5, step: 0.005 },
        { key: 'n_perm', label: '置换次数', type: 'number', default: 1000, min: 0, max: 10000, step: 100 },
      ],
    },
    {
      id: 'biomarker',
      title: '标志物筛选',
      icon: 'biomarker',
      requires: ['de'],
      params: [
        {
          key: 'model', label: '模型', type: 'select', default: 'rf',
          options: [
            { value: 'rf', label: '随机森林' },
            { value: 'lasso', label: 'LASSO' },
            { value: 'plsda', label: 'PLS-DA' },
          ],
        },
        { key: 'n_features', label: '特征数', type: 'number', default: 10, min: 1, max: 200, step: 1 },
        { key: 'cv_folds', label: '交叉验证折数', type: 'number', default: 5, min: 2, max: 20, step: 1 },
        { key: 'time_col', label: '生存时间列', type: 'column', default: null },
        { key: 'event_col', label: '生存事件列', type: 'column', default: null },
      ],
    },
    {
      id: 'annotate',
      title: '代谢物鉴定',
      icon: 'annotate',
      requires: ['transform'],
      params: [
        {
          key: 'adduct', label: '加合物', type: 'select', default: '[M+H]+',
          options: [
            { value: '[M+H]+', label: '[M+H]+' },
            { value: '[M+Na]+', label: '[M+Na]+' },
            { value: '[M+NH4]+', label: '[M+NH4]+' },
            { value: '[M-H]-', label: '[M-H]-' },
            { value: '[M+Cl]-', label: '[M+Cl]-' },
            { value: '[M+HCOO]-', label: '[M+HCOO]-' },
          ],
        },
        { key: 'ppm', label: '质量误差（ppm）', type: 'number', default: 10, min: 0.1, max: 100, step: 0.5 },
        { key: 'formula', label: '推断分子式', type: 'boolean', default: true },
        { key: 'library', label: 'MS2 谱库路径', type: 'text', default: '', placeholder: '/absolute/path/library.msp' },
      ],
    },
  ]

  var STEP_BY_ID = {}
  for (var stepIndex = 0; stepIndex < STEPS.length; stepIndex++) STEP_BY_ID[STEPS[stepIndex].id] = STEPS[stepIndex]

  function stepById(id) {
    var key = typeof id === 'string' ? id : ''
    return STEP_BY_ID[key] || null
  }

  /** 中栏 tab 顺序（概览 / 质控 / 多变量 / 差异 / 通路 / 标志物 / 鉴定 / 记录 / 日志）。 */
  var TAB_KEYS = ['overview', 'qc', 'multivariate', 'de', 'pathway', 'biomarker', 'annotation', 'records', 'log']

  var TAB_META = {
    overview: { titleKey: 'tabOverview' },
    qc: { titleKey: 'tabQc' },
    multivariate: { titleKey: 'tabMultivariate' },
    de: { titleKey: 'tabDe' },
    pathway: { titleKey: 'tabPathway' },
    biomarker: { titleKey: 'tabBiomarker' },
    annotation: { titleKey: 'tabAnnotation' },
    records: { titleKey: 'tabRecords' },
    log: { titleKey: 'tabLog' },
  }

  /** 步骤跑完自动切到对应 tab（没有映射的留在原地）。 */
  var TAB_FOR_STEP = {
    qc: 'qc',
    filter: 'qc',
    impute: 'qc',
    normalize: 'qc',
    transform: 'multivariate',
    pca: 'multivariate',
    plsda: 'multivariate',
    de: 'de',
    batch: 'multivariate',
    enrich: 'pathway',
    biomarker: 'biomarker',
    annotate: 'annotation',
  }

  /**
   * 分析分组 → tab。分组文案取自 DESIGN.md §4（质控与基础 / 多变量判别 / 差异与趋势 /
   * 通路与化学分类 / 标志物与建模 / 批次与网络 / 代谢物鉴定）；未映射的分组落到
   * 「多变量」tab，保证服务端返回的每一项分析都有落点。
   */
  var ANALYSIS_TAB = {
    '质控与基础': 'qc',
    'QC': 'qc',
    '多变量判别': 'multivariate',
    '批次与网络': 'multivariate',
    '差异与趋势': 'de',
    '通路与化学分类': 'pathway',
    '标志物与建模': 'biomarker',
    '代谢物鉴定': 'annotation',
  }

  var ANALYSIS_TAB_FALLBACK = 'multivariate'

  // ==========================================================================
  // 文案表：每行 [key, 中文, English]。DICT_ZH / DICT_EN 由同一份表生成，
  // 从结构上保证两侧 key 集一致（单测仍会断言一次）。参数 label 等来自服务端
  // schema 的文案属于数据，不走字典；locale/*.json 只放宿主侧导航需要的词条。
  // ==========================================================================
  var WORDS = [
    ['nav', '代谢工作台', 'Metabolomics'],
    ['appTitle', '代谢组学工作台', 'Metabolomics Workbench'],
    ['appSubtitle', 'GEO/MetaboLights 检索 · 特征表导入 · 质控 · PLS-DA · 差异 · 通路 · 标志物 · 鉴定', 'GEO/MetaboLights search · Feature table · QC · PLS-DA · DE · Pathways · Biomarkers · Annotation'],
    ['refresh', '刷新', 'Refresh'],
    ['refreshHint', '重新读取工作台状态', 'Re-read workbench state'],
    ['retry', '重试', 'Retry'],
    ['loading', '加载中…', 'Loading…'],
    ['loadingData', '正在读取工作台…', 'Reading workbench…'],
    ['noData', '尚未载入数据集', 'No dataset loaded'],
    ['errorTitle', '出错了', 'Something went wrong'],
    ['crashTitle', '这个面板渲染失败', 'This panel failed to render'],
    ['crashHint', '其它面板仍可使用，可以点击重试恢复。', 'Other panels still work; retry to recover.'],
    ['close', '关闭', 'Close'],
    ['cancel', '取消', 'Cancel'],
    ['copy', '复制', 'Copy'],
    ['copied', '已复制', 'Copied'],
    ['copyFailed', '复制失败（浏览器不允许写入剪贴板）', 'Copy failed (clipboard not allowed)'],
    ['search', '搜索', 'Search'],
    ['all', '全部', 'All'],
    ['none', '无', 'None'],
    ['yes', '是', 'Yes'],
    ['no', '否', 'No'],
    ['unknown', '未知', 'Unknown'],
    ['dataset', '数据集', 'Dataset'],
    ['samples', '样本', 'Samples'],
    ['features', '特征', 'Features'],
    ['polarity', '极性', 'Polarity'],
    ['species', '物种', 'Species'],
    ['colorGroup', '分组', 'Group'],
    ['colorBatch', '批次', 'Batch'],
    ['colorQc', 'QC 样本', 'QC samples'],
    ['colorNone', '单色', 'Single colour'],
    ['engine', '引擎', 'Engine'],
    ['warnings', '警告', 'Warnings'],
    ['notes', '说明', 'Notes'],
    ['current', '当前', 'Current'],
    ['completed', '已完成', 'Done'],
    ['pending', '待运行', 'Pending'],
    ['running', '运行中', 'Running'],
    ['failed', '失败', 'Failed'],
    ['plot', '出图', 'Plot'],
    ['plotting', '出图中…', 'Plotting…'],
    ['noFigure', '暂无图', 'No figure yet'],
    ['figureFail', '图片加载失败（文件可能已被清理）', 'Image failed to load (file may be cleaned)'],
    ['exportBtn', '导出', 'Export'],
    ['exported', '已导出', 'Exported'],
    ['exportFail', '导出失败', 'Export failed'],
    ['copyPath', '复制路径', 'Copy path'],
    ['pathLabel', '文件路径', 'File path'],
    ['tableRows', '共 {v} 行', '{v} rows'],
    ['tableTruncated', '仅展示前 {v} 行', 'showing first {v}'],
    ['emptyTable', '空表', 'Empty table'],
    ['total', '总计', 'Total'],
    ['selectAll', '全选', 'Select all'],
    ['clearSelection', '清空', 'Clear'],
    ['of', '/', '/'],
    ['selectDataset', '切换数据集', 'Switch dataset'],
    ['closeDataset', '关闭数据集', 'Close dataset'],
    ['workdir', '工作目录', 'Workdir'],
    ['engineReady', '引擎就绪', 'Engine ready'],
    ['engineBooting', '引擎启动中', 'Engine starting'],
    ['engineBad', '引擎不可用', 'Engine unavailable'],
    ['engineGuideTitle', 'Python 引擎尚未就绪', 'Python engine is not ready'],
    ['engineGuideBody', '工作台依赖本机 Python 环境（numpy / pandas / scipy / sklearn / matplotlib / statsmodels / openpyxl / h5py）。整页仍可浏览，但导入、计算与出图需要引擎先就绪。', 'The workbench needs a local Python environment (numpy / pandas / scipy / sklearn / matplotlib / statsmodels / openpyxl / h5py). You can browse the page, but import, compute and plotting need the engine.'],
    ['engineGuideSteps', '排查顺序：① 查看 engine.log 的报错；② 确认插件设置里的 python3 路径可执行；③ 重启插件或宿主后点「重试」。', 'Check: 1) engine.log for the traceback; 2) python3 path in plugin settings; 3) restart the plugin or host, then retry.'],
    ['pythonLabel', 'Python', 'Python'],
    ['engineInfo', '引擎信息', 'Engine info'],
    ['statusOk', '正常', 'OK'],
    ['statusWarn', '注意', 'Warn'],
    ['statusFail', '异常', 'Error'],
    ['pipelineTitle', '分析流程', 'Pipeline'],
    ['pipelineHint', '按顺序运行；重跑某步会清空其下游结果', 'Run in order; re-running a step clears its downstream results'],
    ['datasetsTitle', '工作台', 'Workbench'],
    ['datasetNoDataset', '还没有数据集', 'No dataset yet'],
    ['datasetOpenHint', '搜 GEO / MetaboLights、导入 LC-MS 特征表（csv/tsv/xlsx/zip/mzTab/ISA-Tab），或载入内置示例立刻试一遍。', 'Search GEO / MetaboLights, import an LC-MS feature table (csv/tsv/xlsx/zip/mzTab/ISA-Tab), or load the built-in demo.'],
    ['datasetStepsDone', '已完成 {v} 步', '{v} steps done'],
    ['datasetCurrentStep', '当前 {v}', 'current {v}'],
    ['datasetSamples', '样本 {v}', 'samples {v}'],
    ['datasetFeatures', '特征 {v}', 'features {v}'],
    ['datasetPolarity', '极性 {v}', 'polarity {v}'],
    ['datasetAnnotated', '注释 {v}', 'annotated {v}'],
    ['datasetGroupColumn', '分组列 {v}', 'group column {v}'],
    ['stepStatusDone', '已完成', 'Done'],
    ['stepStatusRunning', '运行中', 'Running'],
    ['stepStatusReady', '可运行', 'Ready'],
    ['stepStatusBlocked', '待前置', 'Blocked'],
    ['stepStatusFailed', '失败', 'Failed'],
    ['stepRun', '运行', 'Run'],
    ['stepRerun', '重跑', 'Re-run'],
    ['stepRunning', '运行中…', 'Running…'],
    ['stepParams', '参数', 'Params'],
    ['stepNeedPrereq', '需要先完成：', 'Needs: '],
    ['stepSummaryEmpty', '—', '—'],
    ['runStepDone', '步骤完成', 'Step finished'],
    ['analysisCatalog', '分析目录', 'Analyses'],
    ['analysisHint', '从 /metabolomics/analyses 动态读取的全部分析', 'Every analysis is read from /metabolomics/analyses'],
    ['analysisTitle', '代谢组学分析目录', 'Metabolomics analyses'],
    ['analysisSubtitle', '按分组浏览；参数表单由服务端 schema 生成', 'Browse by group; the param form is generated from the server schema'],
    ['analysisClose', '收起', 'Collapse'],
    ['analysisRun', '运行分析', 'Run analysis'],
    ['analysisRunning', '运行中…', 'Running…'],
    ['analysisNoParams', '该分析没有可调参数', 'This analysis has no tunable params'],
    ['analysisNeedDataset', '请先载入数据集', 'Load a dataset first'],
    ['analysisUnavailable', '不可用', 'Unavailable'],
    ['analysisNeeds', '前置', 'Needs'],
    ['analysisEmpty', '这个分组下还没有可用分析（引擎未就绪或缺少前置步骤）', 'No analysis in this group (engine not ready or prerequisites missing)'],
    ['analysisGroupOther', '其它', 'Other'],
    ['analysisSelectHint', '左侧选一个分析，右侧填参数后运行', 'Pick an analysis on the left, tune params, then run'],
    ['analysisResult', '分析结果', 'Result'],
    ['analysisMethod', '方法', 'Method'],
    ['analysisRetry', '重试', 'Retry'],
    ['analysisFallbackTitle', '该分析暂未在本机引擎注册', 'This analysis is not registered in the local engine yet'],
    ['tabOverview', '概览', 'Overview'],
    ['tabQc', '质控', 'QC'],
    ['tabMultivariate', '多变量', 'Multivariate'],
    ['tabDe', '差异', 'DE'],
    ['tabPathway', '通路', 'Pathways'],
    ['tabBiomarker', '标志物', 'Biomarkers'],
    ['tabAnnotation', '鉴定', 'Annotation'],
    ['tabRecords', '记录', 'Records'],
    ['tabLog', '日志', 'Log'],
    ['overviewTitle', '数据集摘要', 'Dataset summary'],
    ['overviewHint', '先看样本与特征形状、极性与注释覆盖，再决定走哪条分析路线', 'Check sample/feature shape, polarity and annotation coverage first'],
    ['metricSamples', '样本数', 'Samples'],
    ['metricFeatures', '特征数', 'Features'],
    ['metricPolarity', '极性', 'Polarity'],
    ['metricAnnotated', '注释特征', 'Annotated'],
    ['metricGroups', '分组', 'Groups'],
    ['metricEngine', '引擎', 'Engine'],
    ['pcaScores', 'PCA 得分图', 'PCA scores'],
    ['pcaScree', 'PCA 碎石图', 'PCA scree'],
    ['qcTic', 'TIC 图', 'TIC'],
    ['overviewPcaEmpty', '先运行 PCA 步骤，得分图会按分组上色', 'Run the PCA step first to colour scores by group'],
    ['overviewTicEmpty', '先运行质控步骤才会生成 TIC 图', 'Run the QC step to generate the TIC figure'],
    ['miniMapTitle', '样本质控小图', 'Sample QC scatter'],
    ['miniMapHint', '用质控结果表里的 TIC / 缺失率画散点（前端 SVG）；完整图见「质控」tab', 'Inline SVG scatter from TIC / missing-rate columns in the QC table; full figures are under QC'],
    ['nextSteps', '建议下一步', 'Next steps'],
    ['noSuggestions', '暂无建议（引擎未就绪或已验证到终点）', 'No suggestions (engine not ready, or at the end of validated steps)'],
    ['qcTitle', '质控', 'QC'],
    ['qcHint', '质控汇总 / QC-RSD / 缺失模式 / 样本离群 / 样本相关', 'QC summary / QC-RSD / missingness / outliers / correlation'],
    ['qcEmpty', '还没有质控结果，选一个分析运行', 'No QC result yet — pick an analysis and run it'],
    ['qcRsdTitle', 'QC-RSD 分布', 'QC-RSD distribution'],
    ['qcMissingTitle', '特征缺失率', 'Feature missingness'],
    ['qcSampleScatterHint', '质控结果表里没有可用的 TIC/缺失率列，先运行 qc 步骤或 qc_summary', 'No TIC / missing-rate columns in the QC tables yet — run the qc step or qc_summary'],
    ['qcRunStep', '运行质控步骤', 'Run QC step'],
    ['qcRunStepBusy', '质控运行中…', 'Running QC…'],
    ['qcSampleCount', '样本点', 'Sample points'],
    ['mvTitle', '多变量判别', 'Multivariate'],
    ['mvHint', 'PLS-DA / OPLS-DA：VIP、S-plot、置换检验、交叉验证、载荷', 'PLS-DA / OPLS-DA: VIP, S-plot, permutation, cross-validation, loadings'],
    ['mvEmpty', '还没有多变量结果，选一个分析运行', 'No multivariate result yet — pick an analysis and run it'],
    ['mvQuality', '模型质量', 'Model quality'],
    ['mvR2Y', 'R²Y', 'R²Y'],
    ['mvQ2', 'Q²', 'Q²'],
    ['mvPermP', '置换 p', 'Permutation p'],
    ['mvNeedPlsda', '先运行「PLS-DA / OPLS-DA」步骤，VIP 与 S-plot 才有意义', 'Run the PLS-DA step first — VIP and S-plot need it'],
    ['mvVip', 'VIP 条形图', 'VIP bar'],
    ['mvSplot', 'OPLS-DA S-plot', 'OPLS-DA S-plot'],
    ['mvPerm', '置换检验', 'Permutation test'],
    ['mvScores', 'PLS-DA 得分图', 'PLS-DA scores'],
    ['deTitle', '差异', 'Differential analysis'],
    ['deHint', '单变量差异（Welch / Mann-Whitney / limma + BH）与火山图/热图/趋势', 'Univariate DE (Welch / Mann-Whitney / limma + BH) with volcano, heatmap and trends'],
    ['deEmpty', '还没有差异结果，选一个分析运行', 'No DE result yet — pick an analysis and run it'],
    ['deExport', '导出差异表', 'Export DE table'],
    ['deVolcano', '火山图', 'Volcano'],
    ['deHeatmap', '差异热图', 'DE heatmap'],
    ['deFcBar', '倍数变化', 'Fold change'],
    ['deNeedDe', '差异图需要先做单变量差异（de 步骤或 univariate_de）', 'DE figures need univariate DE first (de step or univariate_de)'],
    ['pathwayTitle', '通路与化学分类', 'Pathways and classes'],
    ['pathwayHint', 'Mummichog 式通路 / KEGG ORA / HMDB 化学分类 / MSEA', 'Mummichog-style pathways / KEGG ORA / HMDB class / MSEA'],
    ['pathwayEmpty', '还没有富集结果，选一个分析运行', 'No enrichment result yet — pick an analysis and run it'],
    ['pathwayBar', '富集条形图', 'Enrichment bar'],
    ['pathwayDot', '富集气泡图', 'Enrichment dot'],
    ['pathwayNeedDe', '富集需要先做差异分析', 'Enrichment needs a DE result first'],
    ['biomarkerTitle', '标志物与建模', 'Biomarkers'],
    ['biomarkerHint', 'RF 重要度 / LASSO 组合 / ROC / 组合评分 / 自助法验证 / 生存', 'RF importance / LASSO panel / ROC / score / bootstrap validation / survival'],
    ['biomarkerEmpty', '还没有标志物结果，选一个分析运行', 'No biomarker result yet — pick an analysis and run it'],
    ['biomarkerRoc', 'ROC 曲线', 'ROC curve'],
    ['biomarkerImportance', '随机森林重要度', 'Random-forest importance'],
    ['biomarkerLasso', 'LASSO 路径', 'LASSO path'],
    ['biomarkerNeedGroups', '标志物分析需要分组信息（样本表 group 列）', 'Biomarker analyses need a group column in the sample table'],
    ['annotationTitle', '代谢物鉴定', 'Metabolite annotation'],
    ['annotationHint', '加合物 / 分子式候选 / MS2 库匹配 / 分子网络', 'Adducts / formula candidates / MS2 library match / molecular network'],
    ['annotationEmpty', '还没有鉴定结果，选一个分析运行', 'No annotation result yet — pick an analysis and run it'],
    ['annotationLibrary', 'MS2 谱库路径', 'MS2 library path'],
    ['annotationLibraryPh', '/absolute/path/library.msp', '/absolute/path/library.msp'],
    ['annotationLibraryHint', '支持 msp/mgf；该路径作为 annotate 步骤与 MS2 分析（ms2_library_match / ms2_network）的 library 参数', 'msp/mgf supported; the path is passed as library to the annotate step and the MS2 analyses'],
    ['annotationLibraryApply', '带谱库运行鉴定步骤', 'Run annotation step with library'],
    ['annotationLibraryBusy', '鉴定运行中…', 'Running annotation…'],
    ['annotationMassAcc', '质量精度', 'Mass accuracy'],
    ['annotationNetwork', '分子网络', 'Molecular network'],
    ['annotationNeedMz', '数据集里没有 m/z 列，加合物与分子式推断不可用', 'No m/z column in this dataset — adduct/formula inference unavailable'],
    ['recordsTitle', '记录与导出', 'Records and export'],
    ['recordsHint', '历史步骤、可复现脚本与 0–100 自检', 'Step history, reproducible script and a 0–100 self-check'],
    ['recordsHistory', '步骤历史', 'Step history'],
    ['recordsNoHistory', '还没有运行记录', 'No runs yet'],
    ['recordsAnalyses', '分析运行', 'Analysis runs'],
    ['recordsAudit', '确定性自检', 'Deterministic self-check'],
    ['recordsRunAudit', '运行自检', 'Run check'],
    ['recordsAuditScore', '自检得分', 'Score'],
    ['recordsAuditEmpty', '点「运行自检」检查流程完整性、缺失值与质控情况', 'Run the check to inspect pipeline completeness, missingness and QC'],
    ['recordsExport', '导出', 'Export'],
    ['recordsAnalysis', '分析目录', 'Analyses'],
    ['recordsQuickExports', '快速导出', 'Quick export'],
    ['recordsExporting', '导出中…', 'Exporting…'],
    ['auditHint', '得分不是模型评分，只是流程完整性检查；有问题按提示补齐', 'The score is not a model score; it only checks pipeline completeness'],
    ['auditOk', '通过', 'Pass'],
    ['auditWarn', '注意', 'Warn'],
    ['auditFail', '异常', 'Fail'],
    ['exportExprCsv', '特征表', 'Feature table'],
    ['exportDeCsv', '差异表', 'DE table'],
    ['exportVipCsv', 'VIP 表', 'VIP table'],
    ['exportEnrichCsv', '富集表', 'Enrichment table'],
    ['exportBiomarkerCsv', '标志物表', 'Biomarker table'],
    ['exportSampleCsv', '样本表', 'Sample table'],
    ['exportSummary', '汇总 JSON', 'Summary JSON'],
    ['exportProtocol', '分析协议', 'Analysis protocol'],
    ['exportMethods', 'Methods', 'Methods'],
    ['exportReport', '报告', 'Report'],
    ['exportScript', '可复现脚本', 'Reproducible script'],
    ['logTitle', '活动日志', 'Activity log'],
    ['logRefresh', '刷新日志', 'Refresh log'],
    ['logEmpty', '还没有活动记录', 'No activity yet'],
    ['at', '时间', 'Time'],
    ['actor', '来源', 'Actor'],
    ['action', '动作', 'Action'],
    ['duration', '耗时', 'Duration'],
    ['resultColumn', '结果', 'Result'],
    ['activityAgent', 'Agent', 'Agent'],
    ['activityUser', '用户', 'User'],
    ['activityDataset', '数据集', 'Dataset'],
    ['agentTitle', 'Agent', 'Agent'],
    ['agentPanel', 'Agent', 'Agent'],
    ['agentHint', '也可以在对话里直接让 Agent 操作这个工作台。', 'You can also drive this workbench from the chat with the Agent.'],
    ['agentStatus', '状态', 'Status'],
    ['agentNoDataset', '载入数据集后这里显示当前状态', 'Load a dataset to see its status here'],
    ['agentSuggestions', '建议下一步', 'Next steps'],
    ['agentActivity', '活动', 'Activity'],
    ['agentNoActivity', '暂无活动', 'No activity yet'],
    ['agentCopy', '复制指令', 'Copy prompt'],
    ['agentShapeHint', '样本 × 特征矩阵；缺失值统一按 NaN 处理', 'Samples x features matrix; missing values are handled as NaN'],
    ['suggestionFallback', '继续分析', 'Continue analysis'],
    ['geoTitle', 'GEO 检索', 'GEO search'],
    ['geoSubtitle', '用 NCBI E-utilities 检索代谢组学数据集；中文关键词会自动翻成英文', 'Search metabolomics datasets via NCBI E-utilities; Chinese terms are translated automatically'],
    ['geoTermPh', '如：血清 代谢组学 LC-MS', 'e.g. serum metabolomics LC-MS'],
    ['geoSearchBtn', '检索', 'Search'],
    ['geoSearching', '检索中…', 'Searching…'],
    ['geoHint', '中文可直接搜（内置词典翻译）；结果少时会自动放宽关键词。', 'Chinese queries work (built-in dictionary); keywords are relaxed automatically when few results.'],
    ['geoSpecies', '物种', 'Species'],
    ['geoAll', '全部', 'All'],
    ['geoHuman', '人', 'Human'],
    ['geoMouse', '小鼠', 'Mouse'],
    ['geoStudyType', '研究类型', 'Study type'],
    ['geoStudyAll', '全部', 'All'],
    ['geoStudyArray', '芯片', 'Array'],
    ['geoStudySeq', '测序', 'Sequencing'],
    ['geoMinSamples', '最少样本', 'Min samples'],
    ['geoHasClinical', '只看有人体临床', 'Has clinical data'],
    ['geoMethod', '检测方法', 'Method'],
    ['geoMethodAll', '全部方法', 'All methods'],
    ['geoResults', '检索结果', 'Results'],
    ['geoTotal', '共 {v} 条', '{v} hits'],
    ['geoNoResults', '没有结果，换个关键词或放宽筛选', 'No hits; try another keyword or relax the filters'],
    ['geoLoadingDetail', '读取详情中…', 'Loading details…'],
    ['geoErrorSearch', '检索失败', 'Search failed'],
    ['geoErrorDetail', '详情读取失败', 'Failed to load details'],
    ['geoDetailTitle', '数据集详情', 'Dataset details'],
    ['geoDetailBtn', '点左侧结果查看详情', 'Pick a result on the left to see details'],
    ['geoSummary', '摘要', 'Summary'],
    ['geoDesign', '实验设计', 'Design'],
    ['geoFiles', '文件列表', 'Files'],
    ['geoNoFiles', '没有可下载文件', 'No downloadable file'],
    ['geoSelectAll', '全选', 'Select all'],
    ['geoClear', '清空', 'Clear'],
    ['geoDownloadDir', '下载目录', 'Download dir'],
    ['geoDownloadDirHint', '绝对路径；默认 ~/.dsh/metabolomics/GEO', 'Absolute path; default ~/.dsh/metabolomics/GEO'],
    ['geoPickDir', '选择目录', 'Choose dir'],
    ['geoPicking', '选择中…', 'Choosing…'],
    ['geoPickFailed', '目录选择接口不可用，请手动填写绝对路径', 'Directory picker unavailable; type the absolute path manually'],
    ['geoDownloadBtn', '下载', 'Download'],
    ['geoDownloading', '下载中…', 'Downloading…'],
    ['geoJobDone', '下载完成', 'Download finished'],
    ['geoJobFailed', '下载失败', 'Download failed'],
    ['geoImportBtn', '导入并开始清洗', 'Import and start QC'],
    ['geoImporting', '导入中…', 'Importing…'],
    ['geoImportDone', '已导入并完成质控', 'Imported and QC finished'],
    ['geoOpenAfter', '下载并导入', 'Download & import'],
    ['geoNeedFile', '请至少勾选一个文件', 'Select at least one file'],
    ['geoRelease', '发布时间', 'Released'],
    ['geoPubmed', 'PubMed', 'PubMed'],
    ['geoMatrix', '有矩阵', 'matrix'],
    ['geoRelevance', '相关度', 'relevance'],
    ['geoInvalidDir', '下载目录必须是绝对路径', 'Download dir must be an absolute path'],
    ['mlTitle', 'MetaboLights 检索', 'MetaboLights search'],
    ['mlSubtitle', '用 EBI MetaboLights REST 检索代谢组学数据集（第二数据源）', 'Search metabolomics studies via the EBI MetaboLights REST API (second data source)'],
    ['mlQueryPh', '如：serum metabolomics', 'e.g. serum metabolomics'],
    ['mlSearchBtn', '检索', 'Search'],
    ['mlSearching', '检索中…', 'Searching…'],
    ['mlHint', '中文关键词可能需要翻成英文；端点不可用时会给出可读原因。', 'English keywords work best; a readable reason is shown when the endpoint is unavailable.'],
    ['mlAccession', '编号', 'Accession'],
    ['mlSpeciesLabel', '物种', 'Species'],
    ['mlAssays', '实验数', 'Assays'],
    ['mlFiles', '文件数', 'Files'],
    ['mlSubmission', '提交时间', 'Submitted'],
    ['mlContacts', '联系人', 'Contacts'],
    ['mlPublications', '文献', 'Publications'],
    ['mlDetailTitle', '研究详情', 'Study details'],
    ['mlDetailBtn', '点左侧结果查看详情', 'Pick a result on the left to see details'],
    ['mlNoResults', '没有结果，换个关键词', 'No hits; try another keyword'],
    ['mlUnavailable', '这台机器上的 MetaboLights 接口不可用（见引擎信息）', 'The MetaboLights endpoint is unavailable on this machine (see engine info)'],
    ['mlDownloadDir', '下载目录', 'Download dir'],
    ['mlDownloadBtn', '下载', 'Download'],
    ['jobTimeout', '任务超时，引擎可能仍在计算，请稍后在记录里查看', 'Job timed out; the engine may still be computing — check Records later'],
    ['importTitle', '导入数据', 'Import data'],
    ['importSubtitle', '支持 csv/tsv/xlsx/gz、zip/tar、mzTab、ISA-Tab、msp/mgf、mzML（TIC）', 'csv/tsv/xlsx/gz, zip/tar, mzTab, ISA-Tab, msp/mgf, mzML (TIC)'],
    ['importSourceLocal', '本机路径', 'Local path'],
    ['importSourceUrl', 'URL', 'URL'],
    ['importChooseFile', '选择本地文件（浏览器上传）', 'Choose a local file (browser upload)'],
    ['importUploadHint', '上传适合小表；大文件请用本机路径，避免浏览器拷贝', 'Upload suits small tables; use a local path for big files'],
    ['importFormat', '数据格式', 'Data format'],
    ['importFeatureCol', '特征名列', 'Feature column'],
    ['importFeatureColPh', '特征 id 所在列（可选）', 'Column holding feature ids (optional)'],
    ['importSheet', 'Excel sheet', 'Excel sheet'],
    ['importMember', '压缩包内路径', 'Path inside archive'],
    ['importObject', 'RData 对象名', 'RData object'],
    ['importTranspose', '矩阵转置（特征×样本 → 样本×特征）', 'Transpose (features x samples to samples x features)'],
    ['importSampleSheet', '样本信息表（可选）', 'Sample sheet (optional)'],
    ['importClinical', '临床表（可选）', 'Clinical table (optional)'],
    ['importZeroAsMissing', '0 视为缺失', 'Treat 0 as missing'],
    ['importAdvanced', '高级选项', 'Advanced'],
    ['importDelimiter', '分隔符', 'Delimiter'],
    ['importDelimiterPh', '自动（逗号 / 制表符）', 'auto (comma / tab)'],
    ['importHint', '常见错误：矩阵方向判错 → 勾选转置；压缩包多候选 → 会列出候选文件让你点选。', 'Common errors: wrong matrix orientation (tick transpose); archives with several candidates (pick one).'],
    ['importError', '导入失败', 'Import failed'],
    ['importNoPath', '请填写路径或 URL', 'Fill in a path or URL'],
    ['importCandidates', '发现多个候选，点选一个再试', 'Several candidates found — pick one and retry'],
    ['importInspect', '只探测格式', 'Inspect only'],
    ['importInspectBusy', '探测中…', 'Inspecting…'],
    ['importInspectTitle', '格式探测', 'Format probe'],
    ['importStart', '导入', 'Import'],
    ['importBusy', '导入中…', 'Importing…'],
    ['importPathPh', '/absolute/path/feature_table.csv 或 /path/study.zip', '/absolute/path/feature_table.csv or /path/study.zip'],
    ['importUrlPh', 'https://…（直链）', 'https://… (direct link)'],
    ['demo', '载入示例数据', 'Load demo'],
    ['demoLoad', '载入示例', 'Load demo'],
    ['searchGeo', 'GEO 检索', 'GEO search'],
    ['searchMl', 'MetaboLights 检索', 'MetaboLights'],
    ['importData', '导入数据', 'Import'],
    ['copyInstruction', '复制指令', 'Copy prompt'],
    ['noSelection', '不选择', 'None'],
    ['tableNoteEmpty', '—', '—'],
    ['empty', '—', '—'],
    ['confirm', '确定', 'OK'],
    ['ok', '好', 'OK'],
    ['apply', '应用', 'Apply'],
    ['stepParamsTitle', '步骤参数', 'Step params'],
    ['stepParamsHint', '运行时会带上这里的参数', 'These params are sent when you run the step'],
    ['analysisParamsTitle', '分析参数', 'Analysis params'],
    ['currentDataset', '当前数据集', 'Current dataset'],
    ['refreshPreview', '刷新预览', 'Refresh preview'],
    ['rowsLabel', '行', 'rows'],
    ['downloadDirMissing', '请填写下载目录（绝对路径）', 'Fill in the download dir (absolute path)'],
  ]

  var DICT_ZH = {}
  var DICT_EN = {}
  for (var wordIndex = 0; wordIndex < WORDS.length; wordIndex++) {
    DICT_ZH[WORDS[wordIndex][0]] = WORDS[wordIndex][1]
    DICT_EN[WORDS[wordIndex][0]] = WORDS[wordIndex][2]
  }
  // ==========================================================================
  // 纯函数层：格式化 / 取色 / 二维散点→像素 / payload / 参数 / 归一化 / API / 轮询
  // 不碰 DOM、不 require；可在 node 里直接单测（见文件尾 __MB_TEST__）。
  // ==========================================================================

  function isNil(value) {
    return value === null || value === undefined
  }

  function clamp(value, min, max) {
    var number = Number(value)
    if (!isFinite(number)) return min
    if (number < min) return min
    if (number > max) return max
    return number
  }

  function asNumber(value) {
    if (isNil(value) || value === '') return null
    var number = Number(value)
    return isFinite(number) ? number : null
  }

  function plainText(value, max) {
    var text = isNil(value) ? '' : String(value).replace(/\s+/g, ' ').trim()
    if (typeof max === 'number' && max > 0 && text.length > max) return text.slice(0, max)
    return text
  }

  function clampInt(value, min, max, fallback) {
    var number = asNumber(value)
    if (number === null) return fallback
    return Math.round(clamp(number, min, max))
  }

  function trimFixed(number, digits) {
    return Number(number).toFixed(clamp(digits, 0, 6))
  }

  /** 1234 → 1,234；25000 → 25K；1500000 → 1.5M；空值 → —。 */
  function fmtNumber(value, digits) {
    var number = asNumber(value)
    if (number === null) return '—'
    var fixed = isFinite(Number(digits)) ? clamp(digits, 0, 6) : 2
    var abs = Math.abs(number)
    var sign = number < 0 ? '-' : ''
    if (abs >= 1e9) return sign + trimFixed(abs / 1e9, fixed > 0 ? fixed : 1).replace(/\.0+$/, '') + 'B'
    if (abs >= 1e6) return sign + trimFixed(abs / 1e6, fixed > 0 ? fixed : 1).replace(/\.0+$/, '') + 'M'
    if (abs >= 1e4) return sign + trimFixed(abs / 1e3, fixed > 0 ? fixed : 1).replace(/\.0+$/, '') + 'K'
    if (abs >= 1000) {
      var parts = trimFixed(abs, fixed).split('.')
      var head = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',')
      return sign + head + (parts[1] && parts[1].length ? '.' + parts[1] : '')
    }
    var fixedText = trimFixed(abs, fixed)
    return sign + (fixedText.indexOf('.') >= 0 ? fixedText.replace(/0+$/, '').replace(/\.$/, '') : fixedText)
  }

  /** 0.5 → 50%；37.5 → 37.5%；空值 → —。 */
  function fmtPct(value, digits) {
    var number = asNumber(value)
    if (number === null) return '—'
    var fixed = isFinite(Number(digits)) ? clamp(digits, 0, 4) : 1
    var percent = Math.abs(number) <= 1 ? number * 100 : number
    var text = trimFixed(percent, fixed)
    if (text.indexOf('.') >= 0) text = text.replace(/0+$/, '').replace(/\.$/, '')
    return text + '%'
  }

  /** P 值：0.05 → 0.050；5e-4 → 5.00e-4；0 → 0。 */
  function fmtP(value) {
    var number = asNumber(value)
    if (number === null) return '—'
    if (number === 0) return '0'
    if (number < 1e-3) {
      var text = Number(number).toExponential(2)
      return text.replace('e', 'e')
    }
    if (number < 0.01) return Number(number).toFixed(4)
    return Number(number).toFixed(3)
  }

  /** 字节 → 1024 进制短单位。 */
  function formatBytes(bytes) {
    var number = asNumber(bytes)
    if (number === null || number < 0) return '—'
    var units = ['B', 'KB', 'MB', 'GB', 'TB']
    var exponent = Math.min(Math.floor(Math.log(number) / Math.log(1024)), units.length - 1)
    if (number <= 0) return '0 B'
    var value = number / Math.pow(1024, exponent)
    return (Math.round(value * 10) / 10).toFixed(1).replace(/\.0$/, '') + ' ' + units[exponent]
  }

  function formatDuration(ms) {
    var number = asNumber(ms)
    if (number === null || number < 0) return '—'
    if (number < 1000) return Math.round(number) + ' ms'
    if (number < 60000) return (Math.round(number / 100) / 10).toFixed(1).replace(/\.0$/, '') + ' s'
    var totalSec = Math.round(number / 1000)
    var minutes = Math.floor(totalSec / 60)
    var seconds = totalSec % 60
    if (minutes < 60) return minutes + 'm ' + (seconds < 10 ? '0' + seconds : String(seconds)) + 's'
    var hours = Math.floor(minutes / 60)
    return hours + 'h ' + (minutes % 60) + 'm'
  }

  function formatClock(value) {
    if (isNil(value) || value === '') return '—'
    var date = value instanceof Date ? value : new Date(value)
    if (isNaN(date.getTime())) {
      var text = String(value)
      return text.length > 16 ? text.slice(5, 16).replace('T', ' ') : text
    }
    function pad(number) { return number < 10 ? '0' + number : String(number) }
    return pad(date.getMonth() + 1) + '-' + pad(date.getDate()) + ' ' + pad(date.getHours()) + ':' + pad(date.getMinutes())
  }

  function formatError(error) {
    if (isNil(error)) return ''
    if (typeof error === 'string') return error
    if (error && typeof error.message === 'string' && error.message) return error.message
    return String(error)
  }

  function hexToRgb(hex) {
    var text = String(hex || '').replace('#', '')
    if (text.length === 3) text = text.charAt(0) + text.charAt(0) + text.charAt(1) + text.charAt(1) + text.charAt(2) + text.charAt(2)
    if (!/^[0-9a-fA-F]{6}$/.test(text)) return { r: 128, g: 128, b: 128 }
    return {
      r: parseInt(text.slice(0, 2), 16),
      g: parseInt(text.slice(2, 4), 16),
      b: parseInt(text.slice(4, 6), 16),
    }
  }

  /** 同名同色：数字当调色板下标，字符串取稳定哈希。 */
  function colorForGroup(value) {
    var index = null
    if (typeof value === 'number' && isFinite(value)) {
      index = Math.abs(Math.round(value)) % PALETTE.length
      return PALETTE[index]
    }
    var text = isNil(value) ? '' : String(value)
    if (!text) return PALETTE[PALETTE.length - 1]
    var hash = 0
    for (var cursor = 0; cursor < text.length; cursor++) {
      hash = ((hash << 5) - hash + text.charCodeAt(cursor)) | 0
    }
    index = Math.abs(hash) % PALETTE.length
    return PALETTE[index]
  }

  /** 连续值 → 色带颜色（低→高）；min/max 相同或非法时落到中位色。 */
  function continuousColor(value, min, max) {
    var number = asNumber(value)
    if (number === null) return PALETTE[PALETTE.length - 1]
    var low = asNumber(min)
    var high = asNumber(max)
    if (low === null || high === null || !(high > low)) return CONTINUOUS_STOPS[Math.floor(CONTINUOUS_STOPS.length / 2)]
    var ratio = clamp((number - low) / (high - low), 0, 1)
    var span = CONTINUOUS_STOPS.length - 1
    var position = ratio * span
    var leftIndex = Math.floor(position)
    var rightIndex = Math.min(leftIndex + 1, span)
    var mix = position - leftIndex
    var left = hexToRgb(CONTINUOUS_STOPS[leftIndex])
    var right = hexToRgb(CONTINUOUS_STOPS[rightIndex])
    var channel = function (a, b) {
      var mixed = Math.round(a + (b - a) * mix)
      var hex = mixed.toString(16)
      return hex.length === 1 ? '0' + hex : hex
    }
    return '#' + channel(left.r, right.r) + channel(left.g, right.g) + channel(left.b, right.b)
  }

  function groupBy(list, keyFn) {
    var source = Array.isArray(list) ? list : []
    var pick = typeof keyFn === 'function'
      ? keyFn
      : function (item) { return item && typeof item === 'object' ? item[keyFn] : item }
    var out = {}
    for (var index = 0; index < source.length; index++) {
      var key = String(pick(source[index], index))
      if (!out[key]) out[key] = []
      out[key].push(source[index])
    }
    return out
  }

  // --------------------------------------------------------------------------
  // 二维散点 → 像素：把 (x, y) 等比映射进 width × height 的 SVG（质控小图用）。
  // 语义：
  //   - 默认按屏幕坐标（原点左上、y 向下增）线性映射，不翻转；
  //     需要数学坐标（y 向上增，例如「缺失率越高越靠上」）时传 options.flipY = true。
  //   - options.uniform !== false 时保持等比（默认），多余空间居中留白。
  //   - bbox 省略或非法时用点的包围盒。
  // 返回 [{ x, y, index }]，无效点被跳过（不会打断整批）。
  // --------------------------------------------------------------------------

  function normalizeExtent(bbox) {
    var raw = bbox && typeof bbox === 'object' ? bbox : null
    if (!raw) return null
    var xRange = Array.isArray(raw.x) ? raw.x : (Array.isArray(raw.x_range) ? raw.x_range : null)
    var yRange = Array.isArray(raw.y) ? raw.y : (Array.isArray(raw.y_range) ? raw.y_range : null)
    var x0 = xRange ? asNumber(xRange[0]) : null
    var x1 = xRange ? asNumber(xRange[1]) : null
    var y0 = yRange ? asNumber(yRange[0]) : null
    var y1 = yRange ? asNumber(yRange[1]) : null
    if (x0 === null || x1 === null || y0 === null || y1 === null) return null
    return {
      x: [Math.min(x0, x1), Math.max(x0, x1)],
      y: [Math.min(y0, y1), Math.max(y0, y1)],
    }
  }

  function pointXY(point) {
    if (!point) return null
    if (Array.isArray(point)) {
      var ax = asNumber(point[0])
      var ay = asNumber(point[1])
      if (ax === null || ay === null) return null
      return { x: ax, y: ay }
    }
    if (typeof point === 'object') {
      var x = asNumber(point.x !== undefined ? point.x : point[0])
      var y = asNumber(point.y !== undefined ? point.y : point[1])
      if (x === null || y === null) return null
      return { x: x, y: y }
    }
    return null
  }

  /** 点集包围盒 → { x: [min, max], y: [min, max] }；无有效点返回 null。 */
  function extentOf(points) {
    var source = Array.isArray(points) ? points : []
    var minX = null
    var maxX = null
    var minY = null
    var maxY = null
    for (var index = 0; index < source.length; index++) {
      var point = pointXY(source[index])
      if (!point) continue
      if (minX === null || point.x < minX) minX = point.x
      if (maxX === null || point.x > maxX) maxX = point.x
      if (minY === null || point.y < minY) minY = point.y
      if (maxY === null || point.y > maxY) maxY = point.y
    }
    if (minX === null) return null
    return { x: [minX, maxX], y: [minY, maxY] }
  }

  function miniPlotScreen(points, bbox, width, height, options) {
    var opts = options && typeof options === 'object' ? options : {}
    var boxWidth = asNumber(width)
    var boxHeight = asNumber(height)
    if (boxWidth === null || boxWidth <= 0) boxWidth = 640
    if (boxHeight === null || boxHeight <= 0) boxHeight = 480
    var padding = asNumber(opts.padding)
    if (padding === null || padding < 0) padding = 8
    var source = Array.isArray(points) ? points : []
    var valid = []
    for (var index = 0; index < source.length; index++) {
      var point = pointXY(source[index])
      if (point) valid.push({ x: point.x, y: point.y, index: index })
    }
    if (!valid.length) return []
    var extent = normalizeExtent(bbox) || extentOf(valid)
    var minX = extent.x[0]
    var maxX = extent.x[1]
    var minY = extent.y[0]
    var maxY = extent.y[1]
    var spanX = maxX - minX
    var spanY = maxY - minY
    if (!(spanX > 0)) spanX = 1
    if (!(spanY > 0)) spanY = 1
    var innerWidth = Math.max(boxWidth - padding * 2, 1)
    var innerHeight = Math.max(boxHeight - padding * 2, 1)
    var uniform = opts.uniform !== false
    var scaleX = innerWidth / spanX
    var scaleY = innerHeight / spanY
    var scale = uniform ? Math.min(scaleX, scaleY) : 1
    var drawWidth = uniform ? spanX * scale : innerWidth
    var drawHeight = uniform ? spanY * scale : innerHeight
    var offsetX = padding + (innerWidth - drawWidth) / 2
    var offsetY = padding + (innerHeight - drawHeight) / 2
    var result = []
    for (var cursor = 0; cursor < valid.length; cursor++) {
      var item = valid[cursor]
      var sx = offsetX + (item.x - minX) * (uniform ? scale : scaleX)
      var sy = offsetY + (item.y - minY) * (uniform ? scale : scaleY)
      if (opts.flipY === true) sy = boxHeight - sy
      result.push({ x: Math.round(sx * 100) / 100, y: Math.round(sy * 100) / 100, index: item.index })
    }
    return result
  }

  // --------------------------------------------------------------------------
  // 数据源搜索 payload：只带非空/非默认的可选字段（服务端自己补默认值）。
  //   GEO：term/species/method/min_samples/has_clinical/limit/page/sort（DESIGN §7.1）
  //   MetaboLights：query/limit/page（DESIGN §7.4）
  // --------------------------------------------------------------------------

  function buildSearchPayload(filters) {
    var raw = filters && typeof filters === 'object' ? filters : {}
    var payload = {
      term: plainText(raw.term, 300),
      limit: clampInt(raw.limit, 1, 50, 20),
      page: clampInt(raw.page, 0, 1000, 0),
      sort: 'relevance',
    }
    var sort = plainText(raw.sort)
    if (sort === 'date' || sort === 'samples' || sort === 'relevance') payload.sort = sort
    var species = plainText(raw.species)
    if (species && species !== 'all') payload.species = species
    var method = plainText(raw.method)
    if (method && method !== 'all') payload.method = method
    var minSamples = asNumber(raw.min_samples)
    if (minSamples !== null && minSamples > 0) payload.min_samples = Math.round(minSamples)
    if (raw.has_clinical === true) payload.has_clinical = true
    return payload
  }

  /** MetaboLights 检索 payload（DESIGN §7.4：query / limit / page / assay_type）。 */
  function buildMlSearchPayload(options) {
    var raw = options && typeof options === 'object' ? options : {}
    var payload = {
      query: plainText(raw.query !== undefined ? raw.query : raw.term, 300),
      limit: clampInt(raw.limit, 1, 50, 20),
      page: clampInt(raw.page, 0, 1000, 0),
    }
    var assayType = plainText(raw.assay_type)
    if (assayType) payload.assay_type = assayType
    return payload
  }

  /** 导入参数：空值一律剔除；data_format=auto 不发；布尔只在 true 时发。 */
  function buildImportPayload(options) {
    var raw = options && typeof options === 'object' ? options : {}
    var out = {}
    var path = plainText(raw.path)
    if (path) out.path = path
    var url = plainText(raw.url)
    if (url) out.url = url
    var name = plainText(raw.name)
    if (name) out.name = name
    var format = plainText(raw.data_format)
    if (format && format !== 'auto') out.data_format = format
    var member = plainText(raw.member)
    if (member) out.member = member
    var object = plainText(raw.object)
    if (object) out.object = object
    var sheet = plainText(raw.sheet)
    if (sheet) out.sheet = sheet
    var delimiter = plainText(raw.delimiter)
    if (delimiter) out.delimiter = delimiter
    var sampleSheet = plainText(raw.sample_sheet)
    if (sampleSheet) out.sample_sheet = sampleSheet
    var clinical = plainText(raw.clinical)
    if (clinical) out.clinical = clinical
    var featureCol = plainText(raw.feature_col)
    if (featureCol) out.feature_col = featureCol
    if (raw.transpose === true) out.transpose = true
    if (raw.zero_as_missing === true) out.zero_as_missing = true
    return out
  }
  /** 建议 → 可直接粘给 Agent 的中文指令。 */
  function suggestionInstruction(suggestion) {
    var raw = suggestion && typeof suggestion === 'object' ? suggestion : {}
    var title = plainText(raw.title, 120) || '继续分析'
    var detail = plainText(raw.detail, 400)
    var step = plainText(raw.step, 40)
    var meta = step && STEP_BY_ID[step] ? STEP_BY_ID[step] : null
    var text = '请在工作台执行「' + title + '」'
    if (meta) text += '（步骤：' + meta.title + '，step=' + step + '）'
    else if (step) text += '（step=' + step + '）'
    if (detail) text += '：' + detail
    return text + '。'
  }

  // --------------------------------------------------------------------------
  // 步骤参数：默认值 / 收集（按 descriptor.type 收敛）/ 依赖判定
  // --------------------------------------------------------------------------

  function parseTokenList(value) {
    if (Array.isArray(value)) {
      var out = []
      for (var index = 0; index < value.length; index++) {
        var text = plainText(value[index], 200)
        if (text) out.push(text)
      }
      return out
    }
    var raw = plainText(value, 4000)
    if (!raw) return []
    return raw.split(/[,，;；\s|]+/).map(function (item) { return item.trim() }).filter(Boolean)
  }

  function cloneDefault(value) {
    if (Array.isArray(value)) return value.slice()
    if (value && typeof value === 'object') {
      var out = {}
      var keys = Object.keys(value)
      for (var index = 0; index < keys.length; index++) out[keys[index]] = value[keys[index]]
      return out
    }
    return value
  }

  /** 步骤默认参数（数组/对象复制，避免多次运行共享同一引用）。 */
  function defaultParams(step) {
    var meta = typeof step === 'string' ? STEP_BY_ID[step] : step
    var out = {}
    if (!meta || !Array.isArray(meta.params)) return out
    for (var index = 0; index < meta.params.length; index++) {
      var descriptor = meta.params[index]
      if (!descriptor || !descriptor.key) continue
      out[descriptor.key] = cloneDefault(descriptor.default)
    }
    return out
  }

  function parseJsonParam(text) {
    var raw = plainText(text, 8000)
    if (!raw) return null
    try {
      var parsed = JSON.parse(raw)
      if (parsed && typeof parsed === 'object') return parsed
      return null
    } catch (ignored) {
      return null
    }
  }

  /** 按 descriptor 类型收敛表单值；空值统一成 null（数组类型成 []）。 */
  function collectStepParams(step, values) {
    var meta = typeof step === 'string' ? STEP_BY_ID[step] : step
    var merged = defaultParams(meta)
    var source = values && typeof values === 'object' ? values : {}
    var out = {}
    if (!meta || !Array.isArray(meta.params)) return out
    for (var index = 0; index < meta.params.length; index++) {
      var descriptor = meta.params[index]
      if (!descriptor || !descriptor.key) continue
      var value = source[descriptor.key]
      if (value === undefined) value = merged[descriptor.key]
      var type = descriptor.type || 'text'
      if (type === 'number') {
        out[descriptor.key] = asNumber(value)
        continue
      }
      if (type === 'boolean') {
        out[descriptor.key] = value === true
        continue
      }
      if (type === 'multiselect' || type === 'genes' || type === 'columns' || type === 'array') {
        out[descriptor.key] = parseTokenList(value)
        continue
      }
      if (type === 'json') {
        out[descriptor.key] = parseJsonParam(value)
        continue
      }
      if (type === 'select' || type === 'column' || type === 'file' || type === 'text' || type === 'string') {
        var text = plainText(value, 2000)
        out[descriptor.key] = text || null
        continue
      }
      out[descriptor.key] = value === undefined ? null : value
    }
    return out
  }

  function formatParamValue(descriptor, value) {
    var type = descriptor && descriptor.type ? descriptor.type : 'text'
    if (type === 'multiselect') {
      var labels = {}
      var options = descriptor && Array.isArray(descriptor.options) ? descriptor.options : []
      for (var index = 0; index < options.length; index++) labels[String(options[index].value)] = options[index].label
      return parseTokenList(value).map(function (item) { return labels[item] || item }).join(' / ')
    }
    if (Array.isArray(value)) return value.join(', ')
    if (isNil(value) || value === '') return '—'
    if (typeof value === 'boolean') return value ? '是' : '否'
    return String(value)
  }

  function doneStepSet(dataset) {
    var set = {}
    var steps = dataset && Array.isArray(dataset.steps) ? dataset.steps : []
    for (var index = 0; index < steps.length; index++) set[String(steps[index])] = true
    return set
  }

  /** 前置是否全部完成（requires 与 DESIGN.md §3 一致）。 */
  function isStepReady(step, dataset) {
    var meta = typeof step === 'string' ? STEP_BY_ID[step] : step
    if (!meta || !dataset) return false
    var done = doneStepSet(dataset)
    var requires = Array.isArray(meta.requires) ? meta.requires : []
    for (var index = 0; index < requires.length; index++) {
      if (!done[requires[index]]) return false
    }
    return true
  }

  /** 左栏状态：running > failed > done > ready > blocked。 */
  function stepStatusOf(step, dataset, busyStepId, failedStepId) {
    var meta = typeof step === 'string' ? STEP_BY_ID[step] : step
    if (!meta) return 'blocked'
    if (busyStepId === meta.id) return 'running'
    if (failedStepId === meta.id) return 'failed'
    if (dataset && doneStepSet(dataset)[meta.id]) return 'done'
    if (dataset && isStepReady(meta, dataset)) return 'ready'
    return 'blocked'
  }

  function stepStatusKey(status) {
    if (status === 'done') return 'stepStatusDone'
    if (status === 'running') return 'stepStatusRunning'
    if (status === 'ready') return 'stepStatusReady'
    if (status === 'failed') return 'stepStatusFailed'
    return 'stepStatusBlocked'
  }

  // --------------------------------------------------------------------------
  // 图 URL / 结果归一化（DESIGN.md §2.5 result 信封）
  // --------------------------------------------------------------------------

  function figureSrc(raw) {
    if (typeof raw === 'string' && raw) {
      if (raw.indexOf('data:') === 0 || raw.charAt(0) === '/') return raw
      var nameOnly = raw.split('/').pop()
      if (/^[A-Za-z0-9._-]+\.png$/.test(nameOnly)) return API + '/figure?id=' + encodeURIComponent(nameOnly)
      return ''
    }
    if (!raw || typeof raw !== 'object') return ''
    if (typeof raw.url === 'string' && raw.url) return raw.url
    if (typeof raw.file === 'string' && raw.file) {
      var name = raw.file.split('/').pop()
      if (/^[A-Za-z0-9._-]+\.png$/.test(name)) return API + '/figure?id=' + encodeURIComponent(name)
    }
    return ''
  }

  function normalizeFigure(payload) {
    var raw = payload && typeof payload === 'object'
      ? (payload.figure && typeof payload.figure === 'object' ? payload.figure : payload)
      : {}
    var file = typeof raw.file === 'string' ? raw.file : ''
    var url = figureSrc({ url: typeof raw.url === 'string' ? raw.url : '', file: file })
    if (!url) return null
    return {
      id: isNil(raw.id) ? '' : String(raw.id),
      title: plainText(raw.title || raw.kind || '', 160),
      url: url,
      file: file,
      width: asNumber(raw.width) || 0,
      height: asNumber(raw.height) || 0,
      kind: typeof raw.kind === 'string' ? raw.kind : '',
      bytes: asNumber(raw.bytes),
    }
  }

  function normalizeTable(table, index) {
    var raw = table && typeof table === 'object' ? table : {}
    var columns = []
    if (Array.isArray(raw.columns)) {
      for (var columnIndex = 0; columnIndex < raw.columns.length; columnIndex++) {
        var column = raw.columns[columnIndex] && typeof raw.columns[columnIndex] === 'object' ? raw.columns[columnIndex] : {}
        var key = plainText(column.key || column.name || ('' + columnIndex))
        if (!key) continue
        columns.push({
          key: key,
          label: plainText(column.label || key),
          type: column.type === 'number' || column.numeric === true ? 'number' : 'string',
        })
      }
    }
    var rows = Array.isArray(raw.rows) ? raw.rows : []
    if (!columns.length && rows.length && rows[0] && typeof rows[0] === 'object') {
      var firstKeys = Object.keys(rows[0])
      for (var keyIndex = 0; keyIndex < firstKeys.length; keyIndex++) {
        columns.push({
          key: firstKeys[keyIndex],
          label: firstKeys[keyIndex],
          type: typeof rows[0][firstKeys[keyIndex]] === 'number' ? 'number' : 'string',
        })
      }
    }
    return {
      id: plainText(raw.id || ('table-' + index)),
      title: plainText(raw.title || ''),
      columns: columns,
      rows: rows,
      note: plainText(raw.note || '', 300),
      truncated: raw.truncated === true,
    }
  }

  function normalizeStringList(value) {
    var list = Array.isArray(value) ? value : (isNil(value) || value === '' ? [] : [value])
    var out = []
    for (var index = 0; index < list.length; index++) {
      var text = plainText(list[index], 500)
      if (text) out.push(text)
    }
    return out
  }

  /** 步骤/分析/导出共用的结果信封归一化。 */
  function normalizeResult(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    if (raw.result && typeof raw.result === 'object' && !raw.summary) {
      raw = Object.assign({}, raw.result, {
        analysis_id: raw.analysis_id || raw.result.analysis_id,
        ms: raw.ms || raw.result.ms,
        dataset: raw.dataset || raw.result.dataset,
      })
    }
    var summary = raw.summary && typeof raw.summary === 'object' ? raw.summary : {}
    var bullets = []
    if (Array.isArray(summary.bullets)) {
      for (var index = 0; index < summary.bullets.length; index++) {
        var bullet = plainText(summary.bullets[index], 500)
        if (bullet) bullets.push(bullet)
      }
    }
    var metrics = {}
    if (summary.metrics && typeof summary.metrics === 'object') {
      var metricKeys = Object.keys(summary.metrics)
      for (var metricIndex = 0; metricIndex < metricKeys.length; metricIndex++) {
        metrics[metricKeys[metricIndex]] = summary.metrics[metricKeys[metricIndex]]
      }
    }
    var tables = []
    if (Array.isArray(raw.tables)) {
      for (var tableIndex = 0; tableIndex < raw.tables.length; tableIndex++) {
        tables.push(normalizeTable(raw.tables[tableIndex], tableIndex))
      }
    }
    var figures = []
    if (Array.isArray(raw.figures)) {
      for (var figureIndex = 0; figureIndex < raw.figures.length; figureIndex++) {
        var figure = normalizeFigure(raw.figures[figureIndex])
        if (figure) figures.push(figure)
      }
    }
    var dataOut = []
    if (Array.isArray(raw.data_out)) {
      for (var outIndex = 0; outIndex < raw.data_out.length; outIndex++) {
        var item = raw.data_out[outIndex] && typeof raw.data_out[outIndex] === 'object' ? raw.data_out[outIndex] : {}
        if (!item.path) continue
        dataOut.push({ kind: plainText(item.kind || ''), path: String(item.path), label: plainText(item.label || item.kind || '') })
      }
    }
    return {
      title: plainText(summary.title || raw.title || '', 200),
      summary: { title: plainText(summary.title || '', 200), bullets: bullets, metrics: metrics },
      bullets: bullets,
      metrics: metrics,
      tables: tables,
      figures: figures,
      data_out: dataOut,
      warnings: normalizeStringList(raw.warnings),
      notes: normalizeStringList(raw.notes),
      analysis_id: plainText(raw.analysis_id || ''),
      step: plainText(raw.step || ''),
      ms: asNumber(raw.ms),
      raw: raw,
    }
  }

  /** dataset 摘要归一化：服务端缺字段（老版本/引擎忙）时也能渲染（DESIGN §2.4）。 */
  function normalizeDataset(payload) {
    var raw = payload && typeof payload === 'object'
      ? (payload.dataset && typeof payload.dataset === 'object' ? payload.dataset : payload)
      : {}
    var id = plainText(raw.id || raw.dataset || raw.dataset_id || '')
    var steps = Array.isArray(raw.steps) ? raw.steps.map(function (item) { return plainText(item) }).filter(Boolean) : []
    var sampleColumns = []
    if (Array.isArray(raw.sample_columns)) {
      for (var columnIndex = 0; columnIndex < raw.sample_columns.length; columnIndex++) {
        var column = raw.sample_columns[columnIndex] && typeof raw.sample_columns[columnIndex] === 'object' ? raw.sample_columns[columnIndex] : {}
        var key = plainText(column.key || column.name || '')
        if (!key) continue
        sampleColumns.push({
          key: key,
          kind: plainText(column.kind || 'categorical'),
          n_unique: asNumber(column.n_unique),
        })
      }
    }
    var metaRaw = raw.feature_meta && typeof raw.feature_meta === 'object' ? raw.feature_meta : {}
    var history = []
    if (Array.isArray(raw.history)) {
      var entries = raw.history.slice(-MAX_HISTORY_ROWS)
      for (var historyIndex = 0; historyIndex < entries.length; historyIndex++) {
        var entry = entries[historyIndex] && typeof entries[historyIndex] === 'object' ? entries[historyIndex] : {}
        history.push({
          at: plainText(entry.at || entry.time || ''),
          step: plainText(entry.step || entry.action || ''),
          ok: entry.ok !== false && entry.status !== 'failed',
          ms: asNumber(entry.ms),
          summary: plainText(entry.summary || entry.message || '', 240),
          actor: plainText(entry.actor || ''),
        })
      }
    }
    var runs = []
    if (Array.isArray(raw.analyses_run)) {
      for (var runIndex = 0; runIndex < raw.analyses_run.length; runIndex++) {
        var run = raw.analyses_run[runIndex] && typeof raw.analyses_run[runIndex] === 'object' ? raw.analyses_run[runIndex] : {}
        runs.push({ at: plainText(run.at || ''), id: plainText(run.id || ''), ok: run.ok !== false, ms: asNumber(run.ms) })
      }
    }
    return {
      id: id,
      name: plainText(raw.name || id || ''),
      path: typeof raw.path === 'string' ? raw.path : '',
      source_format: plainText(raw.source_format || raw.format || ''),
      n_samples: asNumber(raw.n_samples),
      n_features: asNumber(raw.n_features),
      species: plainText(raw.species || ''),
      polarity: plainText(raw.polarity || ''),
      feature_meta: {
        has_mz: metaRaw.has_mz === true,
        has_rt: metaRaw.has_rt === true,
        n_annotated: asNumber(metaRaw.n_annotated !== undefined ? metaRaw.n_annotated : metaRaw.annotated),
      },
      sample_columns: sampleColumns,
      group_column: plainText(raw.group_col || raw.group_column || ''),
      steps: steps,
      current: plainText(raw.current || ''),
      warnings: normalizeStringList(raw.warnings),
      notes: normalizeStringList(raw.notes),
      history: history,
      analyses_run: runs,
    }
  }

  function normalizePreview(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    var metaRaw = raw.feature_meta && typeof raw.feature_meta === 'object' ? raw.feature_meta : {}
    var columns = []
    if (Array.isArray(raw.columns)) {
      for (var index = 0; index < raw.columns.length; index++) {
        var item = raw.columns[index] && typeof raw.columns[index] === 'object' ? raw.columns[index] : {}
        var key = plainText(item.key || item.name || '')
        if (!key) continue
        columns.push({ key: key, kind: plainText(item.kind || 'categorical'), n_unique: asNumber(item.n_unique) })
      }
    } else if (Array.isArray(raw.sample_columns)) {
      for (var fallbackIndex = 0; fallbackIndex < raw.sample_columns.length; fallbackIndex++) {
        var fallbackItem = raw.sample_columns[fallbackIndex] && typeof raw.sample_columns[fallbackIndex] === 'object' ? raw.sample_columns[fallbackIndex] : {}
        var fallbackKey = plainText(fallbackItem.key || fallbackItem.name || '')
        if (!fallbackKey) continue
        columns.push({ key: fallbackKey, kind: plainText(fallbackItem.kind || 'categorical'), n_unique: asNumber(fallbackItem.n_unique) })
      }
    }
    return {
      features: normalizeStringList(raw.features).slice(0, 500),
      samples: normalizeStringList(raw.samples).slice(0, 200),
      n_samples: asNumber(raw.n_samples),
      n_features: asNumber(raw.n_features),
      columns: columns,
      steps: Array.isArray(raw.steps) ? raw.steps.map(function (step) { return plainText(step) }) : [],
      current: plainText(raw.current || ''),
      feature_meta: {
        has_mz: metaRaw.has_mz === true,
        has_rt: metaRaw.has_rt === true,
        n_annotated: asNumber(metaRaw.n_annotated !== undefined ? metaRaw.n_annotated : metaRaw.annotated),
      },
    }
  }
  function normalizeHealth(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    var engine = raw.engine && typeof raw.engine === 'object' ? raw.engine : raw
    var capabilities = engine.capabilities && typeof engine.capabilities === 'object' ? engine.capabilities : {}
    var packages = engine.packages && typeof engine.packages === 'object' ? engine.packages : {}
    var ready = engine.ready === true
    if (isNil(engine.ready) && (typeof engine.python === 'string' || typeof engine.mode === 'string')) ready = true
    var steps = Array.isArray(capabilities.steps)
      ? capabilities.steps.map(function (item) { return typeof item === 'string' ? item : plainText(item && item.id) }).filter(Boolean)
      : []
    var metabolights = null
    if (raw.metabolights !== undefined) metabolights = raw.metabolights === true
    else if (engine.metabolights !== undefined) metabolights = engine.metabolights === true
    return {
      ready: ready,
      mode: plainText(engine.mode || ''),
      python: plainText(engine.python || ''),
      platform: plainText(engine.platform || ''),
      version: plainText(engine.version || ''),
      workdir: plainText(raw.workdir || engine.workdir || ''),
      packages: packages,
      capabilities: {
        steps: steps,
        analyses: Array.isArray(capabilities.analyses) ? capabilities.analyses : [],
        plots: Array.isArray(capabilities.plots) ? capabilities.plots : [],
        formats: Array.isArray(capabilities.formats) ? capabilities.formats : [],
      },
      geo: raw.geo !== false,
      metabolights: metabolights,
      error: plainText(raw.error || engine.error || ''),
      lastProgress: engine.lastProgress && typeof engine.lastProgress === 'object' ? engine.lastProgress : null,
    }
  }
  /** 引擎徽章：未就绪时把可读原因带出来（顶栏显示）。 */
  function engineBadge(engine) {
    var health = engine && (engine.capabilities || engine.ready !== undefined) ? engine : normalizeHealth(engine)
    var detail = []
    if (health.python) detail.push('Python ' + health.python)
    if (health.mode) detail.push(health.mode)
    if (!health.ready) {
      var reason = health.error || '引擎尚未启动'
      return {
        ready: false,
        tone: health.error ? 'bad' : 'warn',
        labelKey: health.error ? 'engineBad' : 'engineBooting',
        detail: detail.join(' · '),
        reason: reason,
      }
    }
    return { ready: true, tone: 'ok', labelKey: 'engineReady', detail: detail.join(' · '), reason: '' }
  }

  function normalizeSuggestion(item, index) {
    var raw = item && typeof item === 'object' ? item : { title: String(item || '') }
    return {
      id: plainText(raw.id || ('suggestion-' + index)),
      title: plainText(raw.title || raw.detail || ''),
      detail: plainText(raw.detail || ''),
      action: plainText(raw.action || ''),
      step: plainText(raw.step || ''),
      params: raw.params && typeof raw.params === 'object' ? raw.params : null,
    }
  }

  function normalizeSuggestions(list) {
    if (!Array.isArray(list)) return []
    var out = []
    for (var index = 0; index < list.length; index++) out.push(normalizeSuggestion(list[index], index))
    return out
  }

  function normalizeActivityEntry(entry, index) {
    var raw = entry && typeof entry === 'object' ? entry : {}
    var ok = raw.ok !== false && raw.status !== 'failed' && raw.status !== 'error'
    var action = plainText(raw.action || raw.cmd || raw.step || raw.kind || raw.op || '')
    var text = plainText(raw.summary || raw.message || raw.detail || raw.error || (typeof entry === 'string' ? entry : ''), 300)
    return {
      id: plainText(raw.id || ('activity-' + index)),
      at: plainText(raw.at || raw.time || raw.ts || ''),
      actor: plainText(raw.actor || 'user'),
      action: action,
      text: text,
      ok: ok,
      ms: asNumber(raw.ms),
      dataset: plainText(raw.dataset || ''),
      step: plainText(raw.step || ''),
    }
  }

  function normalizeState(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    var datasets = []
    if (Array.isArray(raw.datasets)) {
      for (var index = 0; index < raw.datasets.length; index++) datasets.push(normalizeDataset(raw.datasets[index]))
    }
    var activity = []
    if (Array.isArray(raw.activity)) {
      for (var activityIndex = 0; activityIndex < raw.activity.length; activityIndex++) {
        activity.push(normalizeActivityEntry(raw.activity[activityIndex], activityIndex))
      }
    }
    var active = plainText(raw.active || raw.active_id || '')
    if (!active && datasets.length) active = datasets[0].id
    return {
      active: active,
      datasets: datasets,
      engine: normalizeHealth(raw.engine ? { engine: raw.engine } : raw),
      suggestions: normalizeSuggestions(raw.suggestions),
      recipes: Array.isArray(raw.recipes) ? raw.recipes : [],
      steps: Array.isArray(raw.steps) ? raw.steps : [],
      activity: activity,
      workdir: plainText(raw.workdir || ''),
    }
  }

  /** /metabolomics/analyses 目录归一化：分组 + 参数 schema + 可用性。 */
  function normalizeAnalyses(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    var list = Array.isArray(raw.analyses) ? raw.analyses : []
    var out = []
    for (var index = 0; index < list.length; index++) {
      var item = list[index] && typeof list[index] === 'object' ? list[index] : {}
      var params = []
      if (Array.isArray(item.params)) {
        for (var paramIndex = 0; paramIndex < item.params.length; paramIndex++) {
          var descriptor = item.params[paramIndex] && typeof item.params[paramIndex] === 'object' ? item.params[paramIndex] : {}
          if (!descriptor.key) continue
          params.push({
            key: plainText(descriptor.key),
            label: plainText(descriptor.label || descriptor.key),
            type: plainText(descriptor.type || 'string'),
            default: descriptor.default,
            min: asNumber(descriptor.min),
            max: asNumber(descriptor.max),
            step: asNumber(descriptor.step),
            placeholder: plainText(descriptor.placeholder || ''),
            options: Array.isArray(descriptor.options)
              ? descriptor.options.map(function (option) {
                  if (option && typeof option === 'object') return { value: plainText(option.value), label: plainText(option.label || option.value) }
                  return { value: plainText(option), label: plainText(option) }
                })
              : null,
          })
        }
      }
      out.push({
        id: plainText(item.id || ''),
        title: plainText(item.title || item.id || ''),
        group: plainText(item.group || '其它'),
        method: plainText(item.method || ''),
        needs: Array.isArray(item.needs) ? item.needs.map(function (need) { return plainText(need) }) : [],
        params: params,
        available: item.available !== false,
        reason: plainText(item.reason || ''),
        native: item.native !== false,
      })
    }
    var groups = Array.isArray(raw.groups) ? raw.groups.map(function (item) { return plainText(item) }).filter(Boolean) : []
    if (!groups.length) {
      var seen = {}
      for (var groupIndex = 0; groupIndex < out.length; groupIndex++) {
        if (!seen[out[groupIndex].group]) { seen[out[groupIndex].group] = true; groups.push(out[groupIndex].group) }
      }
    }
    return {
      analyses: out,
      groups: groups,
      diagnostics: raw.diagnostics && typeof raw.diagnostics === 'object' ? raw.diagnostics : {},
      columns: Array.isArray(raw.columns) ? raw.columns : [],
    }
  }

  /** audit 归一化：score 0–100 取整，level 三态。 */
  function normalizeAudit(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    var checks = []
    var list = Array.isArray(raw.checks) ? raw.checks : []
    for (var index = 0; index < list.length; index++) {
      var item = list[index] && typeof list[index] === 'object' ? list[index] : {}
      checks.push({
        id: plainText(item.id || ('check-' + index)),
        level: item.level === 'fail' ? 'fail' : (item.level === 'warn' ? 'warn' : 'ok'),
        title: plainText(item.title || item.id || ''),
        detail: plainText(item.detail || ''),
        hint: plainText(item.hint || ''),
      })
    }
    var score = asNumber(raw.score)
    return {
      score: score === null ? 0 : Math.round(clamp(score, 0, 100)),
      level: raw.level === 'fail' ? 'fail' : (raw.level === 'warn' ? 'warn' : 'ok'),
      checks: checks,
    }
  }

  function normalizeJob(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    var job = raw.job && typeof raw.job === 'object' ? raw.job : raw
    var pct = asNumber(job.pct)
    return {
      id: plainText(job.id || ''),
      kind: plainText(job.kind || ''),
      status: plainText(job.status || ''),
      pct: pct === null ? 0 : clamp(pct, 0, 100),
      message: plainText(job.message || job.detail || ''),
      result: job.result && typeof job.result === 'object' ? job.result : null,
      error: plainText(job.error || ''),
    }
  }

  /**
   * /jobs 响应状态机：running | done | failed | timeout
   * timeout 由调用方传 elapsedMs/timeoutMs 判定；done 优先级最高（超时判定不覆盖已完成）。
   */
  function jobPolling(response, options) {
    var job = normalizeJob(response)
    var opts = options && typeof options === 'object' ? options : {}
    var elapsedMs = asNumber(opts.elapsedMs)
    if (elapsedMs === null || elapsedMs < 0) elapsedMs = 0
    var timeoutMs = asNumber(opts.timeoutMs)
    if (timeoutMs === null || timeoutMs <= 0) timeoutMs = POLL_TIMEOUT_MS
    var done = job.status === 'done' || job.status === 'success' || job.status === 'completed'
    var failed = job.status === 'failed' || job.status === 'error'
    var timedOut = !done && !failed && elapsedMs >= timeoutMs
    var status = done ? 'done' : (failed ? 'failed' : (timedOut ? 'timeout' : 'running'))
    return {
      id: job.id,
      kind: job.kind,
      status: status,
      pct: job.pct,
      message: job.message,
      result: job.result,
      error: job.error,
      done: done,
      failed: failed,
      timedOut: timedOut,
      running: status === 'running',
    }
  }

  function jobDone(response) {
    return jobPolling(response).done
  }

  function pickJobId(payload) {
    var raw = payload && typeof payload === 'object' ? payload : {}
    if (raw.job && typeof raw.job === 'object' && raw.job.id) return plainText(raw.job.id)
    if (raw.job_id) return plainText(raw.job_id)
    if (raw.id) return plainText(raw.id)
    return ''
  }

  // ==========================================================================
  // API 客户端：全部走同源 /metabolomics/*；fetch 可注入（单测用假 fetch）。
  // 每个方法只描述「一个操作」，HTTP 路由与 Agent 工具（impl.mjs 的 op*）共用同一套实现。
  // 路由以 lib/impl.mjs 的 handle() 为准；**不调用不存在的接口**（例如没有 /pick）。
  // ==========================================================================

  function createApi(fetchImpl, baseUrl) {
    var base = typeof baseUrl === 'string' && baseUrl ? baseUrl.replace(/\/$/, '') : API

    function resolveFetch() {
      if (typeof fetchImpl === 'function') return fetchImpl
      if (typeof globalThis.fetch === 'function') return globalThis.fetch.bind(globalThis)
      return null
    }

    function apiError(message, data, status) {
      var error = new Error(message)
      error.__mbApiError = true
      error.code = ''
      if (data && typeof data === 'object') {
        error.data = data
        if (typeof data.code === 'string' && data.code) error.code = data.code
        if (Array.isArray(data.candidates)) error.candidates = data.candidates
        if (data.missing_module) error.missing_module = String(data.missing_module)
        if (data.reason) error.reason = String(data.reason)
      }
      if (isFinite(Number(status)) && Number(status) > 0) error.status = Number(status)
      return error
    }

    function pickError(data) {
      if (!data || typeof data !== 'object') return ''
      if (typeof data.error === 'string' && data.error) return data.error
      if (typeof data.message === 'string' && data.message) return data.message
      return ''
    }

    function parseResponse(response) {
      if (!response) return Promise.reject(apiError('Host 返回了空响应'))
      var ok = response.ok !== false
      var status = Number(response.status) || 0
      var textPromise = typeof response.text === 'function'
        ? Promise.resolve().then(function () { return response.text() }).catch(function () { return '' })
        : Promise.resolve('')
      return textPromise.then(function (text) {
        var data = null
        if (text) {
          try { data = JSON.parse(text) } catch (ignored) { data = null }
        }
        if (!ok) throw apiError(pickError(data) || ('HTTP ' + status), data, status)
        if (data && data.ok === false) throw apiError(pickError(data) || '请求失败', data, status)
        return data && typeof data === 'object' ? data : {}
      })
    }

    function request(path, settings, extra) {
      var doFetch = resolveFetch()
      if (!doFetch) return Promise.reject(apiError('当前环境不支持 fetch'))
      var options = Object.assign({}, settings || {})
      var extraOptions = extra && typeof extra === 'object' ? extra : {}
      var timeoutMs = isFinite(Number(extraOptions.timeoutMs)) && Number(extraOptions.timeoutMs) > 0
        ? Number(extraOptions.timeoutMs)
        : DEFAULT_API_TIMEOUT_MS
      var controller = null
      var timer = null
      if (extraOptions.signal) {
        options.signal = extraOptions.signal
      } else if (typeof AbortController === 'function' && timeoutMs > 0) {
        controller = new AbortController()
        options.signal = controller.signal
        timer = setTimeout(function () {
          try { controller.abort() } catch (ignored) { /* ignore */ }
        }, timeoutMs)
      }
      function clear() {
        if (timer) { clearTimeout(timer); timer = null }
      }
      var promise = Promise.resolve()
        .then(function () { return doFetch(base + path, options) })
        .then(parseResponse)
      return promise.then(function (value) {
        clear()
        return value
      }, function (error) {
        clear()
        if (error && error.__mbApiError) throw error
        var message = formatError(error)
        if (error && (error.name === 'AbortError' || /aborted/i.test(message))) {
          if (extraOptions.signal) throw error
          throw apiError('请求超时（' + Math.round(timeoutMs / 1000) + ' 秒）：引擎可能还在计算，请稍后刷新重试。')
        }
        throw apiError('网络请求失败：' + (message || '无法连接 Host'))
      })
    }

    function jsonPost(path, body, extra) {
      return request(path, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body || {}),
      }, extra)
    }

    function withActor(body) {
      return Object.assign({ actor: 'user' }, body && typeof body === 'object' ? body : {})
    }

    function datasetQuery(dataset) {
      var id = plainText(dataset)
      return id ? '?dataset=' + encodeURIComponent(id) : ''
    }

    return {
      base: base,
      request: request,
      getHealth: function (extra) { return request('/health', { method: 'GET' }, extra) },
      getState: function (extra) { return request('/state', { method: 'GET' }, extra) },
      getActivity: function (extra) { return request('/activity', { method: 'GET' }, extra) },
      getFormats: function (extra) { return request('/formats', { method: 'GET' }, extra) },
      getSteps: function (extra) { return request('/steps', { method: 'GET' }, extra) },
      getAnalyses: function (dataset, extra) {
        return request('/analyses' + datasetQuery(dataset), { method: 'GET' }, extra)
      },
      analyses: function (dataset, extra) {
        return request('/analyses' + datasetQuery(dataset), { method: 'GET' }, extra)
      },
      getJob: function (id, extra) {
        return request('/jobs?id=' + encodeURIComponent(plainText(id)), { method: 'GET' }, extra)
      },
      geoSearch: function (filters, extra) {
        return jsonPost('/geo/search', buildSearchPayload(filters), Object.assign({ timeoutMs: 120000 }, extra || {}))
      },
      geoDetail: function (gse, extra) {
        return jsonPost('/geo/detail', withActor({ gse: plainText(gse).toUpperCase() }), Object.assign({ timeoutMs: 180000 }, extra || {}))
      },
      geoDownload: function (options, extra) {
        var body = {
          gse: plainText(options && options.gse).toUpperCase(),
          files: Array.isArray(options && options.files) ? options.files.slice() : [],
          dest_dir: plainText(options && options.dest_dir) || DEFAULT_GEO_DIR,
          overwrite: options && options.overwrite === true,
        }
        if (options && options.open_after === true) body.open_after = true
        return jsonPost('/geo/download', withActor(body), Object.assign({ timeoutMs: 90000 }, extra || {}))
      },
      mlSearch: function (options, extra) {
        return jsonPost('/ml/search', buildMlSearchPayload(options), Object.assign({ timeoutMs: 120000 }, extra || {}))
      },
      mlDetail: function (accession, extra) {
        return jsonPost('/ml/detail', withActor({ accession: plainText(accession).toUpperCase() }),
          Object.assign({ timeoutMs: 180000 }, extra || {}))
      },
      mlDownload: function (options, extra) {
        var body = {
          accession: plainText(options && options.accession).toUpperCase(),
          files: Array.isArray(options && options.files) ? options.files.slice() : [],
          dest_dir: plainText(options && options.dest_dir) || DEFAULT_ML_DIR,
          overwrite: options && options.overwrite === true,
        }
        if (options && options.open_after === true) body.open_after = true
        return jsonPost('/ml/download', withActor(body), Object.assign({ timeoutMs: 90000 }, extra || {}))
      },
      /** 浏览器上传：Host 从 x-dsh-filename 取名字，body 直接是文件字节流。 */
      upload: function (file, extra) {
        var name = file && file.name ? String(file.name) : 'upload.csv'
        return request('/dataset/upload', {
          method: 'POST',
          headers: { 'x-dsh-filename': name },
          body: file,
        }, Object.assign({ timeoutMs: HEAVY_API_TIMEOUT_MS }, extra || {}))
      },
      openDataset: function (options, extra) {
        return jsonPost('/dataset/open', withActor(buildImportPayload(options)),
          Object.assign({ timeoutMs: HEAVY_API_TIMEOUT_MS }, extra || {}))
      },
      demo: function (extra) { return jsonPost('/dataset/demo', withActor({}), extra) },
      selectDataset: function (dataset, extra) {
        return jsonPost('/dataset/select', withActor({ dataset: plainText(dataset) }), extra)
      },
      closeDataset: function (dataset, extra) {
        return jsonPost('/dataset/close', withActor({ dataset: plainText(dataset) }), extra)
      },
      inspectSource: function (options, extra) {
        return jsonPost('/inspect', withActor(buildImportPayload(options)),
          Object.assign({ timeoutMs: 180000 }, extra || {}))
      },
      preview: function (dataset, extra) {
        return jsonPost('/preview', withActor({ dataset: plainText(dataset) }), extra)
      },
      samples: function (payload, extra) {
        return jsonPost('/samples', withActor(payload && typeof payload === 'object' ? payload : {}), extra)
      },
      runStep: function (dataset, step, params, extra) {
        return jsonPost('/run', withActor({
          dataset: plainText(dataset),
          step: plainText(step),
          params: params && typeof params === 'object' ? params : {},
        }), Object.assign({ timeoutMs: HEAVY_API_TIMEOUT_MS }, extra || {}))
      },
      plot: function (dataset, kind, params, extra) {
        return jsonPost('/plot', withActor({
          dataset: plainText(dataset),
          kind: plainText(kind),
          params: params && typeof params === 'object' ? params : {},
        }), Object.assign({ timeoutMs: HEAVY_API_TIMEOUT_MS }, extra || {}))
      },
      analysis: function (dataset, id, params, extra) {
        return jsonPost('/analysis', withActor({
          dataset: plainText(dataset),
          id: plainText(id),
          params: params && typeof params === 'object' ? params : {},
        }), Object.assign({ timeoutMs: HEAVY_API_TIMEOUT_MS * 2 }, extra || {}))
      },
      exportData: function (dataset, what, path, extra) {
        var body = { dataset: plainText(dataset), what: plainText(what) }
        if (path) body.out_path = plainText(path)
        return jsonPost('/export', withActor(body), Object.assign({ timeoutMs: HEAVY_API_TIMEOUT_MS }, extra || {}))
      },
      audit: function (dataset, extra) {
        return jsonPost('/audit', withActor({ dataset: plainText(dataset) }), extra)
      },
      suggest: function (dataset, extra) {
        return jsonPost('/suggest', withActor({ dataset: plainText(dataset) }), extra)
      },
      recipes: function (extra) {
        return jsonPost('/recipe', withActor({}), extra)
      },
      recipe: function (id, extra) {
        return jsonPost('/recipe', withActor({ recipe: plainText(id) }), extra)
      },
      recipeRun: function (options, extra) {
        var raw = options && typeof options === 'object' ? options : {}
        return jsonPost('/recipe/run', withActor({
          dataset: plainText(raw.dataset),
          recipe: plainText(raw.recipe) || 'standard',
        }), Object.assign({ timeoutMs: HEAVY_API_TIMEOUT_MS }, extra || {}))
      },
      reset: function (extra) { return jsonPost('/reset', withActor({}), extra) },
      restartEngine: function (extra) { return jsonPost('/engine/restart', withActor({}), extra) },
    }
  }


  // --------------------------------------------------------------------------
  // 展示小工具：特征检索 / 表格排序 / 图缓存键 / 取词 / tab 与分组映射
  // --------------------------------------------------------------------------

  /** 特征 id 检索（前缀优先，其次包含；大小写不敏感）。 */
  function matchFeatures(allFeatures, query, limit) {
    var source = Array.isArray(allFeatures) ? allFeatures : []
    var needle = plainText(query).toUpperCase()
    var max = clampInt(limit, 1, 500, 50)
    var prefix = []
    var contains = []
    for (var index = 0; index < source.length; index++) {
      var feature = String(source[index] || '')
      var upper = feature.toUpperCase()
      if (!needle) { prefix.push(feature); continue }
      if (upper.indexOf(needle) === 0) prefix.push(feature)
      else if (upper.indexOf(needle) > 0) contains.push(feature)
      if (prefix.length >= max) break
    }
    return prefix.concat(contains).slice(0, max)
  }

  function tableColumns(table) {
    return table && Array.isArray(table.columns) ? table.columns : []
  }

  function sortRows(rows, key, direction, columns) {
    var list = Array.isArray(rows) ? rows.slice() : []
    var numeric = false
    if (Array.isArray(columns)) {
      for (var index = 0; index < columns.length; index++) {
        if (columns[index] && columns[index].key === key && columns[index].type === 'number') numeric = true
      }
    }
    var factor = direction === 'desc' ? -1 : 1
    return list.sort(function (a, b) {
      var left = a && typeof a === 'object' ? a[key] : null
      var right = b && typeof b === 'object' ? b[key] : null
      if (isNil(left) && isNil(right)) return 0
      if (isNil(left)) return 1
      if (isNil(right)) return -1
      if (numeric) {
        var leftNumber = asNumber(left)
        var rightNumber = asNumber(right)
        if (leftNumber !== null && rightNumber !== null) return (leftNumber - rightNumber) * factor
      }
      return String(left).localeCompare(String(right)) * factor
    })
  }

  function formatCell(value, numeric) {
    if (isNil(value) || value === '') return '—'
    if (typeof value === 'number') {
      if (!numeric) return String(value)
      var rounded = Math.abs(value) >= 1000 ? value : Number(value.toPrecision(4))
      return String(rounded)
    }
    if (typeof value === 'boolean') return value ? '是' : '否'
    if (Array.isArray(value)) return value.join(', ')
    return String(value)
  }

  function formatTableRows(table, limit) {
    var columns = tableColumns(table)
    var rows = table && Array.isArray(table.rows) ? table.rows : []
    var max = clampInt(limit, 1, 5000, MAX_TABLE_RENDER_ROWS)
    var shown = rows.slice(0, max)
    var out = []
    for (var index = 0; index < shown.length; index++) {
      var row = shown[index] && typeof shown[index] === 'object' ? shown[index] : {}
      var cells = []
      for (var columnIndex = 0; columnIndex < columns.length; columnIndex++) {
        var column = columns[columnIndex]
        cells.push({
          key: column.key,
          text: formatCell(row[column.key], column.type === 'number'),
          numeric: column.type === 'number',
        })
      }
      out.push({ key: 'row-' + index, cells: cells })
    }
    return {
      columns: columns,
      rows: out,
      total: rows.length,
      shown: out.length,
      truncated: rows.length > out.length || (table && table.truncated === true),
    }
  }

  /** PlotBlock 缓存键：kind + 稳定序列化参数（键排序、跳过空值）。 */
  function plotKey(kind, params) {
    var name = plainText(kind)
    var source = params && typeof params === 'object' && !Array.isArray(params) ? params : {}
    var keys = Object.keys(source).sort()
    var parts = []
    for (var index = 0; index < keys.length; index++) {
      var key = keys[index]
      var value = source[key]
      if (isNil(value) || value === '' || value === false) continue
      if (Array.isArray(value)) {
        var joined = value.map(function (item) { return plainText(item) }).filter(Boolean).join(',')
        if (joined) parts.push(key + '=' + joined)
        continue
      }
      if (typeof value === 'object') {
        try { parts.push(key + '=' + JSON.stringify(value)) } catch (ignored) { /* 不可序列化就跳过 */ }
        continue
      }
      parts.push(key + '=' + String(value))
    }
    return name + (parts.length ? '|' + parts.join('&') : '')
  }

  /**
   * 宿主 locale 服务 → 取词函数。宿主不可用时静默回落 DICT_ZH，
   * 保证任何环境（含单测）下 T() 都不会返回裸 key 或抛异常。
   */
  function createTranslator(locale) {
    var bound = null
    try {
      if (locale && typeof locale.bind === 'function') bound = locale.bind(NS)
    } catch (ignored) {
      bound = null
    }
    return function translate(key) {
      if (typeof bound === 'function') {
        try {
          var value = bound(key)
          if (typeof value === 'string' && value && value !== key) return value
        } catch (ignored) { /* 宿主 locale 不可用时静默回落 */ }
      }
      var fallback = DICT_ZH[key]
      return typeof fallback === 'string' ? fallback : String(key)
    }
  }

  /** 服务端分析分组 → tab（未映射的分组落到多变量 tab）。 */
  function analysisGroupTab(group) {
    var name = plainText(group)
    return ANALYSIS_TAB[name] || ANALYSIS_TAB_FALLBACK
  }

  /** 某个 tab 该展示哪些分析 id（TAB_ANALYSES 是白名单，不是 schema）。 */
  function tabAnalyses(tab) {
    var list = TAB_ANALYSES[tab]
    return Array.isArray(list) ? list.slice() : []
  }

  /** 某个 tab 的快捷出图定义。 */
  function tabPlots(tab) {
    var list = TAB_PLOTS[tab]
    return Array.isArray(list) ? list.slice() : []
  }

  /** 数据集分组列：优先 dataset.group_column，其次样本表里像分组的列，最后回落 group。 */
  function groupColumn(dataset) {
    if (!dataset || typeof dataset !== 'object') return ''
    var explicit = plainText(dataset.group_column || '')
    if (explicit) return explicit
    var columns = Array.isArray(dataset.sample_columns) ? dataset.sample_columns : []
    for (var index = 0; index < columns.length; index++) {
      var key = plainText(columns[index] && columns[index].key)
      if (/^(group|class|condition|treatment|phenotype)$/i.test(key)) return key
    }
    for (var cursor = 0; cursor < columns.length; cursor++) {
      var name = plainText(columns[cursor] && columns[cursor].key)
      if (/group|class|cond|treat|pheno/i.test(name)) return name
    }
    for (var fallback = 0; fallback < columns.length; fallback++) {
      var kind = plainText(columns[fallback] && columns[fallback].kind)
      if (kind === 'categorical' && Number(columns[fallback].n_unique) > 1) return plainText(columns[fallback].key)
    }
    return ''
  }

  /** 特征列是否带 m/z（决定加合物/分子式分析是否可达）。 */
  function hasMz(dataset, preview) {
    var meta = (dataset && dataset.feature_meta) || (preview && preview.feature_meta) || {}
    return meta.has_mz === true
  }


  // ==========================================================================
  // 纯函数测试出口。node 环境（typeof window === 'undefined'）挂到
  // globalThis.__MB_TEST__；浏览器里也会挂进 factory 返回的 __test__，便于联调。
  // ==========================================================================
  var MB_TEST = {
    NS: NS,
    API: API,
    POLL_MS: POLL_MS,
    POLL_TIMEOUT_MS: POLL_TIMEOUT_MS,
    DEFAULT_GEO_DIR: DEFAULT_GEO_DIR,
    DEFAULT_ML_DIR: DEFAULT_ML_DIR,
    MAX_HISTORY_ROWS: MAX_HISTORY_ROWS,
    DEFAULT_API_TIMEOUT_MS: DEFAULT_API_TIMEOUT_MS,
    STEPS: STEPS,
    STEP_BY_ID: STEP_BY_ID,
    TAB_KEYS: TAB_KEYS,
    TAB_META: TAB_META,
    TAB_FOR_STEP: TAB_FOR_STEP,
    ANALYSIS_TAB: ANALYSIS_TAB,
    ANALYSIS_TAB_FALLBACK: ANALYSIS_TAB_FALLBACK,
    TAB_ANALYSES: TAB_ANALYSES,
    TAB_PLOTS: TAB_PLOTS,
    PALETTE: PALETTE,
    CONTINUOUS_STOPS: CONTINUOUS_STOPS,
    PLOT_KINDS: PLOT_KINDS,
    EXPORT_KINDS: EXPORT_KINDS,
    QUICK_EXPORTS: QUICK_EXPORTS,
    COLOR_BY: COLOR_BY,
    GEO_METHODS: GEO_METHODS,
    DATA_FORMATS: DATA_FORMATS,
    AUDIT_LEVEL_STYLE: AUDIT_LEVEL_STYLE,
    DICT_ZH: DICT_ZH,
    DICT_EN: DICT_EN,
    WORDS: WORDS,
    clamp: clamp,
    asNumber: asNumber,
    plainText: plainText,
    fmtNumber: fmtNumber,
    fmtPct: fmtPct,
    fmtP: fmtP,
    fmtBytes: formatBytes,
    formatBytes: formatBytes,
    formatDuration: formatDuration,
    formatClock: formatClock,
    formatError: formatError,
    hexToRgb: hexToRgb,
    colorForGroup: colorForGroup,
    continuousColor: continuousColor,
    groupBy: groupBy,
    extentOf: extentOf,
    normalizeExtent: normalizeExtent,
    miniPlotScreen: miniPlotScreen,
    buildSearchPayload: buildSearchPayload,
    buildMlSearchPayload: buildMlSearchPayload,
    buildImportPayload: buildImportPayload,
    suggestionInstruction: suggestionInstruction,
    defaultParams: defaultParams,
    collectStepParams: collectStepParams,
    parseTokenList: parseTokenList,
    formatParamValue: formatParamValue,
    doneStepSet: doneStepSet,
    isStepReady: isStepReady,
    stepStatusOf: stepStatusOf,
    stepStatusKey: stepStatusKey,
    stepById: stepById,
    figureSrc: figureSrc,
    normalizeFigure: normalizeFigure,
    normalizeTable: normalizeTable,
    normalizeResult: normalizeResult,
    normalizeDataset: normalizeDataset,
    normalizePreview: normalizePreview,
    normalizeHealth: normalizeHealth,
    normalizeState: normalizeState,
    normalizeSuggestions: normalizeSuggestions,
    normalizeActivityEntry: normalizeActivityEntry,
    normalizeAnalyses: normalizeAnalyses,
    normalizeAudit: normalizeAudit,
    normalizeJob: normalizeJob,
    normalizeStringList: normalizeStringList,
    engineBadge: engineBadge,
    jobPolling: jobPolling,
    jobDone: jobDone,
    pickJobId: pickJobId,
    createApi: createApi,
    apiClient: createApi,
    createTranslator: createTranslator,
    matchFeatures: matchFeatures,
    sortRows: sortRows,
    formatTableRows: formatTableRows,
    tableColumns: tableColumns,
    formatCell: formatCell,
    plotKey: plotKey,
    analysisGroupTab: analysisGroupTab,
    tabAnalyses: tabAnalyses,
    tabPlots: tabPlots,
    groupColumn: groupColumn,
    hasMz: hasMz,
  }
  // ==========================================================================
  // 浏览器层：React 组件 + 样式 + 槽注册。只 require('react')。
  // ==========================================================================

  function factory(require) {
    var React = require('react')
    var useState = React.useState
    var useEffect = React.useEffect
    var useRef = React.useRef
    var useMemo = React.useMemo

    /** React.createElement 简写（不写 JSX，避免宿主构建链依赖）。 */
    function h(type, props) {
      var children = Array.prototype.slice.call(arguments, 2)
      return React.createElement.apply(React, [type, props].concat(children))
    }

    /** t()：apply() 里用宿主 locale 服务重新绑定；默认回落中文词典。 */
    var T = function (key) {
      var fallback = DICT_ZH[key]
      return typeof fallback === 'string' ? fallback : String(key)
    }

    function warnPlugin(message, error) {
      try { console.warn('[dsh-metabolomics] ' + message, error || '') } catch (ignored) { /* ignore */ }
    }

    /** 模板取词：字典里用 `{v}` 占位，避免依赖宿主 t() 的插值实现。 */
    function tf(translate, key, value) {
      var raw = typeof translate === 'function' ? translate(key) : ''
      var template = typeof raw === 'string' && raw && raw !== key ? raw : (DICT_ZH[key] || '')
      return String(template).replace(/\{v\}/g, isNil(value) ? '' : String(value))
    }

    // ==========================================================================
    // 样式：一个 CSS 字符串，由 <style> 渲染在组件内部（生命周期跟随组件）。
    // 颜色只用 --dsw-alias-* 主题变量，派生色统一走 color-mix，深浅色自适应。
    // 类名一律 mb- 前缀；数据色（PALETTE / CONTINUOUS_STOPS）走内联样式。
    // ==========================================================================
    var CSS = [
      '.mb-root{display:flex;flex-direction:column;height:100%;min-height:72vh;overflow:hidden;position:relative;isolation:isolate;color:var(--dsw-alias-label-primary);background:var(--dsw-alias-bg-base);border-radius:12px;font-size:13px;line-height:1.5;font-variant-numeric:tabular-nums}',
      '.mb-root *,.mb-root *::before,.mb-root *::after{box-sizing:border-box}',
      '.mb-root button,.mb-root input,.mb-root select,.mb-root textarea{font-family:inherit;font-variant-numeric:tabular-nums}',
      '.mb-root ::selection{background:color-mix(in srgb,var(--dsw-alias-brand-primary) 24%,transparent)}',

      /* 顶栏 + 1px 渐变线 */
      '.mb-top{position:relative;display:flex;align-items:center;gap:10px;flex-wrap:wrap;flex:none;min-height:52px;padding:9px 14px;border-bottom:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1)}',
      '.mb-top::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:1px;background:linear-gradient(90deg,transparent,color-mix(in srgb,var(--dsw-alias-brand-primary) 55%,transparent) 26%,color-mix(in srgb,var(--dsw-alias-state-success-primary) 45%,transparent) 58%,transparent);pointer-events:none}',
      '.mb-brand{display:flex;align-items:center;gap:9px;min-width:0}',
      '.mb-brand-mark{display:inline-flex;align-items:center;justify-content:center;flex:none;width:28px;height:28px;border-radius:9px;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 34%,var(--dsw-alias-border-l1));background:linear-gradient(135deg,color-mix(in srgb,var(--dsw-alias-brand-primary) 16%,var(--dsw-alias-bg-layer-1)),var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-brand-primary)}',
      '.mb-title-block{display:flex;flex-direction:column;gap:1px;min-width:0}',
      '.mb-title{font-size:14px;font-weight:600;letter-spacing:.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-subtitle{font-size:11px;color:var(--dsw-alias-label-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-badge{display:inline-flex;align-items:center;gap:6px;max-width:340px;height:22px;padding:0 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:999px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);font-size:11px;line-height:1;cursor:default;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-badge-dot{width:6px;height:6px;border-radius:50%;flex:none;background:currentColor;box-shadow:0 0 0 2px color-mix(in srgb,currentColor 18%,transparent)}',
      '.mb-badge-ok{border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary) 40%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-success-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-state-success-primary)}',
      '.mb-badge-warn{border-color:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 42%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-state-warn-primary)}',
      '.mb-badge-bad{border-color:color-mix(in srgb,var(--dsw-alias-state-error-primary) 42%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-state-error-primary)}',
      '.mb-badge-reason{max-width:200px;overflow:hidden;text-overflow:ellipsis;color:var(--dsw-alias-state-error-primary);font-size:10.5px}',
      '.mb-chips{display:flex;align-items:center;gap:5px;flex-wrap:wrap;min-width:0;flex:1 1 200px}',
      '.mb-chip{display:inline-flex;align-items:center;gap:5px;max-width:210px;height:23px;padding:0 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:999px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);font:inherit;font-size:11.5px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:border-color .14s ease,background .14s ease,color .14s ease}',
      '.mb-chip:hover:not(:disabled){border-color:var(--dsw-alias-border-l2);color:var(--dsw-alias-label-primary)}',
      '.mb-chip-on{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 45%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-brand-primary)}',
      '.mb-chip-label{border:none;background:transparent;color:inherit;font:inherit;font-size:11.5px;cursor:pointer;padding:0;max-width:170px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
      '.mb-chip-x{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;padding:0;border:none;border-radius:50%;background:transparent;font-size:10px;color:var(--dsw-alias-label-tertiary);cursor:pointer}',
      '.mb-chip-x:hover{background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 18%,transparent);color:var(--dsw-alias-state-error-primary)}',
      '.mb-top-actions{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-left:auto;justify-content:flex-end}',

      /* 按钮 / 输入 */
      '.mb-btn{display:inline-flex;align-items:center;justify-content:center;gap:5px;height:28px;padding:0 10px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-primary);font:inherit;font-size:12px;line-height:1;cursor:pointer;white-space:nowrap;transition:border-color .14s ease,background .14s ease,opacity .14s ease,box-shadow .14s ease}',
      '.mb-btn:hover:not(:disabled){border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-overlay)}',
      '.mb-btn:active:not(:disabled){transform:translateY(.5px)}',
      '.mb-btn:disabled{opacity:.45;cursor:default}',
      '.mb-btn-primary{background:var(--dsw-alias-brand-primary);border-color:var(--dsw-alias-brand-primary);color:var(--dsw-alias-bg-base);box-shadow:0 1px 2px color-mix(in srgb,var(--dsw-alias-brand-primary) 32%,transparent)}',
      '.mb-btn-primary:hover:not(:disabled){opacity:.9;background:var(--dsw-alias-brand-primary);border-color:var(--dsw-alias-brand-primary)}',
      '.mb-btn-ghost{background:transparent;border-color:transparent;color:var(--dsw-alias-label-secondary)}',
      '.mb-btn-ghost:hover:not(:disabled){background:var(--dsw-alias-bg-layer-2);border-color:var(--dsw-alias-border-l1);color:var(--dsw-alias-label-primary)}',
      '.mb-btn-sm{height:24px;padding:0 8px;font-size:11.5px;border-radius:6px}',
      '.mb-btn-danger{color:var(--dsw-alias-state-error-primary);border-color:color-mix(in srgb,var(--dsw-alias-state-error-primary) 36%,var(--dsw-alias-border-l1))}',
      '.mb-input{height:28px;padding:0 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font:inherit;font-size:12px;outline:none;min-width:0;transition:border-color .14s ease,box-shadow .14s ease}',
      '.mb-input:focus{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 48%,var(--dsw-alias-border-l1));box-shadow:0 0 0 2px color-mix(in srgb,var(--dsw-alias-brand-primary) 12%,transparent)}',
      '.mb-input:disabled{opacity:.55}',
      '.mb-input::placeholder{color:var(--dsw-alias-label-tertiary)}',
      '.mb-select{cursor:pointer;padding-right:5px}',
      '.mb-textarea{min-height:54px;padding:6px 9px;resize:vertical;line-height:1.45}',
      '.mb-check{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--dsw-alias-label-secondary);cursor:pointer;user-select:none}',
      '.mb-check input{accent-color:var(--dsw-alias-brand-primary);width:14px;height:14px;cursor:pointer}',
      '.mb-form{display:flex;flex-direction:column;gap:7px}',
      '.mb-field{display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0}',
      '.mb-field-col{flex-direction:column;align-items:stretch;gap:4px}',
      '.mb-field-label{flex:0 1 auto;min-width:0;font-size:11.5px;color:var(--dsw-alias-label-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-field .mb-input{flex:1 1 auto;max-width:152px;height:25px;font-size:11.5px}',
      '.mb-field-col .mb-input{max-width:none;width:100%}',
      '.mb-multi{display:flex;flex-wrap:wrap;gap:4px}',
      '.mb-multi-opt{display:inline-flex;align-items:center;gap:4px;height:22px;padding:0 8px;border:1px solid var(--dsw-alias-border-l1);border-radius:6px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);font:inherit;font-size:11px;cursor:pointer;transition:border-color .14s ease,background .14s ease,color .14s ease}',
      '.mb-multi-opt:hover:not(:disabled){border-color:var(--dsw-alias-border-l2);color:var(--dsw-alias-label-primary)}',
      '.mb-multi-opt-on{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 46%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 10%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-brand-primary)}',

      /* 三栏骨架 */
      '.mb-body{display:flex;flex:1 1 auto;min-height:0;overflow:hidden}',
      '.mb-left{display:flex;flex-direction:column;flex:0 0 268px;width:268px;min-height:0;border-right:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1)}',
      '.mb-center{display:flex;flex-direction:column;flex:1 1 auto;min-width:0;min-height:0;background:var(--dsw-alias-bg-base)}',
      '.mb-right{display:flex;flex-direction:column;flex:0 0 300px;width:300px;min-height:0;overflow-y:auto;border-left:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1)}',
      '.mb-panel-head{display:flex;align-items:center;justify-content:space-between;gap:8px;flex:none;padding:9px 12px;border-bottom:1px solid var(--dsw-alias-border-l1);font-size:11px;font-weight:600;letter-spacing:.04em;color:var(--dsw-alias-label-secondary)}',
      '.mb-panel-note{font-weight:400;font-size:10.5px;letter-spacing:0;color:var(--dsw-alias-label-tertiary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-panel-body{display:flex;flex-direction:column;gap:9px;padding:10px 12px 16px;overflow-y:auto;min-height:0}',
      '.mb-banner{display:flex;align-items:center;justify-content:space-between;gap:10px;flex:none;padding:6px 14px;border-bottom:1px solid color-mix(in srgb,var(--dsw-alias-state-warn-primary) 32%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 7%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-label-primary);font-size:11.5px;animation:mb-fade .16s ease}',
      '.mb-banner-bad{border-bottom-color:color-mix(in srgb,var(--dsw-alias-state-error-primary) 34%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 6%,var(--dsw-alias-bg-layer-1))}',
      '.mb-banner-text{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',

      /* 左栏：数据集卡 + 12 张流程卡 */
      '.mb-dataset-card{display:flex;flex-direction:column;gap:5px;padding:10px 11px;border:1px solid var(--dsw-alias-border-l1);border-radius:12px;background:linear-gradient(160deg,color-mix(in srgb,var(--dsw-alias-brand-primary) 5%,var(--dsw-alias-bg-base)),var(--dsw-alias-bg-base) 58%);box-shadow:0 1px 2px color-mix(in srgb,var(--dsw-alias-label-primary) 5%,transparent)}',
      '.mb-dataset-name{font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-dataset-meta{display:flex;flex-wrap:wrap;gap:4px 8px;font-size:11px;color:var(--dsw-alias-label-tertiary)}',
      '.mb-dataset-meta b{font-weight:600;color:var(--dsw-alias-label-secondary)}',
      '.mb-steps{display:flex;flex-direction:column;gap:8px;padding:10px 12px 16px;overflow-y:auto;min-height:0}',
      '.mb-step{display:flex;flex-direction:column;gap:5px;padding:8px 10px;border:1px solid var(--dsw-alias-border-l1);border-radius:12px;background:var(--dsw-alias-bg-base);box-shadow:0 1px 2px color-mix(in srgb,var(--dsw-alias-label-primary) 4%,transparent);transition:border-color .16s ease,background .16s ease,opacity .16s ease}',
      '.mb-step-done{border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary) 28%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-success-primary) 4%,var(--dsw-alias-bg-base))}',
      '.mb-step-running{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 48%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 5%,var(--dsw-alias-bg-base))}',
      '.mb-step-blocked{opacity:.8}',
      '.mb-step-head{display:flex;align-items:center;gap:7px;min-width:0}',
      '.mb-step-dot{position:relative;display:inline-flex;align-items:center;justify-content:center;flex:none;width:20px;height:20px;border-radius:50%;border:1px solid var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);font-size:10.5px;font-weight:600}',
      '.mb-step-dot-done{border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary) 55%,transparent);background:color-mix(in srgb,var(--dsw-alias-state-success-primary) 14%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-state-success-primary)}',
      '.mb-step-dot-running{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 58%,transparent);background:color-mix(in srgb,var(--dsw-alias-brand-primary) 14%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-brand-primary);animation:mb-pulse 1.6s ease-in-out infinite}',
      '.mb-step-dot-pending{border-color:var(--dsw-alias-border-l2);color:var(--dsw-alias-label-tertiary)}',
      '.mb-step-dot-blocked{border-color:var(--dsw-alias-border-l1);color:var(--dsw-alias-label-tertiary)}',
      '.mb-step-title{flex:1 1 auto;min-width:0;font-size:12.5px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-step-status{flex:none;font-size:10.5px;color:var(--dsw-alias-label-tertiary);white-space:nowrap}',
      '.mb-status-done{color:var(--dsw-alias-state-success-primary)}',
      '.mb-status-running{color:var(--dsw-alias-brand-primary)}',
      '.mb-status-failed{color:var(--dsw-alias-state-error-primary)}',
      '.mb-status-ready{color:var(--dsw-alias-label-secondary)}',
      '.mb-step-caret{flex:none;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;padding:0;border:none;border-radius:6px;background:transparent;color:var(--dsw-alias-label-tertiary);cursor:pointer;transition:background .14s ease,color .14s ease}',
      '.mb-step-caret:hover{background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-primary)}',
      '.mb-step-params{display:flex;flex-direction:column;gap:6px;padding:7px 1px 2px;border-top:1px dashed var(--dsw-alias-border-l1);margin-top:2px}',
      '.mb-step-foot{display:flex;align-items:center;gap:6px;flex-wrap:wrap}',
      '.mb-step-summary{flex:1 1 auto;min-width:0;font-size:11px;color:var(--dsw-alias-label-tertiary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',

      /* 中栏 tabs */
      '.mb-tabs{display:flex;align-items:center;gap:2px;flex:none;padding:6px 10px 0;border-bottom:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1);overflow-x:auto;scrollbar-width:thin}',
      '.mb-tab{display:inline-flex;align-items:center;justify-content:center;gap:5px;height:30px;padding:0 11px;border:none;border-bottom:2px solid transparent;border-radius:7px 7px 0 0;background:transparent;color:var(--dsw-alias-label-secondary);font:inherit;font-size:12px;cursor:pointer;white-space:nowrap;transition:color .14s ease,background .14s ease,border-color .14s ease}',
      '.mb-tab:hover{color:var(--dsw-alias-label-primary);background:var(--dsw-alias-bg-layer-2)}',
      '.mb-tab-on{color:var(--dsw-alias-brand-primary);border-bottom-color:var(--dsw-alias-brand-primary);font-weight:600}',
      '.mb-pane{display:flex;flex:1 1 auto;flex-direction:column;gap:11px;min-height:0;padding:12px 14px 18px;overflow-y:auto}',
      '.mb-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;flex:none;min-height:28px}',
      '.mb-toolbar-label{font-size:11px;color:var(--dsw-alias-label-tertiary);white-space:nowrap}',
      '.mb-spacer{flex:1 1 auto}',

      /* 卡片 / 指标 / 区块 */
      '.mb-card{display:flex;flex-direction:column;gap:8px;padding:11px 13px;border:1px solid var(--dsw-alias-border-l1);border-radius:12px;background:var(--dsw-alias-bg-layer-1);box-shadow:0 1px 2px color-mix(in srgb,var(--dsw-alias-label-primary) 4%,transparent)}',
      '.mb-card-title{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11.5px;font-weight:600;color:var(--dsw-alias-label-secondary);letter-spacing:.02em}',
      '.mb-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(104px,1fr));gap:8px}',
      '.mb-metric{display:flex;flex-direction:column;gap:1px;padding:8px 11px;border:1px solid var(--dsw-alias-border-l1);border-radius:10px;background:var(--dsw-alias-bg-layer-1)}',
      '.mb-metric-k{font-size:10.5px;color:var(--dsw-alias-label-tertiary);letter-spacing:.03em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-metric-v{font-size:18px;font-weight:300;line-height:1.25;color:var(--dsw-alias-label-primary);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:11px;align-items:start}',
      '.mb-section-title{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11px;font-weight:600;letter-spacing:.04em;color:var(--dsw-alias-label-secondary);margin-top:2px}',
      '.mb-muted{font-size:11.5px;color:var(--dsw-alias-label-secondary);line-height:1.5}',
      '.mb-inline-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;min-width:0}',
      '.mb-inline-err{padding:5px 8px;border-radius:6px;border:1px solid color-mix(in srgb,var(--dsw-alias-state-error-primary) 40%,transparent);background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 8%,transparent);color:var(--dsw-alias-state-error-primary);font-size:11.5px;word-break:break-word}',
      '.mb-warn-text{color:var(--dsw-alias-state-warn-primary);font-size:11.5px}',
      '.mb-ok-text{color:var(--dsw-alias-state-success-primary);font-size:11.5px}',
      '.mb-divider{flex:none;height:1px;margin:2px 0;background:var(--dsw-alias-border-l1)}',
      '.mb-path{font-size:10.5px;color:var(--dsw-alias-label-tertiary);word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}',
      '.mb-tag{display:inline-flex;align-items:center;height:18px;padding:0 7px;border:1px solid var(--dsw-alias-border-l1);border-radius:999px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);font-size:10.5px;white-space:nowrap}',
      '.mb-tag-ok{border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary) 40%,var(--dsw-alias-border-l1));color:var(--dsw-alias-state-success-primary)}',
      '.mb-tag-warn{border-color:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 40%,var(--dsw-alias-border-l1));color:var(--dsw-alias-state-warn-primary)}',

      /* 图 */
      '.mb-figure{display:flex;flex-direction:column;gap:6px;min-width:0}',
      '.mb-figure-title{font-size:11px;color:var(--dsw-alias-label-secondary);font-weight:500}',
      '.mb-figure img{display:block;max-width:100%;height:auto;border:1px solid var(--dsw-alias-border-l1);border-radius:10px;background:var(--dsw-alias-bg-layer-2)}',
      '.mb-figure-meta{font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',
      '.mb-figure-empty{display:flex;align-items:center;justify-content:center;min-height:120px;border:1px dashed var(--dsw-alias-border-l2);border-radius:10px;background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-tertiary);font-size:11.5px;padding:12px;text-align:center}',

      /* 表格 */
      '.mb-table-wrap{max-height:420px;overflow:auto;border:1px solid var(--dsw-alias-border-l1);border-radius:10px;background:var(--dsw-alias-bg-layer-1)}',
      '.mb-table{width:100%;border-collapse:separate;border-spacing:0;font-size:11.5px}',
      '.mb-table th{position:sticky;top:0;z-index:1;padding:6px 10px;border-bottom:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-secondary);font-weight:600;text-align:left;white-space:nowrap}',
      '.mb-table th.mb-th-sort{cursor:pointer;user-select:none}',
      '.mb-table th.mb-th-sort:hover{color:var(--dsw-alias-label-primary)}',
      '.mb-table td{padding:4px 10px;border-bottom:1px solid color-mix(in srgb,var(--dsw-alias-border-l1) 55%,transparent);color:var(--dsw-alias-label-primary);white-space:nowrap;max-width:340px;overflow:hidden;text-overflow:ellipsis}',
      '.mb-table tbody tr:hover td{background:color-mix(in srgb,var(--dsw-alias-brand-primary) 4%,transparent)}',
      '.mb-td-num{text-align:right;font-variant-numeric:tabular-nums}',
      '.mb-table-note{padding:5px 2px 0;font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',
      '.mb-empty-table{padding:18px 12px;text-align:center;color:var(--dsw-alias-label-tertiary);font-size:11.5px}',

      /* 结果视图 */
      '.mb-result{display:flex;flex-direction:column;gap:10px}',
      '.mb-bullets{margin:0;padding-left:16px;display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--dsw-alias-label-primary)}',
      '.mb-bullets li::marker{color:var(--dsw-alias-brand-primary)}',
      '.mb-result-block{display:flex;flex-direction:column;gap:6px}',
      '.mb-result-block-title{font-size:11px;font-weight:600;color:var(--dsw-alias-label-secondary);letter-spacing:.03em}',
      '.mb-warn-list{display:flex;flex-direction:column;gap:3px;font-size:11.5px;color:var(--dsw-alias-state-warn-primary)}',
      '.mb-note-list{display:flex;flex-direction:column;gap:3px;font-size:11.5px;color:var(--dsw-alias-label-secondary)}',
      '.mb-file-chip{display:inline-flex;align-items:center;gap:6px;max-width:100%;padding:4px 8px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-layer-2);font-size:11px;color:var(--dsw-alias-label-secondary)}',
      '.mb-file-chip .mb-path{flex:1 1 auto;min-width:0}',

      /* 引导 / 空态 / 崩溃 */
      '.mb-guide{display:flex;flex-direction:column;align-items:flex-start;gap:9px;justify-content:center;padding:22px 20px;border:1.5px dashed var(--dsw-alias-border-l2);border-radius:12px;background:var(--dsw-alias-bg-layer-1);text-align:left}',
      '.mb-guide-title{font-size:13.5px;font-weight:600}',
      '.mb-guide-body{font-size:12px;color:var(--dsw-alias-label-secondary);max-width:640px;line-height:1.55}',
      '.mb-guide-hint{font-size:11.5px;color:var(--dsw-alias-label-tertiary);max-width:660px;line-height:1.5}',
      '.mb-notice{display:flex;align-items:center;gap:8px;padding:9px 12px;border:1px dashed var(--dsw-alias-border-l2);border-radius:10px;background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-secondary);font-size:11.5px}',
      '.mb-crash{display:flex;flex-direction:column;align-items:flex-start;gap:8px;margin:14px;padding:16px;border:1px solid color-mix(in srgb,var(--dsw-alias-state-error-primary) 40%,var(--dsw-alias-border-l1));border-radius:12px;background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 5%,var(--dsw-alias-bg-layer-1))}',
      '.mb-crash-title{font-size:13px;font-weight:600;color:var(--dsw-alias-state-error-primary)}',
      '.mb-crash-hint{font-size:11.5px;color:var(--dsw-alias-label-secondary)}',
      '.mb-crash-detail{font-size:10.5px;color:var(--dsw-alias-label-tertiary);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}',
      '.mb-crash-mini{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 16%,transparent);color:var(--dsw-alias-state-error-primary);font-size:11px;font-weight:700}',
      '.mb-loading{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--dsw-alias-label-secondary)}',
      '.mb-spinner{display:inline-block;width:13px;height:13px;border-radius:50%;border:1.6px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 30%,transparent);border-top-color:var(--dsw-alias-brand-primary);animation:mb-spin .8s linear infinite;flex:none}',
      '@keyframes mb-spin{to{transform:rotate(360deg)}}',
      '@keyframes mb-pulse{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,var(--dsw-alias-brand-primary) 30%,transparent)}50%{box-shadow:0 0 0 4px color-mix(in srgb,var(--dsw-alias-brand-primary) 6%,transparent)}}',
      '@keyframes mb-fade{from{opacity:0;transform:translateY(2px)}to{opacity:1;transform:none}}',

      /* 弹层 */
      '.mb-modal-mask{position:fixed;inset:0;z-index:60;display:flex;align-items:center;justify-content:center;padding:24px;background:color-mix(in srgb,var(--dsw-alias-label-primary) 34%,transparent);backdrop-filter:blur(3px);animation:mb-fade .14s ease}',
      '.mb-modal{display:flex;flex-direction:column;width:min(920px,94vw);max-height:88vh;border:1px solid var(--dsw-alias-border-l1);border-radius:14px;background:var(--dsw-alias-bg-base);box-shadow:0 24px 80px color-mix(in srgb,var(--dsw-alias-label-primary) 24%,transparent);overflow:hidden}',
      '.mb-modal-wide{width:min(1100px,95vw)}',
      '.mb-modal-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;flex:none;padding:13px 16px;border-bottom:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1)}',
      '.mb-modal-title{font-size:14px;font-weight:600}',
      '.mb-modal-subtitle{font-size:11.5px;color:var(--dsw-alias-label-secondary);margin-top:1px}',
      '.mb-modal-body{display:flex;flex-direction:column;gap:11px;padding:13px 16px 18px;overflow-y:auto;min-height:0}',
      '.mb-modal-foot{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex-wrap:wrap;flex:none;padding:10px 16px;border-top:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1)}',
      '.mb-advanced{margin-top:1px;padding:9px 12px;border:1px dashed var(--dsw-alias-border-l2);border-radius:10px;background:var(--dsw-alias-bg-layer-1)}',
      '.mb-advanced summary{cursor:pointer;font-size:11.5px;font-weight:600;color:var(--dsw-alias-label-secondary)}',
      '.mb-advanced[open] summary{margin-bottom:7px;color:var(--dsw-alias-label-primary)}',
      '.mb-form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px 10px}',
      '.mb-form-grid .mb-field{flex-direction:column;align-items:stretch;gap:3px}',
      '.mb-form-grid .mb-input{max-width:none;width:100%}',

      /* GEO 检索 */
      '.mb-geo-search{display:flex;align-items:center;gap:7px;flex-wrap:wrap}',
      '.mb-geo-term{flex:1 1 300px;min-width:220px;height:32px;font-size:12.5px}',
      '.mb-geo-filters{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:11px;color:var(--dsw-alias-label-secondary)}',
      '.mb-geo-layout{display:grid;grid-template-columns:minmax(260px,340px) 1fr;gap:12px;min-height:0;flex:1 1 auto}',
      '.mb-geo-list{display:flex;flex-direction:column;gap:7px;overflow-y:auto;max-height:50vh;padding-right:2px}',
      '.mb-geo-card{display:flex;flex-direction:column;gap:4px;padding:9px 11px;border:1px solid var(--dsw-alias-border-l1);border-radius:10px;background:var(--dsw-alias-bg-layer-1);cursor:pointer;text-align:left;font:inherit;transition:border-color .14s ease,background .14s ease,box-shadow .14s ease}',
      '.mb-geo-card:hover{border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2)}',
      '.mb-geo-card-on{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 50%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 6%,var(--dsw-alias-bg-layer-1));box-shadow:0 0 0 1px color-mix(in srgb,var(--dsw-alias-brand-primary) 24%,transparent)}',
      '.mb-geo-gse{font-size:12px;font-weight:600;color:var(--dsw-alias-brand-primary)}',
      '.mb-geo-title{font-size:12px;color:var(--dsw-alias-label-primary);line-height:1.4}',
      '.mb-geo-title-en{font-size:10.5px;color:var(--dsw-alias-label-tertiary);line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}',
      '.mb-geo-meta{display:flex;flex-wrap:wrap;align-items:center;gap:4px 7px;font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',
      '.mb-geo-tag{padding:0 6px;height:17px;display:inline-flex;align-items:center;border-radius:999px;border:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-2);font-size:10px;color:var(--dsw-alias-label-secondary)}',
      '.mb-geo-tag-ok{border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary) 40%,var(--dsw-alias-border-l1));color:var(--dsw-alias-state-success-primary)}',
      '.mb-geo-tag-high{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 44%,var(--dsw-alias-border-l1));color:var(--dsw-alias-brand-primary)}',
      '.mb-geo-tag-mid{border-color:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 40%,var(--dsw-alias-border-l1));color:var(--dsw-alias-state-warn-primary)}',
      '.mb-geo-tag-low{color:var(--dsw-alias-label-tertiary)}',
      '.mb-geo-detail{display:flex;flex-direction:column;gap:9px;min-height:0;overflow-y:auto;padding:1px 2px 4px;max-height:54vh}',
      '.mb-geo-detail-title{font-size:13px;font-weight:600;line-height:1.4}',
      '.mb-geo-kv{display:grid;grid-template-columns:84px 1fr;gap:3px 8px;font-size:11.5px}',
      '.mb-geo-kv dt{color:var(--dsw-alias-label-tertiary)}',
      '.mb-geo-kv dd{margin:0;color:var(--dsw-alias-label-primary);word-break:break-word}',
      '.mb-geo-file{display:flex;align-items:center;gap:7px;padding:5px 8px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-base);font-size:11.5px}',
      '.mb-geo-file-name{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-progress{display:flex;flex-direction:column;gap:4px;min-width:180px;flex:1 1 auto}',
      '.mb-progress-bar{position:relative;height:7px;border-radius:999px;border:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-2);overflow:hidden}',
      '.mb-progress-fill{position:absolute;left:0;top:0;bottom:0;border-radius:999px;background:linear-gradient(90deg,color-mix(in srgb,var(--dsw-alias-brand-primary) 70%,transparent),var(--dsw-alias-brand-primary));transition:width .2s ease}',
      '.mb-progress-text{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',

      /* 分析目录 / 运行器 */
      '.mb-analysis-wrap{display:grid;grid-template-columns:minmax(240px,300px) 1fr;gap:12px;align-items:start;min-height:0}',
      '.mb-analysis-list{display:flex;flex-direction:column;gap:8px;max-height:64vh;overflow-y:auto;padding-right:2px}',
      '.mb-analysis-group{display:flex;flex-direction:column;gap:5px}',
      '.mb-analysis-group-title{font-size:10.5px;font-weight:600;letter-spacing:.05em;color:var(--dsw-alias-label-tertiary)}',
      '.mb-analysis-item{display:flex;flex-direction:column;gap:2px;padding:7px 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;background:var(--dsw-alias-bg-layer-1);cursor:pointer;text-align:left;font:inherit;transition:border-color .14s ease,background .14s ease}',
      '.mb-analysis-item:hover:not(:disabled){border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2)}',
      '.mb-analysis-item:disabled{opacity:.55;cursor:not-allowed}',
      '.mb-analysis-item-on{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 48%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 6%,var(--dsw-alias-bg-layer-1))}',
      '.mb-analysis-main{display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0}',
      '.mb-analysis-name{font-size:12px;font-weight:500;color:var(--dsw-alias-label-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.mb-analysis-method{font-size:10.5px;color:var(--dsw-alias-label-tertiary);line-height:1.4;word-break:break-word}',
      '.mb-analysis-reason{font-size:10.5px;color:var(--dsw-alias-state-warn-primary);line-height:1.4}',
      '.mb-analysis-runner{display:flex;flex-direction:column;gap:10px;min-width:0}',
      '.mb-analysis-head{display:flex;flex-direction:column;gap:2px;min-width:0}',
      '.mb-analysis-title{font-size:13.5px;font-weight:600;line-height:1.35}',

      /* 模型质量（R²Y / Q² / 置换 p） */
      '.mb-quality{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:8px 11px;border:1px solid var(--dsw-alias-border-l1);border-radius:10px;background:var(--dsw-alias-bg-layer-1)}',
      '.mb-quality-item{display:flex;align-items:baseline;gap:5px}',
      '.mb-quality-k{font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',
      '.mb-quality-v{font-size:16px;font-weight:400;color:var(--dsw-alias-label-primary);font-variant-numeric:tabular-nums}',

      /* 质控小图（前端 SVG 散点：TIC × 缺失率） */
      '.mb-minimap{display:flex;align-items:center;justify-content:center;padding:6px;border:1px solid var(--dsw-alias-border-l1);border-radius:10px;background:var(--dsw-alias-bg-layer-2)}',
      '.mb-minimap svg{display:block;width:100%;height:auto;max-height:240px}',

      /* 谱库 / 路径输入行 */
      '.mb-library-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}',
      '.mb-library-row .mb-input{flex:1 1 260px}',
      '.mb-library-hint{font-size:10.5px;color:var(--dsw-alias-label-tertiary);line-height:1.45}',

      /* 候选文件列表（导入 ambiguous 时点选） */
      '.mb-candidates{display:flex;flex-direction:column;gap:4px}',
      '.mb-candidate{display:flex;align-items:center;gap:6px;padding:5px 8px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font:inherit;font-size:11px;cursor:pointer;text-align:left;word-break:break-all}',
      '.mb-candidate:hover{border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2)}',
      '.mb-candidate-on{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 46%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 6%,var(--dsw-alias-bg-layer-1))}',

      /* ML 结果卡（MetaboLights 用同一个列表样式，前缀区分编号） */
      '.mb-ml-accession{font-size:12px;font-weight:600;color:var(--dsw-alias-brand-primary)}',
      '.mb-ml-title{font-size:12px;color:var(--dsw-alias-label-primary);line-height:1.4}',
      '.mb-ml-meta{display:flex;flex-wrap:wrap;align-items:center;gap:4px 7px;font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',

      /* 导出按钮组 */
      '.mb-export-grid{display:flex;flex-wrap:wrap;gap:6px}',
      '.mb-export-item{display:flex;flex-direction:column;gap:2px;align-items:flex-start;min-width:118px;padding:7px 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-primary);font:inherit;font-size:11.5px;cursor:pointer;text-align:left;transition:border-color .14s ease,background .14s ease}',
      '.mb-export-item:hover:not(:disabled){border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2)}',
      '.mb-export-item:disabled{opacity:.5;cursor:default}',
      '.mb-export-hint{font-size:10px;color:var(--dsw-alias-label-tertiary)}',

      /* 右栏 Agent 面板 */
      '.mb-agent-section{display:flex;flex-direction:column;gap:6px;padding:10px 12px;border-bottom:1px solid var(--dsw-alias-border-l1)}',
      '.mb-agent-title{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:10.5px;font-weight:600;letter-spacing:.05em;color:var(--dsw-alias-label-tertiary)}',
      '.mb-suggestion{display:flex;flex-direction:column;gap:4px;padding:8px 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;background:var(--dsw-alias-bg-layer-2);animation:mb-fade .16s ease}',
      '.mb-suggestion-title{font-size:11.5px;font-weight:600;color:var(--dsw-alias-label-primary);line-height:1.4}',
      '.mb-suggestion-detail{font-size:10.5px;color:var(--dsw-alias-label-tertiary);line-height:1.45}',
      '.mb-activity{display:flex;flex-direction:column;gap:6px}',
      '.mb-activity-item{display:flex;gap:7px;font-size:11px;line-height:1.4}',
      '.mb-activity-dot{position:relative;flex:none;width:6px;height:6px;margin-top:4px;border-radius:50%;background:var(--dsw-alias-state-success-primary);box-shadow:0 0 0 2px color-mix(in srgb,var(--dsw-alias-state-success-primary) 16%,transparent)}',
      '.mb-activity-dot-bad{background:var(--dsw-alias-state-error-primary);box-shadow:0 0 0 2px color-mix(in srgb,var(--dsw-alias-state-error-primary) 16%,transparent)}',
      '.mb-activity-main{display:flex;flex-direction:column;gap:1px;min-width:0}',
      '.mb-activity-head{display:flex;align-items:center;gap:5px;flex-wrap:wrap;color:var(--dsw-alias-label-secondary)}',
      '.mb-activity-actor{font-size:9.5px;padding:0 5px;height:15px;display:inline-flex;align-items:center;border-radius:999px;border:1px solid var(--dsw-alias-border-l1);color:var(--dsw-alias-label-tertiary)}',
      '.mb-activity-text{color:var(--dsw-alias-label-tertiary);font-size:10.5px;word-break:break-word}',
      '.mb-stat-row{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11px;color:var(--dsw-alias-label-secondary)}',
      '.mb-stat-row b{font-weight:600;color:var(--dsw-alias-label-primary);font-variant-numeric:tabular-nums;max-width:60%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
      '.mb-history{display:flex;flex-direction:column;gap:4px}',
      '.mb-history-item{display:flex;align-items:center;gap:7px;padding:5px 8px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-layer-1);font-size:11px}',
      '.mb-history-at{flex:none;font-size:10.5px;color:var(--dsw-alias-label-tertiary);font-variant-numeric:tabular-nums}',
      '.mb-history-step{flex:none;font-weight:600;color:var(--dsw-alias-label-secondary)}',
      '.mb-history-summary{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--dsw-alias-label-tertiary)}',
      '.mb-history-ms{flex:none;color:var(--dsw-alias-label-tertiary);font-size:10.5px}',
      '.mb-audit-item{display:flex;align-items:flex-start;gap:7px;padding:6px 8px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-layer-1);font-size:11px}',
      '.mb-audit-item-warn{border-color:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 36%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 5%,var(--dsw-alias-bg-layer-1))}',
      '.mb-audit-item-fail{border-color:color-mix(in srgb,var(--dsw-alias-state-error-primary) 36%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 5%,var(--dsw-alias-bg-layer-1))}',
      '.mb-audit-mark{flex:none;width:16px;height:16px;display:inline-flex;align-items:center;justify-content:center;border-radius:50%;font-size:9.5px;font-weight:700;background:color-mix(in srgb,var(--dsw-alias-state-success-primary) 16%,transparent);color:var(--dsw-alias-state-success-primary)}',
      '.mb-audit-mark-warn{background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 16%,transparent);color:var(--dsw-alias-state-warn-primary)}',
      '.mb-audit-mark-fail{background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 16%,transparent);color:var(--dsw-alias-state-error-primary)}',
      '.mb-audit-main{display:flex;flex-direction:column;gap:1px;min-width:0}',
      '.mb-audit-title{font-weight:600;color:var(--dsw-alias-label-primary)}',
      '.mb-audit-detail{color:var(--dsw-alias-label-tertiary);font-size:10.5px;line-height:1.45}',
      '.mb-score{display:flex;align-items:baseline;gap:6px}',
      '.mb-score-v{font-size:30px;font-weight:250;line-height:1;color:var(--dsw-alias-label-primary)}',
      '.mb-score-label{font-size:11px;color:var(--dsw-alias-label-tertiary)}',

      /* 响应式 */
      '@media (max-width:1180px){.mb-left{flex-basis:240px;width:240px}.mb-right{flex-basis:262px;width:262px}}',
      '@media (max-width:980px){',
      '  .mb-left{flex-basis:216px;width:216px}.mb-right{display:none}',
      '  .mb-geo-layout{grid-template-columns:1fr}',
      '  .mb-analysis-wrap{grid-template-columns:1fr}',
      '  .mb-field .mb-input{max-width:120px}',
      '}',
      '@media (prefers-reduced-motion:reduce){.mb-spinner{animation:none}.mb-step-dot-running{animation:none}}',
    ].join('')

    // ==========================================================================
    // 通用小组件：加载 / 错误 / 图标 / 按钮 / 错误边界 / 侧边栏图标
    // ==========================================================================

    function Spinner() {
      return h('span', { className: 'mb-spinner', 'aria-hidden': 'true' })
    }

    function LoadingLine(props) {
      return h('div', { className: 'mb-loading' }, h(Spinner), h('span', null, (props && props.text) || T('loading')))
    }

    function ErrorBanner(props) {
      return h('div', { className: 'mb-crash', role: 'alert' },
        h('div', { className: 'mb-crash-title' }, (props && props.title) || T('errorTitle')),
        props && props.message ? h('div', { className: 'mb-crash-detail' }, String(props.message)) : null,
        props && props.onRetry
          ? h('button', { type: 'button', className: 'mb-btn mb-btn-sm', onClick: props.onRetry },
              (props && props.retryLabel) || T('retry'))
          : null)
    }

    function InlineError(props) {
      if (!props || !props.message) return null
      return h('div', { className: 'mb-inline-err', role: 'alert' }, String(props.message))
    }

    function Notice(props) {
      return h('div', { className: 'mb-notice' }, (props && props.text) || '')
    }

    /** 内联 SVG 图标（不用 emoji；stroke 用 currentColor 跟随主题）。 */
    function Icon(props) {
      var name = (props && props.name) || 'dot'
      var size = (props && props.size) || 14
      var paths = {
        refresh: 'M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7',
        search: 'M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM15.2 15.2 21 21',
        importIcon: 'M12 3v12m0 0 4-4m-4 4-4-4M4 17v3h16v-3',
        download: 'M12 3v12m0 0 4-4m-4 4-4-4M4 20h16',
        close: 'M6 6l12 12M18 6 6 18',
        play: 'M7 5l12 7-12 7z',
        chart: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
        chevron: 'M9 6l6 6-6 6',
        chevronDown: 'M6 9l6 6 6-6',
        copy: 'M9 9h11v11H9zM5 15V4h11',
        check: 'M4 12.5l5 5L20 6.5',
        folder: 'M3 6h6l2 2h10v11H3z',
        plus: 'M12 5v14M5 12h14',
        flask: 'M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3',
        shield: 'M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z',
        clock: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3.2 2',
        database: 'M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6',
        grid: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
        dot: 'M12 12h.01',
      }
      return h('svg', {
        width: size,
        height: size,
        viewBox: '0 0 24 24',
        fill: name === 'play' ? 'currentColor' : 'none',
        stroke: 'currentColor',
        strokeWidth: props && props.strokeWidth ? props.strokeWidth : 1.7,
        strokeLinecap: 'round',
        strokeLinejoin: 'round',
        'aria-hidden': 'true',
        focusable: 'false',
        style: { flex: 'none', display: 'inline-block' },
      }, h('path', { d: paths[name] || paths.dot }))
    }

    function Btn(props) {
      var children = Array.prototype.slice.call(arguments, 1)
      var extra = props || {}
      // 两种调用方式都支持：Btn(props, ...children) 直呼；h(Btn, props, ...children) 经 React。
      if (!children.length && extra.children !== undefined && extra.children !== null && extra.children !== false) {
        children = Array.isArray(extra.children) ? extra.children : [extra.children]
      }
      var className = 'mb-btn' +
        (extra.variant === 'primary' ? ' mb-btn-primary' : '') +
        (extra.variant === 'ghost' ? ' mb-btn-ghost' : '') +
        (extra.variant === 'danger' ? ' mb-btn-danger' : '') +
        (extra.small ? ' mb-btn-sm' : '') +
        (extra.className ? ' ' + extra.className : '')
      return h('button', {
        key: extra.key,
        type: 'button',
        className: className,
        title: extra.title || undefined,
        disabled: extra.disabled === true,
        onClick: extra.onClick,
      }, extra.icon ? h(Icon, { name: extra.icon, size: extra.small ? 12 : 14 }) : null, children.filter(Boolean))
    }

    /** 渲染崩溃兜底：每个主要区块都包一层，避免整页白屏。 */
    class PluginErrorBoundary extends React.Component {
      constructor(props) {
        super(props)
        this.state = { error: null }
        this.handleReset = this.handleReset.bind(this)
      }

      static getDerivedStateFromError(error) {
        return { error: error || new Error('unknown render error') }
      }

      componentDidCatch(error, info) {
        warnPlugin('UI crashed: ' + formatError(error), info)
      }

      handleReset() {
        this.setState({ error: null })
      }

      render() {
        if (this.state.error) {
          if (this.props && this.props.mini) {
            return h('span', { className: 'mb-crash-mini', title: T('crashTitle') }, '!')
          }
          return h('div', { className: 'mb-crash', role: 'alert' },
            h('div', { className: 'mb-crash-title' }, T('crashTitle')),
            h('div', { className: 'mb-crash-hint' }, T('crashHint')),
            h('div', { className: 'mb-crash-detail' }, formatError(this.state.error)),
            h('button', { type: 'button', className: 'mb-btn mb-btn-primary mb-btn-sm', onClick: this.handleReset }, T('retry')))
        }
        return this.props ? this.props.children : null
      }
    }

    // ==========================================================================
    // 侧边栏图标（纯 SVG：色谱峰 + 特征点，代谢质谱语义）
    // ==========================================================================

    function SidebarGlyph(props) {
      var size = (props && props.size) || 18
      return h('svg', {
        width: size,
        height: size,
        viewBox: '0 0 24 24',
        fill: 'none',
        stroke: 'currentColor',
        strokeWidth: 1.6,
        strokeLinecap: 'round',
        strokeLinejoin: 'round',
        'aria-hidden': 'true',
        focusable: 'false',
      },
        h('path', { d: 'M3 19.5h18' }),
        h('path', { d: 'M5 19.5v-4.2M8.2 19.5V9.4M11.4 19.5v-7.1M14.6 19.5V5.6M17.8 19.5v-5.3' }),
        h('circle', { cx: 14.6, cy: 5.6, r: 1.15, fill: 'currentColor', stroke: 'none' }),
        h('circle', { cx: 8.2, cy: 9.4, r: 1.05, fill: 'currentColor', stroke: 'none' }))
    }


    function SidebarIcon(props) {
      return h(PluginErrorBoundary, { mini: true }, h(SidebarGlyph, { size: props && props.size }))
    }

    // ==========================================================================
    // 通用展示组件：指标卡 / 卡片 / 图 / 表 / 结果信封
    // ==========================================================================

    function MetricCard(props) {
      return h('div', { className: 'mb-metric', title: props.hint || undefined },
        h('div', { className: 'mb-metric-k' }, props.label),
        h('div', { className: 'mb-metric-v' }, props.value, props.unit ? h('small', null, props.unit) : null))
    }

    function SectionCard(props) {
      return h('div', { className: 'mb-card' },
        props.title || props.extra
          ? h('div', { className: 'mb-card-title' },
              h('span', null, props.title || ''),
              props.extra ? h('span', null, props.extra) : null)
          : null,
        props.children)
    }

    /** 单张图：加载 / 失败 / 空态 / 图片（点击在新标签打开原图）。 */
    function FigureBlock(props) {
      var brokenState = useState(false)
      var broken = brokenState[0]
      var setBroken = brokenState[1]
      var figure = props.figure || null
      var url = figure && figure.url ? figure.url : ''
      useEffect(function () {
        setBroken(false)
      }, [url])
      if (props.loading) {
        return h('div', { className: 'mb-figure' },
          props.title ? h('div', { className: 'mb-figure-title' }, props.title) : null,
          h('div', { className: 'mb-figure-empty' }, h(LoadingLine, { text: props.loadingText || T('plotting') })))
      }
      if (props.error) {
        return h('div', { className: 'mb-figure' },
          props.title ? h('div', { className: 'mb-figure-title' }, props.title) : null,
          h('div', { className: 'mb-inline-err' }, props.error),
          props.onReload ? h('div', null, Btn({ small: true, icon: 'refresh', onClick: props.onReload }, T('retry'))) : null)
      }
      if (!url) {
        return h('div', { className: 'mb-figure' },
          props.title ? h('div', { className: 'mb-figure-title' }, props.title) : null,
          h('div', { className: 'mb-figure-empty' }, props.emptyText || T('noFigure')))
      }
      return h('div', { className: 'mb-figure' },
        props.title ? h('div', { className: 'mb-figure-title' }, props.title) : null,
        broken
          ? h('div', { className: 'mb-figure-empty' }, T('figureFail'))
          : h('a', {
              href: url,
              target: '_blank',
              rel: 'noopener noreferrer',
              title: T('plot'),
              style: { display: 'block' },
            }, h('img', {
              src: url,
              alt: figure.title || props.title || 'figure',
              loading: 'lazy',
              onError: function () { setBroken(true) },
            })),
        figure.file && figure.file !== url ? h('div', { className: 'mb-figure-meta', title: figure.file }, figure.file) : null,
        props.onReload && !broken ? h('div', null, Btn({ small: true, variant: 'ghost', icon: 'refresh', onClick: props.onReload }, T('retry'))) : null)
    }

    /** 自动加载 + 缓存展示一张服务端图（固定参数用；动态参数由各 tab 自己管）。 */
    function PlotBlock(props) {
      var key = plotKey(props.kind, props.params)
      var entry = (props.plots && props.plots[key]) || null
      var loadPlot = props.loadPlot
      var enabled = props.enabled !== false
      useEffect(function () {
        if (!enabled || typeof loadPlot !== 'function') return
        if (entry && (entry.loading || entry.figure || entry.error)) return
        loadPlot(props.kind, props.params)
      }, [enabled, key, entry ? (entry.loading ? 2 : (entry.figure ? 1 : 3)) : 0])
      var reload = function () { if (typeof loadPlot === 'function') loadPlot(props.kind, props.params, true) }
      return h(FigureBlock, {
        title: props.title,
        figure: entry && entry.figure ? entry.figure : null,
        loading: !!(entry && entry.loading),
        error: entry && entry.error ? entry.error : '',
        emptyText: props.emptyText || T('noFigure'),
        onReload: enabled ? reload : null,
      })
    }

    /** 服务端表格：列头可排序，最多渲染 MAX_TABLE_RENDER_ROWS 行。 */
    function DataTable(props) {
      var table = props.table || { columns: [], rows: [] }
      var sortState = useState({ key: '', dir: 'asc' })
      var sort = sortState[0]
      var setSort = sortState[1]
      var limit = isFinite(Number(props.maxRows)) ? Number(props.maxRows) : MAX_TABLE_RENDER_ROWS
      var rows = Array.isArray(table.rows) ? table.rows : []
      var view
      if (props.sortable !== false) {
        var ordered = sort.key ? sortRows(rows, sort.key, sort.dir, table.columns) : rows
        view = formatTableRows({ columns: table.columns, rows: ordered, truncated: table.truncated }, limit)
      } else {
        view = formatTableRows(table, limit)
      }
      function clickHeader(column) {
        if (props.sortable === false) return
        setSort(function (previous) {
          if (previous.key === column.key) return { key: column.key, dir: previous.dir === 'asc' ? 'desc' : 'asc' }
          return { key: column.key, dir: 'asc' }
        })
      }
      if (!view.columns.length) {
        return h('div', { className: 'mb-empty-table' }, props.emptyText || T('emptyTable'))
      }
      return h('div', null,
        h('div', { className: 'mb-table-wrap', style: props.maxHeight ? { maxHeight: props.maxHeight } : undefined },
          h('table', { className: 'mb-table' },
            h('thead', null,
              h('tr', null, view.columns.map(function (column) {
                var active = sort.key === column.key
                return h('th', {
                  key: column.key,
                  className: (props.sortable === false ? '' : 'mb-th-sort') + (column.type === 'number' ? ' mb-td-num' : ''),
                  onClick: function () { clickHeader(column) },
                }, column.label + (active ? (sort.dir === 'asc' ? ' ↑' : ' ↓') : ''))
              }))),
            h('tbody', null, view.rows.map(function (row) {
              return h('tr', { key: row.key }, row.cells.map(function (cell) {
                return h('td', { key: cell.key, className: cell.numeric ? 'mb-td-num' : '', title: cell.text }, cell.text)
              }))
            })))),
        h('div', { className: 'mb-table-note' },
          tf(T, 'tableRows', view.total) +
          (view.truncated ? ' · ' + tf(T, 'tableTruncated', view.shown) : '') +
          (table.note ? ' · ' + table.note : '')))
    }

    /**
     * 通用结果信封渲染（DESIGN §2.5）：bullets / metrics / tables / figures /
     * warnings / notes / data_out。标题优先用服务端文案，缺省回落字典词条。
     */
    function ResultView(props) {
      var result = props.result || null
      if (!result) {
        return props.emptyText ? h('div', { className: 'mb-notice' }, props.emptyText) : null
      }
      var blocks = []
      if (typeof props.onExport === 'function') {
        blocks.push(h('div', { key: 'export', className: 'mb-toolbar' },
          Btn({ small: true, icon: 'download', disabled: props.exportBusy === true, onClick: props.onExport },
            props.exportBusy ? T('recordsExporting') : (props.exportLabel || T('exportBtn'))),
          props.exportFile ? h('span', { className: 'mb-file-chip' }, h('span', { className: 'mb-path' }, props.exportFile)) : null))
      }
      var bullets = Array.isArray(result.bullets) ? result.bullets : []
      if (bullets.length) {
        blocks.push(h('div', { key: 'bullets', className: 'mb-card' },
          h('div', { className: 'mb-card-title' }, h('span', null, result.title || T('analysisResult'))),
          h('ul', { className: 'mb-bullets' }, bullets.map(function (bullet, index) {
            return h('li', { key: 'b-' + index }, bullet)
          }))))
      }
      var metrics = result.metrics && typeof result.metrics === 'object' ? result.metrics : {}
      var metricKeys = Object.keys(metrics)
      if (metricKeys.length) {
        blocks.push(h('div', { key: 'metrics', className: 'mb-metrics' }, metricKeys.map(function (key) {
          var value = metrics[key]
          return h(MetricCard, {
            key: key,
            label: key,
            value: typeof value === 'number' ? fmtNumber(value) : formatCell(value, false),
          })
        })))
      }
      if (result.tables && result.tables.length) {
        blocks.push(h('div', { key: 'tables', className: 'mb-card' },
          h('div', { className: 'mb-card-title' },
            h('span', null, T('resultColumn')),
            h('span', null, String(result.tables.length))),
          result.tables.map(function (table, index) {
            return h('div', { key: 'table-' + (table.id || index), className: 'mb-result-block' },
              h('div', { className: 'mb-figure-title' }, table.title || tf(T, 'tableRows', (table.rows || []).length)),
              h(DataTable, { table: table, maxRows: props.maxTableRows || MAX_TABLE_RENDER_ROWS }))
          })))
      }
      if (result.figures && result.figures.length) {
        blocks.push(h('div', { key: 'figures', className: 'mb-grid-2' }, result.figures.map(function (figure, index) {
          return h(FigureBlock, {
            key: 'figure-' + (figure.id || index),
            title: figure.title || T('plot'),
            figure: figure,
          })
        })))
      }
      if (result.warnings && result.warnings.length) {
        blocks.push(h('div', { key: 'warnings', className: 'mb-card' },
          h('div', { className: 'mb-card-title' }, h('span', null, T('warnings')), h('span', null, String(result.warnings.length))),
          h('div', { className: 'mb-warn-list' }, result.warnings.map(function (warning, index) {
            return h('div', { key: 'w-' + index }, '· ' + warning)
          }))))
      }
      if (result.notes && result.notes.length) {
        blocks.push(h('div', { key: 'notes', className: 'mb-card' },
          h('div', { className: 'mb-card-title' }, h('span', null, T('notes')), h('span', null, String(result.notes.length))),
          h('div', { className: 'mb-note-list' }, result.notes.map(function (note, index) {
            return h('div', { key: 'n-' + index }, '· ' + note)
          }))))
      }
      if (result.data_out && result.data_out.length) {
        blocks.push(h('div', { key: 'files', className: 'mb-card' },
          h('div', { className: 'mb-card-title' }, h('span', null, T('pathLabel'))),
          h('div', { className: 'mb-inline-row' }, result.data_out.map(function (item, index) {
            return h('span', { key: 'f-' + index, className: 'mb-file-chip', title: item.path },
              h(Icon, { name: 'download', size: 11 }),
              h('span', null, item.label || item.kind),
              h('span', { className: 'mb-path' }, item.path))
          }))))
      }
      if (!blocks.length) {
        return h('div', { className: 'mb-notice' }, T('stepSummaryEmpty'))
      }
      return h('div', { className: 'mb-result' }, blocks)
    }

    // ==========================================================================
    // 参数表单：按 schema 通用渲染
    //   number / boolean / select / column / multiselect / genes / columns /
    //   text / string / file / json / object
    // ==========================================================================

    function ParamControl(props) {
      var descriptor = props.descriptor || {}
      var value = props.value
      var disabled = props.disabled === true
      var onChange = props.onChange || function () {}
      var type = descriptor.type || 'text'
      if (type === 'number') {
        return h('input', {
          className: 'mb-input',
          type: 'number',
          min: isNil(descriptor.min) ? undefined : descriptor.min,
          max: isNil(descriptor.max) ? undefined : descriptor.max,
          step: isNil(descriptor.step) ? 'any' : descriptor.step,
          value: isNil(value) ? '' : String(value),
          disabled: disabled,
          onChange: function (event) { onChange(event.target.value === '' ? null : Number(event.target.value)) },
        })
      }
      if (type === 'boolean') {
        return h('label', { className: 'mb-check' },
          h('input', {
            type: 'checkbox',
            checked: value === true,
            disabled: disabled,
            onChange: function (event) { onChange(event.target.checked) },
          }),
          h('span', null, value === true ? T('yes') : T('no')))
      }
      if (type === 'select') {
        var options = Array.isArray(descriptor.options) ? descriptor.options : []
        var current = isNil(value) ? '' : String(value)
        var known = false
        for (var optionIndex = 0; optionIndex < options.length; optionIndex++) {
          if (String(options[optionIndex].value) === current) known = true
        }
        var list = options.slice()
        if (current && !known) list.unshift({ value: current, label: current })
        return h('select', {
          className: 'mb-input mb-select',
          value: current,
          disabled: disabled,
          onChange: function (event) { onChange(event.target.value) },
        }, list.map(function (option) {
          return h('option', { key: String(option.value), value: String(option.value) }, option.label)
        }))
      }
      if (type === 'column') {
        var columns = Array.isArray(props.columns) ? props.columns : []
        var columnOptions = [{ value: '', label: T('noSelection') }]
        for (var columnIndex = 0; columnIndex < columns.length; columnIndex++) {
          var column = columns[columnIndex]
          columnOptions.push({
            value: column.key,
            label: column.key + (column.kind && column.kind !== 'categorical' ? ' · ' + column.kind : ''),
          })
        }
        var columnValue = isNil(value) ? '' : String(value)
        var hasColumn = false
        for (var checkIndex = 0; checkIndex < columnOptions.length; checkIndex++) {
          if (columnOptions[checkIndex].value === columnValue) hasColumn = true
        }
        if (columnValue && !hasColumn) columnOptions.push({ value: columnValue, label: columnValue })
        return h('select', {
          className: 'mb-input mb-select',
          value: columnValue,
          disabled: disabled,
          onChange: function (event) { onChange(event.target.value || null) },
        }, columnOptions.map(function (option) {
          return h('option', { key: option.value || 'none', value: option.value }, option.label)
        }))
      }
      if (type === 'multiselect') {
        var optionList = Array.isArray(descriptor.options) ? descriptor.options : []
        var selected = Array.isArray(value) ? value : parseTokenList(value)
        return h('div', { className: 'mb-multi' }, optionList.map(function (option) {
          var on = selected.indexOf(option.value) >= 0
          return h('button', {
            key: String(option.value),
            type: 'button',
            disabled: disabled,
            className: 'mb-multi-opt' + (on ? ' mb-multi-opt-on' : ''),
            onClick: function () {
              var next = selected.slice()
              var index = next.indexOf(option.value)
              if (index >= 0) next.splice(index, 1)
              else next.push(option.value)
              onChange(next)
            },
          }, (on ? '✓ ' : '') + (option.label || option.value))
        }))
      }
      if (type === 'json' || type === 'object') {
        var jsonText = ''
        if (value && typeof value === 'object') {
          try { jsonText = JSON.stringify(value) } catch (ignored) { jsonText = '' }
        } else if (!isNil(value)) {
          jsonText = String(value)
        }
        return h('textarea', {
          className: 'mb-input mb-textarea',
          defaultValue: jsonText,
          disabled: disabled,
          spellCheck: false,
          placeholder: descriptor.placeholder || '{"1":"region A"}',
          onChange: function (event) { onChange(parseJsonParam(event.target.value)) },
        })
      }
      var textValue = isNil(value) ? '' : (Array.isArray(value) ? value.join(', ') : String(value))
      var isList = type === 'genes' || type === 'columns' || type === 'array'
      if (isList) {
        return h('textarea', {
          className: 'mb-input mb-textarea',
          value: textValue,
          disabled: disabled,
          spellCheck: false,
          placeholder: descriptor.placeholder || (type === 'genes' ? 'CDH1, VIM, MKI67' : ''),
          onChange: function (event) { onChange(parseTokenList(event.target.value)) },
        })
      }
      return h('input', {
        className: 'mb-input',
        type: 'text',
        value: textValue,
        disabled: disabled,
        spellCheck: false,
        placeholder: descriptor.placeholder || (type === 'file' ? '/absolute/path' : ''),
        onChange: function (event) { onChange(event.target.value) },
      })
    }

    function ParamForm(props) {
      var params = Array.isArray(props.params) ? props.params : []
      var values = props.values && typeof props.values === 'object' ? props.values : {}
      var onChange = props.onChange || function () {}
      var columns = Array.isArray(props.columns) ? props.columns : []
      if (!params.length) return h('div', { className: 'mb-muted' }, T('analysisNoParams'))
      return h('div', { className: 'mb-form' }, params.map(function (descriptor) {
        if (!descriptor || !descriptor.key) return null
        var label = descriptor.label || descriptor.key
        var control = h(ParamControl, {
          key: 'control-' + descriptor.key,
          descriptor: descriptor,
          columns: columns,
          value: values[descriptor.key],
          disabled: props.disabled,
          onChange: function (next) {
            var patch = {}
            patch[descriptor.key] = next
            onChange(patch)
          },
        })
        var full = descriptor.type === 'multiselect' || descriptor.type === 'genes' ||
          descriptor.type === 'columns' || descriptor.type === 'array' ||
          descriptor.type === 'json' || descriptor.type === 'object' || descriptor.type === 'boolean'
        return h('div', { key: descriptor.key, className: 'mb-field' + (full ? ' mb-field-col' : '') },
          h('span', { className: 'mb-field-label', title: label }, label),
          control)
      }))
    }

    function Modal(props) {
      function onMask(event) {
        if (event.target === event.currentTarget && !props.busy && props.onClose) props.onClose()
      }
      return h('div', { className: 'mb-modal-mask', onMouseDown: onMask },
        h('div', { className: 'mb-modal' + (props.wide ? ' mb-modal-wide' : '') },
          h('div', { className: 'mb-modal-head' },
            h('div', null,
              h('div', { className: 'mb-modal-title' }, props.title || ''),
              props.subtitle ? h('div', { className: 'mb-modal-subtitle' }, props.subtitle) : null),
            h('button', {
              type: 'button',
              className: 'mb-btn mb-btn-sm mb-btn-ghost',
              disabled: props.busy === true,
              title: T('close'),
              onClick: props.onClose,
            }, h(Icon, { name: 'close', size: 13 }))),
          props.children,
          props.footer ? h('div', { className: 'mb-modal-foot' }, props.footer) : null))
    }

    function GuideBlock(props) {
      return h('div', { className: 'mb-guide' },
        h('div', { className: 'mb-guide-title' }, props.title || ''),
        props.body ? h('div', { className: 'mb-guide-body' }, props.body) : null,
        props.hint ? h('div', { className: 'mb-guide-hint' }, props.hint) : null,
        props.actions ? h('div', { className: 'mb-inline-row' }, props.actions) : null)
    }

    /** 弹层里的字段行。 */
    function FormField(props) {
      return h('label', { className: 'mb-field' + (props.full ? ' mb-field-col' : '') },
        h('span', { className: 'mb-field-label' }, props.label),
        props.children)
    }

    // ==========================================================================
    // 顶栏：品牌 + 引擎徽章 + 数据集 chips + 四个动作
    // ==========================================================================

    function TopBar(props) {
      var badge = engineBadge(props.health)
      var datasets = Array.isArray(props.datasets) ? props.datasets : []
      var tone = props.health && props.health.ready ? 'ok' : (props.health && props.health.error ? 'bad' : 'warn')
      return h('header', { className: 'mb-top' },
        h('div', { className: 'mb-brand' },
          h('span', { className: 'mb-brand-mark' }, h(SidebarGlyph, { size: 17 })),
          h('div', { className: 'mb-title-block' },
            h('div', { className: 'mb-title', title: T('appTitle') }, T('appTitle')),
            h('div', { className: 'mb-subtitle', title: T('appSubtitle') }, T('appSubtitle')))),
        h('span', { className: 'mb-badge mb-badge-' + tone, title: badge.detail || badge.reason || '' },
          h('span', { className: 'mb-badge-dot' }),
          T(badge.labelKey) + (badge.detail ? ' · ' + badge.detail : '')),
        !badge.ready && badge.reason
          ? h('span', { className: 'mb-badge-reason', title: badge.reason }, badge.reason)
          : null,
        h('div', { className: 'mb-chips' }, datasets.map(function (dataset) {
          var on = dataset.id === props.activeId
          var title = (dataset.path || dataset.name || dataset.id) +
            (dataset.n_samples ? ' · ' + fmtNumber(dataset.n_samples) + ' ' + T('samples') : '')
          return h('span', { key: dataset.id, className: 'mb-chip' + (on ? ' mb-chip-on' : ''), title: title },
            h('button', {
              type: 'button',
              className: 'mb-chip-label',
              disabled: props.busy === true,
              title: on ? T('current') : T('selectDataset'),
              onClick: function () { if (!on && props.onSelect) props.onSelect(dataset.id) },
            }, dataset.name || dataset.id),
            on && props.onCloseDataset
              ? h('button', {
                  type: 'button',
                  className: 'mb-chip-x',
                  title: T('closeDataset'),
                  disabled: props.busy === true,
                  onClick: function (event) {
                    event.stopPropagation()
                    props.onCloseDataset(dataset.id)
                  },
                }, '×')
              : null)
        })),
        h('div', { className: 'mb-top-actions' },
          Btn({ small: true, icon: 'search', onClick: props.onGeo, disabled: props.busy === true }, T('searchGeo')),
          Btn({
            small: true,
            icon: 'database',
            onClick: props.onMl,
            disabled: props.busy === true || props.metabolights === false,
            title: props.metabolights === false ? T('mlUnavailable') : T('mlTitle'),
          }, T('searchMl')),
          Btn({ small: true, icon: 'importIcon', onClick: props.onImport, disabled: props.busy === true }, T('importData')),
          Btn({ small: true, icon: 'flask', onClick: props.onDemo, disabled: props.busy === true, title: T('demoLoad') }, T('demo')),
          Btn({ small: true, icon: 'refresh', onClick: props.onRefresh, disabled: props.refreshing === true, title: T('refreshHint') }, T('refresh'))))
    }

    // ==========================================================================
    // 左栏：数据集摘要卡 + 12 张流程卡（步骤来自 STEPS）
    // ==========================================================================

    /** 数据集摘要卡：样本 × 特征、极性、注释覆盖、分组列、已完成步骤。 */
    function DatasetCard(props) {
      var dataset = props.dataset
      if (!dataset) {
        return h('div', { className: 'mb-dataset-card' },
          h('div', { className: 'mb-dataset-name' }, T('datasetNoDataset')),
          h('div', { className: 'mb-dataset-meta' }, T('datasetOpenHint')))
      }
      var meta = []
      var group = groupColumn(dataset)
      if (!isNil(dataset.n_samples)) meta.push(h('span', { key: 's' }, T('samples') + ' ', h('b', null, fmtNumber(dataset.n_samples))))
      if (!isNil(dataset.n_features)) meta.push(h('span', { key: 'f' }, T('features') + ' ', h('b', null, fmtNumber(dataset.n_features))))
      if (dataset.polarity) meta.push(h('span', { key: 'p' }, h('b', null, dataset.polarity)))
      if (dataset.feature_meta && !isNil(dataset.feature_meta.n_annotated)) {
        meta.push(h('span', { key: 'a' }, tf(T, 'datasetAnnotated', fmtNumber(dataset.feature_meta.n_annotated))))
      }
      if (dataset.source_format) meta.push(h('span', { key: 'fmt' }, h('b', null, dataset.source_format)))
      if (dataset.species) meta.push(h('span', { key: 'sp' }, h('b', null, dataset.species)))
      return h('div', { className: 'mb-dataset-card' },
        h('div', { className: 'mb-dataset-name', title: dataset.path || dataset.name }, dataset.name || dataset.id),
        h('div', { className: 'mb-dataset-meta' }, meta),
        h('div', { className: 'mb-dataset-meta' },
          h('span', null, tf(T, 'datasetStepsDone', dataset.steps.length)),
          dataset.current ? h('span', null, tf(T, 'datasetCurrentStep', (stepById(dataset.current) || { title: dataset.current }).title)) : null),
        h('div', { className: 'mb-dataset-meta' },
          h('span', null, tf(T, 'datasetGroupColumn', group || T('none'))),
          dataset.feature_meta && dataset.feature_meta.has_mz ? h('span', { className: 'mb-tag' }, 'm/z') : null,
          dataset.feature_meta && dataset.feature_meta.has_rt ? h('span', { className: 'mb-tag' }, 'RT') : null),
        dataset.warnings.length
          ? h('div', { className: 'mb-warn-text', title: dataset.warnings.join('\n') }, T('warnings') + ' · ' + dataset.warnings.length)
          : null)
    }


    function StepCard(props) {
      var step = props.step
      var status = props.status
      var expandedState = useState(status === 'running' || status === 'failed')
      var expanded = expandedState[0]
      var setExpanded = expandedState[1]
      var statusKey = stepStatusKey(status)
      var dotClass = status === 'done'
        ? ' mb-step-dot-done'
        : (status === 'running' ? ' mb-step-dot-running' : (status === 'blocked' ? ' mb-step-dot-blocked' : ' mb-step-dot-pending'))
      var cardClass = 'mb-step' +
        (status === 'done' ? ' mb-step-done' : (status === 'running' ? ' mb-step-running' : (status === 'blocked' ? ' mb-step-blocked' : '')))
      var blockedRequires = []
      var done = doneStepSet(props.dataset)
      var requires = Array.isArray(step.requires) ? step.requires : []
      for (var index = 0; index < requires.length; index++) {
        if (!done[requires[index]]) {
          var required = stepById(requires[index])
          blockedRequires.push(required ? required.title : requires[index])
        }
      }
      var disabled = props.busy === true || !props.dataset || status === 'blocked'
      var summary = props.summary || ''
      var fallback = props.busy ? T('stepRunning')
        : (status === 'blocked' ? T('stepNeedPrereq') + blockedRequires.join('、') : T('stepSummaryEmpty'))
      return h('div', { className: cardClass, 'data-step': step.id },
        h('div', { className: 'mb-step-head' },
          h('span', { className: 'mb-step-dot' + dotClass, title: T(statusKey) }, String(props.index + 1)),
          h('span', { className: 'mb-step-title', title: step.title }, step.title),
          h('span', { className: 'mb-step-status mb-status-' + status }, T(statusKey)),
          h('button', {
            type: 'button',
            className: 'mb-step-caret',
            title: expanded ? T('analysisClose') : T('stepParams'),
            'aria-expanded': expanded ? 'true' : 'false',
            onClick: function () { setExpanded(!expanded) },
          }, h(Icon, { name: expanded ? 'chevronDown' : 'chevron', size: 12 }))),
        expanded
          ? h('div', { className: 'mb-step-params' },
              h(ParamForm, {
                params: step.params,
                values: props.values || {},
                columns: props.columns || [],
                disabled: props.busy === true,
                onChange: props.onParamChange,
              }))
          : null,
        h('div', { className: 'mb-step-foot' },
          Btn({
            small: true,
            variant: status === 'done' ? undefined : 'primary',
            icon: status === 'done' ? 'refresh' : 'play',
            disabled: disabled,
            title: status === 'blocked' ? T('stepNeedPrereq') + blockedRequires.join('、') : T('stepRun'),
            onClick: function () { if (props.onRun) props.onRun(step.id) },
          }, props.busy ? T('stepRunning') : (status === 'done' ? T('stepRerun') : T('stepRun'))),
          h('span', { className: 'mb-step-summary', title: summary || fallback }, summary || fallback)),
        props.error ? h(InlineError, { message: props.error }) : null)
    }

    function PipelinePanel(props) {
      return h('aside', { className: 'mb-left' },
        h('div', { className: 'mb-panel-head' },
          h('span', null, T('datasetsTitle')),
          h('span', { className: 'mb-panel-note' }, T('pipelineHint'))),
        h('div', { className: 'mb-panel-body', style: { paddingBottom: '8px' } },
          h(DatasetCard, { dataset: props.dataset })),
        h('div', { className: 'mb-steps' }, STEPS.map(function (step, index) {
          return h(StepCard, {
            key: step.id,
            step: step,
            index: index,
            status: props.statuses[step.id],
            dataset: props.dataset,
            values: props.values[step.id],
            columns: props.columns,
            summary: props.summaries[step.id],
            error: props.errors[step.id],
            busy: props.busyStep === step.id,
            onParamChange: function (patch) { if (props.onParamChange) props.onParamChange(step.id, patch) },
            onRun: props.onRun,
          })
        })))
    }

    // ==========================================================================
    // 右栏：Agent 面板（状态摘要 / 建议 + 复制指令 / 活动时间线）
    // ==========================================================================

    function AgentPanel(props) {
      var dataset = props.dataset
      var suggestions = Array.isArray(props.suggestions) ? props.suggestions : []
      var activity = Array.isArray(props.activity) ? props.activity : []
      var sections = []
      if (!dataset) {
        sections.push(h('div', { key: 'empty', className: 'mb-agent-section' },
          h('div', { className: 'mb-agent-title' }, T('agentStatus')),
          h('div', { className: 'mb-muted' }, T('agentNoDataset'))))
      } else {
        var currentStep = stepById(dataset.current)
        sections.push(h('div', { key: 'status', className: 'mb-agent-section' },
          h('div', { className: 'mb-agent-title' }, T('agentStatus')),
          h('div', { className: 'mb-stat-row' },
            h('span', null, dataset.name || dataset.id),
            h('b', null, fmtNumber(dataset.n_samples) + ' × ' + fmtNumber(dataset.n_features))),
          h('div', { className: 'mb-stat-row' }, h('span', null, T('completed')), h('b', null, dataset.steps.length + ' / ' + STEPS.length)),
          h('div', { className: 'mb-stat-row' }, h('span', null, T('current')), h('b', null, currentStep ? currentStep.title : T('none'))),
          h('div', { className: 'mb-stat-row' },
            h('span', null, T('datasetGroupColumn', '')),
            h('b', null, groupColumn(dataset) || T('unknown'))),
          h('div', { className: 'mb-stat-row' },
            h('span', null, T('engine')),
            h('b', null, props.health && props.health.ready ? T('engineReady') : T('engineBooting'))),
          h('div', { className: 'mb-muted' }, T('agentShapeHint'))))
      }
      sections.push(h('div', { key: 'suggestions', className: 'mb-agent-section' },
        h('div', { className: 'mb-agent-title' }, T('agentSuggestions')),
        suggestions.length
          ? suggestions.map(function (suggestion) {
              var stepMeta = suggestion.step ? stepById(suggestion.step) : null
              var copied = props.copiedKey === ('suggestion-' + suggestion.id)
              return h('div', { key: suggestion.id, className: 'mb-suggestion' },
                h('div', { className: 'mb-suggestion-title' }, suggestion.title || T('suggestionFallback')),
                suggestion.detail ? h('div', { className: 'mb-suggestion-detail' }, suggestion.detail) : null,
                h('div', { className: 'mb-inline-row' },
                  stepMeta ? h('span', { className: 'mb-tag' }, stepMeta.title) : null,
                  Btn({
                    small: true,
                    variant: 'ghost',
                    icon: 'copy',
                    onClick: function () { if (props.onCopy) props.onCopy(suggestionInstruction(suggestion), 'suggestion-' + suggestion.id) },
                  }, copied ? T('copied') : T('agentCopy'))))
            })
          : h('div', { className: 'mb-muted' }, T('noSuggestions'))))
      sections.push(h('div', { key: 'activity', className: 'mb-agent-section' },
        h('div', { className: 'mb-agent-title' }, T('agentActivity')),
        activity.length
          ? h('div', { className: 'mb-activity' }, activity.slice(-6).reverse().map(function (entry) {
              return h('div', { key: entry.id, className: 'mb-activity-item' },
                h('span', { className: 'mb-activity-dot' + (entry.ok ? '' : ' mb-activity-dot-bad') }),
                h('div', { className: 'mb-activity-main' },
                  h('div', { className: 'mb-activity-head' },
                    h('span', { className: 'mb-activity-actor' }, entry.actor === 'agent' ? T('activityAgent') : T('activityUser')),
                    h('span', null, formatClock(entry.at))),
                  h('div', { className: 'mb-activity-text', title: entry.text }, (entry.action || '—') + (entry.text ? ' · ' + entry.text : ''))))
            }))
          : h('div', { className: 'mb-muted' }, T('agentNoActivity'))))
      return h('aside', { className: 'mb-right' },
        h('div', { className: 'mb-panel-head' }, h('span', null, T('agentPanel')), h('span', { className: 'mb-panel-note' }, T('agentHint'))),
        sections)
    }


    // ==========================================================================
    // 中栏 Tab 框架
    // ==========================================================================

    function TabsBar(props) {
      return h('nav', { className: 'mb-tabs', role: 'tablist' }, TAB_KEYS.map(function (key) {
        return h('button', {
          key: key,
          type: 'button',
          role: 'tab',
          'aria-selected': props.active === key ? 'true' : 'false',
          className: 'mb-tab' + (props.active === key ? ' mb-tab-on' : ''),
          onClick: function () { if (props.onChange) props.onChange(key) },
        }, T(TAB_META[key].titleKey))
      }))
    }

    function TabGuide(props) {
      return h(GuideBlock, {
        title: props.title,
        body: props.body,
        hint: props.hint,
        actions: props.action
          ? Btn({ small: true, variant: 'primary', icon: 'play', onClick: props.action.onClick, disabled: props.action.disabled }, props.action.label)
          : null,
      })
    }

    /**
     * 质控小图：从质控结果表里挑「强度列 × 缺失率列」画前端 SVG 散点。
     * 找不到合适列就不画（提示先跑 qc 步骤），不伪造数据。
     */
    function findScatterTable(tables) {
      var list = Array.isArray(tables) ? tables : []
      for (var index = 0; index < list.length; index++) {
        var table = list[index] || {}
        var columns = Array.isArray(table.columns) ? table.columns : []
        var xKey = ''
        var yKey = ''
        var groupKey = ''
        for (var cursor = 0; cursor < columns.length; cursor++) {
          var column = columns[cursor] || {}
          var key = plainText(column.key)
          if (!key) continue
          if (!groupKey && /^(group|class|condition|batch|qc)$/i.test(key)) groupKey = key
          if (!xKey && column.type === 'number' && /(^|[^a-z])(tic|intensity|abundance|total|mean)$/i.test(key)) xKey = key
          if (!yKey && column.type === 'number' && /(missing|miss|na_frac|nan)/i.test(key)) yKey = key
        }
        if (xKey && yKey && Array.isArray(table.rows) && table.rows.length) {
          return { xKey: xKey, yKey: yKey, groupKey: groupKey, rows: table.rows, title: plainText(table.title || '') }
        }
      }
      return null
    }

    function SampleScatter(props) {
      var data = props.data || null
      if (!data) {
        return h('div', { className: 'mb-figure-empty' }, props.emptyText || T('qcSampleScatterHint'))
      }
      var points = []
      var labels = []
      for (var index = 0; index < data.rows.length; index++) {
        var row = data.rows[index] || {}
        var x = asNumber(row[data.xKey])
        var y = asNumber(row[data.yKey])
        if (x === null || y === null) continue
        points.push([x, y])
        labels.push(row)
      }
      if (!points.length) {
        return h('div', { className: 'mb-figure-empty' }, props.emptyText || T('qcSampleScatterHint'))
      }
      var width = 320
      var height = 220
      var placed = miniPlotScreen(points, null, width, height, { padding: 14, flipY: true })
      return h('div', { className: 'mb-minimap' },
        h('svg', { viewBox: '0 0 ' + width + ' ' + height, width: '100%', height: '220', role: 'img', 'aria-label': T('miniMapTitle') },
          h('line', { x1: 14, y1: height - 14, x2: width - 6, y2: height - 14, stroke: 'currentColor', strokeWidth: 0.6, opacity: 0.35 }),
          h('line', { x1: 14, y1: 6, x2: 14, y2: height - 14, stroke: 'currentColor', strokeWidth: 0.6, opacity: 0.35 }),
          placed.map(function (point, cursor) {
            var row = labels[point.index] || {}
            var color = data.groupKey && !isNil(row[data.groupKey]) ? colorForGroup(row[data.groupKey]) : PALETTE[0]
            return h('circle', {
              key: 'p-' + cursor,
              cx: point.x,
              cy: point.y,
              r: 3.1,
              fill: color,
              opacity: 0.86,
            }, h('title', null, (row.sample || row.name || ('#' + point.index)) + ' · ' + data.xKey + '=' + fmtNumber(row[data.xKey]) + ' · ' + data.yKey + '=' + fmtPct(row[data.yKey])))
          }),
          h('text', { x: width / 2, y: height - 2, textAnchor: 'middle', fontSize: 9, fill: 'currentColor', opacity: 0.6 }, data.xKey),
          h('text', { x: 4, y: 10, fontSize: 9, fill: 'currentColor', opacity: 0.6 }, data.yKey)))
    }

    function OverviewTab(props) {
      var dataset = props.dataset
      // hooks 必须在任何 early return 之前（否则数据集从空到有会触发 hooks 数量变化）
      var colorState = useState('group')
      var colorBy = colorState[0]
      var setColorBy = colorState[1]
      if (!dataset) {
        return h('div', { className: 'mb-pane' },
          h(TabGuide, {
            title: T('datasetNoDataset'),
            body: T('datasetOpenHint'),
            actions: null,
            action: { label: T('importData'), onClick: props.onOpenImport },
          }))
      }
      var health = props.health || {}
      var done = doneStepSet(dataset)
      var pcaReady = done.transform || done.pca
      var engineInfo = health.ready
        ? (health.mode || T('engineReady')) + (health.python ? ' · ' + T('pythonLabel') + ' ' + health.python : '')
        : T('engineBooting')
      var groupKey = groupColumn(dataset)
      var groupCount = null
      var columns = Array.isArray(dataset.sample_columns) ? dataset.sample_columns : []
      for (var columnIndex = 0; columnIndex < columns.length; columnIndex++) {
        if (columns[columnIndex].key === groupKey) groupCount = columns[columnIndex].n_unique
      }
      var meta = dataset.feature_meta || {}
      var metrics = [
        { label: T('metricSamples'), value: fmtNumber(dataset.n_samples) },
        { label: T('metricFeatures'), value: fmtNumber(dataset.n_features) },
        { label: T('metricPolarity'), value: dataset.polarity || T('unknown') },
        { label: T('metricAnnotated'), value: isNil(meta.n_annotated) ? '—' : fmtNumber(meta.n_annotated) },
        { label: T('metricGroups'), value: isNil(groupCount) ? '—' : fmtNumber(groupCount) + (groupKey ? ' · ' + groupKey : '') },
        { label: T('metricEngine'), value: engineInfo },
      ]
      var suggestions = Array.isArray(props.suggestions) ? props.suggestions : []
      var qcResult = (props.stepResults && props.stepResults.qc) || null
      var qcAnalysis = (props.runs && props.runs.qc_summary && props.runs.qc_summary.result) || null
      var scatter = findScatterTable(qcResult ? qcResult.tables : null) ||
        findScatterTable(qcAnalysis ? qcAnalysis.tables : null)
      return h('div', { className: 'mb-pane' },
        h('div', { className: 'mb-toolbar' },
          h('span', { className: 'mb-section-title' }, T('overviewTitle')),
          h('span', { className: 'mb-muted' }, T('overviewHint')),
          h('span', { className: 'mb-spacer' }),
          h('span', { className: 'mb-toolbar-label' }, T('pcaScores')),
          h('select', {
            className: 'mb-input mb-select',
            value: colorBy,
            style: { width: '104px' },
            onChange: function (event) { setColorBy(event.target.value) },
            disabled: !pcaReady,
          }, COLOR_BY.map(function (option) {
            return h('option', { key: option.value, value: option.value }, T(option.labelKey))
          })),
          props.busyStep ? h('span', { className: 'mb-loading' }, h(Spinner), h('span', null, props.busyStep)) : null,
          Btn({ small: true, icon: 'grid', onClick: props.onOpenAnalysis, disabled: props.busy === true }, T('analysisCatalog'))),
        h('div', { className: 'mb-metrics' }, metrics.map(function (metric) {
          return h(MetricCard, { key: metric.label, label: metric.label, value: metric.value })
        })),
        h('div', { className: 'mb-grid-2' },
          h(SectionCard, { title: T('pcaScores'), extra: pcaReady ? null : T('overviewPcaEmpty') },
            h(PlotBlock, {
              plots: props.plots,
              loadPlot: props.loadPlot,
              kind: 'pca_scores',
              params: { color_by: colorBy },
              enabled: !!dataset && pcaReady,
              emptyText: T('overviewPcaEmpty'),
            })),
          h(SectionCard, { title: T('qcTic'), extra: done.qc ? null : T('overviewTicEmpty') },
            h(PlotBlock, {
              plots: props.plots,
              loadPlot: props.loadPlot,
              kind: 'qc_tic',
              params: {},
              enabled: !!dataset && !!done.qc,
              emptyText: T('overviewTicEmpty'),
            }))),
        h('div', { className: 'mb-grid-2' },
          h(SectionCard, { title: T('pcaScree') },
            h(PlotBlock, {
              plots: props.plots,
              loadPlot: props.loadPlot,
              kind: 'pca_variance',
              params: {},
              enabled: !!dataset && pcaReady,
            })),
          h(SectionCard, { title: T('miniMapTitle'), extra: T('miniMapHint') },
            h(SampleScatter, { data: scatter }))),
        h(SectionCard, { title: T('nextSteps'), extra: T('agentSuggestions') },
          suggestions.length
            ? h('div', { className: 'mb-form' }, suggestions.slice(0, 6).map(function (suggestion) {
                var stepMeta = suggestion.step ? stepById(suggestion.step) : null
                return h('div', { key: suggestion.id, className: 'mb-suggestion' },
                  h('div', { className: 'mb-suggestion-title' }, suggestion.title || T('suggestionFallback')),
                  suggestion.detail ? h('div', { className: 'mb-suggestion-detail' }, suggestion.detail) : null,
                  h('div', { className: 'mb-inline-row' },
                    stepMeta
                      ? Btn({
                          small: true,
                          variant: 'ghost',
                          icon: 'play',
                          disabled: props.stepBusy === stepMeta.id,
                          onClick: function () { if (props.onRunStep) props.onRunStep(stepMeta.id) },
                        }, stepMeta.title)
                      : null,
                    Btn({
                      small: true,
                      variant: 'ghost',
                      icon: 'copy',
                      onClick: function () { if (props.onCopy) props.onCopy(suggestionInstruction(suggestion), 'suggestion-' + suggestion.id) },
                    }, props.copiedKey === ('suggestion-' + suggestion.id) ? T('copied') : T('agentCopy'))))
              }))
            : h('div', { className: 'mb-muted' }, T('noSuggestions'))),
        dataset.warnings.length
          ? h(SectionCard, { title: T('warnings'), extra: String(dataset.warnings.length) },
              h('div', { className: 'mb-warn-list' }, dataset.warnings.map(function (warning, index) {
                return h('div', { key: 'w-' + index }, '· ' + warning)
              })))
          : null,
        dataset.notes.length
          ? h(SectionCard, { title: T('notes'), extra: String(dataset.notes.length) },
              h('div', { className: 'mb-note-list' }, dataset.notes.slice(0, 8).map(function (note, index) {
                return h('div', { key: 'n-' + index }, '· ' + note)
              })))
          : null)
    }


    // ==========================================================================
    // 中栏：质控 / 多变量 / 差异 / 通路 / 标志物 / 鉴定
    //
    // 分析目录一律来自 GET /metabolomics/analyses（{id,title,group,method,needs,params,
    // available,reason}），每个 tab 只负责：
    //   1) 用 TAB_ANALYSES 白名单挑出本 tab 相关的分析 id；
    //   2) 用通用参数表单（服务端 schema）渲染并运行；
    //   3) 附带本 tab 的快捷出图（TAB_PLOTS，统一走 POST /plot）。
    // 不在前端硬编码 38 项分析的标题/参数。
    // ==========================================================================

    /** 结果导出按钮（各分析 tab 顶栏共用；what 与 impl.mjs EXPORT_KINDS 一致）。 */
    function exportButton(props, what, label) {
      var state = (props && props.exports && props.exports[what]) || {}
      return Btn({
        small: true,
        icon: 'download',
        disabled: state.busy === true || !props.dataset,
        title: state.file || label,
        onClick: function () { if (props.runExport) props.runExport(what) },
      }, state.busy ? T('recordsExporting') : label)
    }

    /** 从结果 metrics 里挑出模型质量指标（R²Y / Q² / 置换 p）；名字不固定就按模式匹配。 */
    function pickModelQuality(results) {
      var list = Array.isArray(results) ? results : []
      var found = {}
      var aliases = [
        { key: 'mvR2Y', pattern: /^(r2y|r2_y|r2y_cum|r2)$/i },
        { key: 'mvQ2', pattern: /^(q2|q2_cum|q²)$/i },
        { key: 'mvPermP', pattern: /(perm.*p$|^p_perm|perm_p|p_value$)/i },
      ]
      for (var index = 0; index < list.length; index++) {
        var result = list[index]
        var metrics = result && result.metrics && typeof result.metrics === 'object' ? result.metrics : {}
        var keys = Object.keys(metrics)
        for (var cursor = 0; cursor < keys.length; cursor++) {
          for (var aliasIndex = 0; aliasIndex < aliases.length; aliasIndex++) {
            var alias = aliases[aliasIndex]
            if (found[alias.key] !== undefined) continue
            if (alias.pattern.test(keys[cursor])) found[alias.key] = metrics[keys[cursor]]
          }
        }
      }
      return found
    }

    /** tab 顶部的模型质量条（只显示已拿到的指标）。 */
    function ModelQuality(props) {
      var quality = props.quality || {}
      var keys = Object.keys(quality)
      if (!keys.length) return null
      return h('div', { className: 'mb-quality' },
        keys.map(function (key) {
          var value = quality[key]
          return h('div', { key: key, className: 'mb-quality-item' },
            h('span', { className: 'mb-quality-k' }, T(key)),
            h('span', { className: 'mb-quality-v' }, typeof value === 'number' ? (Math.abs(value) < 1 && key !== 'mvPermP' ? fmtNumber(value, 3) : (key === 'mvPermP' ? fmtP(value) : fmtNumber(value, 2))) : formatCell(value, false)))
        }))
    }

    /** 分析参数默认值（对象/数组复制，避免多次运行共享引用）。 */
    function analysisDefaults(descriptors) {
      var out = {}
      var list = Array.isArray(descriptors) ? descriptors : []
      for (var index = 0; index < list.length; index++) {
        var descriptor = list[index]
        if (!descriptor || !descriptor.key) continue
        out[descriptor.key] = cloneDefault(isNil(descriptor.default) ? null : descriptor.default)
      }
      return out
    }

    /**
     * 通用分析 tab：白名单 + 服务端 schema 参数表单 + 结果信封 + 快捷出图。
     * props.ids 可覆盖 TAB_ANALYSES（AnalysisDrawer 用全目录时不传 ids）。
     */
    function AnalysisTab(props) {
      var dataset = props.dataset
      var catalog = props.catalog || { loading: false, error: '', data: normalizeAnalyses({}) }
      var all = catalog.data && Array.isArray(catalog.data.analyses) ? catalog.data.analyses : []
      var ids = Array.isArray(props.ids) ? props.ids : (props.tab ? tabAnalyses(props.tab) : null)
      var items = []
      if (ids && ids.length) {
        for (var idIndex = 0; idIndex < ids.length; idIndex++) {
          for (var itemIndex = 0; itemIndex < all.length; itemIndex++) {
            if (all[itemIndex].id === ids[idIndex]) { items.push(all[itemIndex]); break }
          }
        }
      } else {
        items = all.slice()
      }
      var selectState = useState('')
      var selectedId = selectState[0]
      var setSelectedId = selectState[1]
      var valuesState = useState({})
      var values = valuesState[0]
      var setValues = valuesState[1]
      var meta = null
      for (var index = 0; index < items.length; index++) {
        if (items[index].id === selectedId) meta = items[index]
      }
      if (!meta && items.length) meta = items[0]
      useEffect(function () {
        if (!meta) return
        if (selectedId !== meta.id) {
          setSelectedId(meta.id)
          setValues(analysisDefaults(meta.params))
        }
      }, [dataset ? dataset.id : '', catalog.data, meta ? meta.id : ''])
      var run = (props.runs && meta && props.runs[meta.id]) || null
      function patch(next) {
        setValues(function (previous) { return Object.assign({}, previous, next || {}) })
      }
      function runAnalysis() {
        if (!meta || !dataset || props.onRunAnalysis === undefined) return
        props.onRunAnalysis(meta.id, values || {})
      }
      var groupNodes = items.map(function (item) {
        var disabled = item.available === false
        var on = meta && meta.id === item.id
        return h('button', {
          key: item.id,
          type: 'button',
          className: 'mb-analysis-item' + (on ? ' mb-analysis-item-on' : ''),
          disabled: disabled,
          title: disabled ? (item.reason || T('analysisUnavailable')) : (item.method || item.title),
          onClick: function () {
            setSelectedId(item.id)
            setValues(analysisDefaults(item.params))
          },
        },
          h('div', { className: 'mb-analysis-main' },
            h('div', { className: 'mb-analysis-name' }, item.title || item.id),
            h('span', { className: 'mb-tag' + (disabled ? '' : ' mb-tag-ok') },
              disabled ? T('analysisUnavailable') : T('statusOk'))),
          item.method ? h('div', { className: 'mb-analysis-method' }, item.method) : null,
          disabled && item.reason ? h('div', { className: 'mb-analysis-reason' }, item.reason) : null)
      })
      var runner = null
      if (!dataset) {
        runner = h(Notice, { text: T('analysisNeedDataset') })
      } else if (!items.length) {
        runner = h(Notice, { text: props.emptyText || T('analysisEmpty') })
      } else if (meta) {
        runner = h('div', { className: 'mb-analysis-runner' },
          h('div', { className: 'mb-analysis-head' },
            h('div', { className: 'mb-analysis-title' }, meta.title || meta.id),
            meta.method ? h('div', { className: 'mb-analysis-method' }, T('analysisMethod') + '：' + meta.method) : null),
          meta.needs && meta.needs.length
            ? h('div', { className: 'mb-inline-row' },
                h('span', { className: 'mb-toolbar-label' }, T('analysisNeeds')),
                meta.needs.map(function (need) {
                  var stepMeta = stepById(need)
                  return h('span', { key: 'need-' + need, className: 'mb-tag' }, stepMeta ? stepMeta.title : need)
                }))
            : null,
          meta.available === false && meta.reason ? h('div', { className: 'mb-warn-text' }, meta.reason) : null,
          h(ParamForm, {
            params: meta.params,
            values: values,
            columns: props.columns || [],
            disabled: run && run.busy === true,
            onChange: patch,
          }),
          h('div', { className: 'mb-inline-row' },
            Btn({
              small: true,
              variant: 'primary',
              icon: 'play',
              disabled: !dataset || (run && run.busy === true) || meta.available === false,
              onClick: runAnalysis,
            }, run && run.busy ? T('analysisRunning') : T('analysisRun')),
            props.extraActions || null),
          run && run.error ? h(InlineError, { message: run.error }) : null,
          run && run.result
            ? h(ResultView, {
                result: run.result,
                onExport: props.tabExport && props.runExport
                  ? function () { props.runExport(props.tabExport.what) }
                  : null,
                exportBusy: props.tabExport ? !!(props.exports && props.exports[props.tabExport.what] && props.exports[props.tabExport.what].busy) : false,
                exportLabel: props.tabExport ? T(props.tabExport.labelKey) : '',
                exportFile: props.tabExport && props.exports && props.exports[props.tabExport.what]
                  ? props.exports[props.tabExport.what].file || ''
                  : '',
              })
            : null)
      }
      var plots = Array.isArray(props.plotsDef) ? props.plotsDef : (props.tab ? tabPlots(props.tab) : [])
      var plotNodes = plots.length
        ? h('div', { className: 'mb-grid-2' }, plots.map(function (def, plotIndex) {
            return h(SectionCard, { key: 'plot-' + def.kind + '-' + plotIndex, title: T(def.titleKey) },
              h(PlotBlock, {
                plots: props.plots,
                loadPlot: props.loadPlot,
                kind: def.kind,
                params: def.params || {},
                enabled: !!dataset,
              }))
          }))
        : null
      return h('div', { className: 'mb-pane' },
        h('div', { className: 'mb-toolbar' },
          h('span', { className: 'mb-section-title' }, props.title),
          h('span', { className: 'mb-muted' }, props.hint),
          h('span', { className: 'mb-spacer' }),
          props.toolbarExtra || null),
        props.headerExtra || null,
        catalog.loading ? h(LoadingLine, { text: T('loading') }) : null,
        catalog.error ? h(InlineError, { message: catalog.error }) : null,
        h('div', { className: 'mb-analysis-wrap' },
          h('div', { className: 'mb-analysis-list' }, groupNodes.length ? groupNodes : h('div', { className: 'mb-muted' }, props.emptyText || T('analysisEmpty'))),
          runner),
        plotNodes)
    }

    /** 质控 tab：质控汇总 / QC-RSD / 缺失模式 / 离群 / 相关性 + RSD 与缺失率图。 */
    function QcTab(props) {
      var dataset = props.dataset
      var qcResult = (props.stepResults && props.stepResults.qc) || null
      var qcAnalysis = (props.runs && props.runs.qc_summary && props.runs.qc_summary.result) || null
      var scatter = findScatterTable(qcResult ? qcResult.tables : null) || findScatterTable(qcAnalysis ? qcAnalysis.tables : null)
      var done = doneStepSet(dataset)
      return h(AnalysisTab, Object.assign({}, props, {
        tab: 'qc',
        title: T('qcTitle'),
        hint: T('qcHint'),
        emptyText: T('qcEmpty'),
        headerExtra: h(SectionCard, { title: T('miniMapTitle'), extra: T('miniMapHint') },
          h(SampleScatter, { data: scatter }),
          !done.qc
            ? h('div', { className: 'mb-inline-row' },
                Btn({
                  small: true,
                  variant: 'primary',
                  icon: 'play',
                  disabled: props.busy === true || !dataset,
                  onClick: function () { if (props.onRunStep) props.onRunStep('qc') },
                }, props.stepBusy === 'qc' ? T('qcRunStepBusy') : T('qcRunStep')))
            : null),
        toolbarExtra: h('div', { className: 'mb-inline-row' },
          exportButton(props, 'sample_info_csv', T('exportSampleCsv'))),
      }))
    }

    /** 多变量 tab：VIP / S-plot / 置换 / 交叉验证 / 多类 / 载荷 + R²Y·Q²·置换 p。 */
    function MultivariateTab(props) {
      var stepResult = (props.stepResults && props.stepResults.plsda) || null
      var results = []
      if (stepResult) results.push(stepResult)
      var runIds = ['plsda_cv', 'plsda_permutation', 'plsda_vip']
      for (var index = 0; index < runIds.length; index++) {
        var run = props.runs && props.runs[runIds[index]]
        if (run && run.result) results.push(run.result)
      }
      var quality = pickModelQuality(results)
      return h(AnalysisTab, Object.assign({}, props, {
        tab: 'multivariate',
        title: T('mvTitle'),
        hint: T('mvHint'),
        emptyText: T('mvEmpty'),
        headerExtra: h(ModelQuality, { quality: quality }),
        toolbarExtra: h('div', { className: 'mb-inline-row' },
          exportButton(props, 'vip_csv', T('exportVipCsv'))),
      }))
    }

    function DeTab(props) {
      return h(AnalysisTab, Object.assign({}, props, {
        tab: 'de',
        tabExport: { what: 'de_csv', labelKey: 'deExport' },
        title: T('deTitle'),
        hint: T('deHint'),
        emptyText: T('deEmpty'),
        toolbarExtra: exportButton(props, 'de_csv', T('deExport')),
      }))
    }

    function PathwayTab(props) {
      return h(AnalysisTab, Object.assign({}, props, {
        tab: 'pathway',
        tabExport: { what: 'enrich_csv', labelKey: 'exportEnrichCsv' },
        title: T('pathwayTitle'),
        hint: T('pathwayHint'),
        emptyText: T('pathwayEmpty'),
        toolbarExtra: exportButton(props, 'enrich_csv', T('exportEnrichCsv')),
      }))
    }

    function BiomarkerTab(props) {
      return h(AnalysisTab, Object.assign({}, props, {
        tab: 'biomarker',
        tabExport: { what: 'biomarker_csv', labelKey: 'exportBiomarkerCsv' },
        title: T('biomarkerTitle'),
        hint: T('biomarkerHint'),
        emptyText: T('biomarkerEmpty'),
        toolbarExtra: exportButton(props, 'biomarker_csv', T('exportBiomarkerCsv')),
      }))
    }

    /**
     * 鉴定 tab：加合物 / 分子式 / MS2 库匹配 / 分子网络。
     * 「MS2 谱库」是一个路径输入：填了之后可以作为 annotate 步骤的 library 参数运行；
     * 分析表单里的 library 参数仍由服务端 schema 提供（这里只做便捷入口，不另造接口）。
     */
    function AnnotationTab(props) {
      var dataset = props.dataset
      var mzReady = dataset ? hasMz(dataset) : false
      var libraryState = useState('')
      var library = libraryState[0]
      var setLibrary = libraryState[1]
      function runAnnotate() {
        if (!dataset || !props.onRunStep) return
        var params = defaultParams('annotate')
        params.library = plainText(library)
        props.onRunStep('annotate', params)
      }
      return h(AnalysisTab, Object.assign({}, props, {
        tab: 'annotation',
        title: T('annotationTitle'),
        hint: T('annotationHint'),
        emptyText: T('annotationEmpty'),
        headerExtra: h(SectionCard, { title: T('annotationLibrary'), extra: T('annotationHint') },
          !mzReady ? h('div', { className: 'mb-warn-text' }, T('annotationNeedMz')) : null,
          h('div', { className: 'mb-library-row' },
            h('input', {
              className: 'mb-input',
              value: library,
              spellCheck: false,
              placeholder: T('annotationLibraryPh'),
              onChange: function (event) { setLibrary(event.target.value) },
            }),
            Btn({
              small: true,
              variant: 'primary',
              icon: 'play',
              disabled: !dataset || !plainText(library) || props.busy === true,
              onClick: runAnnotate,
            }, props.stepBusy === 'annotate' ? T('annotationLibraryBusy') : T('annotationLibraryApply'))),
          h('div', { className: 'mb-library-hint' }, T('annotationLibraryHint'))),
      }))
    }

    // ==========================================================================
    // 中栏：记录（历史 + 自检 + 导出）/ 日志
    // ==========================================================================

    function RecordsTab(props) {
      var dataset = props.dataset
      var auditState = useState({ busy: false, error: '', audit: null })
      var audit = auditState[0]
      var setAudit = auditState[1]
      if (!dataset) {
        return h('div', { className: 'mb-pane' }, h(TabGuide, { title: T('datasetNoDataset'), body: T('datasetOpenHint') }))
      }
      function runAudit() {
        setAudit({ busy: true, error: '', audit: null })
        props.api.audit(dataset.id)
          .then(function (payload) { setAudit({ busy: false, error: '', audit: normalizeAudit(payload) }) })
          .catch(function (error) { setAudit({ busy: false, error: formatError(error), audit: null }) })
      }
      var history = (dataset.history || []).slice().reverse()
      var analysisRuns = (dataset.analyses_run || []).slice().reverse()
      return h('div', { className: 'mb-pane' },
        h('div', { className: 'mb-toolbar' },
          h('span', { className: 'mb-section-title' }, T('recordsTitle')),
          h('span', { className: 'mb-muted' }, T('recordsHint')),
          h('span', { className: 'mb-spacer' }),
          Btn({ small: true, icon: 'grid', onClick: props.onOpenAnalysis }, T('recordsAnalysis'))),
        h('div', { className: 'mb-grid-2' },
          h(SectionCard, { title: T('recordsHistory'), extra: tf(T, 'tableRows', history.length) },
            history.length
              ? h('div', { className: 'mb-history' }, history.map(function (entry, index) {
                  return h('div', { key: 'h-' + index, className: 'mb-history-item' },
                    h('span', { className: 'mb-history-at' }, formatClock(entry.at)),
                    h('span', { className: 'mb-history-step' }, (stepById(entry.step) || {}).title || entry.step || '—'),
                    h('span', { className: 'mb-history-summary', title: entry.summary || '' }, entry.summary || ''),
                    h('span', { className: 'mb-history-ms' }, entry.ms ? formatDuration(entry.ms) : ''),
                    entry.ok ? h('span', { className: 'mb-ok-text' }, T('completed')) : h('span', { className: 'mb-warn-text' }, T('failed')))
                }))
              : h('div', { className: 'mb-muted' }, T('recordsNoHistory'))),
          h(SectionCard, {
            title: T('recordsAudit'),
            extra: Btn({ small: true, icon: 'shield', disabled: audit.busy, onClick: runAudit },
              audit.busy ? T('running') : T('recordsRunAudit')),
          },
            audit.error ? h(InlineError, { message: audit.error }) : null,
            audit.audit
              ? h('div', { className: 'mb-form' },
                  h('div', { className: 'mb-score' },
                    h('span', { className: 'mb-score-v' }, String(audit.audit.score)),
                    h('span', { className: 'mb-score-label' },
                      T('recordsAuditScore') + ' · ' +
                      T(AUDIT_LEVEL_STYLE[audit.audit.level] ? AUDIT_LEVEL_STYLE[audit.audit.level].labelKey : 'auditOk'))),
                  h('div', { className: 'mb-muted' }, T('auditHint')),
                  audit.audit.checks.map(function (check) {
                    return h('div', {
                      key: check.id,
                      className: 'mb-audit-item' +
                        (check.level === 'warn' ? ' mb-audit-item-warn' : (check.level === 'fail' ? ' mb-audit-item-fail' : '')),
                    },
                      h('span', {
                        className: 'mb-audit-mark' +
                          (check.level === 'warn' ? ' mb-audit-mark-warn' : (check.level === 'fail' ? ' mb-audit-mark-fail' : '')),
                      }, check.level === 'ok' ? '✓' : (check.level === 'warn' ? '!' : '×')),
                      h('div', { className: 'mb-audit-main' },
                        h('span', { className: 'mb-audit-title' }, check.title || check.id),
                        check.detail ? h('span', { className: 'mb-audit-detail' }, check.detail) : null,
                        check.hint ? h('span', { className: 'mb-audit-detail' }, check.hint) : null))
                  }))
              : h('div', { className: 'mb-muted' }, T('recordsAuditEmpty')))),
        h(SectionCard, { title: T('recordsAnalyses'), extra: tf(T, 'tableRows', analysisRuns.length) },
          analysisRuns.length
            ? h('div', { className: 'mb-history' }, analysisRuns.map(function (entry, index) {
                return h('div', { key: 'a-' + index, className: 'mb-history-item' },
                  h('span', { className: 'mb-history-at' }, formatClock(entry.at)),
                  h('span', { className: 'mb-history-step' }, entry.id || '—'),
                  h('span', { className: 'mb-history-ms' }, entry.ms ? formatDuration(entry.ms) : ''),
                  entry.ok ? h('span', { className: 'mb-ok-text' }, T('completed')) : h('span', { className: 'mb-warn-text' }, T('failed')))
              }))
            : h('div', { className: 'mb-muted' }, T('recordsNoHistory'))),
        h(SectionCard, { title: T('recordsExport'), extra: T('exportBtn') },
          h('div', { className: 'mb-inline-row' },
            h('span', { className: 'mb-toolbar-label' }, T('recordsQuickExports')),
            QUICK_EXPORTS.map(function (what) {
              var state = (props.exports && props.exports[what]) || {}
              var found = null
              for (var index = 0; index < EXPORT_KINDS.length; index++) {
                if (EXPORT_KINDS[index].id === what) found = EXPORT_KINDS[index]
              }
              return Btn({
                key: 'quick-' + what,
                small: true,
                icon: 'download',
                disabled: state.busy === true || !dataset,
                title: state.file || (found ? T(found.labelKey) : what),
                onClick: function () { if (props.runExport) props.runExport(what) },
              }, state.busy ? T('recordsExporting') : (found ? T(found.labelKey) : what))
            })),
          h('div', { className: 'mb-export-grid' }, EXPORT_KINDS.map(function (item) {
            var state = (props.exports && props.exports[item.id]) || {}
            return h('button', {
              key: item.id,
              type: 'button',
              className: 'mb-export-item',
              disabled: state.busy === true,
              title: state.file || T(item.labelKey),
              onClick: function () { if (props.runExport) props.runExport(item.id) },
            },
              h('span', null, state.busy ? T('recordsExporting') : T(item.labelKey)),
              h('span', { className: 'mb-export-hint' }, state.file ? state.file : item.hint),
              state.error ? h('span', { className: 'mb-warn-text' }, state.error) : null)
          }))),
        h(Notice, { text: T('auditHint') }))
    }

    function LogTab(props) {
      var state = useState({ loading: true, error: '', entries: [] })
      var log = state[0]
      var setLog = state[1]
      var mountedRef = useRef(false)
      function load() {
        setLog(function (previous) { return Object.assign({}, previous, { loading: true, error: '' }) })
        props.api.getActivity()
          .then(function (payload) {
            var list = Array.isArray(payload && payload.entries)
              ? payload.entries
              : (Array.isArray(payload && payload.activity) ? payload.activity : [])
            setLog({
              loading: false,
              error: '',
              entries: list.map(function (entry, index) { return normalizeActivityEntry(entry, index) }),
            })
          })
          .catch(function (error) { setLog({ loading: false, error: formatError(error), entries: [] }) })
      }
      useEffect(function () {
        if (mountedRef.current) return
        mountedRef.current = true
        load()
      }, [])
      return h('div', { className: 'mb-pane' },
        h('div', { className: 'mb-toolbar' },
          h('span', { className: 'mb-section-title' }, T('logTitle')),
          h('span', { className: 'mb-spacer' }),
          Btn({ small: true, icon: 'refresh', onClick: load, disabled: log.loading }, T('logRefresh'))),
        log.error ? h(InlineError, { message: log.error }) : null,
        log.loading
          ? h(LoadingLine, { text: T('loading') })
          : (log.entries.length
              ? h('div', { className: 'mb-table-wrap' },
                  h('table', { className: 'mb-table' },
                    h('thead', null, h('tr', null,
                      h('th', null, T('at')),
                      h('th', null, T('actor')),
                      h('th', null, T('action')),
                      h('th', null, T('dataset')),
                      h('th', null, T('duration')),
                      h('th', null, T('resultColumn')))),
                    h('tbody', null, log.entries.map(function (entry) {
                      return h('tr', { key: entry.id },
                        h('td', null, formatClock(entry.at)),
                        h('td', null, entry.actor === 'agent' ? T('activityAgent') : (entry.actor === 'system' ? T('engine') : T('activityUser'))),
                        h('td', null, entry.action || '—'),
                        h('td', null, entry.dataset || '—'),
                        h('td', { className: 'mb-td-num' }, entry.ms ? formatDuration(entry.ms) : '—'),
                        h('td', null, h('span', { className: entry.ok ? 'mb-ok-text' : 'mb-warn-text' }, entry.ok ? T('completed') : T('failed'))))
                    }))))
              : h(Notice, { text: T('logEmpty') })))
    }

    // GEO / MetaboLights 检索弹层：搜索 → 详情 → 手填下载目录 → 下载（轮询）→ 导入并质控
    //
    // 说明：impl.mjs 没有 /pick（原生目录选择）路由，所以下载目录一律**手填绝对路径**，
    // 前端只做绝对路径校验，不调用不存在的接口。两个数据源共用一个 SourceDialog，
    // 差异只在「检索表单 / 结果卡 / 详情字段 / 默认目录 / 下载接口」。
    // ==========================================================================

    /** GEO 结果项归一化（DESIGN §7.1 items 形状）。 */
    function normalizeGeoItem(item) {
      var raw = item && typeof item === 'object' ? item : {}
      return {
        id: plainText(raw.gse || raw.id || raw.accession || ''),
        title: plainText(raw.title || ''),
        titleZh: plainText(raw.title_zh || ''),
        summary: plainText(raw.summary || raw.description || '', 600),
        species: plainText(raw.species || ''),
        nSamples: asNumber(raw.n_samples),
        platform: Array.isArray(raw.platforms) ? raw.platforms.join(', ') : plainText(raw.platform || ''),
        studyType: plainText(raw.study_type || ''),
        release: plainText(raw.release || raw.submission_date || ''),
        method: plainText(raw.method || ''),
        hasMatrix: raw.has_matrix === true,
        hasClinical: raw.has_clinical === true,
        disease: plainText(raw.disease || ''),
        tags: Array.isArray(raw.tags) ? raw.tags.map(function (tag) { return plainText(tag) }).filter(Boolean) : [],
      }
    }

    function normalizeGeoDetail(payload) {
      var raw = payload && typeof payload === 'object' ? payload : {}
      var files = []
      if (Array.isArray(raw.files)) {
        for (var index = 0; index < raw.files.length; index++) {
          var file = raw.files[index] && typeof raw.files[index] === 'object' ? raw.files[index] : {}
          var name = plainText(file.name || '')
          if (!name) continue
          files.push({
            name: name,
            size: asNumber(file.size),
            kind: plainText(file.kind || file.type || ''),
            url: plainText(file.url || ''),
          })
        }
      }
      var clinical = raw.clinical && typeof raw.clinical === 'object' ? raw.clinical : {}
      return {
        id: plainText(raw.gse || raw.accession || ''),
        title: plainText(raw.title || ''),
        titleZh: plainText(raw.title_zh || ''),
        summary: plainText(raw.summary || raw.description || '', 4000),
        design: plainText(raw.design || ''),
        species: plainText(raw.species || ''),
        nSamples: asNumber(raw.n_samples),
        platform: Array.isArray(raw.platforms) ? raw.platforms.join(', ') : plainText(raw.platforms || raw.platform || ''),
        release: plainText(raw.release || raw.submission_date || ''),
        pubmed: plainText(raw.pubmed || ''),
        contacts: plainText(raw.contacts || ''),
        publications: Array.isArray(raw.publications)
          ? raw.publications.map(function (item) { return plainText(item && (item.title || item)) }).filter(Boolean)
          : [],
        hasClinical: raw.has_clinical === true || clinical.from_series_matrix === true,
        clinicalFields: Array.isArray(clinical.fields) ? clinical.fields.map(function (field) { return plainText(field) }) : [],
        files: files,
        suggested: plainText(raw.suggested || ''),
        url: plainText(raw.url || ''),
      }
    }

    /** MetaboLights 结果项归一化（DESIGN §7.4 items 形状）。 */
    function normalizeMlItem(item) {
      var raw = item && typeof item === 'object' ? item : {}
      return {
        id: plainText(raw.accession || raw.id || ''),
        title: plainText(raw.title || ''),
        titleZh: '',
        summary: plainText(raw.description || raw.summary || '', 600),
        species: Array.isArray(raw.species) ? raw.species.map(function (name) { return plainText(name) }).join(', ') : plainText(raw.species || ''),
        nSamples: asNumber(raw.n_assays !== undefined ? raw.n_assays : raw.files_count),
        platform: plainText(raw.study_type || ''),
        studyType: plainText(raw.study_type || ''),
        release: plainText(raw.submission_date || ''),
        method: plainText(raw.method || ''),
        hasMatrix: true,
        hasClinical: false,
        disease: '',
        tags: [],
      }
    }

    /** 通用「来源任务」轮询：POST 返回 {job} → GET /jobs?id= 直到 done/failed/timeout。 */
    function useJob(api) {
      var state = useState({ active: false, status: '', pct: 0, message: '', error: '', result: null, id: '' })
      var job = state[0]
      var setJob = state[1]
      var timerRef = useRef(null)
      var startedRef = useRef(0)
      function stop() {
        if (timerRef.current) {
          clearInterval(timerRef.current)
          timerRef.current = null
        }
      }
      useEffect(function () { return stop }, [])
      function poll(id) {
        api.getJob(id).then(function (payload) {
          var next = jobPolling(payload, { elapsedMs: Date.now() - startedRef.current, timeoutMs: POLL_TIMEOUT_MS })
          setJob({
            active: next.running,
            status: next.status,
            pct: next.pct,
            message: next.message,
            error: next.error,
            result: next.result,
            id: id,
          })
          if (!next.running) stop()
        }).catch(function (error) {
          stop()
          setJob(function (previous) {
            return Object.assign({}, previous, { active: false, status: 'failed', error: formatError(error) })
          })
        })
      }
      function watch(id) {
        var jobId = plainText(id)
        if (!jobId) return
        stop()
        startedRef.current = Date.now()
        setJob({ active: true, status: 'running', pct: 0, message: T('loading'), error: '', result: null, id: jobId })
        poll(jobId)
        timerRef.current = setInterval(function () { poll(jobId) }, POLL_MS)
      }
      function reset() {
        stop()
        setJob({ active: false, status: '', pct: 0, message: '', error: '', result: null, id: '' })
      }
      return [job, watch, reset]
    }

    function ProgressBar(props) {
      var pct = clamp(asNumber(props.pct) || 0, 0, 100)
      return h('div', { className: 'mb-progress' },
        h('div', { className: 'mb-progress-bar' }, h('div', { className: 'mb-progress-fill', style: { width: pct + '%' } })),
        h('div', { className: 'mb-progress-text' },
          h('span', null, (props.label || '') + (props.message ? ' · ' + props.message : '')),
          h('span', null, Math.round(pct) + '%')))
    }

    function isAbsoluteDir(value) {
      var text = plainText(value)
      if (!text) return false
      return text.charAt(0) === '/' || text.charAt(0) === '~'
    }

    /** 结果卡（GEO / ML 共用骨架）。 */
    function SourceCard(props) {
      var item = props.item
      var on = props.active === true
      return h('button', {
        type: 'button',
        className: 'mb-geo-card' + (on ? ' mb-geo-card-on' : ''),
        onClick: props.onClick,
      },
        h('div', { className: 'mb-inline-row' },
          h('span', { className: props.source === 'ml' ? 'mb-ml-accession' : 'mb-geo-gse' }, item.id || '—'),
          item.method ? h('span', { className: 'mb-geo-tag mb-geo-tag-high' }, item.method) : null,
          item.hasMatrix ? h('span', { className: 'mb-geo-tag mb-geo-tag-ok' }, T('geoMatrix')) : null),
        item.titleZh ? h('div', { className: 'mb-geo-title' }, item.titleZh) : null,
        item.title ? h('div', { className: props.source === 'ml' ? 'mb-ml-title' : 'mb-geo-title-en' }, item.title) : null,
        h('div', { className: props.source === 'ml' ? 'mb-ml-meta' : 'mb-geo-meta' },
          item.species ? h('span', null, item.species) : null,
          !isNil(item.nSamples) ? h('span', null, (props.source === 'ml' ? T('mlAssays') : T('samples')) + ' ' + fmtNumber(item.nSamples)) : null,
          item.platform ? h('span', null, item.platform) : null,
          item.release ? h('span', null, item.release) : null),
        item.tags && item.tags.length
          ? h('div', { className: 'mb-inline-row' }, item.tags.slice(0, 5).map(function (tag) {
              return h('span', { key: 'tag-' + tag, className: 'mb-geo-tag' }, tag)
            }))
          : null)
    }

    /** GEO / MetaboLights 的共用检索弹层。 */
    function SourceDialog(props) {
      var source = props.source === 'ml' ? 'ml' : 'geo'
      var api = props.api
      var jobState = useJob(api)
      var job = jobState[0]
      var watchJob = jobState[1]
      var resetJob = jobState[2]

      var searchState = useState({ busy: false, error: '', total: 0, items: [], relaxed: '', usedTerm: '' })
      var search = searchState[0]
      var setSearch = searchState[1]
      var termState = useState('')
      var term = termState[0]
      var setTerm = termState[1]
      var speciesState = useState('all')
      var species = speciesState[0]
      var setSpecies = speciesState[1]
      var methodState = useState('all')
      var method = methodState[0]
      var setMethod = methodState[1]
      var minSamplesState = useState('')
      var minSamples = minSamplesState[0]
      var setMinSamples = minSamplesState[1]
      var clinicalState = useState(false)
      var hasClinical = clinicalState[0]
      var setHasClinical = clinicalState[1]
      var limitState = useState('20')
      var limit = limitState[0]
      var setLimit = limitState[1]

      var detailState = useState({ id: '', busy: false, error: '', data: null })
      var detail = detailState[0]
      var setDetail = detailState[1]
      var filesState = useState({})
      var files = filesState[0]
      var setFiles = filesState[1]
      var dirState = useState(source === 'ml' ? DEFAULT_ML_DIR : DEFAULT_GEO_DIR)
      var destDir = dirState[0]
      var setDestDir = dirState[1]
      var importState = useState({ busy: false, error: '' })
      var importing = importState[0]
      var setImporting = importState[1]

      function runSearch() {
        setSearch(function (previous) { return Object.assign({}, previous, { busy: true, error: '' }) })
        setDetail({ id: '', busy: false, error: '', data: null })
        setFiles({})
        resetJob()
        var request = source === 'ml'
          ? api.mlSearch({ query: term, limit: limit, page: 0 })
          : api.geoSearch({
              term: term,
              species: species,
              method: method,
              min_samples: minSamples,
              has_clinical: hasClinical,
              limit: 20,
              page: 0,
            })
        request.then(function (payload) {
          var raw = payload && typeof payload === 'object' ? payload : {}
          var list = Array.isArray(raw.items) ? raw.items : []
          var items = []
          for (var index = 0; index < list.length; index++) {
            items.push(source === 'ml' ? normalizeMlItem(list[index]) : normalizeGeoItem(list[index]))
          }
          setSearch({
            busy: false,
            error: '',
            notes: Array.isArray(raw.notes) ? raw.notes.filter(function (note) { return plainText(note) }) : [],
            total: asNumber(raw.total) === null ? items.length : asNumber(raw.total),
            items: items,
            relaxed: plainText(raw.relaxed || raw.used_term || ''),
            usedTerm: plainText(raw.used_term || ''),
          })
        }).catch(function (error) {
          setSearch({ busy: false, error: formatError(error), total: 0, items: [], notes: [], relaxed: '', usedTerm: '' })
        })
      }

      function openDetail(id) {
        var target = plainText(id)
        if (!target) return
        setDetail({ id: target, busy: true, error: '', data: null })
        setFiles({})
        resetJob()
        var request = source === 'ml' ? api.mlDetail(target) : api.geoDetail(target)
        request.then(function (payload) {
          var data = normalizeGeoDetail(payload)
          var next = {}
          var suggested = data.suggested || (data.files.length ? data.files[0].name : '')
          for (var index = 0; index < data.files.length; index++) {
            next[data.files[index].name] = data.files[index].name === suggested
          }
          setFiles(next)
          setDetail({ id: target, busy: false, error: '', data: data })
        }).catch(function (error) {
          setDetail({ id: target, busy: false, error: formatError(error), data: null })
        })
      }

      function toggleFile(name) {
        setFiles(function (previous) {
          var next = Object.assign({}, previous)
          next[name] = !next[name]
          return next
        })
      }

      function selectedFiles() {
        var out = []
        var names = Object.keys(files)
        for (var index = 0; index < names.length; index++) {
          if (files[names[index]]) out.push(names[index])
        }
        return out
      }

      function startDownload() {
        var data = detail.data
        if (!data) return
        var chosen = selectedFiles()
        if (!chosen.length) {
          setDetail(function (previous) { return Object.assign({}, previous, { error: T('geoNeedFile') }) })
          return
        }
        if (!isAbsoluteDir(destDir)) {
          setDetail(function (previous) { return Object.assign({}, previous, { error: T('geoInvalidDir') }) })
          return
        }
        setDetail(function (previous) { return Object.assign({}, previous, { error: '' }) })
        var options = {
          files: chosen,
          dest_dir: plainText(destDir),
          overwrite: false,
        }
        if (source === 'ml') options.accession = data.id
        else options.gse = data.id
        var request = source === 'ml' ? api.mlDownload(options) : api.geoDownload(options)
        request.then(function (payload) {
          var jobId = pickJobId(payload)
          if (jobId) watchJob(jobId)
        }).catch(function (error) {
          setDetail(function (previous) { return Object.assign({}, previous, { error: formatError(error) }) })
        })
      }

      function importDownloaded() {
        var data = detail.data
        var result = job.result || {}
        var list = Array.isArray(result.files) ? result.files : []
        var first = null
        for (var index = 0; index < list.length; index++) {
          if (list[index] && list[index].ok !== false && list[index].path) { first = list[index]; break }
        }
        var path = first ? String(first.path) : ''
        if (!path) {
          setImporting({ busy: false, error: T('geoNeedFile') })
          return
        }
        setImporting({ busy: true, error: '' })
        var name = data ? data.id : ''
        api.openDataset({ path: path, name: name }).then(function (payload) {
          var datasetId = payload && payload.dataset ? payload.dataset.id : ''
          if (!datasetId) return payload
          return api.runStep(datasetId, 'qc', {}).catch(function () { return payload })
        }).then(function (payload) {
          setImporting({ busy: false, error: '' })
          if (props.onImported) props.onImported(payload)
        }).catch(function (error) {
          setImporting({ busy: false, error: formatError(error) })
        })
      }

      var items = search.items
      var detailData = detail.data
      var running = job.active === true
      var done = job.status === 'done'
      var failed = job.status === 'failed' || job.status === 'timeout'
      return h(Modal, {
        title: source === 'ml' ? T('mlTitle') : T('geoTitle'),
        subtitle: source === 'ml' ? T('mlSubtitle') : T('geoSubtitle'),
        wide: true,
        busy: running || importing.busy,
        onClose: props.onClose,
      },
        h('div', { className: 'mb-modal-body' },
          h('div', { className: 'mb-geo-search' },
            h('input', {
              className: 'mb-input mb-geo-term',
              value: term,
              spellCheck: false,
              placeholder: source === 'ml' ? T('mlQueryPh') : T('geoTermPh'),
              onChange: function (event) { setTerm(event.target.value) },
              onKeyDown: function (event) { if (event.key === 'Enter') runSearch() },
            }),
            source === 'ml'
              ? h('input', {
                  className: 'mb-input',
                  type: 'number',
                  min: 1,
                  max: 50,
                  step: 1,
                  style: { width: '78px' },
                  value: limit,
                  title: 'limit',
                  onChange: function (event) { setLimit(event.target.value) },
                })
              : null,
            Btn({ variant: 'primary', icon: 'search', disabled: search.busy, onClick: runSearch },
              search.busy ? (source === 'ml' ? T('mlSearching') : T('geoSearching')) : (source === 'ml' ? T('mlSearchBtn') : T('geoSearchBtn')))),
          source === 'geo'
            ? h('div', { className: 'mb-geo-filters' },
                h('span', null, T('geoSpecies')),
                h('select', {
                  className: 'mb-input mb-select',
                  value: species,
                  onChange: function (event) { setSpecies(event.target.value) },
                }, [
                  h('option', { key: 'all', value: 'all' }, T('geoAll')),
                  h('option', { key: 'human', value: 'human' }, T('geoHuman')),
                  h('option', { key: 'mouse', value: 'mouse' }, T('geoMouse')),
                ]),
                h('span', null, T('geoMethod')),
                h('select', {
                  className: 'mb-input mb-select',
                  value: method,
                  onChange: function (event) { setMethod(event.target.value) },
                }, GEO_METHODS.map(function (option) {
                  return h('option', { key: option.value, value: option.value }, option.labelKey ? T(option.labelKey) : option.label)
                })),
                h('span', null, T('geoMinSamples')),
                h('input', {
                  className: 'mb-input',
                  type: 'number',
                  min: 0,
                  step: 1,
                  style: { width: '74px' },
                  value: minSamples,
                  onChange: function (event) { setMinSamples(event.target.value) },
                }),
                h('label', { className: 'mb-check' },
                  h('input', {
                    type: 'checkbox',
                    checked: hasClinical,
                    onChange: function (event) { setHasClinical(event.target.checked) },
                  }),
                  h('span', null, T('geoHasClinical'))),
                h('span', { className: 'mb-muted' }, T('geoHint')))
            : h('div', { className: 'mb-geo-filters' }, h('span', { className: 'mb-muted' }, T('mlHint'))),
          search.error ? h(InlineError, { message: search.error }) : null,
          (search.notes && search.notes.length)
            ? h('div', { className: 'mb-muted', style: { marginTop: '6px' } },
                search.notes.map(function (note, index) { return h('div', { key: 'gn' + index }, '· ' + note) }))
            : null,
          h('div', { className: 'mb-geo-layout' },
            h('div', { className: 'mb-geo-list' },
              search.total
                ? h('div', { className: 'mb-toolbar-label' }, tf(T, 'geoTotal', search.total))
                : null,
              items.map(function (item) {
                return h(SourceCard, {
                  key: item.id,
                  item: item,
                  source: source,
                  active: detail.id === item.id,
                  onClick: function () { openDetail(item.id) },
                })
              }),
              !items.length && !search.busy ? h('div', { className: 'mb-muted' }, source === 'ml' ? T('mlNoResults') : T('geoNoResults')) : null,
              search.busy ? h(LoadingLine, { text: T('loading') }) : null),
            h('div', { className: 'mb-geo-detail' },
              !detail.id ? h('div', { className: 'mb-muted' }, source === 'ml' ? T('mlDetailBtn') : T('geoDetailBtn')) : null,
              detail.busy ? h(LoadingLine, { text: T('geoLoadingDetail') }) : null,
              detail.error ? h(InlineError, { message: detail.error }) : null,
              detailData
                ? h('div', { className: 'mb-form' },
                    h('div', { className: 'mb-geo-detail-title' }, detailData.titleZh || detailData.title || detailData.id),
                    detailData.titleZh && detailData.title ? h('div', { className: 'mb-geo-title-en' }, detailData.title) : null,
                    h('div', { className: 'mb-geo-kv' },
                      h('dt', null, source === 'ml' ? T('mlAccession') : 'GSE'),
                      h('dd', null, detailData.id || '—'),
                      detailData.species ? h('dt', null, T('species')) : null,
                      detailData.species ? h('dd', null, detailData.species) : null,
                      !isNil(detailData.nSamples) ? h('dt', null, source === 'ml' ? T('mlAssays') : T('samples')) : null,
                      !isNil(detailData.nSamples) ? h('dd', null, fmtNumber(detailData.nSamples)) : null,
                      detailData.platform ? h('dt', null, source === 'ml' ? T('geoStudyType') : 'GPL') : null,
                      detailData.platform ? h('dd', null, detailData.platform) : null,
                      detailData.release ? h('dt', null, source === 'ml' ? T('mlSubmission') : T('geoRelease')) : null,
                      detailData.release ? h('dd', null, detailData.release) : null,
                      detailData.pubmed ? h('dt', null, T('geoPubmed')) : null,
                      detailData.pubmed ? h('dd', null, detailData.pubmed) : null,
                      detailData.contacts ? h('dt', null, T('mlContacts')) : null,
                      detailData.contacts ? h('dd', null, detailData.contacts) : null,
                      detailData.hasClinical ? h('dt', null, T('geoHasClinical')) : null,
                      detailData.hasClinical ? h('dd', null, detailData.clinicalFields.join('、') || T('yes')) : null),
                    detailData.summary
                      ? h('div', { className: 'mb-muted', style: { whiteSpace: 'pre-wrap' } }, detailData.summary.slice(0, 1200))
                      : null,
                    detailData.design
                      ? h('details', { className: 'mb-advanced' },
                          h('summary', null, T('geoDesign')),
                          h('div', { className: 'mb-muted', style: { whiteSpace: 'pre-wrap' } }, detailData.design.slice(0, 2000)))
                      : null,
                    h('div', { className: 'mb-section-title' },
                      h('span', null, T('geoFiles')),
                      h('span', { className: 'mb-toolbar-label' }, detailData.files.length ? fmtNumber(detailData.files.length) : '')),
                    h('div', { className: 'mb-inline-row' },
                      Btn({
                        small: true,
                        variant: 'ghost',
                        onClick: function () {
                          var all = {}
                          for (var index = 0; index < detailData.files.length; index++) all[detailData.files[index].name] = true
                          setFiles(all)
                        },
                      }, T('geoSelectAll')),
                      Btn({ small: true, variant: 'ghost', onClick: function () { setFiles({}) } }, T('geoClear'))),
                    h('div', { className: 'mb-form' }, detailData.files.map(function (file) {
                      return h('label', { key: file.name, className: 'mb-geo-file' },
                        h('input', {
                          type: 'checkbox',
                          checked: files[file.name] === true,
                          onChange: function () { toggleFile(file.name) },
                        }),
                        h('span', { className: 'mb-geo-file-name', title: file.name }, file.name),
                        file.kind ? h('span', { className: 'mb-geo-tag' }, file.kind) : null,
                        !isNil(file.size) ? h('span', { className: 'mb-toolbar-label' }, formatBytes(file.size)) : null)
                    })),
                    detailData.files.length ? null : h('div', { className: 'mb-muted' }, T('geoNoFiles')),
                    h('div', { className: 'mb-library-row' },
                      h('span', { className: 'mb-field-label' }, source === 'ml' ? T('mlDownloadDir') : T('geoDownloadDir')),
                      h('input', {
                        className: 'mb-input',
                        value: destDir,
                        spellCheck: false,
                        placeholder: DEFAULT_GEO_DIR,
                        onChange: function (event) { setDestDir(event.target.value) },
                      }),
                      Btn({
                        small: true,
                        variant: 'primary',
                        icon: 'download',
                        disabled: running || !selectedFiles().length,
                        onClick: startDownload,
                      }, running ? T('geoDownloading') : (source === 'ml' ? T('mlDownloadBtn') : T('geoDownloadBtn')))),
                    h('div', { className: 'mb-library-hint' },
                      isAbsoluteDir(destDir)
                        ? (source === 'ml' ? T('mlDownloadDir') : T('geoDownloadDirHint'))
                        : T('geoInvalidDir')),
                    running ? h(ProgressBar, { pct: job.pct, message: job.message, label: T('geoDownloading') }) : null,
                    done ? h('div', { className: 'mb-ok-text' }, T('geoJobDone') + (job.result && job.result.dir ? ' · ' + job.result.dir : '')) : null,
                    failed ? h(InlineError, { message: job.error || T('geoJobFailed') }) : null,
                    done
                      ? h('div', { className: 'mb-inline-row' },
                          Btn({
                            small: true,
                            variant: 'primary',
                            icon: 'importIcon',
                            disabled: importing.busy,
                            onClick: importDownloaded,
                          }, importing.busy ? T('geoImporting') : T('geoImportBtn')),
                          h('span', { className: 'mb-muted' }, T('geoImportDone')))
                      : null,
                    importing.error ? h(InlineError, { message: importing.error }) : null)
                : null))
        ))
    }

    function GeoDialog(props) {
      return h(SourceDialog, Object.assign({}, props, { source: 'geo' }))
    }

    function MlDialog(props) {
      return h(SourceDialog, Object.assign({}, props, { source: 'ml' }))
    }

    // 导入弹层：本机路径 / URL / 浏览器上传 + 数据格式 + 矩阵方向 + 样本表/临床表 + 候选点选
    //   接口：POST /dataset/open、POST /inspect、POST /dataset/upload（body=文件字节，x-dsh-filename）
    // ==========================================================================

    function formatOptions(formats) {
      var raw = formats && typeof formats === 'object' ? formats : {}
      var list = Array.isArray(raw.formats) ? raw.formats : []
      var out = []
      for (var index = 0; index < list.length; index++) {
        var item = list[index]
        if (typeof item === 'string') {
          out.push({ value: item, label: item })
          continue
        }
        if (!item || !item.id) continue
        if (item.available === false) continue
        out.push({ value: String(item.id), label: plainText(item.label || item.id) + (item.hint ? ' · ' + plainText(item.hint) : '') })
      }
      if (!out.length) {
        for (var fallback = 0; fallback < DATA_FORMATS.length; fallback++) {
          out.push({ value: DATA_FORMATS[fallback], label: DATA_FORMATS[fallback] })
        }
      }
      return out
    }

    function ImportDialog(props) {
      var api = props.api
      var sourceState = useState('path')
      var source = sourceState[0]
      var setSource = sourceState[1]
      var optionsState = useState({
        path: '',
        url: '',
        data_format: 'auto',
        feature_col: '',
        member: '',
        object: '',
        sheet: '',
        delimiter: '',
        sample_sheet: '',
        clinical: '',
        transpose: false,
        zero_as_missing: false,
      })
      var options = optionsState[0]
      var setOptions = optionsState[1]
      var busyState = useState({ inspect: false, open: false, error: '', notes: [], candidates: [], info: null })
      var busy = busyState[0]
      var setBusy = busyState[1]
      function patch(next) {
        setOptions(function (previous) { return Object.assign({}, previous, next || {}) })
      }
      function payload() {
        var raw = Object.assign({}, options)
        if (source === 'url') raw.path = ''
        if (source === 'path') raw.url = ''
        return raw
      }
      function runInspect() {
        setBusy(function (previous) { return Object.assign({}, previous, { inspect: true, error: '' }) })
        api.inspectSource(payload()).then(function (result) {
          var raw = result && typeof result === 'object' ? result : {}
          setBusy({
            inspect: false,
            open: false,
            error: '',
            notes: normalizeStringList(raw.notes),
            candidates: Array.isArray(raw.candidates) ? raw.candidates : [],
            info: {
              format: plainText(raw.format || ''),
              n_samples: asNumber(raw.n_samples),
              n_features: asNumber(raw.n_features),
              has_mz: raw.has_mz === true,
              has_rt: raw.has_rt === true,
              size: asNumber(raw.size),
            },
          })
        }).catch(function (error) {
          setBusy({ inspect: false, open: false, error: formatError(error), notes: [], candidates: [], info: null })
        })
      }
      function runOpen() {
        setBusy(function (previous) { return Object.assign({}, previous, { open: true, error: '' }) })
        api.openDataset(payload()).then(function (result) {
          setBusy(function (previous) { return Object.assign({}, previous, { open: false }) })
          if (props.onImported) props.onImported(result)
        }).catch(function (error) {
          setBusy({
            inspect: false,
            open: false,
            error: formatError(error),
            notes: [],
            candidates: Array.isArray(error && error.candidates) ? error.candidates : [],
            info: null,
          })
        })
      }
      function upload(file) {
        if (!file) return
        setBusy({ inspect: false, open: true, error: '', notes: [], candidates: [], info: null })
        api.upload(file).then(function (result) {
          setBusy(function (previous) { return Object.assign({}, previous, { open: false }) })
          if (props.onImported) props.onImported(result)
        }).catch(function (error) {
          setBusy({ inspect: false, open: false, error: formatError(error), notes: [], candidates: [], info: null })
        })
      }
      var formats = formatOptions(props.formats)
      var info = busy.info
      return h(Modal, {
        title: T('importTitle'),
        subtitle: T('importSubtitle'),
        wide: true,
        busy: busy.open || busy.inspect,
        onClose: props.onClose,
      },
        h('div', { className: 'mb-modal-body' },
          h('div', { className: 'mb-inline-row' },
            Btn({ small: true, variant: source === 'path' ? 'primary' : undefined, onClick: function () { setSource('path') } }, T('importSourceLocal')),
            Btn({ small: true, variant: source === 'url' ? 'primary' : undefined, onClick: function () { setSource('url') } }, T('importSourceUrl')),
            Btn({ small: true, variant: source === 'upload' ? 'primary' : undefined, onClick: function () { setSource('upload') } }, T('importChooseFile'))),
          source === 'upload'
            ? h('div', { className: 'mb-form' },
                h('input', {
                  className: 'mb-input',
                  type: 'file',
                  disabled: busy.open,
                  onChange: function (event) {
                    var file = event.target.files && event.target.files[0]
                    upload(file)
                  },
                }),
                h('div', { className: 'mb-muted' }, T('importUploadHint')))
            : h('div', { className: 'mb-form-grid' },
                h(FormField, { label: source === 'url' ? 'URL' : T('importSourceLocal'), full: true },
                  h('input', {
                    className: 'mb-input',
                    value: source === 'url' ? options.url : options.path,
                    spellCheck: false,
                    placeholder: source === 'url' ? T('importUrlPh') : T('importPathPh'),
                    onChange: function (event) {
                      if (source === 'url') patch({ url: event.target.value })
                      else patch({ path: event.target.value })
                    },
                  })),
                h(FormField, { label: T('importFormat') },
                  h('select', {
                    className: 'mb-input mb-select',
                    value: options.data_format,
                    onChange: function (event) { patch({ data_format: event.target.value }) },
                  }, formats.map(function (option) {
                    return h('option', { key: option.value, value: option.value }, option.label)
                  }))),
                h(FormField, { label: T('importFeatureCol') },
                  h('input', {
                    className: 'mb-input',
                    value: options.feature_col,
                    spellCheck: false,
                    placeholder: T('importFeatureColPh'),
                    onChange: function (event) { patch({ feature_col: event.target.value }) },
                  })),
                h(FormField, { label: T('importDelimiter') },
                  h('input', {
                    className: 'mb-input',
                    value: options.delimiter,
                    spellCheck: false,
                    placeholder: T('importDelimiterPh'),
                    onChange: function (event) { patch({ delimiter: event.target.value }) },
                  })),
                h(FormField, { label: T('importSheet') },
                  h('input', {
                    className: 'mb-input',
                    value: options.sheet,
                    spellCheck: false,
                    placeholder: 'Sheet1',
                    onChange: function (event) { patch({ sheet: event.target.value }) },
                  })),
                h(FormField, { label: T('importMember') },
                  h('input', {
                    className: 'mb-input',
                    value: options.member,
                    spellCheck: false,
                    onChange: function (event) { patch({ member: event.target.value }) },
                  })),
                h(FormField, { label: T('importObject') },
                  h('input', {
                    className: 'mb-input',
                    value: options.object,
                    spellCheck: false,
                    onChange: function (event) { patch({ object: event.target.value }) },
                  }))),
          source === 'upload'
            ? null
            : h('details', { className: 'mb-advanced' },
                h('summary', null, T('importAdvanced')),
                h('div', { className: 'mb-form-grid' },
                  h(FormField, { label: T('importSampleSheet'), full: true },
                    h('input', {
                      className: 'mb-input',
                      value: options.sample_sheet,
                      spellCheck: false,
                      placeholder: '/absolute/path/sample_info.csv',
                      onChange: function (event) { patch({ sample_sheet: event.target.value }) },
                    })),
                  h(FormField, { label: T('importClinical'), full: true },
                    h('input', {
                      className: 'mb-input',
                      value: options.clinical,
                      spellCheck: false,
                      placeholder: '/absolute/path/clinical.csv',
                      onChange: function (event) { patch({ clinical: event.target.value }) },
                    })),
                  h('label', { className: 'mb-check' },
                    h('input', {
                      type: 'checkbox',
                      checked: options.transpose,
                      onChange: function (event) { patch({ transpose: event.target.checked }) },
                    }),
                    h('span', null, T('importTranspose'))),
                  h('label', { className: 'mb-check' },
                    h('input', {
                      type: 'checkbox',
                      checked: options.zero_as_missing,
                      onChange: function (event) { patch({ zero_as_missing: event.target.checked }) },
                    }),
                    h('span', null, T('importZeroAsMissing'))))),
          h('div', { className: 'mb-muted' }, T('importHint')),
          busy.error ? h(InlineError, { message: busy.error }) : null,
          info
            ? h(SectionCard, { title: T('importInspectTitle') },
                h('div', { className: 'mb-inline-row' },
                  h('span', { className: 'mb-tag' }, info.format || T('unknown')),
                  !isNil(info.n_samples) ? h('span', { className: 'mb-tag' }, T('samples') + ' ' + fmtNumber(info.n_samples)) : null,
                  !isNil(info.n_features) ? h('span', { className: 'mb-tag' }, T('features') + ' ' + fmtNumber(info.n_features)) : null,
                  info.has_mz ? h('span', { className: 'mb-tag mb-tag-ok' }, 'm/z') : null,
                  info.has_rt ? h('span', { className: 'mb-tag mb-tag-ok' }, 'RT') : null,
                  !isNil(info.size) ? h('span', { className: 'mb-tag' }, formatBytes(info.size)) : null),
                busy.notes.length
                  ? h('div', { className: 'mb-note-list' }, busy.notes.map(function (note, index) {
                      return h('div', { key: 'note-' + index }, '· ' + note)
                    }))
                  : null)
            : null,
          busy.candidates.length
            ? h(SectionCard, { title: T('importCandidates'), extra: String(busy.candidates.length) },
                h('div', { className: 'mb-candidates' }, busy.candidates.map(function (candidate) {
                  var value = typeof candidate === 'string' ? candidate : plainText(candidate && (candidate.member || candidate.name || candidate.path))
                  var on = options.member === value
                  return h('button', {
                    key: value,
                    type: 'button',
                    className: 'mb-candidate' + (on ? ' mb-candidate-on' : ''),
                    onClick: function () { patch({ member: value }) },
                  }, value)
                })))
            : null),
        h('div', { className: 'mb-modal-foot' },
          Btn({ small: true, icon: 'search', disabled: busy.inspect || busy.open, onClick: runInspect },
            busy.inspect ? T('importInspectBusy') : T('importInspect')),
          Btn({ small: true, variant: 'primary', icon: 'importIcon', disabled: busy.open || busy.inspect, onClick: runOpen },
            busy.open ? T('importBusy') : T('importStart'))))
    }

    // ==========================================================================
    // 分析目录抽屉：/analyses 按 group 分组 + 参数表单 + 运行 + 结果
    // ==========================================================================

    function AnalysisDrawer(props) {
      var catalog = props.catalog || { loading: false, error: '', data: normalizeAnalyses({}) }
      var dataset = props.dataset
      var data = catalog.data
      var selectState = useState('')
      var selectedId = selectState[0]
      var setSelectedId = selectState[1]
      var valuesState = useState({})
      var values = valuesState[0]
      var setValues = valuesState[1]
      var meta = null
      var list = Array.isArray(data.analyses) ? data.analyses : []
      for (var index = 0; index < list.length; index++) {
        if (list[index].id === selectedId) meta = list[index]
      }
      function select(item) {
        if (!item || item.available === false) return
        setSelectedId(item.id)
        setValues(analysisDefaults(item.params))
      }
      function patch(next) {
        setValues(function (previous) { return Object.assign({}, previous, next || {}) })
      }
      var grouped = groupBy(list, 'group')
      var groupNames = Array.isArray(data.groups) ? data.groups.slice() : []
      var extraGroups = Object.keys(grouped)
      for (var groupIndex = 0; groupIndex < extraGroups.length; groupIndex++) {
        if (groupNames.indexOf(extraGroups[groupIndex]) < 0) groupNames.push(extraGroups[groupIndex])
      }
      var run = (props.runs && meta && props.runs[meta.id]) || null
      var groupNodes = groupNames.map(function (group) {
        var items = grouped[group] || []
        if (!items.length) return null
        return h('div', { key: 'group-' + group, className: 'mb-analysis-group' },
          h('div', { className: 'mb-analysis-group-title' }, group),
          items.map(function (item) {
            var disabled = item.available === false
            return h('button', {
              key: item.id,
              type: 'button',
              className: 'mb-analysis-item' + (meta && meta.id === item.id ? ' mb-analysis-item-on' : ''),
              disabled: disabled,
              title: disabled ? (item.reason || T('analysisUnavailable')) : (item.method || item.title),
              onClick: function () { select(item) },
            },
              h('div', { className: 'mb-analysis-main' },
                h('div', { className: 'mb-analysis-name' }, item.title || item.id),
                h('span', { className: 'mb-tag' + (disabled ? '' : ' mb-tag-ok') },
                  disabled ? T('analysisUnavailable') : T('statusOk'))),
              item.method ? h('div', { className: 'mb-analysis-method' }, item.method) : null,
              disabled && item.reason ? h('div', { className: 'mb-analysis-reason' }, item.reason) : null)
          }))
      })
      var runner = null
      if (!dataset) {
        runner = h(Notice, { text: T('analysisNeedDataset') })
      } else if (!list.length) {
        runner = h(Notice, { text: catalog.loading ? T('loading') : T('analysisEmpty') })
      } else if (!meta) {
        runner = h(Notice, { text: T('analysisSelectHint') })
      } else {
        runner = h('div', { className: 'mb-analysis-runner' },
          h('div', { className: 'mb-analysis-head' },
            h('div', { className: 'mb-analysis-title' }, meta.title || meta.id),
            meta.method ? h('div', { className: 'mb-analysis-method' }, T('analysisMethod') + '：' + meta.method) : null),
          meta.needs && meta.needs.length
            ? h('div', { className: 'mb-inline-row' },
                h('span', { className: 'mb-toolbar-label' }, T('analysisNeeds')),
                meta.needs.map(function (need) {
                  var stepMeta = stepById(need)
                  return h('span', { key: 'need-' + need, className: 'mb-tag' }, stepMeta ? stepMeta.title : need)
                }))
            : null,
          meta.available === false && meta.reason ? h('div', { className: 'mb-warn-text' }, meta.reason) : null,
          h(ParamForm, {
            params: meta.params,
            values: values,
            columns: props.columns || [],
            disabled: run && run.busy === true,
            onChange: patch,
          }),
          h('div', { className: 'mb-inline-row' },
            Btn({
              small: true,
              variant: 'primary',
              icon: 'play',
              disabled: !dataset || (run && run.busy === true) || meta.available === false,
              onClick: function () { if (props.onRun) props.onRun(meta.id, values || {}) },
            }, run && run.busy ? T('analysisRunning') : T('analysisRun'))),
          run && run.error ? h(InlineError, { message: run.error }) : null,
          run && run.result
            ? h(ResultView, {
                result: run.result,
                onExport: props.tabExport && props.runExport
                  ? function () { props.runExport(props.tabExport.what) }
                  : null,
                exportBusy: props.tabExport ? !!(props.exports && props.exports[props.tabExport.what] && props.exports[props.tabExport.what].busy) : false,
                exportLabel: props.tabExport ? T(props.tabExport.labelKey) : '',
                exportFile: props.tabExport && props.exports && props.exports[props.tabExport.what]
                  ? props.exports[props.tabExport.what].file || ''
                  : '',
              })
            : null)
      }
      return h(Modal, {
        title: T('analysisTitle'),
        subtitle: T('analysisSubtitle'),
        wide: true,
        onClose: props.onClose,
      },
        h('div', { className: 'mb-modal-body' },
          catalog.loading ? h(LoadingLine, { text: T('loading') }) : null,
          catalog.error ? h(InlineError, { message: catalog.error }) : null,
          h('div', { className: 'mb-analysis-wrap' },
            h('div', { className: 'mb-analysis-list' }, groupNodes),
            runner)),
        h('div', { className: 'mb-modal-foot' },
          Btn({ icon: 'close', onClick: props.onClose, disabled: run && run.busy === true }, T('close'))))
    }

    // ==========================================================================
    // 整页工作台：状态容器 + 三栏布局 + 三个弹层（GEO / MetaboLights / 导入）+ 分析抽屉
    //   · 左栏状态由 STEPS + dataset.steps 派生（stepStatusOf）
    //   · 步骤/分析结果只保存在本会话（切数据集清空；后端 history/steps 仍保留）
    //   · 所有故障单点包 PluginErrorBoundary，不黑整页
    // ==========================================================================

    function pickDataset(datasets, id) {
      var list = Array.isArray(datasets) ? datasets : []
      for (var index = 0; index < list.length; index++) {
        if (list[index].id === id) return list[index]
      }
      return null
    }

    function datasetIdOf(payload) {
      if (!payload) return ''
      if (typeof payload === 'string') return plainText(payload)
      if (typeof payload !== 'object') return ''
      if (payload.dataset && typeof payload.dataset === 'object') return plainText(payload.dataset.id || '')
      return plainText(payload.id || payload.active || '')
    }

    /** 步骤卡上的结果摘要：优先 bullets，其次 metrics，最后图表计数。 */
    function resultSummaryText(result) {
      if (!result) return ''
      var bullets = Array.isArray(result.bullets) ? result.bullets : []
      if (bullets.length) return bullets.slice(0, 2).join('；')
      var metrics = result.metrics && typeof result.metrics === 'object' ? result.metrics : {}
      var keys = Object.keys(metrics)
      if (keys.length) {
        var parts = []
        for (var index = 0; index < keys.length && parts.length < 4; index++) {
          var value = metrics[keys[index]]
          parts.push(keys[index] + ' ' + (typeof value === 'number' ? fmtNumber(value) : formatCell(value, false)))
        }
        return parts.join(' · ')
      }
      var counts = []
      if (result.tables && result.tables.length) counts.push(result.tables.length + ' ' + T('resultColumn'))
      if (result.figures && result.figures.length) counts.push(result.figures.length + ' ' + T('plot'))
      return counts.join(' · ')
    }

    function WorkbenchBody() {
      var api = useMemo(function () { return createApi() }, [])
      var bootState = useState({ loading: true, error: '', health: normalizeHealth({}), formats: null })
      var boot = bootState[0]
      var setBoot = bootState[1]
      var bootNonceState = useState(0)
      var bootNonce = bootNonceState[0]
      var setBootNonce = bootNonceState[1]
      var stateState = useState(function () { return normalizeState({}) })
      var snapshot = stateState[0]
      var setSnapshot = stateState[1]
      var previewState = useState(function () { return normalizePreview({}) })
      var preview = previewState[0]
      var setPreview = previewState[1]
      var valuesState = useState({})
      var stepValues = valuesState[0]
      var setStepValues = valuesState[1]
      var resultsState = useState({})
      var stepResults = resultsState[0]
      var setStepResults = resultsState[1]
      var summariesState = useState({})
      var stepSummaries = summariesState[0]
      var setStepSummaries = summariesState[1]
      var errorsState = useState({})
      var stepErrors = errorsState[0]
      var setStepErrors = errorsState[1]
      var busyStepState = useState('')
      var stepBusy = busyStepState[0]
      var setStepBusy = busyStepState[1]
      var failedState = useState('')
      var failedStep = failedState[0]
      var setFailedStep = failedState[1]
      var plotsState = useState({})
      var plots = plotsState[0]
      var setPlots = plotsState[1]
      var exportsState = useState({})
      var exportsMap = exportsState[0]
      var setExports = exportsState[1]
      var catalogState = useState({ loading: false, error: '', data: normalizeAnalyses({}) })
      var catalog = catalogState[0]
      var setCatalog = catalogState[1]
      var runsState = useState({})
      var runs = runsState[0]
      var setRuns = runsState[1]
      var tabState = useState('overview')
      var tab = tabState[0]
      var setTab = tabState[1]
      var dialogState = useState({ geo: false, ml: false, import: false, analysis: false })
      var dialogs = dialogState[0]
      var setDialogs = dialogState[1]
      var copiedState = useState('')
      var copiedKey = copiedState[0]
      var setCopiedKey = copiedState[1]
      var refreshingState = useState(false)
      var refreshing = refreshingState[0]
      var setRefreshing = refreshingState[1]
      var busyState = useState(false)
      var busy = busyState[0]
      var setBusy = busyState[1]
      var bannerState = useState({ kind: '', text: '' })
      var banner = bannerState[0]
      var setBanner = bannerState[1]
      var activeRef = useRef('')
      var catalogRef = useRef('')
      var copyTimerRef = useRef(null)

      // 首屏：health + state + formats 并发拉取；bootNonce 用于「重试」。
      useEffect(function () {
        var cancelled = false
        setBoot(function (previous) { return Object.assign({}, previous, { loading: true, error: '' }) })
        Promise.all([
          api.getHealth().catch(function () { return null }),
          api.getState(),
          api.getFormats().catch(function () { return null }),
        ]).then(function (list) {
          if (cancelled) return
          var next = normalizeState(list[1])
          var health = normalizeHealth(list[0] || {})
          if (!health.ready && next.engine.ready) health = next.engine
          setSnapshot(next)
          setBoot({ loading: false, error: '', health: health, formats: list[2] })
        }).catch(function (error) {
          if (cancelled) return
          setBoot({ loading: false, error: formatError(error), health: normalizeHealth({}), formats: null })
        })
        return function () { cancelled = true }
      }, [bootNonce, api])

      // 切换数据集：清掉本会话的图/结果/分析结果，重新拉 preview 与 analyses 目录。
      useEffect(function () {
        if (activeRef.current === snapshot.active) return undefined
        activeRef.current = snapshot.active
        catalogRef.current = snapshot.active
        setStepResults({})
        setStepSummaries({})
        setStepErrors({})
        setStepBusy('')
        setFailedStep('')
        setExports({})
        setPlots({})
        setStepValues({})
        setRuns({})
        setCopiedKey('')
        if (!snapshot.active) {
          setPreview(normalizePreview({}))
          setCatalog({ loading: false, error: '', data: normalizeAnalyses({}) })
          return undefined
        }
        var cancelled = false
        api.preview(snapshot.active)
          .then(function (payload) { if (!cancelled) setPreview(normalizePreview(payload)) })
          .catch(function () { if (!cancelled) setPreview(normalizePreview({})) })
        setCatalog(function (previous) { return Object.assign({}, previous, { loading: true, error: '' }) })
        api.analyses(snapshot.active)
          .then(function (payload) {
            if (cancelled) return
            setCatalog({ loading: false, error: '', data: normalizeAnalyses(payload) })
          })
          .catch(function (error) {
            if (cancelled) return
            setCatalog({ loading: false, error: formatError(error), data: normalizeAnalyses({}) })
          })
        return function () { cancelled = true }
      }, [snapshot.active, api])

      useEffect(function () {
        return function () {
          if (copyTimerRef.current) clearTimeout(copyTimerRef.current)
        }
      }, [])

      var dataset = pickDataset(snapshot.datasets, snapshot.active)
      var health = snapshot.engine && snapshot.engine.ready ? snapshot.engine : boot.health
      var columns = dataset && dataset.sample_columns && dataset.sample_columns.length
        ? dataset.sample_columns
        : (preview.columns && preview.columns.length ? preview.columns : [])

      var statuses = {}
      var paramValues = {}
      for (var stepsLoop = 0; stepsLoop < STEPS.length; stepsLoop++) {
        var stepMeta = STEPS[stepsLoop]
        statuses[stepMeta.id] = stepStatusOf(stepMeta, dataset, stepBusy, failedStep)
        paramValues[stepMeta.id] = Object.assign({}, defaultParams(stepMeta), stepValues[stepMeta.id] || {})
      }

      function refreshState(options) {
        var opts = options && typeof options === 'object' ? options : {}
        if (!opts.silent) setRefreshing(true)
        return api.getState().then(function (payload) {
          var next = normalizeState(payload)
          setSnapshot(next)
          setBoot(function (previous) {
            var merged = previous.health
            if (next.engine.ready && !merged.ready) merged = next.engine
            return Object.assign({}, previous, { health: merged })
          })
          if (!opts.silent) setRefreshing(false)
          return next
        }).catch(function (error) {
          if (!opts.silent) {
            setRefreshing(false)
            setBanner({ kind: 'bad', text: formatError(error) })
          }
          return null
        })
      }

      function refreshPreview(targetId) {
        var id = plainText(targetId) || snapshot.active
        if (!id) {
          setPreview(normalizePreview({}))
          return Promise.resolve(null)
        }
        return api.preview(id).then(function (payload) {
          var normalized = normalizePreview(payload)
          setPreview(normalized)
          return normalized
        }).catch(function () { return null })
      }

      function refreshCatalog() {
        var id = snapshot.active
        if (!id) return Promise.resolve(null)
        setCatalog(function (previous) { return Object.assign({}, previous, { loading: true, error: '' }) })
        return api.analyses(id).then(function (payload) {
          var data = normalizeAnalyses(payload)
          setCatalog({ loading: false, error: '', data: data })
          return data
        }).catch(function (error) {
          setCatalog({ loading: false, error: formatError(error), data: normalizeAnalyses({}) })
          return null
        })
      }

      function loadPlot(kind, params, force) {
        var id = snapshot.active
        if (!id) return
        var safeParams = params && typeof params === 'object' ? params : {}
        var key = plotKey(kind, safeParams)
        var entry = plots[key]
        if (!force && entry && (entry.loading || entry.figure)) return
        setPlots(function (previous) {
          var next = Object.assign({}, previous)
          next[key] = { loading: true, error: '', figure: null }
          return next
        })
        api.plot(id, kind, safeParams).then(function (payload) {
          var figure = normalizeFigure(payload && payload.figure ? payload.figure : payload)
          setPlots(function (previous) {
            var next = Object.assign({}, previous)
            next[key] = { loading: false, error: figure ? '' : T('noFigure'), figure: figure }
            return next
          })
        }).catch(function (error) {
          setPlots(function (previous) {
            var next = Object.assign({}, previous)
            next[key] = { loading: false, error: formatError(error), figure: null }
            return next
          })
        })
      }

      function plotOnce(kind, params) {
        var id = snapshot.active
        if (!id) return Promise.reject(new Error(T('analysisNeedDataset')))
        return api.plot(id, kind, params && typeof params === 'object' ? params : {}).then(function (payload) {
          var figure = normalizeFigure(payload && payload.figure ? payload.figure : payload)
          if (!figure) throw new Error(T('noFigure'))
          return figure
        })
      }

      function runExport(what) {
        var id = snapshot.active
        if (!id) return
        setExports(function (previous) {
          var next = Object.assign({}, previous)
          next[what] = Object.assign({}, previous[what] || {}, { busy: true, error: '' })
          return next
        })
        api.exportData(id, what).then(function (payload) {
          var file = plainText(payload && (payload.file || payload.path)) ||
            plainText(payload && payload.result && payload.result.file)
          setExports(function (previous) {
            var next = Object.assign({}, previous)
            next[what] = { busy: false, file: file, error: '' }
            return next
          })
          refreshState({ silent: true })
        }).catch(function (error) {
          setExports(function (previous) {
            var next = Object.assign({}, previous)
            next[what] = { busy: false, file: '', error: formatError(error) }
            return next
          })
        })
      }

      function runAnalysis(analysisId, params) {
        var id = snapshot.active
        if (!id) return
        setRuns(function (previous) {
          var next = Object.assign({}, previous)
          var previousRun = previous[analysisId] || {}
          next[analysisId] = { busy: true, error: '', result: previousRun.result || null }
          return next
        })
        api.analysis(id, analysisId, params && typeof params === 'object' ? params : {}).then(function (payload) {
          var result = normalizeResult(payload)
          setRuns(function (previous) {
            var next = Object.assign({}, previous)
            next[analysisId] = { busy: false, error: '', result: result }
            return next
          })
          refreshState({ silent: true })
        }).catch(function (error) {
          setRuns(function (previous) {
            var next = Object.assign({}, previous)
            next[analysisId] = { busy: false, error: formatError(error), result: null }
            return next
          })
        })
      }

      function handleRunStep(stepId, params) {
        var step = stepById(stepId)
        var id = snapshot.active
        if (!step || !id) return Promise.resolve(null)
        var requestParams = params && typeof params === 'object' && !Array.isArray(params)
          ? params
          : collectStepParams(step, stepValues[stepId] || {})
        setStepBusy(stepId)
        setFailedStep('')
        setBanner({ kind: '', text: '' })
        setStepErrors(function (previous) {
          var next = Object.assign({}, previous)
          delete next[stepId]
          return next
        })
        return api.runStep(id, stepId, requestParams).then(function (payload) {
          var result = normalizeResult(payload)
          setStepResults(function (previous) {
            var next = Object.assign({}, previous)
            next[stepId] = result
            return next
          })
          setStepSummaries(function (previous) {
            var next = Object.assign({}, previous)
            next[stepId] = resultSummaryText(result)
            return next
          })
          setStepValues(function (previous) {
            var next = Object.assign({}, previous)
            next[stepId] = Object.assign({}, requestParams)
            return next
          })
          setStepBusy('')
          setPlots({})
          setBanner({ kind: '', text: tf(T, 'runStepDone', step.title) })
          refreshCatalog()
          if (TAB_FOR_STEP[stepId]) setTab(TAB_FOR_STEP[stepId])
          refreshState({ silent: true }).then(function () {
            refreshPreview()
          })
          return result
        }).catch(function (error) {
          setStepBusy('')
          setFailedStep(stepId)
          setStepErrors(function (previous) {
            var next = Object.assign({}, previous)
            next[stepId] = formatError(error)
            return next
          })
          return null
        })
      }

      function handleParamChange(stepId, patch) {
        var step = stepById(stepId)
        setStepValues(function (previous) {
          var next = Object.assign({}, previous)
          next[stepId] = Object.assign({}, step ? defaultParams(step) : {}, previous[stepId] || {}, patch || {})
          return next
        })
      }

      function handleSelectDataset(id) {
        if (!id || id === snapshot.active || busy) return
        setBusy(true)
        setBanner({ kind: '', text: '' })
        api.selectDataset(id).then(function () {
          return refreshState({ silent: true })
        }).then(function () {
          setBusy(false)
        }).catch(function (error) {
          setBusy(false)
          setBanner({ kind: 'bad', text: formatError(error) })
        })
      }

      function handleCloseDataset(id) {
        if (!id || busy) return
        setBusy(true)
        api.closeDataset(id).then(function () {
          return refreshState({ silent: true })
        }).then(function () {
          setBusy(false)
        }).catch(function (error) {
          setBusy(false)
          setBanner({ kind: 'bad', text: formatError(error) })
        })
      }

      function handleImported(payload) {
        var targetId = datasetIdOf(payload)
        setDialogs({ geo: false, ml: false, import: false, analysis: false })
        setTab('overview')
        setBusy(true)
        refreshState({ silent: true }).then(function (next) {
          var active = next ? next.active : ''
          if (targetId && active && targetId !== active) {
            return api.selectDataset(targetId).then(function () { return refreshState({ silent: true }) })
          }
          return null
        }).then(function () {
          setBusy(false)
          refreshPreview(targetId)
          refreshCatalog()
        }).catch(function (error) {
          setBusy(false)
          setBanner({ kind: 'bad', text: formatError(error) })
        })
      }

      function loadDemo() {
        if (busy) return
        setBusy(true)
        setBanner({ kind: '', text: '' })
        api.demo()
          .then(function (payload) { handleImported(payload) })
          .catch(function (error) {
            setBusy(false)
            setBanner({ kind: 'bad', text: formatError(error) })
          })
      }

      function copyText(text, key) {
        var value = plainText(text, 4000)
        if (!value) return
        var markCopied = function () {
          setCopiedKey(key || 'copy')
          if (copyTimerRef.current) clearTimeout(copyTimerRef.current)
          copyTimerRef.current = setTimeout(function () { setCopiedKey('') }, 1800)
        }
        var fallback = function () {
          try {
            if (typeof document === 'undefined' || !document.body) return false
            var area = document.createElement('textarea')
            area.value = value
            area.setAttribute('readonly', 'readonly')
            area.style.position = 'fixed'
            area.style.top = '-1000px'
            area.style.opacity = '0'
            document.body.appendChild(area)
            area.select()
            var ok = typeof document.execCommand === 'function' ? document.execCommand('copy') : false
            document.body.removeChild(area)
            return ok === true
          } catch (ignored) {
            return false
          }
        }
        var clipboard = null
        try { clipboard = typeof navigator === 'undefined' ? null : navigator.clipboard } catch (ignored) { clipboard = null }
        if (clipboard && typeof clipboard.writeText === 'function') {
          Promise.resolve(clipboard.writeText(value)).then(markCopied).catch(function () {
            if (fallback()) markCopied()
            else setBanner({ kind: 'bad', text: T('copyFailed') })
          })
          return
        }
        if (fallback()) markCopied()
        else setBanner({ kind: 'bad', text: T('copyFailed') })
      }

      function openDialog(name) {
        setDialogs(function (previous) {
          var next = Object.assign({}, previous)
          next[name] = true
          return next
        })
      }

      function closeDialog(name) {
        setDialogs(function (previous) {
          var next = Object.assign({}, previous)
          next[name] = false
          return next
        })
      }

      var tabProps = {
        dataset: dataset,
        preview: preview,
        columns: columns,
        api: api,
        catalog: catalog,
        runs: runs,
        plots: plots,
        loadPlot: loadPlot,
        stepResults: stepResults,
        stepBusy: stepBusy,
        exports: exportsMap,
        runExport: runExport,
        runAnalysis: runAnalysis,
        onRunAnalysis: runAnalysis,
        plotOnce: plotOnce,
        runStep: handleRunStep,
        onRunStep: handleRunStep,
        onAttached: function () { refreshState({ silent: true }); refreshCatalog() },
        onOpenAnalysis: function () { openDialog('analysis') },
        busy: busy,
      }

      function renderTab() {
        if (tab === 'qc') return h(QcTab, tabProps)
        if (tab === 'multivariate') return h(MultivariateTab, tabProps)
        if (tab === 'de') return h(DeTab, tabProps)
        if (tab === 'pathway') return h(PathwayTab, tabProps)
        if (tab === 'biomarker') return h(BiomarkerTab, tabProps)
        if (tab === 'annotation') return h(AnnotationTab, tabProps)
        if (tab === 'records') return h(RecordsTab, tabProps)
        if (tab === 'log') return h(LogTab, { api: api })
        return h(OverviewTab, Object.assign({}, tabProps, {
          health: health,
          suggestions: snapshot.suggestions,
          busyStep: stepBusy ? (stepById(stepBusy) || { title: stepBusy }).title : '',
          copiedKey: copiedKey,
          onCopy: copyText,
          onOpenImport: function () { openDialog('import') },
        }))
      }

      var bannerNode = banner.text
        ? h('div', { className: 'mb-banner' + (banner.kind === 'bad' ? ' mb-banner-bad' : '') },
            h('span', { className: 'mb-banner-text', title: banner.text }, banner.text),
            Btn({
              small: true,
              variant: 'ghost',
              icon: 'close',
              onClick: function () { setBanner({ kind: '', text: '' }) },
            }, T('close')))
        : null

      var engineNode = !boot.loading && !boot.error && health && !health.ready
        ? h('div', { className: 'mb-banner' },
            h('span', { className: 'mb-banner-text', title: T('engineGuideBody') + ' ' + T('engineGuideSteps') },
              T('engineGuideTitle') + ' · ' + (health.error || T('engineBooting'))))
        : null

      var centerNode = boot.loading
        ? h('div', { className: 'mb-pane' }, h(LoadingLine, { text: T('loadingData') }))
        : (boot.error
            ? h('div', { className: 'mb-pane' },
                h(ErrorBanner, {
                  title: T('errorTitle'),
                  message: boot.error,
                  retryLabel: T('retry'),
                  onRetry: function () { setBootNonce(function (previous) { return previous + 1 }) },
                }))
            : h(PluginErrorBoundary, { key: 'pane-' + tab }, renderTab()))

      return h('div', { className: 'mb-root' },
        h('style', { 'data-plugin': NS }, CSS),
        h(PluginErrorBoundary, null,
          h(TopBar, {
            health: health,
            metabolights: health ? health.metabolights : null,
            datasets: snapshot.datasets,
            activeId: snapshot.active,
            busy: busy || refreshing,
            refreshing: refreshing,
            onSelect: handleSelectDataset,
            onCloseDataset: handleCloseDataset,
            onGeo: function () { openDialog('geo') },
            onMl: function () { openDialog('ml') },
            onImport: function () { openDialog('import') },
            onDemo: loadDemo,
            onRefresh: function () {
              refreshState({ silent: false }).then(function (next) {
                if (next) { refreshPreview(); refreshCatalog() }
              })
            },
          })),
        bannerNode,
        h('div', { className: 'mb-body' },
          h(PluginErrorBoundary, null,
            h(PipelinePanel, {
              dataset: dataset,
              statuses: statuses,
              values: paramValues,
              columns: columns,
              summaries: stepSummaries,
              errors: stepErrors,
              busyStep: stepBusy,
              onParamChange: handleParamChange,
              onRun: handleRunStep,
            })),
          h('div', { className: 'mb-center' },
            engineNode,
            h(PluginErrorBoundary, null, h(TabsBar, { active: tab, onChange: setTab })),
            centerNode),
          h(PluginErrorBoundary, null,
            h(AgentPanel, {
              dataset: dataset,
              health: health,
              suggestions: snapshot.suggestions,
              activity: snapshot.activity,
              copiedKey: copiedKey,
              onCopy: copyText,
            }))),
        dialogs.geo
          ? h(PluginErrorBoundary, { key: 'dialog-geo' },
              h(GeoDialog, {
                api: api,
                onImported: handleImported,
                onClose: function () { closeDialog('geo') },
              }))
          : null,
        dialogs.ml
          ? h(PluginErrorBoundary, { key: 'dialog-ml' },
              h(MlDialog, {
                api: api,
                onImported: handleImported,
                onClose: function () { closeDialog('ml') },
              }))
          : null,
        dialogs.import
          ? h(PluginErrorBoundary, { key: 'dialog-import' },
              h(ImportDialog, {
                api: api,
                formats: boot.formats,
                onImported: handleImported,
                onClose: function () { closeDialog('import') },
              }))
          : null,
        dialogs.analysis
          ? h(PluginErrorBoundary, { key: 'dialog-analysis' },
              h(AnalysisDrawer, {
                dataset: dataset,
                columns: columns,
                catalog: catalog,
                runs: runs,
                onRun: runAnalysis,
                onClose: function () { closeDialog('analysis') },
              }))
          : null)
    }

    /** 整页入口：整体再包一层错误边界（单点异常不黑整页）。 */
    function WorkbenchPage() {
      return h(PluginErrorBoundary, null, h(WorkbenchBody))
    }

    // ==========================================================================
    // 插件入口：locale 字典 + 两个槽（整页 main + 侧边栏入口 sidebar.panellist）
    // ==========================================================================

    function apply(ctx) {
      var locale = ctx && ctx.locale
      if (locale && typeof locale.register === 'function') {
        var registerDictionaries = function () {
          try {
            return locale.register(NS, { zh: DICT_ZH, en: DICT_EN })
          } catch (error) {
            warnPlugin('failed to register locale dictionaries', error)
            return undefined
          }
        }
        try {
          if (typeof ctx.effect === 'function') ctx.effect(registerDictionaries, NS + ': locale')
          else registerDictionaries()
        } catch (error) {
          warnPlugin('failed to install locale effect', error)
        }
        T = createTranslator(locale)
      }

      var slots = ctx && ctx.slots
      if (!slots && ctx && typeof ctx.get === 'function') {
        try { slots = ctx.get('slots') } catch (error) { slots = null }
      }
      if (!slots || typeof slots.register !== 'function' || typeof slots.inject !== 'function') {
        warnPlugin('slots service unavailable; UI not registered')
        return
      }

      var disposers = []
      function keep(handle) {
        if (typeof handle === 'function') disposers.push(handle)
        return handle
      }

      try {
        if (typeof ctx.effect === 'function') {
          ctx.effect(function () {
            return function () {
              while (disposers.length) {
                var dispose = disposers.pop()
                try { dispose() } catch (ignored) { /* 单个 disposer 失败不影响其它 */ }
              }
            }
          }, NS + ': slot disposers')
        }
      } catch (error) {
        warnPlugin('failed to install slot cleanup', error)
      }

      try {
        keep(slots.inject('main', function () {
          return keep(slots.register({ name: 'main', key: 'metabolomics', locale: NS }, WorkbenchPage))
        }))
      } catch (error) {
        warnPlugin('failed to register main slot', error)
      }

      try {
        keep(slots.inject('sidebar.panellist', function () {
          return keep(slots.register({
            name: 'sidebar.panellist',
            id: 'metabolomics',
            order: 6,
            label: function () { return T('nav') },
            locale: NS,
          }, SidebarIcon))
        }))
      } catch (error) {
        warnPlugin('failed to register sidebar.panellist slot', error)
      }
    }

    // __test__ 只给自测/联调用（宿主不消费）：纯函数 + 组件引用。
    return {
      name: NS,
      inject: ['slots','locale'],
      apply: apply,
      __test__: Object.assign({}, MB_TEST, {
        components: {
          WorkbenchPage: WorkbenchPage,
          WorkbenchBody: WorkbenchBody,
          PipelinePanel: PipelinePanel,
          StepCard: StepCard,
          DatasetCard: DatasetCard,
          TopBar: TopBar,
          AgentPanel: AgentPanel,
          TabsBar: TabsBar,
          OverviewTab: OverviewTab,
          QcTab: QcTab,
          MultivariateTab: MultivariateTab,
          DeTab: DeTab,
          PathwayTab: PathwayTab,
          BiomarkerTab: BiomarkerTab,
          AnnotationTab: AnnotationTab,
          AnalysisTab: AnalysisTab,
          SampleScatter: SampleScatter,
          ModelQuality: ModelQuality,
          RecordsTab: RecordsTab,
          LogTab: LogTab,
          SourceDialog: SourceDialog,
          GeoDialog: GeoDialog,
          MlDialog: MlDialog,
          ImportDialog: ImportDialog,
          AnalysisDrawer: AnalysisDrawer,
          ResultView: ResultView,
          DataTable: DataTable,
          ParamForm: ParamForm,
          ParamControl: ParamControl,
          FigureBlock: FigureBlock,
          PlotBlock: PlotBlock,
          ProgressBar: ProgressBar,
          Modal: Modal,
          MetricCard: MetricCard,
          SectionCard: SectionCard,
          FormField: FormField,
          GuideBlock: GuideBlock,
          InlineError: InlineError,
          Notice: Notice,
          Spinner: Spinner,
          LoadingLine: LoadingLine,
          ErrorBanner: ErrorBanner,
          Icon: Icon,
          Btn: Btn,
          PluginErrorBoundary: PluginErrorBoundary,
          SidebarIcon: SidebarIcon,
          SidebarGlyph: SidebarGlyph,
        },
      }),
    }
  }

  // ==========================================================================
  // 顶层守卫：node 里只暴露纯函数给单测；浏览器里注册 Client 模块。
  // ==========================================================================

  if (typeof window === 'undefined') {
    globalThis.__MB_TEST__ = MB_TEST
  } else if (window.__ModuleLoader__ && typeof window.__ModuleLoader__.load === 'function') {
    window.__ModuleLoader__.load({ id: NS, factory: factory })
  }
})()
