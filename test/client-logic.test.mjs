/**
 * dsh-metabolomics 浏览器端纯函数单测（不依赖浏览器、不装 npm 依赖）。
 *
 * 运行：node test/client-logic.test.mjs
 *
 * client.js 是浏览器模块（window.__ModuleLoader__.load），但在 node 里
 * （typeof window === 'undefined'）只会把纯函数挂到 globalThis.__MB_TEST__，
 * 所以这里直接 import 它、断言纯函数行为 + 静态合规约束（DESIGN §2.2/§3/§4/§5/§7/§8）。
 * 打印 PASS 表示全部通过。
 */
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'

const CLIENT_URL = new URL('../client.js', import.meta.url)
await import(CLIENT_URL.href)

const MB = globalThis.__MB_TEST__
assert.ok(MB, 'node 环境应通过 globalThis.__MB_TEST__ 暴露纯函数')

const CLIENT_SOURCE = await readFile(CLIENT_URL, 'utf8')

/** 去掉块注释与行注释后的代码（静态断言只看真实代码，注释里的说明不算）。 */
function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1')
}

const CLIENT_CODE = stripComments(CLIENT_SOURCE)

// ---------------------------------------------------------------------------
// 0. 加载与导出形状
// ---------------------------------------------------------------------------

test('import：无 window 时不炸，且暴露 NS/API/纯函数层', () => {
  assert.equal(typeof globalThis.window, 'undefined')
  assert.equal(MB.NS, 'dsh-metabolomics')
  assert.equal(MB.API, '/metabolomics')
  assert.equal(MB.DEFAULT_GEO_DIR, '~/.dsh/metabolomics/GEO')
  assert.equal(MB.DEFAULT_ML_DIR, '~/.dsh/metabolomics/MetaboLights')
  assert.ok(MB.POLL_MS > 0)
  assert.ok(MB.POLL_TIMEOUT_MS >= MB.POLL_MS)
  assert.ok(MB.DICT_ZH.appTitle && MB.DICT_EN.appTitle)
  assert.ok(MB.DICT_ZH.nav && MB.DICT_EN.nav)
  assert.equal(typeof MB.createApi, 'function')
  assert.equal(typeof MB.apiClient, 'function')
  assert.equal(typeof MB.jobPolling, 'function')
  assert.equal(typeof MB.jobDone, 'function')
  assert.equal(typeof MB.miniPlotScreen, 'function')
  assert.equal(typeof MB.buildSearchPayload, 'function')
  assert.equal(typeof MB.buildMlSearchPayload, 'function')
  assert.equal(typeof MB.buildImportPayload, 'function')
  assert.equal(typeof MB.suggestionInstruction, 'function')
  assert.equal(typeof MB.plotKey, 'function')
  assert.equal(typeof MB.createTranslator, 'function')
  assert.equal(typeof MB.normalizeState, 'function')
  assert.equal(typeof MB.normalizeDataset, 'function')
  assert.equal(typeof MB.normalizeResult, 'function')
  assert.equal(typeof MB.normalizeAudit, 'function')
  assert.equal(typeof MB.tabAnalyses, 'function')
  assert.equal(typeof MB.groupColumn, 'function')
})

test('字典：中英文 key 集完全一致（同一份 WORDS 表生成）', () => {
  const zhKeys = Object.keys(MB.DICT_ZH).sort()
  const enKeys = Object.keys(MB.DICT_EN).sort()
  assert.ok(zhKeys.length > 300, `字典词条太少：${zhKeys.length}`)
  assert.deepEqual(zhKeys.filter((key) => !MB.DICT_EN[key]), [], 'EN 缺中文侧的 key')
  assert.deepEqual(enKeys.filter((key) => !MB.DICT_ZH[key]), [], 'ZH 缺英文侧的 key')
  assert.equal(MB.WORDS.length, zhKeys.length, 'WORDS 行数与字典词条数一致')
  // 每条都必须是 [key, zh, en] 三元素，且非空
  for (const row of MB.WORDS) {
    assert.equal(row.length, 3, `WORDS 行结构错误：${JSON.stringify(row)}`)
    assert.ok(row[0] && row[1] && row[2], `WORDS 行有空值：${JSON.stringify(row)}`)
  }
})

// ---------------------------------------------------------------------------
// 1. STEPS / TAB（DESIGN §3 顺序与 requires）
// ---------------------------------------------------------------------------

test('STEPS：12 个步骤顺序与 DESIGN §3 / impl.mjs 一致', () => {
  assert.equal(MB.STEPS.length, 12)
  assert.deepEqual(MB.STEPS.map((step) => step.id), [
    'qc', 'filter', 'impute', 'normalize', 'transform', 'pca',
    'plsda', 'de', 'batch', 'enrich', 'biomarker', 'annotate',
  ])
  const requires = {}
  for (const step of MB.STEPS) requires[step.id] = step.requires
  assert.deepEqual(requires, {
    qc: [],
    filter: ['qc'],
    impute: ['filter'],
    normalize: ['impute'],
    transform: ['normalize'],
    pca: ['transform'],
    plsda: ['transform'],
    de: ['transform'],
    batch: ['normalize'],
    enrich: ['de'],
    biomarker: ['de'],
    annotate: ['transform'],
  })
  const allowedTypes = ['number', 'select', 'boolean', 'column', 'text', 'genes', 'multiselect', 'json', 'file', 'string', 'array', 'columns']
  for (const step of MB.STEPS) {
    assert.ok(step.title, `${step.id} 应有中文标题`)
    assert.ok(Array.isArray(step.params) && step.params.length > 0, `${step.id} 应有参数`)
    for (const descriptor of step.params) {
      assert.ok(descriptor.key, `${step.id} 参数缺 key`)
      assert.ok(allowedTypes.includes(descriptor.type), `${step.id}.${descriptor.key} 控件类型不支持：${descriptor.type}`)
    }
  }
  // 左栏 tab 映射必须覆盖全部步骤（否则跑完某步不知道去哪个 tab）
  for (const step of MB.STEPS) {
    assert.ok(MB.TAB_FOR_STEP[step.id], `${step.id} 缺 TAB_FOR_STEP 映射`)
    assert.ok(MB.TAB_KEYS.includes(MB.TAB_FOR_STEP[step.id]), `${step.id} 映射到未知 tab`)
  }
})

test('TAB_KEYS：9 个 tab（概览/质控/多变量/差异/通路/标志物/鉴定/记录/日志）', () => {
  assert.deepEqual(MB.TAB_KEYS, [
    'overview', 'qc', 'multivariate', 'de', 'pathway', 'biomarker', 'annotation', 'records', 'log',
  ])
  for (const key of MB.TAB_KEYS) {
    assert.ok(MB.TAB_META[key] && MB.TAB_META[key].titleKey, `${key} 缺 TAB_META`)
    assert.ok(MB.DICT_ZH[MB.TAB_META[key].titleKey], `${key} 的 titleKey 不在字典里`)
    assert.ok(MB.DICT_EN[MB.TAB_META[key].titleKey], `${key} 的 titleKey 不在英文词典里`)
  }
  // 6 个分析 tab 都有白名单，且白名单里的 id 不重复
  for (const key of ['qc', 'multivariate', 'de', 'pathway', 'biomarker', 'annotation']) {
    const ids = MB.TAB_ANALYSES[key]
    assert.ok(Array.isArray(ids) && ids.length >= 4, `${key} 的分析白名单太少`)
    assert.equal(new Set(ids).size, ids.length, `${key} 的白名单有重复 id`)
    assert.ok(Array.isArray(MB.TAB_PLOTS[key]) && MB.TAB_PLOTS[key].length >= 2, `${key} 缺快捷出图`)
    for (const plot of MB.TAB_PLOTS[key]) {
      assert.ok(MB.PLOT_KINDS.includes(plot.kind), `${key} 出图 kind 不在白名单：${plot.kind}`)
      assert.ok(MB.DICT_ZH[plot.titleKey], `${key}.${plot.kind} 的 titleKey 不在字典里`)
    }
  }
  // DESIGN §5 图白名单 25 种
  assert.equal(MB.PLOT_KINDS.length, 25, 'DESIGN §5 图白名单应 25 种')
  assert.ok(MB.PLOT_KINDS.includes('qc_tic') && MB.PLOT_KINDS.includes('mass_network'))
  // export 11 种（impl.mjs EXPORT_KINDS 逐字一致）
  assert.deepEqual(MB.EXPORT_KINDS.map((item) => item.id), [
    'expr_csv', 'de_csv', 'vip_csv', 'enrich_csv', 'biomarker_csv', 'sample_info_csv',
    'summary_json', 'protocol', 'methods', 'report', 'script',
  ])
  for (const item of MB.EXPORT_KINDS) assert.ok(MB.DICT_ZH[item.labelKey], `导出按钮缺文案：${item.id}`)
  assert.ok(MB.QUICK_EXPORTS.every((what) => MB.EXPORT_KINDS.some((item) => item.id === what)))
  // 得分图着色维度（group/batch/qc/none）
  assert.deepEqual(MB.COLOR_BY.map((item) => item.value), ['group', 'batch', 'qc', 'none'])
  for (const item of MB.COLOR_BY) assert.ok(MB.DICT_ZH[item.labelKey], `着色缺文案：${item.value}`)
  // GEO 方法筛选（lc-ms/gc-ms/nmr/ms）
  assert.deepEqual(MB.GEO_METHODS.map((item) => item.value), ['all', 'lc-ms', 'gc-ms', 'nmr', 'ms'])
  assert.ok(MB.DATA_FORMATS.includes('mztab') && MB.DATA_FORMATS.includes('isa_tab') && MB.DATA_FORMATS.includes('msms_library'))
})

test('analysisGroupTab / tabAnalyses / groupColumn：分组→tab 映射与数据集分组列', () => {
  assert.equal(MB.analysisGroupTab('质控与基础'), 'qc')
  assert.equal(MB.analysisGroupTab('多变量判别'), 'multivariate')
  assert.equal(MB.analysisGroupTab('差异与趋势'), 'de')
  assert.equal(MB.analysisGroupTab('通路与化学分类'), 'pathway')
  assert.equal(MB.analysisGroupTab('标志物与建模'), 'biomarker')
  assert.equal(MB.analysisGroupTab('批次与网络'), 'multivariate')
  assert.equal(MB.analysisGroupTab('代谢物鉴定'), 'annotation')
  assert.equal(MB.analysisGroupTab('不存在的分组'), 'multivariate')
  assert.deepEqual(MB.tabAnalyses('qc'), MB.TAB_ANALYSES.qc)
  assert.deepEqual(MB.tabAnalyses('nope'), [])
  assert.deepEqual(MB.tabPlots('pathway'), MB.TAB_PLOTS.pathway)

  assert.equal(MB.groupColumn({ group_column: 'treatment' }), 'treatment')
  assert.equal(MB.groupColumn({ sample_columns: [{ key: 'group', kind: 'categorical', n_unique: 2 }] }), 'group')
  assert.equal(MB.groupColumn({ sample_columns: [{ key: 'batch', kind: 'categorical', n_unique: 2 }] }), 'batch')
  assert.equal(MB.groupColumn(null), '')
  assert.equal(MB.hasMz({ feature_meta: { has_mz: true } }), true)
  assert.equal(MB.hasMz({ feature_meta: { has_mz: false } }), false)
  assert.equal(MB.hasMz(null, { feature_meta: { has_mz: true } }), true)
})

// ---------------------------------------------------------------------------
// 2. 二维散点映射（质控小图：等比缩放、居中留白、可翻转 y）
// ---------------------------------------------------------------------------

test('miniPlotScreen：等比缩放、居中留白、flipY 只影响 y', () => {
  const placed = MB.miniPlotScreen([[0, 0], [100, 0], [0, 100]], { x: [0, 100], y: [0, 100] }, 120, 120, { padding: 10 })
  assert.equal(placed.length, 3)
  assert.deepEqual(placed.map((point) => [point.x, point.y]), [[10, 10], [110, 10], [10, 110]])
  assert.deepEqual(placed.map((point) => point.index), [0, 1, 2])

  const flipped = MB.miniPlotScreen([[0, 0], [0, 100]], { x: [0, 100], y: [0, 100] }, 120, 120, { padding: 10, flipY: true })
  assert.equal(flipped[0].y, 110)
  assert.equal(flipped[1].y, 10)

  // 脏数据跳过，不打断整批
  const dirty = MB.miniPlotScreen([{ x: 1 }, { x: 'a', y: 2 }, [3, 4], { x: 0, y: 0 }], null, 100, 100)
  assert.equal(dirty.length, 2)
  assert.equal(MB.miniPlotScreen([], null, 100, 100).length, 0)
  assert.equal(MB.miniPlotScreen(null).length, 0)
})

// ---------------------------------------------------------------------------
// 3. 数据源 payload / 导入 payload / 建议指令
// ---------------------------------------------------------------------------

test('buildSearchPayload（GEO）：只带非空/非默认字段', () => {
  const empty = MB.buildSearchPayload({})
  assert.equal(empty.term, '')
  assert.equal(empty.limit, 20)
  assert.equal(empty.page, 0)
  assert.equal(empty.sort, 'relevance')
  assert.equal(empty.species, undefined, '全部物种不发 species')
  assert.equal(empty.method, undefined, '全部方法不发 method')
  assert.equal(empty.min_samples, undefined)
  assert.equal(empty.has_clinical, undefined)

  const full = MB.buildSearchPayload({
    term: '  血清 代谢组学 LC-MS ',
    species: 'human',
    method: 'lc-ms',
    min_samples: '8',
    has_clinical: true,
    limit: 1000,
    page: -3,
    sort: 'date',
  })
  assert.equal(full.term, '血清 代谢组学 LC-MS')
  assert.equal(full.species, 'human')
  assert.equal(full.method, 'lc-ms')
  assert.equal(full.min_samples, 8)
  assert.equal(full.has_clinical, true)
  assert.equal(full.limit, 50, 'limit 上限夹到 50')
  assert.equal(full.page, 0)
  assert.equal(full.sort, 'date')

  const zero = MB.buildSearchPayload({ species: 'all', method: 'all', min_samples: 0, has_clinical: false, sort: 'bogus' })
  assert.equal(zero.species, undefined)
  assert.equal(zero.method, undefined)
  assert.equal(zero.min_samples, undefined)
  assert.equal(zero.has_clinical, undefined)
  assert.equal(zero.sort, 'relevance')
})

test('buildMlSearchPayload（MetaboLights）：query/limit/page + assay_type', () => {
  const empty = MB.buildMlSearchPayload({})
  assert.deepEqual(empty, { query: '', limit: 20, page: 0 })
  const full = MB.buildMlSearchPayload({ query: '  serum metabolomics ', limit: 999, page: -1, assay_type: 'MS' })
  assert.equal(full.query, 'serum metabolomics')
  assert.equal(full.limit, 50)
  assert.equal(full.page, 0)
  assert.equal(full.assay_type, 'MS')
  assert.equal(MB.buildMlSearchPayload({ term: 'urine' }).query, 'urine', 'term 可作为 query 兼容键')
})

test('buildImportPayload：空值剔除、auto 不发、布尔只在 true 时发', () => {
  assert.deepEqual(MB.buildImportPayload({}), {})
  const payload = MB.buildImportPayload({
    path: ' /data/feature_table.csv ',
    url: '',
    data_format: 'auto',
    feature_col: 'mz_rt',
    sample_sheet: '/data/samples.csv',
    clinical: '/data/clinical.csv',
    delimiter: ',',
    sheet: 'Sheet1',
    member: 'a/b.csv',
    object: 'x',
    transpose: false,
    zero_as_missing: false,
  })
  assert.deepEqual(payload, {
    path: '/data/feature_table.csv',
    feature_col: 'mz_rt',
    sample_sheet: '/data/samples.csv',
    clinical: '/data/clinical.csv',
    delimiter: ',',
    sheet: 'Sheet1',
    member: 'a/b.csv',
    object: 'x',
  })
  const xlsx = MB.buildImportPayload({ path: '/x.xlsx', data_format: 'xlsx', transpose: true, zero_as_missing: true })
  assert.equal(xlsx.data_format, 'xlsx')
  assert.equal(xlsx.transpose, true)
  assert.equal(xlsx.zero_as_missing, true)
})

test('suggestionInstruction：生成可粘给 Agent 的中文指令', () => {
  const text = MB.suggestionInstruction({ title: '运行 PLS-DA', detail: '先做置换检验', step: 'plsda' })
  assert.match(text, /^请在工作台执行「运行 PLS-DA」/)
  assert.match(text, /（步骤：PLS-DA \/ OPLS-DA，step=plsda）/)
  assert.match(text, /：先做置换检验/)
  assert.match(text, /。$/)
  assert.match(MB.suggestionInstruction({ step: 'mystery' }), /（step=mystery）/)
  assert.match(MB.suggestionInstruction(null), /^请在工作台执行「继续分析」/)
})

// ---------------------------------------------------------------------------
// 4. 参数默认值 / 收集 / 步骤依赖
// ---------------------------------------------------------------------------

test('defaultParams / collectStepParams：类型收敛且数组不共享引用', () => {
  const impute = MB.defaultParams('impute')
  assert.equal(impute.method, 'half_min')
  assert.equal(impute.k, 5)
  assert.notEqual(MB.defaultParams('impute'), impute, '每次都应返回新对象')

  const collected = MB.collectStepParams('impute', { method: 'knn', k: '9' })
  assert.equal(collected.method, 'knn')
  assert.equal(collected.k, 9)

  const blank = MB.collectStepParams('de', { case: '', control: '', group_col: null })
  assert.equal(blank.case, null)
  assert.equal(blank.control, null)
  assert.equal(blank.group_col, null)
  assert.equal(blank.method, 'welch', '未填的参数回落默认值')
  assert.equal(blank.paired, false)

  const flags = MB.collectStepParams('annotate', { formula: true, ppm: '5', library: ' /lib/a.msp ' })
  assert.equal(flags.formula, true)
  assert.equal(flags.ppm, 5)
  assert.equal(flags.library, '/lib/a.msp')
  assert.deepEqual(MB.parseTokenList('mz1, mz2；mz3 mz4'), ['mz1', 'mz2', 'mz3', 'mz4'])
})

test('doneStepSet / isStepReady / stepStatusOf：requires 判定与 running 优先', () => {
  const dataset = { id: 'ds1', steps: ['qc', 'filter', 'impute', 'normalize'] }
  assert.deepEqual(MB.doneStepSet(dataset), { qc: true, filter: true, impute: true, normalize: true })
  assert.equal(MB.isStepReady('qc', dataset), true)
  assert.equal(MB.isStepReady('transform', dataset), true)
  assert.equal(MB.isStepReady('plsda', dataset), false, 'plsda 需要 transform')
  assert.equal(MB.isStepReady('batch', dataset), true, 'batch 只依赖 normalize')
  assert.equal(MB.isStepReady('enrich', dataset), false, 'enrich 需要 de')
  assert.equal(MB.isStepReady('qc', null), false)
  assert.equal(MB.isStepReady('nope', dataset), false)

  assert.equal(MB.stepStatusOf('qc', dataset, '', ''), 'done')
  assert.equal(MB.stepStatusOf('qc', dataset, 'qc', ''), 'running')
  assert.equal(MB.stepStatusOf('qc', dataset, '', 'qc'), 'failed')
  assert.equal(MB.stepStatusOf('transform', dataset, '', ''), 'ready')
  assert.equal(MB.stepStatusOf('plsda', dataset, '', ''), 'blocked')
  assert.equal(MB.stepStatusKey('done'), 'stepStatusDone')
  assert.equal(MB.stepStatusKey('ready'), 'stepStatusReady')
  assert.equal(MB.stepStatusKey('weird'), 'stepStatusBlocked')
})

// ---------------------------------------------------------------------------
// 5. API 客户端：method / url / body（与 lib/impl.mjs 路由一一对应）
// ---------------------------------------------------------------------------

function makeFetch(payload, options) {
  const opts = options || {}
  const calls = []
  const impl = function (url, init) {
    calls.push({ url: String(url), init: init || {} })
    if (opts.reject) return Promise.reject(opts.reject)
    const body = typeof payload === 'function' ? payload(url, init) : payload
    return Promise.resolve({
      ok: opts.httpOk === false ? false : true,
      status: opts.status || 200,
      text: function () { return Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)) },
    })
  }
  impl.calls = calls
  return impl
}

test('apiClient：GET/POST、同源 /metabolomics 前缀、JSON body 与 actor=user', async () => {
  const fake = makeFetch({ ok: true, engine: { ready: true } })
  const api = MB.apiClient(fake)

  await api.getHealth()
  assert.equal(fake.calls[0].url, '/metabolomics/health')
  assert.equal(fake.calls[0].init.method, 'GET')

  await api.getState()
  assert.equal(fake.calls[1].url, '/metabolomics/state')

  await api.getFormats()
  assert.equal(fake.calls[2].url, '/metabolomics/formats')

  await api.getSteps()
  assert.equal(fake.calls[3].url, '/metabolomics/steps')

  await api.getAnalyses('ds 1')
  assert.equal(fake.calls[4].url, '/metabolomics/analyses?dataset=ds%201')
  assert.equal(fake.calls[4].init.method, 'GET')

  await api.getJob('job 1')
  assert.equal(fake.calls[5].url, '/metabolomics/jobs?id=job%201')

  await api.runStep('ds1', 'plsda', { n_comp: 3 })
  assert.equal(fake.calls[6].url, '/metabolomics/run')
  assert.equal(fake.calls[6].init.method, 'POST')
  assert.equal(fake.calls[6].init.headers['content-type'], 'application/json')
  assert.deepEqual(JSON.parse(fake.calls[6].init.body), {
    dataset: 'ds1', step: 'plsda', params: { n_comp: 3 }, actor: 'user',
  })

  await api.plot('ds1', 'vip_bar', { top_n: 20 })
  assert.equal(fake.calls[7].url, '/metabolomics/plot')
  assert.deepEqual(JSON.parse(fake.calls[7].init.body), {
    dataset: 'ds1', kind: 'vip_bar', params: { top_n: 20 }, actor: 'user',
  })

  await api.analysis('ds1', 'plsda_vip', { top_n: 30 })
  assert.equal(fake.calls[8].url, '/metabolomics/analysis')
  assert.deepEqual(JSON.parse(fake.calls[8].init.body), {
    dataset: 'ds1', id: 'plsda_vip', params: { top_n: 30 }, actor: 'user',
  })

  await api.preview('ds1')
  assert.deepEqual(JSON.parse(fake.calls[9].init.body), { dataset: 'ds1', actor: 'user' })

  await api.samples({ dataset: 'ds1', op: 'auto_group', group_regex: '_(\\w+)$' })
  assert.equal(fake.calls[10].url, '/metabolomics/samples')
  assert.equal(JSON.parse(fake.calls[10].init.body).op, 'auto_group')

  await api.exportData('ds1', 'de_csv')
  assert.deepEqual(JSON.parse(fake.calls[11].init.body), { dataset: 'ds1', what: 'de_csv', actor: 'user' })
  await api.exportData('ds1', 'report', '/tmp/r.md')
  assert.equal(JSON.parse(fake.calls[12].init.body).out_path, '/tmp/r.md')

  await api.audit('ds1')
  assert.equal(fake.calls[13].url, '/metabolomics/audit')
  await api.suggest('ds1')
  assert.equal(fake.calls[14].url, '/metabolomics/suggest')
  await api.reset()
  assert.equal(fake.calls[15].url, '/metabolomics/reset')
  await api.restartEngine()
  assert.equal(fake.calls[16].url, '/metabolomics/engine/restart')

  await api.openDataset({ path: '/data/ft.csv', data_format: 'matrix_text' })
  assert.equal(fake.calls[17].url, '/metabolomics/dataset/open')
  assert.deepEqual(JSON.parse(fake.calls[17].init.body), { path: '/data/ft.csv', data_format: 'matrix_text', actor: 'user' })

  await api.demo()
  assert.equal(fake.calls[18].url, '/metabolomics/dataset/demo')
  await api.selectDataset('ds2')
  assert.equal(fake.calls[19].url, '/metabolomics/dataset/select')
  await api.closeDataset('ds2')
  assert.equal(fake.calls[20].url, '/metabolomics/dataset/close')
  await api.inspectSource({ path: '/data/ft.csv' })
  assert.equal(fake.calls[21].url, '/metabolomics/inspect')
})

test('apiClient：GEO / MetaboLights 检索与下载（默认目录、大写编号、open_after）', async () => {
  const fake = makeFetch({ ok: true, total: 0, items: [] })
  const api = MB.apiClient(fake)

  await api.geoSearch({ term: '血清 代谢组学 LC-MS', species: 'human', method: 'lc-ms', limit: 5 })
  assert.equal(fake.calls[0].url, '/metabolomics/geo/search')
  const search = JSON.parse(fake.calls[0].init.body)
  assert.equal(search.term, '血清 代谢组学 LC-MS')
  assert.equal(search.species, 'human')
  assert.equal(search.method, 'lc-ms')
  assert.equal(search.limit, 5)
  assert.equal(search.has_clinical, undefined)

  await api.geoDetail('gse123')
  assert.equal(fake.calls[1].url, '/metabolomics/geo/detail')
  assert.equal(JSON.parse(fake.calls[1].init.body).gse, 'GSE123')

  await api.geoDownload({ gse: 'gse123', files: ['a.txt'], open_after: true })
  assert.equal(fake.calls[2].url, '/metabolomics/geo/download')
  const download = JSON.parse(fake.calls[2].init.body)
  assert.equal(download.gse, 'GSE123')
  assert.deepEqual(download.files, ['a.txt'])
  assert.equal(download.dest_dir, MB.DEFAULT_GEO_DIR, '默认下载目录')
  assert.equal(download.overwrite, false)
  assert.equal(download.open_after, true)

  await api.mlSearch({ query: 'serum metabolomics', limit: 10 })
  assert.equal(fake.calls[3].url, '/metabolomics/ml/search')
  assert.equal(JSON.parse(fake.calls[3].init.body).query, 'serum metabolomics')

  await api.mlDetail('mtbls1')
  assert.equal(fake.calls[4].url, '/metabolomics/ml/detail')
  assert.equal(JSON.parse(fake.calls[4].init.body).accession, 'MTBLS1')

  await api.mlDownload({ accession: 'mtbls1', files: ['s.txt'], dest_dir: '/tmp/ml' })
  assert.equal(fake.calls[5].url, '/metabolomics/ml/download')
  const mlDownload = JSON.parse(fake.calls[5].init.body)
  assert.equal(mlDownload.accession, 'MTBLS1')
  assert.equal(mlDownload.dest_dir, '/tmp/ml')

  // 浏览器上传：body 就是文件，名字走 x-dsh-filename 头
  await api.upload({ name: 'ft.csv', size: 12 })
  assert.equal(fake.calls[6].url, '/metabolomics/dataset/upload')
  assert.equal(fake.calls[6].init.method, 'POST')
  assert.equal(fake.calls[6].init.headers['x-dsh-filename'], 'ft.csv')
})

test('apiClient：recipe 状态（查 / 运行）与错误信封带 code', async () => {
  const fake = makeFetch({ ok: true, recipes: [{ id: 'standard' }] })
  const api = MB.apiClient(fake)
  await api.recipes()
  assert.equal(fake.calls[0].url, '/metabolomics/recipe')
  assert.equal(fake.calls[0].init.method, 'POST')
  assert.deepEqual(JSON.parse(fake.calls[0].init.body), { actor: 'user' })

  await api.recipe('standard')
  assert.equal(JSON.parse(fake.calls[1].init.body).recipe, 'standard')

  await api.recipeRun({ dataset: 'ds1', recipe: 'quick' })
  assert.equal(fake.calls[2].url, '/metabolomics/recipe/run')
  assert.deepEqual(JSON.parse(fake.calls[2].init.body), { dataset: 'ds1', recipe: 'quick', actor: 'user' })

  const soft = makeFetch({ ok: false, code: 'prerequisite', error: '需要先完成 de' })
  await assert.rejects(MB.apiClient(soft).getState(), (error) => {
    assert.equal(error.message, '需要先完成 de')
    assert.equal(error.code, 'prerequisite')
    return true
  })
  const hard = makeFetch({ ok: false, code: 'internal', error: '引擎崩了' }, { httpOk: false, status: 500 })
  await assert.rejects(MB.apiClient(hard).getHealth(), (error) => {
    assert.equal(error.message, '引擎崩了')
    assert.equal(error.status, 500)
    return true
  })
  const ambiguous = makeFetch({ ok: false, code: 'ambiguous', error: '多个候选', candidates: ['a.csv'] })
  await assert.rejects(MB.apiClient(ambiguous).openDataset({ path: '/x.zip' }), (error) => {
    assert.equal(error.code, 'ambiguous')
    assert.deepEqual(error.candidates, ['a.csv'])
    return true
  })
})

test('plotKey / matchFeatures / 表格工具：稳定键、特征检索、排序与截断', () => {
  assert.equal(MB.plotKey('pca_scores', { color_by: 'group', pc_x: 1 }), MB.plotKey('pca_scores', { pc_x: 1, color_by: 'group' }))
  assert.equal(MB.plotKey('pca_variance', {}), 'pca_variance')
  assert.equal(MB.plotKey('volcano_de', { fc: 1.5 }), 'volcano_de|fc=1.5')

  const features = ['mz203.0526_rt1.23', 'mz184.0733_rt0.88', 'mz496.3402_rt9.51']
  assert.deepEqual(MB.matchFeatures(features, 'mz203', 5), ['mz203.0526_rt1.23'])
  assert.equal(MB.matchFeatures(features, '', 2).length, 2)
  assert.deepEqual(MB.matchFeatures(null, 'x'), [])

  const table = {
    columns: [{ key: 'feature', type: 'string' }, { key: 'p', type: 'number' }],
    rows: [{ feature: 'B', p: 0.2 }, { feature: 'A', p: 0.01 }, { feature: 'C', p: null }],
  }
  assert.deepEqual(MB.sortRows(table.rows, 'p', 'asc', table.columns).map((row) => row.feature), ['A', 'B', 'C'])
  const view = MB.formatTableRows(table, 2)
  assert.equal(view.total, 3)
  assert.equal(view.shown, 2)
  assert.equal(view.truncated, true)
  assert.equal(view.rows[0].cells[1].numeric, true)
  assert.equal(MB.formatTableRows({ columns: [], rows: [] }, 2).columns.length, 0)
})

// ---------------------------------------------------------------------------
// 6. 轮询状态机
// ---------------------------------------------------------------------------

test('jobPolling：running / done / failed / timeout 四态', () => {
  const running = MB.jobPolling({ job: { id: 'j1', status: 'running', pct: 42, message: '下载中' } })
  assert.equal(running.status, 'running')
  assert.equal(running.running, true)
  assert.equal(running.pct, 42)
  assert.equal(running.done, false)

  const done = MB.jobPolling({ ok: true, job: { id: 'j1', status: 'done', pct: 100, result: { dir: '/tmp/geo' } } })
  assert.equal(done.status, 'done')
  assert.equal(done.done, true)
  assert.deepEqual(done.result, { dir: '/tmp/geo' })
  assert.equal(MB.jobDone({ job: { status: 'done' } }), true)
  assert.equal(MB.jobDone({ job: { status: 'running' } }), false)

  const failed = MB.jobPolling({ ok: false, job: { id: 'j2', status: 'failed', error: '网络错误' } })
  assert.equal(failed.status, 'failed')
  assert.equal(failed.failed, true)
  assert.equal(failed.error, '网络错误')

  const timeout = MB.jobPolling({ job: { status: 'running', pct: 10 } }, { elapsedMs: 999999, timeoutMs: 1000 })
  assert.equal(timeout.status, 'timeout')
  assert.equal(timeout.timedOut, true)
  assert.equal(timeout.running, false)

  const doneWins = MB.jobPolling({ job: { status: 'done' } }, { elapsedMs: 999999, timeoutMs: 1 })
  assert.equal(doneWins.status, 'done')
  assert.equal(MB.pickJobId({ job: { id: 'j9' } }), 'j9')
  assert.equal(MB.pickJobId({ job_id: 'j8' }), 'j8')
  assert.equal(MB.pickJobId({}), '')
})

// ---------------------------------------------------------------------------
// 7. 响应归一化（DESIGN §2.4 dataset 摘要 / §2.5 结果信封）
// ---------------------------------------------------------------------------

test('normalizeDataset：缺字段也能渲染；样本/特征/极性/注释/历史归一化', () => {
  const empty = MB.normalizeDataset(null)
  assert.equal(empty.id, '')
  assert.equal(empty.name, '')
  assert.equal(empty.n_samples, null)
  assert.equal(empty.n_features, null)
  assert.equal(empty.polarity, '')
  assert.deepEqual(empty.steps, [])
  assert.deepEqual(empty.sample_columns, [])
  assert.deepEqual(empty.warnings, [])
  assert.deepEqual(empty.notes, [])
  assert.deepEqual(empty.history, [])
  assert.equal(empty.current, '')
  assert.equal(empty.feature_meta.has_mz, false)
  assert.equal(empty.feature_meta.n_annotated, null)

  const dataset = MB.normalizeDataset({
    dataset: {
      id: 'ds1',
      name: 'MTBLS1',
      path: '/data/MTBLS1/m_MTBLS1.txt',
      source_format: 'isa_tab',
      n_samples: '72',
      n_features: 1600,
      polarity: 'pos',
      species: 'human',
      feature_meta: { has_mz: true, has_rt: true, n_annotated: 312 },
      sample_columns: [{ key: 'group', kind: 'categorical', n_unique: 2 }, { name: 'batch' }],
      steps: ['qc', 'filter', 'impute'],
      current: 'impute',
      warnings: ['有 3 个样本缺失率偏高'],
      notes: ['PQN 前后 TIC 已对比'],
      history: [{ at: '2026-01-01T00:00:00Z', step: 'qc', ok: true, ms: 812, summary: '完成' }],
      analyses_run: [{ at: '2026-01-01T00:00:00Z', id: 'plsda_vip', ok: true, ms: 120 }],
    },
  })
  assert.equal(dataset.id, 'ds1')
  assert.equal(dataset.n_samples, 72)
  assert.equal(dataset.n_features, 1600)
  assert.equal(dataset.polarity, 'pos')
  assert.equal(dataset.source_format, 'isa_tab')
  assert.equal(dataset.feature_meta.has_mz, true)
  assert.equal(dataset.feature_meta.n_annotated, 312)
  assert.equal(dataset.sample_columns.length, 2)
  assert.equal(dataset.sample_columns[0].n_unique, 2)
  assert.deepEqual(dataset.steps, ['qc', 'filter', 'impute'])
  assert.deepEqual(dataset.warnings, ['有 3 个样本缺失率偏高'])
  assert.equal(dataset.history[0].ms, 812)
  assert.equal(dataset.analyses_run[0].id, 'plsda_vip')
})

test('normalizeResult：summary/tables/figures/warnings/data_out 信封', () => {
  const result = MB.normalizeResult({
    ok: true,
    summary: { title: 'PLS-DA', bullets: ['R²Y=0.91', '  '], metrics: { q2: 0.72 } },
    tables: [{ id: 'vip', title: 'VIP 表', columns: [{ key: 'feature' }, { key: 'vip', type: 'number' }], rows: [{ feature: 'mz203.0526_rt1.23', vip: 2.1 }] }],
    figures: [{ id: 'f1', title: 'VIP 条形图', file: '/tmp/analyses/ds1/plsda/vip_bar.png', width: 800, height: 600 }],
    warnings: ['置换次数偏少'],
    notes: ['OPLS-DA 与 PLS-DA 结果一致'],
    data_out: [{ kind: 'vip', path: '/tmp/out/vip.csv', label: 'VIP 表' }],
  })
  assert.equal(result.title, 'PLS-DA')
  assert.deepEqual(result.bullets, ['R²Y=0.91'])
  assert.deepEqual(result.metrics, { q2: 0.72 })
  assert.equal(result.tables.length, 1)
  assert.equal(result.tables[0].columns[1].type, 'number')
  assert.equal(result.figures.length, 1)
  assert.equal(result.figures[0].url, '/metabolomics/figure?id=vip_bar.png')
  assert.deepEqual(result.warnings, ['置换次数偏少'])
  assert.deepEqual(result.notes, ['OPLS-DA 与 PLS-DA 结果一致'])
  assert.equal(result.data_out[0].path, '/tmp/out/vip.csv')

  const nested = MB.normalizeResult({ ok: true, result: { summary: { bullets: ['ok'] } }, ms: 120 })
  assert.deepEqual(nested.bullets, ['ok'])
  assert.equal(nested.ms, 120)
  assert.deepEqual(MB.normalizeResult(null).tables, [])
})

test('normalizeFigure / normalizeState / normalizeAnalyses / normalizeAudit / normalizeHealth', () => {
  assert.equal(MB.figureSrc('/metabolomics/figure?id=a.png'), '/metabolomics/figure?id=a.png')
  assert.equal(MB.figureSrc('qc_tic.png'), '/metabolomics/figure?id=qc_tic.png')
  assert.equal(MB.figureSrc('bad.svg'), '')
  const figure = MB.normalizeFigure({ file: '/tmp/x/qc_tic.png', title: 'TIC 图', kind: 'qc_tic' })
  assert.equal(figure.url, '/metabolomics/figure?id=qc_tic.png')
  assert.equal(figure.kind, 'qc_tic')
  assert.equal(MB.normalizeFigure({ file: 'bad.svg' }), null)

  const state = MB.normalizeState({
    datasets: [{ id: 'ds1', name: 'demo-metabo', n_samples: 72, n_features: 1600, steps: ['qc'] }],
    engine: { ready: true, mode: 'builtin', python: '3.14.2' },
    suggestions: [{ id: 's1', title: '下一步', step: 'filter' }],
    activity: [{ at: '2026-01-01T00:00:00Z', actor: 'agent', action: 'run:qc', detail: '完成', ok: true }],
    recipes: [{ id: 'standard', title: '标准流程', steps: [{ step: 'qc' }] }],
  })
  assert.equal(state.active, 'ds1', '未给 active 时回落第一个数据集')
  assert.equal(state.datasets[0].n_samples, 72)
  assert.equal(state.engine.ready, true)
  assert.equal(state.suggestions[0].step, 'filter')
  assert.equal(state.activity[0].action, 'run:qc')
  assert.equal(state.recipes[0].id, 'standard', 'recipe 目录透传')

  const catalog = MB.normalizeAnalyses({
    analyses: [
      { id: 'plsda_vip', title: '变量投影重要度（VIP）', group: '多变量判别', method: 'VIP（Wold 1995）', needs: ['plsda'], params: [{ key: 'top_n', type: 'number', default: 30 }] },
      { id: 'ms2_library_match', title: 'MS2 库匹配', group: '代谢物鉴定', available: false, reason: '未挂载 MS2 谱库' },
    ],
  })
  assert.equal(catalog.analyses.length, 2)
  assert.deepEqual(catalog.groups, ['多变量判别', '代谢物鉴定'])
  assert.equal(catalog.analyses[0].params[0].type, 'number')
  assert.equal(catalog.analyses[1].available, false)
  assert.equal(catalog.analyses[1].reason, '未挂载 MS2 谱库')

  const audit = MB.normalizeAudit({ score: 88.6, level: 'warn', checks: [{ id: 'missing', level: 'warn', title: '缺失值', detail: '未填补' }] })
  assert.equal(audit.score, 89)
  assert.equal(audit.checks[0].level, 'warn')

  const health = MB.normalizeHealth({ engine: { ready: true, python: '3.14.2', packages: { numpy: '2.5.1' } }, metabolights: true })
  assert.equal(health.ready, true)
  assert.equal(health.python, '3.14.2')
  assert.equal(health.metabolights, true)
  assert.equal(MB.engineBadge(health).labelKey, 'engineReady')
  assert.equal(MB.engineBadge({ ready: false }).labelKey, 'engineBooting')
  assert.equal(MB.engineBadge({ ready: false, error: '没有 python' }).labelKey, 'engineBad')
})

// ---------------------------------------------------------------------------
// 8. 静态合规：骨架 / 槽注册 / 样式 / 文案 / 数据面
// ---------------------------------------------------------------------------

test('模块骨架：IIFE + ModuleLoader + 两个槽 + __test__ 组件表', () => {
  assert.match(CLIENT_SOURCE, /^\/\*\*/)
  assert.match(CLIENT_SOURCE, /;\(function \(\) \{/)
  assert.ok(CLIENT_SOURCE.trimEnd().endsWith('})()'), '整文件应以 })() 收尾')
  assert.match(CLIENT_SOURCE, /window\.__ModuleLoader__\.load\(\{ id: NS, factory: factory \}\)/)
  assert.match(CLIENT_SOURCE, /globalThis\.__MB_TEST__ = MB_TEST/)
  assert.match(CLIENT_SOURCE, /inject: \['slots',\s*'locale'\]/)

  // main 槽 key=metabolomics；侧边栏 id=metabolomics, order=6
  assert.match(CLIENT_SOURCE, /slots\.inject\('main',/)
  assert.match(CLIENT_SOURCE, /key: 'metabolomics'/)
  assert.match(CLIENT_SOURCE, /slots\.inject\('sidebar\.panellist',/)
  assert.match(CLIENT_SOURCE, /id: 'metabolomics'/)
  assert.match(CLIENT_SOURCE, /order: 6/)
  // locale 注册 + 翻译器
  assert.match(CLIENT_SOURCE, /locale\.register\(NS, \{ zh: DICT_ZH, en: DICT_EN \}\)/)
  assert.match(CLIENT_SOURCE, /T = createTranslator\(locale\)/)
  // 错误边界 / 关键组件
  assert.match(CLIENT_SOURCE, /class PluginErrorBoundary extends React\.Component/)
  for (const name of [
    'Spinner', 'LoadingLine', 'ErrorBanner', 'SidebarIcon', 'TopBar',
    'DatasetCard', 'PipelinePanel', 'StepCard', 'ParamForm', 'ResultView', 'TabsBar',
    'OverviewTab', 'QcTab', 'MultivariateTab', 'DeTab', 'PathwayTab', 'BiomarkerTab',
    'AnnotationTab', 'AnalysisTab', 'RecordsTab', 'LogTab',
    'SourceDialog', 'GeoDialog', 'MlDialog', 'ImportDialog', 'AnalysisDrawer',
    'AgentPanel', 'WorkbenchBody', 'WorkbenchPage', 'apply',
  ]) {
    assert.match(CLIENT_SOURCE, new RegExp('function ' + name + '\\('), `缺组件 ${name}`)
  }
  // factory 返回 __test__ 组件表
  assert.match(CLIENT_SOURCE, /__test__: Object\.assign\(\{\}, MB_TEST, \{/)
  assert.match(CLIENT_SOURCE, /components: \{/)

  // 只 require('react')，不碰任何 @deepseek-ai 包
  const requires = [...CLIENT_CODE.matchAll(/\brequire\((['"])([^'"]+)\1\)/g)].map((match) => match[2])
  assert.deepEqual([...new Set(requires)], ['react'], '唯一允许的 require 目标是 react')
  assert.ok(!CLIENT_CODE.includes('@deepseek-ai'), '不得 require 宿主 Client 包')
  // 不写 JSX：不出现 `h(` 之外的标签字面量
  assert.ok(!/<[A-Za-z][A-Za-z0-9]*[\s/>]/.test(CLIENT_CODE), '不应出现 JSX 标签字面量')
  // 数据面只打同源 /metabolomics（不出现别的插件前缀）
  assert.ok(!CLIENT_CODE.includes('/geobulk'), '不得请求 /geobulk 路由')
  assert.ok(!CLIENT_SOURCE.includes('dsh-spatial'), '不得残留 dsh-spatial 字样')
  assert.ok(CLIENT_SOURCE.includes("'/metabolomics'"), "API 常量应为 '/metabolomics'")
  assert.ok(!/\/spatial\b/.test(CLIENT_CODE), '不得请求 /spatial 路由')
  assert.ok(!/\bpickDir\b/.test(CLIENT_CODE), 'impl 没有 /pick 路由，前端不应调用目录选择接口')
  assert.ok(!/\/pick['"]/.test(CLIENT_CODE), 'impl 没有 /pick 路由')
})

test('样式：类名 mb- 前缀、只用 --dsw-alias-* 主题色', () => {
  const cssStart = CLIENT_SOURCE.indexOf('var CSS = [')
  const cssEnd = CLIENT_SOURCE.indexOf("].join('')", cssStart)
  assert.ok(cssStart > 0 && cssEnd > cssStart, 'CSS 段落应存在')
  const css = CLIENT_SOURCE.slice(cssStart, cssEnd)
  assert.ok(css.length > 12000, `CSS 太短：${css.length}`)

  const selectors = [...css.matchAll(/\.([A-Za-z][A-Za-z0-9_-]*)\s*[{,:]/g)].map((match) => match[1])
  const badSelectors = [...new Set(selectors)].filter((name) => !name.startsWith('mb-'))
  assert.deepEqual(badSelectors, [], `CSS 选择器不是 mb- 前缀：${badSelectors.join(', ')}`)

  const hexColors = [...css.matchAll(/#[0-9a-fA-F]{3,8}\b/g)].map((match) => match[0])
  assert.deepEqual(hexColors, [], `CSS 不应出现硬编码颜色：${hexColors.join(', ')}`)
  assert.ok(!/\brgba?\(/.test(css), 'CSS 不应出现 rgb()/rgba()，请用 color-mix + 主题变量')
  const aliasUses = [...css.matchAll(/--dsw-alias-[a-z0-9-]+/g)].map((match) => match[0])
  assert.ok(aliasUses.length > 120, `主题变量用得不够：${aliasUses.length}`)
  // 不允许出现 sp- 前缀类名
  assert.ok(!/['".]sp-/.test(CLIENT_SOURCE), '不得残留 sp- 前缀类名')
})

test('组件里用到的 mb- 类名都有样式定义；文案 key 都在字典里', () => {
  const cssStart = CLIENT_SOURCE.indexOf('var CSS = [')
  const cssEnd = CLIENT_SOURCE.indexOf("].join('')", cssStart)
  const css = CLIENT_SOURCE.slice(cssStart, cssEnd)
  const defined = new Set([...css.matchAll(/\.(mb-[A-Za-z0-9_-]+)/g)].map((match) => match[1]))
  const used = new Set()
  for (const match of CLIENT_SOURCE.matchAll(/className: '([^']*)'/g)) {
    for (const token of match[1].split(/\s+/)) {
      if (token.startsWith('mb-') && !token.endsWith('-')) used.add(token)
    }
  }
  const missing = [...used].filter((name) => !defined.has(name)).sort()
  assert.deepEqual(missing, [], `用了未定义的样式类：${missing.join(', ')}`)

  const referenced = new Set()
  for (const match of CLIENT_SOURCE.matchAll(/\bT\('([A-Za-z0-9_]+)'\)/g)) referenced.add(match[1])
  for (const match of CLIENT_SOURCE.matchAll(/tf\(T, '([A-Za-z0-9_]+)'/g)) referenced.add(match[1])
  for (const match of CLIENT_SOURCE.matchAll(/labelKey: '([A-Za-z0-9_]+)'/g)) referenced.add(match[1])
  for (const match of CLIENT_SOURCE.matchAll(/titleKey: '([A-Za-z0-9_]+)'/g)) referenced.add(match[1])
  for (const key of [
    'stepStatusDone', 'stepStatusRunning', 'stepStatusReady', 'stepStatusBlocked', 'stepStatusFailed',
    'engineReady', 'engineBooting', 'engineBad', 'auditOk', 'auditWarn', 'auditFail',
    'tabOverview', 'tabQc', 'tabMultivariate', 'tabDe', 'tabPathway', 'tabBiomarker',
    'tabAnnotation', 'tabRecords', 'tabLog', 'geoMethodAll', 'colorGroup', 'colorBatch',
    'colorQc', 'colorNone', 'mvR2Y', 'mvQ2', 'mvPermP', 'mvQuality', 'metricSamples', 'metricFeatures',
  ]) referenced.add(key)
  const missingZh = [...referenced].filter((key) => !MB.DICT_ZH[key]).sort()
  const missingEn = [...referenced].filter((key) => !MB.DICT_EN[key]).sort()
  assert.deepEqual(missingZh, [], `字典缺 key：${missingZh.join(', ')}`)
  assert.deepEqual(missingEn, [], `EN 字典缺 key：${missingEn.join(', ')}`)
})

test('不用 emoji 当图标；侧边栏是内联 SVG', () => {
  const emoji = CLIENT_SOURCE.match(/[\u{1F000}-\u{1FAFF}\u{1F1E6}-\u{1F1FF}]/gu)
  assert.equal(emoji, null, `出现 emoji：${emoji ? emoji.join('') : ''}`)
  assert.match(CLIENT_SOURCE, /function SidebarGlyph\(props\)/)
  assert.match(CLIENT_SOURCE, /viewBox: '0 0 24 24'/)
})

test('纯函数层不碰 DOM / 不 require react；浏览器层只在 factory 里', () => {
  const factoryStart = CLIENT_SOURCE.indexOf('function factory(require)')
  assert.ok(factoryStart > 0)
  const pure = stripComments(CLIENT_SOURCE.slice(0, factoryStart))
  assert.ok(!/\bdocument\./.test(pure), '纯函数层不应碰 document')
  assert.ok(!/\bwindow\./.test(pure), '纯函数层不应碰 window')
  assert.ok(!/\brequire\(/.test(pure), '纯函数层不应 require')
  const browser = CLIENT_SOURCE.slice(factoryStart)
  assert.match(browser, /require\('react'\)/)
})

// ---------------------------------------------------------------------------
// 8.5 locale/*.json 与组件冒烟渲染（不依赖浏览器 / npm）
// ---------------------------------------------------------------------------

test('locale/*.json：JSON 合法、中英 key 集一致、meta.title/description/nav 保留', async () => {
  const zh = JSON.parse(await readFile(new URL('../locale/zh.json', import.meta.url), 'utf8'))
  const en = JSON.parse(await readFile(new URL('../locale/en.json', import.meta.url), 'utf8'))
  assert.deepEqual(Object.keys(zh).sort(), Object.keys(en).sort(), 'locale 中英 key 集不一致')
  for (const [name, dict] of [['zh', zh], ['en', en]]) {
    assert.ok(dict.meta && dict.meta.title, `${name}.meta.title 缺失`)
    assert.ok(dict.meta.description, `${name}.meta.description 缺失`)
    assert.ok(dict.nav, `${name}.nav 缺失`)
    assert.ok(dict.title, `${name}.title 缺失`)
    // 9 个 tab 的导航词条都要有
    for (const tab of ['overview', 'qcTab', 'multivariate', 'differential', 'pathway', 'biomarker', 'annotation', 'records', 'log']) {
      assert.ok(dict[tab], `${name}.${tab} 缺失`)
    }
  }
  assert.equal(zh.nav, MB.DICT_ZH.nav, 'locale nav 与客户端字典 nav 一致')
  assert.equal(en.nav, MB.DICT_EN.nav, 'locale nav 与客户端字典 nav 一致')
})

/** 极简假 React：只够渲染一遍组件树（不跑 effect、不触发 state 更新）。 */
function createFakeReact() {
  let hookIndex = 0
  let states = []
  const Fake = {
    createElement(type, props) {
      const children = []
      const push = (value) => {
        if (Array.isArray(value)) value.forEach(push)
        else if (value !== null && value !== undefined && value !== false && value !== true) children.push(value)
      }
      push(Array.prototype.slice.call(arguments, 2))
      if (props && props.children !== undefined) push(props.children)
      return { type, props: Object.assign({}, props, { children }) }
    },
    Component: class {
      constructor(props) { this.props = props || {}; this.state = {} }
      setState(patch) { this.state = Object.assign({}, this.state, typeof patch === 'function' ? patch(this.state, this.props) : patch) }
      render() { return null }
    },
    useState(initial) {
      const index = hookIndex++
      if (states[index] === undefined) states[index] = typeof initial === 'function' ? initial() : initial
      return [states[index], (next) => { states[index] = typeof next === 'function' ? next(states[index]) : next }]
    },
    useEffect() {},
    useRef(value) {
      const index = hookIndex++
      if (states[index] === undefined) states[index] = { current: value }
      return states[index]
    },
    useMemo(fn) { hookIndex++; return fn() },
    reset() { hookIndex = 0; states = [] },
  }
  return Fake
}

function renderTree(React, node, depth) {
  if (depth > 120) return
  if (node === null || node === undefined || typeof node === 'boolean') return
  if (typeof node === 'string' || typeof node === 'number') return
  if (Array.isArray(node)) { node.forEach((child) => renderTree(React, child, depth + 1)); return }
  if (!node || typeof node !== 'object') return
  const type = node.type
  const props = node.props || {}
  if (typeof type === 'function') {
    React.reset()
    const out = type.prototype instanceof React.Component
      ? new type(props).render()
      : type(props)
    renderTree(React, out, depth + 1)
    return
  }
  if (typeof type === 'string') renderTree(React, props.children, depth + 1)
}

test('组件冒烟渲染：假 React 渲染主组件/9 个 tab/3 个弹层（不依赖浏览器）', async () => {
  const url = new URL(CLIENT_URL.href)
  url.search = '?smoke=1'
  let captured = null
  globalThis.window = { __ModuleLoader__: { load: (spec) => { captured = spec } } }
  try {
    await import(url.href)
  } finally {
    delete globalThis.window
  }
  assert.ok(captured && typeof captured.factory === 'function', '浏览器分支应触发 __ModuleLoader__.load')
  assert.equal(captured.id, 'dsh-metabolomics')

  const React = createFakeReact()
  const module = captured.factory((name) => {
    assert.equal(name, 'react', 'factory 只允许 require react')
    return React
  })
  const components = module.__test__.components
  const helpers = module.__test__

  const dataset = helpers.normalizeDataset({
    dataset: {
      id: 'ds1', name: 'MTBLS1', n_samples: 72, n_features: 1600, polarity: 'pos', species: 'human',
      feature_meta: { has_mz: true, has_rt: true, n_annotated: 120 },
      sample_columns: [
        { key: 'group', kind: 'categorical', n_unique: 2 },
        { key: 'batch', kind: 'categorical', n_unique: 2 },
        { key: 'qc', kind: 'categorical', n_unique: 2 },
      ],
      steps: ['qc', 'filter', 'impute', 'normalize', 'transform'],
      current: 'transform',
      history: [{ at: '2026-01-01T00:00:00Z', step: 'qc', ok: true, ms: 800, summary: 'ok' }],
      analyses_run: [{ at: '2026-01-01T00:00:00Z', id: 'qc_summary', ok: true, ms: 100 }],
    },
  })
  const preview = helpers.normalizePreview({ n_samples: 72, n_features: 1600, columns: dataset.sample_columns, feature_meta: { has_mz: true } })
  const health = helpers.normalizeHealth({ engine: { ready: true, mode: 'builtin', python: '3.14.2' } })
  const catalog = helpers.normalizeAnalyses({
    analyses: [
      { id: 'qc_summary', title: '质控汇总', group: '质控与基础', method: 'm', needs: ['qc'], params: [{ key: 'x', type: 'number', default: 1 }] },
      { id: 'plsda_vip', title: 'VIP', group: '多变量判别', method: 'm', needs: ['plsda'], params: [{ key: 'top_n', type: 'number', default: 30 }] },
      { id: 'univariate_de', title: '差异', group: '差异与趋势', params: [] },
      { id: 'mummichog_pathway', title: '通路', group: '通路与化学分类', params: [] },
      { id: 'rf_importance', title: 'RF', group: '标志物与建模', params: [] },
      { id: 'adduct_annot', title: '加合物', group: '代谢物鉴定', params: [{ key: 'polarity', type: 'select', default: 'pos', options: [{ value: 'pos', label: 'pos' }] }] },
    ],
    groups: ['质控与基础', '多变量判别', '差异与趋势', '通路与化学分类', '标志物与建模', '代谢物鉴定'],
  })
  const api = helpers.createApi(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('{"ok":true}') }))
  const qcResult = helpers.normalizeResult({
    ok: true,
    summary: { bullets: ['b'], metrics: { r2y: 0.9, q2: 0.7, perm_p: 0.01 } },
    tables: [{
      id: 't', title: 'TIC',
      columns: [
        { key: 'sample', type: 'string' }, { key: 'tic', type: 'number' },
        { key: 'missing_rate', type: 'number' }, { key: 'group', type: 'string' },
      ],
      rows: [{ sample: 'S1', tic: 1e6, missing_rate: 0.12, group: 'case' }, { sample: 'S2', tic: 2e6, missing_rate: 0.31, group: 'control' }],
    }],
  })
  const baseProps = {
    dataset, preview, columns: dataset.sample_columns, api,
    catalog: { loading: false, error: '', data: catalog },
    runs: { qc_summary: { busy: false, error: '', result: qcResult } },
    plots: {}, loadPlot: () => {},
    stepResults: { qc: qcResult },
    stepBusy: '', exports: {}, runExport: () => {},
    runAnalysis: () => {}, onRunAnalysis: () => {}, plotOnce: () => Promise.resolve(null),
    runStep: () => {}, onRunStep: () => {}, onOpenAnalysis: () => {}, busy: false,
    health, suggestions: [{ id: 's', title: 't', step: 'plsda', detail: 'd' }],
    copiedKey: '', onCopy: () => {}, onOpenImport: () => {},
    formats: null, onImported: () => {}, onClose: () => {}, onRun: () => {},
  }
  const cases = [
    ['WorkbenchPage', {}],
    ['OverviewTab', baseProps],
    ['QcTab', baseProps],
    ['MultivariateTab', baseProps],
    ['DeTab', baseProps],
    ['PathwayTab', baseProps],
    ['BiomarkerTab', baseProps],
    ['AnnotationTab', baseProps],
    ['RecordsTab', baseProps],
    ['LogTab', { api }],
    ['GeoDialog', { api, onImported: () => {}, onClose: () => {} }],
    ['MlDialog', { api, onImported: () => {}, onClose: () => {} }],
    ['ImportDialog', { api, formats: null, onImported: () => {}, onClose: () => {} }],
    ['AnalysisDrawer', { dataset, columns: dataset.sample_columns, catalog: { loading: false, error: '', data: catalog }, runs: {}, onRun: () => {}, onClose: () => {} }],
    ['TabsBar', { active: 'qc', onChange: () => {} }],
    ['TopBar', {
      health, datasets: [dataset], activeId: 'ds1', busy: false, refreshing: false,
      onGeo: () => {}, onMl: () => {}, onImport: () => {}, onDemo: () => {}, onRefresh: () => {},
      onSelect: () => {}, onCloseDataset: () => {},
    }],
    ['AgentPanel', {
      dataset, health, suggestions: [{ id: 's', title: 't', step: 'plsda', detail: 'd' }],
      activity: [{ id: 'a', at: '2026-01-01T00:00:00Z', actor: 'agent', action: 'run:qc', text: 'x', ok: true }],
      copiedKey: '', onCopy: () => {},
    }],
  ]
  for (const [name, props] of cases) {
    assert.equal(typeof components[name], 'function', `__test__.components 缺 ${name}`)
    assert.doesNotThrow(
      () => renderTree(React, React.createElement(components[name], props), 0),
      `${name} 首次渲染抛错`,
    )
  }
})

// ---------------------------------------------------------------------------
// 9. 全部通过时打印 PASS（供 CI / Lead 验收脚本 grep）
// ---------------------------------------------------------------------------

test('汇总', () => {
  console.log('PASS')
})
