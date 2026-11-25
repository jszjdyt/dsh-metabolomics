/**
 * dsh-metabolomics — Host 业务实现（唯一操作实现）
 *
 * 一条操作只写一次：浏览器 UI 走 HTTP 路由，Agent 走 metabolomics 工具，
 * 两边入口都调本文件里的 op* 函数（见 cordis-plugin-dev 的「一个操作两个调用方」）。
 *
 * 契约来源：DESIGN.md（§2.2 命令表 / §2.6 Host 层 / §3 步骤 / §5 图 / §6 格式 / §7 数据源 / §8 工具）。
 * Python 侧以 lib/mb_engine.py 实际分发的 cmd 与 args 为准。
 *
 * 本文件按需用带 rev 的 URL 动态 import（见 index.js 的 loadImpl），
 * 因此不要在 lib 模块之间写静态 import；需要 engine.mjs 时用 loadEngineMod()。
 */

import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { pipeline } from 'node:stream/promises'

const require_path_join = (...parts) => path.join(...parts)
const require_path_basename = (value) => path.basename(value)

const PREFIX = '/metabolomics'
const MAX_JSON_BODY = 4 * 1024 * 1024
const MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
const ACTIVITY_LIMIT = 200
const STEP_TIMEOUT = 15 * 60 * 1000
const ANALYSIS_TIMEOUT = 20 * 60 * 1000
const DOWNLOAD_TIMEOUT = 45 * 60 * 1000
const IMPORT_TIMEOUT = 20 * 60 * 1000
const PING_TTL = 30_000

/** 12 个流程步骤（DESIGN §3：id/title/requires/params 默认值逐字一致，顺序即 UI 左栏卡片顺序）。 */
export const STEPS = [
  {
    id: 'qc', title: '样本质控', requires: [],
    params: { group_col: null, qc_label: '', min_features: 0 },
  },
  {
    id: 'filter', title: '特征过滤', requires: ['qc'],
    params: { min_sample_frac: 0.5, min_intensity: 0, rsd_max: null },
  },
  {
    id: 'impute', title: '缺失值填补', requires: ['filter'],
    params: { method: 'half_min', k: 5 },
  },
  {
    id: 'normalize', title: '归一化', requires: ['impute'],
    params: { method: 'pqn' },
  },
  {
    id: 'transform', title: '变换/标度', requires: ['normalize'],
    params: { transform: 'log2', scale: 'auto' },
  },
  {
    id: 'pca', title: '主成分分析', requires: ['transform'],
    params: { n_comp: 3, color_by: 'group' },
  },
  {
    id: 'plsda', title: 'PLS-DA / OPLS-DA', requires: ['transform'],
    params: { method: 'plsda', n_comp: 3, cv_folds: 7, permutations: 200 },
  },
  {
    id: 'de', title: '单变量差异', requires: ['transform'],
    params: { group_col: null, case: null, control: null, method: 'welch', paired: false, fdr: 0.05, fc: 1.5 },
  },
  {
    id: 'batch', title: '批次校正', requires: ['normalize'],
    params: { batch_key: null, method: 'rlsc', qc_label: '' },
  },
  {
    id: 'enrich', title: '通路富集', requires: ['de'],
    params: { method: 'mummichog', db: 'kegg', p_cut: 0.05, n_perm: 1000 },
  },
  {
    id: 'biomarker', title: '标志物筛选', requires: ['de'],
    params: { model: 'rf', n_features: 10, cv_folds: 5, time_col: null, event_col: null },
  },
  {
    id: 'annotate', title: '代谢物鉴定', requires: ['transform'],
    params: { adduct: '[M+H]+', ppm: 10, formula: true, library: '' },
  },
]
export const STEP_IDS = STEPS.map((s) => s.id)

/** 图类型白名单（DESIGN §5，与 mb_registry.PLOT_KINDS 一致）。 */
export const PLOT_KINDS = [
  'qc_tic', 'qc_rsd', 'qc_missing', 'sample_corr', 'sample_dendro', 'pca_scores', 'pca_variance',
  'hotelling_t2', 'plsda_scores', 'oplsda_splot', 'vip_bar', 'permutation_test', 'volcano_de',
  'fc_bar', 'heatmap_top', 'enrich_bar', 'enrich_dot', 'roc_curve', 'rf_importance', 'lasso_path',
  'trend_plot', 'dose_response', 'mass_accuracy', 'mass_network', 'batch_pca',
  'de_heatmap_top', 'batch_effect', 'batch_corrected_tic', 'corr_heatmap', 'corr_network',
  'km_metabolite', 'calibration_curve',
]

/** 导出类型（与 mb_registry.EXPORT_KINDS 逐字一致）。 */
export const EXPORT_KINDS = [
  'expr_csv', 'de_csv', 'vip_csv', 'enrich_csv', 'biomarker_csv', 'sample_info_csv',
  'summary_json', 'protocol', 'methods', 'report', 'script',
]

/** 数据格式（DESIGN §6；mb_io 就绪前 formats 命令也给这组 id 兜底）。 */
export const DATA_FORMATS = [
  'auto', 'matrix_text', 'xlsx', 'geo_series', 'archive', 'mztab', 'isa_tab',
  'msms_library', 'mzml_tic', 'rds', 'rdata',
]

/** 样本表子操作（DESIGN §2.2 samples）。 */
export const SAMPLE_OPS = ['list', 'update', 'auto_group', 'import_map', 'join_obs', 'derive', 'set_group', 'remove']

// ---------------------------------------------------------------------------
// HTTP 小工具
// ---------------------------------------------------------------------------

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]', '::1', '0.0.0.0'])

export function hostIsLoopback(req) {
  const raw = String((req && req.headers && req.headers.host) || '')
  if (!raw) return true
  const name = raw.startsWith('[') ? raw.slice(0, raw.indexOf(']') + 1) : raw.split(':')[0]
  return LOOPBACK_HOSTS.has(name.toLowerCase())
}

function sendJson(res, status, value) {
  const body = Buffer.from(JSON.stringify(value), 'utf8')
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': String(body.length),
  })
  res.end(body)
}

function opError(code, message, status = 400, details = null) {
  const body = { ok: false, code, error: message }
  if (details && typeof details === 'object') Object.assign(body, details)
  return { status, body }
}

function toFailure(error) {
  const code = (error && error.code) || 'internal'
  const message = String((error && error.message) || error)
  const details = error && error.details && typeof error.details === 'object' ? error.details : null
  if (code === 'prerequisite' || code === 'bad_request' || code === 'not_found' || code === 'unsupported' || code === 'ambiguous') {
    return opError(code, message, 400, details)
  }
  if (code === 'timeout') return opError(code, message, 504, details)
  if (code === 'no_python' || code === 'no_r' || code === 'engine_exited' || code === 'internal' || code === 'disposed') {
    return opError(code, message, 500, details)
  }
  if (code === 'busy') return opError(code, message, 429, details)
  return opError(code, message, 500, details)
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > limit) {
        reject(Object.assign(new Error(`请求体超过上限（${Math.round(limit / 1024)} KB）`), { code: 'bad_request' }))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', (error) => reject(Object.assign(error, { code: 'bad_request' })))
  })
}

async function readJson(req) {
  const buffer = await readBody(req, MAX_JSON_BODY)
  if (!buffer.length) return {}
  try {
    const value = JSON.parse(buffer.toString('utf8'))
    return value && typeof value === 'object' ? value : {}
  } catch (error) {
    throw Object.assign(new Error(`请求 JSON 解析失败：${error.message}`), { code: 'bad_request' })
  }
}

function sanitizeName(name) {
  const base = String(name || 'upload')
  return path.basename(base).replace(/[^\w.\-\u4e00-\u9fa5]+/g, '_').slice(0, 160) || 'upload'
}

function nowIso() {
  return new Date().toISOString()
}

function errorText(error) {
  return String(error && error.message ? error.message : error)
}

/** 把「Python 模块还没落盘」的 internal 错误翻译成可读的 unsupported（DESIGN 硬性要求）。 */
function readableError(error, label) {
  const message = errorText(error)
  if ((error && error.code === 'internal') && /No module named ['"]?(mb_|sp_)/.test(message)) {
    const missing = message.replace(/^.*No module named ['"]?/, '').replace(/['"].*$/, '')
    return Object.assign(
      new Error(`${label}尚未就绪：Python 模块 ${missing}.py 未安装。其它功能不受影响。`),
      { code: 'unsupported', details: { missing_module: missing } },
    )
  }
  return error
}

async function loadEngineMod() {
  try {
    return await import(new URL(`./engine.mjs?rev=${Date.now()}-${Math.random().toString(36).slice(2)}`, import.meta.url).href)
  } catch (error) {
    // 某些宿主加载器给出的 import.meta.url 不能作为 new URL 的 base（会抛 Invalid URL）：
    // 退回静态相对 import，仍能拿到同一份 engine.mjs。
    try { return await import('./engine.mjs') } catch { throw error }
  }
}

// ---------------------------------------------------------------------------
// 引擎
// ---------------------------------------------------------------------------

async function ensureEngine(runtime, { allowStart = true } = {}) {
  if (runtime.engine) {
    if (runtime.engine.running) return runtime.engine
    if (!allowStart) return runtime.engine
  } else {
    const mod = await loadEngineMod()
    const config = runtime.config && typeof runtime.config === 'object' ? runtime.config : {}
    runtime.engine = mod.createEngine({
      python: typeof config.python === 'string' && config.python.trim() ? config.python.trim() : undefined,
      workdir: typeof config.workdir === 'string' && config.workdir.trim() ? config.workdir.trim() : undefined,
      timeout: Number(config.timeoutMs) > 0 ? Number(config.timeoutMs) : undefined,
      config,
    })
    if (typeof runtime.ctx?.effect === 'function') {
      const engine = runtime.engine
      try {
        runtime.ctx.effect(() => () => { engine.dispose().catch(() => {}) }, 'dsh-metabolomics: engine')
      } catch (error) {
        // 插件重装/热重载后，工具闭包里捕获的 ctx 可能已经被销毁；这里失败不能拖垮引擎启动。
        try { console.warn('[dsh-metabolomics] ctx.effect 失败（可忽略）：', error && error.message) } catch { /* ignore */ }
      }
    }
  }
  if (!allowStart) return runtime.engine
  const now = Date.now()
  if (runtime.engineFail && now - runtime.engineFail.at < 60_000) throw runtime.engineFail.error
  try {
    await runtime.engine.start()
    runtime.engineFail = null
  } catch (error) {
    runtime.engineFail = { at: now, error }
    throw error
  }
  return runtime.engine
}

export async function engineStatus(runtime) {
  let engine = runtime.engine
  if (!engine) {
    try {
      await ensureEngine(runtime, { allowStart: false })
      engine = runtime.engine
    } catch (error) {
      return { ready: false, mode: 'none', error: errorText(error), python: null, packages: {}, r: null }
    }
  }
  if (!engine) return { ready: false, mode: 'none', error: 'Python 引擎未初始化', python: null, packages: {}, r: null }
  const status = typeof engine.status === 'function' ? engine.status() : { ready: false }
  if (!status.ready && !status.error && runtime.engineFail) {
    status.error = errorText(runtime.engineFail.error)
  }
  return status
}

/** 引擎请求（带长超时选项）。timeout 允许传毫秒数或 {timeout} 两种写法。 */
async function engineRequest(runtime, cmd, args, timeout) {
  const engine = await ensureEngine(runtime)
  const ms = timeout && typeof timeout === 'object' ? Number(timeout.timeout) : Number(timeout)
  return engine.request(cmd, args, Number.isFinite(ms) && ms > 0 ? { timeout: ms } : {})
}

/** ping 结果缓存（30s）：健康检查/状态轮询不必每次都跑完整探测。 */
async function enginePing(runtime) {
  const now = Date.now()
  if (runtime.pingInfo && now - runtime.pingAt < PING_TTL) return runtime.pingInfo
  try {
    const info = await engineRequest(runtime, 'ping', {}, 90_000)
    runtime.pingInfo = info
    runtime.pingAt = now
    return info
  } catch (error) {
    if (runtime.pingInfo) return runtime.pingInfo
    throw error
  }
}

/** 引擎信息 → 前端/工具统一视图（不暴露内部字段）。 */
function engineView(status, ping) {
  const info = ping || {}
  return {
    ready: Boolean(status && status.ready),
    mode: info.mode || (status && status.mode) || (status && status.ready ? 'builtin' : 'none'),
    python: info.python || (status && status.python) || null,
    version: (status && status.version) || null,
    platform: info.platform || null,
    plugin: info.plugin || null,
    packages: info.packages || (status && status.packages) || {},
    modules: info.modules || null,
    capabilities: info.capabilities || null,
    geo: info.geo === undefined ? null : Boolean(info.geo),
    metabolights: info.metabolights === undefined ? null : Boolean(info.metabolights),
    config: info.config || null,
    error: (status && status.error) || null,
    lastProgress: (status && status.lastProgress) || null,
  }
}

// ---------------------------------------------------------------------------
// runtime 状态：活动日志 / 任务
// ---------------------------------------------------------------------------

export function logActivity(runtime, entry) {
  if (!runtime.activity) runtime.activity = []
  const item = {
    at: nowIso(),
    actor: entry.actor || 'system',
    action: String(entry.action || ''),
    dataset: entry.dataset || null,
    detail: entry.detail == null ? '' : String(entry.detail).slice(0, 400),
    ok: entry.ok !== false,
    ms: Number.isFinite(entry.ms) ? entry.ms : null,
  }
  runtime.activity.push(item)
  if (runtime.activity.length > ACTIVITY_LIMIT) runtime.activity = runtime.activity.slice(-ACTIVITY_LIMIT)
  return item
}

export function activityOf(runtime, since = 0) {
  const list = runtime.activity || []
  return { ok: true, entries: list.slice(Math.max(0, Number(since) || 0)) }
}

function jobs(runtime) {
  if (!runtime.jobs) runtime.jobs = new Map()
  return runtime.jobs
}

function newJob(runtime, kind, meta = {}) {
  const id = randomUUID().slice(0, 10)
  const job = {
    id,
    kind,
    status: 'running',
    pct: 0,
    message: '准备中…',
    startedAt: nowIso(),
    finishedAt: null,
    result: null,
    error: null,
    code: null,
    ...meta,
  }
  jobs(runtime).set(id, job)
  return job
}

function finishJob(job, { ok, result, error, code }) {
  job.status = ok ? 'done' : 'failed'
  job.finishedAt = nowIso()
  job.pct = ok ? 100 : job.pct
  if (ok) job.result = result
  else {
    job.error = String(error || '未知错误')
    job.code = code || 'internal'
  }
  return job
}

function jobView(runtime, id) {
  const job = jobs(runtime).get(String(id || ''))
  if (!job) throw Object.assign(new Error('任务不存在或已过期'), { code: 'not_found' })
  // 下载中的进度从引擎 lastProgress 同步
  if (job.status === 'running' && runtime.engine && runtime.engine.progress) {
    const progress = runtime.engine.progress
    if (progress && progress.message) job.message = progress.message
    if (progress && Number.isFinite(progress.pct)) job.pct = Math.max(0, Math.min(99, progress.pct))
  }
  return { job }
}

// ---------------------------------------------------------------------------
// 健康检查 / 状态
// ---------------------------------------------------------------------------

export async function opHealth(runtime) {
  let status = null
  let startError = null
  try {
    await ensureEngine(runtime)
  } catch (error) {
    // Python 不可用只降级，不 throw：/health 必须始终 200 且给出可读原因
    startError = error
  }
  try {
    status = await engineStatus(runtime)
  } catch (error) {
    status = { ready: false, mode: 'none', error: errorText(error) }
  }
  let ping = null
  try {
    if (status && status.ready) ping = await enginePing(runtime)
  } catch { ping = null }
  const engine = engineView(status, ping)
  if (!engine.error && startError) engine.error = errorText(startError)
  const capabilities = engine.capabilities || null
  return {
    ok: true,
    engine,
    steps: (capabilities && capabilities.steps) || STEPS.map((s) => ({ ...s, available: false })),
    workdir: runtime.engine ? runtime.engine.workdir : null,
    geo: engine.geo === null ? Boolean(ping && ping.geo) : engine.geo,
    metabolights: engine.metabolights === null ? Boolean(ping && ping.metabolights) : engine.metabolights,
  }
}

export async function opState(runtime) {
  // 与 /health 一致：允许在这里拉起引擎，否则首次进页面会把「未启动」误报成「不可用」
  try { await ensureEngine(runtime) } catch { /* 未就绪时下面按错误对象降级 */ }
  const status = await engineStatus(runtime)
  let datasets = []
  let active = null
  let suggestions = []
  let recipes = []
  if (status && status.ready) {
    try {
      const list = await engineRequest(runtime, 'list', {})
      datasets = list.datasets || []
      active = list.active || null
      if (active) {
        const sug = await engineRequest(runtime, 'suggest', { dataset: active })
        suggestions = sug.suggestions || []
      }
      const rec = await engineRequest(runtime, 'recipe', {})
      recipes = rec.recipes || []
    } catch { /* 引擎忙/未就绪时保持空 */ }
  }
  let ping = null
  if (status && status.ready) {
    try { ping = await enginePing(runtime) } catch { ping = null }
  }
  const engine = engineView(status, ping)
  const capabilities = engine.capabilities || null
  return {
    ok: true,
    active,
    datasets,
    engine,
    steps: (capabilities && capabilities.steps) || STEPS.map((s) => ({ ...s, available: false })),
    plots: (capabilities && capabilities.plots) || PLOT_KINDS.map((id) => ({ id, owner: null, available: false })),
    recipes,
    suggestions,
    activity: (runtime.activity || []).slice(-50),
    workdir: runtime.engine ? runtime.engine.workdir : null,
    geo: engine.geo === null ? Boolean(ping && ping.geo) : engine.geo,
    metabolights: engine.metabolights === null ? Boolean(ping && ping.metabolights) : engine.metabolights,
  }
}

// ---------------------------------------------------------------------------
// GEO（数据源 §7.1–7.3）
// ---------------------------------------------------------------------------

export async function opGeoSearch(runtime, args = {}) {
  const started = Date.now()
  const term = String(args.term || args.query || '')
  if (!term) throw Object.assign(new Error('请提供检索关键词（term，中文或英文）'), { code: 'bad_request' })
  let result
  try {
    result = await engineRequest(runtime, 'geo_search', {
      term,
      species: args.species || 'all',
      method: args.method || 'all',
      study_type: args.study_type || 'all',
      min_samples: Number(args.min_samples) || 0,
      has_clinical: Boolean(args.has_clinical),
      limit: Number(args.limit) || 20,
      page: Number(args.page) || 0,
      sort: args.sort || 'relevance',
    }, { timeout: 120000 })
  } catch (error) {
    throw readableError(error, 'GEO 检索')
  }
  logActivity(runtime, {
    actor: args.actor || 'user',
    action: 'geo:search',
    detail: `${term} → ${result.total || 0} 条`,
    ok: true,
    ms: Date.now() - started,
  })
  return { ok: true, ...result }
}

export async function opGeoDetail(runtime, args = {}) {
  const started = Date.now()
  const gse = String(args.gse || args.accession || '')
  if (!gse) throw Object.assign(new Error('请提供 GSE 编号（gse，如 GSE13861）'), { code: 'bad_request' })
  let result
  try {
    result = await engineRequest(runtime, 'geo_detail', { gse }, { timeout: 180000 })
  } catch (error) {
    throw readableError(error, 'GEO 详情')
  }
  logActivity(runtime, { actor: args.actor || 'user', action: 'geo:detail', detail: gse, ok: true, ms: Date.now() - started })
  return { ok: true, ...result }
}

/** 异步下载 GEO 文件：立即返回 job，前端轮询 /jobs。 */
export async function opGeoDownload(runtime, args = {}) {
  const gse = String(args.gse || args.accession || '')
  if (!gse) throw Object.assign(new Error('请提供 GSE 编号（gse）'), { code: 'bad_request' })
  const destDir = String(args.dest_dir || '').trim()
  const job = newJob(runtime, 'geo_download', { gse, destDir })
  const actor = args.actor || 'user'
  ;(async () => {
    const started = Date.now()
    try {
      const engine = await ensureEngine(runtime)
      job.message = '连接 NCBI…'
      const requested = (Array.isArray(args.files) ? args.files : []).map((item) => String(item || '').trim()).filter(Boolean)
      const targetDir = destDir || path.join(engine.workdir || engine.dirs?.root || '', 'downloads', 'GEO')
      const reusable = args.overwrite ? [] : reuseLocalFiles(engine, targetDir, requested)
      let result
      if (!args.overwrite && requested.length && reusable.length === requested.length) {
        // 路径已存在且哈希一致 → 直接复用，不访问 NCBI
        result = { dir: targetDir, gse, files: reusable, ms: 0, reused: true, notes: ['文件已存在且 sha256 一致，按哈希复用'] }
        job.pct = 100
        job.message = '已存在且哈希一致，复用本地文件'
      } else {
        result = await engine.request('geo_download', {
          gse,
          files: requested,
          dest_dir: destDir || undefined,
          overwrite: Boolean(args.overwrite),
        }, { timeout: DOWNLOAD_TIMEOUT })
        var downloadInputs = rememberLocalFiles(engine, (result && result.files) || [])
        job.pct = 100
        job.message = '下载完成'
      }
      let opened = null
      if (args.open_after && result && Array.isArray(result.files)) {
        const first = result.files.find((f) => f.ok !== false)
        if (first) {
          try {
            opened = await opOpen(runtime, { path: first.path, name: gse, actor })
            await engine.request('run', { dataset: opened.dataset.id, step: 'qc', params: {} }, { timeout: STEP_TIMEOUT })
            job.message = '已下载并导入，样本质控完成'
          } catch (error) {
            job.message = `下载完成，但自动导入失败：${errorText(error)}`
          }
        }
      }
      finishJob(job, { ok: true, result: { ...result, opened } })
      logActivity(runtime, {
        actor,
        action: 'geo:download',
        detail: `${gse} → ${result.dir || destDir}（${(result.files || []).length} 个文件）`,
        ok: true,
        ms: Date.now() - started,
      })
    } catch (error) {
      finishJob(job, { ok: false, error: readableError(error, 'GEO 下载').message, code: (error && error.code) || 'internal' })
      logActivity(runtime, { actor, action: 'geo:download', detail: `${gse}：${errorText(error)}`, ok: false, ms: Date.now() - started })
    }
  })()
  return { ok: true, job: jobView(runtime, job.id).job }
}

// ---------------------------------------------------------------------------
// MetaboLights（第二数据源 §7.4）
// ---------------------------------------------------------------------------

export async function opMlSearch(runtime, args = {}) {
  const started = Date.now()
  const query = String(args.query || args.term || '')
  let result
  try {
    result = await engineRequest(runtime, 'ml_search', {
      query,
      limit: Number(args.limit) || 20,
      page: Number(args.page) || 0,
      assay_type: args.assay_type || undefined,
    }, { timeout: 120000 })
  } catch (error) {
    throw readableError(error, 'MetaboLights 检索')
  }
  logActivity(runtime, {
    actor: args.actor || 'user',
    action: 'ml:search',
    detail: `${query || '（全部）'} → ${result.total || 0} 条`,
    ok: true,
    ms: Date.now() - started,
  })
  return { ok: true, ...result }
}

export async function opMlDetail(runtime, args = {}) {
  const started = Date.now()
  const accession = String(args.accession || args.gse || '')
  if (!accession) throw Object.assign(new Error('请提供 MetaboLights 编号（accession，如 MTBLS1）'), { code: 'bad_request' })
  let result
  try {
    result = await engineRequest(runtime, 'ml_detail', { accession }, { timeout: 180000 })
  } catch (error) {
    throw readableError(error, 'MetaboLights 详情')
  }
  logActivity(runtime, { actor: args.actor || 'user', action: 'ml:detail', detail: accession, ok: true, ms: Date.now() - started })
  return { ok: true, ...result }
}

/** 异步下载 MetaboLights 文件：立即返回 job，前端轮询 /jobs。 */
export async function opMlDownload(runtime, args = {}) {
  const accession = String(args.accession || args.gse || '')
  if (!accession) throw Object.assign(new Error('请提供 MetaboLights 编号（accession）'), { code: 'bad_request' })
  const destDir = String(args.dest_dir || '').trim()
  const job = newJob(runtime, 'ml_download', { accession, destDir })
  const actor = args.actor || 'user'
  ;(async () => {
    const started = Date.now()
    try {
      const engine = await ensureEngine(runtime)
      job.message = '连接 EBI MetaboLights…'
      const requested = (Array.isArray(args.files) ? args.files : []).map((item) => String(item || '').trim()).filter(Boolean)
      const targetDir = destDir || path.join(engine.workdir || engine.dirs?.root || '', 'downloads', 'MetaboLights')
      const reusable = args.overwrite ? [] : reuseLocalFiles(engine, targetDir, requested)
      let result
      if (!args.overwrite && requested.length && reusable.length === requested.length) {
        result = { dir: targetDir, accession, files: reusable, ms: 0, reused: true, notes: ['文件已存在且 sha256 一致，按哈希复用'] }
        job.pct = 100
        job.message = '已存在且哈希一致，复用本地文件'
      } else {
        result = await engine.request('ml_download', {
          accession,
          files: requested,
          dest_dir: destDir || undefined,
          overwrite: Boolean(args.overwrite),
        }, { timeout: DOWNLOAD_TIMEOUT })
        var downloadInputs = rememberLocalFiles(engine, (result && result.files) || [])
        job.pct = 100
        job.message = '下载完成'
      }
      let opened = null
      if (args.open_after && result && Array.isArray(result.files)) {
        const first = result.files.find((f) => f.ok !== false)
        if (first) {
          try {
            opened = await opOpen(runtime, { path: first.path, name: accession, actor })
            await engine.request('run', { dataset: opened.dataset.id, step: 'qc', params: {} }, { timeout: STEP_TIMEOUT })
            job.message = '已下载并导入，样本质控完成'
          } catch (error) {
            job.message = `下载完成，但自动导入失败：${errorText(error)}`
          }
        }
      }
      finishJob(job, { ok: true, result: { ...result, opened } })
      logActivity(runtime, {
        actor,
        action: 'ml:download',
        detail: `${accession} → ${result.dir || destDir}（${(result.files || []).length} 个文件）`,
        ok: true,
        ms: Date.now() - started,
      })
    } catch (error) {
      finishJob(job, { ok: false, error: readableError(error, 'MetaboLights 下载').message, code: (error && error.code) || 'internal' })
      logActivity(runtime, { actor, action: 'ml:download', detail: `${accession}：${errorText(error)}`, ok: false, ms: Date.now() - started })
    }
  })()
  return { ok: true, job: jobView(runtime, job.id).job }
}

export function opJob(runtime, id) {
  return { ok: true, ...jobView(runtime, id) }
}

export function opJobs(runtime) {
  return { ok: true, jobs: Array.from(jobs(runtime).values()).slice(-30) }
}

// ---------------------------------------------------------------------------
// 数据集（DESIGN §2.2 open/demo/preview）
// ---------------------------------------------------------------------------

/** 数据集摘要的展示字段（代谢组学里矩阵的列叫「特征」，不是表达矩阵的行/列名）。 */
function datasetLine(dataset) {
  if (!dataset || typeof dataset !== 'object') return '未知数据集'
  const parts = [`${dataset.n_samples || '?'} 样本`, `${dataset.n_features || '?'} 特征`]
  if (dataset.source_format) parts.push(`格式 ${dataset.source_format}`)
  return `${dataset.name || dataset.id || '数据集'}（${parts.join(' × ')}）`
}

export async function opOpen(runtime, args = {}) {
  const started = Date.now()
  // body 逐字对齐 mb_engine.cmd_open 的 options（DESIGN §2.2 open args）
  const payload = {
    path: args.path || undefined,
    url: args.url || undefined,
    name: args.name || undefined,
    data_format: args.data_format || args.format || 'auto',
    member: args.member || undefined,
    object: args.object || undefined,
    transpose: args.transpose === undefined ? undefined : Boolean(args.transpose),
    delimiter: args.delimiter || undefined,
    sheet: args.sheet || undefined,
    sample_sheet: args.sample_sheet || undefined,
    clinical: args.clinical || undefined,
    feature_col: args.feature_col || undefined,
    mz_col: args.mz_col || undefined,
    rt_col: args.rt_col || undefined,
    group_col: args.group_col || undefined,
    zero_as_missing: args.zero_as_missing === undefined ? undefined : Boolean(args.zero_as_missing),
  }
  if (args.demo) payload.demo = true
  for (const key of Object.keys(payload)) if (payload[key] === undefined) delete payload[key]
  if (!payload.path && !payload.url && !payload.demo) {
    throw Object.assign(new Error('请提供 path（本地特征表/压缩包）或 url（直链）'), { code: 'bad_request' })
  }
  let result
  try {
    result = await engineRequest(runtime, 'open', payload, IMPORT_TIMEOUT)
  } catch (error) {
    throw readableError(error, '特征表导入')
  }
  const dataset = result.dataset || {}
  logActivity(runtime, {
    actor: args.actor || 'user',
    action: 'dataset:open',
    dataset: dataset.id || null,
    detail: datasetLine(dataset),
    ok: true,
    ms: Date.now() - started,
  })
  return { ok: true, ...result }
}

export async function opDemo(runtime, args = {}) {
  const started = Date.now()
  let result
  try {
    result = await engineRequest(runtime, 'open', { demo: true, name: args.name || 'demo-metabo' }, { timeout: 180000 })
  } catch (error) {
    throw readableError(error, '示例数据')
  }
  const dataset = result.dataset || {}
  logActivity(runtime, {
    actor: args.actor || 'user',
    action: 'dataset:demo',
    dataset: dataset.id || null,
    detail: datasetLine(dataset),
    ok: true,
    ms: Date.now() - started,
  })
  return { ok: true, ...result }
}

export async function opUpload(runtime, { filename, buffer, actor } = {}) {
  const started = Date.now()
  const engine = await ensureEngine(runtime)
  const safe = sanitizeName(filename)
  const dest = path.join(engine.dirs.uploads, `${Date.now().toString(36)}-${randomUUID().slice(0, 8)}-${safe}`)
  if (buffer) {
    await new Promise((resolve, reject) => {
      const stream = createWriteStream(dest)
      stream.on('error', reject)
      stream.on('finish', resolve)
      stream.end(buffer)
    })
  }
  try {
    return await opOpen(runtime, { path: dest, name: safe, actor: actor || 'user' })
  } catch (error) {
    try { unlinkSync(dest) } catch { /* ignore */ }
    throw error
  } finally {
    logActivity(runtime, {
      actor: actor || 'user',
      action: 'dataset:upload',
      detail: safe,
      ok: true,
      ms: Date.now() - started,
    })
  }
}

export async function opSelect(runtime, args = {}) {
  const engine = await ensureEngine(runtime)
  const list = await engineRequest(runtime, 'list', {})
  const target = String(args.dataset || '')
  if (!target) throw Object.assign(new Error('请提供 dataset（数据集 id）'), { code: 'bad_request' })
  if (!list.datasets.some((item) => item.id === target)) {
    throw Object.assign(new Error(`数据集 ${target} 不存在（可选：${(list.datasets || []).map((d) => d.id).join('、') || '无'}）`), { code: 'not_found' })
  }
  const result = await engineRequest(runtime, 'select', { dataset: target })
  runtime.activeDataset = target
  logActivity(runtime, { actor: args.actor || 'user', action: 'dataset:select', dataset: target, ok: true })
  return { ok: true, ...result }
}

export async function opClose(runtime, args = {}) {
  const engine = await ensureEngine(runtime)
  const result = await engineRequest(runtime, 'close', { dataset: args.dataset || undefined })
  logActivity(runtime, { actor: args.actor || 'user', action: 'dataset:close', dataset: args.dataset || null, ok: true })
  return { ok: true, ...result }
}

export async function opPreview(runtime, args = {}) {
  const engine = await ensureEngine(runtime)
  const result = await engineRequest(runtime, 'preview', { dataset: args.dataset || undefined })
  return { ok: true, ...result }
}

export async function opInspect(runtime, args = {}) {
  if (!args.path && !args.url) {
    throw Object.assign(new Error('请提供 path（本地文件/目录）或 url（直链）'), { code: 'bad_request' })
  }
  const payload = {
    path: args.path || undefined,
    url: args.url || undefined,
    data_format: args.data_format || args.format || 'auto',
    member: args.member || undefined,
    object: args.object || undefined,
    delimiter: args.delimiter || undefined,
    sheet: args.sheet || undefined,
    transpose: args.transpose === undefined ? undefined : Boolean(args.transpose),
  }
  for (const key of Object.keys(payload)) if (payload[key] === undefined) delete payload[key]
  try {
    const result = await engineRequest(runtime, 'inspect', payload, { timeout: 5 * 60 * 1000 })
    return { ok: true, ...result }
  } catch (error) {
    throw readableError(error, '格式探测')
  }
}

export async function opFormats(runtime) {
  let formats = []
  let ids = []
  try {
    const result = await engineRequest(runtime, 'formats', {})
    formats = result.formats || []
    ids = result.ids || []
  } catch { /* 引擎未就绪时给静态清单 */ }
  if (!formats.length) {
    formats = DATA_FORMATS.filter((id) => id !== 'auto').map((id) => ({ id, label: id, hint: '', available: true }))
  }
  if (!ids.length) ids = formats.map((item) => item.id).filter((id) => id && id !== 'auto')
  return { ok: true, formats, ids }
}

export async function opSamples(runtime, args = {}) {
  const started = Date.now()
  const op = String(args.op || 'list')
  if (!SAMPLE_OPS.includes(op)) {
    throw Object.assign(new Error(`不支持的样本操作：${op}（可选 ${SAMPLE_OPS.join('/')}）`), { code: 'bad_request' })
  }
  const payload = { dataset: args.dataset || undefined, op }
  const forward = ['samples', 'groups', 'column', 'group_regex', 'path', 'key', 'name', 'source', 'value', 'pattern']
  for (const key of forward) if (args[key] !== undefined) payload[key] = args[key]
  // derive 的新列名兼容两种写法：name / samples_op_name
  if (op === 'derive' && payload.name === undefined && args.samples_op_name !== undefined) payload.name = args.samples_op_name
  const result = await engineRequest(runtime, 'samples', payload, { timeout: 5 * 60 * 1000 })
  logActivity(runtime, {
    actor: args.actor || 'user',
    action: `samples:${op}`,
    dataset: args.dataset || null,
    detail: op === 'import_map' || op === 'join_obs' ? String(args.path || '') : '',
    ok: true,
    ms: Date.now() - started,
  })
  return { ok: true, ...result }
}



/** 本地已存在文件走 engine 的下载缓存（openDownloadCache）：同内容只登记一次，不复制。 */
function reuseLocalFiles(engine, destDir, requested) {
  const reused = []
  for (const name of requested) {
    const file = path.join(destDir, name)
    if (!existsSync(file)) continue
    try {
      const hit = engine.acquireFile(file, { name })
      if (hit && hit.sha256) {
        reused.push({ name, path: file, bytes: hit.bytes, ok: true, skipped: true, reused: true, sha256: hit.sha256 })
      }
    } catch { /* 读不到就当未命中，重新下载 */ }
  }
  return reused
}

/** 下载落地后把文件登记进下载缓存，返回 sha256 清单（可进 withTask 的 inputs 幂等键）。 */
function rememberLocalFiles(engine, files) {
  const inputs = []
  for (const item of files || []) {
    if (!item || !item.ok || !item.path || !existsSync(item.path)) continue
    try {
      const hit = engine.acquireFile(item.path, { name: item.name || path.basename(item.path) })
      if (hit && hit.sha256) {
        item.sha256 = hit.sha256
        inputs.push({ role: 'download', sha256: hit.sha256, path: hit.path || item.path, bytes: hit.bytes })
      }
    } catch { /* 登记失败不影响下载本身 */ }
  }
  return inputs
}

// ---------------------------------------------------------------------------
// P0 修复用的任务级 helper（冻结数据 + 独立状态 + 不可变产物）
// ---------------------------------------------------------------------------

/** 从 op 结果里挑出需要搬进 runs/<run_id>/ 的产物。 */
function artifactsOf(result, { link = false, extra = [] } = {}) {
  const out = []
  const seen = new Set()
  const push = (item) => {
    if (!item || !item.source) return
    const key = String(item.source)
    if (seen.has(key)) return
    seen.add(key)
    out.push(item)
  }
  for (const fig of (result && result.figures) || []) {
    if (fig && fig.file) push({ kind: 'figure', source: fig.file, name: fig.filename || path.basename(fig.file), link })
  }
  if (result && result.figure && result.figure.file) {
    push({ kind: 'figure', source: result.figure.file, name: result.figure.filename || path.basename(result.figure.file), link })
  }
  if (result && result.file) push({ kind: 'file', source: result.file, name: path.basename(result.file), link })
  for (const item of extra) push(item)
  return out
}

/** 结果里附上 run_id / 冻结的数据版本（UI 与证据链都用得到）。 */
function withRunMeta(result, task) {
  if (!result || typeof result !== 'object') return result
  return {
    ...result,
    run_id: task.runId,
    run_reused: Boolean(task.reused),
    data_version: task.frozen
      ? { id: task.frozen.id, revision: task.frozen.revision, artifact_id: task.frozen.artifactId }
      : (result.data_version || null),
  }
}

// ---------------------------------------------------------------------------
// 执行：步骤 / 图 / 分析 / 导出 / 自检 / 建议 / 预设流程
// ---------------------------------------------------------------------------

export async function opRun(runtime, args = {}) {
  const started = Date.now()
  const step = String(args.step || '')
  if (!STEP_IDS.includes(step)) {
    throw Object.assign(new Error(`未知步骤：${step}（可选 ${STEP_IDS.join('/')}）`), { code: 'bad_request' })
  }
  const engine = await ensureEngine(runtime)
  const params = args.params && typeof args.params === 'object' ? args.params : {}
  // P0-2：先冻结数据版本，之后只认这个 id（UI 切 dataset 不会把结果写到别的数据集）
  const task = await engine.withTask({
    op: `run:${step}`,
    dataset: args.dataset,
    params: { step, params },
    label: step,
    timeoutMs: 15 * 60 * 1000,
    reuse: args.reuse === true,
  }, async (ctx) => {
    const result = await ctx.request('run', { dataset: ctx.frozen.id, step, params }, { timeout: 15 * 60 * 1000 })
    return {
      result: { ...result, dataset_frozen: { id: ctx.frozen.id, revision: ctx.frozen.revision } },
      artifacts: artifactsOf(result),
      steps: [{ step, status: 'ok', ms: result && result.ms }],
    }
  })
  const result = withRunMeta(task.result || {}, task)
  logActivity(runtime, {
    actor: args.actor || 'agent',
    action: `run:${step}`,
    dataset: task.frozen ? task.frozen.id : (args.dataset || null),
    detail: ((result.summary && result.summary.bullets) || []).slice(0, 2).join('；'),
    ok: true,
    run: task.runId,
    ms: Date.now() - started,
  })
  return { ok: true, ...result }
}

export async function opPlot(runtime, args = {}) {
  const started = Date.now()
  const kind = String(args.kind || '')
  if (!PLOT_KINDS.includes(kind)) {
    throw Object.assign(new Error(`未知图类型：${kind}（可选 ${PLOT_KINDS.join('/')}）`), { code: 'bad_request' })
  }
  const engine = await ensureEngine(runtime)
  const params = args.params && typeof args.params === 'object' ? args.params : {}
  const style = args.style && typeof args.style === 'object' ? args.style : {}
  const task = await engine.withTask({
    op: `plot:${kind}`,
    dataset: args.dataset,
    params: { kind, params, style },
    label: kind,
    timeoutMs: 5 * 60 * 1000,
    reuse: args.reuse !== false,
  }, async (ctx) => {
    const result = await ctx.request('plot', { dataset: ctx.frozen.id, kind, params }, { timeout: 5 * 60 * 1000 })
    const artifacts = artifactsOf(result)
    // 有的引擎只回顶层 file（没有 figures[]）：按图登记，不能当成普通文件
    if (!artifacts.some((item) => item.kind === 'figure') && result && result.file) {
      artifacts.unshift({ kind: 'figure', source: result.file, name: result.filename || path.basename(result.file) })
    }
    return { result, artifacts }
  })
  if (task.reused) {
    const cached = ((task.run && task.run.artifacts) || []).find((item) => item.kind === 'figure')
    const result = { ...(task.result || {}), reused: true, file: cached ? cached.file : (task.result && task.result.file) }
    return { ok: true, figure: figureView(result, kind), ...result, run_id: task.runId, run_reused: true }
  }
  const result = withRunMeta(task.result || {}, task)
  logActivity(runtime, { actor: args.actor || 'user', action: `plot:${kind}`, dataset: task.frozen ? task.frozen.id : (args.dataset || null), ok: true, run: task.runId, ms: Date.now() - started })
  return { ok: true, figure: figureView(result, kind), ...result }
}

function figureView(result, kind) {
  const file = (result && result.file) || null
  if (!file) return null
  return {
    id: result.id,
    url: `${PREFIX}/figure?id=${encodeURIComponent(path.basename(file))}`,
    file,
    width: result.width,
    height: result.height,
    kind,
  }
}

export async function opSteps(runtime, args = {}) {
  let result
  try {
    result = await engineRequest(runtime, 'steps', { reload: Boolean(args.reload) }, { timeout: 120000 })
  } catch (error) {
    throw readableError(error, '步骤目录')
  }
  return { ok: true, ...result }
}

export async function opAnalyses(runtime, args = {}) {
  let result
  try {
    result = await engineRequest(runtime, 'analyses', { dataset: args.dataset || undefined })
  } catch (error) {
    const failure = readableError(error, '分析目录')
    // 还没载入数据集时，用 /steps 的静态 id 目录兜底：UI 首次进页面不应 400/白屏（DESIGN §4）
    if (failure.code === 'not_found') {
      const steps = await engineRequest(runtime, 'steps', {}).catch(() => null)
      const ids = (steps && Array.isArray(steps.analyses) ? steps.analyses : [])
      if (ids.length) {
        return {
          ok: true,
          static: true,
          groups: ['未载入数据集'],
          analyses: ids.map((id) => ({
            id,
            title: id,
            group: '未载入数据集',
            method: '',
            needs: [],
            params: [],
            available: false,
            reason: '先导入特征表或载入示例（action=demo）',
            module: null,
          })),
        }
      }
    }
    throw failure
  }
  return { ok: true, ...result }
}

export async function opAnalysis(runtime, args = {}) {
  const started = Date.now()
  const id = String(args.analysis || args.id || '')
  if (!id) throw Object.assign(new Error('请提供分析 id'), { code: 'bad_request' })
  const engine = await ensureEngine(runtime)
  const params = args.params && typeof args.params === 'object' ? args.params : {}
  const task = await engine.withTask({
    op: `analysis:${id}`,
    dataset: args.dataset,
    params: { id, params },
    label: id,
    timeoutMs: 20 * 60 * 1000,
    reuse: args.reuse !== false,
  }, async (ctx) => {
    const result = await ctx.request('analysis', { dataset: ctx.frozen.id, id, params }, { timeout: 20 * 60 * 1000 })
    return { result, artifacts: artifactsOf(result) }
  })
  const result = task.reused ? { ...(task.result || {}), reused: true } : withRunMeta(task.result || {}, task)
  logActivity(runtime, {
    actor: args.actor || 'agent',
    action: `analysis:${id}`,
    dataset: task.frozen ? task.frozen.id : (args.dataset || null),
    detail: ((result.summary && result.summary.bullets) || []).slice(0, 2).join('；'),
    ok: true,
    run: task.runId,
    ms: Date.now() - started,
  })
  return { ok: true, ...result, run_id: task.runId }
}

export async function opExport(runtime, args = {}) {
  const started = Date.now()
  const what = String(args.what || 'summary_json')
  if (!EXPORT_KINDS.includes(what)) throw Object.assign(new Error(`未知导出类型：${what}`), { code: 'bad_request' })
  const engine = await ensureEngine(runtime)
  const outPath = args.out_path || args.path || undefined
  const task = await engine.withTask({
    op: `export:${what}`,
    dataset: args.dataset,
    params: { what, outPath },
    label: what,
    timeoutMs: 5 * 60 * 1000,
    reuse: false,
  }, async (ctx) => {
    const result = await ctx.request('export', { dataset: ctx.frozen.id, what, path: outPath }, { timeout: 5 * 60 * 1000 })
    return {
      result,
      // 导出文件可能很大：只登记路径 + sha256，不复制
      artifacts: artifactsOf(result, { link: true }),
      steps: [{ step: `export:${what}`, status: 'ok' }],
    }
  })
  const result = withRunMeta(task.result || {}, task)
  logActivity(runtime, {
    actor: args.actor || 'user',
    action: `export:${what}`,
    dataset: task.frozen ? task.frozen.id : (args.dataset || null),
    detail: result.file,
    ok: true,
    run: task.runId,
    ms: Date.now() - started,
  })
  return { ok: true, ...result }
}

export async function opAudit(runtime, args = {}) {
  const result = await engineRequest(runtime, 'audit', { dataset: args.dataset || undefined })
  return { ok: true, ...result }
}

export async function opSuggest(runtime, args = {}) {
  const result = await engineRequest(runtime, 'suggest', { dataset: args.dataset || undefined })
  return { ok: true, ...result }
}

export async function opRecipe(runtime, args = {}) {
  const id = args.recipe || args.id
  const execute = args.run === true || Boolean(args.params && args.params.run === true)
  if (execute) return opRecipeRun(runtime, { ...args, recipe: id || 'standard' })
  if (!id) {
    const result = await engineRequest(runtime, 'recipe', {})
    return { ok: true, ...result }
  }
  const result = await engineRequest(runtime, 'recipe', { id: String(id) })
  return { ok: true, recipe: result }
}

/** 运行预设流程：顺序执行，每步一条活动日志（便于失败恢复）。 */
/**
 * 运行预设流程：顺序执行，每步一条独立状态（P0-2）。
 *
 *   - 数据集版本在任务开始时冻结，之后每步显式带同一个 dataset id（UI 切 dataset 不漂移）；
 *   - 每步成功/失败/跳过写进 run store（steps 表 + manifest.json）；
 *   - 任何一步失败整体 ok:false（旧实现失败也返回 ok:true）；
 *   - 同参数重复提交幂等；参数/数据版本变化产生新 run_id；失败/取消零已提交产物。
 */
export async function opRecipeRun(runtime, args = {}) {
  const engine = await ensureEngine(runtime)
  const recipeId = String(args.recipe || args.id || 'standard')
  const definition = await engine.request('recipe', { id: recipeId })
  const steps = (definition.steps || []).map((item) => ({ step: String(item.step), params: item.params || {} }))
  const captured = { results: [] }
  let task = null
  try {
    task = await engine.withTask({
      op: `recipe:${recipeId}`,
      dataset: args.dataset,
      params: { recipe: recipeId, steps, options: args.params && typeof args.params === 'object' ? args.params : {} },
      label: `recipe:${recipeId}`,
      timeoutMs: 15 * 60 * 1000 * Math.max(1, steps.length),
      reuse: args.reuse !== false,
      revise: Boolean(args.revise),
    }, async (ctx) => {
      const results = []
      let failure = null
      for (const item of steps) {
        if (failure) {
          results.push({ step: item.step, ok: false, status: 'skipped', error: `前置步骤 ${failure.step} 失败，已跳过` })
          continue
        }
        const startedAt = Date.now()
        try {
          const result = await ctx.request('run', { dataset: ctx.frozen.id, step: item.step, params: item.params }, { timeout: 15 * 60 * 1000 })
          results.push({ step: item.step, ok: true, status: 'ok', ms: Date.now() - startedAt, summary: result.summary })
          logActivity(runtime, {
            actor: args.actor || 'user',
            action: `run:${item.step}`,
            dataset: ctx.frozen.id,
            detail: ((result.summary && result.summary.bullets) || [])[0] || '',
            ok: true,
            run: ctx.run.runId,
          })
        } catch (error) {
          const message = error && error.message ? error.message : String(error)
          failure = { step: item.step, error: message, code: error && error.code }
          results.push({ step: item.step, ok: false, status: 'failed', error: message, code: failure.code })
          logActivity(runtime, {
            actor: args.actor || 'user',
            action: `run:${item.step}`,
            dataset: ctx.frozen.id,
            detail: message,
            ok: false,
            run: ctx.run.runId,
          })
        }
      }
      captured.results = results
      const done = results.filter((item) => item.ok).length
      const failed = results.filter((item) => item.status === 'failed').length
      // 注意：engine.withTask 看到 failed 步骤会 abort 并抛 step_failed（不会以 ok:true 提交）
      return {
        result: {
          // engine.withTask 会检查记录到的步骤：有 failed 就 abort + 抛 step_failed，
          // 所以能走到这里的都是全部成功（result.ok 必须为 true，而不是恒定 false）。
          ok: failed === 0,
          status: failed ? 'partial' : 'ok',
          recipe: recipeId,
          frozen: { id: ctx.frozen.id, revision: ctx.frozen.revision, artifact_id: ctx.frozen.artifactId },
          results,
          done,
          failed,
          total: steps.length,
          error: failure ? `${failure.step} 失败：${failure.error}` : undefined,
        },
        steps: results.map((item) => ({ step: item.step, status: item.status, error: item.error })),
        artifacts: [],
      }
    })
  } catch (error) {
    const code = (error && error.code) || 'internal'
    if (['bad_request', 'not_found', 'stale_artifact', 'prerequisite'].includes(code)) throw error
    const results = captured.results
    const failedCount = results.filter((item) => item.status === 'failed').length
    const runId = (error && error.runId) || null
    const payload = {
      ok: false,
      status: results.length ? 'partial' : (code === 'cancelled' ? 'cancelled' : 'failed'),
      recipe: recipeId,
      run_id: runId,
      results,
      done: results.filter((item) => item.ok).length,
      failed: failedCount || 1,
      total: steps.length,
      code,
      error: error && error.message ? error.message : String(error),
      run_reused: false,
    }
    logActivity(runtime, {
      actor: args.actor || 'user',
      action: `recipe:${recipeId}`,
      ok: false,
      run: runId,
      detail: payload.error,
    })
    return payload
  }
  if (task.reused) return { ...(task.result || {}), run_id: task.runId, run_reused: true }
  if (!task.run || task.run.status !== 'committed') {
    return { ok: false, status: 'failed', recipe: recipeId, run_id: task.runId, results: captured.results, total: steps.length }
  }
  return { ...(task.result || {}), run_id: task.runId, run_reused: false }
}

/** 开发/排障用：重启 Python worker（重载 lib/*.py 改动）。 */
export async function opEngineRestart(runtime) {
  const engine = runtime.engine
  runtime.engine = null
  runtime.pingInfo = null
  runtime.pingAt = 0
  runtime.engineFail = null
  if (engine && typeof engine.dispose === 'function') {
    try { await engine.dispose() } catch { /* ignore */ }
  }
  const fresh = await ensureEngine(runtime)
  const ping = await fresh.request('ping', {}, { timeout: 90_000 })
  return { ok: true, engine: { ready: true, python: ping.python, mode: ping.mode, modules: ping.modules } }
}

export async function opReset(runtime, args = {}) {
  const engine = await ensureEngine(runtime)
  await engineRequest(runtime, 'reset', {}).catch(() => {})
  runtime.activity = []
  runtime.pingInfo = null
  runtime.pingAt = 0
  jobs(runtime).clear()
  logActivity(runtime, { actor: args.actor || 'user', action: 'reset', ok: true })
  return { ok: true, datasets: [] }
}

export function opFigure(runtime, id) {
  const name = sanitizeName(String(id || ''))
  if (!/^[A-Za-z0-9._-]+\.png$/.test(name)) {
    throw Object.assign(new Error('非法图片 id'), { code: 'bad_request' })
  }
  const engine = runtime.engine
  if (!engine) throw Object.assign(new Error('引擎未就绪'), { code: 'no_python' })
  const found = findFigure(engine, name)
  if (found) return { file: found }
  throw Object.assign(new Error('图片不存在（可能已被清理，请重新生成）'), { code: 'not_found' })
}

function findFigure(engine, name) {
  if (!engine) return null
  const root = engine.workdir || (engine.dirs && engine.dirs.root)
  if (!root) return null
  const stack = [path.join(root, 'analyses'), path.join(root, 'figures'), path.join(root, 'exports')]
  const seen = new Set()
  while (stack.length) {
    const dir = stack.pop()
    if (!dir || seen.has(dir)) continue
    seen.add(dir)
    let entries
    try { entries = readdirSync(dir, { withFileTypes: true }) } catch { continue }
    for (const entry of entries) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) stack.push(full)
      else if (entry.name === name) return full
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// Agent 工具
// ---------------------------------------------------------------------------

/** 工具输出必须是 lossless JSON：去掉 undefined、把 NaN/Infinity 转 null。 */
function sanitizeValue(value, depth = 0) {
  if (depth > 16) return null
  if (value === undefined) return undefined
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value === 'bigint') return Number(value)
  if (typeof value === 'function' || typeof value === 'symbol') return null
  if (Array.isArray(value)) {
    return value.map((item) => {
      const next = sanitizeValue(item, depth + 1)
      return next === undefined ? null : next
    })
  }
  if (value && typeof value === 'object') {
    const out = {}
    for (const [key, item] of Object.entries(value)) {
      const next = sanitizeValue(item, depth + 1)
      if (next !== undefined) out[key] = next
    }
    return out
  }
  return value
}

function compactResult(result) {
  if (!result || typeof result !== 'object') return result
  const out = { ...result }
  if (Array.isArray(out.tables)) {
    out.tables = out.tables.map((table) => ({
      id: table.id,
      title: table.title,
      columns: table.columns,
      total: table.total,
      rows: Array.isArray(table.rows) ? table.rows.slice(0, 20) : [],
      truncated: table.rows && table.rows.length > 20 ? true : table.truncated,
    }))
  }
  if (out.dataset && typeof out.dataset === 'object') {
    out.dataset = {
      id: out.dataset.id,
      name: out.dataset.name,
      n_samples: out.dataset.n_samples,
      n_features: out.dataset.n_features,
      source_format: out.dataset.source_format,
      species: out.dataset.species,
      polarity: out.dataset.polarity,
      feature_meta: out.dataset.feature_meta,
      steps: out.dataset.steps,
      current: out.dataset.current,
    }
  }
  if (out.preview && typeof out.preview === 'object') {
    out.preview = {
      features: (out.preview.features || []).slice(0, 12),
      samples: (out.preview.samples || []).slice(0, 10),
      n_samples: out.preview.n_samples,
      n_features: out.preview.n_features,
      feature_meta: out.preview.feature_meta,
      group_key: out.preview.group_key,
      groups: out.preview.groups,
      value_kind: out.preview.value_kind,
      steps: out.preview.steps,
      column_count: (out.preview.columns || []).length,
      clinical: out.preview.clinical,
      library: out.preview.library,
      n_missing: out.preview.n_missing,
    }
  }
  return out
}

function bulletText(result) {
  const bullets = (result && result.summary && result.summary.bullets) || []
  return bullets.length ? bullets.join('；') : ''
}

function figureLines(result) {
  const figures = (result && result.figures) || []
  return figures.map((fig) => fig.file).filter(Boolean)
}

function searchLine(result) {
  const items = result.items || []
  if (!items.length) return '没有命中'
  return `${result.total || items.length} 条，返回 ${items.length} 条：` +
    items.slice(0, 8).map((item) => {
      const id = item.gse || item.accession || '?'
      const bits = []
      if (item.n_samples) bits.push(`${item.n_samples} 样本`)
      if (item.has_clinical) bits.push('有临床')
      if (item.method && item.method !== 'unknown') bits.push(item.method)
      return `${id}（${bits.join('，') || '—'}）`
    }).join('；')
}

export async function runTool(runtime, args = {}) {
  const action = String(args.action || 'status')
  const dataset = args.dataset || undefined
  const actor = 'agent'
  switch (action) {
    case 'status': {
      const state = await opState(runtime)
      runtime.lastState = { datasets: state.datasets, active: state.active }
      const engines = state.engine
      return {
        message: state.datasets.length
          ? `代谢组学工作台就绪：${state.datasets.length} 个数据集，当前 ${state.active || '—'}。` +
            `Python ${engines.python || '?'}${engines.metabolights ? '，MetaboLights 可用' : ''}。` +
            ((state.suggestions || [])[0] ? `建议下一步：${state.suggestions[0].title}` : '')
          : `工作台已就绪但没有数据集（Python ${engines.python || '?'}）。可以 action=search 检索 GEO，action=ml_search 检索 MetaboLights，action=open 导入本地特征表，或 action=demo 载入示例。`,
        ...state,
      }
    }
    case 'search': {
      if (String(args.source || '') === 'metabolights' || args.source === 'ml') {
        const result = await opMlSearch(runtime, { ...args, query: args.term || args.query, actor })
        return { message: `MetaboLights：${searchLine(result)}`, ...result }
      }
      const result = await opGeoSearch(runtime, { ...args, actor })
      return { message: `GEO：${searchLine(result)}`, ...result }
    }
    case 'detail': {
      const result = await opGeoDetail(runtime, { ...args, actor })
      return {
        message: `${result.gse || args.gse}：${result.n_samples || '?'} 样本，${(result.files || []).length} 个可下载文件，建议下载 ${result.suggested || '—'}`,
        ...result,
      }
    }
    case 'download': {
      const result = await opGeoDownload(runtime, { ...args, actor })
      const finished = await waitJob(runtime, result.job.id, 20 * 60 * 1000)
      if (finished.status === 'failed') throw Object.assign(new Error(finished.error || 'GEO 下载失败'), { code: finished.code || 'internal' })
      const dir = (finished.result && finished.result.dir) || args.dest_dir
      return { message: `GEO 下载完成 → ${dir}${finished.message ? `（${finished.message}）` : ''}`, ...finished }
    }
    case 'ml_search': {
      const result = await opMlSearch(runtime, { ...args, query: args.query || args.term, actor })
      return { message: `MetaboLights：${searchLine(result)}`, ...result }
    }
    case 'ml_detail': {
      const result = await opMlDetail(runtime, { ...args, actor })
      return {
        message: `${result.accession || args.accession}：${result.title || ''}；${(result.files || []).length} 个文件`,
        ...result,
      }
    }
    case 'ml_download': {
      const result = await opMlDownload(runtime, { ...args, actor })
      const finished = await waitJob(runtime, result.job.id, 20 * 60 * 1000)
      if (finished.status === 'failed') throw Object.assign(new Error(finished.error || 'MetaboLights 下载失败'), { code: finished.code || 'internal' })
      const dir = (finished.result && finished.result.dir) || args.dest_dir
      return { message: `MetaboLights 下载完成 → ${dir}${finished.message ? `（${finished.message}）` : ''}`, ...finished }
    }
    case 'open': {
      const result = await opOpen(runtime, { ...args, actor })
      const ds = result.dataset || {}
      const meta = ds.feature_meta || {}
      return {
        message: `已导入 ${datasetLine(ds)}；分组列 ${result.preview?.group_key || ds.group_key || '未设置'}；` +
          `m/z ${meta.has_mz ? '有' : '无'}、RT ${meta.has_rt ? '有' : '无'}、注释 ${meta.n_annotated ?? meta.annotated ?? 0} 个。建议下一步：run step=qc。`,
        ...compactResult(result),
      }
    }
    case 'demo': {
      const result = await opDemo(runtime, { actor })
      const ds = result.dataset || {}
      const groups = (result.preview && result.preview.groups) || []
      return {
        message: `已载入内置示例：${ds.n_samples} 样本 × ${ds.n_features} 特征（${groups.join('/') || 'case/control/QC'}，带 mz/rt、HMDB/KEGG 注释、MS2 谱库与临床列）。建议下一步：run step=qc。`,
        ...compactResult(result),
      }
    }
    case 'inspect': {
      const result = await opInspect(runtime, args)
      const meta = [
        result.format ? `格式 ${result.format}` : '',
        result.n_samples ? `${result.n_samples} 样本` : '',
        result.n_features ? `${result.n_features} 特征` : '',
        result.has_mz ? '有 m/z' : '',
        result.has_rt ? '有 RT' : '',
      ].filter(Boolean)
      return {
        message: `探测结果：${meta.join('，') || '未知'}${(result.candidates || []).length ? `；${result.candidates.length} 个候选（用 member 指定）` : ''}`,
        ...result,
      }
    }
    case 'preview': {
      const result = await opPreview(runtime, { dataset })
      const meta = result.feature_meta || {}
      return {
        message: `${result.n_samples} 样本 × ${result.n_features} 特征；分组列 ${result.group_key || '未设置'}` +
          `（分组 ${(result.groups || []).join('/') || '—'}）；值类型 ${result.value_kind || '?'}；` +
          `m/z ${meta.has_mz ? '有' : '无'}、RT ${meta.has_rt ? '有' : '无'}、注释 ${meta.n_annotated ?? meta.annotated ?? 0}；` +
          `缺失 ${result.n_missing ?? '?'}/${result.n_values ?? '?'}；已跑步骤 ${(result.steps || []).join('/') || '无'}。`,
        ...result,
      }
    }
    case 'samples': {
      const result = await opSamples(runtime, { ...args, dataset, actor })
      return {
        message: `样本表已更新：${result.total ?? (result.sample_info || []).length} 个样本；分组列 ${result.group_column || '未设置'}` +
          `（${(result.groups || []).join('/') || '—'}）${result.matched !== undefined ? `；匹配 ${result.matched} 个` : ''}`,
        ...result,
      }
    }
    case 'run': {
      const result = await opRun(runtime, { ...args, dataset, actor })
      const figs = figureLines(result)
      return {
        message: `${(result.summary && result.summary.title) || result.step}：${bulletText(result)}${figs.length ? `\n图：${figs.join('、')}` : ''}`,
        ...compactResult(result),
      }
    }
    case 'plot': {
      const result = await opPlot(runtime, { ...args, dataset, actor })
      return { message: `已出图（${result.kind || args.kind}）：${result.file}`, ...result }
    }
    case 'analyses': {
      const result = await opAnalyses(runtime, { dataset })
      const available = (result.analyses || []).filter((item) => item.available)
      return {
        message: `${(result.analyses || []).length} 个分析（可用 ${available.length} 个），分组：${(result.groups || []).join('、')}。可用：${available.slice(0, 12).map((a) => a.id).join('、') || '（对应模块尚未就绪）'}`,
        ...result,
      }
    }
    case 'analysis': {
      const result = await opAnalysis(runtime, { ...args, dataset, actor })
      const figs = figureLines(result)
      return {
        message: `${result.title}（${result.method || ''}）：${bulletText(result)}${figs.length ? `\n图：${figs.join('、')}` : ''}`,
        ...compactResult(result),
      }
    }
    case 'export': {
      const result = await opExport(runtime, { ...args, dataset, actor })
      return { message: `已导出 ${result.what} → ${result.file}（${Math.round((result.bytes || 0) / 1024)} KB）`, ...result }
    }
    case 'audit': {
      const result = await opAudit(runtime, { dataset })
      const bad = (result.checks || []).filter((c) => c.level !== 'ok')
      return {
        message: `自检得分 ${result.score}/100（${result.level}）` +
          (bad.length ? `；${bad.length} 项待改进：${bad.slice(0, 6).map((c) => c.title).join('、')}` : '；预处理与下游覆盖都齐全'),
        ...result,
      }
    }
    case 'suggest': {
      const result = await opSuggest(runtime, { dataset })
      return {
        message: (result.suggestions || []).map((s) => `${s.title}${s.step ? `（step=${s.step}）` : ''}`).join('；') || '暂无建议',
        ...result,
      }
    }
    case 'recipe': {
      const result = await opRecipe(runtime, { ...args, actor })
      if (result.recipes) {
        return { message: `预设流程：${result.recipes.map((r) => `${r.id}（${r.title}）`).join('；')}`, ...result }
      }
      if (result.results) {
        return {
          message: `预设流程 ${result.recipe}：完成 ${result.done}/${result.total} 步` +
            (result.done === result.total ? '' : `；中断于 ${result.results.filter((r) => !r.ok).map((r) => r.step).join('、')}`),
          ...result,
        }
      }
      return { message: `预设流程 ${result.recipe.id}：${(result.recipe.steps || []).map((s) => s.step).join('→')}`, ...result }
    }
    case 'formats': {
      const result = await opFormats(runtime)
      return { message: `支持 ${(result.formats || []).length} 种格式：${(result.formats || []).map((f) => f.id).join('、')}`, ...result }
    }
    case 'reset': {
      const result = await opReset(runtime, { actor })
      return { message: '工作台已清空（数据集、任务与活动日志一并重置）。', ...result }
    }
    case 'close': {
      const result = await opClose(runtime, { ...args, dataset, actor })
      return { message: result.closed ? `已关闭数据集 ${dataset || ''}` : '没有可关闭的数据集', ...result }
    }
    default:
      throw Object.assign(
        new Error(`未知 action "${action}"，可用：${TOOL_ACTIONS.join('/')}`),
        { code: 'bad_request' },
      )
  }
}

const TOOL_ACTIONS = [
  'status', 'search', 'detail', 'download', 'ml_search', 'ml_detail', 'ml_download',
  'open', 'demo', 'inspect', 'preview', 'run', 'plot', 'analyses', 'analysis',
  'export', 'audit', 'suggest', 'recipe', 'samples', 'formats', 'reset', 'close',
]

async function waitJob(runtime, id, timeoutMs) {
  const started = Date.now()
  for (;;) {
    const view = jobView(runtime, id).job
    if (view.status !== 'running') return view
    if (Date.now() - started > timeoutMs) {
      throw Object.assign(new Error('下载任务超时，请稍后用 /jobs 查询'), { code: 'timeout' })
    }
    await new Promise((resolve) => setTimeout(resolve, 800))
  }
}

function buildToolDefinition(hostRuntime) {
  const def = {
    name: 'metabolomics',
    description:
      '代谢组学分析工作台（本机 Host 插件，带可视化页面与 Python 计算引擎，LC-MS/GC-MS 特征表为核心、MS1/MS2 鉴定为辅）。用于：检索并下载 GEO / EBI MetaboLights 代谢组学数据；导入自有特征表（csv/tsv/txt/gz、xlsx、zip/tar、GEO series matrix、mzTab、ISA-Tab、msp/mgf 谱库、mzML TIC）；做 12 步流程（样品 QC → 特征过滤 → 缺失值填补 → 归一化 → 变换/标度 → PCA → PLS-DA/OPLS-DA → 单变量差异 → 批次校正 → 通路富集 → 标志物筛选 → 代谢物鉴定）；跑 36 项高级分析（VIP/S-plot/置换检验/交叉验证/多组 ANOVA/时间趋势/剂量响应/Mummichog/ORA/MSEA/RF/LASSO/ROC/组合评分/生存/RLSC/ComBat/共表达网络/加合物/分子式/MS2 匹配/分子网络）；出 25 种出版图；导出特征表、差异表、VIP、富集表、标志物、汇总 JSON、Methods、报告与可复现脚本。' +
      'action: status=查看工作台状态与建议下一步; ' +
      'search=检索 GEO 代谢组学数据（term 关键词，中文或英文；可带 species/method/study_type/min_samples/has_clinical/limit/page/sort；source=metabolights 时转 MetaboLights）; detail=GEO 系列详情与可下载文件列表（gse）; download=下载 GEO 文件到 dest_dir（gse+dest_dir，可 open_after=true 自动导入并质控）; ' +
      'ml_search=检索 MetaboLights（query，可带 limit/page/assay_type）; ml_detail=MetaboLights 研究详情（accession，如 MTBLS1）; ml_download=下载 MetaboLights 文件（accession+dest_dir，可 open_after）; ' +
      'open=导入本地/URL 特征数据（path 或 url；data_format 可强制 matrix_text/xlsx/geo_series/archive/mztab/isa_tab/msms_library/mzml_tic/rds/rdata；样本表用 sample_sheet/clinical 按样本名合并；列名映射用 feature_col/mz_col/rt_col/group_col）; demo=载入内置示例（72 样本 × 1600 特征，带 mz/rt、注释、MS2 谱库与临床列）; inspect=只探测格式与维度; preview=查看特征/样本列、缺失与注释情况; ' +
      'samples=样本表管理（op=list/update/auto_group/import_map/join_obs/derive/set_group/remove；样本名正则分组用 auto_group，临床/样本信息表用 import_map/join_obs）; ' +
      'run=按顺序跑流程步骤（qc→filter→impute→normalize→transform→pca→plsda→de→batch→enrich→biomarker→annotate，step 必填，params 可选）; ' +
      'plot=出图（kind=qc_tic/qc_rsd/qc_missing/sample_corr/sample_dendro/pca_scores/pca_variance/hotelling_t2/plsda_scores/oplsda_splot/vip_bar/permutation_test/volcano_de/fc_bar/heatmap_top/enrich_bar/enrich_dot/roc_curve/rf_importance/lasso_path/trend_plot/dose_response/mass_accuracy/mass_network/batch_pca）; ' +
      'analyses=列出全部高级分析及其可用性; analysis=运行高级分析（analysis 传 id，params 传参数）; ' +
      'export=导出（what=expr_csv/de_csv/vip_csv/enrich_csv/biomarker_csv/sample_info_csv/summary_json/protocol/methods/report/script，可用 out_path）; ' +
      'audit=确定性自检（预处理完整性、统计前提、下游覆盖）; suggest=建议下一步; recipe=预设流程（standard/quick/batch/annotate；给 recipe + run=true 可直接执行）; formats=支持的数据格式; reset=清空工作台; close=关闭数据集。' +
      '用户提到代谢组学、LC-MS/GC-MS/NMR、特征表/峰表、mz/RT、代谢物、通路富集（Mummichog/KEGG/HMDB）、PLS-DA/OPLS-DA、VIP、差异代谢物、标志物/ROC、加合物/分子式/MS2 鉴定，或要求 GEO/MetaboLights 代谢数据时，优先用本工具驱动这个工作台（与页面按钮同一套实现），不要自己写脚本重跑同类分析。' +
      '给结论前建议先 audit；不确定状态先 status；图的绝对路径可直接用 read_image 查看。',
    parameters: {
      action: {
        type: 'string',
        enum: TOOL_ACTIONS,
        description: '要执行的操作',
      },
      dataset: { type: 'string', description: '数据集 id（如 ds1）；省略则用当前活动数据集' },
      term: { type: 'string', description: '检索关键词（中文或英文，如「血清 代谢组学 LC-MS」「serum metabolomics」）' },
      gse: { type: 'string', description: 'GEO 系列编号（如 GSE13861；action=detail/download）' },
      accession: { type: 'string', description: 'MetaboLights 研究编号（如 MTBLS1；action=ml_detail/ml_download）' },
      source: { type: 'string', description: 'action=search 的数据源：geo（默认）/ metabolights' },
      species: { type: 'string', description: '物种筛选：human / mouse / all（默认 all）' },
      method: { type: 'string', description: '检索/分析的方法筛选：all / lc-ms / gc-ms / nmr / ms（默认 all）' },
      min_samples: { type: 'number', description: '最少样本数（默认 0）' },
      has_clinical: { type: 'boolean', description: '只看有人体临床信息的系列' },
      limit: { type: 'number', description: '每页条数（默认 20）' },
      page: { type: 'number', description: '页码（从 0 开始）' },
      sort: { type: 'string', description: '排序：relevance / date / samples（默认 relevance）' },
      dest_dir: { type: 'string', description: '下载目录绝对路径（默认 workdir/downloads，GEO 在 GEO/ 子目录）' },
      files: { type: 'array', description: '要下载的文件名列表（省略=建议文件）', items: { type: 'string' } },
      overwrite: { type: 'boolean', description: '下载时覆盖已存在文件（默认 false）' },
      open_after: { type: 'boolean', description: '下载后自动导入并跑样本质控' },
      path: { type: 'string', description: '本地文件/目录绝对路径（action=open/inspect；样本表用 sample_sheet）' },
      url: { type: 'string', description: 'http(s)/ftp 直链（action=open/inspect）' },
      // 参数名不能叫 `format`（JSON Schema 保留字），用 data_format
      data_format: { type: 'string', enum: DATA_FORMATS, description: '强制数据格式（默认 auto 自动识别）' },
      member: { type: 'string', description: '压缩包内的相对路径（候选见 inspect 的 candidates）' },
      object: { type: 'string', description: 'RData 内的对象名' },
      transpose: { type: 'boolean', description: '矩阵方向（true=特征×样本，导入时自动转置）' },
      delimiter: { type: 'string', description: '文本分隔符（默认自动嗅探：, / \\t / ;）' },
      sheet: { type: 'string', description: 'Excel 的 sheet 名' },
      sample_sheet: { type: 'string', description: '样本信息表路径（CSV/TSV/XLSX，按样本名合并）' },
      clinical: { type: 'string', description: '临床表路径（含时间/事件列，用于生存分析）' },
      feature_col: { type: 'string', description: '特征 id 所在的列名（默认取首列或行名）' },
      mz_col: { type: 'string', description: 'm/z 列名（如 mz、mass、m/z）' },
      rt_col: { type: 'string', description: '保留时间列名（如 rt、retention、RT(min)）' },
      group_col: { type: 'string', description: '样本分组列名（导入后自动设为分组列）' },
      zero_as_missing: { type: 'boolean', description: '把 0 当作缺失值（默认 false）' },
      name: { type: 'string', description: '数据集显示名' },
      op: { type: 'string', enum: SAMPLE_OPS, description: 'action=samples 的子操作（默认 list）' },
      samples: { type: 'array', description: '样本列表（op=update 的 [{id,name,group}] 或 op=remove 的样本名）', items: { type: 'object' } },
      groups: { type: 'array', description: '分组列表（op=update 时记录分组顺序）', items: { type: 'string' } },
      column: { type: 'string', description: 'action=samples op=set_group 的分组列名' },
      group_regex: { type: 'string', description: 'action=samples op=auto_group/derive 的正则（第 1 个捕获组=大组）' },
      step: { type: 'string', enum: STEP_IDS, description: '流程步骤（action=run 必填）' },
      params: { type: 'object', description: '步骤/分析/图的参数（如 qc: qc_label/min_features；filter: min_sample_frac/min_intensity/rsd_max；impute: method/k；normalize: method；transform: transform/scale；pca: n_comp/color_by；plsda: method/n_comp/cv_folds/permutations；de: group_col/case/control/method/paired/fdr/fc；batch: batch_key/method/qc_label；enrich: method/db/p_cut/n_perm；biomarker: model/n_features/cv_folds；annotate: adduct/ppm/formula/library）', additionalProperties: true },
      analysis: { type: 'string', description: 'action=analysis 的分析 id（见 action=analyses 返回的 id）' },
      kind: { type: 'string', enum: PLOT_KINDS, description: 'action=plot 的图类型' },
      what: { type: 'string', enum: EXPORT_KINDS, description: 'action=export 的导出类型' },
      out_path: { type: 'string', description: 'action=export 的目标文件绝对路径（可选）' },
      recipe: { type: 'string', description: 'action=recipe 的预设 id：standard / quick / batch / annotate' },
      run: { type: 'boolean', description: 'action=recipe 时是否直接执行该预设流程（默认 false 只返回步骤定义）' },
      actor: { type: 'string', description: '调用方标记（user/agent），只用于活动日志（默认 agent）' },
    },
    output: {
      schema: { type: 'object' },
      render: (_args, value) => {
        const text = value && typeof value.message === 'string' ? value.message : JSON.stringify(value)
        return [{ type: 'text', text: String(text) }]
      },
    },
    execute: async (args) => {
      const fresh = await import(new URL(`./impl.mjs?rev=${Date.now()}-${Math.random().toString(36).slice(2)}`, import.meta.url).href)
      return sanitizeValue(await fresh.runTool(hostRuntime, args || {}))
    },
  }
  // 防呆：参数名撞 JSON Schema 保留关键字会让 API 判整个 tools 数组非法
  const reservedKeywords = new Set(['type', 'enum', 'const', 'multipleOf', 'maximum', 'exclusiveMaximum', 'minimum', 'exclusiveMinimum', 'maxLength', 'minLength', 'pattern', 'maxItems', 'minItems', 'uniqueItems', 'maxContains', 'minContains', 'maxProperties', 'minProperties', 'required', 'dependentRequired', 'properties', 'patternProperties', 'additionalProperties', 'propertyNames', 'items', 'prefixItems', 'additionalItems', 'contains', 'title', 'description', 'default', 'deprecated', 'readOnly', 'writeOnly', 'examples', 'format', 'contentEncoding', 'contentMediaType', 'contentSchema', 'if', 'then', 'else', 'allOf', 'anyOf', 'oneOf', 'not'])
  const clash = Object.keys(def.parameters).filter((key) => reservedKeywords.has(key))
  if (clash.length) throw new Error(`[dsh-metabolomics] 参数名 ${clash.join('、')} 是 JSON Schema 保留关键字，请改名`)
  return def
}

const TOOL_SCHEMA_VERSION = 1

export function ensureToolRegistered(runtime) {
  if (runtime.toolRegistered && runtime.toolSchemaVersion === TOOL_SCHEMA_VERSION) return runtime.toolRegistered
  try {
    const ctx = runtime.ctx
    const tools = ctx && typeof ctx.get === 'function' ? ctx.get('tools') : ctx && ctx.tools
    if (!tools || typeof tools.register !== 'function') {
      runtime.toolRegistered = false
      return false
    }
    if (runtime.toolRegistered) {
      if (typeof runtime.toolRegistered !== 'function') return runtime.toolRegistered
      try { runtime.toolRegistered() } catch (error) {
        try { console.warn('[dsh-metabolomics] old tool dispose failed:', error && error.message ? error.message : error) } catch { /* ignore */ }
      }
      runtime.toolRegistered = false
    }
    const dispose = tools.register(buildToolDefinition(runtime))
    runtime.toolRegistered = typeof dispose === 'function' ? dispose : true
    runtime.toolSchemaVersion = TOOL_SCHEMA_VERSION
    return true
  } catch (error) {
    runtime.toolRegistered = false
    try { console.warn('[dsh-metabolomics] agent tool unavailable:', error && error.message ? error.message : error) } catch { /* ignore */ }
    return false
  }
}

// ---------------------------------------------------------------------------
// Prompt section
// ---------------------------------------------------------------------------

export function renderPromptSection(runtime) {
  const state = runtime.lastState
  if (!state || !state.datasets || state.datasets.length === 0) return ''
  const active = state.datasets.find((item) => item.id === state.active)
  const lines = ['代谢组学分析工作台（Host 插件 dsh-metabolomics）已在使用中。']
  if (active) {
    const done = (active.steps || []).length ? active.steps.join('、') : '无'
    lines.push(`当前数据集：${active.name || active.id}（${active.n_samples} 样本 × ${active.n_features} 特征），已完成步骤：${done}。`)
  }
  lines.push(
    '用户提到 GEO/MetaboLights、LC-MS/GC-MS 特征表、质控/归一化、PCA/PLS-DA、差异代谢物、通路富集（Mummichog/KEGG/HMDB）、' +
      '标志物/ROC、加合物与代谢物鉴定时，优先调用 metabolomics 工具操作这个工作台' +
      '（与页面按钮同一套实现），不要另写脚本重跑；图可用 read_image 查看返回的 PNG 路径。',
  )
  return lines.join('\n')
}

export function ensurePromptSection(runtime) {
  if (runtime.promptRegistered) return runtime.promptRegistered
  try {
    const ctx = runtime.ctx
    const systemPrompt = ctx && typeof ctx.get === 'function' ? ctx.get('systemPrompt') : ctx && ctx.systemPrompt
    if (!systemPrompt || typeof systemPrompt.section !== 'function') {
      runtime.promptRegistered = false
      return false
    }
    const dispose = systemPrompt.section({
      name: 'dsh-metabolomics-workbench',
      order: 770,
      text: () => renderPromptSection(runtime),
    })
    runtime.promptRegistered = typeof dispose === 'function' ? dispose : true
    return true
  } catch (error) {
    runtime.promptRegistered = false
    try { console.warn('[dsh-metabolomics] prompt section unavailable:', error && error.message ? error.message : error) } catch { /* ignore */ }
    return false
  }
}

// ---------------------------------------------------------------------------
// 上传
// ---------------------------------------------------------------------------

async function handleUpload(runtime, req, res) {
  const filename = sanitizeName(req.headers['x-dsh-filename'] || 'upload.csv')
  const engine = await ensureEngine(runtime)
  const dest = path.join(engine.dirs.uploads, `${Date.now().toString(36)}-${randomUUID().slice(0, 8)}-${filename}`)
  let size = 0
  await pipeline(
    req,
    async function* count(source) {
      for await (const chunk of source) {
        size += chunk.length
        if (size > MAX_UPLOAD_BYTES) throw Object.assign(new Error('文件超过 8 GiB 上限'), { code: 'bad_request' })
        yield chunk
      }
    },
    createWriteStream(dest),
  )
  const result = await opOpen(runtime, { path: dest, name: filename, actor: 'user' })
  return { ok: true, ...result }
}

// ---------------------------------------------------------------------------
// 路由（DESIGN §2.6：GET /health /state /formats /analyses /activity /jobs /figure /steps
//              POST /geo/* /ml/* /dataset/* /inspect /preview /samples /run /plot /analysis
//                   /export /audit /suggest /recipe/run /reset）
// ---------------------------------------------------------------------------

async function serveFigure(runtime, url, res) {
  const id = url.searchParams.get('id')
  const entry = opFigure(runtime, id)
  let stat
  try { stat = statSync(entry.file) } catch {
    throw Object.assign(new Error('图片文件已被清理，请重新生成'), { code: 'not_found' })
  }
  res.writeHead(200, {
    'content-type': 'image/png',
    'cache-control': 'private, max-age=3600',
    'content-length': String(stat.size),
  })
  await pipeline(createReadStream(entry.file), res)
}

export async function handle(runtime, req, res) {
  ensureToolRegistered(runtime)
  ensurePromptSection(runtime)
  if (!hostIsLoopback(req)) {
    return sendJson(res, 403, { ok: false, code: 'forbidden', error: '拒绝：只接受本机（回环地址）请求' })
  }
  const url = new URL(req.url || '/', 'http://127.0.0.1')
  if (!url.pathname.startsWith(PREFIX)) {
    return sendJson(res, 404, { ok: false, code: 'not_found', error: '未知路由' })
  }
  const route = url.pathname.slice(PREFIX.length) || '/'
  const method = (req.method || 'GET').toUpperCase()
  const key = `${method} ${route}`
  try {
    switch (key) {
      case 'GET /health':
        return sendJson(res, 200, await opHealth(runtime))
      case 'GET /state': {
        const state = await opState(runtime)
        runtime.lastState = { datasets: state.datasets, active: state.active }
        return sendJson(res, 200, state)
      }
      case 'GET /formats':
        return sendJson(res, 200, await opFormats(runtime))
      case 'GET /steps':
        return sendJson(res, 200, await opSteps(runtime, { reload: url.searchParams.get('reload') === '1' }))
      case 'GET /analyses':
        return sendJson(res, 200, await opAnalyses(runtime, { dataset: url.searchParams.get('dataset') || undefined }))
      case 'GET /activity':
        return sendJson(res, 200, activityOf(runtime, Number(url.searchParams.get('since') || 0)))
      case 'GET /jobs':
        return sendJson(res, 200, url.searchParams.get('id') ? opJob(runtime, url.searchParams.get('id')) : opJobs(runtime))
      case 'GET /figure':
        return await serveFigure(runtime, url, res)
      case 'POST /geo/search':
        return sendJson(res, 200, await opGeoSearch(runtime, await readJson(req)))
      case 'POST /geo/detail':
        return sendJson(res, 200, await opGeoDetail(runtime, await readJson(req)))
      case 'POST /geo/download':
        return sendJson(res, 200, await opGeoDownload(runtime, await readJson(req)))
      case 'POST /ml/search':
        return sendJson(res, 200, await opMlSearch(runtime, await readJson(req)))
      case 'POST /ml/detail':
        return sendJson(res, 200, await opMlDetail(runtime, await readJson(req)))
      case 'POST /ml/download':
        return sendJson(res, 200, await opMlDownload(runtime, await readJson(req)))
      case 'POST /dataset/open':
        return sendJson(res, 200, await opOpen(runtime, await readJson(req)))
      case 'POST /dataset/demo':
        return sendJson(res, 200, await opDemo(runtime, await readJson(req)))
      case 'POST /dataset/upload':
        return sendJson(res, 200, await handleUpload(runtime, req, res))
      case 'POST /dataset/select':
        return sendJson(res, 200, await opSelect(runtime, await readJson(req)))
      case 'POST /dataset/close':
        return sendJson(res, 200, await opClose(runtime, await readJson(req)))
      case 'POST /inspect':
        return sendJson(res, 200, await opInspect(runtime, await readJson(req)))
      case 'POST /preview':
        return sendJson(res, 200, await opPreview(runtime, await readJson(req)))
      case 'POST /samples':
        return sendJson(res, 200, await opSamples(runtime, await readJson(req)))
      case 'POST /run':
        return sendJson(res, 200, await opRun(runtime, await readJson(req)))
      case 'POST /plot':
        return sendJson(res, 200, await opPlot(runtime, await readJson(req)))
      case 'POST /analysis':
        return sendJson(res, 200, await opAnalysis(runtime, await readJson(req)))
      case 'POST /export':
        return sendJson(res, 200, await opExport(runtime, await readJson(req)))
      case 'POST /audit':
        return sendJson(res, 200, await opAudit(runtime, await readJson(req)))
      case 'POST /suggest':
        return sendJson(res, 200, await opSuggest(runtime, await readJson(req)))
      case 'POST /recipe':
        return sendJson(res, 200, await opRecipe(runtime, await readJson(req)))
      case 'POST /recipe/run':
        return sendJson(res, 200, await opRecipeRun(runtime, await readJson(req)))
      case 'POST /reset':
        return sendJson(res, 200, await opReset(runtime, await readJson(req)))
      case 'POST /engine/restart':
        return sendJson(res, 200, await opEngineRestart(runtime))
      default:
        return sendJson(res, 404, { ok: false, code: 'not_found', error: `未知路由 ${key}` })
    }
  } catch (error) {
    const failure = toFailure(error)
    if (failure && failure.body) return sendJson(res, failure.status, failure.body)
    return sendJson(res, 500, { ok: false, code: 'internal', error: errorText(error) })
  }
}
