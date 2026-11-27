/**
 * 工作台统计口径契约测试（node:test）——dsh-metabolomics
 *
 * 口径卡的职责是「把这次分析用了什么设计/预处理/检验/阈值」说清楚，
 * 且**取不到就写 null**。这里用最小 dataset summary 夹具钉死：
 * 齐全 → ok_to_cite、缺字段 → caveat、n<2 → warn、空数据集 → null 而不是编造。
 *
 * 运行：node --test test/test_stats_contract.mjs
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

import { buildWorkbenchContract, PLUGIN_VERSION } from '../lib/impl.mjs'

const pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'))

function baseDataset(overrides = {}) {
  return {
    id: 'ds_demo',
    name: 'MTBLS_demo',
    path: '/data/MTBLS_demo/features.csv',
    source_format: 'matrix_text',
    n_samples: 8,
    n_features: 1600,
    species: 'human',
    polarity: 'pos',
    value_kind: 'scaled',
    created_at: '2026-11-27T09:00:00Z',
    feature_meta: { has_mz: true, has_rt: true, n_annotated: 210 },
    group_key: 'group',
    groups: ['case', 'control'],
    group_counts: { case: 4, control: 4 },
    donor_key: 'subject',
    n_donors: 8,
    batch_key: 'batch',
    n_qc: 6,
    sample_columns: [
      { key: 'sample', kind: 'text', n_unique: 8 },
      { key: 'group', kind: 'categorical', n_unique: 2 },
      { key: 'batch', kind: 'categorical', n_unique: 2 },
      { key: 'subject', kind: 'categorical', n_unique: 8 },
    ],
    steps: ['qc', 'filter', 'impute', 'normalize', 'transform', 'batch', 'plsda', 'de'],
    history: [
      { at: '2026-11-27T10:00:00Z', step: 'qc', ok: true, ms: 1200, params: { qc_label: 'QC', min_features: 100 }, summary: 'QC 通过' },
      { at: '2026-11-27T10:01:00Z', step: 'normalize', ok: true, ms: 800, params: { method: 'pqn' }, summary: 'PQN' },
      { at: '2026-11-27T10:02:00Z', step: 'transform', ok: true, ms: 500, params: { transform: 'log2', scale: 'auto' }, summary: 'log2 + auto' },
      { at: '2026-11-27T10:03:00Z', step: 'batch', ok: true, ms: 900, params: { method: 'rlsc', batch_key: 'batch', qc_label: 'QC' }, summary: 'RLSC' },
      { at: '2026-11-27T10:04:00Z', step: 'plsda', ok: true, ms: 3400, params: { method: 'oplsda', n_comp: 2, cv_folds: 7, permutations: 200 }, summary: 'OPLS-DA' },
      {
        at: '2026-11-27T10:05:00Z', step: 'de', ok: true, ms: 2100,
        params: { method: 'welch', group_col: 'group', case: 'case', control: 'control', paired: false, fdr: 0.05, fc: 1.5 },
        summary: '差异分析完成',
      },
    ],
    analyses_run: [{ at: '2026-11-27T10:06:00Z', id: 'plsda_vip', ok: true, ms: 1200 }],
    ...overrides,
  }
}

test('口径 1) 齐全：分组 n / 批次与 QC / 方法阈值 / 值类型都可回查', () => {
  const contract = buildWorkbenchContract(baseDataset())
  assert.equal(contract.schema, 'dsh-workbench.stats_contract/1')
  assert.equal(contract.plugin, 'dsh-metabolomics')
  assert.equal(contract.plugin_version, PLUGIN_VERSION)
  assert.equal(contract.plugin_version, pkg.version, 'PLUGIN_VERSION 必须与 package.json 一致')
  assert.equal(contract.dataset.n_samples, 8)
  assert.equal(contract.dataset.n_features, 1600)
  assert.equal(contract.dataset.polarity, 'pos')
  assert.equal(contract.dataset.value_kind, 'scaled')
  assert.deepEqual(contract.design.groups, ['case', 'control'])
  assert.deepEqual(contract.design.n_per_group, { case: 4, control: 4 })
  assert.equal(contract.design.n_donors, 8)
  assert.equal(contract.design.batch_key, 'batch')
  assert.equal(contract.design.n_batches, 2)
  assert.equal(contract.design.batch_corrected, true)
  assert.equal(contract.design.qc.n_qc, 6)
  assert.equal(contract.design.qc.label, 'QC')
  assert.equal(contract.design.batch_columns.join(','), 'batch')
  // 最近一次分析（de）
  assert.equal(contract.inference.last_step, 'de')
  assert.equal(contract.inference.method.method, 'welch')
  assert.equal(contract.inference.design.case, 'case')
  assert.equal(contract.inference.design.control, 'control')
  assert.equal(contract.inference.thresholds.fdr, 0.05)
  assert.equal(contract.inference.thresholds.fc, 1.5)
  // 关键步骤的真实参数（归一化 / PLS-DA / 差异 / 批次）
  assert.equal(contract.workflow.normalize.method, 'pqn')
  assert.equal(contract.workflow.transform.transform, 'log2')
  assert.equal(contract.workflow.plsda.method, 'oplsda')
  assert.equal(contract.workflow.plsda.permutations, 200)
  assert.equal(contract.workflow.de.fdr, 0.05)
  assert.equal(contract.workflow.batch.method, 'rlsc')
  assert.equal(contract.identity.n_steps_done, 8)
  assert.equal(contract.identity.fingerprint, null, '没有内容指纹时写 null，不编 hash')
  assert.deepEqual(contract.caveats.filter((item) => item.level === 'warn'), [])
  assert.equal(contract.ok_to_cite, true)
  assert.doesNotThrow(() => JSON.stringify(contract), '契约必须可 JSON 序列化')
})

test('口径 2) 缺分组 / 未校正批次 / 最近失败 / QC 未知 → caveats 并拒绝引用', () => {
  const contract = buildWorkbenchContract(baseDataset({
    group_key: '',
    groups: [],
    group_counts: {},
    donor_key: '',
    n_donors: null,
    value_kind: 'raw',
    n_qc: null,
    batch_key: '',
    sample_columns: [{ key: 'batch', kind: 'categorical', n_unique: 3 }],
    feature_meta: { has_mz: false, has_rt: false, n_annotated: 0 },
    steps: ['qc'],
    history: [{ at: '2026-11-27T10:00:00Z', step: 'qc', ok: false, ms: 900, params: { qc_label: '' }, summary: '样本信息缺失' }],
    analyses_run: [],
  }))
  const codes = contract.caveats.map((item) => item.code)
  assert.ok(codes.includes('no_groups'))
  assert.ok(codes.includes('batch_not_corrected'))
  assert.ok(codes.includes('last_step_failed'))
  assert.ok(codes.includes('qc_unknown'))
  assert.ok(codes.includes('no_donor_column'))
  assert.ok(codes.includes('raw_intensity'))
  assert.ok(codes.includes('no_multivariate'))
  assert.ok(codes.includes('no_annotation'))
  assert.ok(codes.includes('workflow_params_missing'))
  assert.equal(contract.ok_to_cite, false)
  assert.equal(contract.inference.last_step, 'qc')
  assert.equal(contract.inference.ok, false)
})

test('口径 3) 每组 n<2 必须警告；无历史时只给 info（不误报成错误）', () => {
  const lowN = buildWorkbenchContract(baseDataset({ group_counts: { case: 1, control: 4 } }))
  assert.ok(lowN.caveats.some((item) => item.code === 'min_group_n_below_2'))
  assert.equal(lowN.ok_to_cite, false)

  const noHistory = buildWorkbenchContract(baseDataset({ history: [], analyses_run: [], steps: [] }))
  assert.equal(noHistory.inference, null)
  assert.deepEqual(noHistory.workflow, {})
  assert.ok(noHistory.caveats.some((item) => item.code === 'no_history' && item.level === 'info'))
})

test('口径 4) 空数据集不抛错，字段为 null/空而不是编造', () => {
  const contract = buildWorkbenchContract({})
  assert.equal(contract.dataset.n_samples, null)
  assert.equal(contract.dataset.n_features, null)
  assert.equal(contract.dataset.value_kind, '')
  assert.deepEqual(contract.design.groups, [])
  assert.deepEqual(contract.design.n_per_group, {})
  assert.equal(contract.design.qc.n_qc, null)
  assert.equal(contract.design.batch_key, '')
  assert.equal(contract.inference, null)
  assert.deepEqual(contract.workflow, {})
  assert.equal(contract.identity.fingerprint, null)
  assert.ok(contract.caveats.some((item) => item.code === 'no_data'))
  assert.ok(contract.caveats.some((item) => item.code === 'no_groups'))
  assert.ok(contract.caveats.some((item) => item.code === 'scale_unknown'))
  assert.equal(contract.ok_to_cite, false)
})
