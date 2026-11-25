#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-metabolomics — P0-3 回归测试：配对唯一 ID / 事件显式映射 / 零显著富集

对应 BIO_P0_BACKLOG.md §4 P0-3：
  * 配对必须用显式唯一 ID（支持复合 pair_col；重复键 → bad_request；无配对列不再按行序配对，
    缺配对信息 → prerequisite）；
  * 生存事件列必须显式映射；未知事件值默认验证失败（`event_unknown="na"` 才按 NA 丢弃并 warning）；
  * `step_enrich` 零显著时输出合法空表 + n_significant=0 + warning（不再退化为 top 50），
    全量排序富集交给 `metabolite_set_enrich`(MSEA)；
  * `_significant` 里 fc（倍数）与 log2(FC) 的比较口径一致。

用法：cd plugins/dsh-metabolomics && MPLBACKEND=Agg python3 test/test_p0_pairing_events.py
成功打印 PASS 并以 0 退出；失败打印 FAIL 并以 1 退出。
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mb-mpl-"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mb_biomarker as B  # noqa: E402
import mb_de as D  # noqa: E402
import mb_enrich as E  # noqa: E402
from mb_common import EngineError  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="mb-p0-pair-")
PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  PASS {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))


def out(tag: str) -> str:
    path = os.path.join(OUTDIR, tag)
    os.makedirs(path, exist_ok=True)
    return path


def expect_error(name: str, fn, code: str) -> None:
    try:
        fn()
        check(name, False, "没有抛错")
    except EngineError as error:
        check(name, error.code == code, f"{error.code}: {str(error.message)[:80]}")
    except Exception as error:  # noqa: BLE001
        check(name, False, f"{type(error).__name__}: {error}")


def make_ctx(X, samples, sample_info, features=None, X_kind="lognorm", results=None):
    arr = np.asarray(X, dtype=np.float64)
    n, p = arr.shape
    features = list(features) if features is not None else [f"F{j + 1:04d}" for j in range(p)]
    feature_info = pd.DataFrame(
        {"name": [f"M{j + 1}" for j in range(p)], "mz": np.nan, "rt": np.nan},
        index=pd.Index(features, name="feature"),
    )
    info = pd.DataFrame(sample_info, index=pd.Index([str(s) for s in samples], name="sample"))
    return {
        "dataset_id": "ds_p0_pair",
        "name": "p0-pairing",
        "features": features,
        "samples": [str(s) for s in samples],
        "X": np.asarray(arr, dtype=np.float32),
        "X_kind": X_kind,
        "raw": None,
        "feature_info": feature_info,
        "sample_info": info,
        "group_key": "group",
        "groups": ["case", "control"],
        "matrix_order": "samples_by_features",
        "species": "human",
        "results": {} if results is None else results,
        "library": None,
        "rng": np.random.default_rng(0),
        "workdir": OUTDIR,
        "outdir": OUTDIR,
        "engine": {"python": sys.executable, "mode": "builtin"},
        "notes": [],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# 1) 配对：唯一 ID / 复合键 / 重复键 / 缺配对信息
# ---------------------------------------------------------------------------


def paired_ctx(subject_column: str = "subject", shuffle: bool = True):
    """6 对样本：控制组按 P01..P06，实验组按反序（乱序配对反例）。"""
    rng = np.random.default_rng(17)
    n_pairs = 6
    n_features = 20
    base = rng.uniform(1000.0, 5000.0, (n_pairs, n_features))
    case_values = base + 2.0
    order_control = list(range(n_pairs))
    order_case = list(reversed(range(n_pairs))) if shuffle else list(range(n_pairs))
    X = np.vstack([base[order_control], case_values[order_case]])
    labels = ["control"] * n_pairs + ["case"] * n_pairs
    subjects = [f"P{i + 1:02d}" for i in order_control] + [f"P{i + 1:02d}" for i in order_case]
    samples = [f"S{i + 1:02d}" for i in range(2 * n_pairs)]
    info = {"group": labels, subject_column: subjects,
            "visit": ["V1"] * n_pairs + ["V1"] * n_pairs}
    ctx = make_ctx(X, samples, info)
    ctx["group_key"] = "group"
    return ctx


def test_pairing() -> None:
    print("\n[1] 配对必须用显式唯一 ID")
    ctx = paired_ctx()
    try:
        comp = D._resolve_comparison(ctx, {"case": "case", "control": "control"})
        warnings: list[str] = []
        case_idx, control_idx = D._paired_indices(comp, {}, warnings)
    except Exception:  # noqa: BLE001
        check("显式配对列可配对", False, traceback.format_exc(limit=2))
        return
    check("显式配对列可配对", len(case_idx) == len(control_idx) == 6,
          f"{len(case_idx)} vs {len(control_idx)}")
    info = ctx["sample_info"]
    match = all(str(info.iloc[int(a)]["subject"]) == str(info.iloc[int(b)]["subject"])
                for a, b in zip(case_idx, control_idx))
    check("乱序样本仍按 subject 一一配对", match,
          str([(info.iloc[int(a)]["subject"], info.iloc[int(b)]["subject"])
               for a, b in zip(case_idx, control_idx)][:3]))

    # 无配对列 → prerequisite（不得再按行序配对）
    no_pair = paired_ctx()
    no_pair["sample_info"] = no_pair["sample_info"].drop(columns=["subject"])
    expect_error("无配对列 → prerequisite（不再按行序配对）",
                 lambda: D._paired_indices(
                     D._resolve_comparison(no_pair, {"case": "case", "control": "control"}),
                     {}, []), "prerequisite")

    # 重复配对键 → bad_request
    dup = paired_ctx()
    dup["sample_info"] = dup["sample_info"].copy()
    dup["sample_info"].iloc[1, dup["sample_info"].columns.get_loc("subject")] = "P01"
    expect_error("重复配对键 → bad_request",
                 lambda: D._paired_indices(
                     D._resolve_comparison(dup, {"case": "case", "control": "control"}),
                     {}, []), "bad_request")

    # 复合配对键：3 个受试者 × 2 次 visit → subject 在组内重复（单独用必须 bad_request），
    # subject+visit 才唯一（6 对）。旧实现会按 subject 取第一条，等于把两次 visit 混成一对。
    comp_ctx = paired_ctx()
    n_pairs = 6
    base_subjects = [f"P{i + 1:02d}" for i in range(3)] * 2   # P01,P02,P03,P01,P02,P03
    base_visits = ["V1", "V2"] * 3                            # 组内 6 行 = 3×2 唯一组合
    subjects = base_subjects + base_subjects
    visits = base_visits + base_visits
    comp_ctx["sample_info"] = comp_ctx["sample_info"].copy()
    comp_ctx["sample_info"]["subject"] = subjects
    comp_ctx["sample_info"]["visit"] = visits
    comp = D._resolve_comparison(comp_ctx, {"case": "case", "control": "control"})
    try:
        case_idx, control_idx = D._paired_indices(comp, {"pair_col": ["subject", "visit"]}, [])
        ok = len(case_idx) == n_pairs == len(control_idx)
        check("复合 pair_col（list：[subject, visit]）可配对", ok, f"{len(case_idx)} 对")
    except Exception:  # noqa: BLE001
        check("复合 pair_col（list：[subject, visit]）可配对", False, traceback.format_exc(limit=2))
    try:
        case_idx, control_idx = D._paired_indices(comp, {"pair_col": "subject+visit"}, [])
        check("复合 pair_col（字符串 'subject+visit'）可配对", len(case_idx) == n_pairs,
              f"{len(case_idx)} 对")
    except Exception:  # noqa: BLE001
        check("复合 pair_col（字符串 'subject+visit'）可配对", False, traceback.format_exc(limit=2))
    # 只给 subject（重复键）仍应 bad_request
    expect_error("复合场景下只给重复的 subject → bad_request",
                 lambda: D._paired_indices(
                     D._resolve_comparison(comp_ctx, {"case": "case", "control": "control"}),
                     {"pair_col": "subject"}, []), "bad_request")

    # 端到端：step_de paired 使用新配对
    try:
        res = D.step_de(ctx, {"paired": True, "case": "case", "control": "control"},
                        out("de-paired"))
        check("step_de paired 端到端可跑", res["store"]["de"]["paired"] is True)
    except Exception:  # noqa: BLE001
        check("step_de paired 端到端可跑", False, traceback.format_exc(limit=2))


# ---------------------------------------------------------------------------
# 2) 事件列显式映射
# ---------------------------------------------------------------------------


def survival_ctx(events: list[str], times: list[float] | None = None):
    n = len(events)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n, 10))
    samples = [f"S{i + 1:02d}" for i in range(n)]
    info = {"group": ["case", "control"] * (n // 2),
            "os_time": list(times) if times is not None else list(np.arange(n) + 1.0),
            "os_event": list(events)}
    return make_ctx(X, samples, info)


def test_events() -> None:
    print("\n[2] 生存事件列显式映射")
    ctx = survival_ctx(["dead", "alive", "censored", "dead", "alive", "censored"])
    try:
        times, events, time_col, event_col = B._survival_columns(ctx, {})
        check("已知事件值映射正确", list(map(int, events)) == [1, 0, 0, 1, 0, 0],
              f"{list(map(int, events))}")
        check("事件列名回传", event_col == "os_event", str(event_col))
    except Exception:  # noqa: BLE001
        check("已知事件值映射正确", False, traceback.format_exc(limit=2))

    # 未知事件值：默认必须验证失败（不得静默记为删失）
    unknown = survival_ctx(["dead", "lost", "alive", "dead", "alive", "lost"])
    expect_error("未知事件值默认 → bad_request",
                 lambda: B._survival_columns(unknown, {}), "bad_request")

    # event_unknown="na"：按 NA 丢弃并 warning（这些样本的时间置 NaN，下游自动剔除）
    try:
        times, events, _tc, _ec = B._survival_columns(unknown, {"event_unknown": "na"})
        dropped = int(np.isnan(times).sum())
        check("event_unknown='na'：未知事件按 NA 丢弃", dropped == 2, f"{dropped} 个被置 NaN")
    except Exception:  # noqa: BLE001
        check("event_unknown='na'：未知事件按 NA 丢弃", False, traceback.format_exc(limit=2))

    # 显式影射：progression=1 / stable=0
    mapped = survival_ctx(["progression", "stable", "progression", "stable", "progression", "stable"])
    try:
        times, events, _tc, _ec = B._survival_columns(
            mapped, {"event_map": {"progression": 1, "stable": 0}})
        check("显式 event_map 生效", list(map(int, events)) == [1, 0, 1, 0, 1, 0],
              f"{list(map(int, events))}")
    except Exception:  # noqa: BLE001
        check("显式 event_map 生效", False, traceback.format_exc(limit=2))

    # 0/1 数值事件仍正常
    numeric = survival_ctx(["0", "1", "0", "1", "0", "1"])
    try:
        _t, events, _tc, _ec = B._survival_columns(numeric, {})
        check("0/1 数值事件仍可解析", list(map(int, events)) == [0, 1, 0, 1, 0, 1],
              f"{list(map(int, events))}")
    except Exception:  # noqa: BLE001
        check("0/1 数值事件仍可解析", False, traceback.format_exc(limit=2))


# ---------------------------------------------------------------------------
# 3) 零显著富集：合法空表 + MSEA 承担全量排序
# ---------------------------------------------------------------------------


def enrich_ctx(p_adj_value: float = 1.0):
    rng = np.random.default_rng(8)
    n, p = 30, 40
    X = rng.uniform(1000.0, 5000.0, (n, p))
    samples = [f"S{i + 1:02d}" for i in range(n)]
    info = {"group": ["control"] * 15 + ["case"] * 15}
    features = [f"F{j + 1:04d}" for j in range(p)]
    de_table = pd.DataFrame({
        "feature": features,
        "log2FC": rng.normal(0.0, 0.3, p),
        "p": np.full(p, 0.9),
        "p_adj": np.full(p, float(p_adj_value)),
        "regulation": np.full(p, "ns", dtype=object),
    })
    results = {"de": {"table": de_table, "case": "case", "control": "control",
                      "log2fc_basis": "log2-mean-difference"}}
    return make_ctx(X, samples, info, features=features, results=results)


def test_zero_significant() -> None:
    print("\n[3] 零显著富集：合法空表 + warning")
    ctx = enrich_ctx(p_adj_value=1.0)
    try:
        payload = E.step_enrich(ctx, {"p_cut": 0.05}, out("enrich-zero"))
    except EngineError as error:
        check("零显著 step_enrich 不抛 prerequisite（应返回空结果）", False,
              f"{error.code}: {str(error.message)[:100]}")
        return
    except Exception:  # noqa: BLE001
        check("零显著 step_enrich 可跑", False, traceback.format_exc(limit=2))
        return
    metrics = payload.get("summary", {}).get("metrics") or {}
    check("零显著：n_significant = 0", int(metrics.get("n_significant", -1)) == 0,
          str(metrics.get("n_significant")))
    check("零显著：n_pathways = 0", int(metrics.get("n_pathways", -1)) == 0,
          str(metrics.get("n_pathways")))
    tables = payload.get("tables") or []
    check("零显著：返回合法空表", bool(tables) and int(tables[0].get("total") or 0) == 0,
          f"{len(tables)} 张表, total={tables[0].get('total') if tables else None}")
    check("零显著：给出 warning", any("显著" in str(w) for w in payload.get("warnings") or []),
          str(payload.get("warnings"))[:140])
    joined = " ".join(str(w) for w in payload.get("warnings") or []) + \
             " ".join(str(n) for n in payload.get("notes") or [])
    check("零显著：提示用 MSEA（全量排序富集）", "MSEA" in joined or "msea" in joined,
          joined[:140])
    store = (payload.get("store") or {}).get("enrich") or {}
    sig = store.get("significant")
    check("零显著：store.significant 为空（不再退化 top50）",
          isinstance(sig, list) and len(sig) == 0, str(sig)[:80])

    # 分析级别同样把零显著当合法结果（空表 + n_significant=0），不再 prerequisite
    for analysis_id, runner in (
        ("mummichog_pathway", E.analysis_mummichog_pathway),
        ("ora_kegg", E.analysis_ora_kegg),
        ("ora_hmdb_class", E.analysis_ora_hmdb_class),
    ):
        try:
            analysis_payload = runner(enrich_ctx(p_adj_value=1.0), {"p_cut": 0.05},
                                      out(f"enrich-zero-{analysis_id}"))
        except EngineError as error:
            check(f"零显著分析 {analysis_id} 返回空结果（不抛错）", False,
                  f"{error.code}: {str(error.message)[:80]}")
            continue
        analysis_metrics = analysis_payload.get("summary", {}).get("metrics") or {}
        analysis_tables = analysis_payload.get("tables") or []
        check(f"零显著分析 {analysis_id}：n_significant=0 且空表",
              int(analysis_metrics.get("n_significant", -1)) == 0
              and bool(analysis_tables) and int(analysis_tables[0].get("total") or 0) == 0,
              f"n_sig={analysis_metrics.get('n_significant')}, total="
              f"{analysis_tables[0].get('total') if analysis_tables else None}")


def test_significant_fc_semantics() -> None:
    print("\n[4] _significant 的 fc 口径（倍数 vs log2FC）")
    frame = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "log2FC": [np.log2(1.5), np.log2(2.5), 0.0],
        "p": [0.01, 0.01, 0.01],
        "p_adj": [0.01, 0.01, 0.01],
    })
    sig15 = E._significant(frame, 0.05, 1.5)
    check("fc=1.5 保留 |log2FC|=log2(1.5) 的特征（含边界）", set(sig15["feature"]) == {"a", "b"},
          str(list(sig15["feature"])))
    sig25 = E._significant(frame, 0.05, 2.5)
    check("fc=2.5 只保留 |log2FC|≥log2(2.5) 的特征", set(sig25["feature"]) == {"b"},
          str(list(sig25["feature"])))
    sig_none = E._significant(frame, 0.05)
    check("不给 fc 时按 p_adj 选显著", set(sig_none["feature"]) == {"a", "b", "c"},
          str(list(sig_none["feature"])))


# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("P0-3 回归：配对唯一 ID / 事件显式映射 / 零显著富集")
    print("=" * 78)
    test_pairing()
    test_events()
    test_zero_significant()
    test_significant_fc_semantics()
    print("\n" + "=" * 78)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
