"""
dsh-metabolomics — 代谢物鉴定（DESIGN.md §3 `annotate` + §4 group=代谢物鉴定）

方法（文献 / 真实实现）：
  * 加合物推断：按常见加合物质量差在特征间配对，反推中性质量（MS-DIAL / CAMERA 思路；内置实现）。
  * 分子式候选：CHNOPS 组合枚举 + 质量精度 ppm + RDBE 与七黄金规则过滤
    （Kind & Fiehn 2007 "Seven Golden Rules"；内置实现）。
  * MS2 库匹配：余弦相似度 + 前体 m/z 容差；**优先用 matchms 官方实现**
    （matchms.importing.load_from_msp / load_from_mgf 解析谱库，
    matchms.similarity.CosineGreedy 计分，matchms.filtering.default_filters 清洗峰），
    matchms 不可用时回退本模块内置余弦（Stein 1994 语义）。method 字段写明走哪条路。
  * 分子网络：余弦相似度网络 + 连通分量（GNPS 分子网络思路，Wang 2016）；
    余弦同样优先走 matchms CosineGreedy 矩阵，不可用时回退内置实现。
"""

from __future__ import annotations

import math
import os
import threading
import warnings

import numpy as np
import pandas as pd

import mb_common as sc

_POS = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+"]
_NEG = ["[M-H]-", "[M+Cl]-", "[M+HCOO]-"]


def _p(params, key, default=None):
    value = (params or {}).get(key, default)
    return default if value in (None, "", "null") else value


def _p_float(params, key, default):
    try:
        return float(_p(params, key, default))
    except (TypeError, ValueError):
        return float(default)


def _p_int(params, key, default):
    try:
        return int(float(_p(params, key, default)))
    except (TypeError, ValueError):
        return int(default)


def _outdir(ctx, outdir) -> str:
    return sc.ensure_dir(outdir or ctx.get("outdir") or ".")


def _features(ctx) -> list[str]:
    return [str(item) for item in (ctx.get("features") or [])]


def _info_frame(ctx) -> pd.DataFrame:
    frame = ctx.get("feature_info")
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(index=pd.Index(_features(ctx), name="feature"))


def _mz_rt(ctx) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = _features(ctx)
    frame = _info_frame(ctx)
    n = len(names)
    mz = np.full(n, np.nan)
    rt = np.full(n, np.nan)
    if "mz" in frame.columns:
        values = pd.to_numeric(frame["mz"], errors="coerce").to_numpy(dtype=float)
        if values.size == n:
            mz = values
    if "rt" in frame.columns:
        values = pd.to_numeric(frame["rt"], errors="coerce").to_numpy(dtype=float)
        if values.size == n:
            rt = values
    if not np.isfinite(mz).any():
        raise sc.prerequisite("特征表里没有 m/z 列（需要 mz/rt 列）才能做加合物/分子式推断")
    return mz, rt, names


def _polarity(ctx) -> str:
    frame = _info_frame(ctx)
    if "polarity" in frame.columns:
        values = [str(v).lower() for v in frame["polarity"].dropna().tolist()]
        if values:
            pos = sum(1 for v in values if v.startswith("p") or "+" in v)
            neg = sum(1 for v in values if v.startswith("n") or "-" in v)
            return "pos" if pos >= neg else "neg"
    overall = str((ctx.get("polarity") if isinstance(ctx, dict) else "") or "").lower()
    if "neg" in overall or overall.startswith("n"):
        return "neg"
    return "pos"


def _spectra(ctx, params) -> list[dict]:
    """取 MS2 谱库：ctx["library"]（dataset.library）> params["library"] 路径。"""
    library = ctx.get("library")
    if isinstance(library, dict):
        spectra = library.get("spectra")
        if isinstance(spectra, list) and spectra:
            return spectra
    path = str(_p(params, "library", "") or "")
    if path:
        return _parse_spectra_file(path)
    raise sc.prerequisite("还没有 MS2 谱库：请导入 msp/mgf（见「代谢物鉴定」页），或在参数里给 library 路径")


def _parse_spectra_file(path: str) -> list[dict]:
    """msp/mgf 谱库 → 统一 spectrum dict。

    优先用 matchms 官方解析器（load_from_msp / load_from_mgf，含元数据调和），
    解析不出东西或 matchms 不可用时回退本模块内置解析。
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise sc.prerequisite(f"谱库文件不存在：{path}")
    state = _matchms_or_none()
    if state:
        spectra = _parse_spectra_matchms(state, path)
        if spectra:
            return spectra
    return _parse_spectra_builtin(path)


def _parse_spectra_matchms(state: dict, path: str) -> list[dict]:
    """matchms.importing.load_from_msp / load_from_mgf → 统一 spectrum dict（失败返回 []）。"""
    lowered = path.lower()
    if lowered.endswith(".msp"):
        loader = state.get("load_from_msp")
    elif lowered.endswith(".mgf"):
        loader = state.get("load_from_mgf")
    else:
        loader = None
    if loader is None:
        return []
    spectra: list[dict] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index, spectrum in enumerate(loader(path)):
                arrays = (np.asarray(spectrum.peaks.mz, dtype=float),
                          np.asarray(spectrum.peaks.intensities, dtype=float))
                if arrays[0].size == 0:
                    continue
                metadata = dict(getattr(spectrum, "metadata", {}) or {})
                precursor = metadata.get("precursor_mz")
                try:
                    precursor = float(precursor) if precursor is not None else None
                except (TypeError, ValueError):
                    precursor = None
                spectra.append({
                    "id": str(metadata.get("compound_name") or metadata.get("name")
                              or f"ms{index + 1}"),
                    "precursor_mz": precursor,
                    "peaks": [[float(mz), float(intensity)]
                              for mz, intensity in zip(arrays[0], arrays[1])],
                    "formula": metadata.get("formula"),
                    "inchikey": metadata.get("inchikey"),
                    "adduct": metadata.get("adduct") or metadata.get("ionmode"),
                })
    except Exception:  # noqa: BLE001 - 解析失败走内置解析
        return []
    return spectra


def _parse_spectra_builtin(path: str) -> list[dict]:
    """极简 msp/mgf 解析（matchms 不可用时的回退）。"""
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    spectra: list[dict] = []
    if path.lower().endswith(".mgf"):
        blocks = text.split("BEGIN IONS")
        for block in blocks[1:]:
            precursor = None
            peaks = []
            name = None
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith("END IONS"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().upper()
                    if key in ("PEPMASS", "PEPMASS "):
                        try:
                            precursor = float(value.split()[0])
                        except (TypeError, ValueError):
                            precursor = None
                    elif key in ("TITLE", "NAME"):
                        name = value.strip()
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append([float(parts[0]), float(parts[1])])
                    except (TypeError, ValueError):
                        continue
            if peaks:
                spectra.append({"id": name or f"mgf{len(spectra) + 1}",
                                "precursor_mz": precursor, "peaks": peaks})
        return spectra
    # msp：空行分段
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        meta: dict = {}
        peaks = []
        in_peaks = False
        for line in block.splitlines():
            line = line.strip()
            if not line:
                in_peaks = True
                continue
            if not in_peaks and ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip().lower()] = value.strip()
                continue
            if ":" in line and not in_peaks:
                continue
            parts = line.replace("\t", " ").split()
            if len(parts) >= 2:
                try:
                    peaks.append([float(parts[0]), float(parts[1])])
                    in_peaks = True
                except (TypeError, ValueError):
                    continue
        precursor = None
        for key in ("precursormz", "precursor_mz", "parent", "pepmass", "precursor"):
            if key in meta:
                try:
                    precursor = float(str(meta[key]).split()[0])
                    break
                except (TypeError, ValueError):
                    continue
        if peaks:
            spectra.append({"id": meta.get("name") or meta.get("title") or f"msp{len(spectra) + 1}",
                            "precursor_mz": precursor, "peaks": peaks,
                            "formula": meta.get("formula"), "inchikey": meta.get("inchikey"),
                            "adduct": meta.get("adduct") or meta.get("ion_mode")})
    return spectra


def _norm_peaks(peaks, int_precision: int = 3) -> dict:
    """峰表 → {mz 四舍五入: 相对强度}，最高峰归一化。"""
    out: dict = {}
    for peak in peaks or []:
        try:
            mz, intensity = float(peak[0]), float(peak[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(mz) or not np.isfinite(intensity) or intensity <= 0:
            continue
        out[round(mz, int_precision)] = max(out.get(round(mz, int_precision), 0.0), intensity)
    if not out:
        return {}
    top = max(out.values())
    return {mz: value / top for mz, value in out.items()}


def _cosine(a: dict, b: dict, tolerance: float = 0.02) -> float:
    """余弦相似度（按 m/z 容差匹配峰；Stein 1994）。"""
    if not a or not b:
        return 0.0
    keys_b = sorted(b)
    num = 0.0
    for mz, value in a.items():
        best_key = None
        best_diff = None
        for key in keys_b:
            diff = abs(key - mz)
            if best_key is None or diff < best_diff:
                best_key, best_diff = key, diff
        if best_key is not None and best_diff is not None and best_diff <= tolerance:
            num += value * b[best_key]
    denom = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return float(num / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# matchms 官方实现（MS2 相似度 / 谱库解析的优先路径）
# ---------------------------------------------------------------------------

MS2_TOLERANCE = 0.02       # CosineGreedy 峰匹配容差（Da，默认；可用 params.tolerance 覆盖）

_MATCHMS_LOCK = threading.Lock()
_MATCHMS_STATE: dict = {
    "checked": False,
    "ok": False,
    "version": None,
    "error": None,
    "spectrum": None,
    "cosine_greedy": None,
    "load_from_msp": None,
    "load_from_mgf": None,
    "default_filters": None,
}


def matchms_status() -> dict:
    """惰性探测 matchms（缺包/版本不兼容只记录原因，不抛异常）。"""
    with _MATCHMS_LOCK:
        if _MATCHMS_STATE.get("checked"):
            return dict(_MATCHMS_STATE)
        state: dict = {"checked": True, "ok": False, "version": None, "error": None,
                       "spectrum": None, "cosine_greedy": None, "load_from_msp": None,
                       "load_from_mgf": None, "default_filters": None}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import matchms  # noqa: PLC0415 - 惰性 import：只在 MS2 分析时需要
                from matchms import Spectrum  # noqa: PLC0415
                from matchms.filtering import default_filters  # noqa: PLC0415
                from matchms.importing import load_from_mgf, load_from_msp  # noqa: PLC0415
                from matchms.similarity import CosineGreedy  # noqa: PLC0415
            state.update({
                "ok": True,
                "version": str(getattr(matchms, "__version__", "") or ""),
                "spectrum": Spectrum,
                "cosine_greedy": CosineGreedy,
                "load_from_msp": load_from_msp,
                "load_from_mgf": load_from_mgf,
                "default_filters": default_filters,
            })
        except Exception as error:  # noqa: BLE001 - 缺包/ABI 不兼容都走内置回退
            state["error"] = f"{type(error).__name__}: {error}"
        _MATCHMS_STATE.update(state)
        return dict(_MATCHMS_STATE)


def _matchms_or_none() -> dict | None:
    state = matchms_status()
    return state if state.get("ok") else None


def _peaks_arrays(peaks):
    """峰表 → (mz 升序数组, 强度数组)；无效峰丢弃。"""
    mz: list[float] = []
    intensity: list[float] = []
    for peak in peaks or []:
        try:
            value_mz, value_intensity = float(peak[0]), float(peak[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(value_mz) or not np.isfinite(value_intensity) or value_intensity <= 0:
            continue
        mz.append(value_mz)
        intensity.append(value_intensity)
    if not mz:
        return None
    order = np.argsort(np.asarray(mz, dtype=float))
    return np.asarray(mz, dtype=float)[order], np.asarray(intensity, dtype=float)[order]


def _matchms_spectrum(state: dict, peaks, precursor_mz=None, name=None):
    """峰表 → matchms Spectrum（并过一遍 matchms.filtering.default_filters）。"""
    arrays = _peaks_arrays(peaks)
    if arrays is None:
        return None
    mz, intensity = arrays
    metadata: dict = {}
    try:
        if precursor_mz is not None and np.isfinite(float(precursor_mz)):
            metadata["precursor_mz"] = float(precursor_mz)
    except (TypeError, ValueError):
        pass
    if name:
        metadata["compound_name"] = str(name)
    try:
        spectrum = state["spectrum"](mz=mz, intensities=intensity, metadata=metadata)
    except Exception:  # noqa: BLE001 - 单张谱失败不影响其它谱
        return None
    try:
        filtered = state["default_filters"](spectrum)
        if filtered is not None and len(filtered.peaks.mz):
            spectrum = filtered
    except Exception:  # noqa: BLE001 - 清洗失败就用原谱
        pass
    return spectrum


def _matchms_greedy(state: dict, tolerance: float):
    """CosineGreedy 实例（不裁噪、不去前体：峰匹配只由 tolerance 决定，便于与内置实现对照）。"""
    try:
        return state["cosine_greedy"](tolerance=float(tolerance), noise_cutoff=0.0,
                                      remove_precursor=False, mz_power=0.0,
                                      intensity_power=1.0)
    except TypeError:  # 老版本 matchms 没有这些关键字
        return state["cosine_greedy"](tolerance=float(tolerance))


def _matchms_score(state: dict, greedy, spectrum_a, spectrum_b) -> float:
    """两张 matchms Spectrum 的 CosineGreedy 得分（0..1）。"""
    if spectrum_a is None or spectrum_b is None:
        return 0.0
    try:
        value = greedy.pair(spectrum_a, spectrum_b)
    except Exception:  # noqa: BLE001
        return 0.0
    score = 0.0
    try:
        score = float(np.asarray(value)["score"])
    except Exception:  # noqa: BLE001
        try:
            score = float(value[0])
        except Exception:  # noqa: BLE001
            score = 0.0
    return float(score) if np.isfinite(score) else 0.0


def _matchms_matrix(state: dict, greedy, spectra: list) -> np.ndarray:
    """余弦矩阵（CosineGreedy.matrix，失败则逐对 pair 回退）。"""
    n = len(spectra)
    out = np.zeros((n, n), dtype=float)
    if n == 0:
        return out
    try:
        scores = greedy.matrix(spectra, progress_bar=False)
        array = scores.to_array()
        values = np.asarray(array["score"], dtype=float)
        if values.shape == (n, n):
            return np.nan_to_num(values, nan=0.0)
    except Exception:  # noqa: BLE001 - 矩阵接口失败就逐对算
        pass
    for i in range(n):
        for j in range(i + 1, n):
            value = _matchms_score(state, greedy, spectra[i], spectra[j])
            out[i, j] = value
            out[j, i] = value
    return out


def _feature_spectra_map(names: list[str], spectra: list[dict]) -> dict:
    """特征自己的实验 MS2 谱：谱图带 feature 关联字段，或 id/name 与特征同名。

    demo 谱库里与特征 m/z 精确对应的那批谱就带 `feature` 字段——它们既是库谱，也是
    该特征的实验谱，所以自匹配得分应 ≈1（DESIGN §9 验收点）。
    """
    name_set = {str(name) for name in names}
    mapping: dict = {}
    for spectrum in spectra:
        if not spectrum.get("peaks"):
            continue
        reference = ""
        for key in ("feature", "feature_id", "featureid"):
            value = spectrum.get(key)
            if value:
                reference = str(value)
                break
        if not reference:
            for key in ("id", "name"):
                value = spectrum.get(key)
                if value is not None and str(value) in name_set:
                    reference = str(value)
                    break
        if reference in name_set:
            mapping[reference] = spectrum
    return mapping


def _method_ms2(state: dict | None, kind: str, ppm=None, tolerance=None) -> str:
    """MS2 分析的真实实现字符串（供 summary.method 使用）。"""
    if state:
        version = str(state.get("version") or "").strip()
        version_txt = f"matchms {version}" if version else "matchms"
        if kind == "library":
            return (f"matchms.similarity.CosineGreedy（前体 ±{ppm} ppm，"
                    f"峰容差 {tolerance} Da；{version_txt}）")
        return (f"matchms.similarity.CosineGreedy 余弦矩阵（峰容差 {tolerance} Da）"
                f"+ 连通分量（GNPS 思路；{version_txt}）")
    reason = matchms_status().get("error") or "未检测到 matchms"
    if kind == "library":
        return f"内置余弦实现（matchms 不可用：{reason}）"
    return f"内置余弦矩阵 + 连通分量（matchms 不可用：{reason}）"


def _finish(payload: dict, method: str) -> dict:
    """把真实实现写进结果信封（summary.method）。"""
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary["method"] = str(method)
    return payload


# ---------------------------------------------------------------------------
# 加合物 / 分子式
# ---------------------------------------------------------------------------


def _adduct_table(polarity: str) -> list[str]:
    return list(_POS if polarity == "pos" else _NEG)


def _pair_adducts(mz: np.ndarray, rt: np.ndarray, polarity: str, ppm: float) -> list[dict]:
    """在特征间搜索加合物对：同一中性质量 + 不同加合物应差固定质量。"""
    adducts = _adduct_table(polarity)
    pairs: list[dict] = []
    index = np.where(np.isfinite(mz))[0]
    # 性能护栏：特征太多时只在前 3000 个（按 m/z 排序）里找加合物对
    if index.size > 3000:
        index = index[np.argsort(mz[index])[:3000]]
    for a_index in index:
        for b_index in index:
            if b_index <= a_index:
                continue
            if np.isfinite(rt[a_index]) and np.isfinite(rt[b_index]) and abs(rt[a_index] - rt[b_index]) > 0.6:
                continue
            for add_a in adducts:
                mass_a = float(mz[a_index]) * mb_charge(add_a) - mb_shift(add_a)
                if not np.isfinite(mass_a) or mass_a <= 0:
                    continue
                for add_b in adducts:
                    if add_a == add_b:
                        continue
                    mass_b = float(mz[b_index]) * mb_charge(add_b) - mb_shift(add_b)
                    if not np.isfinite(mass_b) or mass_b <= 0:
                        continue
                    err = abs(mass_a - mass_b) / max(abs(mass_a), 1e-9) * 1e6
                    if 0 <= err <= ppm:
                        pairs.append({
                            "feature_a": int(a_index + 1), "adduct_a": add_a,
                            "feature_b": int(b_index + 1), "adduct_b": add_b,
                            "neutral_mass": float((mass_a + mass_b) / 2),
                            "ppm": float(err),
                            "rt_a": float(rt[a_index]) if np.isfinite(rt[a_index]) else None,
                            "rt_b": float(rt[b_index]) if np.isfinite(rt[b_index]) else None,
                        })
                        break
    # 同一对特征只保留一个最佳加合物组合
    best: dict = {}
    for item in pairs:
        key = (item["feature_a"], item["feature_b"])
        if key not in best or item["ppm"] < best[key]["ppm"]:
            best[key] = item
    return sorted(best.values(), key=lambda item: item["ppm"])


def mb_shift(adduct: str) -> float:
    return float(sc.ADDUCTS.get(adduct, (sc.PROTON, 1))[0])


def mb_charge(adduct: str) -> int:
    return int(sc.ADDUCTS.get(adduct, (sc.PROTON, 1))[1])


def _formula_candidates(neutral_mass: float, ppm: float, max_c: int = 50, hetero: int = 6,
                        max_results: int = 20) -> list[dict]:
    """CHNOPS 组合枚举 + RDBE/七黄金规则过滤（Kind & Fiehn 2007）。"""
    mass = {el: m for el, m in sc.ELEMENT_MASS.items()}
    tol = abs(neutral_mass) * ppm * 1e-6
    out: list[dict] = []
    c_max = max(1, min(int(max_c), int(neutral_mass / mass["C"]) + 1))
    for c in range(1, c_max + 1):
        rem_c = neutral_mass - c * mass["C"]
        if rem_c < -tol:
            continue
        n_max = int(rem_c / mass["N"]) + 1 if rem_c > 0 else 0
        for n in range(0, min(n_max, hetero) + 1):
            rem_n = rem_c - n * mass["N"]
            if rem_n < -tol:
                continue
            o_max = int(rem_n / mass["O"]) + 1 if rem_n > 0 else 0
            for o in range(0, min(o_max, hetero) + 1):
                rem_o = rem_n - o * mass["O"]
                if rem_o < -tol:
                    continue
                p_max = int(rem_o / mass["P"]) + 1 if rem_o > 0 else 0
                for p in range(0, min(p_max, 2) + 1):
                    rem_p = rem_o - p * mass["P"]
                    if rem_p < -tol:
                        continue
                    s_max = int(rem_p / mass["S"]) + 1 if rem_p > 0 else 0
                    for s in range(0, min(s_max, 2) + 1):
                        rem_s = rem_p - s * mass["S"]
                        if rem_s < -tol:
                            continue
                        h = int(round(rem_s / mass["H"]))
                        if h < 0 or h > 2 * c + n + 2:
                            continue
                        counts = {"C": c, "H": h, "N": n, "O": o}
                        if p:
                            counts["P"] = p
                        if s:
                            counts["S"] = s
                        formula_mass = sc.formula_mass(counts)
                        err = (formula_mass - neutral_mass) / neutral_mass * 1e6
                        if abs(err) > ppm:
                            continue
                        rdbe = sc.rdbe(counts)
                        if rdbe < -0.5 or rdbe > 40:
                            continue
                        # 七黄金规则（简化）：H/C、N/C、杂原子/C 的合理范围
                        hc = h / c
                        if not (0.1 <= hc <= 3.5):
                            continue
                        if n > 0 and n / c > 1.3:
                            continue
                        hetero_total = n + o + p + s
                        if hetero_total / c > 1.5:
                            continue
                        if h % 2 == 1 and n % 2 == 1:
                            continue  # 氮规则
                        out.append({"formula": sc.formula_str(counts), "mass": formula_mass,
                                    "ppm": err, "rdbe": rdbe, "C": c, "H": h, "N": n, "O": o,
                                    "S": p * 0 + s, "P": p})
                        if len(out) >= max_results * 30:
                            break
                    if len(out) >= max_results * 30:
                        break
                if len(out) >= max_results * 30:
                    break
            if len(out) >= max_results * 30:
                break
        if len(out) >= max_results * 30:
            break
    out.sort(key=lambda item: abs(item["ppm"]))
    return out[:max_results]


# ---------------------------------------------------------------------------
# STEP
# ---------------------------------------------------------------------------


def step_annotate(ctx, params, outdir):
    mz, rt, names = _mz_rt(ctx)
    polarity = str(_p(params, "polarity", "") or _polarity(ctx)).lower()
    polarity = "neg" if polarity.startswith("n") or "neg" in polarity else "pos"
    ppm = _p_float(params, "ppm", 10.0)
    want_formula = str(_p(params, "formula", True)).lower() not in ("false", "0", "no")
    library_path = str(_p(params, "library", "") or "")
    outdir = _outdir(ctx, outdir)

    adducts = _adduct_table(polarity)
    frame = _info_frame(ctx)
    rows = []
    for index, name in enumerate(names):
        if not np.isfinite(mz[index]):
            continue
        neutral = mz[index] * mb_charge(adducts[0]) - mb_shift(adducts[0])
        rows.append({
            "feature": name,
            "mz": float(mz[index]),
            "rt": float(rt[index]) if np.isfinite(rt[index]) else None,
            "adduct_default": adducts[0],
            "neutral_mass": float(neutral),
            "name": frame["name"].iloc[index] if "name" in frame.columns and index < len(frame) else None,
        })
    table_rows = rows[:500]
    figures = []
    fig, ax = sc.new_fig(6.4, 4.2)
    finite_mz = mz[np.isfinite(mz) & (mz > 0)]
    ax.hist(finite_mz, bins=48, color=sc.PALETTE[0])
    ax.set_xlabel("m/z", fontsize=9)
    ax.set_ylabel("features", fontsize=9)
    ax.set_title("Feature m/z distribution", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figures.append(sc.save_fig(fig, outdir, "mz-distribution", title=None))
    method = "特征中性质量推断（按默认加合物的质量差；内置实现）"
    bullets = [f"实现：{method}",
               f"{len(rows)} 个带 m/z 的特征（极性 {polarity}），按 {adducts[0]} 反推中性质量",
               "下一步可跑「加合物推断」「分子式候选」或导入 MS2 库做匹配"]
    result = sc.result(
        summary={"title": "代谢物鉴定（准备）", "bullets": bullets,
                 "metrics": {"n_features": len(rows), "polarity": polarity, "ppm": ppm,
                             "implementation": method}},
        tables=[sc.table("annotate", "特征的中性质量推断",
                         columns=[{"key": "feature", "label": "特征", "type": "string"},
                                  {"key": "mz", "label": "m/z", "type": "number"},
                                  {"key": "rt", "label": "RT", "type": "number"},
                                  {"key": "adduct_default", "label": "加合物", "type": "string"},
                                  {"key": "neutral_mass", "label": "中性质量", "type": "number"}],
                         rows=table_rows,
                         note=f"按 {adducts[0]} 反推；多加合物需跑「加合物推断」")],
        figures=figures,
        store={"annotation": {"polarity": polarity, "ppm": ppm, "n_features": len(rows),
                              "implementation": method}},
    )
    return _finish(result, method)


# ---------------------------------------------------------------------------
# ANALYSES
# ---------------------------------------------------------------------------


def analysis_adduct_annot(ctx, params, outdir):
    mz, rt, names = _mz_rt(ctx)
    polarity = "neg" if str(_p(params, "polarity", _polarity(ctx))).lower().startswith("n") else "pos"
    ppm = _p_float(params, "ppm", 10.0)
    pairs = _pair_adducts(mz, rt, polarity, ppm)
    outdir = _outdir(ctx, outdir)
    bullets = [f"在 {int(np.isfinite(mz).sum())} 个特征里找到 {len(pairs)} 对加合物关系（ppm≤{ppm}，极性 {polarity}）"]
    if not pairs:
        bullets.append("没有配对：可能每个代谢物只有一个加合物，或 ppm 太严（可放宽到 20）")
    else:
        bullets.append("前 3 对：" + "；".join(
            f"#{p['feature_a']}({p['adduct_a']}) ↔ #{p['feature_b']}({p['adduct_b']}) {p['ppm']:.1f}ppm"
            for p in pairs[:3]))
    figures = []
    if pairs:
        fig, ax = sc.new_fig(6.4, 4.2)
        ax.hist([p["ppm"] for p in pairs], bins=30, color=sc.PALETTE[2])
        ax.set_xlabel("mass error (ppm)", fontsize=9)
        ax.set_ylabel("adduct pairs", fontsize=9)
        ax.set_title("Adduct pairing mass accuracy", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        figures.append(sc.save_fig(fig, outdir, "adduct-pairs", title=None))
    method = "特征间加合物质量差配对（MS-DIAL / CAMERA 思路；内置实现）"
    bullets = [f"实现：{method}"] + bullets
    payload = sc.result(
        summary={"title": "加合物推断", "bullets": bullets,
                 "metrics": {"n_pairs": len(pairs), "ppm": ppm, "polarity": polarity,
                             "implementation": method}},
        tables=[sc.table("adducts", "加合物配对（同一中性质量）",
                         columns=[{"key": "feature_a", "label": "特征 A", "type": "number"},
                                  {"key": "adduct_a", "label": "加合物 A", "type": "string"},
                                  {"key": "feature_b", "label": "特征 B", "type": "number"},
                                  {"key": "adduct_b", "label": "加合物 B", "type": "string"},
                                  {"key": "neutral_mass", "label": "中性质量", "type": "number"},
                                  {"key": "ppm", "label": "误差 ppm", "type": "number"},
                                  {"key": "rt_a", "label": "RT A", "type": "number"},
                                  {"key": "rt_b", "label": "RT B", "type": "number"}],
                         rows=pairs)],
        figures=figures,
        store={"adducts": {"n_pairs": len(pairs), "pairs": pairs[:2000],
                           "implementation": method}},
    )
    return _finish(payload, method)


def analysis_formula_candidates(ctx, params, outdir):
    mz, rt, names = _mz_rt(ctx)
    polarity = "neg" if str(_p(params, "polarity", _polarity(ctx))).lower().startswith("n") else "pos"
    ppm = _p_float(params, "ppm", 5.0)
    max_c = _p_int(params, "max_c", 50)
    hetero = _p_int(params, "hetero_atoms", 6)
    feature = str(_p(params, "feature", "") or "")
    index_map = {name: i for i, name in enumerate(names)}
    if feature:
        if feature not in index_map:
            raise sc.prerequisite(f"特征「{feature}」不在数据集里（用 feature 名称，如 mz203.0526_rt1.23）")
        targets = [index_map[feature]]
    else:
        # 默认取 m/z 最小的 10 个特征做演示
        finite = np.where(np.isfinite(mz))[0]
        if finite.size == 0:
            raise sc.prerequisite("没有 m/z")
        targets = list(finite[np.argsort(mz[finite])[:10]])
    rows = []
    for target in targets:
        adduct = _POS[0] if polarity == "pos" else _NEG[0]
        neutral = mz[target] * mb_charge(adduct) - mb_shift(adduct)
        for cand in _formula_candidates(neutral, ppm, max_c=max_c, hetero=hetero):
            rows.append({"feature": names[target], "mz": float(mz[target]),
                         "neutral_mass": float(neutral), "adduct": adduct, **cand})
    if not rows:
        raise sc.prerequisite(f"ppm≤{ppm} 内没有找到合法 CHNOPS 分子式：可放宽 ppm 或提高 max_c")
    frame = pd.DataFrame(rows)
    outdir = _outdir(ctx, outdir)
    fig, ax = sc.new_fig(6.4, 4.2)
    ax.hist(frame["ppm"], bins=30, color=sc.PALETTE[1])
    ax.set_xlabel("mass error (ppm)", fontsize=9)
    ax.set_ylabel("formula candidates", fontsize=9)
    ax.set_title("Formula candidate mass accuracy", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure = sc.save_fig(fig, outdir, "formula-accuracy", title=None)
    bullets = [f"{len(targets)} 个特征共 {len(frame)} 个候选分子式（ppm≤{ppm}，RDBE 与七黄金规则过滤）",
               "误差最小的候选：" + "、".join(f"{row['formula']}({row['ppm']:+.2f}ppm)" for row in rows[:3])]
    method = "CHNOPS 枚举 + RDBE / 七黄金规则（Kind & Fiehn 2007；内置实现）"
    bullets = [f"实现：{method}"] + bullets
    payload = sc.result(
        summary={"title": "分子式候选", "bullets": bullets,
                 "metrics": {"n_features": len(targets), "n_candidates": len(rows), "ppm": ppm,
                             "implementation": method}},
        tables=[sc.table("formulas", "分子式候选",
                         columns=[{"key": "feature", "label": "特征", "type": "string"},
                                  {"key": "neutral_mass", "label": "中性质量", "type": "number"},
                                  {"key": "formula", "label": "分子式", "type": "string"},
                                  {"key": "mass", "label": "理论质量", "type": "number"},
                                  {"key": "ppm", "label": "误差 ppm", "type": "number"},
                                  {"key": "rdbe", "label": "RDBE", "type": "number"}],
                         rows=rows)],
        figures=[figure],
        store={"formulas": frame.head(2000)},
    )
    return _finish(payload, method)


def analysis_ms2_library_match(ctx, params, outdir):
    """MS2 库匹配：前体 m/z 容差筛候选 + 余弦打分。

    余弦优先用 matchms 官方实现（CosineGreedy）；库谱解析优先用 matchms
    load_from_msp/load_from_mgf。特征自己的实验 MS2 谱来自 library 里带 feature 关联的谱，
    没有实验谱时只给前体候选（score=0 表示算不了相似度，不伪造分数）。
    """
    mz, rt, names = _mz_rt(ctx)
    spectra = _spectra(ctx, params)
    ppm = _p_float(params, "ppm", 20.0)
    min_score = _p_float(params, "min_score", 0.5)
    top_n = _p_int(params, "top_n", 5)
    tolerance = _p_float(params, "tolerance", MS2_TOLERANCE)
    state = _matchms_or_none()
    greedy = _matchms_greedy(state, tolerance) if state else None
    method = _method_ms2(state, "library", ppm=ppm, tolerance=tolerance)

    queries = _feature_spectra_map(names, spectra)
    query_spectra = {}
    if state:
        for feature, spectrum in queries.items():
            query_spectra[feature] = _matchms_spectrum(
                state, spectrum.get("peaks"), spectrum.get("precursor_mz"), spectrum.get("id"))

    rows: list[dict] = []
    n_scored = 0
    for spectrum in spectra:
        peaks = spectrum.get("peaks")
        if not peaks:
            continue
        precursor = spectrum.get("precursor_mz")
        if precursor is None or not np.isfinite(float(precursor)):
            continue
        library_spectrum = _matchms_spectrum(state, peaks, precursor,
                                             spectrum.get("id")) if state else None
        library_peaks = _norm_peaks(peaks) if not state else None
        err = np.abs(mz - float(precursor)) / max(float(precursor), 1e-9) * 1e6
        for index in np.where(err <= ppm)[0]:
            score = 0.0
            reference = queries.get(names[index])
            if reference is not None:
                if state and library_spectrum is not None and query_spectra.get(names[index]) is not None:
                    score = _matchms_score(state, greedy,
                                           query_spectra[names[index]], library_spectrum)
                    n_scored += 1
                elif library_peaks:
                    score = _cosine(_norm_peaks(reference.get("peaks")), library_peaks)
                    n_scored += 1
            rows.append({"library_id": spectrum.get("id"), "feature": names[index],
                         "precursor_mz": float(precursor), "feature_mz": float(mz[index]),
                         "ppm": float(err[index]), "score": score,
                         "library_formula": spectrum.get("formula")})
    if not rows:
        raise sc.prerequisite(
            f"没有特征的前体 m/z 落在库谱前体 ±{ppm}ppm 内：检查极性/加合物，或认为库里没有匹配（也可放宽 ppm）")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["score", "ppm"], ascending=[False, True])
    matched = frame[frame["score"] >= min_score]
    frame = frame.head(max(50, top_n * 30))
    outdir = _outdir(ctx, outdir)
    fig, ax = sc.new_fig(6.4, 4.2)
    ax.hist(frame["score"], bins=30, color=sc.PALETTE[3])
    ax.set_xlabel("cosine score", fontsize=9)
    ax.set_ylabel("candidates", fontsize=9)
    ax.set_title("MS2 library matching scores", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure = sc.save_fig(fig, outdir, "ms2-match-scores", title=None)
    bullets = [f"实现：{method}",
               f"库谱 {len(spectra)} 张 vs 特征 {len(names)} 个：{len(frame)} 条候选（±{ppm}ppm）",
               f"余弦 ≥{min_score} 的匹配：{len(matched)} 条（实际算了 {n_scored} 对余弦）"]
    if len(queries):
        top_score = float(frame["score"].max()) if len(frame) else 0.0
        bullets.append(f"其中 {len(queries)} 个特征自带实验 MS2 谱；最高余弦 {top_score:.3f}"
                       f"（同一张谱自匹配≈1，是链路自检点）")
    else:
        bullets.append("特征没有实验 MS2 谱：本次只按前体质量给出候选（score=0 表示无法算相似度）")
    payload = sc.result(
        summary={"title": "MS2 库匹配", "bullets": bullets,
                 "metrics": {"n_spectra": len(spectra), "n_candidates": len(frame),
                             "n_matched": int(len(matched)), "n_scored": n_scored,
                             "n_query_spectra": len(queries),
                             "implementation": method}},
        tables=[sc.table("ms2", "MS2 库匹配候选",
                         columns=[{"key": "feature", "label": "特征", "type": "string"},
                                  {"key": "library_id", "label": "库谱", "type": "string"},
                                  {"key": "feature_mz", "label": "特征 m/z", "type": "number"},
                                  {"key": "precursor_mz", "label": "库前体 m/z", "type": "number"},
                                  {"key": "ppm", "label": "误差 ppm", "type": "number"},
                                  {"key": "score", "label": "余弦", "type": "number"},
                                  {"key": "library_formula", "label": "库分子式", "type": "string"}],
                         rows=frame.to_dict(orient="records"))],
        figures=[figure],
        store={"ms2_match": frame.head(2000)},
    )
    return _finish(payload, method)


def analysis_ms2_network(ctx, params, outdir):
    """分子网络：余弦矩阵 + 阈值连边 + 连通分量（GNPS 思路，Wang 2016）。

    余弦矩阵优先由 matchms.similarity.CosineGreedy 计算（matchms 不可用时回退内置余弦）。
    """
    spectra = _spectra(ctx, params)
    min_score = _p_float(params, "min_score", 0.6)
    max_nodes = _p_int(params, "max_nodes", 200)
    tolerance = _p_float(params, "tolerance", MS2_TOLERANCE)
    if len(spectra) < 3:
        raise sc.prerequisite("谱库少于 3 张谱，无法构建分子网络")
    spectra = spectra[:max_nodes]
    state = _matchms_or_none()
    greedy = _matchms_greedy(state, tolerance) if state else None
    method = _method_ms2(state, "network", tolerance=tolerance)

    normalised = [(str(s.get("id")), _norm_peaks(s.get("peaks")), s.get("precursor_mz"), s)
                  for s in spectra]
    normalised = [item for item in normalised if item[1]]
    if len(normalised) < 3:
        raise sc.prerequisite("有效谱图少于 3 张")

    edges: list[tuple[int, int, float]] = []
    if state:
        matchms_spectra = [_matchms_spectrum(state, item[3].get("peaks"),
                                             item[3].get("precursor_mz"), item[0])
                           for item in normalised]
        valid = [index for index, spectrum in enumerate(matchms_spectra) if spectrum is not None]
        matrix = np.zeros((len(normalised), len(normalised)), dtype=float)
        if valid:
            sub = _matchms_matrix(state, greedy, [matchms_spectra[i] for i in valid])
            for i_pos, i in enumerate(valid):
                for j_pos, j in enumerate(valid):
                    matrix[i, j] = sub[i_pos, j_pos]
        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                if matrix[i, j] >= min_score:
                    edges.append((i, j, float(matrix[i, j])))
    else:
        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                score = _cosine(normalised[i][1], normalised[j][1], tolerance=tolerance)
                if score >= min_score:
                    edges.append((i, j, score))
    # 连通分量
    parent = list(range(len(normalised)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, _score in edges:
        union(i, j)
    components: dict = {}
    for index in range(len(normalised)):
        components.setdefault(find(index), []).append(index)
    outdir = _outdir(ctx, outdir)
    fig, ax = sc.new_fig(7.2, 6.0)
    try:
        import networkx as nx

        graph = nx.Graph()
        for index, (name, _peaks, precursor, _raw) in enumerate(normalised):
            graph.add_node(index, label=name, mz=precursor)
        for i, j, score in edges:
            graph.add_edge(i, j, weight=score)
        positions = nx.spring_layout(graph, seed=0, k=0.6) if graph.number_of_nodes() > 0 else {}
        for i, j, score in edges:
            if i in positions and j in positions:
                ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]],
                        color="#b8bcc4", linewidth=0.6 + 1.6 * score, zorder=1)
        colors = {}
        for root, members in components.items():
            color = sc.PALETTE[len(colors) % len(sc.PALETTE)]
            for member in members:
                colors[member] = color
        xs = [positions[i][0] for i in positions]
        ys = [positions[i][1] for i in positions]
        cs = [colors.get(i, "#8c8c8c") for i in positions]
        ax.scatter(xs, ys, s=42, c=cs, zorder=2, edgecolors="white", linewidths=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    except Exception:  # noqa: BLE001 - 没有 networkx 时退回余弦直方图
        ax.hist([e[2] for e in edges] or [0], bins=20, color=sc.PALETTE[0])
        ax.set_xlabel("cosine", fontsize=9); ax.set_ylabel("edges", fontsize=9)
    ax.set_title(f"MS2 molecular network (cosine >= {min_score})", fontsize=10)
    figure = sc.save_fig(fig, outdir, "ms2-network", title=None)
    rows = [{"spectrum_a": normalised[i][0], "spectrum_b": normalised[j][0], "cosine": float(score)}
            for i, j, score in edges]
    largest = max((len(v) for v in components.values()), default=0)
    mean_score = float(np.mean([e[2] for e in edges])) if edges else 0.0
    bullets = [f"实现：{method}",
               f"{len(normalised)} 张谱、{len(edges)} 条边（余弦≥{min_score}），"
               f"{len(components)} 个分子网络",
               f"最大网络含 {largest} 张谱" + (f"；连边平均余弦 {mean_score:.3f}" if edges else
                                               "；阈值内没有连边（可放宽 min_score 看相似度分布）")]
    payload = sc.result(
        summary={"title": "分子网络", "bullets": bullets,
                 "metrics": {"n_spectra": len(normalised), "n_edges": len(edges),
                             "n_components": len(components), "largest": int(largest),
                             "mean_edge_cosine": mean_score,
                             "tolerance": tolerance, "implementation": method}},
        tables=[sc.table("ms2_network", "谱间余弦边",
                         columns=[{"key": "spectrum_a", "label": "谱 A", "type": "string"},
                                  {"key": "spectrum_b", "label": "谱 B", "type": "string"},
                                  {"key": "cosine", "label": "余弦", "type": "number"}],
                         rows=rows)],
        figures=[figure],
        store={"ms2_network": {"edges": rows[:2000], "n_components": len(components),
                               "implementation": method}},
    )
    return _finish(payload, method)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


def plot_mass_accuracy(ctx, params=None, outdir=None):
    params = dict(params or {})
    mz, rt, names = _mz_rt(ctx)
    frame = _info_frame(ctx)
    outdir = _outdir(ctx, outdir)
    if "mz" in frame.columns and "formula" in frame.columns:
        rows = []
        for index, name in enumerate(names):
            formula = frame["formula"].iloc[index] if index < len(frame) else None
            value = frame["mz"].iloc[index] if index < len(frame) else None
            if isinstance(formula, str) and formula and np.isfinite(mz[index]):
                try:
                    theory = sc.formula_mass(formula)
                    adduct = _POS[0]
                    rows.append(sc.ppm_error(float(value), sc.adduct_mz(theory, adduct)))
                except Exception:  # noqa: BLE001
                    continue
        if rows:
            fig, ax = sc.new_fig(6.2, 4.2)
            ax.hist(rows, bins=40, color=sc.PALETTE[0])
            ax.axvline(0, color="#d1495b", linewidth=0.8)
            ax.set_xlabel("mass error (ppm)", fontsize=9)
            ax.set_ylabel("features", fontsize=9)
            ax.set_title("Mass accuracy (measured vs theoretical)", fontsize=10)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            return sc.save_fig(fig, outdir, "mass-accuracy", title=None)
    fig, ax = sc.new_fig(6.2, 4.2)
    ax.hist(mz[np.isfinite(mz)], bins=48, color=sc.PALETTE[0])
    ax.set_xlabel("m/z", fontsize=9); ax.set_ylabel("features", fontsize=9)
    ax.set_title("Feature m/z distribution", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return sc.save_fig(fig, outdir, "mass-accuracy", title=None)


def plot_mass_network(ctx, params=None, outdir=None):
    params = dict(params or {})
    out = analysis_ms2_network(ctx, params, _outdir(ctx, outdir))
    figures = out.get("figures") or []
    if figures:
        return figures[0]
    raise sc.prerequisite("分子网络没有生成图")


STEPS = [
    {
        "id": "annotate",
        "title": "代谢物鉴定",
        "icon": "annotate",
        "requires": ["transform"],
        "params": [
            {"key": "polarity", "label": "极性", "type": "select", "default": "pos",
             "options": [{"value": "pos", "label": "正离子"}, {"value": "neg", "label": "负离子"}]},
            {"key": "adduct", "label": "默认加合物", "type": "text", "default": "[M+H]+"},
            {"key": "ppm", "label": "质量容差 ppm", "type": "number", "default": 10},
            {"key": "formula", "label": "推断分子式", "type": "boolean", "default": True},
            {"key": "library", "label": "MS2 谱库路径（可空）", "type": "text", "default": ""},
        ],
        "run": step_annotate,
    },
]

ANALYSES = [
    {
        "id": "adduct_annot", "title": "加合物推断", "group": "代谢物鉴定",
        "method": "特征间加合物质量差配对（MS-DIAL / CAMERA 思路；内置实现）",
        "needs": ["mz"],
        "params": [
            {"key": "polarity", "label": "极性", "type": "select", "default": "pos",
             "options": [{"value": "pos", "label": "正离子"}, {"value": "neg", "label": "负离子"}]},
            {"key": "ppm", "label": "质量容差 ppm", "type": "number", "default": 10},
        ],
        "run": analysis_adduct_annot,
    },
    {
        "id": "formula_candidates", "title": "分子式候选", "group": "代谢物鉴定",
        "method": "CHNOPS 枚举 + RDBE/七黄金规则（Kind & Fiehn 2007；内置实现）",
        "needs": ["mz"],
        "params": [
            {"key": "feature", "label": "指定特征（可空=最小的 10 个）", "type": "text", "default": ""},
            {"key": "ppm", "label": "质量容差 ppm", "type": "number", "default": 5},
            {"key": "max_c", "label": "最大碳数", "type": "number", "default": 50},
            {"key": "hetero_atoms", "label": "杂原子上限", "type": "number", "default": 6},
        ],
        "run": analysis_formula_candidates,
    },
    {
        "id": "ms2_library_match", "title": "MS2 库匹配", "group": "代谢物鉴定",
        "method": "余弦相似度 + 前体容差（Stein 1994；matchms.similarity.CosineGreedy 官方实现，缺 matchms 时回退内置余弦）",
        "needs": ["ms2"],
        "params": [
            {"key": "ppm", "label": "前体容差 ppm", "type": "number", "default": 20},
            {"key": "min_score", "label": "最低余弦", "type": "number", "default": 0.5},
            {"key": "top_n", "label": "返回条数", "type": "number", "default": 5},
            {"key": "library", "label": "谱库路径（可空）", "type": "text", "default": ""},
        ],
        "run": analysis_ms2_library_match,
    },
    {
        "id": "ms2_network", "title": "分子网络", "group": "代谢物鉴定",
        "method": "余弦网络 + 连通分量（GNPS 分子网络思路，Wang 2016；matchms CosineGreedy 余弦矩阵，缺 matchms 时回退内置）",
        "needs": ["ms2"],
        "params": [
            {"key": "min_score", "label": "最低余弦", "type": "number", "default": 0.6},
            {"key": "max_nodes", "label": "最多谱图数", "type": "number", "default": 200},
            {"key": "tolerance", "label": "峰匹配容差 Da", "type": "number", "default": 0.02},
        ],
        "run": analysis_ms2_network,
    },
]

PLOTS = {
    "mass_accuracy": plot_mass_accuracy,
    "mass_network": plot_mass_network,
}

__all__ = ["STEPS", "ANALYSES", "PLOTS"]
