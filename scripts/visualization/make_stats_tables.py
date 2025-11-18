#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel cohort-level stats:
- C-index: 使用原始 predicted（不取反），配对bootstrap并行 + p-value vs Our
- AUROC: 统一使用 -predicted 计算（DeLong 95%CI + p-value vs Our）
- Cohorts:
    Internal: ["seer"]
    External: ["cg_3", "multicenter"]

Usage:
python make_stats_tables_parallel_cindex_raw_auc_flip.py --base_dir /path/to/csvs \
  --internal_out internal_stats.csv --external_out external_stats.csv \
  --n_boot 1000 --seed 20251012 --n_jobs 0
"""

import os
import math
import argparse
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

# ---------- optional scipy ----------
try:
    from scipy.stats import norm
    def norm_cdf(x: np.ndarray) -> np.ndarray:
        return norm.cdf(x)
except Exception:
    import numpy as _np
    def norm_cdf(x: _np.ndarray) -> _np.ndarray:
        return 0.5 * (1.0 + _np.erf(x / math.sqrt(2.0)))

# ---------- model maps ----------
MODELS_EXTERNAL = {
    "Our model": "df_ours",
    "TNM": "df_tnm",
    "BCLC": "df_bclc",
    "CNLC": "df_cnlc",
    "CLAUDE": "df_claude-3-5-sonnet-20241022",
    "DEEPSEEK-r1": "df_deepseek-r1",
    "GPT-4o": "df_gpt-4o-2024-08-06",
    "GEMINI-2.5-pro": "df_gemini-2_5-pro",
    "GPT-5": "df_gpt-5",
}
MODELS_INTERNAL = {
    "Our model": "df_ours",
    "TNM": "df_tnm",
    "BCLC": "df_bclc",
    "CNLC": "df_cnlc",
    "XGB": "df_xgb",
    "SVM": "df_svm",
    "Bayes": "df_bayes",
    "MLP": "df_mlp",
}
EXTERNAL_SUFFIXES = ["cg_3", "multicenter"]
INTERNAL_SUFFIXES = ["seer"]

# ---------- IO ----------
def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for k, v in cols.items():
        if k in ["time", "t"]:
            mapping[v] = "time"
        elif k in ["event", "status", "evt", "e"]:
            mapping[v] = "event"
        elif k in ["predicted", "score", "risk", "prediction", "pred"]:
            mapping[v] = "predicted"
        elif k in ["staging", "model", "method"]:
            mapping[v] = "staging"
    out = df.rename(columns=mapping)
    keep = [c for c in ["time", "event", "predicted", "staging"] if c in out.columns]
    return out[keep].copy()

def _load_concat(base_dir: str, method_prefix: str, suffixes: List[str]) -> Optional[pd.DataFrame]:
    parts = []
    for suf in suffixes:
        path = os.path.join(base_dir, f"{method_prefix}_{suf}.csv")
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                df = _standardize_cols(df)
                for col in ["time", "event", "predicted"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["time", "event", "predicted"])
                parts.append(df)
            except Exception as e:
                print(f"[WARN] Failed to read {path}: {e}")
    if not parts:
        return None
    merged = pd.concat(parts, axis=0, ignore_index=True)
    return merged.reset_index(drop=True)

# ---------- C-index ----------
def cindex_point(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    # lifelines 假定分数越大=风险越高。这里按你的要求不取反，直接用原始 predicted。
    return float(concordance_index(time, score, event_observed=event))

def _boot_chunk_worker(
    idx_chunk: np.ndarray,
    method_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ref_name: str
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    methods = list(method_arrays.keys())
    boot_vals = {m: [] for m in methods}
    diffs = {m: [] for m in methods if m != ref_name}
    for idx in idx_chunk:
        cache = {}
        for m in methods:
            t, e, s = method_arrays[m]  # s 是原始 predicted（未取反）
            ci = cindex_point(t[idx], e[idx], s[idx])
            cache[m] = ci
            boot_vals[m].append(ci)
        ref_ci = cache[ref_name]
        for m in methods:
            if m == ref_name:
                continue
            diffs[m].append(cache[m] - ref_ci)
    return boot_vals, diffs

def cindex_bootstrap_ci_and_p_parallel(
    data_by_method: Dict[str, pd.DataFrame],
    method_names: List[str],
    ref_name: str,
    alpha: float,
    n_boot: int,
    seed: int,
    n_jobs: int
) -> Dict[str, Dict[str, float]]:
    # 对齐长度
    n = None
    for m in method_names:
        df = data_by_method.get(m)
        if df is None: continue
        n = len(df) if n is None else min(n, len(df))
    if n is None or n <= 1:
        return {m: {"cindex": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_vs_ref": np.nan}
                for m in method_names}

    # 截断并转数组（使用原始 predicted）
    method_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m in method_names:
        df = data_by_method.get(m)
        if df is None or len(df) < n:
            return {mm: {"cindex": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_vs_ref": np.nan}
                    for mm in method_names}
        df = df.iloc[:n].reset_index(drop=True)
        method_arrays[m] = (
            df["time"].values.astype(float),
            df["event"].values.astype(int),
            df["predicted"].values.astype(float),  # 不取反
        )

    # 点估计
    point = {m: cindex_point(*method_arrays[m]) for m in method_names}

    # 生成所有 bootstrap 索引
    rng = np.random.default_rng(seed)
    idx_matrix = rng.integers(low=0, high=n, size=(n_boot, n), endpoint=False)

    # 切块
    if n_jobs is None or n_jobs == 0:
        n_jobs = max(1, mp.cpu_count())
    chunk_sizes = np.full(n_jobs, n_boot // n_jobs, dtype=int)
    chunk_sizes[: (n_boot % n_jobs)] += 1
    chunks = []
    start = 0
    for cs in chunk_sizes:
        if cs > 0:
            chunks.append(idx_matrix[start:start+cs])
        start += cs

    # 并行
    agg_vals = {m: [] for m in method_names}
    agg_diff = {m: [] for m in method_names if m != ref_name}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_boot_chunk_worker, ch, method_arrays, ref_name) for ch in chunks]
        for fut in as_completed(futures):
            boot_vals, diffs = fut.result()
            for m, lst in boot_vals.items():
                agg_vals[m].extend(lst)
            for m, lst in diffs.items():
                agg_diff[m].extend(lst)

    # 汇总
    out = {}
    for m in method_names:
        arr = np.asarray(agg_vals[m], dtype=float)
        if arr.size == 0:
            out[m] = {"cindex": point.get(m, np.nan), "ci_low": np.nan, "ci_high": np.nan, "p_vs_ref": np.nan}
            continue
        lo = np.quantile(arr, alpha/2)
        hi = np.quantile(arr, 1 - alpha/2)
        p = np.nan
        if m != ref_name:
            diff = np.asarray(agg_diff[m], dtype=float)
            if diff.size > 0:
                # 经验双侧 p（保持现状）
                p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        out[m] = {"cindex": point[m], "ci_low": lo, "ci_high": hi, "p_vs_ref": p}
    return out

# ---------- DeLong (AUROC) ----------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def _fast_delong(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float]:
    assert set(np.unique(label)).issubset({0, 1})
    pos = pred[label == 1]
    neg = pred[label == 0]
    m = len(pos); n = len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan
    X = np.concatenate((pos, neg))
    Tx = _compute_midrank(X)
    Tpos = Tx[:m]; Tneg = Tx[m:]
    auc = (Tpos.sum() - m*(m+1)/2) / (m*n)
    V10 = (Tpos - (m+1)/2) / n
    V01 = 1 - (Tneg - (m+1)/2) / m
    s10 = V10.var(ddof=1); s01 = V01.var(ddof=1)
    auc_var = s10 / m + s01 / n
    return float(auc), float(auc_var)

def _delong_covariance(p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    def _auc_struct(pred):
        pos = pred[y == 1]; neg = pred[y == 0]
        m = len(pos); n = len(neg)
        X = np.concatenate((pos, neg))
        Tx = _compute_midrank(X)
        Tpos = Tx[:m]; Tneg = Tx[m:]
        auc = (Tpos.sum() - m*(m+1)/2) / (m*n)
        V10 = (Tpos - (m+1)/2) / n
        V01 = 1 - (Tneg - (m+1)/2) / m
        return auc, V10, V01, m, n

    auc1, V10_1, V01_1, m, n = _auc_struct(p1)
    auc2, V10_2, V01_2, _m, _n = _auc_struct(p2)
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan
    s10_1 = V10_1.var(ddof=1); s01_1 = V01_1.var(ddof=1)
    s10_2 = V10_2.var(ddof=1); s01_2 = V01_2.var(ddof=1)
    c10 = np.cov(V10_1, V10_2, ddof=1)[0,1]
    c01 = np.cov(V01_1, V01_2, ddof=1)[0,1]
    var1 = s10_1 / m + s01_1 / n
    var2 = s10_2 / m + s01_2 / n
    cov12 = c10 / m + c01 / n
    return var1, var2, cov12

def delong_test_two_sided(p_ref: np.ndarray, p_alt: np.ndarray, y: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
    auc_ref, var_ref = _fast_delong(p_ref, y)
    auc_alt, var_alt = _fast_delong(p_alt, y)
    var1, var2, cov12 = _delong_covariance(p_ref, p_alt, y)
    if any(np.isnan([auc_ref, auc_alt, var1, var2, cov12])):
        return np.nan, (auc_ref, auc_alt, auc_alt - auc_ref)
    diff = auc_alt - auc_ref
    denom = math.sqrt(max(1e-12, var1 + var2 - 2*cov12))
    z = diff / denom
    p = 2 * (1 - norm_cdf(abs(z)))
    return float(p), (float(auc_ref), float(auc_alt), float(diff))

def delong_ci(p: np.ndarray, y: np.ndarray, alpha: float=0.05) -> Tuple[float, float, float]:
    auc, auc_var = _fast_delong(p, y)
    if np.isnan(auc) or np.isnan(auc_var):
        return np.nan, np.nan, np.nan
    se = math.sqrt(max(1e-12, auc_var))
    z = 1.959964
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return float(auc), float(lo), float(hi)

# ---------- Main per-cohort ----------
def compute_stats_for_cohort(
    base_dir: str,
    models_map: Dict[str, str],
    suffixes: List[str],
    cohort_name: str,
    out_csv: str,
    n_boot: int,
    seed: int,
    n_jobs: int
):
    # Load
    data_by_method = {}
    for nice, pref in models_map.items():
        df = _load_concat(base_dir, pref, suffixes)
        if df is None or len(df) == 0:
            print(f"[INFO] {cohort_name}: missing or empty for {nice} ({pref})")
            data_by_method[nice] = None
        else:
            data_by_method[nice] = df

    lengths = [len(df) for df in data_by_method.values() if df is not None]
    if not lengths:
        print(f"[WARN] {cohort_name}: no data found.")
        return
    n = min(lengths)
    for k in list(data_by_method.keys()):
        df = data_by_method[k]
        data_by_method[k] = None if (df is None or len(df) < n) else df.iloc[:n].reset_index(drop=True)

    method_names = list(models_map.keys())
    if "Our model" not in method_names:
        raise ValueError("Our model must be included.")

    # --- labels: event==1 为正类 ---
    y = None
    if data_by_method.get("Our model") is not None:
        y = (data_by_method["Our model"]["event"].values.astype(int) == 1).astype(int)
    else:
        for m in method_names:
            if data_by_method.get(m) is not None:
                y = (data_by_method[m]["event"].values.astype(int) == 1).astype(int)
                break
    if y is None:
        print(f"[WARN] {cohort_name}: cannot get labels.")
        return

    # --- C-index（原始 predicted，未取反） ---
    cidx_stats = cindex_bootstrap_ci_and_p_parallel(
        data_by_method=data_by_method,
        method_names=method_names,
        ref_name="Our model",
        alpha=0.05,
        n_boot=n_boot,
        seed=seed,
        n_jobs=n_jobs
    )

    # --- AUROC（DeLong）统一使用 -predicted ---
    auroc_rows = {}
    ref_scores = None if data_by_method.get("Our model") is None else -data_by_method["Our model"]["predicted"].values.astype(float)
    for m in method_names:
        dfm = data_by_method.get(m, None)
        if dfm is None:
            auroc_rows[m] = {"auroc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_vs_ref": np.nan}
            continue
        scores_neg = -dfm["predicted"].values.astype(float)   # 这里取反
        auc, lo, hi = delong_ci(scores_neg, y)
        p = np.nan
        if (ref_scores is not None) and (m != "Our model"):
            p, _ = delong_test_two_sided(ref_scores, scores_neg, y)
        auroc_rows[m] = {"auroc": auc, "ci_low": lo, "ci_high": hi, "p_vs_ref": p}

    # --- Save tidy CSV ---
    rows = []
    for m in method_names:
        cstat = cidx_stats.get(m, {"cindex": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_vs_ref": np.nan})
        rows.append({
            "Cohort": cohort_name,
            "Method": m,
            "Metric": "C-index",
            "Estimate": cstat["cindex"],
            "CI_lower_95": cstat["ci_low"],
            "CI_upper_95": cstat["ci_high"],
            "P_value_vs_Our": cstat["p_vs_ref"]
        })
        astat = auroc_rows.get(m, {"auroc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_vs_ref": np.nan})
        rows.append({
            "Cohort": cohort_name,
            "Method": m,
            "Metric": "AUROC",
            "Estimate": astat["auroc"],
            "CI_lower_95": astat["ci_low"],
            "CI_upper_95": astat["ci_high"],
            "P_value_vs_Our": astat["p_vs_ref"]
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv} with {len(out_df)} rows.")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Parallel stats: raw C-index & flipped AUROC.")
    parser.add_argument("--base_dir", type=str, default=".", help="Directory with df_<method>_<suffix>.csv")
    parser.add_argument("--internal_out", type=str, default="internal_stats.csv", help="Output CSV for internal cohort.")
    parser.add_argument("--external_out", type=str, default="external_stats.csv", help="Output CSV for external cohort.")
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap iterations for C-index.")
    parser.add_argument("--seed", type=int, default=20251012, help="Random seed for bootstrap.")
    parser.add_argument("--n_jobs", type=int, default=0, help="Processes to use (0 = CPU count).")
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs > 0 else max(1, mp.cpu_count())

    # Internal (seer)
    compute_stats_for_cohort(
        base_dir=args.base_dir,
        models_map=MODELS_INTERNAL,
        suffixes=INTERNAL_SUFFIXES,
        cohort_name="Internal (seer)",
        out_csv=args.internal_out,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=n_jobs
    )
    # External (cg_3 + multicenter)
    compute_stats_for_cohort(
        base_dir=args.base_dir,
        models_map=MODELS_EXTERNAL,
        suffixes=EXTERNAL_SUFFIXES,
        cohort_name="External (cg_3 + multicenter)",
        out_csv=args.external_out,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=n_jobs
    )

if __name__ == "__main__":
    main()
