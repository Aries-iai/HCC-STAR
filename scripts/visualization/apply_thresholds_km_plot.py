#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apply_thresholds_km_plot.py

读取阈值 JSON（既支持 predicted_thresholds 也支持 cdf_thresholds）。
- 若为 cdf_thresholds：需提供 --ecdf-path（.pkl 或 .npz），先把每条样本的 predicted(月) -> CDF 值，再按 CDF 阈值分组；
- 若为 predicted_thresholds：直接在 predicted 轴分组。

分别绘制：
  - Internal (seer)
  - External (cg_3 / multicenter / late_stage 合并)
"""

# 若你的图里会有中文或希腊字母，建议装并使用 Noto Sans CJK：
# mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Arial", "Helvetica", "DejaVu Sans"]
# mpl.rcParams["axes.unicode_minus"] = False  # 避免负号变方框


import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# ---------- 默认配置 ----------
CSV_PREFIX = "df_ours"
SUFFIX_TO_GROUP = {
    "seer": "Internal",
    "cg_3": "External",
    "multicenter": "External",
    "late_stage": "External",
}
BASE_DIR = "."
OUT_DIR = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin"
LABEL_ORDER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # A,B,C,...
COLOR_PALETTE = ["#008A45","#00A9A5","#F27873","#C0392B","#00A9A5","#5F5F5E","#B384BA","#4B3F72","#D9C2DD","#F27873","#C0392B","#FFD373"]#["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]  # K<=6

# ---------- 工具 ----------
def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _digitize_groups(x: np.ndarray, thresholds):
    thr = np.asarray(sorted(thresholds), dtype=float)
    return np.digitize(x, bins=thr, right=False)  # 0..K

def _groups_to_labels(groups: np.ndarray, values_for_order: np.ndarray, higher_is_better=True):
    uniq = np.unique(groups)
    # 以每组的 values_for_order 均值决定 A/B/... 的次序
    means = [(g, values_for_order[groups == g].mean()) for g in uniq]
    means.sort(key=lambda t: t[1], reverse=higher_is_better)
    labels_pool = LABEL_ORDER[:len(uniq)]
    mapping = {g: labels_pool[i] for i, (g, _) in enumerate(means)}
    out = np.empty(groups.shape, dtype=object)
    for g in uniq:
        out[groups == g] = mapping[g]
    # 返回标签顺序（按“更好→更差”）
    ordered_labels = [mapping[g] for g, _ in means]
    return out, ordered_labels

def load_one(prefix, suffix, base_dir):
    path = os.path.join(base_dir, f"{prefix}_{suffix}.csv")
    if not os.path.exists(path):
        print(f"[WARN] 跳过缺失：{path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    need = {"time", "event", "predicted"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺列：{sorted(list(missing))}")
    df["time"] = _to_numeric(df["time"])
    df["event"] = _to_numeric(df["event"])
    df["predicted"] = _to_numeric(df["predicted"])
    df = df.dropna(subset=["time", "event", "predicted"]).copy()
    df["dataset_suffix"] = suffix
    df["cohort"] = SUFFIX_TO_GROUP.get(suffix, "External")
    return df

def load_all(prefix, base_dir):
    parts = [load_one(prefix, s, base_dir) for s in SUFFIX_TO_GROUP.keys()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        print("[ERROR] 没有读到任何 CSV"); sys.exit(1)
    return pd.concat(parts, ignore_index=True)

# ---------- ECDF ----------
def load_ecdf(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        import pickle
        with open(path, "rb") as f:
            ecdf = pickle.load(f)
    elif ext == ".npz":
        d = np.load(path)
        ecdf = {"sorted_times": np.asarray(d["sorted_times"], dtype=float), "n": int(d["n"])}
    else:
        raise ValueError("ECDF must be .pkl or .npz")
    if "sorted_times" not in ecdf or "n" not in ecdf:
        raise ValueError("Bad ECDF file (need 'sorted_times' and 'n').")
    ecdf["sorted_times"] = np.asarray(ecdf["sorted_times"], dtype=float)
    ecdf["n"] = int(ecdf["n"])
    return ecdf

def ecdf_value(ecdf_obj, t):
    st = ecdf_obj["sorted_times"]; n = ecdf_obj["n"]
    t_arr = np.asarray(t, dtype=float)
    return np.searchsorted(st, t_arr, side="right").astype(float) / float(n)

# ---------- 绘图 ----------
def km_plot(df: pd.DataFrame, values_for_cut: np.ndarray, thresholds, cohort_name: str, out_prefix: str, title: str, higher_is_better=True):
    if df.empty:
        print(f"[WARN] {cohort_name} 为空，跳过")
        return
    groups = _digitize_groups(values_for_cut, thresholds)
    labels, ordered = _groups_to_labels(groups, values_for_order=values_for_cut, higher_is_better=higher_is_better)
    df_plot = df.copy()
    df_plot["stage_supervised"] = labels

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    for i, lab in enumerate(ordered):
        g = df_plot[df_plot["stage_supervised"] == lab]
        if g.empty: 
            continue
        kmf.fit(g["time"].values, event_observed=g["event"].astype(int).values, label=f"{lab} (N={len(g)})")
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        kmf.plot_survival_function(ci_show=False, color=color, linewidth=2)

    plt.title(title)
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Our Staging Label", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    plt.savefig(f"{out_prefix}.png", dpi=300)
    plt.savefig(f"{out_prefix}.svg")
    plt.close()
    print(f"[OK] 保存：{out_prefix}.png | .svg")

# ---------- 主流程 ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-json", type=str, default="ecdf_train_thresholds.json")
    ap.add_argument("--csv-prefix", type=str, default=CSV_PREFIX)
    ap.add_argument("--base-dir", type=str, default=BASE_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--ecdf-path", type=str, default="", help="若 JSON 为 cdf_thresholds 则必须提供 .pkl/.npz")
    return ap.parse_args()

def main():
    args = parse_args()

    with open(args.threshold_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    thr_pred = cfg.get("predicted_thresholds", None)
    thr_cdf  = cfg.get("cdf_thresholds", None)
    k = cfg.get("k_selected", None)

    if thr_cdf is None and thr_pred is None:
        raise ValueError(f"{args.threshold_json} 既无 'cdf_thresholds' 也无 'predicted_thresholds'。")

    # 加载数据
    df_all = load_all(args.csv_prefix, args.base_dir)
    df_internal = df_all.query("cohort=='Internal'").copy()
    df_external = df_all.query("cohort=='External'").copy()

    # 选择分组轴：cdf 或 predicted
    if thr_cdf is not None:
        if not args.ecdf_path:
            raise ValueError("该 JSON 使用 cdf_thresholds，请提供 --ecdf-path 指向训练好的 ECDF。")
        ecdf = load_ecdf(args.ecdf_path)
        # 在两个 cohort 上分别计算 cdf(predicted)
        if not df_internal.empty:
            x_int = ecdf_value(ecdf, df_internal["predicted"].values)
            km_plot(
                df_internal, x_int, thr_cdf,
                cohort_name="Internal",
                out_prefix=os.path.join(args.out_dir, f"{args.csv_prefix}_km_internal_by_supervised_stage"),
                title=f"",
                higher_is_better=True
            )
        if not df_external.empty:
            x_ext = ecdf_value(ecdf, df_external["predicted"].values)
            km_plot(
                df_external, x_ext, thr_cdf,
                cohort_name="External",
                out_prefix=os.path.join(args.out_dir, f"{args.csv_prefix}_km_external_by_supervised_stage"),
                title=f"",
                higher_is_better=True
            )
        print(f"[INFO] 使用 CDF 阈值：{thr_cdf}")
        if "predicted_thresholds_equivalent" in cfg:
            print(f"[INFO] 训练分布等价月份阈值：{cfg['predicted_thresholds_equivalent']}")
    else:
        # 直接在 predicted 轴分组
        if not df_internal.empty:
            km_plot(
                df_internal, df_internal["predicted"].values, thr_pred,
                cohort_name="Internal",
                out_prefix=os.path.join(args.out_dir, f"{args.csv_prefix}_km_internal_by_supervised_stage"),
                title=f"KM · Internal (Pred thresholds, K={k}) · {args.csv_prefix}",
                higher_is_better=True
            )
        if not df_external.empty:
            km_plot(
                df_external, df_external["predicted"].values, thr_pred,
                cohort_name="External",
                out_prefix=os.path.join(args.out_dir, f"{args.csv_prefix}_km_external_by_supervised_stage"),
                title=f"KM · External (Pred thresholds, K={k}) · {args.csv_prefix}",
                higher_is_better=True
            )
        print(f"[INFO] 使用 Predicted 阈值：{thr_pred}")

if __name__ == "__main__":
    main()
