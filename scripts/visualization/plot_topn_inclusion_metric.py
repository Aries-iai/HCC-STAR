#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Top-N inclusion accuracy (Science-like, color palette, legend above)

- Ground truth for both internal/external: tx_actual (single string)
- Metric: inclusion — 1 if gt in top-k predictions, else 0
- N in {1, 2, 3} for both cohorts

Outputs:
  - external_topn_stats.csv
  - internal_topn_stats.csv
  - external_topn_accuracy.(png|svg)
  - internal_topn_accuracy.(png|svg)
"""

import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =================== Science-like Style ===================

def set_science_style():
    """Clean, Science-like base style."""
    mpl.rcParams.update({
        "figure.figsize": (10.5, 4.8),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "grid.alpha": 0.0,
        "axes.grid": False,
        "figure.autolayout": False,  # we'll control layout manually
    })

def okabe_ito_colors(n: int) -> List[str]:
    """
    Colorblind-friendly Okabe–Ito palette (up to 8).
    Order: orange, skyblue, green, yellow, blue, vermilion, purple, gray
    """
    base = ["#4B3F72", "#468BCA", "#00A9A5", "#C0392B", "#468BCA", "#D9C2DD", "#5F5F5E", "#008A45", "#F27873", "#00A9A5", "#B384BA", "#7DD2F6"]
    # base = ["#008A45","#468BCA","#7DD2F6","#80C5A2","#00A9A5","#5F5F5E","#B384BA","#4B3F72","#D9C2DD","#F27873","#C0392B","#FFD373"]
    if n <= len(base):
        return base[:n]
    # repeat if more needed
    out = []
    while len(out) < n:
        out += base
    return out[:n]


# =================== Normalization / I/O ===================

def normalize_label(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower().replace('-', '_')
    while '__' in s:
        s = s.replace('__', '_')
    return s

def normalize_list(lst: List[str]) -> List[str]:
    return [normalize_label(z) for z in lst if isinstance(z, str) and z.strip() != ""]

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def detect_method_keys(samples: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for s in samples:
        for k in s.keys():
            if not k.startswith('tx_'):
                continue
            if k in ('tx_actual', 'tx_actual_all', 'tx_svm'):
                continue
            keys.add(k)
    return sorted(keys)

def pretty_method_name(tx_key: str) -> str:
    base = tx_key[3:] if tx_key.startswith('tx_') else tx_key
    mapping = {
        "ours": "Our model",
        "tnm": "TNM",
        "bclc": "BCLC",
        "cnlc": "CNLC",
        "claude-3-5-sonnet-20241022": "Claude",
        "deepseek-r1": "DeepSeek-R1",
        "gpt-4o-2024-08-06": "GPT-4o",
        "gemini-2_5-pro": "Gemini-2.5-pro",
        "gpt-5": "GPT-5",
        "bayesian": "Bayes",
        "xgboost": "XGBoost",
    }
    if base in mapping:
        return mapping[base]
    return base.replace('_', '-').upper()

def extract_pred_list(sample: Dict[str, Any], key: str) -> List[str]:
    vals = sample.get(key, [])
    if isinstance(vals, list):
        return normalize_list(vals)
    if isinstance(vals, str):
        return normalize_list([vals])
    return []


# =================== Metric (Top-N inclusion) ===================

def list_to_topk_dict(pred_list: List[str], k: int) -> Dict[str, str]:
    out = {}
    for i in range(min(k, len(pred_list))):
        out[f"top{i+1}"] = pred_list[i]
    return out

def calculate_topn_accuracy(majority_voted_treatments: Dict[str, str], gt_treatment: str, n: int) -> int:
    if gt_treatment is None or str(gt_treatment).strip() == "":
        return 0
    top_predictions = []
    for i in range(1, n + 1):
        top_key = f'top{i}'
        if top_key in majority_voted_treatments and majority_voted_treatments[top_key] is not None:
            top_predictions.append(majority_voted_treatments[top_key])
    return 1 if gt_treatment in top_predictions else 0


# =================== Scoring ===================

def score_topn(samples: List[Dict[str, Any]], method_keys: List[str], n_values: Iterable[int]) -> Dict[str, Dict[int, float]]:
    n_values = list(sorted(set(int(n) for n in n_values)))
    sums = {n: defaultdict(int) for n in n_values}
    counts = defaultdict(int)

    for s in samples:
        gt = s.get('tx_actual', None)
        if gt is None or str(gt).strip() == "":
            continue
        gt_norm = normalize_label(gt)

        for mk in method_keys:
            preds = extract_pred_list(s, mk)
            for n in n_values:
                topd = list_to_topk_dict(preds, k=n)
                acc = calculate_topn_accuracy(topd, gt_norm, n=n)
                sums[n][mk] += acc
            counts[mk] += 1

    out = {mk: {n: 0.0 for n in n_values} for mk in method_keys}
    for mk in method_keys:
        c = counts.get(mk, 0)
        if c > 0:
            for n in n_values:
                out[mk][n] = sums[n][mk] / c
    return out


# =================== Plotting (Science + Color) ===================

def order_methods_by_top1(scores_topn: Dict[str, Dict[int, float]]) -> List[str]:
    # Our model first; others by Top-1 desc.
    top1 = {mk: d.get(1, 0.0) for mk, d in scores_topn.items()}
    items = list(top1.items())
    items.sort(key=lambda kv: (0 if kv[0] == 'tx_ours' else 1, -kv[1], kv[0]))
    return [k for k, _ in items]

def plot_grouped_bars_science_color(
    scores_topn: Dict[str, Dict[int, float]],
    n_values: Iterable[int],
    title: str,
    ylabel: str,
    save_prefix: str
):
    methods = order_methods_by_top1(scores_topn)
    labels = [pretty_method_name(m) for m in methods]
    n_values = list(sorted(set(int(n) for n in n_values)))
    data = {n: [scores_topn[m].get(n, 0.0) for m in methods] for n in n_values}

    set_science_style()
    colors = okabe_ito_colors(len(n_values))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(methods))
    width = min(0.8 / len(n_values), 0.25)

    # 为避免图例与柱子（尤其 Our model）重叠，把图例放在顶部并留白
    top_margin = 0.05  # 预留上边距给 legend
    plt.subplots_adjust(top=1.0 - top_margin)

    offsets = (np.arange(len(n_values)) - (len(n_values)-1)/2.0) * width
    for i, n in enumerate(n_values):
        ax.bar(x + offsets[i], [round(v * 100, 1) for v in data[n]], width, label=f"Top-{n} ACC",
               color=colors[i], edgecolor='black', linewidth=0.6)

    # ticks & labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    # print(title)
    if 'External' in title:
        ax.set_ylim(0, 85) 
    else:
        ax.set_ylim(0, 100)
    # ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)

    # legend on top, centered, outside axes
    ax.legend(ncol=len(n_values), loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=12)

    # value labels for Top-1 only
    if 1 in n_values:
        i = n_values.index(1)
        for xi, v in zip(x, data[1]):
            # 控制文本不越界
            ytxt = min(v*100 + 0.02, 98)
            ax.text(xi + offsets[i], ytxt, f"{v*100:.1f}", ha="center", va="bottom", fontsize=6, weight='bold', clip_on=True)

    # clean spines
    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)

    plt.savefig(f"{save_prefix}.png")
    plt.savefig(f"{save_prefix}.svg")
    plt.close()


# =================== CSV Export ===================

def export_topn_csv(scores_topn: Dict[str, Dict[int, float]], n_values: Iterable[int], out_csv: str):
    n_values = list(sorted(set(int(n) for n in n_values)))
    rows = []
    for mk, d in scores_topn.items():
        row = {"MethodKey": mk, "Method": pretty_method_name(mk)}
        for n in n_values:
            row[f"Top{n}_Accuracy"] = d.get(n, 0.0)
        rows.append(row)
    rows = sorted(rows, key=lambda r: (0 if r["MethodKey"] == "tx_ours" else 1, -r.get("Top1_Accuracy", 0.0), r["MethodKey"]))
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# =================== Main ===================

def main():
    parser = argparse.ArgumentParser(description="Top-N inclusion accuracy (Science-style color) for internal/external cohorts.")
    parser.add_argument("--cg3", type=str, default="staging_survival_cg_3.jsonl", help="Path to cg_3 jsonl")
    parser.add_argument("--multicenter", type=str, default="staging_survival_multicenter.jsonl", help="Path to multicenter jsonl")
    parser.add_argument("--seer", type=str, default="staging_survival_seer.jsonl", help="Path to seer jsonl")
    parser.add_argument("--n_values", type=int, nargs="+", default=[1, 2, 3], help="Top-N values to compute")
    parser.add_argument("--external_csv", type=str, default="external_topn_stats.csv")
    parser.add_argument("--internal_csv", type=str, default="internal_topn_stats.csv")
    args = parser.parse_args()

    # Load data
    cg3 = read_jsonl(args.cg3)
    mc  = read_jsonl(args.multicenter)
    seer = read_jsonl(args.seer)

    external_samples = cg3 + mc
    ext_methods = detect_method_keys(external_samples)
    int_methods = detect_method_keys(seer)

    if not external_samples:
        print("[WARN] No external samples loaded.")
    if not seer:
        print("[WARN] No internal samples loaded.")
    if not ext_methods and not int_methods:
        print("[WARN] No tx_* methods detected.")

    # External: Top-1/2/3 using tx_actual
    ext_scores_topn = score_topn(external_samples, ext_methods, n_values=args.n_values)
    export_topn_csv(ext_scores_topn, args.n_values, args.external_csv)
    ext_scores_topn = {k: v for k, v in ext_scores_topn.items() if k != "tx_ours_sft"}
    print(ext_scores_topn)
    plot_grouped_bars_science_color(
        ext_scores_topn, args.n_values,
        title="External: Top-1 / Top-2 / Top-3 Accuracy",
        # title="",
        ylabel="Top-N Accuracy",
        save_prefix="visual_fin_1014_2/external_topn_accuracy"
    )
    print("[OK] External Top-N plots & CSV saved.")

    # Internal: Top-1/2/3 using tx_actual
    int_scores_topn = score_topn(seer, int_methods, n_values=args.n_values)
    int_scores_topn = {k: v for k, v in int_scores_topn.items() if k != "tx_ours_sft"}
    print(int_scores_topn)
    export_topn_csv(int_scores_topn, args.n_values, args.internal_csv)
    plot_grouped_bars_science_color(
        int_scores_topn, args.n_values,
        title="Internal: Top-1 / Top-2 / Top-3 Accuracy",
        # title="",
        ylabel="Top-N Accuracy",
        save_prefix="visual_fin_1014_2/internal_topn_accuracy"
    )
    print("[OK] Internal Top-N plots & CSV saved.")

if __name__ == "__main__":
    main()
