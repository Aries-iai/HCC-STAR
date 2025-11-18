#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

# ======== Color Config (user-editable) ========
LIGHT_COLOR = "#87c0fa"   # Physician (light blue)
DARK_COLOR  = "#4B3F72"   # Physician + AI (deep blue)
AI_COLOR    = "#F39C12"   # AI only (orange)

# ======== Helpers ========

EXPECTED_COLS = {"Group","patient_id","time_s","gt_first","pred_list","Top1","Top2","Top3"}

def _display_labels(cats):
    return ["Our model" if c == "AI" else c for c in cats]

def read_per_case(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"per_case.csv is missing columns: {missing}")
    # force types
    for k in ["Top1","Top2","Top3"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["Group"] = df["Group"].astype(str)
    return df

def time_filter_values(vals: np.ndarray, mode: str) -> np.ndarray:
    arr = vals[~np.isnan(vals)]
    if arr.size == 0:
        return arr
    mode = (mode or "iqr1.5").lower()
    if mode == "none":
        return arr
    if mode in ("iqr1.5","iqr1.0","iqr0.5"):
        k = {"iqr1.5":1.5,"iqr1.0":1.0,"iqr0.5":0.5}[mode]
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k*iqr, q3 + k*iqr
        return arr[(arr >= lo) & (arr <= hi)]
    if mode == "mad2.5":
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0: 
            return arr
        thr = 2.5 * 1.4826 * mad
        return arr[(arr >= med - thr) & (arr <= med + thr)]
    if mode == "pct2-98":
        lo, hi = np.percentile(arr, [2,98])
        return arr[(arr >= lo) & (arr <= hi)]
    if mode == "pct5-95":
        lo, hi = np.percentile(arr, [5,95])
        return arr[(arr >= lo) & (arr <= hi)]
    return arr

def compute_accuracy(df: pd.DataFrame, topk_col: str) -> float:
    # micro-average over cases in this subgroup
    if df.empty:
        return float("nan")
    vals = pd.to_numeric(df[topk_col], errors="coerce")
    if vals.size == 0:
        return float("nan")
    return float(np.nanmean(vals) * 100.0)

def accuracy_panels_from_per_case(per_case: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # ---- 动态发现可用的 base：xxx (no AI) / xxx + AI ----
    bases = set()
    for g in per_case["Group"].astype(str).unique():
        g = g.strip()
        if g.endswith("(no AI)"):
            bases.add(g[:-7].strip())              # remove " (no AI)"
        elif g.endswith("+ AI"):
            bases.add(g[:-4].strip())              # remove " + AI"

    # 希望的展示顺序：Resident, Attending, Physician, 其余按字母序
    preferred = ["Resident", "Attending", "Physician"]
    rest = sorted([b for b in bases if b not in preferred])
    ordered_bases = [b for b in preferred if b in bases] + rest

    # 面板容器
    cats = []
    panels = {t: {"light": [], "dark": []} for t in ["Top-1 (%)","Top-2 (%)","Top-3 (%)"]}

    # AI only
    ai_df = per_case[per_case["Group"] == "AI only"]
    if not ai_df.empty:
        cats.append("AI")
        for title, col in [("Top-1 (%)","Top1"),("Top-2 (%)","Top2"),("Top-3 (%)","Top3")]:
            panels[title]["light"].append(compute_accuracy(ai_df, col))
            panels[title]["dark"].append(np.nan)

    # 动态加入各 base
    for base in ordered_bases:
        light_df = per_case[per_case["Group"] == f"{base} (no AI)"]
        dark_df  = per_case[per_case["Group"] == f"{base} + AI"]
        if light_df.empty and dark_df.empty:
            continue
        cats.append(base)
        for title, col in [("Top-1 (%)","Top1"),("Top-2 (%)","Top2"),("Top-3 (%)","Top3")]:
            panels[title]["light"].append(compute_accuracy(light_df, col))
            panels[title]["dark"].append(compute_accuracy(dark_df, col))

    if not cats:
        return []

    x = np.arange(len(cats))
    width = 0.35
    saved = []

    def _draw_with_center_ai(ax, title, light_vals, dark_vals):
        bars_light, bars_dark = [], []
        for idx in range(len(cats)):
            lv, dv = light_vals[idx], dark_vals[idx]
            has_l = isinstance(lv, (int, float)) and not math.isnan(lv)
            has_d = isinstance(dv, (int, float)) and not math.isnan(dv)

            if has_l and not has_d:
                b = ax.bar(x[idx], lv, width, color=LIGHT_COLOR)[0]   # AI-only 或单柱 → 居中
                bars_light.append(b); bars_dark.append(None)
            else:
                bl = ax.bar(x[idx] - width/2, lv if has_l else np.nan, width, color=LIGHT_COLOR)[0]
                bd = ax.bar(x[idx] + width/2, dv if has_d else np.nan, width, color=DARK_COLOR)[0]
                bars_light.append(bl); bars_dark.append(bd)

        # AI-only 单柱改为 AI 颜色
        if "AI" in cats:
            ai_idx = cats.index("AI")
            if bars_light[ai_idx] is not None:
                bars_light[ai_idx].set_color(AI_COLOR)
            ref = light_vals[ai_idx]  # AI only 的准确率（百分比）
            if isinstance(ref, (int, float)) and not math.isnan(ref):
                # 虚线
                ax.axhline(ref, linestyle="--", linewidth=1.2, color='gray', alpha=0.9)

        # 数值标注
        for b, v in zip(bars_light, light_vals):
            if b is not None and isinstance(v, (int, float)) and not math.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for b, v in zip(bars_dark, dark_vals):
            if b is not None and isinstance(v, (int, float)) and not math.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # 轴与范围（y 轴刻度显示到 100，顶部留少量空间但不超过 103）
        # ax.set_title(title.replace("(%)","Accuracy"))
        ax.set_ylabel(title.replace("(%)","Accuracy (%)"))
        # ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(_display_labels(cats))
        numeric_vals = [v for v in (light_vals + dark_vals) if isinstance(v, (int, float)) and not math.isnan(v)]
        vmax = max(numeric_vals) if numeric_vals else 80.0
        ylim_top = max(100.0, min(108.0, vmax + 8))
        ax.set_ylim(0, ylim_top)
        yticks = np.linspace(0, 100, 11)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(t)}" for t in yticks])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # 全局图例（上方）
        # ai_patch    = mpatches.Patch(color=AI_COLOR,    label="Our model")
        # light_patch = mpatches.Patch(color=LIGHT_COLOR, label="Physician")
        # dark_patch  = mpatches.Patch(color=DARK_COLOR,  label="Physician + Our model")
        # fig.legend(handles=[light_patch, dark_patch, ai_patch], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.00), frameon=False)

        # 图例（外置）
        light_patch = mpatches.Patch(color=LIGHT_COLOR, label="Physician")
        dark_patch  = mpatches.Patch(color=DARK_COLOR,  label="Physician + AI")
        ai_patch    = mpatches.Patch(color=AI_COLOR,    label="AI only")
        ax.legend(handles=[ai_patch, light_patch, dark_patch], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), frameon=False)
        plt.tight_layout(rect=[0,0,0.85,1])

    for title in ["Top-1 (%)","Top-2 (%)","Top-3 (%)"]:
        light_vals = panels[title]["light"]
        dark_vals  = panels[title]["dark"]
        fig, ax = plt.subplots(figsize=(6, 4.8))
        _draw_with_center_ai(ax, title, light_vals, dark_vals)
        base = title.split()[0].lower().replace("-","")
        png = os.path.join(outdir, f"{base}.png")
        svg = os.path.join(outdir, f"{base}.svg")
        plt.savefig(png, dpi=300, bbox_inches='tight')
        plt.savefig(svg, bbox_inches='tight')
        plt.close(fig)
        saved.append(png)

    return saved


def time_plot_from_per_case(per_case: pd.DataFrame, outdir: str, time_filter: str):
    os.makedirs(outdir, exist_ok=True)
    # physician-only groups
    res_no = per_case[per_case["Group"] == "Resident (no AI)"]["time_s"].to_numpy(dtype=float)
    res_ai = per_case[per_case["Group"] == "Resident + AI"]["time_s"].to_numpy(dtype=float)
    att_no = per_case[per_case["Group"] == "Attending (no AI)"]["time_s"].to_numpy(dtype=float)
    att_ai = per_case[per_case["Group"] == "Attending + AI"]["time_s"].to_numpy(dtype=float)

    # filter
    res_no = time_filter_values(res_no, time_filter)
    res_ai = time_filter_values(res_ai, time_filter)
    att_no = time_filter_values(att_no, time_filter)
    att_ai = time_filter_values(att_ai, time_filter)

    cats = ["Resident","Attending"]
    light_vals = [float(np.nanmean(res_no)) if res_no.size else np.nan,
                  float(np.nanmean(att_no)) if att_no.size else np.nan]
    dark_vals  = [float(np.nanmean(res_ai)) if res_ai.size else np.nan,
                  float(np.nanmean(att_ai)) if att_ai.size else np.nan]

    if all(np.isnan(light_vals)) and all(np.isnan(dark_vals)):
        return None

    x = np.arange(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars_light = ax.bar(x - width/2, light_vals, width, label="Physician")
    bars_dark  = ax.bar(x + width/2, dark_vals,  width, label="Physician + AI")
    for b in bars_light: b.set_color(LIGHT_COLOR)
    for b in bars_dark:  b.set_color(DARK_COLOR)

    for bars, vals in [(bars_light, light_vals), (bars_dark, dark_vals)]:
        for b, v in zip(bars, vals):
            if isinstance(v, (int,float)) and not math.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Decision Time (mean seconds)")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)

    vmax = max([v for v in light_vals+dark_vals if isinstance(v,(int,float)) and not math.isnan(v)] + [10])
    ax.set_ylim(0, vmax * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    light_patch = mpatches.Patch(color=LIGHT_COLOR, label="Physician")
    dark_patch  = mpatches.Patch(color=DARK_COLOR,  label="Physician + AI")
    ax.legend(handles=[light_patch, dark_patch], loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.tight_layout(rect=[0,0,0.85,1])
    png = os.path.join(outdir, "time.png")
    svg = os.path.join(outdir, "time.svg")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    return png

def combined_4panels_from_per_case(per_case: pd.DataFrame, outdir: str, time_filter: str):
    os.makedirs(outdir, exist_ok=True)

    # ---- 动态发现 base ----
    bases = set()
    for g in per_case["Group"].astype(str).unique():
        g = g.strip()
        if g.endswith("(no AI)"):
            bases.add(g[:-7].strip())
        elif g.endswith("+ AI"):
            bases.add(g[:-4].strip())
    preferred = ["Resident", "Attending", "Physician"]
    rest = sorted([b for b in bases if b not in preferred])
    ordered_bases = [b for b in preferred if b in bases] + rest

    # 准确率数据容器
    cats = []
    series = {t: {"light": [], "dark": []} for t in ["Top-1 (%)","Top-2 (%)","Top-3 (%)"]}

    # AI only
    ai_df = per_case[per_case["Group"] == "AI only"]
    if not ai_df.empty:
        cats.append("AI")
        for title, col in [("Top-1 (%)","Top1"),("Top-2 (%)","Top2"),("Top-3 (%)","Top3")]:
            series[title]["light"].append(compute_accuracy(ai_df, col))
            series[title]["dark"].append(np.nan)
        

    # 动态加入各 base
    for base in ordered_bases:
        light_df = per_case[per_case["Group"] == f"{base} (no AI)"]
        dark_df  = per_case[per_case["Group"] == f"{base} + AI"]
        if light_df.empty and dark_df.empty:
            continue
        cats.append(base)
        for title, col in [("Top-1 (%)","Top1"),("Top-2 (%)","Top2"),("Top-3 (%)","Top3")]:
            series[title]["light"].append(compute_accuracy(light_df, col))
            series[title]["dark"].append(compute_accuracy(dark_df, col))

    # 时间数据
    def _get_times(tag):
        return per_case[per_case["Group"] == tag]["time_s"].to_numpy(dtype=float)

    # 尝试优先 Resident / Attending；若没有则回退到 Physician
    res_no = time_filter_values(_get_times("Resident (no AI)"), time_filter)
    res_ai = time_filter_values(_get_times("Resident + AI"), time_filter)
    att_no = time_filter_values(_get_times("Attending (no AI)"), time_filter)
    att_ai = time_filter_values(_get_times("Attending + AI"), time_filter)

    # 如果没有 Resident/Attending，则尝试 Physician
    use_physician_only = (res_no.size == 0 and res_ai.size == 0 and att_no.size == 0 and att_ai.size == 0)
    if use_physician_only:
        phy_no = time_filter_values(_get_times("Physician (no AI)"), time_filter)
        phy_ai = time_filter_values(_get_times("Physician + AI"), time_filter)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    width = 0.2
    x = np.arange(len(cats)) * 0.8

    def draw_acc(ax, title, light_vals, dark_vals):
        bars_light, bars_dark = [], []
        for idx in range(len(cats)):
            lv, dv = light_vals[idx], dark_vals[idx]
            has_l = isinstance(lv, (int, float)) and not math.isnan(lv)
            has_d = isinstance(dv, (int, float)) and not math.isnan(dv)
            if has_l and not has_d:
                b = ax.bar(x[idx], lv, width, color=LIGHT_COLOR)[0]   # 单柱居中
                bars_light.append(b); bars_dark.append(None)
            else:
                bl = ax.bar(x[idx] - width/2, lv if has_l else np.nan, width, color=LIGHT_COLOR)[0]
                bd = ax.bar(x[idx] + width/2, dv if has_d else np.nan, width, color=DARK_COLOR)[0]
                bars_light.append(bl); bars_dark.append(bd)

        if "AI" in cats:
            ai_idx = cats.index("AI")
            if bars_light[ai_idx] is not None:
                bars_light[ai_idx].set_color(AI_COLOR)
            ref = light_vals[ai_idx]  # AI only 的准确率（百分比）
            if isinstance(ref, (int, float)) and not math.isnan(ref):
                # 虚线
                ax.axhline(ref, linestyle="--", linewidth=1.2, color='gray', alpha=0.9)

        for b, v in zip(bars_light, light_vals):
            if b is not None and isinstance(v, (int, float)) and not math.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for b, v in zip(bars_dark, dark_vals):
            if b is not None and isinstance(v, (int, float)) and not math.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(_display_labels(cats))

        numeric_vals = [v for v in (light_vals + dark_vals) if isinstance(v, (int, float)) and not math.isnan(v)]
        vmax = max(numeric_vals) if numeric_vals else 80.0
        ylim_top = max(100.0, min(108.0, vmax + 8))
        ax.set_ylim(0, ylim_top)
        yticks = np.linspace(0, 100, 11)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(t)}" for t in yticks])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 三个准确率子图
    draw_acc(fig.add_subplot(gs[0,0]), "Top-1 Accuracy", series["Top-1 (%)"]["light"], series["Top-1 (%)"]["dark"])
    draw_acc(fig.add_subplot(gs[0,1]), "Top-2 Accuracy", series["Top-2 (%)"]["light"], series["Top-2 (%)"]["dark"])
    draw_acc(fig.add_subplot(gs[1,0]), "Top-3 Accuracy", series["Top-3 (%)"]["light"], series["Top-3 (%)"]["dark"])

    # 时间面板（若只有 Physician，就显示 Physician；否则显示 Resident/Attending）
    ax4 = fig.add_subplot(gs[1,1])
    if use_physician_only:
        bases_time = ["Physician"]
        light_vals = [float(np.nanmean(phy_no)) if phy_no.size else np.nan]
        dark_vals  = [float(np.nanmean(phy_ai)) if phy_ai.size else np.nan]
    else:
        bases_time = ["Resident", "Attending"]
        light_vals = [float(np.nanmean(res_no)) if res_no.size else np.nan,
                      float(np.nanmean(att_no)) if att_no.size else np.nan]
        dark_vals  = [float(np.nanmean(res_ai)) if res_ai.size else np.nan,
                      float(np.nanmean(att_ai)) if att_ai.size else np.nan]

    bar_width = 0.12
    xb = np.arange(len(bases_time)) * 0.7
    bars_light = ax4.bar(xb - bar_width/2, light_vals, bar_width, label="Physician", color=LIGHT_COLOR)
    bars_dark  = ax4.bar(xb + bar_width/2, dark_vals,  bar_width, label="Physician + AI", color=DARK_COLOR)
    # xb = np.arange(len(bases_time))*0.8
    # bars_light = ax4.bar(xb - width/2, light_vals, width, label="Physician", color=LIGHT_COLOR)
    # bars_dark  = ax4.bar(xb + width/2, dark_vals,  width, label="Physician + AI", color=DARK_COLOR)
    for bars, vals in [(bars_light, light_vals), (bars_dark, dark_vals)]:
        for b, v in zip(bars, vals):
            if isinstance(v, (int,float)) and not math.isnan(v):
                ax4.text(b.get_x()+b.get_width()/2, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax4.set_title("Annotation Time (mean seconds)")
    ax4.set_ylabel("Seconds")
    ax4.set_xticks(xb)
    ax4.set_xticklabels(bases_time)
    vmax = max([v for v in light_vals+dark_vals if isinstance(v,(int,float)) and not math.isnan(v)] + [10])
    ax4.set_ylim(0, vmax * 1.15)
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 全局图例（上方）
    ai_patch    = mpatches.Patch(color=AI_COLOR,    label="Our model")
    light_patch = mpatches.Patch(color=LIGHT_COLOR, label="Physician")
    dark_patch  = mpatches.Patch(color=DARK_COLOR,  label="Physician + Our model")
    fig.legend(handles=[light_patch, dark_patch, ai_patch], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.00), frameon=False, fontsize=14)

    plt.tight_layout(rect=[0,0,1,0.95])
    png = os.path.join(outdir, "combined_4panels.png")
    svg = os.path.join(outdir, "combined_4panels.svg")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser(description="Plot from per_case.csv only")
    ap.add_argument("--per-case", required=True, help="Path to per_case.csv")
    ap.add_argument("--outdir", required=True, help="Output dir")
    ap.add_argument("--time-filter", type=str, default="iqr1.5",
                    choices=["none","iqr1.5","iqr1.0","iqr0.5","mad2.5","pct2-98","pct5-95"],
                    help="Outlier filter for time means")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = read_per_case(args.per_case)

    # Accuracy
    accuracy_panels_from_per_case(df, args.outdir)

    # Time
    time_plot_from_per_case(df, args.outdir, args.time_filter)

    # Combined
    combined_4panels_from_per_case(df, args.outdir, args.time_filter)

    print(f"[OK] Plots saved to: {args.outdir}")

if __name__ == "__main__":
    main()
