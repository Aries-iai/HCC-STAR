# -*- coding: utf-8 -*-
"""
Two figures with clustered shared bars (rectangular, no rounded corners)

Figure 1 (OUT_TOPN_GROUPED_FIG):
- Clusters: External | Internal
- Bars per cluster: Top-1, Top-2, Top-3
- Each bar is shared (ours & ours_sft in one bar); cap = |ours - sft|
  * cap color = pos_cap_color if (ours >= sft) else neg_color

Figure 2 (OUT_CINDEX_GROUPED_FIG):
- Clusters: External | Internal
- Bars per cluster: Overall, 1yr, 3yr, 5yr
- Same shared-bar logic

CLI palettes:
  --bar-colors "#4B3F72,#FFD373,#80C5A2"      # per-category bar colors
  --pos-cap-colors "#4B3F72,#CC9E00,#2B8F6B"  # per-category positive cap colors (default: same as bar)
  --neg-colors "#C0392B,#F27873,#9b1d20"      # negative caps (ours<sft)

CSV dumps kept: OUT_TOPN_SUMMARY, OUT_CINDEX_SUMMARY
"""

import os, json, re, argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ===================== 路径 =====================
# External JSONL：合并 cg_3 + multicenter
JSONL_CG = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_cg_3.jsonl"
JSONL_MC = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_multicenter_merged.jsonl"
# Internal JSONL（seer）
JSONL_SEER = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_seer.jsonl"

# External CSV：C-index（ours / ours_sft）
CSV_OURS_CG  = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/df_ours_cg_3.csv"
CSV_SFT_CG   = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/df_ours_sft_cg_3.csv"
CSV_OURS_MC  = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/df_ours_multicenter.csv"
CSV_SFT_MC   = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/df_ours_sft_multicenter.csv"
# Internal CSV（seer）
CSV_OURS_SEER = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/df_ours_seer.csv"
CSV_SFT_SEER  = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/df_ours_sft_seer.csv"

# 输出路径（和你此前一致）
OUT_TOPN_SUMMARY       = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin/topn_merged_summary.csv"
OUT_CINDEX_SUMMARY     = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin/cindex_merged_summary.csv"
OUT_TOPN_GROUPED_FIG   = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin/topn_internal_external_stylized.svg"
OUT_CINDEX_GROUPED_FIG = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin/cindex_internal_external_stylized.svg"

# ===================== 读写 & 计算工具 =====================
_USE_LIFELINES = True
try:
    from lifelines.utils import concordance_index as _lifelines_cindex
except Exception:
    _USE_LIFELINES = False

def concordance_index(time_arr, event_arr, score_arr):
    time_arr = np.asarray(time_arr, float)
    event_arr = np.asarray(event_arr, int)
    score_arr = np.asarray(score_arr, float)
    if _USE_LIFELINES:
        return float(_lifelines_cindex(time_arr, score_arr, event_arr))
    # 兜底（慢）
    conc = ties = perm = 0.0
    n = len(time_arr)
    for i in range(n - 1):
        ti, ei, si = time_arr[i], event_arr[i], score_arr[i]
        for j in range(i + 1, n):
            tj, ej, sj = time_arr[j], event_arr[j], score_arr[j]
            if ti == tj: continue
            if ti < tj and ei == 1:
                perm += 1; conc += (si > sj); ties += (si == sj)
            elif tj < ti and ej == 1:
                perm += 1; conc += (sj > si); ties += (si == sj)
    return (conc + 0.5 * ties) / perm if perm > 0 else np.nan

def load_jsonl(p: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(p):
        print(f"[WARN] JSONL not found: {p}")
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def read_and_concat_csv(csv_paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"[WARN] CSV not found: {p}")
            continue
        df = pd.read_csv(p)
        for c in ["time", "event", "predicted"]:
            if c not in df.columns:
                raise ValueError(f"{p} 缺少列: {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["time","event","predicted"])
        if not df.empty:
            dfs.append(df[["time","event","predicted"]])
    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame(columns=["time","event","predicted"])

# ====== Top-N 计算 ======
WEIGHTS_POS = [1.0, 0.6, 0.3]  # top1/top2/top3

def build_weighted_list(tx_list: List[str]) -> List[Tuple[str, float]]:
    if not isinstance(tx_list, list): return []
    return [(t, WEIGHTS_POS[i]) for i, t in enumerate(tx_list[:3])]

def pick_topk_after_threshold(weighted_items: List[Tuple[str, float]], thr: float, k: int) -> List[str]:
    filt = [x for x in weighted_items if x[1] >= thr]
    filt.sort(key=lambda x: -x[1])  # 稳定排序：同权保序
    return [t for t,_ in filt[:k]]

def rebuild_voting_from_weighted(weighted_items: List[Tuple[str,float]], thr: float):
    tops = pick_topk_after_threshold(weighted_items, thr, 3)
    vr = {"top1": None, "top2": None, "top3": None}
    for i, t in enumerate(tops):
        vr[f"top{i+1}"] = t
    return vr

def calculate_topn_accuracy_from_voting(voting_result, gt_treatment, n):
    if gt_treatment is None or voting_result is None:
        return 0
    preds = []
    for i in range(1, n+1):
        v = voting_result.get(f"top{i}")
        if v is not None:
            preds.append(v)
    return 1 if gt_treatment in preds else 0

def compute_topn_from_jsonls(jsonl_paths: List[str],
                             thr_list: List[float],
                             ours_key: str) -> Dict[float, Tuple[float,float,float,int]]:
    rows = []
    for p in jsonl_paths:
        rows += load_jsonl(p)
    results = {}
    for thr in thr_list:
        n = 0; a1=a2=a3=0.0
        for r in rows:
            gt = r.get("tx_actual")
            tx_list = r.get(ours_key, [])
            weighted = build_weighted_list(tx_list)
            vr = rebuild_voting_from_weighted(weighted, thr)
            a1 += calculate_topn_accuracy_from_voting(vr, gt, 1)
            a2 += calculate_topn_accuracy_from_voting(vr, gt, 2)
            a3 += calculate_topn_accuracy_from_voting(vr, gt, 3)
            n += 1
        results[thr] = (a1/n if n else 0.0, a2/n if n else 0.0, a3/n if n else 0.0, n)
    return results

# ====== C-index ======
def compute_cindex(df: pd.DataFrame) -> Tuple[float,int]:
    if df is None or df.empty:
        return float("nan"), 0
    return float(concordance_index(df["time"], df["predicted"], df["event"])), len(df)

def capped_cindex_from_df(df: pd.DataFrame, horizon_months: int) -> float:
    if df is None or df.empty:
        return float("nan")
    t = df["time"].to_numpy(float)
    e = df["event"].to_numpy(int)
    s = df["predicted"].to_numpy(float)
    t_cap = np.minimum(t, horizon_months)
    e_cap = ((e == 1) & (t <= horizon_months)).astype(int)
    return float(concordance_index(t_cap, e_cap, s))

# ===================== 颜色 & 绘图 =====================
_HEX_RE = re.compile(r'^#?[0-9A-Fa-f]{6}$')
DEFAULT_BAR_COLORS = ["#4B3F72", "#FFD373", "#80C5A2", "#468BCA", "#D9C2DD", "#5F5F5E"]
DEFAULT_NEG_COLORS = ["#C0392B", "#F27873", "#9b1d20"]

def parse_color_list(s: str):
    if not s:
        return []
    parts = [p.strip() for p in re.split(r'[,\s]+', s) if p.strip()]
    out = []
    for p in parts:
        if not _HEX_RE.match(p):
            raise ValueError(f"非法颜色值: {p}")
        out.append(p if p.startswith("#") else f"#{p}")
    return out

def draw_shared_bar_rect(ax, x, ours, sft,
                         *, ymax=1.0, bar_width=0.28,
                         bar_color="#4B3F72", pos_cap_color="#4B3F72", neg_cap_color="#C0392B",
                         annotate=True):
    """直角共享柱：底座=min(ours,sft)，帽子=|ours-sft|，正负帽颜色可配。"""
    if not (np.isfinite(ours) and np.isfinite(sft)):
        return
    ours = float(ours); sft = float(sft)
    base_h = max(0.0, min(ymax, min(ours, sft)))
    top_h  = max(0.0, min(ymax, max(ours, sft)))
    delta  = ours - sft
    cap_h  = max(0.0, min(ymax, abs(delta)))

    # 底座
    ax.add_patch(Rectangle((x - bar_width/2, 0.0), bar_width, base_h, linewidth=0, facecolor=bar_color, alpha=1.0))
    # 帽子
    if cap_h > 0:
        hat_color = pos_cap_color if delta >= 0 else neg_cap_color
        ax.add_patch(Rectangle((x - bar_width/2, base_h), bar_width, cap_h, linewidth=0, facecolor=hat_color, alpha=0.95))

    if annotate:
        ax.text(x, top_h + 0.02, f"{max(ours, sft):.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

def clustered_shared_bars(ax,
                          cohorts: List[str],
                          categories: List[str],
                          data: Dict[str, Dict[str, Tuple[float,float]]],
                          *,
                          ymax=1.0,
                          cluster_gap=1.2,
                          intra_gap=0.18,
                          bar_width=0.28,
                          bar_colors=None,
                          pos_cap_colors=None,
                          neg_colors=None,
                          title=None,
                          ylabel=None,
                          legend_loc="upper left"):
    """
    cohorts    = ["External", "Internal"]
    categories = ["Top-1","Top-2","Top-3"] 或 ["Overall","1yr","3yr","5yr"]
    data       = {
        "External": {"Top-1": (ours,sft), ...},
        "Internal": {"Top-1": (ours,sft), ...},
    }
    """
    bar_colors     = bar_colors     or DEFAULT_BAR_COLORS
    pos_cap_colors = pos_cap_colors or bar_colors
    neg_colors     = neg_colors     or DEFAULT_NEG_COLORS

    n_clusters = len(cohorts)
    n_cats = len(categories)

    # x 位置
    cluster_centers = np.arange(n_clusters) * cluster_gap
    # 每簇内类别条的中心
    cat_offsets = (np.arange(n_cats) - (n_cats - 1)/2.0) * (bar_width + intra_gap)

    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # 画柱
    for ci, cohort in enumerate(cohorts):
        cx = cluster_centers[ci]
        for ki, cat in enumerate(categories):
            x = cx + cat_offsets[ki]
            ours, sft = data.get(cohort, {}).get(cat, (np.nan, np.nan))
            draw_shared_bar_rect(
                ax, x, ours, sft,
                ymax=ymax, bar_width=bar_width,
                bar_color=bar_colors[ki % len(bar_colors)],
                pos_cap_color=pos_cap_colors[ki % len(pos_cap_colors)],
                neg_cap_color=neg_colors[ki % len(neg_colors)],
            )
        # 簇中线
        ax.axvline(cx, ymin=0.0, ymax=1.0, color="gray", linestyle="--", alpha=0.30)

    # 轴与标题
    ax.set_xticks(cluster_centers, cohorts, fontsize=12)
    ax.set_xlim(cluster_centers[0] - cluster_gap*0.6, cluster_centers[-1] + cluster_gap*0.6)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title, fontsize=13, pad=8)

    # 图例（按类别配色）
    proxies = [Rectangle((0,0),1,1,facecolor=bar_colors[i % len(bar_colors)], label=categories[i]) for i in range(n_cats)]
    ax.legend(handles=proxies, frameon=False, loc=legend_loc)

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar-colors", type=str, default="", help="类别柱色（逗号/空格分隔，按类别顺序循环）")
    parser.add_argument("--pos-cap-colors", type=str, default="", help="正向帽子色卡（按类别顺序循环；默认同柱色）")
    parser.add_argument("--neg-colors", type=str, default="", help="ours<sft 的帽子色卡（循环）")
    args = parser.parse_args()

    BAR_COLORS     = parse_color_list(args.bar_colors) if args.bar_colors else DEFAULT_BAR_COLORS
    POS_CAP_COLORS = parse_color_list(args.pos_cap_colors) if args.pos_cap_colors else BAR_COLORS
    NEG_COLORS     = parse_color_list(args.neg_colors) if args.neg_colors else DEFAULT_NEG_COLORS

    # ---------- Top-N ----------
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # external：cg_3 + multicenter
    topn_ours_ext     = compute_topn_from_jsonls([JSONL_CG, JSONL_MC], thresholds, "tx_ours")
    topn_ours_sft_ext = compute_topn_from_jsonls([JSONL_CG, JSONL_MC], thresholds, "tx_ours_sft")
    # internal：seer
    topn_ours_int     = compute_topn_from_jsonls([JSONL_SEER], thresholds, "tx_ours")
    topn_ours_sft_int = compute_topn_from_jsonls([JSONL_SEER], thresholds, "tx_ours_sft")

    t_show = 0.5
    # 保存外部点的摘要（与之前一致）
    o1,o2,o3,n_ext = topn_ours_ext[t_show]
    s1,s2,s3,_     = topn_ours_sft_ext[t_show]
    pd.DataFrame(
        [{"model":"ours","thr":t_show,"top1":o1,"top2":o2,"top3":o3,"N":n_ext},
         {"model":"ours_sft","thr":t_show,"top1":s1,"top2":s2,"top3":s3,"N":n_ext}]
    ).to_csv(OUT_TOPN_SUMMARY, index=False)

    # 组装 Top-N 数据：两个簇，各三条
    topn_categories = ["Top-1","Top-2","Top-3"]
    topn_data = {
        "External": {
            "Top-1": (topn_ours_ext[t_show][0], topn_ours_sft_ext[t_show][0]),
            "Top-2": (topn_ours_ext[t_show][1], topn_ours_sft_ext[t_show][1]),
            "Top-3": (topn_ours_ext[t_show][2], topn_ours_sft_ext[t_show][2]),
        },
        "Internal": {
            "Top-1": (topn_ours_int[t_show][0], topn_ours_sft_int[t_show][0]),
            "Top-2": (topn_ours_int[t_show][1], topn_ours_sft_int[t_show][1]),
            "Top-3": (topn_ours_int[t_show][2], topn_ours_sft_int[t_show][2]),
        }
    }

    # ---------- C-index ----------
    df_ours_ext     = read_and_concat_csv([CSV_OURS_CG, CSV_OURS_MC])
    df_ours_sft_ext = read_and_concat_csv([CSV_SFT_CG,  CSV_SFT_MC])
    df_ours_int     = read_and_concat_csv([CSV_OURS_SEER])
    df_ours_sft_int = read_and_concat_csv([CSV_SFT_SEER])

    def cidx_from_df(df: pd.DataFrame) -> Tuple[float,int]:
        if df is None or df.empty: return float("nan"), 0
        t = df["time"].to_numpy(float)
        e = df["event"].to_numpy(int)
        s = df["predicted"].to_numpy(float)
        return float(concordance_index(t, e, s)), len(t)

    c_o_ext, n_o_ext = cidx_from_df(df_ours_ext)
    c_s_ext, n_s_ext = cidx_from_df(df_ours_sft_ext)
    c_o_int, n_o_int = cidx_from_df(df_ours_int)
    c_s_int, n_s_int = cidx_from_df(df_ours_sft_int)

    # 截尾 1/3/5 年
    c1_ext = capped_cindex_from_df(df_ours_ext, 12)
    c1_sft = capped_cindex_from_df(df_ours_sft_ext, 12)
    c3_ext = capped_cindex_from_df(df_ours_ext, 36)
    c3_sft = capped_cindex_from_df(df_ours_sft_ext, 36)
    c5_ext = capped_cindex_from_df(df_ours_ext, 60)
    c5_sft = capped_cindex_from_df(df_ours_sft_ext, 60)

    c1_int = capped_cindex_from_df(df_ours_int, 12)
    c1_sit = capped_cindex_from_df(df_ours_sft_int, 12)
    c3_int = capped_cindex_from_df(df_ours_int, 36)
    c3_sit = capped_cindex_from_df(df_ours_sft_int, 36)
    c5_int = capped_cindex_from_df(df_ours_int, 60)
    c5_sit = capped_cindex_from_df(df_ours_sft_int, 60)

    cidx_categories = ["Overall","1yr","3yr","5yr"]
    cidx_data = {
        "External": {
            "Overall": (c_o_ext, c_s_ext),
            "1yr":     (c1_ext, c1_sft),
            "3yr":     (c3_ext, c3_sft),
            "5yr":     (c5_ext, c5_sft),
        },
        "Internal": {
            "Overall": (c_o_int, c_s_int),
            "1yr":     (c1_int, c1_sit),
            "3yr":     (c3_int, c3_sit),
            "5yr":     (c5_int, c5_sit),
        }
    }

    # ===================== 只输出两张图 =====================
    os.makedirs(os.path.dirname(OUT_TOPN_GROUPED_FIG), exist_ok=True)

    # Figure 1: Top-N（External/ Internal 两簇；每簇三条）
    fig1, ax1 = plt.subplots(figsize=(11.5, 5.2))
    clustered_shared_bars(
        ax1,
        cohorts=["External","Internal"],
        categories=topn_categories,
        data=topn_data,
        ymax=1.0,
        cluster_gap=2.2,
        intra_gap=0.24,
        bar_width=0.38,
        bar_colors=BAR_COLORS,
        pos_cap_colors=POS_CAP_COLORS,
        neg_colors=NEG_COLORS,
        title=f"Top-N Accuracy (thr={0.5})",
        ylabel="Accuracy",
        legend_loc="upper left"
    )
    for sp in ["top","right"]:
        ax1.spines[sp].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(OUT_TOPN_GROUPED_FIG, dpi=300)
    plt.close(fig1)

    # Figure 2: C-index（External/ Internal 两簇；每簇四条）
    fig2, ax2 = plt.subplots(figsize=(12.5, 5.4))
    clustered_shared_bars(
        ax2,
        cohorts=["External","Internal"],
        categories=cidx_categories,
        data=cidx_data,
        ymax=0.83,
        cluster_gap=2.6,
        intra_gap=0.22,
        bar_width=0.34,
        bar_colors=BAR_COLORS,
        pos_cap_colors=POS_CAP_COLORS,
        neg_colors=NEG_COLORS,
        # title="C-index (Overall / 1yr / 3yr / 5yr)",
        title="",
        ylabel="C-index",
        legend_loc="upper left"
    )
    for sp in ["top","right"]:
        ax2.spines[sp].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(OUT_CINDEX_GROUPED_FIG, dpi=300)
    plt.close(fig2)

    # Dumps
    print("Saved:")
    print(f" - {OUT_TOPN_SUMMARY}")
    print(f" - {OUT_CINDEX_SUMMARY}")
    print(f" - {OUT_TOPN_GROUPED_FIG}")
    print(f" - {OUT_CINDEX_GROUPED_FIG}")

if __name__ == "__main__":
    main()
