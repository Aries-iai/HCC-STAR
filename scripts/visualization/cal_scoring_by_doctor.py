#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import argparse
import re, os, sys
from typing import List

# -------- 自定义颜色入口 --------
CUSTOM_COLORS: List[str] = []  # 可在此写死，如 ["#1f77b4","#d62728","#2ca02c"]
HEX_RE = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")

DEFAULT_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]

def build_palette(user_hex_csv: str | None, fallback: List[str] = None) -> List[str]:
    if fallback is None:
        fallback = DEFAULT_PALETTE
    # 优先级：命令行 --colors > 顶部 CUSTOM_COLORS > 默认
    if user_hex_csv and user_hex_csv.strip():
        cand = [c.strip() for c in user_hex_csv.split(",")]
        colors = [c for c in cand if HEX_RE.match(c)]
        if colors:
            return colors
    if CUSTOM_COLORS:
        colors = [c for c in CUSTOM_COLORS if HEX_RE.match(c)]
        if colors:
            return colors
    return fallback

def color_for_index(palette: List[str], idx: int) -> str:
    return palette[idx % len(palette)]

# ---------- 聚合 Top-N ----------
def load_and_aggregate_topn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["name"] = df["name"].astype(str)

    pretty = {
        "ours": "Ours",
        "gpt4o": "GPT-4o",
        "deepseek-r1": "DeepSeek-R1",
        "zr.jsonl": "Director Discussion Result",  # 主任讨论结果
    }

    rows = []
    # 模型
    for _, r in df[df["type"] == "MODEL"].iterrows():
        rows.append({
            "label": pretty.get(r["name"], r["name"]),
            "mean_topn1": r["mean_topn1"],
            "mean_topn2": r["mean_topn2"],
            "mean_topn3": r["mean_topn3"],
        })
    # 住院医师 zy_*
    zy = df[(df["type"] == "DOCTOR") & (df["name"].str.startswith("zy_"))]
    if len(zy):
        rows.append({
            "label": "Resident Physician",
            "mean_topn1": zy["mean_topn1"].mean(),
            "mean_topn2": zy["mean_topn2"].mean(),
            "mean_topn3": zy["mean_topn3"].mean(),
        })
    # 主治医生 zz_*
    zz = df[(df["type"] == "DOCTOR") & (df["name"].str.startswith("zz_"))]
    if len(zz):
        rows.append({
            "label": "Attending Physician",
            "mean_topn1": zz["mean_topn1"].mean(),
            "mean_topn2": zz["mean_topn2"].mean(),
            "mean_topn3": zz["mean_topn3"].mean(),
        })

    out = pd.DataFrame(rows)
    order = ["Ours","GPT-4o","DeepSeek-R1","Resident Physician","Attending Physician"]
    out["xorder"] = out["label"].apply(lambda x: order.index(x) if x in order else 999)
    return out.sort_values("xorder").drop(columns=["xorder"]).reset_index(drop=True)

# ---------- 读推理打分 ----------
def load_reason_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    name_map = {0: "Ours", 1: "GPT-4o", 2: "DeepSeek-R1"}
    df["model_name"] = df["model"].map(name_map).fillna(df["model"].astype(str))
    # 只保留 A/B/C，且计算总分(A+B+C)
    need_cols = ["patient_id","model","model_name","A_pts","B_pts","C_pts"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"reason_scores.csv 缺少列: {missing}")
    df["TOTAL_ABC"] = df["A_pts"].fillna(0) + df["B_pts"].fillna(0) + df["C_pts"].fillna(0)
    return df

# ---------- 逐分项“相同”标记（A/B/C/TOTAL 分别判定） ----------
TARGET_MODELS = {"Ours","GPT-4o","DeepSeek-R1"}

def build_equal_flags(df_reason: pd.DataFrame) -> pd.DataFrame:
    """
    返回含 per-patient 标记的 DataFrame：
    columns: patient_id, eq_A, eq_B, eq_C, eq_TOTAL
    判定规则：同一 patient_id 下，三模型都在场，且该分项（或 TOTAL_ABC）唯一值个数==1（dropna=False）
    """
    flags = []
    for pid, g in df_reason.groupby("patient_id", sort=False):
        g = g[g["model_name"].isin(TARGET_MODELS)]
        has_three = (g["model_name"].nunique() == 3)
        if not has_three:
            flags.append({"patient_id": pid, "eq_A": False, "eq_B": False, "eq_C": False, "eq_TOTAL": False})
            continue
        eq_A = g["A_pts"].nunique(dropna=False) == 1
        eq_B = g["B_pts"].nunique(dropna=False) == 1
        eq_C = g["C_pts"].nunique(dropna=False) == 1
        eq_TOTAL = g["TOTAL_ABC"].nunique(dropna=False) == 1
        flags.append({"patient_id": pid, "eq_A": eq_A, "eq_B": eq_B, "eq_C": eq_C, "eq_TOTAL": eq_TOTAL})
    return pd.DataFrame(flags)

# ---------- 画图 ----------
def plot_topn(df_topn: pd.DataFrame, out_svg: str, out_png: str, palette: List[str]):
    labels = df_topn["label"].tolist()
    metrics = ["mean_topn1","mean_topn2","mean_topn3"]
    series_labels = ["Top-1", "Top-2", "Top-3"]

    # 仅画前三个模型 (Ours, GPT-4o, DeepSeek-R1)
    x = np.arange(len(labels[0:3])) * 0.8
    width = 0.20
    plt.figure(figsize=(5, 5))

    bar_containers = []
    for i, (m, slabel) in enumerate(zip(metrics, series_labels)):
        color = color_for_index(palette, i)
        vals = df_topn[m].values[0:3].astype(float)
        # 若数据为0–1比例，转成百分制显示
        if np.nanmax(vals) <= 1.1:
            vals = vals * 100.0
        bars = plt.bar(
            x + (i - 1) * width, vals, width,
            label=slabel, color=color, edgecolor="white", linewidth=0.4
        )
        bar_containers.append((bars, vals))

    # 轴与标签
    plt.xticks(x[0:3], labels[0:3], rotation=0, ha="center")
    plt.yticks(np.arange(0, 110, 10))   # 纵轴刻度 0–100
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)")
    plt.legend(ncol=3, frameon=False, loc="upper center")
    plt.tight_layout()

    # —— 在柱顶添加数值 —— #
    for bars, vals in bar_containers:
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                plt.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + 1,  # 数值稍高于柱顶
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold"
                )

    plt.savefig(out_svg)
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_reason_parts(df: pd.DataFrame, flags_df: pd.DataFrame, out_svg: str, out_png: str, palette: List[str]):
    # 仅 A/B/C 的映射后均值（逐分项剔除相同病例）
    parts = ["A_pts","B_pts","C_pts"]
    labels = ["Completeness (A)","Correctness (B)","Safety & Contraindications (C)"]
    models = ["Ours","GPT-4o","DeepSeek-R1"]

    # 合并标记
    dfm = df.merge(flags_df, on="patient_id", how="left")
    dfm[["eq_A","eq_B","eq_C"]] = dfm[["eq_A","eq_B","eq_C"]].fillna(False)

    # 针对每个分项，屏蔽相同病例
    part_to_flag = {"A_pts":"eq_A", "B_pts":"eq_B", "C_pts":"eq_C"}
    means = {}
    for m in models:
        vals = []
        for p in parts:
            flag_col = part_to_flag[p]
            mask = ~(dfm[flag_col].astype(bool))  # True=保留, False=剔除该分项的病例
            series = pd.to_numeric(dfm.loc[(dfm["model_name"]==m) & mask, p], errors="coerce").dropna()
            vals.append(series.mean() if len(series) else np.nan)
        means[m] = vals

    # 画图
    x = np.arange(len(parts)) * 0.8
    width = 0.20
    plt.figure(figsize=(10,6))
    for i, m in enumerate(models):
        color = color_for_index(palette, i)
        plt.bar(x + (i-1)*width, means[m], width,
                label=m, color=color, edgecolor="white", linewidth=0.6)

    plt.xticks(x, labels, rotation=0, ha="center")
    plt.ylabel("Score (mean)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_svg)
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_total_reason_box(df: pd.DataFrame, flags_df: pd.DataFrame, out_svg: str, out_png: str, palette: List[str]):
    # fig C：仅剔除 TOTAL_ABC 三模型相同的病例
    models = ["Ours","GPT-4o","DeepSeek-R1"]
    dfm = df.merge(flags_df[["patient_id","eq_TOTAL"]], on="patient_id", how="left")
    dfm["eq_TOTAL"] = dfm["eq_TOTAL"].fillna(False)

    mask_keep = ~(dfm["eq_TOTAL"].astype(bool))
    data = [dfm.loc[(dfm["model_name"] == m) & mask_keep, "TOTAL_ABC"].dropna().values for m in models]

    colors = [color_for_index(palette, i) for i in range(len(models))]
    plt.figure(figsize=(8,6))
    bp = plt.boxplot(data, labels=models, showmeans=True, patch_artist=True)

    # 着色
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
    for whisk in bp['whiskers']:
        whisk.set_color("#333333"); whisk.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color("#333333"); cap.set_linewidth(1.0)
    for median in bp['medians']:
        median.set_color("#111111"); median.set_linewidth(1.2)
    if 'means' in bp:
        for mean in bp['means']:
            mean.set_marker('o'); mean.set_markerfacecolor("#ffffff")
            mean.set_markeredgecolor("#111111"); mean.set_markersize(5)

    plt.ylabel("Total Reasoning Score (A+B+C)")
    plt.tight_layout()
    plt.savefig(out_svg); plt.savefig(out_png, dpi=300); plt.close()

def main(topn_csv: str, reason_csv: str, out_dir: str = ".", colors_csv: str | None = None):
    palette = build_palette(colors_csv)

    # ---- Top-N 照旧 ----
    df_topn = load_and_aggregate_topn(topn_csv)
    os.makedirs(out_dir, exist_ok=True)
    df_topn.to_csv(f"{out_dir}/topn_aggregated_for_plot.csv", index=False)

    # ---- Reason 分数：加载 -> 构建“逐分项相同”标记 -> 出图 ----
    df_reason = load_reason_scores(reason_csv)
    df_reason.to_csv(f"{out_dir}/reason_scores_AtoC_only_raw.csv", index=False)

    # 构建标记
    flags_df = build_equal_flags(df_reason)
    flags_df.to_csv(f"{out_dir}/reason_equal_flags_per_patient.csv", index=False)

    # 也导出每一类被剔除的 patient_id 方便审计
    for col in ["eq_A","eq_B","eq_C","eq_TOTAL"]:
        ids = flags_df.loc[flags_df[col]==True, "patient_id"]
        ids.to_csv(f"{out_dir}/dropped_patient_ids_{col}.csv", index=False, header=["patient_id"])

    # 出图：fig a 用 topn；fig b/c 传入 flags_df 做“逐分项/total”剔除
    plot_topn(df_topn, f"{out_dir}/fig_a_topn_grouped.svg", f"{out_dir}/fig_a_topn_grouped.png", palette)
    plot_reason_parts(df_reason, flags_df, f"{out_dir}/fig_b_reason_parts_grouped.svg", f"{out_dir}/fig_b_reason_parts_grouped.png", palette)
    plot_total_reason_box(df_reason, flags_df, f"{out_dir}/fig_c_total_reason_boxplot.svg", f"{out_dir}/fig_c_total_reason_boxplot.png", palette)

    # 控制台统计
    print(f"[Info] patients total={df_reason['patient_id'].nunique()} | "
          f"drop_by_part: A={int(flags_df['eq_A'].sum())}, "
          f"B={int(flags_df['eq_B'].sum())}, "
          f"C={int(flags_df['eq_C'].sum())} | "
          f"drop_by_TOTAL={int(flags_df['eq_TOTAL'].sum())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Top-N & Reasoning scores with per-part skipping of equal cases")
    parser.add_argument("topn_csv", nargs="?", default="topn_scores.csv", help="Top-N 结果 CSV 路径")
    parser.add_argument("reason_csv", nargs="?", default="reason_scores.csv", help="Reasoning 结果 CSV 路径")
    parser.add_argument("out_dir", nargs="?", default="/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/f20", help="输出目录")
    parser.add_argument("--colors", type=str, default="", help="逗号分隔 HEX 色，如 \"#1f77b4,#d62728,#2ca02c\"")
    args = parser.parse_args()

    main(args.topn_csv, args.reason_csv, args.out_dir, colors_csv=args.colors)