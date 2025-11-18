#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- 配置区 ----------------------- #
BASE_DIR = "."  # CSV 所在目录

# 只在 Internal(seer) 加上 ML 方法
MODELS_INTERNAL = {
    "Our model": "df_ours",
    "TNM": "df_tnm",
    "BCLC": "df_bclc",
    "CNLC": "df_cnlc",
    # "XGB": "df_xgb",
    # "SVM": "df_svm",
    # "Bayes": "df_bayes",
    # "MLP": "df_mlp",
    # "GPT-4o": "df_gpt-4o-2024-08-06",
    # "GEMINI-2.5-pro": "df_gemini-2_5-pro",
}

# External 只保留四条（如需对比 LLM 可解注释）
MODELS_EXTERNAL = {
    "Our model": "df_ours",
    "TNM": "df_tnm",
    "BCLC": "df_bclc",
    "CNLC": "df_cnlc",
    # "CLAUDE": "df_claude-3-5-sonnet-20241022",
    # "DEEPSEEK-r1": "df_deepseek-r1",
    # "GPT-4o": "df_gpt-4o-2024-08-06",
    # "GEMINI-2.5-pro": "df_gemini-2_5-pro",
    # "GPT-5": "df_gpt-5",
}

INTERNAL_SUFFIXES = ["seer"]
# EXTERNAL_SUFFIXES = ["cg_3", "multicenter", "late_stage"]
EXTERNAL_SUFFIXES = ["cg_3", "multicenter"]
HORIZONS = [12.0, 24.0, 36.0, 48.0, 60.0]  # 单位：月

# 你可以在这里直接填颜色（方式 1）
# 例如：CUSTOM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
CUSTOM_COLORS = []  # 留空则使用默认色；或用命令行 --colors 传（方式 2）
# ---------------------------------------------------- #

# -------------------- 默认颜色（兜底方案） -------------------- #
DEFAULT_COLOR_MAP = {
    "Our model": "#1f77b4",   # 蓝
    "TNM":       "#d62728",   # 红
    "BCLC":      "#2ca02c",   # 绿
    "CNLC":      "#9467bd",   # 紫
    "XGB":       "#ff7f0e",   # 橙
    "SVM":       "#17becf",   # 青
    "Bayes":     "#8c564b",   # 棕
    "MLP":       "#e377c2",   # 粉
    "GPT-4o":    "#7f7f7f",   # 灰
    "GEMINI-2.5-pro": "#bcbd22", # 橄榄
    "CLAUDE":    "#000000",   # 黑
    "DEEPSEEK-r1": "#98df8a", # 浅绿
    "GPT-5":     "#ff9896",   # 浅红
}
FALLBACK_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#6b6ecf", "#9c9ede",
    "#8ca252", "#b5cf6b", "#cedb9c", "#e7ba52", "#e7969c", "#ad494a"
]

PLOT_ORDER_INTERNAL = ["Our model","TNM","BCLC","CNLC","XGB","SVM","Bayes","MLP","GPT-4o","GEMINI-2.5-pro"]
PLOT_ORDER_EXTERNAL = ["Our model","TNM","BCLC","CNLC","CLAUDE","DEEPSEEK-r1","GPT-4o","GEMINI-2.5-pro","GPT-5"]

# 合并顺序，保证跨图一致着色（去重保序）
ALL_MODEL_ORDER = list(dict.fromkeys(PLOT_ORDER_INTERNAL + PLOT_ORDER_EXTERNAL))

HEX_RE = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")

def build_color_map(custom_hex_list):
    """
    根据用户的 hex 列表按 ALL_MODEL_ORDER 依次分配颜色。
    - 若 custom_hex_list 为空或 None：优先使用 DEFAULT_COLOR_MAP；缺失项用 FALLBACK_PALETTE 按名称哈希兜底。
    - 若 custom_hex_list 提供：循环复用并按 ALL_MODEL_ORDER 分配，确保同名模型在各图颜色一致。
    """
    if not custom_hex_list:
        # 默认策略：先用 DEFAULT_COLOR_MAP；无定义的按哈希从 FALLBACK 取
        cmap = {}
        for name in ALL_MODEL_ORDER:
            if name in DEFAULT_COLOR_MAP:
                cmap[name] = DEFAULT_COLOR_MAP[name]
            else:
                idx = abs(hash(name)) % len(FALLBACK_PALETTE)
                cmap[name] = FALLBACK_PALETTE[idx]
        return cmap

    # 清洗与校验 hex
    clean_list = []
    for h in custom_hex_list:
        h = h.strip()
        if HEX_RE.match(h):
            clean_list.append(h)
    if not clean_list:
        # 传了但都不合法 → 回退默认
        return build_color_map(None)

    # 轮转分配
    cmap = {}
    m = len(clean_list)
    for i, name in enumerate(ALL_MODEL_ORDER):
        cmap[name] = clean_list[i % m]
    return cmap

# ===== 你的核心函数（保持一致） =====
def km_censoring_survival(time, event):
    t = np.asarray(time, float); e = np.asarray(event, int)
    uniq = np.unique(t)
    n = np.array([(t >= ti).sum() for ti in uniq], dtype=float)
    d_c = np.array([((t == ti) & (e == 0)).sum() for ti in uniq], dtype=float)
    surv = np.ones_like(uniq, dtype=float); prod = 1.0
    for i, (nk, dk) in enumerate(zip(n, d_c)):
        if nk > 0: prod *= (1.0 - dk / nk)
        surv[i] = prod
    def G(q):
        q = np.asarray(q, float); out = np.ones_like(q, float)
        for i, qi in enumerate(q):
            idx = np.searchsorted(uniq, qi, side='right') - 1
            out[i] = 1.0 if idx < 0 else surv[idx]
        return out
    return G

def c_index(time, event, score):
    t = np.asarray(time, float); e = np.asarray(event, int); s = np.asarray(score, float)
    conc = ties = perm = 0.0; n = len(t)
    for i in range(n-1):
        for j in range(i+1, n):
            if t[i] == t[j]: continue
            if t[i] < t[j] and e[i]==1:
                perm += 1; conc += (s[i] > s[j]); ties += (s[i] == s[j])
            elif t[j] < t[i] and e[j]==1:
                perm += 1; conc += (s[j] > s[i]); ties += (s[i] == s[j])
    return (conc + 0.5*ties)/perm if perm>0 else np.nan

def auto_orient_scores(time, event, score):
    return (-score if c_index(time, event, score) < 0.5 else score)

def tdroc_ipcw(time, event, score, t0):
    t = np.asarray(time, float); e = np.asarray(event, int); s = np.asarray(score, float)
    G = km_censoring_survival(t, e); eps = 1e-8
    cases = (e == 1) & (t <= t0)   # cumulative/dynamic positives
    ctrls = (t > t0)               # survivors past t0
    if cases.sum() < 3 or ctrls.sum() < 3:
        return np.array([0,1]), np.array([0,1]), np.nan
    w_cases = 1.0 / np.maximum(G(t[cases]), eps)
    w_ctrls = 1.0 / np.maximum(G(np.array([t0]*ctrls.sum())), eps)
    s_cases, s_ctrls = s[cases], s[ctrls]
    thr = np.unique(np.concatenate([s_cases, s_ctrls]))
    thr = np.r_[np.inf, thr[::-1], -np.inf]   # strictly descending
    Wc, Wn = w_cases.sum(), w_ctrls.sum()
    tpr = [(w_cases*(s_cases>th)).sum()/Wc for th in thr]
    fpr = [(w_ctrls*(s_ctrls>th)).sum()/Wn for th in thr]
    tpr, fpr = np.asarray(tpr), np.asarray(fpr)
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc
# =================================================== #

def _read_one_csv(prefix: str, suffix: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, f"{prefix}_{suffix}.csv")
    if not os.path.exists(path):
        print(f"[WARN] 缺失：{path} —— 跳过")
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=["time", "event", "predicted"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df = df.dropna(subset=["time", "event", "predicted"]).copy()
    return df

def _load_cohort_arrays(prefix: str, suffixes: list[str]):
    dfs = []
    for suff in suffixes:
        dfi = _read_one_csv(prefix, suff)
        if not dfi.empty:
            dfs.append(dfi)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    t = df["time"].to_numpy(float)
    e = df["event"].to_numpy(int)
    s = df["predicted"].to_numpy(float)
    s = auto_orient_scores(t, e, s)
    return t, e, s

def _gather_data_for_cohort(cohort_name: str, suffixes: list[str], models: dict[str, str]):
    out = {}
    for model_name, prefix in models.items():
        arrs = _load_cohort_arrays(prefix, suffixes)
        if arrs is None:
            print(f"[INFO] {cohort_name}: {model_name} 无可用数据，已跳过")
            continue
        out[model_name] = arrs
        t, e, s = arrs
        print(f"{cohort_name:<8} | {model_name:<12} N={len(t):<5} Events={int(e.sum()):<5}")
    return out

def _plot_tdroc_for_cohort(cohort_name: str, data: dict, horizons: list[float], plot_order: list[str], color_map: dict):
    if not data:
        print(f"[WARN] {cohort_name} 无数据，跳过绘图")
        return
    for H in horizons:
        plt.figure(figsize=(6.6, 5.6))
        any_plotted = False
        for model_name in plot_order:
            if model_name not in data:
                continue
            t, e, s = data[model_name]
            fpr, tpr, auc = tdroc_ipcw(t, e, s, t0=H)
            label = f"{model_name} (AUC={auc:.3f})" if not np.isnan(auc) else f"{model_name} (AUC=NA)"
            plt.plot(
                fpr, tpr,
                label=label,
                linewidth=2.2,
                alpha=0.95,
                color=color_map.get(model_name, "#444444")
            )
            any_plotted = True

        plt.plot([0,1], [0,1], linestyle="--", linewidth=1.2, alpha=0.8, color="#9ca3af")
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        # plt.title(f"{cohort_name} · Time-dependent ROC at {int(H)} months")
        plt.legend(loc="lower right", frameon=False)
        plt.tight_layout()

        out_svg = f"/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2/tdROC_{cohort_name.lower()}_{int(H)}m.svg"
        out_png = f"/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2/tdROC_{cohort_name.lower()}_{int(H)}m.png"
        if any_plotted:
            plt.savefig(out_svg)
            plt.savefig(out_png, dpi=300)
            print(f"[OK] 保存 {cohort_name} ROC：{out_svg} | {out_png}")
        plt.close()

def parse_args():
    ap = argparse.ArgumentParser(description="Time-dependent ROC with customizable colors")
    ap.add_argument("--base_dir", type=str, default=BASE_DIR, help="CSV 目录（默认当前）")
    ap.add_argument("--colors", type=str, default="", help="逗号分隔的 16 进制颜色列表，如 '#1f77b4,#d62728,#2ca02c'")
    return ap.parse_args()

def main():
    args = parse_args()
    global BASE_DIR
    BASE_DIR = args.base_dir

    # 颜色来源优先级：命令行 --colors > 顶部 CUSTOM_COLORS > 默认
    color_list = []
    if args.colors.strip():
        color_list = [s.strip() for s in args.colors.split(",") if s.strip()]
    elif CUSTOM_COLORS:
        color_list = CUSTOM_COLORS

    color_map = build_color_map(color_list)

    print("========== 加载数据并计算基础指标 ==========")
    data_internal = _gather_data_for_cohort("Internal", INTERNAL_SUFFIXES, MODELS_INTERNAL)
    data_external = _gather_data_for_cohort("External", EXTERNAL_SUFFIXES, MODELS_EXTERNAL)

    print("\n========== 绘制 ROC（Internal，含 ML） ==========")
    _plot_tdroc_for_cohort(
        "Internal",
        data_internal,
        HORIZONS,
        plot_order=PLOT_ORDER_INTERNAL,
        color_map=color_map
    )

    print("\n========== 绘制 ROC（External，仅四条） ==========")
    _plot_tdroc_for_cohort(
        "External",
        data_external,
        HORIZONS,
        plot_order=PLOT_ORDER_EXTERNAL,
        color_map=color_map
    )

if __name__ == "__main__":
    main()
