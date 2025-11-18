# -*- coding: utf-8 -*-
"""
Paper-style C-index bar chart (Internal vs External) — Parallel + Values Dump
- Internal(seer) 对比: Our/TNM/BCLC/CNLC + XGB/SVM/Bayes/MLP + LLMs（可按需增减）
- External(cg_3/multicenter/late_stage) 对比: Our/TNM/BCLC/CNLC + LLMs（可按需增减）
- 自动汇总同一 cohort 的多个 CSV（按前缀+后缀命名）
- Overall + 1/2/3/4/5 年截尾任务；Overall 计算 Our vs TNM/BCLC/CNLC 的配对 bootstrap p 值
- 并行：对每个任务下各模型的 bootstrap 并行；p 值并行
- 输出:
    /share/home/.../visual_fin/cindex_bar_tasks_pval_internal.(svg|png|txt)
    /share/home/.../visual_fin/cindex_bar_tasks_pval_external.(svg|png|txt)

新增：
- 支持 --colors "#1f77b4,#d62728,#2ca02c,..." 指定所有模型的颜色顺序（覆盖默认固定配色）
"""

import os, time, math, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count

# ------- tqdm for progress (optional) -------
USE_PROGRESS = True
try:
    from tqdm import tqdm
except Exception:
    USE_PROGRESS = False
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

# ------------------------ Fast C-index (lifelines if available) ------------------------ #
_USE_LIFELINES = True
try:
    from lifelines.utils import concordance_index as _lifelines_cindex
except Exception:
    _USE_LIFELINES = False

def concordance_index(time_arr, event_arr, score_arr):
    """Compatible wrapper: lifelines if available, else O(n^2) Python."""
    time_arr = np.asarray(time_arr, float)
    event_arr = np.asarray(event_arr, int)
    score_arr = np.asarray(score_arr, float)
    if _USE_LIFELINES:
        return float(_lifelines_cindex(time_arr, score_arr, event_arr))
    conc = ties = perm = 0.0
    n = len(time_arr)
    for i in range(n - 1):
        ti, ei, si = time_arr[i], event_arr[i], score_arr[i]
        for j in range(i + 1, n):
            tj, ej, sj = time_arr[j], event_arr[j], score_arr[j]
            if ti == tj:
                continue
            if ti < tj and ei == 1:
                perm += 1
                if si > sj:
                    conc += 1
                elif si == sj:
                    ties += 1
            elif tj < ti and ej == 1:
                perm += 1
                if sj > si:
                    conc += 1
                elif si == sj:
                    ties += 1
    return (conc + 0.5 * ties) / perm if perm > 0 else np.nan

# ------------------------ Bootstrap helpers ------------------------ #
def make_boot_indices(n, n_boot=200, seed=2025):
    rng = np.random.default_rng(seed)
    return [rng.choice(n, size=n, replace=True) for _ in range(n_boot)]

def bootstrap_cindex_with_indices(t, e, s, boot_indices):
    vals = np.empty(len(boot_indices), float)
    for i, idx in enumerate(boot_indices):
        vals[i] = concordance_index(t[idx], e[idx], s[idx])
    mean_c = float(np.nanmean(vals))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return mean_c, float(lo), float(hi)

def paired_bootstrap_pvalue_with_indices(t, e, s1, s2, boot_indices):
    diffs = np.empty(len(boot_indices), float)
    for i, idx in enumerate(boot_indices):
        c1 = concordance_index(t[idx], e[idx], s1[idx])
        c2 = concordance_index(t[idx], e[idx], s2[idx])
        diffs[i] = c1 - c2
    # 双尾
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(max(p, 1.0 / len(boot_indices)), 1.0))

# ------------------------ Data utils ------------------------ #
def _read_csv(prefix: str, suffix: str):
    path = f"{prefix}_{suffix}.csv"
    if not os.path.exists(path):
        print(f'[MISS] {path}')
        return None
    try:
        df = pd.read_csv(path, usecols=["time", "event", "predicted"])
        df = df.dropna(subset=["time", "event", "predicted"])
        # 如需排除 time==1，取消下一行注释
        # df = df[df["time"].astype(float) != 1.0]
        if df.empty:
            return None
        return df
    except Exception as ex:
        print(f'[READ-ERR] {path}: {ex}')
        return None

def load_scores_from_prefixes(csv_prefix: str, suffixes, auto_flip=True):
    """读取并合并同一模型在多个后缀的数据：返回 (t, e, s)，若全缺返回 None"""
    dfs = []
    for suf in suffixes:
        d = _read_csv(csv_prefix, suf)
        if d is not None:
            dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    t = df["time"].to_numpy(float)
    e = df["event"].to_numpy(int)
    s = df["predicted"].to_numpy(float)
    if auto_flip:
        c0 = concordance_index(t, e, s)
        if np.isfinite(c0) and c0 < 0.5:
            s = -s
    return t, e, s

def cap_survival_data(t, e, s, horizon_months):
    t_new = np.minimum(t, horizon_months)
    e_new = ((e == 1) & (t <= horizon_months)).astype(int)
    return t_new, e_new, s

# ------------------------ Color parsing ------------------------ #
_HEX_RE = re.compile(r'^#?[0-9A-Fa-f]{6}$')

def parse_color_list(s: str):
    """
    将用户输入的颜色字符串解析成 hex 列表：
    - 支持逗号或空格分隔
    - 支持带/不带 '#'，统一返回带 '#'
    - 自动去重前后空格
    """
    if not s:
        return []
    parts = [p.strip() for p in re.split(r'[,\s]+', s) if p.strip()]
    colors = []
    for p in parts:
        if not _HEX_RE.match(p):
            raise ValueError(f"非法颜色值: {p}（需要 6 位十六进制，如 #1f77b4）")
        colors.append(p if p.startswith('#') else f'#{p}')
    return colors

def assign_colors(model_order, user_colors, fixed_colors, extra_palette):
    """
    返回与 model_order 等长的颜色列表：
    - 若 user_colors 非空：按顺序分配并循环复用（覆盖固定色）
    - 否则：使用固定映射 fixed_colors，其余按 extra_palette 依次取色
    """
    if user_colors:
        out = []
        k = 0
        for _ in model_order:
            out.append(user_colors[k % len(user_colors)])
            k += 1
        return out
    # 默认行为（原先逻辑）
    colors, k = [], 0
    for m in model_order:
        if m in fixed_colors:
            colors.append(fixed_colors[m])
        else:
            colors.append(extra_palette[k % len(extra_palette)])
            k += 1
    return colors

# ------------------------ Plot utils ------------------------ #
def ensure_dir(p):
    d = os.path.dirname(p)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)

def _auto_ylim_from_results(all_results, model_order, tasks,
                            floor=0.4, ceil=1.0, pad_frac=0.04):
    lows, highs = [], []
    for task_name, _ in tasks:
        for m in model_order:
            if m in all_results.get(task_name, {}):
                lo = all_results[task_name][m][1]
                hi = all_results[task_name][m][2]
                if np.isfinite(lo): lows.append(lo)
                if np.isfinite(hi): highs.append(hi)
    if not lows or not highs:
        return (floor, ceil)
    lo_raw, hi_raw = float(min(lows)), float(max(highs))
    span = max(hi_raw - lo_raw, 1e-6)
    pad = max(0.01, span * pad_frac)
    y_lo = max(floor, lo_raw - pad)
    y_hi = min(ceil, hi_raw + pad)
    if y_hi - y_lo < 0.1:
        y_lo = max(floor, y_hi - 0.1)
    return y_lo, y_hi

# ------------------------ Parallel runners ------------------------ #
def compute_model_for_task(model_name, arrs, horizon, n_boot, boot_cache):
    """单模型在某 task（horizon）下的 bootstrap 结果。"""
    t, e, s = arrs
    if horizon is not None and horizon > 0:
        t_, e_, s_ = cap_survival_data(t, e, s, horizon)
    else:
        t_, e_, s_ = t, e, s
    n = len(t_)
    if n <= 1:
        return model_name, (np.nan, np.nan, np.nan, (t_, e_, s_))
    if n not in boot_cache:
        boot_cache[n] = make_boot_indices(n, n_boot=n_boot, seed=2025)
    mean_c, lo, hi = bootstrap_cindex_with_indices(t_, e_, s_, boot_cache[n])
    return model_name, (mean_c, lo, hi, (t_, e_, s_))

def compute_p_for_comp(comp_model, our_arrs, comp_arrs, n_boot, boot_cache_p):
    """Overall 下 Our vs comp 的配对 p 值（截尾前数据）。"""
    t0, e0, s0 = our_arrs
    t1, e1, s1 = comp_arrs
    m = min(len(t0), len(t1))
    if m < 10:
        return comp_model, np.nan
    t0, e0, s0 = t0[:m], e0[:m], s0[:m]
    t1, e1, s1 = t1[:m], e1[:m], s1[:m]
    if m not in boot_cache_p:
        boot_cache_p[m] = make_boot_indices(m, n_boot=n_boot, seed=2026)
    p = paired_bootstrap_pvalue_with_indices(t0, e0, s0, s1, boot_cache_p[m])
    return comp_model, p

# ------------------------ Cohort runner (Parallel) ------------------------ #
def run_one_cohort(
    cohort_name: str,
    model_spec: dict,
    suffixes: list,
    n_boot: int = 200,
    pval_only_overall: bool = True,
    auto_flip: bool = True,
    horizons=(None, 12, 24, 36, 48, 60),
    out_prefix=None,
    n_jobs: int = -1,   # 并行核数；-1=全部
    user_colors=None,   # <<<<<< 新增：用户传入颜色（list[str] 或 None）
):
    # ---------- 读取数据 ----------
    base_data = {}
    for disp_name, csv_prefix in model_spec.items():
        arrs = load_scores_from_prefixes(csv_prefix, suffixes, auto_flip=auto_flip)
        if arrs is None:
            print(f"[INFO] {cohort_name}: 缺少 {csv_prefix}_*.csv，已跳过 {disp_name}")
            continue
        base_data[disp_name] = arrs
    if not base_data:
        print(f"[ERROR] {cohort_name}: 无可用数据，终止该 cohort。")
        return

    labels_models = list(base_data.keys())
    tasks = [("Overall", None)] + [(f"{int(h/12)} yr", h) for h in horizons if (h is not None and h > 0)]

    if not _USE_LIFELINES:
        print("[WARN] lifelines 未安装，使用 O(n^2) 纯 Python C-index（较慢）")

    # ---------- 结果容器 ----------
    all_results = {t: {} for t, _ in tasks}
    all_pvals   = {t: {} for t, _ in tasks}

    # 并行参数
    if n_jobs in (0, 1):
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, cpu_count() + 1 + n_jobs)

    for (task_name, horizon) in (tqdm(tasks, desc=f"Tasks[{cohort_name}]") if USE_PROGRESS else tasks):
        boot_cache = {}
        iter_models = labels_models
        if USE_PROGRESS:
            print(f"[RUN] {cohort_name} @ {task_name} — models: {len(iter_models)}, jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(compute_model_for_task)(m, base_data[m], horizon, n_boot, boot_cache)
            for m in iter_models
        )
        for m, val in results:
            all_results[task_name][m] = val

        # p 值：只 Overall（或全任务），并行 Our vs TNM/BCLC/CNLC
        do_pval = (task_name == "Overall") if pval_only_overall else True
        comps = [m for m in ["TNM", "BCLC", "CNLC"] if m in base_data]
        if do_pval and ("Our model" in base_data) and comps:
            boot_cache_p = {}
            p_list = Parallel(n_jobs=min(n_jobs, len(comps)), backend="loky")(
                delayed(compute_p_for_comp)(comp, base_data["Our model"], base_data[comp], n_boot, boot_cache_p)
                for comp in comps
            )
            for comp, p in p_list:
                all_pvals[task_name][comp] = p

    # ---------- 颜色 & 顺序 ----------
    fixed_colors = {"Our model": "#1f77b4", "TNM": "#ff7f0e", "BCLC": "#2ca02c", "CNLC": "#d62728"}
    extra_palette = [
        "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD",
        "#E24A33", "#348ABD", "#988ED5", "#8EBA42", "#E5A11A", "#777777"
    ]
    model_order = [m for m in ["Our model", "TNM", "BCLC", "CNLC"] if m in labels_models] + \
                  [m for m in labels_models if m not in {"Our model","TNM","BCLC","CNLC"}]

    # <<<<<< 关键改动：根据用户输入决定颜色序列 >>>>>>
    colors = assign_colors(model_order, user_colors, fixed_colors, extra_palette)

    # ---------- 画图 ----------
    n_tasks, n_models = len(tasks), len(model_order)
    width = 0.8 / max(n_models, 1)
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for j, model in enumerate(model_order):
        means = [all_results[task][model][0] for task, _ in tasks]
        ci_l  = [all_results[task][model][1] for task, _ in tasks]
        ci_h  = [all_results[task][model][2] for task, _ in tasks]
        yerr  = np.vstack([np.array(means) - np.array(ci_l),
                           np.array(ci_h) - np.array(means)])
        ax.bar(x + (j - n_models/2 + 0.5) * width,
               means, width,
               yerr=yerr,
               color=colors[j], label=model,
               error_kw=dict(elinewidth=1.3, capsize=4, capthick=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels([task for task, _ in tasks],fontsize=14)
    ax.set_ylabel("C-index",fontsize=14)
    # ax.set_title(f"C-index across tasks · {cohort_name}", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---------- 自适应 y 轴 + p 值括号 ----------
    y_lo, y_hi = _auto_ylim_from_results(all_results, model_order, tasks, floor=0.45, ceil=1.02, pad_frac=0.04)
    uppers = []
    for task_name, _ in tasks:
        for m in model_order:
            if m in all_results[task_name]:
                uppers.append(all_results[task_name][m][2])
    y_top_data = max(uppers) if uppers else 0.8
    ax.set_ylim(y_lo, y_hi)
    span_final = max(1e-6, y_hi - y_lo)

    def _annotate(ax, x1, x2, y, text, h):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                linewidth=1.2, color="#444444", clip_on=False)
        ax.text((x1 + x2) / 2, y + h * 1.05, text,
                ha="center", va="bottom", fontsize=9, clip_on=False)

    try:
        i_overall = [i for i, (name, _) in enumerate(tasks) if name == "Overall"][0]
        comps_overall = [m for m in ["TNM","BCLC","CNLC"] if m in model_order and m in all_pvals.get("Overall", {})]
        need_p = ("Overall" in all_pvals) and ("Our model" in model_order) and (len(comps_overall) > 0)
        if need_p:
            bracket_h = 0.018 * span_final
            step = 0.045 * span_final
            base_y = min(y_hi - 0.02*span_final, y_top_data + 0.010 * span_final)
            x_our = x[i_overall] + (model_order.index("Our model") - n_models/2 + 0.5) * width
            for k, comp in enumerate(comps_overall):
                p = all_pvals["Overall"][comp]
                if np.isnan(p): 
                    continue
                text = "P < 1e-4" if p < 1e-4 else f"P = {p:.3f}"
                x_comp = x[i_overall] + (model_order.index(comp) - n_models/2 + 0.5) * width
                _annotate(ax, x_our, x_comp, base_y + step * k, text, h=bracket_h)
    except Exception:
        pass

    # ---------- 图例 ----------
    if len(model_order) > 7:
        # ax.legend(title="Models", loc="upper left", bbox_to_anchor=(1.01, 1.0),
        #           ncol=1, frameon=False, borderaxespad=0.0)
        # plt.subplots_adjust(right=0.82)

        ax.legend(title="", loc="upper right",
          bbox_to_anchor=(1.0, 1.06), bbox_transform=ax.transAxes,
          ncol=1, frameon=False, borderaxespad=0.2,fontsize=10)
        # 不再需要挤右边留白
        plt.subplots_adjust(right=0.98)
    else:
        ncol = 2 if len(model_order) <= 6 else 3
        ax.legend(title="", loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=ncol, frameon=False,fontsize=10)
        plt.subplots_adjust(top=0.88)

    # ---------- 保存（图 & 数值） ----------
    if out_prefix is None:
        out_prefix = f"cindex_bar_tasks_pval_{cohort_name.lower()}"
    out_dir = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2"
    os.makedirs(out_dir, exist_ok=True)
    out_svg = os.path.join(out_dir, f"{out_prefix}.svg")
    out_png = os.path.join(out_dir, f"{out_prefix}.png")
    out_txt = os.path.join(out_dir, f"{out_prefix}.txt")

    fig.tight_layout()
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVED] {cohort_name}: {out_svg} / .png")

    # ---- 写出数值文件（C-index & CI & p）----
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"# C-index Results · {cohort_name}\n")
        f.write("# Mean [95% CI]\n\n")
        for (task_name, _) in tasks:
            f.write(f"## Task: {task_name}\n")
            for m in model_order:
                if m not in all_results[task_name]: 
                    continue
                mean_c, lo, hi, _ = all_results[task_name][m]
                if np.isnan(mean_c):
                    f.write(f"- {m}: NaN\n")
                else:
                    f.write(f"- {m}: {mean_c:.4f} [{lo:.4f}, {hi:.4f}]\n")
            if task_name in all_pvals and all_pvals[task_name]:
                f.write("  * P-values (Our vs comparator):\n")
                for comp in ["TNM","BCLC","CNLC"]:
                    if comp in all_pvals[task_name]:
                        p = all_pvals[task_name][comp]
                        if np.isnan(p):
                            f.write(f"    - Our vs {comp}: NaN\n")
                        else:
                            ptext = "< 1e-4" if p < 1e-4 else f"= {p:.4g}"
                            f.write(f"    - Our vs {comp}: P {ptext}\n")
            f.write("\n")
    print(f"[SAVED] values: {out_txt}")

# ------------------------ Main ------------------------ #
def main():
    parser = argparse.ArgumentParser(description="C-index bar plot with optional custom colors.")
    parser.add_argument("--colors", type=str, default="", help="用逗号或空格分隔的十六进制颜色，如: \"#1f77b4,#d62728,#2ca02c\"")
    args = parser.parse_args()

    # 解析用户颜色（若提供）
    user_colors = parse_color_list(args.colors) if args.colors else None

    # ====== 参数区 ======
    INTERNAL_SUFFIXES = ["seer"]
    EXTERNAL_SUFFIXES = ["cg_3", "multicenter", "late_stage"]

    MODELS_INTERNAL = {
        "Our model": "df_ours",
        "TNM": "df_tnm",
        "BCLC": "df_bclc",
        "CNLC": "df_cnlc",
        "XGB": "df_xgb",
        "SVM": "df_svm",
        "Bayes": "df_bayes",
        "MLP": "df_mlp",
        "GPT-4o": "df_gpt-4o-2024-08-06",
        "GEMINI-2.5-pro": "df_gemini-2_5-pro",
    }
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

    N_BOOT = 200
    PVAL_ONLY_OVERALL = True
    AUTO_FLIP = True
    # HORIZONS = (None, 12, 24, 36, 48, 60)  # None=Overall，其余为月
    HORIZONS = (None, 12, 36, 60)  # None=Overall，其余为月
    N_JOBS = -1  # -1 使用全部核；可改为具体数字

    # ====== 跑 Internal（含 ML + LLMs） ======
    run_one_cohort(
        cohort_name="Internal",
        model_spec=MODELS_INTERNAL,
        suffixes=INTERNAL_SUFFIXES,
        n_boot=N_BOOT,
        pval_only_overall=PVAL_ONLY_OVERALL,
        auto_flip=AUTO_FLIP,
        horizons=HORIZONS,
        out_prefix="cindex_bar_tasks_pval_internal_135",
        n_jobs=N_JOBS,
        user_colors=user_colors,   # <<<<<< 传入
    )

    # ====== 跑 External（含 LLMs） ======
    run_one_cohort(
        cohort_name="External",
        model_spec=MODELS_EXTERNAL,
        suffixes=EXTERNAL_SUFFIXES,
        n_boot=N_BOOT,
        pval_only_overall=PVAL_ONLY_OVERALL,
        auto_flip=AUTO_FLIP,
        horizons=HORIZONS,
        out_prefix="cindex_bar_tasks_pval_external_135",
        n_jobs=N_JOBS,
        user_colors=user_colors,   # <<<<<< 传入
    )

if __name__ == "__main__":
    main()
