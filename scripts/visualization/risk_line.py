import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ------------------ 文件路径配置 ------------------
ECDF_PATH = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/ecdf_train.pkl"
DATA_PATHS = {
    # "The training cohort": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_multicenter.jsonl",
    "The internal validation cohort": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_seer.jsonl",
    "The external testing cohort": [
        "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_cg_3.jsonl",
        # "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_late_stage_chungggeng.jsonl",
        "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_multicenter.jsonl",
    ],
}
PNG_PATH = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin/survival_risk_groups_full_subplot.png"
SVG_PATH = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin/survival_risk_groups_full_subplot.svg"

# ------------------ 工具函数 ------------------
def read_jsonl(path):
    if isinstance(path, str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif isinstance(path, list):
        res = []
        for ph in path:
            with open(ph, "r", encoding="utf-8") as f:
                res.extend([json.loads(line) for line in f if line.strip()])
        return res
    else:
        print(f'[Error] cannot parse jsonl {path}')
        return []

def load_ecdf(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_cdf_values(ecdf_obj, t_array):
    # t_array 是 predicted_survival（越大=越低风险）
    sorted_times = np.asarray(ecdf_obj["sorted_times"])
    n = ecdf_obj["n"]
    return np.searchsorted(sorted_times, t_array, side='right') / n

def assign_risk_group(cdf_vals):
    # 高风险=最低四分位；低风险=最高四分位；其余为中等风险
    return np.where(cdf_vals < 0.25, 'High-risk',
           np.where(cdf_vals >= 0.75, 'Low-risk', 'Mid'))

# ---------- 计算 2 年 PFS（24 月） ----------
def compute_two_year_pfs_for_cohort(df_records, ecdf_obj, t_months=24.0):
    """
    输入：该 cohort 的原始记录（list[dict]），以及 ecdf 对象
    输出：{'Low-risk': p_low, 'High-risk': p_high} （0-1 概率）
    """
    df = pd.DataFrame(df_records)
    # 依据训练集 ecdf 计算分组（与绘图一致）
    cdf_vals = compute_cdf_values(ecdf_obj, df["predicted_survival"].values)
    df["risk_group"] = assign_risk_group(cdf_vals)
    df = df[df["risk_group"].isin(["Low-risk", "High-risk"])].copy()

    kmf = KaplanMeierFitter()
    res = {}
    for grp in ["Low-risk", "High-risk"]:
        m = df["risk_group"] == grp
        if m.sum() == 0:
            res[grp] = np.nan
            continue
        kmf.fit(df.loc[m, "survival_time"], df.loc[m, "event"], label=grp)
        # lifelines：在给定时间点的生存概率估计
        # 兼容不同版本 lifelines：predict(t) 或 survival_function_at_times([t])
        try:
            p = float(kmf.predict(t_months))
        except Exception:
            p = float(kmf.survival_function_at_times([t_months]).values.squeeze())
        res[grp] = p
    return res

# ---------------- 绘图函数 ----------------
def plot_risk_survival_full(data_dict, ecdf_obj, png_path, svg_path):
    # 动态确定行数（按传入 cohort 数）
    n_rows = len(data_dict)
    fig = plt.figure(figsize=(12, 3* n_rows + 4))
    outer_grid = GridSpec(n_rows, 1, height_ratios=[1.8]*n_rows, hspace=0.2)
    colors = {"High-risk": "orangered", "Low-risk": "steelblue"}

    for idx, (cohort_name, df_raw) in enumerate(data_dict.items()):
        df = pd.DataFrame(df_raw)
        cdf_vals = compute_cdf_values(ecdf_obj, df["predicted_survival"].values)
        df["risk_group"] = assign_risk_group(cdf_vals)
        df = df[df["risk_group"].isin(["Low-risk", "High-risk"])].copy()

        time_max = int(df["survival_time"].max()) + 1
        # 以 12 个月为间隔显示 Number at risk
        time_points = np.arange(0, max(time_max, 25), 12)

        outer_subgrid = GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_grid[idx], height_ratios=[2.5, 0.7, 0.5], hspace=0.1
        )
        ax_km = fig.add_subplot(outer_subgrid[0])
        ax_risk = fig.add_subplot(outer_subgrid[1], sharex=ax_km)
        ax_censor = fig.add_subplot(outer_subgrid[2], sharex=ax_km)

        # KM 曲线
        kmf = KaplanMeierFitter()
        for group in ["Low-risk", "High-risk"]:
            mask = df["risk_group"] == group
            if mask.sum() == 0:
                continue
            kmf.fit(df.loc[mask, "survival_time"], df.loc[mask, "event"], label=group)
            kmf.plot(ax=ax_km, ci_show=True, color=colors[group], linewidth=2)

        # log-rank p 值与一个简单 HR（事件率 / 总随访时间）
        ix1 = df["risk_group"] == "Low-risk"
        ix2 = df["risk_group"] == "High-risk"
        pval = np.nan
        hr = np.nan
        if ix1.sum() > 0 and ix2.sum() > 0:
            result = logrank_test(
                df.loc[ix1, "survival_time"], df.loc[ix2, "survival_time"],
                df.loc[ix1, "event"], df.loc[ix2, "event"]
            )
            pval = result.p_value
            # 简易 HR 近似（不是 Cox HR，仅用于图上参考）
            hr = (np.sum(df.loc[ix2, "event"]) / (np.sum(df.loc[ix2, "survival_time"]) + 1e-12)) / \
                 (np.sum(df.loc[ix1, "event"]) / (np.sum(df.loc[ix1, "survival_time"]) + 1e-12))

        # KM 图设定
        print(f"{chr(97+idx)}. {cohort_name}")
        if idx == 0:
            ax_km.set_title(f"a. The internal test cohort", loc="left", fontsize=12, weight="bold")
        else:
            ax_km.set_title(f"b. The external test cohort", loc="left", fontsize=12, weight="bold")
        # ax_km.set_title(f"{chr(97+idx)}. {cohort_name}", loc="left", fontsize=12, weight="bold")
        ax_km.set_ylabel("Progression-free survival", fontsize=9)
        ax_km.legend(loc="upper right")
        ax_km.grid(True, linestyle="--", alpha=0.3)
        ax_km.text(0.04, 0.05, f"p = {pval:.2e}" + (f"\nHR = {hr:.2f}" if np.isfinite(hr) else ""),
                   transform=ax_km.transAxes, fontsize=9)

        # Number at risk
        for i, group in enumerate(["Low-risk", "High-risk"]):
            mask = df["risk_group"] == group
            if mask.sum() == 0:
                continue
            n_at_risk = [int(np.sum((df.loc[mask, "survival_time"] >= t))) for t in time_points]
            for x, y in zip(time_points, n_at_risk):
                ax_risk.text(x, 1.4 - 0.5 * i, str(y), ha="center", color=colors[group], fontsize=9)
        ax_risk.set_yticks([1.4, 0.9])
        ax_risk.set_yticklabels(["Low-risk", "High-risk"], fontsize=9)
        ax_risk.set_xticks(time_points)
        ax_risk.set_ylabel("Number\nat risk", fontsize=9)
        ax_risk.set_ylim(0.5, 1.8)

        # Censoring 条纹图
        for group, color in colors.items():
            mask = (df["risk_group"] == group) & (df["event"] == 0)
            for t in df.loc[mask, "survival_time"]:
                ax_censor.axvline(t, color=color, lw=0.7, alpha=0.7)
        ax_censor.set_yticks([])
        ax_censor.set_ylabel("n.censor", fontsize=9)
        ax_censor.set_xlabel("Time (Months)", fontsize=11)
        ax_censor.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.show()

# ---------------- 主调用 ----------------
if __name__ == "__main__":
    ecdf = load_ecdf(ECDF_PATH)
    data_dict = {name: read_jsonl(path) for name, path in DATA_PATHS.items()}
    # 绘图
    plot_risk_survival_full(data_dict, ecdf, png_path=PNG_PATH, svg_path=SVG_PATH)

    # ======== 计算并打印 2 年 PFS（24 月） ========
    # 识别“internal”与“external”两个 cohort（名称里大小写不敏感匹配）
    internal_key = None
    external_key = None
    for k in data_dict.keys():
        lk = k.lower()
        if "internal" in lk and internal_key is None:
            internal_key = k
        if ("external" in lk or "testing" in lk or "validation" in lk) and external_key is None and "internal" not in lk:
            external_key = k

    # 若找不到，回退到插入顺序的前两个
    keys = list(data_dict.keys())
    if internal_key is None and len(keys) >= 1:
        internal_key = keys[0]
    if external_key is None and len(keys) >= 2:
        external_key = keys[1]

    # 计算 2 年 PFS
    two_year_internal = compute_two_year_pfs_for_cohort(data_dict[internal_key], ecdf, t_months=24.0)
    two_year_external = compute_two_year_pfs_for_cohort(data_dict[external_key], ecdf, t_months=24.0)

    low_i = two_year_internal.get("Low-risk", np.nan) * 100.0
    low_e = two_year_external.get("Low-risk", np.nan) * 100.0
    high_i = two_year_internal.get("High-risk", np.nan) * 100.0
    high_e = two_year_external.get("High-risk", np.nan) * 100.0

    # 英文句式输出
    print(
        f"In the internal and external validation cohorts, the 2-year progression-free survival (PFS) "
        f"rates were {low_i:.1f}% and {low_e:.1f}% in the low-risk group and "
        f"{high_i:.1f}% and {high_e:.1f}% in the high-risk group."
    )
