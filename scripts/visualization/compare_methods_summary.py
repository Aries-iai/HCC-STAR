#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple methods (models) across datasets in a summary JSON/JSONL (tts only).
Outputs four figures (tts only):
  1) external_treatment_multi_model.svg
  2) external_survival_multi_model.svg
  3) internal_treatment_multi_model.svg
  4) internal_survival_multi_model.svg

CSV (tts only):
  - topn_tts_summary.csv
  - cindex_tts_summary.csv

Aggregation:
  Internal = datasets in --internal (default: {"seer"})
  External = all other datasets
  Top-N    weighted by counts.used_for_tx
  C-index  weighted by counts.used_for_surv
"""

import os, json, argparse, re
from typing import Dict, Any, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------- Utils --------------------
_HEX_RE = re.compile(r'^#?[0-9A-Fa-f]{6}$')

def parse_color_list(s: str) -> List[str]:
    if not s: return []
    parts = [p.strip() for p in re.split(r'[,\s]+', s) if p.strip()]
    out = []
    for p in parts:
        if not _HEX_RE.match(p):
            raise ValueError(f"非法颜色值: {p}")
        out.append(p if p.startswith("#") else f"#{p}")
    return out

def deep_get(d: Dict[str,Any], keys: Iterable[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or (k not in cur):
            return default
        cur = cur[k]
    return cur

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Return a list of top-level dict items. JSON => [obj]; JSONL => [each line obj]."""
    rows = []
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return rows
        # try JSON first
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                rows.append(obj)
            elif isinstance(obj, list):
                rows.extend(obj)
            else:
                print(f"[WARN] Unsupported JSON root type in {path}: {type(obj)}")
        except json.JSONDecodeError:
            # treat as JSONL
            for line in txt.splitlines():
                line = line.strip()
                if not line: continue
                rows.append(json.loads(line))
    return rows

def merge_root_objects(root_objs: List[Dict[str,Any]]) -> Dict[str, Any]:
    """
    Merge by dataset->model->mode. Last write wins if conflict.
    """
    merged: Dict[str, Any] = {}
    for obj in root_objs:
        for dataset, v in obj.items():
            if not isinstance(v, dict): 
                continue
            merged.setdefault(dataset, {})
            for model, mv in v.items():
                if not isinstance(mv, dict):
                    continue
                merged[dataset].setdefault(model, {})
                for mode, mmv in mv.items():
                    if not isinstance(mmv, dict):
                        continue
                    merged[dataset][model][mode] = mmv
    return merged

# -------------------- Aggregation --------------------
TopNCats = ["top1","top2","top3"]
CidxCats = [("cindex_overall","Overall"),
            ("cindex_1yr","1yr"),
            ("cindex_3yr","3yr"),
            ("cindex_5yr","5yr")]

def weighted_mean(vals: List[Tuple[float, float]]) -> float:
    """vals: list of (value, weight)"""
    num = 0.0; den = 0.0
    for v, w in vals:
        if v is None: 
            continue
        if not (np.isfinite(v) and np.isfinite(w)):
            continue
        if w <= 0: 
            continue
        num += v * w
        den += w
    return (num / den) if den > 0 else np.nan

def collect_models(merged: Dict[str,Any]) -> List[str]:
    models = set()
    for _, dsv in merged.items():
        if not isinstance(dsv, dict): continue
        for model in dsv.keys():
            models.add(model)
    return sorted(models)

def cohort_of(dataset: str, internal_names: set) -> str:
    return "Internal" if dataset in internal_names else "External"

def aggregate_by_cohort(
    merged: Dict[str,Any],
    internal_names: set
) -> Tuple[
    Dict[str, Dict[str, Dict[str, Tuple[float,float]]]],   # topn_data[cohort][top?][model]=(nontts, tts)
    Dict[str, Dict[str, Dict[str, Tuple[float,float]]]]    # cidx_data[cohort][cat][model]=(nontts, tts)
]:
    """
    For each cohort (External/Internal), for each category, for each model:
      produce (nontts_value, tts_value) as weighted means across datasets in the cohort.
    """
    models = collect_models(merged)

    topn_data: Dict[str, Dict[str, Dict[str, Tuple[float,float]]]] = { "External": {}, "Internal": {} }
    cidx_data: Dict[str, Dict[str, Dict[str, Tuple[float,float]]]] = { "External": {}, "Internal": {} }
    for coh in ["External","Internal"]:
        for cat in TopNCats:
            topn_data[coh][cat] = {}
        for _, cat_name in CidxCats:
            cidx_data[coh][cat_name] = {}

    for model in models:
        accum_topn = {
            "External": {cat: {"nontts": [], "tts": []} for cat in TopNCats},
            "Internal": {cat: {"nontts": [], "tts": []} for cat in TopNCats},
        }
        accum_cidx = {
            "External": {cat_name: {"nontts": [], "tts": []} for _,cat_name in CidxCats},
            "Internal": {cat_name: {"nontts": [], "tts": []} for _,cat_name in CidxCats},
        }

        for dataset, dsv in merged.items():
            if model not in dsv: 
                continue
            coh = cohort_of(dataset, internal_names)
            for mode in ("nontts","tts"):
                mv = dsv[model].get(mode)
                if not isinstance(mv, dict): 
                    continue
                used_tx   = deep_get(mv, ["counts","used_for_tx"], 0) or 0
                used_surv = deep_get(mv, ["counts","used_for_surv"], 0) or 0

                # treatment Top-N
                for cat in TopNCats:
                    v = deep_get(mv, ["metrics","treatment", cat])
                    if v is None: 
                        continue
                    if mode == "nontts":
                        accum_topn[coh][cat]["nontts"].append((float(v), float(used_tx)))
                    else:
                        accum_topn[coh][cat]["tts"].append((float(v), float(used_tx)))

                # survival C-index
                for k, cat_name in CidxCats:
                    v = deep_get(mv, ["metrics","survival", k])
                    if v is None:
                        continue
                    if mode == "nontts":
                        accum_cidx[coh][cat_name]["nontts"].append((float(v), float(used_surv)))
                    else:
                        accum_cidx[coh][cat_name]["tts"].append((float(v), float(used_surv)))

        # finalize weighted means
        for coh in ("External","Internal"):
            for cat in TopNCats:
                nontts_val = weighted_mean(accum_topn[coh][cat]["nontts"])
                tts_val    = weighted_mean(accum_topn[coh][cat]["tts"])
                if np.isfinite(nontts_val) or np.isfinite(tts_val):
                    topn_data[coh][cat][model] = (nontts_val, tts_val)

            for cat_name in [c[1] for c in CidxCats]:
                nontts_val = weighted_mean(accum_cidx[coh][cat_name]["nontts"])
                tts_val    = weighted_mean(accum_cidx[coh][cat_name]["tts"])
                if np.isfinite(nontts_val) or np.isfinite(tts_val):
                    cidx_data[coh][cat_name][model] = (nontts_val, tts_val)

    return topn_data, cidx_data

# -------------------- Plotting (tts only) --------------------
def flat_clusters_single_bars(
    ax,
    cluster_labels: List[str],            # x keys（内部定位用）
    models: List[str],                    # legend entries
    values: Dict[str, Dict[str, float]],  # values[cluster_label][model] = tts_value
    *,
    ymax: float,
    cluster_gap: float = 1.8,
    model_gap: float = 0.10,
    bar_width: float = 0.26,
    auto_shrink: bool = True,
    safety_margin: float = 0.06,
    model_colors: List[str] = None,
    ylabel: str = "",
    title: str = "",
    legend_loc: str = "upper left",
    display_labels: List[str] | None = None,   # <<< 新增：可传入自定义横轴显示文本
):
    if model_colors is None or len(model_colors) == 0:
        model_colors = ["#4B3F72", "#468BCA", "#80C5A2", "#FFD373",
                        "#5F5F5E", "#B384BA", "#00A9A5", "#F27873"]

    n_clusters = len(cluster_labels)
    n_models   = len(models)

    def block_width(bw: float) -> float:
        return n_models * bw + (n_models - 1) * model_gap

    max_block = max(cluster_gap - safety_margin, 0.3)
    if auto_shrink:
        bw = min(bar_width, (max_block - (n_models - 1) * model_gap) / max(n_models, 1))
        bar_width = max(0.06, bw)
    blk_w = block_width(bar_width)

    cluster_centers = np.arange(n_clusters) * cluster_gap
    model_offsets   = (np.arange(n_models) - (n_models - 1) / 2.0) * (bar_width + model_gap)

    # —— 横轴显示：External / Internal；仅当后缀是 top1/top2/top3 时保留并大写为 -Top1/2/3
    
    # —— 横轴显示：默认规则（External/Internal；仅保留 -Top1/2/3），除非调用方传入 display_labels
    def pretty_label(raw: str) -> str:
        parts = raw.split("-", 1)
        cohort = parts[0].capitalize()  # external -> External
        if len(parts) > 1:
            suf = parts[1].lower()
            if suf in ("top1", "top2", "top3"):
                return f"{cohort}-" + suf.capitalize()  # External-Top1/2/3
        return cohort  # 其它后缀（cindex / survival 等）都去掉

    labels_to_show = display_labels if display_labels is not None \
                     else [pretty_label(cl) for cl in cluster_labels]

    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(cluster_centers[0] - blk_w*0.6, cluster_centers[-1] + blk_w*0.6)
    try:
        ax.set_xticks(cluster_centers, labels_to_show, fontsize=11)
    except TypeError:
        ax.set_xticks(cluster_centers)
        ax.set_xticklabels(labels_to_show, fontsize=11)


    for ci, clabel in enumerate(cluster_labels):
        cx = cluster_centers[ci]
        ax.axvline(cx, ymin=0, ymax=1, color="gray", linestyle="--", alpha=0.22)
        for mi, model in enumerate(models):
            x  = cx + model_offsets[mi]
            vv = values.get(clabel, {}).get(model, np.nan)
            if not np.isfinite(vv):
                continue
            h  = max(0.0, min(ymax, float(vv)))
            ax.add_patch(Rectangle((x - bar_width/2, 0.0), bar_width, h,
                                   linewidth=0, facecolor=model_colors[mi % len(model_colors)], alpha=1.0))
            ax.annotate(f"{h:.2f}", xy=(x, h), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10.5, fontweight="bold", clip_on=False, zorder=5)

    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title, fontsize=13, pad=8)
    proxies = [Rectangle((0,0),1,1,facecolor=model_colors[i % len(model_colors)], label=models[i])
               for i in range(n_models)]
    # ax.legend(handles=proxies, frameon=False, loc=legend_loc, ncols=min(4, n_models))
    # === Legend（只调整图例顺序，不影响柱子顺序）===
    # 1) 先按原 models 生成一个图例顺序列表
    legend_models = list(models)

    # 2) 把 qwen_8b 调到第二个（大小写不敏感；如只匹配精确名就去掉 lower()）
    try:
        idx = next(i for i, m in enumerate(legend_models) if m.lower() == "qwen_8b")
        q = legend_models.pop(idx)
        legend_models.insert(1, q)
    except StopIteration:
        pass  # 没找到就不处理

    # 3) 生成图例句柄；颜色用“模型在原 models 中的位置”来取，确保和柱子颜色一致
    def color_for_model(m):
        i = models.index(m)  # 以原 models 的索引取颜色
        return model_colors[i % len(model_colors)] if model_colors else None

    proxies = [
        Rectangle((0, 0), 1, 1, facecolor=color_for_model(m), label=m)
        for m in legend_models
    ]
    ax.legend(handles=proxies, frameon=False, loc=legend_loc, ncols=min(4, len(legend_models)))


# -------------------- CSV dumps (tts only) --------------------
def dump_topn_tts_csv(path: str, data_flat_tts: Dict[str, Dict[str, float]]):
    rows = []
    for cluster_label, model_map in data_flat_tts.items():
        for model, tts_val in model_map.items():
            rows.append({"cluster": cluster_label, "model": model, "tts": tts_val})
    pd.DataFrame(rows).to_csv(path, index=False)

def dump_cidx_tts_csv(path: str, data_flat_tts: Dict[str, Dict[str, float]]):
    rows = []
    for cluster_label, model_map in data_flat_tts.items():
        for model, tts_val in model_map.items():
            rows.append({"cluster": cluster_label, "model": model, "tts": tts_val})
    pd.DataFrame(rows).to_csv(path, index=False)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True,
                    help="多个 JSON/JSONL 用逗号分隔（每个文件可为JSON或JSONL，根结构为你给的那种）")
    ap.add_argument("--internal", type=str, default="seer",
                    help="Internal 数据集名单，逗号分隔（默认：seer）")
    ap.add_argument("--outdir", type=str, default="./visual_fin",
                    help="输出目录（默认 ./visual_fin）")
    ap.add_argument("--model-colors", type=str, default="",
                    help="每个模型的柱色（按模型顺序循环）")
    ap.add_argument("--ymax-topn", type=float, default=1.0)
    ap.add_argument("--ymax-cidx", type=float, default=0.85)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    OUT_EXT_TREAT = os.path.join(args.outdir, "external_treatment_multi_model.svg")
    OUT_EXT_SURV  = os.path.join(args.outdir, "external_survival_multi_model.svg")
    OUT_INT_TREAT = os.path.join(args.outdir, "internal_treatment_multi_model.svg")
    OUT_INT_SURV  = os.path.join(args.outdir, "internal_survival_multi_model.svg")
    OUT_TOPN_TTS_CSV = os.path.join(args.outdir, "topn_tts_summary.csv")
    OUT_CIDX_TTS_CSV = os.path.join(args.outdir, "cindex_tts_summary.csv")

    internal_names = set([s.strip() for s in args.internal.split(",") if s.strip()])
    MODEL_COLORS = parse_color_list(args.model_colors) if args.model_colors else []

    # load & merge
    root_objs: List[Dict[str,Any]] = []
    for p in [p.strip() for p in args.inputs.split(",") if p.strip()]:
        root_objs.extend(load_json_or_jsonl(p))
    merged = merge_root_objects(root_objs)

    # aggregate
    topn_data, cidx_data = aggregate_by_cohort(merged, internal_names)

    # model order
    models = collect_models(merged)

    def reorder_models(ms: List[str]) -> List[str]:
        if 'qwen_8b' in ms:
            ms2 = [m for m in ms if m != "qwen_8b"]
            ms2.insert(1, "qwen_8b")  # 放到第二位
            return ms2
        return ms

    models = reorder_models(models)
    if len(models) == 0:
        print("[ERROR] 未在输入中解析到任何模型。请检查输入文件格式。")
        return

    # ---- 扁平簇（只 tts）构造 ----
    # Top-N -> flat dicts
    topn_flat_tts: Dict[str, Dict[str, float]] = {}
    for coh in ("External", "Internal"):
        for cat in ("top1","top2","top3"):
            clabel = f"{coh.lower()}-{cat.replace('top','top')}"
            topn_flat_tts[clabel] = {}
            model_map = topn_data.get(coh, {}).get(cat, {})
            for m in models:
                if m in model_map:
                    _, tts_val = model_map[m]
                    topn_flat_tts[clabel][m] = tts_val

    # C-index -> flat dicts
    cidx_flat_tts: Dict[str, Dict[str, float]] = {}
    ci_map = {"Overall":"cindex", "1yr":"1yr-survival", "3yr":"3yr-survival", "5yr":"5yr-survival"}
    for coh in ("External", "Internal"):
        for cat_name, suffix in ci_map.items():
            clabel = f"{coh.lower()}-{suffix}"
            cidx_flat_tts[clabel] = {}
            model_map = cidx_data.get(coh, {}).get(cat_name, {})
            for m in models:
                if m in model_map:
                    _, tts_val = model_map[m]
                    cidx_flat_tts[clabel][m] = tts_val

    # ---- 导出 CSV（tts only，整体各一份）----
    dump_topn_tts_csv(OUT_TOPN_TTS_CSV, topn_flat_tts)
    dump_cidx_tts_csv(OUT_CIDX_TTS_CSV, cidx_flat_tts)

    # ---- External / Internal × (Treatment / Survival) 四张图 ----
    ext_treat_clusters = ["external-top1", "external-top2", "external-top3"]
    ext_surv_clusters  = ["external-cindex", "external-1yr-survival", "external-3yr-survival", "external-5yr-survival"]
    int_treat_clusters = ["internal-top1", "internal-top2", "internal-top3"]
    int_surv_clusters  = ["internal-cindex", "internal-1yr-survival", "internal-3yr-survival", "internal-5yr-survival"]

    # 取值映射
    def pick_values(clusters: List[str], topn_src, cidx_src):
        vals = {}
        for cl in clusters:
            if cl in topn_src:
                vals[cl] = topn_src[cl]
            if cl in cidx_src:
                vals[cl] = cidx_src[cl]
        return vals

    # 1) External - Treatment
    fig_e_t, ax_e_t = plt.subplots(figsize=(max(10.0, 6.5 + 0.6*len(models)), 5.2))
    flat_clusters_single_bars(
        ax_e_t,
        cluster_labels=ext_treat_clusters,
        models=models,
        values=pick_values(ext_treat_clusters, topn_flat_tts, cidx_flat_tts),
        ymax=args.ymax_topn,
        cluster_gap=2.0, model_gap=0.08, bar_width=0.25, auto_shrink=True,
        model_colors=MODEL_COLORS,
        ylabel="Accuracy",
        # title="External — Treatment Top-N (tts only; weighted by used_for_tx)",
        title="",
        legend_loc="upper left",
        display_labels=["Top-1", "Top-2", "Top-3"],
    )
    for sp in ("top","right"): ax_e_t.spines[sp].set_visible(False)
    fig_e_t.tight_layout(); fig_e_t.savefig(OUT_EXT_TREAT, dpi=300); plt.close(fig_e_t)

    # 2) External - Survival
    fig_e_s, ax_e_s = plt.subplots(figsize=(max(12.0, 7.0 + 0.6*len(models)), 5.4))
    flat_clusters_single_bars(
        ax_e_s,
        cluster_labels=ext_surv_clusters,
        models=models,
        values=pick_values(ext_surv_clusters, topn_flat_tts, cidx_flat_tts),
        ymax=args.ymax_cidx,
        cluster_gap=1.9, model_gap=0.08, bar_width=0.24, auto_shrink=True,
        model_colors=MODEL_COLORS,
        ylabel="C-index",
        title="",
        legend_loc="upper left",
        display_labels=["Overall","1-year","3-year","5-year"],   # <<< 这里显式指定
    )
    for sp in ("top","right"): ax_e_s.spines[sp].set_visible(False)
    fig_e_s.tight_layout(); fig_e_s.savefig(OUT_EXT_SURV, dpi=300); plt.close(fig_e_s)

    # 3) Internal - Treatment
    fig_i_t, ax_i_t = plt.subplots(figsize=(max(10.0, 6.5 + 0.6*len(models)), 5.2))
    flat_clusters_single_bars(
        ax_i_t,
        cluster_labels=int_treat_clusters,
        models=models,
        values=pick_values(int_treat_clusters, topn_flat_tts, cidx_flat_tts),
        ymax=args.ymax_topn,
        cluster_gap=2.0, model_gap=0.08, bar_width=0.25, auto_shrink=True,
        model_colors=MODEL_COLORS,
        ylabel="Accuracy",
        title="",
        legend_loc="upper left",
        display_labels=["Top-1", "Top-2", "Top-3"],
    )
    for sp in ("top","right"): ax_i_t.spines[sp].set_visible(False)
    fig_i_t.tight_layout(); fig_i_t.savefig(OUT_INT_TREAT, dpi=300); plt.close(fig_i_t)

    # 4) Internal - Survival
    fig_i_s, ax_i_s = plt.subplots(figsize=(max(12.0, 7.0 + 0.6*len(models)), 5.4))
    flat_clusters_single_bars(
        ax_i_s,
        cluster_labels=int_surv_clusters,
        models=models,
        values=pick_values(int_surv_clusters, topn_flat_tts, cidx_flat_tts),
        ymax=args.ymax_cidx,
        cluster_gap=1.9, model_gap=0.08, bar_width=0.24, auto_shrink=True,
        model_colors=MODEL_COLORS,
        ylabel="C-index",
        title="",
        legend_loc="upper left",
        display_labels=["Overall","1-year","3-year","5-year"],   # <<< 同上
    )
    for sp in ("top","right"): ax_i_s.spines[sp].set_visible(False)
    fig_i_s.tight_layout(); fig_i_s.savefig(OUT_INT_SURV, dpi=300); plt.close(fig_i_s)

    print("Saved:")
    print(f" - {OUT_TOPN_TTS_CSV}")
    print(f" - {OUT_CIDX_TTS_CSV}")
    print(f" - {OUT_EXT_TREAT}")
    print(f" - {OUT_EXT_SURV}")
    print(f" - {OUT_INT_TREAT}")
    print(f" - {OUT_INT_SURV}")

if __name__ == "__main__":
    main()
