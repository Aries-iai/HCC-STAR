# -*- coding: utf-8 -*-
"""
External only: multicenter / cg_3
- Ours：多回复 -> 位置投票 -> 赋固定权重 (1.0, 0.6, 0.3) -> 阈值筛选 -> 取 Top-N
- 外部 LLM：解析 <hard_check_treatment> 分数字典，按阈值≥t 过滤后降序取 Top-N
- BCLC/CNLC：用规范化函数映射到推荐列表，再取 Top-N

输出：
  A) 各数据集分别的 Top-1/2/3 vs threshold 折线图（单独）
  B) pooled（multicenter + cg_3 合并样本）Top-1/2/3 三张图：每种方法只画一条曲线

新增：
  - 通过 --colors 或顶部 CUSTOM_COLORS 自定义颜色（十六进制列表，逗号分隔）
  - 颜色在所有图中对齐：同名方法颜色一致
"""

import json, re, ast, os, argparse
from collections import Counter
from typing import Tuple, Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt

# -------------------- 自定义颜色入口 --------------------
# 方式 1：在这里填写 HEX 列表（留空则使用 --colors 或默认）
CUSTOM_COLORS: List[str] = []
HEX_RE = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")

# -------------------- 中英映射 --------------------
cn2en = {
    "手术切除": "Surgical_resection",
    "消融治疗": "Ablation",
    "肝移植": "Liver_transplantation",
    "介入治疗": "Interventional_therapy",
    "介入治疗联合消融治疗": "Interventional_therapy_plus_ablation",
    "手术切除联合消融治疗": "Surgical_resection_plus_ablation",
    "全身抗肿瘤治疗": "Systemic_anti-tumor_therapy",
    "介入治疗联合全身抗肿瘤治疗": "Interventional_therapy_plus_systemic_anti-tumor_therapy",
    "放射治疗": "Radiotherapy",
    "对症支持治疗": "Symptomatic_support",
    "姑息治疗": "Palliative_care",
}

# -------------------- 指南映射 --------------------
_BCLC_MAP = {
    "0":  ("Ablation", "Surgical_resection"),
    "A":  ("Surgical_resection", "Ablation", "Liver_transplantation"),
    "B":  ("Liver_transplantation", "Interventional_therapy", "Systemic_anti-tumor_therapy"),
    "C":  ("Systemic_anti-tumor_therapy",),
    "D":  ("Symptomatic_support", "Palliative_care"),
}
_CNLC_MAP = {
    "Ia":   ("Surgical_resection", "Ablation", "Liver_transplantation"),
    "Ib":   ("Surgical_resection", "Interventional_therapy", "Ablation",
             "Interventional_therapy_plus_ablation", "Liver_transplantation"),
    "IIa":  ("Surgical_resection", "Interventional_therapy",
             "Surgical_resection_plus_ablation", "Interventional_therapy_plus_ablation",
             "Liver_transplantation"),
    "IIb":  ("Interventional_therapy", "Surgical_resection", "Systemic_anti-tumor_therapy"),
    "IIIa": ("Interventional_therapy_plus_systemic_anti-tumor_therapy",
             "Surgical_resection", "Radiotherapy"),
    "IIIb": ("Systemic_anti-tumor_therapy", "Interventional_therapy", "Radiotherapy"),
    "IV":   ("Symptomatic_support", "Liver_transplantation", "Palliative_care"),
}

# -------------------- Data 配置 --------------------
CONFIG = {
    "multicenter": {
        "paths": {
            "main_jsonl": "/share/home/cuipeng/cuipeng_a100/siyan/hcc_external_center_processed_20250920_v1_full_4o_v0_formatted_v4_qwen3_32b_0924_step_step_10_07t.jsonl",
            "claude": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/merged_avg/hcc_external_center_processed_20250920_v1_full_4o_v0_formatted_v4_with_0-shot_claude-3-5-sonnet-20241022_responses_merged_avg.jsonl",
            "deepseek": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/merged_avg/hcc_external_center_processed_20250920_v1_full_4o_v0_formatted_v4_with_0-shot_deepseek-r1_responses_merged_avg.jsonl",
            "gemini": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/merged_avg/hcc_external_center_processed_20250920_v1_full_4o_v0_formatted_v4_with_0-shot_gemini-2_5-pro_responses_merged_avg.jsonl",
            "gpt4o":  "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/merged_avg/hcc_external_center_processed_20250920_v1_full_4o_v0_formatted_v4_with_0-shot_gpt-4o-2024-08-06_responses_merged_avg.jsonl",
            "gpt5":   "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/merged_avg/hcc_external_center_processed_20250920_v1_full_4o_v0_formatted_v4_with_0-shot_gpt-5_responses_merged_avg.jsonl",
        },
        "weights": (1.0, 0.6, 0.3),
        "externals": ["Claude-3.5-Sonnet", "DeepSeek-R1", "Gemini-2.5-pro", "GPT-4o", "GPT-5"],
        "stage_field": "staging",
    },
    "cg_3": {
        "paths": {
            "main_jsonl": "/share/home/cuipeng/cuipeng_a100/siyan/comined_chunggeng_20250919_full_v0_qwen3_32b_0924_step_step_10_5_turns_fixed.jsonl",
            "claude": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/comined_chunggeng_20250919_full_v0_0-shot_claude-3-5-sonnet-20241022_responses_dedup.jsonl",
            "deepseek": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/comined_chunggeng_20250919_full_v0_0-shot_deepseek-r1_responses_dedup.jsonl",
            "gemini": "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/comined_chunggeng_20250919_full_v0_0-shot_gemini-2_5-pro_responses_dedup.jsonl",
            "gpt4o":  "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/comined_chunggeng_20250919_full_v0_0-shot_gpt-4o-2024-08-06_responses_dedup.jsonl",
            "gpt5":   "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/comined_chunggeng_20250919_full_v0_0-shot_gpt-5_responses_dedup.jsonl",
        },
        "weights": (1.0, 0.6, 0.3),
        "externals": ["Claude-3.5-Sonnet", "DeepSeek-R1", "Gemini-2.5-pro", "GPT-4o", "GPT-5"],
        "stage_field": "staging",
    }
}

# 统一的“方法顺序”（跨数据集与 pooled 保持一致）
GLOBAL_METHOD_ORDER = ["Ours"] + CONFIG["multicenter"]["externals"] + ["BCLC", "CNLC"]

# 默认颜色（当未提供自定义颜色时用这个表再兜底）
DEFAULT_COLOR_MAP = {
    "Ours": "#1f77b4",
    "Claude-3.5-Sonnet": "#d62728",
    "DeepSeek-R1": "#2ca02c",
    "Gemini-2.5-pro": "#9467bd",
    "GPT-4o": "#ff7f0e",
    "GPT-5": "#17becf",
    "BCLC": "#8c564b",
    "CNLC": "#e377c2",
}
FALLBACK_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#6b6ecf", "#9c9ede",
    "#8ca252", "#b5cf6b", "#cedb9c", "#e7ba52", "#e7969c", "#ad494a"
]

def build_color_map(custom_hex_list: List[str], model_order: List[str]) -> Dict[str, str]:
    """
    若提供 custom_hex_list：按 model_order 循环分配。
    否则：用 DEFAULT_COLOR_MAP；缺失项按名称哈希从 FALLBACK_PALETTE 取。
    """
    if custom_hex_list:
        clean = [h.strip() for h in custom_hex_list if HEX_RE.match(h.strip())]
        if not clean:
            return build_color_map([], model_order)
        cmap = {}
        m = len(clean)
        for i, name in enumerate(model_order):
            cmap[name] = clean[i % m]
        return cmap
    # 默认策略
    cmap = {}
    for name in model_order:
        if name in DEFAULT_COLOR_MAP:
            cmap[name] = DEFAULT_COLOR_MAP[name]
        else:
            idx = abs(hash(name)) % len(FALLBACK_PALETTE)
            cmap[name] = FALLBACK_PALETTE[idx]
    return cmap

# -------------------- 基础工具 --------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

_ROMAN_MAP = str.maketrans({"Ⅰ":"I","Ⅱ":"II","Ⅲ":"III","Ⅳ":"IV","Ⅴ":"V","Ⅵ":"VI","Ⅶ":"VII","Ⅷ":"VIII","Ⅸ":"IX","Ⅹ":"X"})
def norm_bclc_stage(s: str) -> str:
    if not s: return ""
    s = re.sub(r"BCLC|STAGE|分期|期|：|:|\s+", "", str(s).strip().upper())
    return s if s in {"0","A","B","C","D"} else ""
def norm_cnlc_stage(s: str) -> str:
    if not s: return ""
    x = str(s).translate(_ROMAN_MAP).replace("ＣＮＬＣ","CNLC").replace("ａ","a").replace("ｂ","b")
    x = re.sub(r"CNLC|STAGE|分期|期|：|:|\s+", "", x, flags=re.IGNORECASE).upper()
    x = x.replace("1A","IA").replace("1B","IB").replace("2A","IIA").replace("2B","IIB") \
         .replace("3A","IIIA").replace("3B","IIIB").replace("4","IV")
    mp = {"IA":"Ia","IB":"Ib","IIA":"IIa","IIB":"IIb","IIIA":"IIIa","IIIB":"IIIb","IV":"IV"}
    return mp.get(x, "")
def bclc_recommended(stage_raw: str):
    return _BCLC_MAP.get(norm_bclc_stage(stage_raw), ())
def cnlc_recommended(stage_raw: str):
    return _CNLC_MAP.get(norm_cnlc_stage(stage_raw), ())

def parse_treatment(text: str):
    m = re.search(r"<hard_check_treatment>(.*?)</hard_check_treatment>", text, re.DOTALL | re.IGNORECASE)
    if not m: return None
    raw = m.group(1).strip()
    # 尝试多种解析
    candidates = [raw, raw.replace("'", '"')]
    for body in candidates:
        try:
            obj = json.loads(body)
            if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
                obj = obj["scores"]
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
            obj = obj["scores"]
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def get_top_n_treatments(score_dict: dict, n=3) -> List[str]:
    if not isinstance(score_dict, dict): return []
    items = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k,_ in items[:n]]

def major_voting_topn(dict_list: List[dict], n=3) -> Dict[str, str]:
    pos_votes = {f"top{i}":[] for i in range(1, n+1)}
    for d in dict_list:
        if not isinstance(d, dict): continue
        tops = get_top_n_treatments(d, n=n)
        for i,t in enumerate(tops):
            pos_votes[f"top{i+1}"].append(t)
    winners = {}
    for i in range(1, n+1):
        votes = pos_votes[f"top{i}"]
        winners[f"top{i}"] = Counter(votes).most_common(1)[0][0] if votes else None
    return winners

def build_uuid_to_tx_map(jsonl_path: str, threshold: float) -> dict:
    rows = load_jsonl(jsonl_path)
    u2tx = {}
    for rec in rows:
        uid = rec.get("extra_info",{}).get("uuid") or rec.get("uuid")
        if not uid: continue
        text = ""
        if isinstance(rec.get("llm_response"), dict):
            text = rec["llm_response"].get("text") or ""
        scores = parse_treatment(text)
        items = []
        if isinstance(scores, dict):
            for k,v in scores.items():
                try: v = float(v)
                except Exception: continue
                if v >= threshold: items.append((k, v))
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        u2tx[uid] = [k for k,_ in items]
    return u2tx

def guideline_topn_bclc(stage_raw: str, n: int) -> List[str]:
    rec = list(bclc_recommended(stage_raw))
    return rec[:n] if rec else []

def guideline_topn_cnlc(stage_raw: str, n: int) -> List[str]:
    rec = list(cnlc_recommended(stage_raw))
    return rec[:n] if rec else []

def plot_accuracy_curves(thresholds, curves: Dict[str, List[float]], title: str, save_name: str,
                         color_map: Dict[str, str], plot_order: List[str]):
    plt.figure(figsize=(9,6))
    any_line = False
    for label in plot_order:
        if label not in curves:  # 该方法在当前数据集可能不存在
            continue
        ys = curves[label]
        plt.plot(thresholds, ys, marker='o', linewidth=2.2, label=label,
                 color=color_map.get(label, "#444444"), alpha=0.95)
        any_line = True
    plt.ylim(0.0, 1.0)
    plt.xlabel("threshold")
    plt.ylabel("Top-N Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    if any_line:
        plt.savefig(save_name + ".svg")
        plt.savefig(save_name + ".png", dpi=200)
        print(f"[Saved] {save_name}.svg / .png")
    plt.close()

# -------------------- GT 提取（健壮） --------------------
def _safe_floatable(v) -> bool:
    try:
        float(v); return True
    except Exception:
        return False

def _try_parse_scores_like(x: Any) -> Dict[str, float] | None:
    if isinstance(x, dict):
        if all(_safe_floatable(v) for v in x.values()):
            return {k: float(v) for k, v in x.items()}
        if "treatment" in x and isinstance(x["treatment"], str):
            t = x["treatment"]; return {cn2en.get(t, t): 1.0}
        if "scores" in x:
            return _try_parse_scores_like(x["scores"])
        return None
    if isinstance(x, str):
        for s in (x, x.replace("'", '"')):
            try:
                obj = json.loads(s)
                r = _try_parse_scores_like(obj)
                if r: return r
            except Exception:
                pass
        try:
            obj = ast.literal_eval(x)
            r = _try_parse_scores_like(obj)
            if r: return r
        except Exception:
            return None
    return None

def get_gt_best_from_field(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return cn2en.get(val, val)
    if isinstance(val, dict):
        if "treatment" in val and isinstance(val["treatment"], str):
            t = val["treatment"]; return cn2en.get(t, t)
        sc = _try_parse_scores_like(val.get("scores", val))
        if isinstance(sc, dict) and len(sc):
            k = max(sc.items(), key=lambda kv: float(kv[1]))[0]
            return cn2en.get(k, k)
    return None

def extract_gt_best_and_stage(rec: dict, dataset: str):
    """返回 (gt_best_en, bclc_stage_raw, cnlc_stage_raw, uuid, ours_reply_texts:list[str])"""
    uid = rec.get("extra_info",{}).get("uuid") or rec.get("uuid")
    replies = rec.get("result_ours", [])
    texts = [r if isinstance(r, str) else str(r) for r in replies] if isinstance(replies, list) else []
    gt_best = get_gt_best_from_field(rec.get("treatment"))
    st = rec.get(CONFIG[dataset]["stage_field"], {}) or {}
    bclc = st.get("bclc", "")
    cnlc = st.get("cnlc", "")
    return gt_best, bclc, cnlc, uid, texts

# -------------------- Ours 综合字典（投票 + 权重） --------------------
def ours_major_dict_from_texts(reply_texts: List[str], weights: Tuple[float,float,float]) -> dict:
    per_reply_dicts = []
    for t in reply_texts:
        d = parse_treatment(t)
        per_reply_dicts.append(d if isinstance(d, dict) else {})
    winners = major_voting_topn(per_reply_dicts, n=3)  # {'top1','top2','top3'}
    w1, w2, w3 = weights
    out = {}
    if winners.get("top1"): out[winners["top1"]] = w1
    if winners.get("top2") and winners["top2"] not in out: out[winners["top2"]] = w2
    if winners.get("top3") and winners["top3"] not in out: out[winners["top3"]] = w3
    return out

# -------------------- 单数据集阈值扫描（保留原曲线 + 返回原始0/1以便 pooled 合并） --------------------
def run_one_dataset(name: str, thresholds: List[float], color_map: Dict[str, str]):
    cfg = CONFIG[name]
    rows = load_jsonl(cfg["paths"]["main_jsonl"])
    print(f"[{name}] loaded {len(rows)} rows")

    # 可用样本
    samples = []
    for rec in rows:
        gt_best, bclc, cnlc, uid, texts = extract_gt_best_and_stage(rec, name)
        if not uid or not gt_best:
            continue
        samples.append({"uuid": uid, "gt_best": gt_best, "bclc": bclc, "cnlc": cnlc, "ours_texts": texts})
    print(f"[{name}] usable samples: {len(samples)}")

    methods = ["Ours"] + cfg["externals"] + ["BCLC", "CNLC"]

    # 原始0/1存储：acc_raw[method][topk][thr_idx] = list(0/1)
    acc_raw = {m: {1:[[] for _ in thresholds], 2:[[] for _ in thresholds], 3:[[] for _ in thresholds]} for m in methods}

    # 预先把外部不同阈值的 uuid->tx 列表缓存起来，避免重复 I/O
    ext_maps_by_thr = {m:{} for m in cfg["externals"]}  # m -> {thr_idx: {uuid:[tx...]}}
    for ti, thr in enumerate(thresholds):
        for key, label in [("claude","Claude-3.5-Sonnet"),
                           ("deepseek","DeepSeek-R1"),
                           ("gemini","Gemini-2.5-pro"),
                           ("gpt4o","GPT-4o"),
                           ("gpt5","GPT-5")]:
            if key in cfg["paths"]:
                u2tx = build_uuid_to_tx_map(cfg["paths"][key], threshold=thr)
                ext_maps_by_thr[label][ti] = u2tx

    # 扫描阈值
    for ti, thr in enumerate(thresholds):
        for s in samples:
            gt = s["gt_best"]

            # Ours
            mvd = ours_major_dict_from_texts(s["ours_texts"], weights=cfg["weights"])
            ours_items = sorted([(k,v) for k,v in mvd.items() if v >= thr], key=lambda kv: (-kv[1], kv[0]))
            ours_t1 = [t for t,_ in ours_items[:1]]
            ours_t2 = [t for t,_ in ours_items[:2]]
            ours_t3 = [t for t,_ in ours_items[:3]]
            acc_raw["Ours"][1][ti].append(1 if gt in ours_t1 else 0)
            acc_raw["Ours"][2][ti].append(1 if gt in ours_t2 else 0)
            acc_raw["Ours"][3][ti].append(1 if gt in ours_t3 else 0)

            # 外部
            for m in cfg["externals"]:
                lst = ext_maps_by_thr[m][ti].get(s["uuid"], [])
                acc_raw[m][1][ti].append(1 if gt in lst[:1] else 0)
                acc_raw[m][2][ti].append(1 if gt in lst[:2] else 0)
                acc_raw[m][3][ti].append(1 if gt in lst[:3] else 0)

            # BCLC / CNLC
            b_top1 = guideline_topn_bclc(s["bclc"], 1); b_top2 = guideline_topn_bclc(s["bclc"], 2); b_top3 = guideline_topn_bclc(s["bclc"], 3)
            c_top1 = guideline_topn_cnlc(s["cnlc"], 1); c_top2 = guideline_topn_cnlc(s["cnlc"], 2); c_top3 = guideline_topn_cnlc(s["cnlc"], 3)
            acc_raw["BCLC"][1][ti].append(1 if gt in b_top1 else 0)
            acc_raw["BCLC"][2][ti].append(1 if gt in b_top2 else 0)
            acc_raw["BCLC"][3][ti].append(1 if gt in b_top3 else 0)
            acc_raw["CNLC"][1][ti].append(1 if gt in c_top1 else 0)
            acc_raw["CNLC"][2][ti].append(1 if gt in c_top2 else 0)
            acc_raw["CNLC"][3][ti].append(1 if gt in c_top3 else 0)

    # 由原始0/1得到均值曲线（用于各自单独图）
    curves = {"top1":{m:[] for m in methods},
              "top2":{m:[] for m in methods},
              "top3":{m:[] for m in methods}}
    for m in methods:
        for ti in range(len(thresholds)):
            curves["top1"][m].append(float(np.mean(acc_raw[m][1][ti])) if acc_raw[m][1][ti] else 0.0)
            curves["top2"][m].append(float(np.mean(acc_raw[m][2][ti])) if acc_raw[m][2][ti] else 0.0)
            curves["top3"][m].append(float(np.mean(acc_raw[m][3][ti])) if acc_raw[m][3][ti] else 0.0)

    # 单数据集绘图（按全局顺序画，缺的自动跳过）
    out_prefix = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2"
    plot_accuracy_curves(thresholds, curves["top1"], f"[{name}] Top-1 Accuracy vs threshold",
                         f"{out_prefix}/{name}_top1_accuracy_vs_threshold",
                         color_map, GLOBAL_METHOD_ORDER)
    plot_accuracy_curves(thresholds, curves["top2"], f"[{name}] Top-2 Accuracy vs threshold",
                         f"{out_prefix}/{name}_top2_accuracy_vs_threshold",
                         color_map, GLOBAL_METHOD_ORDER)
    plot_accuracy_curves(thresholds, curves["top3"], f"[{name}] Top-3 Accuracy vs threshold",
                         f"{out_prefix}/{name}_top3_accuracy_vs_threshold",
                         color_map, GLOBAL_METHOD_ORDER)

    return {
        "thresholds": thresholds,
        "methods": methods,
        "curves": curves,   # mean per dataset
        "raw": acc_raw,     # 0/1 lists per threshold (用于 pooled)
    }

# -------------------- pooled：合并 multicenter + cg_3 样本后，每方法只画一条曲线 --------------------
def plot_pooled_curves(results_list: List[dict], thresholds: List[float], color_map: Dict[str, str],
                       save_prefix: str = "pooled"):
    """
    results_list: [res_mc, res_cg] 其中每个包含 'raw' (acc_raw 结构), 'methods'
    acc_raw[method][topk][thr_idx] = list(0/1)
    """
    # 方法集合取交集（通常一致）
    methods = set(results_list[0]["methods"])
    for r in results_list[1:]:
        methods &= set(r["methods"])
    methods = sorted(methods, key=lambda x: GLOBAL_METHOD_ORDER.index(x) if x in GLOBAL_METHOD_ORDER else 1e9)

    def pooled_curve_for_topk(topk: int) -> Dict[str, List[float]]:
        curves = {m: [] for m in methods}
        for ti in range(len(thresholds)):
            for m in methods:
                # 拼接两个数据集在该 threshold 下的 0/1 列表
                pooled = []
                for r in results_list:
                    pooled.extend(r["raw"][m][topk][ti])
                curves[m].append(float(np.mean(pooled)) if pooled else 0.0)
        return curves

    curves_top1 = pooled_curve_for_topk(1)
    curves_top2 = pooled_curve_for_topk(2)
    curves_top3 = pooled_curve_for_topk(3)

    out_prefix = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2"
    plot_accuracy_curves(thresholds, curves_top1, "[External] Top-1 Accuracy vs threshold",
                         f"{out_prefix}/{save_prefix}_top1_accuracy_vs_threshold",
                         color_map, GLOBAL_METHOD_ORDER)
    plot_accuracy_curves(thresholds, curves_top2, "[External] Top-2 Accuracy vs threshold",
                         f"{out_prefix}/{save_prefix}_top2_accuracy_vs_threshold",
                         color_map, GLOBAL_METHOD_ORDER)
    plot_accuracy_curves(thresholds, curves_top3, "[External] Top-3 Accuracy vs threshold",
                         f"{out_prefix}/{save_prefix}_top3_accuracy_vs_threshold",
                         color_map, GLOBAL_METHOD_ORDER)

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="External Top-N accuracy vs threshold with customizable colors")
    ap.add_argument("--colors", type=str, default="",
                    help="逗号分隔的 16 进制颜色列表，如 '#1f77b4,#d62728,#2ca02c,#9467bd'")
    ap.add_argument("--thr_start", type=float, default=0.1, help="阈值起点（含）")
    ap.add_argument("--thr_end", type=float, default=0.9, help="阈值终点（含）")
    ap.add_argument("--thr_num", type=int, default=9, help="阈值采样个数（线性等距）")
    return ap.parse_args()

# -------------------- 入口 --------------------
if __name__ == "__main__":
    args = parse_args()
    # 阈值网格
    thresholds = [round(x, 2) for x in np.linspace(args.thr_start, args.thr_end, args.thr_num)]

    # 颜色来源优先级：--colors > CUSTOM_COLORS > DEFAULT
    if args.colors.strip():
        color_list = [s.strip() for s in args.colors.split(",") if s.strip()]
    else:
        color_list = CUSTOM_COLORS or []

    color_map = build_color_map(color_list, GLOBAL_METHOD_ORDER)

    # 分别跑两套（保留各自图 & 原始0/1）
    res_mc = run_one_dataset("multicenter", thresholds, color_map)
    res_cg = run_one_dataset("cg_3", thresholds, color_map)

    # 合并样本池，得到每方法只一条曲线
    plot_pooled_curves([res_mc, res_cg], thresholds, color_map, save_prefix="pooled_external")

    print("Done.")
