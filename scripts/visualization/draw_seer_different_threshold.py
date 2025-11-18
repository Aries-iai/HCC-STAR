# -*- coding: utf-8 -*-
import json
import re
import ast
import os
import argparse
from collections import Counter, defaultdict
from datetime import datetime, date
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# --- 统一的治疗名集合 ---
Tx = {
    "Surgical_resection",
    "Ablation",
    "Liver_transplantation",
    "Interventional_therapy",
    "Interventional_therapy_plus_ablation",
    "Surgical_resection_plus_ablation",
    "Systemic_anti-tumor_therapy",
    "Interventional_therapy_plus_systemic_anti-tumor_therapy",
    "Radiotherapy",
    "Symptomatic_support",
    "Palliative_care",
}

# ========== 分期标准化 ==========
_ROMAN_MAP = str.maketrans({
    "Ⅰ": "I","Ⅱ": "II","Ⅲ":"III","Ⅳ":"IV","Ⅴ":"V","Ⅵ":"VI","Ⅶ":"VII","Ⅷ":"VIII","Ⅸ":"IX","Ⅹ":"X"
})

def norm_bclc_stage(stage: str) -> str:
    if not stage:
        return ""
    s = stage.strip().upper()
    s = re.sub(r"BCLC|STAGE|分期|期|：|:|\s+", "", s)
    m = re.match(r"^(0|[ABCD])$", s)
    return m.group(1) if m else ""

def norm_cnlc_stage(stage: str) -> str:
    if not stage:
        return ""
    s = stage.strip()
    s = s.translate(_ROMAN_MAP)
    s = s.replace("ＣＮＬＣ", "CNLC").replace("ａ", "a").replace("ｂ", "b")
    s = re.sub(r"CNLC|STAGE|分期|期|：|:|\s+", "", s, flags=re.IGNORECASE)
    s = s.upper()
    s = s.replace("1A", "IA").replace("1B", "IB")
    s = s.replace("2A", "IIA").replace("2B", "IIB")
    s = s.replace("3A", "IIIA").replace("3B", "IIIB")
    s = s.replace("4", "IV")
    mapping = {"IA":"Ia","IB":"Ib","IIA":"IIa","IIB":"IIb","IIIA":"IIIa","IIIB":"IIIb","IV":"IV"}
    return mapping.get(s, "")

# ========== 指南映射表 ==========
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

def bclc_recommended(stage_raw: str) -> Tuple[str, ...]:
    s = norm_bclc_stage(stage_raw)
    return _BCLC_MAP.get(s, ())

def cnlc_recommended(stage_raw: str) -> Tuple[str, ...]:
    s = norm_cnlc_stage(stage_raw)
    return _CNLC_MAP.get(s, ())

# ========== 通用小工具 ==========
def _safe_to_int(x):
    try:
        if x is None: return None
        return int(float(x))
    except Exception:
        return None

def _safe_to_01_or_none(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, (int, float)):
        if x == 0: return 0
        if x == 1: return 1
        return None
    s = str(x).strip().lower()
    if s in {"0", "false"}: return 0
    if s in {"1", "true"}:  return 1
    if s in {"none", "null", ""}: return None
    return None

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def parse_hard_check_survival(solution_str):
    m = re.search(r"<hard_check_survival>(.*?)</hard_check_survival>", solution_str, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    obj = None
    try:
        fixed_raw = raw.encode().decode('unicode_escape')
        obj = json.loads(fixed_raw)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    if "survival_months" not in obj and "survival" in obj and isinstance(obj["survival"], dict):
        obj = obj["survival"]
    pred_months = _safe_to_int(obj.get("survival_months"))
    pred_1yr = _safe_to_01_or_none(obj.get("survival_1yr"))
    pred_3yr = _safe_to_01_or_none(obj.get("survival_3yr"))
    pred_5yr = _safe_to_01_or_none(obj.get("survival_5yr"))
    return {"survival_months": pred_months,"survival_1yr": pred_1yr,"survival_3yr": pred_3yr,"survival_5yr": pred_5yr}

def parse_treatment(res_text):
    m = re.search(r"<hard_check_treatment>(.*?)</hard_check_treatment>", res_text, re.DOTALL)
    if not m:
        return None
    try:
        fixed = m.group(1).strip().encode().decode('unicode_escape')
        hard = json.loads(fixed.replace("'", '"'))
        if isinstance(hard, dict) and 'scores' in hard:
            hard = hard['scores']
        return hard if isinstance(hard, dict) else None
    except json.JSONDecodeError:
        return None

def _is_na(x):
    if x is None: return True
    if isinstance(x, float) and np.isnan(x): return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "na", "none", "null"}: return True
    return False

def map_survival(if_death):
    if if_death == '否' or if_death == 0: return 0
    if if_death: return 1
    return None

def calculate_survival_months(start_date, end_date) -> int:
    start_date = start_date.replace('UK','01') if isinstance(start_date,str) else start_date
    end_date = end_date.replace('UK','01') if isinstance(end_date,str) else end_date
    def to_datetime(d):
        if isinstance(d,(datetime,date)): return datetime(d.year,d.month,d.day)
        if isinstance(d,str):
            for fmt in ("%Y/%m/%d","%Y-%m-%d"):
                try: return datetime.strptime(d, fmt)
                except ValueError: continue
        return None
    s = to_datetime(start_date); e = to_datetime(end_date)
    if not s or not e: return None
    months = (e.year - s.year)*12 + (e.month - s.month)
    if e.day < s.day: months -= 1
    return months

def get_month(sample):
    if sample is None: return None
    survival_month = sample.get('survival_month', None)
    if isinstance(survival_month, str) and '\n' in survival_month:
        survival_month = survival_month.split('\n')[-1].strip()
    if _is_na(survival_month): return None
    if_death = map_survival(sample.get('death'))
    if _is_na(if_death): if_death = None

    in_time = sample.get('in_time'); death_time = sample.get('death_time')
    if _is_na(in_time): in_time = None
    if _is_na(death_time): death_time = None

    if if_death is None:
        if in_time and death_time: if_death = 1
        elif in_time and survival_month: if_death = 0
        else: return None

    if isinstance(survival_month, str):
        parsed_int = _safe_to_int(survival_month)
        if parsed_int is not None: return {'survival_month': parsed_int, 'if_death': if_death}
        if if_death:
            if in_time and death_time:
                sm = calculate_survival_months(in_time, death_time)
                return {'survival_month': sm, 'if_death': if_death} if sm is not None else None
            elif death_time is None and in_time and survival_month:
                sm = calculate_survival_months(in_time, survival_month)
                return {'survival_month': sm, 'if_death': 0} if sm is not None else None
            else:
                return None
        else:
            if in_time and survival_month:
                sm = calculate_survival_months(in_time, survival_month)
                return {'survival_month': sm, 'if_death': if_death} if sm is not None else None
            else:
                return None
    elif isinstance(survival_month, (int,float)):
        parsed = _safe_to_int(survival_month)
        return {'survival_month': parsed, 'if_death': if_death} if parsed is not None else None
    return None

def get_staging(sample):
    def stage_from_tnm(t: str, n: str, m: str):
        t, n, m = str(t or "").upper(), str(n or "").upper(), str(m or "").upper()
        if 'M1' in m: return -5, 'IV'
        if 'N1' in n: return -4, 'IIIC'
        if 'T1' in t: return 0, 'I'
        elif 'T2' in t: return -1, 'II'
        elif 'T3' in t: return -2, 'IIIA'
        elif 'T4' in t: return -3, 'IIIB'
        return None
    def get_bclc(bclc):
        if bclc is None or (isinstance(bclc,float) and pd.isna(bclc)): return None
        s = norm_bclc_stage(str(bclc)); mp = {"0":0,"A":-1,"B":-2,"C":-3,"D":-4}
        return (mp[s], s) if s in mp else None
    def get_cnlc(cnlc):
        if cnlc is None or (isinstance(cnlc,float) and pd.isna(cnlc)): return None
        s = norm_cnlc_stage(str(cnlc)); mp = {"Ia":0,"Ib":-1,"IIa":-2,"IIb":-3,"IIIa":-4,"IIIb":-5,"IV":-6}
        return (mp[s], s) if s in mp else None
    def get_ajcc_tnm(ajcc_tnm):
        if ajcc_tnm is None or (isinstance(ajcc_tnm,float) and pd.isna(ajcc_tnm)): return None
        if isinstance(ajcc_tnm, dict):
            T,N,M = ajcc_tnm.get('T'), ajcc_tnm.get('N'), ajcc_tnm.get('M')
            if T is None or N is None or M is None: return None
            return stage_from_tnm(T,N,M)
        s = str(ajcc_tnm).upper()
        if "IIIA" in s: return -2,"IIIa"
        if "IIIB" in s: return -3,"IIIb"
        if "IIIC" in s: return -4,"III"
        if "III"  in s: return -2,"IIIa"
        if "IV"   in s: return -5,"IV"
        if "II"   in s: return -1,"II"
        if "I"    in s: return 0,"I"
        return None
    bclc = sample.get('bclc'); cnlc = sample.get('cnlc')
    ajcc_tnm = sample.get('ajcc_tnm') or sample.get('tnm')
    return {'bclc': get_bclc(bclc), 'cnlc': get_cnlc(cnlc), 'ajcc_tnm': get_ajcc_tnm(ajcc_tnm)}

# ========== 评估工具（Top-N） ==========
def get_top_n_treatments(pred_dict, n=3):
    if not isinstance(pred_dict, dict): return []
    items = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    return [k for k,_ in items[:n]]

def major_voting_topn(treatment_predictions_list, n=3):
    if not treatment_predictions_list:
        return {'top1': None, 'top2': None, 'top3': None}
    top_positions = {f'top{i+1}': [] for i in range(n)}
    for pdict in treatment_predictions_list:
        if not isinstance(pdict, dict):
            continue
        tops = get_top_n_treatments(pdict, n)
        for i, t in enumerate(tops):
            top_positions[f'top{i+1}'].append(t)
    result = {}
    for i in range(n):
        votes = top_positions[f'top{i+1}']
        if votes:
            c = Counter(votes).most_common(1)
            result[f'top{i+1}'] = c[0][0] if c else None
        else:
            result[f'top{i+1}'] = None
    return result

def build_uuid_to_tx_map(jsonl_path: str, threshold: float = 0.5) -> dict:
    rows = load_jsonl(jsonl_path)
    uuid2tx = {}
    for rec in rows:
        uuid = rec.get("extra_info", {}).get("uuid") or rec.get("uuid")
        if not uuid:
            continue
        text = ""
        llm_resp = rec.get("llm_response", {})
        if isinstance(llm_resp, dict):
            text = llm_resp.get("text", "") or ""
        tdict = parse_treatment(text)
        # 直接按分数阈值筛选 + 降序
        items = []
        if isinstance(tdict, dict):
            for t, v in tdict.items():
                try:
                    v = float(v)
                except Exception:
                    continue
                if v >= threshold:
                    items.append((t, v))
        items.sort(key=lambda x: (-x[1], x[0]))
        uuid2tx[uuid] = [t for t,_ in items]
    return uuid2tx

# --------- 颜色解析与分配（新增） ---------
_HEX_RE = re.compile(r'^#?[0-9A-Fa-f]{6}$')

def parse_color_list(s: str) -> List[str]:
    """支持逗号/空格分隔；可带/不带 #；统一返回带 # 的 hex 列表。"""
    if not s:
        return []
    parts = [p.strip() for p in re.split(r'[,\s]+', s) if p.strip()]
    colors = []
    for p in parts:
        if not _HEX_RE.match(p):
            raise ValueError(f"非法颜色值: {p}（需 6 位十六进制，如 #1f77b4）")
        colors.append(p if p.startswith('#') else f'#{p}')
    return colors

def assign_colors(labels: List[str], user_colors: Optional[List[str]]) -> Optional[List[str]]:
    """按 labels 顺序返回颜色；若 user_colors 为空则返回 None 使用 Matplotlib 默认色。"""
    if not user_colors:
        return None
    out = []
    for i in range(len(labels)):
        out.append(user_colors[i % len(user_colors)])
    return out

def plot_accuracy_curves(thresholds, curves: Dict[str, List[float]], title: str, save_name: str,
                         model_order: Optional[List[str]] = None,
                         user_colors: Optional[List[str]] = None):
    plt.figure(figsize=(9, 6))
    labels = model_order if model_order else list(curves.keys())
    colors = assign_colors(labels, user_colors)

    for i, label in enumerate(labels):
        ys = curves[label]
        if colors:
            plt.plot(thresholds, ys, marker='o', linewidth=2, label=label, color=colors[i])
        else:
            plt.plot(thresholds, ys, marker='o', linewidth=2, label=label)

    plt.ylim(0.0, 1.0)
    plt.xlabel("threshold")
    plt.ylabel("Top-N Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name + ".svg")
    plt.savefig(save_name + ".png", dpi=200)
    plt.close()
    print(f"[Saved] {save_name}.svg / .png")

# ================================ 主程序 ================================
if __name__ == "__main__":
    # --- 参数：仅新增 --colors，不影响你原来的硬编码路径 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--colors", type=str, default="",
                        help='用逗号或空格分隔的十六进制颜色，如 "#1f77b4,#d62728,#2ca02c"')
    args = parser.parse_args()
    user_colors = parse_color_list(args.colors) if args.colors else None

    # 主集（含 result_ours）
    res_path = '/share/home/cuipeng/data/liver_rl_full_with_tag_v_0915_concat_seer_test_v0916_filter_exclude_0_qwen3_32b_0924_step_step_10_tts_08t.jsonl'

    # 外部 LLM 集
    gemini_jsonl = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/liver_rl_full_with_tag_v_0915_concat_seer_test_v0916_filter_exclude_0_fixed_all_fixed_seer_0-shot_gemini-2_5-pro-preview-03-25_responses_dedup.jsonl"
    gpt4o_jsonl  = "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/res_llms/liver_rl_full_with_tag_v_0915_concat_seer_test_v0916_filter_exclude_0_fixed_all_fixed_seer_0-shot_gpt-4o-2024-08-06_responses_dedup.jsonl"

    data = load_jsonl(res_path)
    print(f"Loaded {len(data)} samples from {res_path}")

    samples = []
    miss_counts = defaultdict(int)

    for sample in data:
        replies = sample.get('result_ours', [])
        if not isinstance(replies, list) or len(replies) == 0:
            miss_counts["no_replies"] += 1
            continue

        # 分期（用于 BCLC/CNLC）
        st = get_staging({
            'bclc': sample['reward_model']['ground_truth'].get('bclc'),
            'cnlc': sample['reward_model']['ground_truth'].get('cnlc'),
            'ajcc_tnm': sample['reward_model']['ground_truth'].get('tnm'),
        })
        if not (st["bclc"] and st["cnlc"] and st["ajcc_tnm"]):
            miss_counts["stage_missing_any"] += 1
            continue

        # GT 最优治疗（英文）
        name_gt_treatment = 'None'
        best_score = -1.0
        for item in sample['reward_model']['ground_truth'].get('top3_treatments', []):
            t_raw = item.get('treatment')
            sc = float(item.get('score', 0.0))
            t_en = cn2en.get(t_raw, t_raw)
            if sc > best_score:
                best_score = sc
                name_gt_treatment = t_en

        # 解析每个回复的 {treatment: score}
        per_reply_tdicts = []
        for reply in replies:
            tdict = parse_treatment(reply)
            per_reply_tdicts.append(tdict if isinstance(tdict, dict) else {})

        # === Ours 的“major voting”综合 dict（固定权重 1.0/0.8/0.6）===
        mv = major_voting_topn(per_reply_tdicts, n=3)
        mv_dict = {}
        if mv.get('top1'): mv_dict[mv['top1']] = 1.0
        if mv.get('top2') and mv['top2'] != mv.get('top1'): mv_dict[mv['top2']] = 0.8
        if mv.get('top3') and mv['top3'] not in {mv.get('top1'), mv.get('top2')}: mv_dict[mv['top3']] = 0.6

        cur_uuid = sample.get("extra_info", {}).get("uuid") or sample.get("uuid")
        if not cur_uuid:
            miss_counts["no_uuid"] += 1
            continue

        samples.append({
            "uuid": cur_uuid,
            "bclc_stage": st["bclc"][1],
            "cnlc_stage": st["cnlc"][1],
            "gt_best": name_gt_treatment,
            "ours_major_voting_dict": mv_dict,      # <--- 之后按阈值筛选
        })

    print(f"[Stats] usable samples: {len(samples)} | misses: {dict(miss_counts)}")

    # 阈值扫描
    thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]

    methods = ["Ours", "Gemini-2.5-pro", "GPT-4o", "BCLC", "CNLC"]
    curves_top1 = {m: [] for m in methods}
    curves_top2 = {m: [] for m in methods}
    curves_top3 = {m: [] for m in methods}

    # 辅助：指南固定 TopN
    def guideline_topn(stage: str, mapper: Dict[str, Tuple[str, ...]], n: int) -> List[str]:
        rec = mapper.get(stage, ())
        return list(rec[:n]) if rec else []

    for thr in thresholds:
        # 其他 LLM：随阈值重建 uuid->tx 列表
        uuid2tx_gemini = build_uuid_to_tx_map(gemini_jsonl, threshold=thr)
        uuid2tx_gpt4o  = build_uuid_to_tx_map(gpt4o_jsonl,  threshold=thr)

        acc = {m: {1: [], 2: [], 3: []} for m in methods}

        for s in samples:
            gt = s["gt_best"]

            # ===== Ours：对“综合 dict”按阈值筛选，再按分值排序取 TopN =====
            mvd = s["ours_major_voting_dict"] or {}
            # 候选按分值降序（1.0 > 0.8 > 0.6），仅保留 >= thr
            ours_candidates = [kv for kv in mvd.items() if kv[1] >= thr]
            ours_candidates.sort(key=lambda x: (-x[1], x[0]))
            ours_top1 = [t for t,_ in ours_candidates[:1]]
            ours_top2 = [t for t,_ in ours_candidates[:2]]
            ours_top3 = [t for t,_ in ours_candidates[:3]]

            # ===== Gemini / GPT-4o：各自阈值筛选后的降序列表 =====
            g_list = uuid2tx_gemini.get(s["uuid"], [])
            o_list = uuid2tx_gpt4o.get(s["uuid"], [])
            gem_top1, gem_top2, gem_top3 = g_list[:1], g_list[:2], g_list[:3]
            gpt_top1, gpt_top2, gpt_top3 = o_list[:1], o_list[:2], o_list[:3]

            # ===== BCLC / CNLC：固定序 =====
            bclc_top1 = guideline_topn(s["bclc_stage"], _BCLC_MAP, 1)
            bclc_top2 = guideline_topn(s["bclc_stage"], _BCLC_MAP, 2)
            bclc_top3 = guideline_topn(s["bclc_stage"], _BCLC_MAP, 3)

            cnlc_top1 = guideline_topn(s["cnlc_stage"], _CNLC_MAP, 1)
            cnlc_top2 = guideline_topn(s["cnlc_stage"], _CNLC_MAP, 2)
            cnlc_top3 = guideline_topn(s["cnlc_stage"], _CNLC_MAP, 3)

            # ===== 计算 Top-N accuracy =====
            acc["Ours"][1].append(1 if gt in ours_top1 else 0)
            acc["Ours"][2].append(1 if gt in ours_top2 else 0)
            acc["Ours"][3].append(1 if gt in ours_top3 else 0)

            acc["Gemini-2.5-pro"][1].append(1 if gt in gem_top1 else 0)
            acc["Gemini-2.5-pro"][2].append(1 if gt in gem_top2 else 0)
            acc["Gemini-2.5-pro"][3].append(1 if gt in gem_top3 else 0)

            acc["GPT-4o"][1].append(1 if gt in gpt_top1 else 0)
            acc["GPT-4o"][2].append(1 if gt in gpt_top2 else 0)
            acc["GPT-4o"][3].append(1 if gt in gpt_top3 else 0)

            acc["BCLC"][1].append(1 if gt in bclc_top1 else 0)
            acc["BCLC"][2].append(1 if gt in bclc_top2 else 0)
            acc["BCLC"][3].append(1 if gt in bclc_top3 else 0)

            acc["CNLC"][1].append(1 if gt in cnlc_top1 else 0)
            acc["CNLC"][2].append(1 if gt in cnlc_top2 else 0)
            acc["CNLC"][3].append(1 if gt in cnlc_top3 else 0)

        # 聚合阈值点
        for m in methods:
            curves_top1[m].append(float(np.mean(acc[m][1])) if acc[m][1] else 0.0)
            curves_top2[m].append(float(np.mean(acc[m][2])) if acc[m][2] else 0.0)
            curves_top3[m].append(float(np.mean(acc[m][3])) if acc[m][3] else 0.0)

        print(f"[thr={thr:.2f}] " + " | ".join(
            f"{m}: T1={curves_top1[m][-1]:.3f}, T2={curves_top2[m][-1]:.3f}, T3={curves_top3[m][-1]:.3f}"
            for m in methods
        ))

    # -------- 输出 3 张折线图（加入自定义颜色） --------
    plot_accuracy_curves(
        thresholds, curves_top1,
        "[Internal] Top-1 Accuracy vs threshold",
        "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2/internal_top1_accuracy_vs_threshold",
        model_order=methods, user_colors=user_colors
    )
    plot_accuracy_curves(
        thresholds, curves_top2,
        "[Internal] Top-2 Accuracy vs threshold",
        "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2/internal_top2_accuracy_vs_threshold",
        model_order=methods, user_colors=user_colors
    )
    plot_accuracy_curves(
        thresholds, curves_top3,
        "[Internal] Top-3 Accuracy vs threshold",
        "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/visual_fin_1014_2/internal_top3_accuracy_vs_threshold",
        model_order=methods, user_colors=user_colors
    )

    print("Done.")
