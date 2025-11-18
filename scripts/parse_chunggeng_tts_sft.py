# -*- coding: utf-8 -*-
"""
严格版：主流程（Ours）若 TNM/BCLC/CNLC 任一缺失则跳过；
使用主流程通过样本形成 UUID 白名单，所有输出数据集严格按该白名单构建，条数一致。
外部 LLM 若缺失该 UUID 的记录或解析失败，则用主流程 GT 并给默认预测兜底，但不丢样本。

>>> 新增：并列一个 Ours-SFT 分支（res_path_sft），解析与融合方法与 Ours 一致：
- 导出 df_ours_sft_cg_3.csv（长度与白名单一致）
- 在 staging_survival_cg_3.jsonl 中新增 tx_ours_sft 与 predicted_survival_sft
"""

import json
import re
import ast
import os
from collections import Counter
from datetime import datetime, date
from typing import Iterable, Set, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import exp
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

# ------------------ 治疗名中英映射 ------------------
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
    if not stage: return ""
    s = stage.strip().upper()
    s = re.sub(r"BCLC|STAGE|分期|期|：|:|\s+", "", s)
    m = re.match(r"^(0|[ABCD])$", s)
    return m.group(1) if m else ""

def norm_cnlc_stage(stage: str) -> str:
    if not stage: return ""
    s = stage.strip().translate(_ROMAN_MAP)
    s = s.replace("ＣＮＬＣ", "CNLC").replace("ａ", "a").replace("ｂ", "b")
    s = re.sub(r"CNLC|STAGE|分期|期|：|:|\s+", "", s, flags=re.IGNORECASE).upper()
    s = s.replace("1A", "IA").replace("1B", "IB").replace("2A", "IIA").replace("2B", "IIB") \
         .replace("3A", "IIIA").replace("3B", "IIIB").replace("4", "IV")
    mapping = {"IA":"Ia","IB":"Ib","IIA":"IIa","IIB":"IIb","IIIA":"IIIa","IIIB":"IIIb","IV":"IV"}
    return mapping.get(s, "")

_BCLC_MAP = {
    "0":  ("Ablation", "Surgical_resection"),
    "A":  ("Surgical_resection", "Ablation", "Liver_transplantation"),
    "B":  ("Liver_transplantation", "Interventional_therapy", "Systemic_anti-tumor_therapy"),
    "C":  ("Systemic_anti-tumor_therapy",),
    "D":  ("Symptomatic_support", "Palliative_care"),
}
_CNLC_MAP = {
    "Ia": ("Surgical_resection", "Ablation", "Liver_transplantation"),
    "Ib": ("Surgical_resection","Interventional_therapy","Ablation",
           "Interventional_therapy_plus_ablation","Liver_transplantation"),
    "IIa":("Surgical_resection","Interventional_therapy",
           "Surgical_resection_plus_ablation","Interventional_therapy_plus_ablation","Liver_transplantation"),
    "IIb":("Interventional_therapy","Surgical_resection","Systemic_anti-tumor_therapy"),
    "IIIa":("Interventional_therapy_plus_systemic_anti-tumor_therapy","Surgical_resection","Radiotherapy"),
    "IIIb":("Systemic_anti-tumor_therapy","Interventional_therapy","Radiotherapy"),
    "IV": ("Symptomatic_support","Liver_transplantation","Palliative_care"),
}

def bclc_recommended(stage_raw: str) -> Tuple[str, ...]:
    s = norm_bclc_stage(stage_raw)
    return _BCLC_MAP.get(s, ())

def cnlc_recommended(stage_raw: str) -> Tuple[str, ...]:
    s = norm_cnlc_stage(stage_raw)
    return _CNLC_MAP.get(s, ())

# ------------------ KM 绘图 ------------------
def plot_km(df, save_path, time_col='time', event_col='event'):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    df_plot = df[[time_col, event_col, 'staging']].copy()
    df_plot[time_col] = pd.to_numeric(df_plot[time_col], errors='coerce')
    df_plot[event_col] = pd.to_numeric(df_plot[event_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[time_col, event_col, 'staging'])
    for group_name, group_data in df_plot.groupby('staging'):
        kmf.fit(group_data[time_col], event_observed=group_data[event_col], label=str(group_name))
        kmf.plot_survival_function(ci_show=False)
    plt.title("Kaplan–Meier (Overall)")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Staging")
    plt.tight_layout()
    plt.savefig(save_path, format='svg'); plt.close()

def plot_km_capped(df, save_path, horizon_months, time_col='time', event_col='event'):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    df_plot = df[[time_col, event_col, 'staging']].copy()
    df_plot[time_col] = pd.to_numeric(df_plot[time_col], errors='coerce')
    df_plot[event_col] = pd.to_numeric(df_plot[event_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[time_col, event_col, 'staging'])
    time_capped = np.minimum(df_plot[time_col].values, horizon_months)
    event_within = ((df_plot[event_col].values == 1) &
                    (df_plot[time_col].values <= horizon_months)).astype(int)
    df_plot['time_capped'] = time_capped; df_plot['event_within'] = event_within
    for group_name, group_data in df_plot.groupby('staging'):
        kmf.fit(group_data['time_capped'], event_observed=group_data['event_within'], label=str(group_name))
        kmf.plot_survival_function(ci_show=False)
    plt.title(f"Kaplan–Meier (Task: ≤{horizon_months//12} year survival)")
    plt.xlabel("Time (months)"); plt.ylabel("Survival Probability")
    plt.xlim(0, horizon_months); plt.axvline(horizon_months, linestyle='--', linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.5); plt.legend(title="Staging")
    plt.tight_layout(); plt.savefig(save_path, format='svg'); plt.close()

# ------------------ 工具函数 ------------------
def _safe_to_int(x):
    try:
        if x is None: return None
        return int(float(x))
    except Exception:
        return None

def _safe_to_01_or_none(x):
    if x is None: return None
    if isinstance(x, bool): return 1 if x else 0
    if isinstance(x, (int, float)):
        if x == 0: return 0
        if x == 1: return 1
        return None
    s = str(x).strip().lower()
    if s in {"0","false"}: return 0
    if s in {"1","true"}:  return 1
    if s in {"none","null",""}: return None
    return None

def _months_score_with_event(pred_months, gt_months, event, gt_flags=None, tau_death=12.0, tau_censor=6.0):
    if pred_months is None: return 0.0
    if event == 1:
        if gt_months is None: return 0.0
        diff = abs(pred_months - gt_months)
        return exp(-diff / float(tau_death))
    if event == 0:
        t_c = gt_months if gt_months is not None else _max_passed_threshold_months_from_flags(gt_flags or {})
        if t_c is None: return 0.5
        d = max(0.0, float(t_c) - float(pred_months))
        return exp(-d / float(tau_censor))
    return 0.5

def _max_passed_threshold_months_from_flags(flags):
    mapping = [("survival_1yr", 12), ("survival_3yr", 36), ("survival_5yr", 60)]
    passed = [m for k, m in mapping if flags.get(k) == 1]
    return max(passed) if passed else None

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def parse_hard_check_survival(solution_str):
    m = re.search(r"<hard_check_survival>(.*?)</hard_check_survival>", solution_str, re.DOTALL)
    if not m: return None
    raw = m.group(1).strip()
    try:
        fixed_raw = raw.encode().decode('unicode_escape')
        obj = json.loads(fixed_raw)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            return None
    if not isinstance(obj, dict): return None
    if "survival_months" not in obj and isinstance(obj.get("survival"), dict):
        obj = obj["survival"]
    return {
        "survival_months": _safe_to_int(obj.get("survival_months")),
        "survival_1yr": _safe_to_01_or_none(obj.get("survival_1yr")),
        "survival_3yr": _safe_to_01_or_none(obj.get("survival_3yr")),
        "survival_5yr": _safe_to_01_or_none(obj.get("survival_5yr")),
    }

def parse_treatment(res):
    m = re.search(r"<hard_check_treatment>(.*?)</hard_check_treatment>", res, re.DOTALL)
    if not m: return None
    try:
        fixed = m.group(1).strip().encode().decode('unicode_escape')
        hard = json.loads(fixed.replace("'", '"'))
        if 'scores' in hard: hard = hard['scores']
        return hard
    except json.JSONDecodeError:
        return None

def get_treatment(t):
    if isinstance(t, dict):
        res = t.get('scores')
        try:
            if isinstance(res, str):
                res = ast.literal_eval(res)
                return res.get('treatment')
        except Exception:
            return None
    if isinstance(t, str):
        return cn2en.get(t, t)
    return None

def evaluate_survival(gt, res):
    def cal_correctness(pred, gtv):
        if gtv is None: return None
        if pred is None: return 0.0
        if isinstance(pred, bool): pred = 1 if pred else 0
        if isinstance(gtv, bool): gtv = 1 if gtv else 0
        return 1.0 if pred == gtv else 0.0

    survival_months = gt.get('survival_month')
    survival_1yr = gt.get('survival_1yr')
    survival_3yr = gt.get('survival_3yr')
    survival_5yr = gt.get('survival_5yr')

    pred = parse_hard_check_survival(res)
    if pred is None: return None
    pm = pred.get('survival_months')
    if pm is None: return None

    p1 = 1 if pm > 12 else 0
    p3 = 1 if pm > 36 else 0
    p5 = 1 if pm > 60 else 0

    mapped = min((pm / 100), 1) * 100
    survival_rank = 'D' if mapped < 4 else ('C' if mapped < 12 else ('B' if mapped < 24 else 'A'))

    risk = pm
    survival_score = _months_score_with_event(
        pred_months=pm, gt_months=survival_months, event=gt.get('if_death'),
        gt_flags={'survival_1yr': survival_1yr, 'survival_3yr': survival_3yr,
                  'survival_5yr': survival_5yr, 'survival_months': survival_months}
    )
    return {
        'acc_1yr': cal_correctness(p1, survival_1yr),
        'acc_3yr': cal_correctness(p3, survival_3yr),
        'acc_5yr': cal_correctness(p5, survival_5yr),
        'survival_score': survival_score,
        'survival_months': pm,
        'risk_score': risk,
        'survival_rank': survival_rank,
    }

def map_survival(if_death):
    if if_death == '否' or if_death == 0: return 0
    if if_death: return 1
    return None

def calculate_survival_months(start_date, end_date) -> int:
    start_date = start_date.replace('UK','01'); end_date = end_date.replace('UK','01')
    def to_dt(d):
        if isinstance(d, (datetime, date)): return datetime(d.year, d.month, d.day)
        if isinstance(d, str):
            for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
                try: return datetime.strptime(d, fmt)
                except ValueError: continue
        return None
    s = to_dt(start_date); e = to_dt(end_date)
    if not s or not e: return None
    months = (e.year - s.year) * 12 + (e.month - s.month)
    if e.day < s.day: months -= 1
    return months

def _is_na(x):
    if x is None: return True
    if isinstance(x, float) and np.isnan(x): return True
    if isinstance(x, str) and x.strip().lower() in {"","nan","na","none","null"}: return True
    return False

def get_month(sample):
    if sample is None: return None
    sv = sample.get('survival_month', None)
    if isinstance(sv, str) and '\n' in sv: sv = sv.split('\n')[-1].strip()
    if _is_na(sv): return None
    ideath = map_survival(sample.get('death'))
    if _is_na(ideath): ideath = None
    if ideath is None:
        in_time = None if _is_na(sample.get('in_time')) else sample.get('in_time')
        death_time = None if _is_na(sample.get('death_time')) else sample.get('death_time')
        if in_time and death_time: ideath = 1
        elif in_time and sv: ideath = 0
        else: return None
    in_time = None if _is_na(sample.get('in_time')) else sample.get('in_time')
    death_time = None if _is_na(sample.get('death_time')) else sample.get('death_time')
    if isinstance(sv, str):
        v = _safe_to_int(sv)
        if v is not None: return {'survival_month': v, 'if_death': ideath}
        if ideath:
            if in_time and death_time:
                sm = calculate_survival_months(in_time, death_time); 
                if sm is None: return None
                return {'survival_month': sm, 'if_death': ideath}
            elif death_time is None and in_time and sv:
                sm = calculate_survival_months(in_time, sv); 
                if sm is None: return None
                return {'survival_month': sm, 'if_death': 0}
            else: return None
        else:
            if in_time and sv:
                sm = calculate_survival_months(in_time, sv); 
                if sm is None: return None
                return {'survival_month': sm, 'if_death': ideath}
            else: return None
    elif isinstance(sv, (int, float)):
        v = _safe_to_int(sv)
        if v is None: return None
        return {'survival_month': v, 'if_death': ideath}
    return None

def get_staging(sample):
    def stage_from_tnm(t, n, m):
        t, n, m = str(t or "").upper(), str(n or "").upper(), str(m or "").upper()
        if 'M1' in m: return -5, 'IV'
        if 'N1' in n: return -4, 'IIIC'
        if 'T1' in t: return 0, 'I'
        if 'T2' in t: return -1, 'II'
        if 'T3' in t: return -2, 'IIIA'
        if 'T4' in t: return -3, 'IIIB'
        return None
    def get_bclc(b):
        if b is None or (isinstance(b, float) and pd.isna(b)): return None
        s = norm_bclc_stage(str(b))
        score = {"0":0,"A":-1,"B":-2,"C":-3,"D":-4}
        return (score[s], s) if s in score else None
    def get_cnlc(c):
        if c is None or (isinstance(c, float) and pd.isna(c)): return None
        s = norm_cnlc_stage(str(c))
        score = {"Ia":0,"Ib":-1,"IIa":-2,"IIb":-3,"IIIa":-4,"IIIb":-5,"IV":-6}
        return (score[s], s) if s in score else None
    def get_ajcc_tnm(tnm):
        if tnm is None or (isinstance(tnm, float) and pd.isna(tnm)): return None
        if isinstance(tnm, dict):
            T,N,M = tnm.get('T'), tnm.get('N'), tnm.get('M')
            if T is None or N is None or M is None: return None
            return stage_from_tnm(T,N,M)
        s = str(tnm).upper()
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

def remove_outliers(values, threshold=3):
    if len(values) <= 1: return values
    q1 = np.percentile(values, 25); q3 = np.percentile(values, 75); iqr = q3 - q1
    lb = q1 - threshold * iqr; ub = q3 + threshold * iqr
    filt = [v for v in values if lb <= v <= ub]
    return filt if len(filt) > 0 else values

def get_top_n_treatments(treatment_scores, n=3):
    if not isinstance(treatment_scores, dict):
        return []
    sorted_treatments = sorted(treatment_scores.items(), key=lambda x: x[1], reverse=True)
    return [treatment for treatment, score in sorted_treatments[:n]]

def major_voting_treatment(treatment_results_list):
    if not treatment_results_list:
        return None
    top1_votes, top2_votes, top3_votes = [], [], []
    for treatment_result in treatment_results_list:
        if treatment_result is None:
            continue
        top_treatments = get_top_n_treatments(treatment_result, n=3)
        if len(top_treatments) >= 1: top1_votes.append(top_treatments[0])
        if len(top_treatments) >= 2: top2_votes.append(top_treatments[1])
        if len(top_treatments) >= 3: top3_votes.append(top_treatments[2])

    def get_majority_vote(votes):
        if not votes:
            return None
        counter = Counter(votes)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else None

    top1_winner = get_majority_vote(top1_votes)
    top2_winner = get_majority_vote(top2_votes)
    top3_winner = get_majority_vote(top3_votes)

    final_result = {}
    all_treatments = set()
    for treatment_result in treatment_results_list:
        if treatment_result is not None:
            all_treatments.update(treatment_result.keys())
    for treatment in all_treatments:
        final_result[treatment] = 0.0
    if top1_winner:
        final_result[top1_winner] = 1.0
    if top2_winner and top2_winner != top1_winner:
        final_result[top2_winner] = 0.6
    if top3_winner and top3_winner not in [top1_winner, top2_winner]:
        final_result[top3_winner] = 0.3

    return final_result, {'top1': top1_winner, 'top2': top2_winner, 'top3': top3_winner}

def _tx_list_from_scores_dict(scores: dict, threshold: float = 0.5):
    if not isinstance(scores, dict): return []
    items = []
    for t, v in scores.items():
        try: v = float(v)
        except Exception: continue
        if v >= threshold: items.append((t, v))
    items.sort(key=lambda x: (-x[1], x[0]))
    return [t for t,_ in items]

def _extract_gt_and_stage(rec):
    rm = rec.get("reward_model", {})
    if isinstance(rm, dict) and isinstance(rm.get("ground_truth"), dict):
        gt = rm["ground_truth"]
        gt_survival = {"survival_month": gt.get("survival_months"), "death": gt.get("if_death")}
        stage_input = {"bclc": gt.get("bclc"), "cnlc": gt.get("cnlc"), "ajcc_tnm": gt.get("tnm") or gt.get("ajcc_tnm")}
        return gt_survival, stage_input
    stage = rec.get("staging", {}); surv = rec.get("survival_status", {})
    if isinstance(stage, dict) and isinstance(surv, dict):
        gt_survival = {"survival_month": surv.get("survival_month"), "death": surv.get("death")}
        stage_input = {"bclc": stage.get("bclc"), "cnlc": stage.get("cnlc"), "ajcc_tnm": stage.get("ajcc_tnm") or stage.get("tnm")}
        return gt_survival, stage_input
    return None, None

def get_new_gt_survival(sample):
    if 'record_0' in sample.get('new_survival', {}) and sample['survival_status'].get('survival_month') is None:
        sample['survival_status']['survival_month'] = sample['new_survival']['record_0'].replace(' 00:00:00','')
        return get_month(sample['survival_status'])
    if sample['survival_status'].get('death') is None and 'if_death' in sample.get('new_survival', {}):
        sample['survival_status']['death'] = sample['new_survival']['if_death']
        return get_month(sample['survival_status'])
    return None

def build_uuid_to_tx_map(jsonl_path: str, threshold: float = 0.5) -> dict:
    rows = load_jsonl(jsonl_path)
    print(f"[build_uuid_to_tx_map] Loading {len(rows)} from {jsonl_path}")
    uuid2tx, bad = {}, 0
    for rec in rows:
        uuid = rec.get("uuid") or rec.get("extra_info", {}).get("uuid")
        if not uuid:
            bad += 1; continue
        text = ""
        llm_resp = rec.get("llm_response", {})
        if isinstance(llm_resp, dict): text = llm_resp.get("text", "") or ""
        tdict = parse_treatment(text)
        uuid2tx[uuid] = _tx_list_from_scores_dict(tdict, threshold=threshold) if isinstance(tdict, dict) else []
    print(f"[build_uuid_to_tx_map] Done. Missing/invalid rows: {bad}")
    return uuid2tx

# ------------------ 外部 LLM 处理（按白名单） ------------------
def process_llm_jsonl_with_whitelist(jsonl_path: str,
                                     out_csv_path: str,
                                     whitelist_uuids: Set[str],
                                     fallback: Dict[str, Dict[str, Any]]):
    """
    仅遍历 whitelist_uuids，保证与主流程样本数一致。
    若外部缺该 uuid 或解析失败，则使用 GT=fallback，预测给默认值（不丢样本）。
    """
    rows = load_jsonl(jsonl_path)
    # 建立 uuid -> rec 映射
    idx = {}
    for rec in rows:
        u = rec.get("uuid") or rec.get("extra_info", {}).get("uuid") or rec.get("id") or rec.get("sample_id")
        if u: idx[u] = rec

    staging_labels, times_gt, events, preds_months = [], [], [], []
    used_fallback, pred_defaulted, missing_in_external = 0, 0, 0

    for u in whitelist_uuids:
        fb = fallback.get(u, {})
        fb_gt = fb.get('gt_survival') or {}
        # 强制使用主流程 GT
        gt_survival_raw = {'survival_month': fb_gt.get('survival_month'), 'death': fb_gt.get('death')}
        used_fallback += 1

        gt_survival = get_month(gt_survival_raw)
        if gt_survival is None or gt_survival.get('survival_month') in (None, 0) or gt_survival['survival_month'] < 0:
            # 理论上不会发生，因为白名单已通过主流程
            continue

        rec = idx.get(u)
        reply_text = ""
        if rec is None:
            missing_in_external += 1
        else:
            llm_resp = rec.get("llm_response", {})
            if isinstance(llm_resp, dict): reply_text = llm_resp.get("text") or ""

        survival_res = evaluate_survival(gt_survival, reply_text) if reply_text else None
        if survival_res is None:
            pred_defaulted += 1
            survival_res = {
                "acc_1yr": 0, "acc_3yr": 0, "acc_5yr": 0,
                "survival_score": 0, "survival_months": 6,
                "risk_score": 6, "survival_rank": "C",
            }

        staging_labels.append(survival_res["survival_rank"])
        preds_months.append(survival_res["risk_score"])
        times_gt.append(gt_survival["survival_month"])
        events.append(gt_survival["if_death"])

    print(f"[process_llm_jsonl_with_whitelist] {os.path.basename(jsonl_path)} | "
          f"whitelist={len(whitelist_uuids)}, used_fallback={used_fallback}, "
          f"pred_defaulted={pred_defaulted}, missing_in_external={missing_in_external}")

    df = pd.DataFrame({"staging": staging_labels, "time": times_gt, "event": events, "predicted": preds_months})
    df.to_csv(out_csv_path, index=False)
    print(f"[process_llm_jsonl_with_whitelist] Saved: {out_csv_path} | N={len(df)}")
    return df

# >>> NEW: SFT 主流程解析（按白名单，与 Ours 相同规则） ------------------
def _pick_replies_from_record(rec):
    """
    兼容不同字段名：优先 result_ours / result / replies / result_sft / result_ours_sft
    """
    for key in ("result_ours", "result", "replies", "result_sft", "result_ours_sft"):
        v = rec.get(key)
        if isinstance(v, list) and len(v) > 0:
            return v
    return None

def process_ours_like_jsonl_with_whitelist(jsonl_path: str,
                                           whitelist_uuids: Set[str],
                                           fallback: Dict[str, Dict[str, Any]],
                                           out_csv_path: str,
                                           tx_threshold: float = 0.5):
    """
    与主流程 Ours 相同的解析与融合逻辑，但严格按 Ours 白名单遍历。
    若该 jsonl 缺失 uuid 或解析失败，则：保留样本，用主流程 GT + 默认预测月数 6 兜底。
    返回：(df, uuid2_tx, uuid2_pred_months, uuid2_stage_label)
    """
    rows = load_jsonl(jsonl_path)
    idx = {}
    for rec in rows:
        u = rec.get("uuid") or rec.get("extra_info", {}).get("uuid") or rec.get("id") or rec.get("sample_id")
        if u: idx[u] = rec

    staging_labels, times_gt, events, preds_months = [], [], [], []
    uuid2_tx, uuid2_pred, uuid2_stage = {}, {}, {}

    for u in whitelist_uuids:
        fb = fallback.get(u, {})
        fb_gt = fb.get('gt_survival') or {}
        gt_survival = get_month({'survival_month': fb_gt.get('survival_month'), 'death': fb_gt.get('death')})
        if gt_survival is None:
            # 极端异常，理论上不会发生
            continue

        rec = idx.get(u)
        replies = _pick_replies_from_record(rec) if rec else None

        valid_survival_results, valid_treatment_predictions = [], []
        if replies:
            for reply in replies:
                sres = evaluate_survival(gt_survival, reply)
                if sres is None:
                    sres = {'acc_1yr':0,'acc_3yr':0,'acc_5yr':0,'survival_score':0,'survival_months':6,'risk_score':6,'survival_rank':'C'}
                tres = parse_treatment(reply)
                if tres is None:
                    tres = {
                        'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0.,
                        'Interventional_therapy': 0., 'Interventional_therapy_plus_ablation': 0.,
                        'Surgical_resection_plus_ablation': 0., 'Systemic_anti-tumor_therapy': 0.,
                        'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                        'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.
                    }
                valid_survival_results.append(sres)
                valid_treatment_predictions.append(tres)
        else:
            # 缺记录：直接兜底
            valid_survival_results = [{'acc_1yr':0,'acc_3yr':0,'acc_5yr':0,'survival_score':0,'survival_months':6,'risk_score':6,'survival_rank':'C'}]
            valid_treatment_predictions = [{
                'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0.,
                'Interventional_therapy': 0., 'Interventional_therapy_plus_ablation': 0.,
                'Surgical_resection_plus_ablation': 0., 'Systemic_anti-tumor_therapy': 0.,
                'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.
            }]

        months = [r['survival_months'] for r in valid_survival_results if r['survival_months'] is not None]
        months = remove_outliers(months) if len(months) else [6]
        final_months = float(np.mean(months)) if len(months) else 6.0
        mapped = min((final_months / 100), 1) * 100
        survival_rank = 'D' if mapped < 4 else ('C' if mapped < 12 else ('B' if mapped < 24 else 'A'))
        risk = final_months

        # major voting 取 tx >= threshold
        tx = []
        mv = major_voting_treatment(valid_treatment_predictions)
        if mv and len(mv) == 2:
            final_tx_scores = mv[0]
            tx = [t for t, v in final_tx_scores.items() if float(v) >= tx_threshold]

        staging_labels.append(survival_rank)
        preds_months.append(risk)
        times_gt.append(gt_survival['survival_month'])
        events.append(gt_survival['if_death'])
        uuid2_tx[u] = tx
        uuid2_pred[u] = risk
        uuid2_stage[u] = survival_rank

    df = pd.DataFrame({"staging": staging_labels, "time": times_gt, "event": events, "predicted": preds_months})
    df.to_csv(out_csv_path, index=False)
    print(f"[process_ours_like_jsonl_with_whitelist] Saved: {out_csv_path} | N={len(df)}")
    return df, uuid2_tx, uuid2_pred, uuid2_stage

# ======================= 主流程 =======================
if __name__ == "__main__":
    # ====== 路径（按需替换）======
    res_path = 'xxx'
    # >>> NEW: SFT 路径
    res_path_sft = 'xxx'

    claude_jsonl = "xxx"
    ds_jsonl     = "xxx"
    gemini_jsonl = "xxx"
    gpt4o_jsonl  = "xxx"
    gpt5_jsonl   = "xxx"

    # ====== 外部模型的 tx 映射（可选）======
    uuid2tx_claude = build_uuid_to_tx_map(claude_jsonl, threshold=0.5)
    uuid2tx_ds     = build_uuid_to_tx_map(ds_jsonl,     threshold=0.5)
    uuid2tx_gemini = build_uuid_to_tx_map(gemini_jsonl, threshold=0.5)
    uuid2tx_gpt4o  = build_uuid_to_tx_map(gpt4o_jsonl,  threshold=0.5)
    uuid2tx_gpt5   = build_uuid_to_tx_map(gpt5_jsonl,   threshold=0.5)

    data = load_jsonl(res_path)
    print(f"Loaded {len(data)} samples from {res_path}")

    # ====== 主流程严格筛选 ======
    fres = []
    whitelist = []  # 通过严格筛选的 uuid 列表（保证三种分期都在）

    your_staging_labels = []
    survival_months_pred = []
    survival_months_gt = []
    death_events = []

    tnm_risk_scores = []
    bclc_risk_scores = []
    cnlc_risk_scores = []

    tnm_staging_labels = []
    bclc_staging_labels = []
    cnlc_staging_labels = []

    survival_miss = {'count': 0}
    bclc_miss = {'count': 0}
    cnlc_miss = {'count': 0}
    ajcc_miss = {'count': 0}

    parse_error_survivial = 0
    parse_error_treatment = 0
    survival_0 = 0

    for sample in data:
        replies = sample.get('result_ours')
        if not isinstance(replies, list) or len(replies) == 0:
            parse_error_survivial += 1
            parse_error_treatment += 1
            continue

        # —— 严格要求：三种分期必须都有，否则跳过
        staging = get_staging(sample.get('staging', {}))
        bclc = staging['bclc']; cnlc = staging['cnlc']; ajcc_tnm = staging['ajcc_tnm']
        if bclc is None: bclc_miss['count'] += 1; continue
        if cnlc is None: cnlc_miss['count'] += 1; continue
        if ajcc_tnm is None: ajcc_miss['count'] += 1; continue

        gt_survival = get_month(sample.get('survival_status', {}))
        if gt_survival is None:
            new_gt = get_new_gt_survival(sample) if 'new_gt_survival' in sample else None
            if new_gt is None:
                survival_miss['count'] += 1
                continue
            gt_survival = new_gt

        if gt_survival['survival_month'] == 0 or gt_survival['survival_month'] < 0:
            survival_0 += 1; continue

        # 汇总多回复
        valid_survival_results, valid_treatment_predictions = [], []
        for reply in replies:
            sres = evaluate_survival(gt_survival, reply)
            if sres is None:
                parse_error_survivial += 1
                sres = {'acc_1yr':0,'acc_3yr':0,'acc_5yr':0,'survival_score':0,'survival_months':6,'risk_score':0,'survival_rank':'C'}
            tres = parse_treatment(reply)
            if tres is None:
                parse_error_treatment += 1
                tres = {
                    'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0.,
                    'Interventional_therapy': 0., 'Interventional_therapy_plus_ablation': 0.,
                    'Surgical_resection_plus_ablation': 0., 'Systemic_anti-tumor_therapy': 0.,
                    'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                    'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.
                }
            valid_survival_results.append(sres)
            valid_treatment_predictions.append(tres)

        if len(valid_survival_results) == 0:
            continue

        months = [r['survival_months'] for r in valid_survival_results if r['survival_months'] is not None]
        if len(months) == 0: continue
        months = remove_outliers(months)
        final_months = float(np.mean(months))

        final_1yr = 1 if final_months > 12 else 0
        final_3yr = 1 if final_months > 36 else 0
        final_5yr = 1 if final_months > 60 else 0

        mapped = min((final_months / 100), 1) * 100
        survival_rank = 'D' if mapped < 4 else ('C' if mapped < 12 else ('B' if mapped < 24 else 'A'))
        risk = final_months

        # 记录 ours
        your_staging_labels.append(survival_rank)
        survival_months_pred.append(risk)
        survival_months_gt.append(gt_survival['survival_month'])
        death_events.append(gt_survival['if_death'])

        tnm_risk_scores.append(ajcc_tnm[0]); tnm_staging_labels.append(ajcc_tnm[1])
        bclc_risk_scores.append(bclc[0]);     bclc_staging_labels.append(bclc[1])
        cnlc_risk_scores.append(cnlc[0]);     cnlc_staging_labels.append(cnlc[1])

        cur_uuid = sample.get("uuid", sample.get("extra_info", {}).get("uuid", ""))
        whitelist.append(cur_uuid)

        # major voting
        treatment_result = major_voting_treatment(valid_treatment_predictions)
        if treatment_result is None or len(treatment_result) != 2:
            parse_error_treatment += 1; continue
        final_treatment_result, voting_result = treatment_result
        # treatment ≥0.5 列表（仅用于 fres['tx_ours']）
        bt = []
        threshold_bt = 0.5
        for tname in final_treatment_result:
            if float(final_treatment_result[tname]) >= threshold_bt:
                bt.append(tname)

        fres.append(
            {
                'uuid': cur_uuid,
                'survival_time': gt_survival['survival_month'],
                'predicted_survival': risk,
                'event': gt_survival['if_death'],
                'ajcc_tnm': ajcc_tnm[1],
                'bclc': bclc[1],
                'cnlc': cnlc[1],
                'tx_actual': get_treatment(sample.get('treatment')),
                'tx_ours': bt,
                # >>> 下面两个字段稍后由 SFT 解析回填
                'tx_ours_sft': [],
                'predicted_survival_sft': None,

                'tx_claude-3-5-sonnet-20241022': uuid2tx_claude.get(cur_uuid, []),
                'tx_deepseek-r1': uuid2tx_ds.get(cur_uuid, []),
                'tx_gemini-2_5-pro': uuid2tx_gemini.get(cur_uuid, []),
                'tx_gpt-4o-2024-08-06': uuid2tx_gpt4o.get(cur_uuid, []),
                'tx_gpt-5': uuid2tx_gpt5.get(cur_uuid, []),
                'tx_bclc': list(bclc_recommended(bclc[1])),
                'tx_cnlc': list(cnlc_recommended(cnlc[1])),
            }
        )

    whitelist_set = set(whitelist)
    print(f"[Ours] Kept N={len(whitelist_set)} samples after strict staging filter.")
    print(f"Missing survival count: {survival_miss['count']}, BCLC miss: {bclc_miss['count']}, CNLC miss: {cnlc_miss['count']}, AJCC miss: {ajcc_miss['count']}")

    # ====== fallback：仅来自主流程通过样本 ======
    fallback_map = {
        row['uuid']: {
            'gt_survival': {'survival_month': row['survival_time'], 'death': row['event']},
            'stage_input': {'bclc': row['bclc'], 'cnlc': row['cnlc'], 'ajcc_tnm': row['ajcc_tnm']},
        }
        for row in fres
    }

    # >>> NEW: 处理 Ours-SFT（与 Ours 相同融合/兜底，严格按白名单）
    df_ours_sft, uuid2tx_sft, uuid2pred_sft, uuid2stage_sft = process_ours_like_jsonl_with_whitelist(
        jsonl_path=res_path_sft,
        whitelist_uuids=whitelist_set,
        fallback=fallback_map,
        out_csv_path="df_ours_sft_cg_3.csv",
        tx_threshold=0.5
    )

    # 将 SFT 结果回填到 fres
    uuid2row = {row['uuid']: row for row in fres}
    for u in whitelist_set:
        if u in uuid2row:
            uuid2row[u]['tx_ours_sft'] = uuid2tx_sft.get(u, [])
            uuid2row[u]['predicted_survival_sft'] = uuid2pred_sft.get(u, None)

    # ====== 外部 LLM（严格按白名单；缺则兜底但不丢样本）======
    process_llm_jsonl_with_whitelist(claude_jsonl, "data/visualization/df_claude-3-5-sonnet-20241022_cg_3.csv", whitelist_set, fallback_map)
    process_llm_jsonl_with_whitelist(ds_jsonl,     "data/visualization/df_deepseek-r1_cg_3.csv",                 whitelist_set, fallback_map)
    process_llm_jsonl_with_whitelist(gemini_jsonl, "data/visualization/df_gemini-2_5-pro_cg_3.csv",              whitelist_set, fallback_map)
    process_llm_jsonl_with_whitelist(gpt4o_jsonl,  "data/visualization/df_gpt-4o-2024-08-06_cg_3.csv",           whitelist_set, fallback_map)
    process_llm_jsonl_with_whitelist(gpt5_jsonl,   "data/visualization/df_gpt-5_cg_3.csv",                       whitelist_set, fallback_map)

    # ====== 导出 ours 及各分期 df（与白名单等长，必相等）======
    df_ours = pd.DataFrame({'staging': your_staging_labels, 'time': survival_months_gt, 'event': death_events, 'predicted': survival_months_pred})
    df_tnm  = pd.DataFrame({'staging': tnm_staging_labels,  'time': survival_months_gt, 'event': death_events, 'predicted': tnm_risk_scores})
    df_bclc = pd.DataFrame({'staging': bclc_staging_labels, 'time': survival_months_gt, 'event': death_events, 'predicted': bclc_risk_scores})
    df_cnlc = pd.DataFrame({'staging': cnlc_staging_labels, 'time': survival_months_gt, 'event': death_events, 'predicted': cnlc_risk_scores})

    # --- 校验长度一致性 ---
    assert len(df_ours) == len(df_tnm) == len(df_bclc) == len(df_cnlc) == len(whitelist_set), "主流程各 df 条数不一致！"
    assert len(df_ours_sft) == len(whitelist_set), "Ours-SFT df 条数与白名单不一致！"

    # ====== 绘图 & 指标（保留对 Ours 的原有输出；如需 SFT 也画，解开注释）======
    plot_km(df_ours, 'data/visualization/km_overall_cg_3.svg')
    plot_km_capped(df_ours, 'data/visualization/km_1yr_cg_3.svg', horizon_months=12)
    plot_km_capped(df_ours, 'data/visualization/km_2yr_cg_3.svg', horizon_months=24)
    plot_km_capped(df_ours, 'data/visualization/km_3yr_cg_3.svg', horizon_months=36)
    plot_km_capped(df_ours, 'data/visualization/km_4yr_cg_3.svg', horizon_months=48)
    plot_km_capped(df_ours, 'data/visualization/km_5yr_cg_3.svg', horizon_months=60)

    print(f'Total Number of Samples (all outputs): {len(df_ours)}')
    print('Our concordance index: {:.4f}'.format(concordance_index(df_ours['time'], df_ours['predicted'], df_ours['event'])))
    print('Our-SFT concordance index: {:.4f}'.format(concordance_index(df_ours_sft['time'], df_ours_sft['predicted'], df_ours_sft['event'])))

    if len(df_tnm) > 0:
        print('TNM concordance index: {:.4f}'.format(concordance_index(df_tnm['time'], df_tnm['predicted'], df_tnm['event'])))
        print('BCLC concordance index: {:.4f}'.format(concordance_index(df_bclc['time'], df_bclc['predicted'], df_bclc['event'])))
        print('CNLC concordance index: {:.4f}'.format(concordance_index(df_cnlc['time'], df_cnlc['predicted'], df_cnlc['event'])))

    # ====== 导出 ======
    df_ours.to_csv('data/visualization/df_ours_cg_3.csv', index=False)
    df_tnm.to_csv('data/visualization/df_tnm_cg_3.csv', index=False)
    df_bclc.to_csv('data/visualization/df_bclc_cg_3.csv', index=False)
    df_cnlc.to_csv('data/visualization/df_cnlc_cg_3.csv', index=False)
    # Ours-SFT 已在函数内导出 df_ours_sft_cg_3.csv

    # 保存融合明细（仅白名单样本；已加入 tx_ours_sft & predicted_survival_sft）
    with open("data/visualization/staging_survival_cg_3.jsonl", "w", encoding="utf-8") as f:
        for row in fres:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
