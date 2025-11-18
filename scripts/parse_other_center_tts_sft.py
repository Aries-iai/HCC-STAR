#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import ast
import pandas as pd
from numpy import exp
import os
from collections import defaultdict, Counter
from tqdm import tqdm
from datetime import datetime, date
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Set, Tuple

# -------------------- 中文→英文治疗名 --------------------
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

# -------------------- 工具：清洗 source 为安全文件名 --------------------
def _sanitize_name(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

# -------------------- 分期标准化 --------------------
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
    mapping = {
        "IA": "Ia", "IB": "Ib",
        "IIA": "IIa", "IIB": "IIb",
        "IIIA": "IIIa", "IIIB": "IIIb",
        "IV": "IV"
    }
    return mapping.get(s, "")

# -------------------- 指南推荐映射 --------------------
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

def is_recommended(treatment: str, stage_raw: str, scheme: str) -> bool:
    scheme = scheme.strip().upper()
    rec = bclc_recommended(stage_raw) if scheme == "BCLC" else cnlc_recommended(stage_raw)
    return treatment in rec

# -------------------- KM 绘图 --------------------
def plot_km(df, save_path, time_col='time', event_col='event'):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    df_plot = df[[time_col, event_col, 'staging']].copy()
    df_plot[time_col] = pd.to_numeric(df_plot[time_col], errors='coerce')
    df_plot[event_col] = pd.to_numeric(df_plot[event_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[time_col, event_col, 'staging'])
    for group_name, group_data in df_plot.groupby('staging'):
        kmf.fit(durations=group_data[time_col],
                event_observed=group_data[event_col],
                label=str(group_name))
        kmf.plot_survival_function(ci_show=False)
    plt.title("Kaplan–Meier (Overall)")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Staging")
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()

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
    df_plot['time_capped'] = time_capped
    df_plot['event_within'] = event_within
    for group_name, group_data in df_plot.groupby('staging'):
        kmf.fit(durations=group_data['time_capped'],
                event_observed=group_data['event_within'],
                label=str(group_name))
        kmf.plot_survival_function(ci_show=False)
    plt.title(f"Kaplan–Meier (Task: ≤{horizon_months//12} year survival)")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.xlim(0, horizon_months)
    plt.axvline(horizon_months, linestyle='--', linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Staging")
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()

# -------------------- 安全转换 --------------------
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

# -------------------- 生存评分 --------------------
def _months_score_with_event(pred_months, gt_months, event,
                             gt_flags=None, tau_death=12.0, tau_censor=6.0):
    if pred_months is None:
        return 0.0
    if event == 1:
        if gt_months is None:
            return 0.0
        diff = abs(pred_months - gt_months)
        return exp(-diff / float(tau_death))
    if event == 0:
        if gt_months is not None:
            t_c = gt_months
        else:
            t_c = _max_passed_threshold_months_from_flags(gt_flags or {})
            if t_c is None:
                return 0.5
        d = max(0.0, float(t_c) - float(pred_months))
        return exp(-d / float(tau_censor))
    return 0.5

def _max_passed_threshold_months_from_flags(flags):
    mapping = [("survival_1yr", 12), ("survival_3yr", 36), ("survival_5yr", 60)]
    passed = [m for k, m in mapping if flags.get(k) == 1]
    return max(passed) if passed else None

# -------------------- IO --------------------
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def read_jsonl_files(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
    return data

def save_jsonl(records, path, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------- 解析器 --------------------
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
    return {
        "survival_months": pred_months,
        "survival_1yr": pred_1yr,
        "survival_3yr": pred_3yr,
        "survival_5yr": pred_5yr,
    }

def parse_treatment(res):
    m = re.search(r"<hard_check_treatment>(.*?)</hard_check_treatment>", res, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    l = raw.find("{")
    r = raw.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    body = raw[l:r+1].strip()
    obj = None
    try:
        obj = json.loads(body)
    except Exception:
        try:
            obj = json.loads(body.replace("'", '"'))
        except Exception:
            try:
                obj = ast.literal_eval(body)
            except Exception:
                return None
    if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
        obj = obj["scores"]
    return obj if isinstance(obj, dict) else None

# -------------------- 提取/映射 --------------------
def get_treatment(treatment):
    if isinstance(treatment, dict):
        res = treatment['scores']
        try:
            if isinstance(res, str):
                res = ast.literal_eval(res)
                return res['treatment']
        except Exception:
            return None
    if isinstance(treatment, str):
        return cn2en.get(treatment)

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

def calculate_topn_accuracy_from_voting(voting_result, gt_treatment, n):
    if gt_treatment is None or voting_result is None:
        return 0
    top_predictions = []
    for i in range(1, n+1):
        top_key = f'top{i}'
        if top_key in voting_result and voting_result[top_key] is not None:
            top_predictions.append(voting_result[top_key])
    return 1 if gt_treatment in top_predictions else 0

def evaluate_survival(gt, res):
    def cal_correctness(pred_survival, gt_survival):
        if gt_survival is None:
            return None
        if pred_survival is None:
            return 0.0
        if isinstance(pred_survival, bool):
            pred_survival = 1 if pred_survival else 0
        if isinstance(gt_survival, bool):
            gt_survival = 1 if gt_survival else 0
        return 1.0 if pred_survival == gt_survival else 0.0

    survival_months = gt.get('survival_month')
    survival_1yr = gt.get('survival_1yr')
    survival_3yr = gt.get('survival_3yr')
    survival_5yr = gt.get('survival_5yr')
    pred = parse_hard_check_survival(res)
    if pred is None:
        print("parse_hard_check_survival failed")
        return None
    pred_survival_month = pred.get('survival_months')
    print(pred_survival_month)
    if pred_survival_month is None:
        return None
    pred_survival_1yr = 1 if pred_survival_month > 12 else 0
    pred_survival_3yr = 1 if pred_survival_month > 36 else 0
    pred_survival_5yr = 1 if pred_survival_month > 60 else 0

    mapped_month = min((pred_survival_month / 100), 1) * 100
    if 0 <= mapped_month < 4:
        survival_rank = 'D'
    elif 4 <= mapped_month < 12:
        survival_rank = 'C'
    elif 12 <= mapped_month < 24:
        survival_rank = 'B'
    else:
        survival_rank = 'A'

    risk = pred_survival_month
    acc_1yr = cal_correctness(pred_survival_1yr, survival_1yr)
    acc_3yr = cal_correctness(pred_survival_3yr, survival_3yr)
    acc_5yr = cal_correctness(pred_survival_5yr, survival_5yr)

    survival_score = _months_score_with_event(
        pred_months=pred_survival_month,
        gt_months=survival_months,
        event=gt.get('if_death'),
        gt_flags={
            'survival_1yr': survival_1yr,
            'survival_3yr': survival_3yr,
            'survival_5yr': survival_5yr,
            'survival_months': survival_months
        }
    )

    return {
        'acc_1yr': acc_1yr,
        'acc_3yr': acc_3yr,
        'acc_5yr': acc_5yr,
        'survival_score': survival_score,
        'survival_months': pred_survival_month,
        'risk_score': risk,
        'survival_rank': survival_rank,
    }

def map_survival(if_death):
    if if_death == '否' or if_death == 0:
        return 0
    if if_death:
        return 1
    if if_death is not None:
        print('if_death: ',if_death)
    return None

def calculate_survival_months(start_date, end_date) -> int:
    start_date = start_date.replace('UK','01')
    end_date = end_date.replace('UK','01')
    def to_datetime(d):
        if isinstance(d, (datetime, date)):
            return datetime(d.year, d.month, d.day)
        if isinstance(d, str):
            for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
                try:
                    return datetime.strptime(d, fmt)
                except ValueError:
                    continue
        return None
    start = to_datetime(start_date)
    end = to_datetime(end_date)
    if not start or not end:
        return None
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return months

def get_month(sample):
    if sample is None:
        return None
    survival_month = sample['survival_month']
    if '\n' in str(survival_month):
        survival_month = survival_month.split('\n')[-1].strip()
    if_death = map_survival(sample['death'])
    if if_death is None:
        if 'in_time' in sample and 'death_time' in sample and sample['death_time'] and sample['in_time']:
            if_death = 1
        elif 'in_time' in sample and 'survival_month' in sample and sample['survival_month'] and sample['in_time']:
            if_death = 0
        else:
            return None
    if 'in_time' in sample and 'death_time' in sample:
        in_time = sample['in_time']; death_time = sample['death_time']
    if isinstance(survival_month, str):
        try:
            return {'survival_month': _safe_to_int(survival_month), 'if_death': if_death}
        except ValueError:
            if if_death:
                if in_time and death_time:
                    sm = calculate_survival_months(in_time, death_time)
                    if sm is None:
                        print(f'sm is None, in_time: {in_time}, death_time: {death_time}, survival_month: {survival_month}')
                    return {'survival_month': sm, 'if_death': if_death}
                elif death_time is None and in_time and survival_month:
                    sm = calculate_survival_months(in_time, survival_month)
                    if sm is None:
                        print(f'sm is None, in_time: {in_time}, death_time: {death_time}, survival_month: {survival_month}')
                    return {'survival_month': sm, 'if_death': 0}
                else:
                    return None
            else:
                if in_time and survival_month:
                    sm = calculate_survival_months(in_time, survival_month)
                    if sm is None:
                        print(f'sm is None, in_time: {in_time}, survival_month: {survival_month}')
                    return {'survival_month': sm, 'if_death': if_death}
                else:
                    return None
    elif isinstance(survival_month, int):
        return {'survival_month': survival_month, 'if_death': if_death}
    elif isinstance(survival_month, float):
        return {'survival_month': _safe_to_int(survival_month), 'if_death': if_death}
    else:
        return None

def get_staging(sample):
    def stage_from_tnm(t: str, n: str, m: str):
        if not isinstance(t, str): t = str(t or "")
        if not isinstance(n, str): n = str(n or "")
        if not isinstance(m, str): m = str(m or "")
        t, n, m = t.upper(), n.upper(), m.upper()
        if 'M1' in m:
            return -5, 'IV'
        if 'N1' in n:
            return -4, 'IIIC'
        if 'T1' in t:
            return 0, 'I'
        elif 'T2' in t:
            return -1, 'II'
        elif 'T3' in t:
            return -2, 'IIIA'
        elif 'T4' in t:
            return -3, 'IIIB'
        return None

    def get_bclc(bclc):
        if bclc is None or (isinstance(bclc, float) and pd.isna(bclc)):
            return None
        s = norm_bclc_stage(str(bclc))
        score_map = {"0": 0, "A": -1, "B": -2, "C": -3, "D": -4}
        return (score_map[s], s) if s in score_map else None

    def get_cnlc(cnlc):
        if cnlc is None or (isinstance(cnlc, float) and pd.isna(cnlc)):
            return None
        s = norm_cnlc_stage(str(cnlc))
        score_map = {"Ia": 0, "Ib": -1, "IIa": -2, "IIb": -3, "IIIa": -4, "IIIb": -5, "IV": -6}
        return (score_map[s], s) if s in score_map else None

    def get_ajcc_tnm(ajcc_tnm):
        if ajcc_tnm is None or (isinstance(ajcc_tnm, float) and pd.isna(ajcc_tnm)):
            return None
        if isinstance(ajcc_tnm, dict):
            T, N, M = ajcc_tnm.get('T'), ajcc_tnm.get('N'), ajcc_tnm.get('M')
            if T is None or N is None or M is None:
                return None
            return stage_from_tnm(T, N, M)
        s = str(ajcc_tnm).upper()
        if "IIIA" in s: return -2, "IIIa"
        if "IIIB" in s: return -3, "IIIb"
        if "IIIC" in s: return -4, "III"
        if "III"  in s: return -2, "IIIa"
        if "IV"   in s: return -5, "IV"
        if "II"   in s: return -1, "II"
        if "I"    in s: return 0,  "I"
        return None

    bclc = sample.get('bclc')
    cnlc = sample.get('cnlc')
    ajcc_tnm = sample.get('ajcc_tnm') or sample.get('tnm')
    return {
        'bclc': get_bclc(bclc),
        'cnlc': get_cnlc(cnlc),
        'ajcc_tnm': get_ajcc_tnm(ajcc_tnm),
    }

def _extract_gt_and_stage(rec):
    rm = rec.get("reward_model", {})
    if isinstance(rm, dict) and isinstance(rm.get("ground_truth"), dict):
        gt = rm["ground_truth"]
        gt_survival = {"survival_month": gt.get("survival_months"), "death": gt.get("if_death")}
        stage_input = {"bclc": gt.get("bclc"), "cnlc": gt.get("cnlc"), "ajcc_tnm": gt.get("tnm") or gt.get("ajcc_tnm")}
        return gt_survival, stage_input
    stage = rec.get("staging", {})
    surv  = rec.get("survival_status", {})
    if isinstance(stage, dict) and isinstance(surv, dict):
        gt_survival = {"survival_month": surv.get("survival_month"), "death": surv.get("death")}
        stage_input = {"bclc": stage.get("bclc"), "cnlc": stage.get("cnlc"), "ajcc_tnm": stage.get("ajcc_tnm") or stage.get("tnm")}
        return gt_survival, stage_input
    return None, None

# -------------------- 评估指标 --------------------
def treatment_eval_score_2(gt, pred, thres):
    if gt is None or pred is None:
        return 0.0
    if not isinstance(pred, dict):
        print(f'Invalid pred format: {pred}')
        return 0.0
    if 'scores' in pred:
        pred = pred['scores']
    if not isinstance(pred, dict):
        print(f'Invalid pred format after extracting scores: {pred}')
        return 0.0
    if gt not in pred:
        return 0.0
    gt_score = pred[gt]
    if gt_score < thres:
        return 0.0
    count_above_thres = sum(1 for score in pred.values() if float(score) >= thres)
    if count_above_thres == 0:
        return 0.0
    return 1.0 / count_above_thres

def weighted_topn_similarity_v2(score1, score2, n=1):
    def get_rank_dict(score_dict):
        sorted_items = sorted(score_dict.items(), key=lambda x: -x[1])
        rank_dict = {}
        last_score = None
        rank = -1
        for key, score in sorted_items:
            if score != last_score:
                rank += 1
            rank_dict[key] = rank
            last_score = score
        return rank_dict
    rank1 = get_rank_dict(score1); rank2 = get_rank_dict(score2)
    sim_score = 0.0; score_list = []
    for key in rank1:
        if key in rank2:
            r1, r2 = rank1[key], rank2[key]
            if r1 < n and r2 < n:
                sim_score += 1 / (1 + abs(r1 - r2))
                score_list.append((key, r1, r2))
    return {'score': sim_score / n, 'score_list': score_list}

def weighted_topn_similarity(score1, score2, n=1):
    def get_rank_dict(score_dict):
        sorted_items = sorted(score_dict.items(), key=lambda x: (-x[1], x[0]))
        seen = set(); rank_dict = {}; rank = 0
        for key, score in sorted_items:
            if key not in seen:
                rank_dict[key] = rank; seen.add(key); rank += 1
        return rank_dict
    rank1 = get_rank_dict(score1); rank2 = get_rank_dict(score2)
    sim_score = 0.0; score_list = []
    for key in rank1:
        if key in rank2:
            r1, r2 = rank1[key], rank2[key]
            if r1 < n and r2 < n:
                sim_score += 1 / (1 + abs(r1 - r2))
                score_list.append((key, r1, r2))
    return {'score': sim_score / n, 'score_list': score_list}

def save_histogram(data, filename="histogram.png", bins=20):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.xlabel("Value"); plt.ylabel("Frequency"); plt.title("Distribution of Data")
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()
    print(f"直方图已保存到 {filename}")

# -------------------- 新/缺失生存填补 --------------------
def get_new_gt_survival(sample):
    if 'record_0' in sample['new_survival'] and sample['survival_status']['survival_month'] is None:
        sample['survival_status']['survival_month'] = sample['new_survival']['record_0'].replace(' 00:00:00','')
        return get_month(sample['survival_status'])
    if sample['survival_status']['death'] is None and 'if_death' in sample['new_survival']:
        sample['survival_status']['death'] = sample['new_survival']['if_death']
        return get_month(sample['survival_status'])
    print('original survival: ', sample['survival_status'])
    print('new_survival:', sample['new_survival'])
    return None

# -------------------- 汇总/异常值工具 --------------------
def add_miss(miss_dict, source):
    miss_dict['count'] += 1
    miss_dict[source] = miss_dict.get(source, 0) + 1

def add_res(res_dict, source, value):
    res_dict['sum'].append(value)
    res_dict[source] = res_dict.get(source, []) + [value]

def remove_outliers(values, threshold=1):
    if len(values) <= 1:
        return values
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    print(f"原始值: {values}")
    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    print(f"筛选范围: [{lower_bound}, {upper_bound}]")
    print(f"筛选后: {filtered}")
    return filtered if len(filtered) > 0 else values

# -------------------- 由 LLM JSONL 构建 uuid→tx 列表映射 --------------------
def _tx_list_from_scores_dict(scores: dict, threshold: float = 0.5):
    if not isinstance(scores, dict):
        return []
    items = []
    for t, v in scores.items():
        try:
            v = float(v)
        except Exception:
            continue
        if v >= threshold:
            items.append((t, v))
    items.sort(key=lambda x: (-x[1], x[0]))
    return [t for t, _ in items]

def build_uuid_to_tx_map(jsonl_path: str, threshold: float = 0.5) -> dict:
    rows = load_jsonl(jsonl_path)
    print(f"[build_uuid_to_tx_map] Loading {len(rows)} from {jsonl_path}")
    uuid2tx = {}; bad = 0
    for rec in rows:
        uuid = None
        try:
            uuid = rec.get("uuid", rec.get("extra_info", {}).get("uuid"))
        except Exception:
            pass
        if not uuid:
            bad += 1; continue
        text = ""
        llm_resp = rec.get("llm_response", {})
        if isinstance(llm_resp, dict):
            text = llm_resp.get("text", "") or ""
        tdict = parse_treatment(text)
        if tdict is None or not isinstance(tdict, dict):
            uuid2tx[uuid] = []
            continue
        uuid2tx[uuid] = _tx_list_from_scores_dict(tdict, threshold=threshold)
    print(f"[build_uuid_to_tx_map] Done. Missing/invalid rows: {bad}")
    return uuid2tx

# -------------------- 解析外部 LLM jsonl → CSV（支持按 source 拆分） --------------------
def process_llm_jsonl(jsonl_path: str, out_csv_path: str, uuid2source: dict = None):
    rows = load_jsonl(jsonl_path)
    print(f"[process_llm_jsonl] Loaded {len(rows)} rows from {jsonl_path}")

    staging_labels, times_gt, events, preds_months, uuids = [], [], [], [], []

    miss_gt = parse_error_survival = skip_stage_fail = skip_zero_or_neg = 0

    for rec in rows:
        uu = rec.get("uuid", rec.get("extra_info", {}).get("uuid"))
        uuids.append(uu)

        gt_survival_raw, stage_input = _extract_gt_and_stage(rec)
        if gt_survival_raw is None or stage_input is None:
            miss_gt += 1
            staging_labels.append(None); preds_months.append(None); times_gt.append(None); events.append(None)
            continue

        gt_survival = get_month(gt_survival_raw)
        if gt_survival is None:
            miss_gt += 1
            staging_labels.append(None); preds_months.append(None); times_gt.append(None); events.append(None)
            continue
        if not gt_survival.get("survival_month") or gt_survival["survival_month"] <= 0:
            skip_zero_or_neg += 1
            staging_labels.append(None); preds_months.append(None); times_gt.append(None); events.append(None)
            continue

        st = get_staging(stage_input)
        if st["bclc"] is None or st["cnlc"] is None or st["ajcc_tnm"] is None:
            skip_stage_fail += 1
            staging_labels.append(None); preds_months.append(None); times_gt.append(None); events.append(None)
            continue

        reply_text = ""
        llm_resp = rec.get("llm_response", {})
        if isinstance(llm_resp, dict):
            reply_text = llm_resp.get("text") or ""
        survival_res = evaluate_survival(gt_survival, reply_text)
        if survival_res is None:
            parse_error_survival += 1
            survival_res = {"survival_rank":"C","risk_score":6}
        staging_labels.append(survival_res["survival_rank"])
        preds_months.append(survival_res["risk_score"])
        times_gt.append(gt_survival["survival_month"])
        events.append(gt_survival["if_death"])

    df = pd.DataFrame({
        "uuid": uuids,
        "staging": staging_labels,
        "time": times_gt,
        "event": events,
        "predicted": preds_months
    }).dropna(subset=["time","event","predicted","staging"])
    df.to_csv(out_csv_path, index=False)
    print(f"[process_llm_jsonl] Saved: {out_csv_path} | N={len(df)} | GT miss={miss_gt}, stage_fail={skip_stage_fail}, parse_err={parse_error_survival}, zero/neg={skip_zero_or_neg}")

    # 如果提供了 uuid→source 映射，则按 source 拆分保存
    if uuid2source is not None:
        df["source"] = df["uuid"].map(lambda u: _sanitize_name(uuid2source.get(u, "unknown")))
        for src, dfg in df.groupby("source"):
            base = os.path.splitext(out_csv_path)[0]
            p = f"{base}_{src}.csv"
            dfg.drop(columns=["source"]).to_csv(p, index=False)
            print(f"[process_llm_jsonl] Per-source saved: {p} | n={len(dfg)}")

    return df

# >>> NEW: 取 replies 的通用函数（兼容不同 key）
def _pick_replies(rec):
    for k in ("result_ours", "result", "replies", "result_sft", "result_ours_sft"):
        v = rec.get(k)
        if isinstance(v, list) and len(v) > 0:
            return v
    return None

# >>> NEW: 并列处理 SFT（严格按 Ours 白名单 UUID；缺记录/解析失败兜底）
def process_ours_like_jsonl_with_whitelist(jsonl_path: str,
                                           whitelist_uuids: Set[str],
                                           fallback_map: dict,
                                           uuid2source_map: dict,
                                           out_csv_path: str,
                                           tx_threshold: float = 0.5):
    rows = load_jsonl(jsonl_path)
    idx = {}
    for rec in rows:
        u = rec.get("uuid", rec.get("extra_info", {}).get("uuid"))
        if u: idx[u] = rec

    staging_labels, times_gt, events, preds_months, sources = [], [], [], [], []
    uuid2_tx, uuid2_pred = {}, {}

    for u in whitelist_uuids:
        fb = fallback_map.get(u, {})
        gt_survival = get_month({'survival_month': fb.get('survival_time'), 'death': fb.get('event')})
        if gt_survival is None:
            continue

        rec = idx.get(u, {})
        replies = _pick_replies(rec)

        valid_survival_results, valid_treatment_predictions = [], []
        if replies:
            for reply in replies:
                sres = evaluate_survival(gt_survival, reply)
                if sres is None:
                    sres = {"acc_1yr":0,"acc_3yr":0,"acc_5yr":0,"survival_score":0,"survival_months":6,"risk_score":6,"survival_rank":"C"}
                tres = parse_treatment(reply)
                if tres is None:
                    tres = {'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0., 'Interventional_therapy': 0.,
                            'Interventional_therapy_plus_ablation': 0., 'Surgical_resection_plus_ablation': 0.,
                            'Systemic_anti-tumor_therapy': 0., 'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                            'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.}
                valid_survival_results.append(sres)
                valid_treatment_predictions.append(tres)
        else:
            valid_survival_results = [{"acc_1yr":0,"acc_3yr":0,"acc_5yr":0,"survival_score":0,"survival_months":6,"risk_score":6,"survival_rank":"C"}]
            valid_treatment_predictions = [{'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0., 'Interventional_therapy': 0.,
                                            'Interventional_therapy_plus_ablation': 0., 'Surgical_resection_plus_ablation': 0.,
                                            'Systemic_anti-tumor_therapy': 0., 'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                                            'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.}]

        months = [r['survival_months'] for r in valid_survival_results if r['survival_months'] is not None]
        months = remove_outliers(months) if len(months) else [6]
        # 与主流程一致：均值 + 0.5*std
        final_months = float(np.mean(months) + 0.5 * (np.std(months) if len(months) > 1 else 0.0))
        mapped = min((final_months / 100), 1) * 100
        survival_rank = 'D' if mapped < 4 else ('C' if mapped < 12 else ('B' if mapped < 24 else 'A'))
        risk = final_months

        # major voting（与主流程相同权重 1.0/0.6/0.3）
        votes = {'top1': [], 'top2': [], 'top3': []}
        def _topn(d, n=3):
            if not isinstance(d, dict): return []
            return [k for k,_ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]]
        for pdict in valid_treatment_predictions:
            tops = _topn(pdict, 3)
            if len(tops)>=1: votes['top1'].append(tops[0])
            if len(tops)>=2: votes['top2'].append(tops[1])
            if len(tops)>=3: votes['top3'].append(tops[2])
        mv_soft = {}
        def _winner(vs): 
            return Counter(vs).most_common(1)[0][0] if vs else None
        t1 = _winner(votes['top1']); t2 = _winner(votes['top2']); t3 = _winner(votes['top3'])
        if t1: mv_soft[t1] = 1.0
        if t2 and t2 != t1: mv_soft[t2] = 0.6
        if t3 and t3 not in (t1, t2): mv_soft[t3] = 0.3
        tx = [t for t,v in mv_soft.items() if float(v) >= tx_threshold]

        staging_labels.append(survival_rank)
        preds_months.append(risk)
        times_gt.append(gt_survival['survival_month'])
        events.append(gt_survival['if_death'])
        sources.append(_sanitize_name(uuid2source_map.get(u, "unknown")))
        uuid2_tx[u] = tx
        uuid2_pred[u] = risk

    df = pd.DataFrame({"uuid": list(whitelist_uuids),
                       "staging": staging_labels,
                       "time": times_gt,
                       "event": events,
                       "predicted": preds_months,
                       "source": sources})
    df = df.dropna(subset=["time","event","predicted","staging"])
    df.to_csv(out_csv_path, index=False)
    print(f"[process_ours_like_jsonl_with_whitelist] Saved: {out_csv_path} | N={len(df)}")
    # 按 source 拆分
    for src, dfg in df.groupby("source"):
        p = f"{os.path.splitext(out_csv_path)[0]}_{src}.csv"
        dfg.drop(columns=["source"]).to_csv(p, index=False)
        print(f"[SFT per-source] saved: {p} | n={len(dfg)}")
    return df, uuid2_tx, uuid2_pred

# ============================= 主程序 =============================
if __name__ == "__main__":
    # Ours 主数据
    res_path = 'xxx'
    # >>> NEW: SFT 数据
    res_path_sft = 'xxx'

    # 外部 LLM 结果路径
    claude_jsonl = "xxx"
    ds_jsonl = "xxx"
    gemini_jsonl = "xxx"
    gpt4o_jsonl = "xxx"
    gpt5_jsonl = "xxx"

    # ==== 先构建外部 LLM 的 uuid→tx 列表（>=0.5） ====
    uuid2tx_claude = build_uuid_to_tx_map(claude_jsonl, threshold=0.5)
    uuid2tx_ds = build_uuid_to_tx_map(ds_jsonl, threshold=0.5)
    uuid2tx_gemini = build_uuid_to_tx_map(gemini_jsonl, threshold=0.5)
    uuid2tx_gpt4o = build_uuid_to_tx_map(gpt4o_jsonl, threshold=0.5)
    uuid2tx_gpt5 = build_uuid_to_tx_map(gpt5_jsonl, threshold=0.5)

    # ==== 载入主数据，并记录 uuid→source，供后续按 source 拆分 ====
    data = load_jsonl(res_path)
    print(f"Loaded {len(data)} samples from {res_path}")
    uuid2source = {}
    for sample in data:
        u = sample.get("uuid", sample.get("extra_info", {}).get("uuid"))
        s = _sanitize_name(sample.get("source", "unknown"))
        if u:
            uuid2source[u] = s

    # ---- 收集器（含 source 列表） ----
    your_staging_labels = []; survival_months_pred = []; survival_months_gt = []; death_events = []; sources_ours = []
    tnm_risk_scores = []; bclc_risk_scores = []; cnlc_risk_scores = []
    tnm_death_events = []; bclc_death_events = []; cnlc_death_events = []
    tnm_survival_months_gt = []; bclc_survival_months_gt = []; cnlc_survival_months_gt = []
    tnm_sources = []; bclc_sources = []; cnlc_sources = []
    tnm_staging_labels = []; bclc_staging_labels = []; cnlc_staging_labels = []

    fres = []  # 将包含 'source' 字段，便于后续分源落盘

    top1_scores_treatment_list = {'sum': []}
    treatment_score_2_list = {'sum': []}
    if_treatment_hit_list = {'sum': []}
    top1_accuracy_list = {'sum': []}
    top2_accuracy_list = {'sum': []}
    top3_accuracy_list = {'sum': []}

    survival_miss = {'count': 0}
    bclc_miss = {'count': 0}
    cnlc_miss = {'count': 0}
    ajcc_miss = {'count': 0}

    parse_error_survivial = 0
    parse_error_treatment = 0
    survival_0 = 0

    whitelist = []  # >>> NEW: Ours 白名单 UUID

    for sample in data:
        replies = sample['result_ours']
        if not isinstance(replies, list) or len(replies) == 0:
            print(f"No replies for sample: {sample.get('uuid', 'unknown')}")
            parse_error_survivial += 1; parse_error_treatment += 1
            continue

        source_raw = sample.get('source', 'unknown')
        src = _sanitize_name(source_raw)

        staging = get_staging(sample['staging'])
        bclc = staging['bclc']; cnlc = staging['cnlc']; ajcc_tnm = staging['ajcc_tnm']
        if bclc is None: add_miss(bclc_miss, src)
        if cnlc is None: add_miss(cnlc_miss, src)
        if ajcc_tnm is None: add_miss(ajcc_miss, src)
        if bclc is None or cnlc is None or ajcc_tnm is None:
            continue

        gt_survival = get_month(sample['survival_status'])
        if gt_survival is None:
            if 'new_gt_survival' in sample:
                new_gt_survival = get_new_gt_survival(sample)
                if new_gt_survival is not None:
                    gt_survival = new_gt_survival
                else:
                    print('original survival: ', sample['survival_status'])
                    print('new_survival:', sample['new_survival'])
                    add_miss(survival_miss, src); continue
            else:
                add_miss(survival_miss, src); continue

        if gt_survival['survival_month'] == 0:
            survival_0 += 1; continue
        if gt_survival['survival_month'] < 0:
            survival_0 += 1; print(sample.get("uuid")); continue

        valid_survival_results = []; valid_treatment_results = []
        for reply in replies:
            survival_res = evaluate_survival(gt_survival, reply)
            if survival_res is None:
                parse_error_survivial += 1
                survival_res = {
                    'acc_1yr': 0, 'acc_3yr': 0, 'acc_5yr': 0,
                    'survival_score': 0, 'survival_months': 6,
                    'risk_score': 0, 'survival_rank': 'C',
                }
            treatment_res = parse_treatment(reply)
            if treatment_res is None:
                parse_error_treatment += 1
                treatment_res = {
                    'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0.,
                    'Interventional_therapy': 0., 'Interventional_therapy_plus_ablation': 0.,
                    'Surgical_resection_plus_ablation': 0., 'Systemic_anti-tumor_therapy': 0.,
                    'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                    'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.
                }
            valid_survival_results.append(survival_res)
            valid_treatment_results.append(treatment_res)

        if len(valid_survival_results) == 0:
            parse_error_survivial += 1; continue
        if len(valid_treatment_results) == 0:
            parse_error_treatment += 1; continue

        # 生存：去异常后平均 + 0.5*std（保持你当前逻辑）
        months = [r['survival_months'] for r in valid_survival_results if r['survival_months'] is not None]
        if len(months) == 0:
            continue
        filtered_months = remove_outliers(months)
        final_months = np.mean(filtered_months) + 0.5 * np.std(filtered_months)
        final_1yr = 1 if final_months > 12 else 0
        final_3yr = 1 if final_months > 36 else 0
        final_5yr = 1 if final_months > 60 else 0
        mapped_month = min((final_months / 100), 1) * 100
        if 0 <= mapped_month < 4: survival_rank = 'D'
        elif 4 <= mapped_month < 12: survival_rank = 'C'
        elif 12 <= mapped_month < 24: survival_rank = 'B'
        else: survival_rank = 'A'
        risk = final_months

        def cal_correctness(pred_survival, gt_survival):
            if gt_survival is None: return None
            if pred_survival is None: return 0.0
            if isinstance(pred_survival, bool): pred_survival = 1 if pred_survival else 0
            if isinstance(gt_survival, bool): gt_survival = 1 if gt_survival else 0
            return 1.0 if pred_survival == gt_survival else 0.0

        survival_months_gt_val = gt_survival.get('survival_month')
        survival_1yr_gt = gt_survival.get('survival_1yr', None)
        survival_3yr_gt = gt_survival.get('survival_3yr', None)
        survival_5yr_gt = gt_survival.get('survival_5yr', None)
        if_death_gt = gt_survival.get('if_death')
        acc_1yr = cal_correctness(final_1yr, survival_1yr_gt)
        acc_3yr = cal_correctness(final_3yr, survival_3yr_gt)
        acc_5yr = cal_correctness(final_5yr, survival_5yr_gt)

        final_survival_score = _months_score_with_event(
            pred_months=final_months,
            gt_months=survival_months_gt_val,
            event=if_death_gt,
            gt_flags={
                'survival_1yr': survival_1yr_gt,
                'survival_3yr': survival_3yr_gt,
                'survival_5yr': survival_5yr_gt,
                'survival_months': survival_months_gt_val
            }
        )

        final_survival_res = {
            'acc_1yr': acc_1yr,
            'acc_3yr': acc_3yr,
            'acc_5yr': acc_5yr,
            'survival_score': final_survival_score,
            'survival_months': final_months,
            'risk_score': risk,
            'survival_rank': survival_rank,
        }

        # major voting（与你的主流程一致：1.0/0.6/0.3）
        treatment_result = major_voting_treatment(valid_treatment_results)
        if treatment_result is None or len(treatment_result) != 2:
            parse_error_treatment += 1; continue
        final_treatment_result, voting_result = treatment_result
        bt = []
        threshold_bt = 0.5
        for tname in final_treatment_result:
            if float(final_treatment_result[tname]) >= threshold_bt:
                bt.append(tname)

        name_gt_treatment = get_treatment(sample['treatment'])
        gt_treatment = {name_gt_treatment: 1}

        treatment_eval_res_top1 = weighted_topn_similarity_v2(gt_treatment, final_treatment_result, n=1)
        treatment_score_2 = treatment_eval_score_2(name_gt_treatment, final_treatment_result, thres=0.9)
        if_treatment_hit = 1 if treatment_score_2 != 0.0 else 0
        top1_accuracy = calculate_topn_accuracy_from_voting(voting_result, name_gt_treatment, 1)
        top2_accuracy = calculate_topn_accuracy_from_voting(voting_result, name_gt_treatment, 2)
        top3_accuracy = calculate_topn_accuracy_from_voting(voting_result, name_gt_treatment, 3)

        add_res(if_treatment_hit_list, src, if_treatment_hit)
        add_res(treatment_score_2_list, src, treatment_score_2)
        add_res(top1_scores_treatment_list, src, treatment_eval_res_top1['score'])
        add_res(top1_accuracy_list, src, top1_accuracy)
        add_res(top2_accuracy_list, src, top2_accuracy)
        add_res(top3_accuracy_list, src, top3_accuracy)

        # ours
        your_staging_labels.append(final_survival_res['survival_rank'])
        survival_months_pred.append(final_survival_res['risk_score'])
        survival_months_gt.append(gt_survival['survival_month'])
        death_events.append(gt_survival['if_death'])
        sources_ours.append(src)

        # baselines（携带 source）
        if ajcc_tnm is not None:
            tnm_risk_scores.append(ajcc_tnm[0]); tnm_staging_labels.append(ajcc_tnm[1])
            tnm_survival_months_gt.append(gt_survival['survival_month']); tnm_death_events.append(gt_survival['if_death'])
            tnm_sources.append(src)
        if bclc is not None:
            bclc_risk_scores.append(bclc[0]); bclc_staging_labels.append(bclc[1])
            bclc_survival_months_gt.append(gt_survival['survival_month']); bclc_death_events.append(gt_survival['if_death'])
            bclc_sources.append(src)
        if cnlc is not None:
            cnlc_risk_scores.append(cnlc[0]); cnlc_staging_labels.append(cnlc[1])
            cnlc_survival_months_gt.append(gt_survival['survival_month']); cnlc_death_events.append(gt_survival['if_death'])
            cnlc_sources.append(src)

        for pair in [{'survival_rank':final_survival_res['survival_rank']},
                     {'risk_score':final_survival_res['risk_score']},
                     {'survival_month':gt_survival['survival_month']},
                     {'if_death':gt_survival['if_death']}]:
            for key in pair:
                value = pair[key]
                if value is None or value is np.nan:
                    print(key,': ', value)

        cur_uuid = sample.get("uuid", sample.get("extra_info", {}).get("uuid", ""))
        whitelist.append(cur_uuid)  # >>> NEW: 进入白名单
        fres.append(
            {
                'uuid': cur_uuid,
                'source': src,
                'survival_time': gt_survival['survival_month'],
                'predicted_survival': final_survival_res['risk_score'],
                # >>> NEW: 预留并回填 SFT
                'predicted_survival_sft': None,
                'event': gt_survival['if_death'],
                'ajcc_tnm': ajcc_tnm[1],
                'bclc': bclc[1],
                'cnlc': cnlc[1],
                'tx_actual': name_gt_treatment,
                'tx_ours': bt,
                'tx_ours_sft': [],  # >>> NEW: 预留
                'tx_claude-3-5-sonnet-20241022': uuid2tx_claude.get(cur_uuid, []),
                'tx_deepseek-r1': uuid2tx_ds.get(cur_uuid, []),
                'tx_gemini-2_5-pro': uuid2tx_gemini.get(cur_uuid, []),
                'tx_gpt-4o-2024-08-06': uuid2tx_gpt4o.get(cur_uuid, []),
                'tx_gpt-5': uuid2tx_gpt5.get(cur_uuid, []),
                'tx_bclc': list(bclc_recommended(bclc[1])) if bclc_recommended(bclc[1]) else '',
                'tx_cnlc': list(cnlc_recommended(cnlc[1])) if cnlc_recommended(cnlc[1]) else '',
            }
        )

    # ================== 并列 SFT：严格按 Ours 白名单 ==================
    whitelist_set = set(whitelist)
    fallback_map = {r['uuid']: {'survival_time': r['survival_time'], 'event': r['event']} for r in fres}
    df_ours_sft, uuid2tx_sft, uuid2pred_sft = process_ours_like_jsonl_with_whitelist(
        jsonl_path=res_path_sft,
        whitelist_uuids=whitelist_set,
        fallback_map=fallback_map,
        uuid2source_map=uuid2source,
        out_csv_path="df_ours_sft_multicenter.csv",
        tx_threshold=0.5
    )
    # 回填到 fres
    u2row = {r['uuid']: r for r in fres}
    for u in whitelist_set:
        if u in u2row:
            u2row[u]['tx_ours_sft'] = uuid2tx_sft.get(u, [])
            u2row[u]['predicted_survival_sft'] = uuid2pred_sft.get(u, None)

    # ================= 外部 LLM 结果（总体+分源） =================
    process_llm_jsonl(claude_jsonl, "data/visualization/df_claude-3-5-sonnet-20241022_multicenter.csv", uuid2source=uuid2source)
    process_llm_jsonl(ds_jsonl, "data/visualization/df_deepseek-r1_multicenter.csv", uuid2source=uuid2source)
    process_llm_jsonl(gemini_jsonl, "data/visualization/df_gemini-2_5-pro_multicenter.csv", uuid2source=uuid2source)
    process_llm_jsonl(gpt4o_jsonl, "data/visualization/df_gpt-4o-2024-08-06_multicenter.csv", uuid2source=uuid2source)
    process_llm_jsonl(gpt5_jsonl, "data/visualization/df_gpt-5_multicenter.csv", uuid2source=uuid2source)

    print(f"Missing survival data for {survival_miss}")
    print(f"Missing BCLC staging for {bclc_miss}")
    print(f"Missing CNLC staging for {cnlc_miss}")
    print(f"Missing AJCC TNM staging for {ajcc_miss}")
    print(f"Parse error survival for {parse_error_survivial} samples")
    print(f"Parse error treatment for {parse_error_treatment} samples")

    # ================= 组装主 DF（含 source，用于分源落盘；整体落盘时去掉 source） =================
    df_ours = pd.DataFrame({
        'staging': your_staging_labels,
        'time': survival_months_gt,
        'event': death_events,
        'predicted': survival_months_pred,
        'source': sources_ours,
    })
    df_tnm = pd.DataFrame({
        'staging': tnm_staging_labels,
        'time': tnm_survival_months_gt,
        'event': tnm_death_events,
        'predicted': tnm_risk_scores,
        'source': tnm_sources,
    })
    df_bclc = pd.DataFrame({
        'staging': bclc_staging_labels,
        'time': bclc_survival_months_gt,
        'event': bclc_death_events,
        'predicted': bclc_risk_scores,
        'source': bclc_sources,
    })
    df_cnlc = pd.DataFrame({
        'staging': cnlc_staging_labels,
        'time': cnlc_survival_months_gt,
        'event': cnlc_death_events,
        'predicted': cnlc_risk_scores,
        'source': cnlc_sources,
    })

    # ====== 整体 KM 与 C-index（保持原输出） ======
    plot_km(df_ours.drop(columns=['source']), 'data/visualization/km_overall_multicenter.svg')
    for H in [12,24,36,48,60]:
        plot_km_capped(df_ours.drop(columns=['source']), f'data/visualization/km_{H//12}yr_multicenter.svg', horizon_months=H)

    print(f'Total Number of Samples: {len(df_ours)}')
    c_index = concordance_index(df_ours['time'], df_ours['predicted'], df_ours['event'])
    print('Our concordance index: {:.4f}'.format(c_index), '| ', len(df_ours))
    # >>> NEW: Our-SFT 的 C-index
    if len(df_ours_sft):
        c_index_sft = concordance_index(df_ours_sft['time'], df_ours_sft['predicted'], df_ours_sft['event'])
        print('Our-SFT concordance index: {:.4f}'.format(c_index_sft), '| ', len(df_ours_sft))

    if len(df_tnm):
        c_index = concordance_index(df_tnm['time'], df_tnm['predicted'], df_tnm['event'])
        print('TNM concordance index: {:.4f}'.format(c_index), '| ', len(df_tnm))
    if len(df_bclc):
        c_index = concordance_index(df_bclc['time'], df_bclc['predicted'], df_bclc['event'])
        print('BCLC concordance index: {:.4f}'.format(c_index), '| ', len(df_bclc))
    if len(df_cnlc):
        c_index = concordance_index(df_cnlc['time'], df_cnlc['predicted'], df_cnlc['event'])
        print('CNLC concordance index: {:.4f}'.format(c_index), '| ', len(df_cnlc))

    # 原 treatment 评估打印
    for key in top1_scores_treatment_list:
        scores = top1_scores_treatment_list[key]
        if len(scores) > 0:
            print(f'Treatment Top-1 Similarity Score ({key}): {np.mean(scores):.4f} ± {np.std(scores):.4f} | N={len(scores)}')
    for key in treatment_score_2_list:
        scores = treatment_score_2_list[key]
        if len(scores) > 0:
            print(f'Treatment Score-2 ({key}): {np.mean(scores):.4f} ± {np.std(scores):.4f} | N={len(scores)}')
    for key in if_treatment_hit_list:
        scores = if_treatment_hit_list[key]
        if len(scores) > 0:
            print(f'Treatment If-Hit ({key}): {np.mean(scores):.4f} ± {np.std(scores):.4f} | N={len(scores)}')

    print("\n" + "="*50)
    print("TOP-N ACCURACY RESULTS (Major Voting)")
    print("="*50)
    for key in top1_accuracy_list:
        scores = top1_accuracy_list[key]
        if len(scores) > 0:
            print(f'Treatment Top-1 Accuracy ({key}): {np.mean(scores):.4f} ± {np.std(scores):.4f} | N={len(scores)}')
    for key in top2_accuracy_list:
        scores = top2_accuracy_list[key]
        if len(scores) > 0:
            print(f'Treatment Top-2 Accuracy ({key}): {np.mean(scores):.4f} ± {np.std(scores):.4f} | N={len(scores)}')
    for key in top3_accuracy_list:
        scores = top3_accuracy_list[key]
        if len(scores) > 0:
            print(f'Treatment Top-3 Accuracy ({key}): {np.mean(scores):.4f} ± {np.std(scores):.4f} | N={len(scores)}')

    # ====== 整体落盘（保持原文件名） ======
    df_ours.drop(columns=['source']).to_csv('data/visualization/df_ours_multicenter.csv', index=False)
    df_tnm.drop(columns=['source']).to_csv('data/visualization/df_tnm_multicenter.csv', index=False)
    df_bclc.drop(columns=['source']).to_csv('data/visualization/df_bclc_multicenter.csv', index=False)
    df_cnlc.drop(columns=['source']).to_csv('data/visualization/df_cnlc_multicenter.csv', index=False)
    save_jsonl(fres, "data/visualization/staging_survival_multicenter_sft.jsonl")

    # ====== 新增：按 source 拆分落盘（包含 SFT） ======
    def _split_and_save_df(df, prefix):
        if "source" not in df.columns:
            return
        for src, dfg in df.groupby("source"):
            out_path = f"{prefix}_{src}.csv"
            dfg.drop(columns=["source"]).to_csv(out_path, index=False)
            print(f"[per-source] saved: {out_path} | n={len(dfg)}")

    _split_and_save_df(df_ours, "df_ours")
    _split_and_save_df(df_tnm, "df_tnm")
    _split_and_save_df(df_bclc, "df_bclc")
    _split_and_save_df(df_cnlc, "df_cnlc")
    # >>> NEW: SFT 分源
    _split_and_save_df(df_ours_sft, "df_ours_sft")

    # staging_survival_<source>.jsonl
    fres_by_src = defaultdict(list)
    for r in fres:
        s = _sanitize_name(r.get("source", "unknown"))
        fres_by_src[s].append(r)
    for s, recs in fres_by_src.items():
        p = f"data/visualization/staging_survival_{s}.jsonl"
        save_jsonl(recs, p, mode="w")
        print(f"[per-source] saved: {p} | n={len(recs)}")
