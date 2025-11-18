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
    "Ia": ("Surgical_resection","Ablation","Liver_transplantation"),
    "Ib": ("Surgical_resection","Interventional_therapy","Ablation","Interventional_therapy_plus_ablation","Liver_transplantation"),
    "IIa":("Surgical_resection","Interventional_therapy","Surgical_resection_plus_ablation","Interventional_therapy_plus_ablation","Liver_transplantation"),
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

# ========== KM 绘图 ==========
def plot_km(df, save_path, time_col='time', event_col='event'):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    df_plot = df[[time_col, event_col, 'staging']].copy()
    df_plot[time_col] = pd.to_numeric(df_plot[time_col], errors='coerce')
    df_plot[event_col] = pd.to_numeric(df_plot[event_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[time_col, event_col, 'staging'])
    for group_name, group_data in df_plot.groupby('staging'):
        kmf.fit(durations=group_data[time_col], event_observed=group_data[event_col], label=str(group_name))
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
    event_within = ((df_plot[event_col].values == 1) & (df_plot[time_col].values <= horizon_months)).astype(int)
    df_plot['time_capped'] = time_capped; df_plot['event_within'] = event_within
    for group_name, group_data in df_plot.groupby('staging'):
        kmf.fit(durations=group_data['time_capped'], event_observed=group_data['event_within'], label=str(group_name))
        kmf.plot_survival_function(ci_show=False)
    plt.title(f"Kaplan–Meier (Task: ≤{horizon_months//12} year survival)")
    plt.xlabel("Time (months)"); plt.ylabel("Survival Probability")
    plt.xlim(0, horizon_months); plt.axvline(horizon_months, linestyle='--', linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.5); plt.legend(title="Staging")
    plt.tight_layout(); plt.savefig(save_path, format='svg'); plt.close()

# ========== 工具函数 ==========
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

def _max_passed_threshold_months_from_flags(flags):
    mapping = [("survival_1yr", 12), ("survival_3yr", 36), ("survival_5yr", 60)]
    passed = [m for k, m in mapping if flags.get(k) == 1]
    return max(passed) if passed else None

def _months_score_with_event(pred_months, gt_months, event, gt_flags=None, tau_death=12.0, tau_censor=6.0):
    if pred_months is None: return 0.0
    if event == 1:
        if gt_months is None: return 0.0
        diff = abs(pred_months - gt_months)
        return exp(-diff / float(tau_death))
    if event == 0:
        if gt_months is not None:
            t_c = gt_months
        else:
            t_c = _max_passed_threshold_months_from_flags(gt_flags or {})
            if t_c is None: return 0.5
        d = max(0.0, float(t_c) - float(pred_months))
        return exp(-d / float(tau_censor))
    return 0.5

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def parse_hard_check_survival(solution_str):
    m = re.search(r"<hard_check_survival>(.*?)</hard_check_survival>", solution_str, re.DOTALL)
    if not m: return None
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
    if not isinstance(obj, dict): return None
    if "survival_months" not in obj and "survival" in obj and isinstance(obj["survival"], dict):
        obj = obj["survival"]
    pred_months = _safe_to_int(obj.get("survival_months"))
    pred_1yr = _safe_to_01_or_none(obj.get("survival_1yr"))
    pred_3yr = _safe_to_01_or_none(obj.get("survival_3yr"))
    pred_5yr = _safe_to_01_or_none(obj.get("survival_5yr"))
    return {"survival_months": pred_months,"survival_1yr": pred_1yr,"survival_3yr": pred_3yr,"survival_5yr": pred_5yr}

def parse_treatment(res):
    hard_check_match = re.search(r"<hard_check_treatment>(.*?)</hard_check_treatment>", res, re.DOTALL)
    if not hard_check_match:
        return None
    try:
        fixed = hard_check_match.group(1).strip().encode().decode('unicode_escape')
        hard = json.loads(fixed.replace("'", '"'))
        if 'scores' in hard: hard = hard['scores']
    except json.JSONDecodeError:
        return None
    return hard

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
        return cn2en.get(treatment, treatment)

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
    pred_survival_month = pred.get('survival_months')
    if pred_survival_month is None: return None

    pred_survival_1yr = 1 if pred_survival_month > 12 else 0
    pred_survival_3yr = 1 if pred_survival_month > 36 else 0
    pred_survival_5yr = 1 if pred_survival_month > 60 else 0

    mapped_month = min((pred_survival_month / 100), 1) * 100
    if 0 <= mapped_month < 4: survival_rank = 'D'
    elif 4 <= mapped_month < 12: survival_rank = 'C'
    elif 12 <= mapped_month < 24: survival_rank = 'B'
    else: survival_rank = 'A'

    risk = pred_survival_month
    acc_1yr = cal_correctness(pred_survival_1yr, survival_1yr)
    acc_3yr = cal_correctness(pred_survival_3yr, survival_3yr)
    acc_5yr = cal_correctness(pred_survival_5yr, survival_5yr)

    survival_score = _months_score_with_event(
        pred_months=pred_survival_month,
        gt_months=survival_months,
        event=gt.get('if_death'),
        gt_flags={'survival_1yr': survival_1yr,'survival_3yr': survival_3yr,'survival_5yr': survival_5yr,'survival_months': survival_months}
    )

    return {
        'acc_1yr': acc_1yr,'acc_3yr': acc_3yr,'acc_5yr': acc_5yr,
        'survival_score': survival_score,'survival_months': pred_survival_month,
        'risk_score': risk,'survival_rank': survival_rank,
    }

def map_survival(if_death):
    if if_death == '否' or if_death == 0: return 0
    if if_death: return 1
    return None

def calculate_survival_months(start_date, end_date) -> int:
    start_date = start_date.replace('UK','01'); end_date = end_date.replace('UK','01')
    def to_datetime(d):
        if isinstance(d, (datetime, date)): return datetime(d.year, d.month, d.day)
        if isinstance(d, str):
            for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
                try: return datetime.strptime(d, fmt)
                except ValueError: continue
        return None
    start = to_datetime(start_date); end = to_datetime(end_date)
    if not start or not end: return None
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day: months -= 1
    return months

def _is_na(x):
    if x is None: return True
    if isinstance(x, float) and np.isnan(x): return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "na", "none", "null"}: return True
    return False

def get_month(sample):
    if sample is None: return None
    survival_month = sample.get('survival_month', None)
    if isinstance(survival_month, str) and '\n' in survival_month:
        survival_month = survival_month.split('\n')[-1].strip()
    if _is_na(survival_month): return None

    if_death = map_survival(sample.get('death')); 
    if _is_na(if_death): if_death = None

    if if_death is None:
        in_time = sample.get('in_time'); death_time = sample.get('death_time')
        if _is_na(in_time): in_time = None
        if _is_na(death_time): death_time = None
        if in_time and death_time: if_death = 1
        elif in_time and survival_month: if_death = 0
        else: return None

    in_time = sample.get('in_time'); death_time = sample.get('death_time')
    if _is_na(in_time): in_time = None
    if _is_na(death_time): death_time = None

    if isinstance(survival_month, str):
        parsed_int = _safe_to_int(survival_month)
        if parsed_int is not None:
            return {'survival_month': parsed_int, 'if_death': if_death}
        if if_death:
            if in_time and death_time:
                sm = calculate_survival_months(in_time, death_time); 
                if sm is None: return None
                return {'survival_month': sm, 'if_death': if_death}
            elif death_time is None and in_time and survival_month:
                sm = calculate_survival_months(in_time, survival_month); 
                if sm is None: return None
                return {'survival_month': sm, 'if_death': 0}
            else: return None
        else:
            if in_time and survival_month:
                sm = calculate_survival_months(in_time, survival_month); 
                if sm is None: return None
                return {'survival_month': sm, 'if_death': if_death}
            else: return None
    elif isinstance(survival_month, (int, float)):
        parsed = _safe_to_int(survival_month)
        if parsed is None: return None
        return {'survival_month': parsed, 'if_death': if_death}
    return None

def get_staging(sample):
    def stage_from_tnm(t: str, n: str, m: str):
        t, n, m = str(t or "").upper(), str(n or "").upper(), str(m or "").upper()
        if 'M1' in m: return -5, 'IV'
        if 'N1' in n: return -4, 'IIIC'
        if 'T1' in t: return 0, 'I'
        if 'T2' in t: return -1, 'II'
        if 'T3' in t: return -2, 'IIIA'
        if 'T4' in t: return -3, 'IIIB'
        return None
    def get_bclc(bclc):
        if bclc is None or (isinstance(bclc, float) and pd.isna(bclc)): return None
        s = norm_bclc_stage(str(bclc)); score_map = {"0":0,"A":-1,"B":-2,"C":-3,"D":-4}
        return (score_map[s], s) if s in score_map else None
    def get_cnlc(cnlc):
        if cnlc is None or (isinstance(cnlc, float) and pd.isna(cnlc)): return None
        s = norm_cnlc_stage(str(cnlc)); score_map = {"Ia":0,"Ib":-1,"IIa":-2,"IIb":-3,"IIIa":-4,"IIIb":-5,"IV":-6}
        return (score_map[s], s) if s in score_map else None
    def get_ajcc_tnm(ajcc_tnm):
        if ajcc_tnm is None or (isinstance(ajcc_tnm, float) and pd.isna(ajcc_tnm)): return None
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
    bclc = sample.get('bclc'); cnlc = sample.get('cnlc'); ajcc_tnm = sample.get('ajcc_tnm') or sample.get('tnm')
    return {'bclc': get_bclc(bclc),'cnlc': get_cnlc(cnlc),'ajcc_tnm': get_ajcc_tnm(ajcc_tnm)}

def save_jsonl(records, path, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def add_miss(miss_dict, source):
    miss_dict['count'] += 1
    if source in miss_dict: miss_dict[source] += 1
    else: miss_dict[source] = 1
    
def add_res(res_dict, source, value):
    res_dict['sum'].append(value)
    if source in res_dict: res_dict[source].append(value)
    else: res_dict[source] = [value]

def remove_outliers(values, threshold=1):
    if len(values) <= 1: return values
    q1 = np.percentile(values, 25); q3 = np.percentile(values, 95); iqr = q3 - q1
    lower_bound = q1 - threshold * iqr; upper_bound = q3 + threshold * iqr
    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    return filtered if len(filtered) > 0 else values

def _tx_list_from_scores_dict(scores: dict, threshold: float = 0.5):
    if not isinstance(scores, dict): return []
    items = []
    for t, v in scores.items():
        try: v = float(v)
        except Exception: continue
        if v >= threshold: items.append((t, v))
    items.sort(key=lambda x: (-x[1], x[0]))
    return [t for t,_ in items]

def build_uuid_to_tx_map(jsonl_path: str, threshold: float = 0.5) -> dict:
    rows = load_jsonl(jsonl_path)
    print(f"[build_uuid_to_tx_map] Loading {len(rows)} from {jsonl_path}")
    uuid2tx = {}; bad = 0
    for rec in rows:
        uuid = rec.get("extra_info", {}).get("uuid") or rec.get("uuid")
        if not uuid:
            bad += 1; continue
        text = ""
        llm_resp = rec.get("llm_response", {})
        if isinstance(llm_resp, dict): text = llm_resp.get("text", "") or ""
        tdict = parse_treatment(text)
        uuid2tx[uuid] = _tx_list_from_scores_dict(tdict, threshold=threshold) if isinstance(tdict, dict) else []
    print(f"[build_uuid_to_tx_map] Done. Missing/invalid rows: {bad}")
    return uuid2tx

def process_llm_jsonl(jsonl_path: str, out_csv_path: str):
    rows = load_jsonl(jsonl_path)
    print(f"[process_llm_jsonl] Loaded {len(rows)} rows from {jsonl_path}")
    staging_labels, times_gt, events, preds_months = [], [], [], []
    miss_gt = 0; parse_error_survival = 0; skip_stage_fail = 0; skip_zero_or_neg = 0

    for rec in rows:
        try:
            gt = rec["reward_model"]["ground_truth"]
        except Exception:
            miss_gt += 1; continue

        s_survival_status = {"survival_month": gt.get("survival_months"), "death": gt.get("if_death")}
        gt_survival = get_month(s_survival_status)
        if gt_survival is None:
            miss_gt += 1; continue
        if not gt_survival["survival_month"] or gt_survival["survival_month"] <= 0:
            skip_zero_or_neg += 1; continue

        stage_input = {"bclc": gt.get("bclc"), "cnlc": gt.get("cnlc"), "ajcc_tnm": gt.get("tnm")}
        st = get_staging(stage_input)
        if st["bclc"] is None or st["cnlc"] is None or st["ajcc_tnm"] is None:
            skip_stage_fail += 1; continue

        reply_text = ""
        llm_resp = rec.get("llm_response", {})
        if isinstance(llm_resp, dict): reply_text = llm_resp.get("text") or ""

        survival_res = evaluate_survival(gt_survival, reply_text)
        if survival_res is None:
            parse_error_survival += 1
            survival_res = {"acc_1yr":0,"acc_3yr":0,"acc_5yr":0,"survival_score":0,"survival_months":6,"risk_score":6,"survival_rank":"C"}

        staging_labels.append(survival_res["survival_rank"])
        preds_months.append(survival_res["risk_score"])
        times_gt.append(gt_survival["survival_month"])
        events.append(gt_survival["if_death"])

    print(f"[process_llm_jsonl] GT missing: {miss_gt}, parse_error_survival: {parse_error_survival}, "
          f"skip_stage_fail(BCLC/CNLC/TNM): {skip_stage_fail}, skip_zero_or_neg: {skip_zero_or_neg}")

    df = pd.DataFrame({"staging": staging_labels,"time": times_gt,"event": events,"predicted": preds_months})
    df.to_csv(out_csv_path, index=False)
    print(f"[process_llm_jsonl] Saved: {out_csv_path} | N={len(df)}")
    return df

# >>> NEW: 用于并列处理 SFT jsonl（与 Ours 同逻辑，按 Ours 白名单遍历，缺失则兜底）
def _pick_replies(rec):
    for k in ("result_ours","result","replies","result_sft","result_ours_sft"):
        v = rec.get(k)
        if isinstance(v, list) and len(v) > 0:
            return v
    return None

def process_ours_like_jsonl_with_whitelist(jsonl_path: str,
                                           whitelist_uuids: Set[str],
                                           fallback_map: dict,
                                           out_csv_path: str,
                                           tx_threshold: float = 0.5):
    rows = load_jsonl(jsonl_path)
    idx = {}
    for rec in rows:
        u = rec.get("extra_info", {}).get("uuid") or rec.get("uuid") or rec.get("id") or rec.get("sample_id")
        if u: idx[u] = rec

    staging_labels, times_gt, events, preds_months = [], [], [], []
    uuid2_tx, uuid2_pred = {}, {}

    for u in whitelist_uuids:
        fb = fallback_map.get(u, {})
        gt_survival = get_month({'survival_month': fb.get('survival_time'), 'death': fb.get('event')})
        if gt_survival is None:
            # 极端情况：白名单必然有 GT，这里保守跳过
            continue

        rec = idx.get(u)
        replies = _pick_replies(rec) if rec else None

        valid_survival_results, valid_treatment_predictions = [], []
        if replies:
            for reply in replies:
                sres = evaluate_survival(gt_survival, reply)
                # print(sres)
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
            # 缺记录：统一兜底
            valid_survival_results = [{"acc_1yr":0,"acc_3yr":0,"acc_5yr":0,"survival_score":0,"survival_months":6,"risk_score":6,"survival_rank":"C"}]
            valid_treatment_predictions = [{'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0., 'Interventional_therapy': 0.,
                                            'Interventional_therapy_plus_ablation': 0., 'Surgical_resection_plus_ablation': 0.,
                                            'Systemic_anti-tumor_therapy': 0., 'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                                            'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.}]

        months = [r['survival_months'] for r in valid_survival_results if r['survival_months'] is not None]
        months = remove_outliers(months) if len(months) else [6]
        final_months = float(np.mean(months)) if len(months) else 6.0
        mapped = min((final_months / 100), 1) * 100
        survival_rank = 'D' if mapped < 4 else ('C' if mapped < 12 else ('B' if mapped < 24 else 'A'))
        risk = final_months

        # 简洁版 majority voting（沿用你在 Ours 的阈值筛选）
        mv_soft = {}
        top1 = None; top2 = None; top3 = None
        # 统计 top-1/2/3 多数票
        def get_top_n_treatments(pred_dict, n=3):
            if not isinstance(pred_dict, dict): return []
            sorted_items = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
            return [item[0] for item in sorted_items[:n]]
        votes = {'top1': [], 'top2': [], 'top3': []}
        for pdict in valid_treatment_predictions:
            tops = get_top_n_treatments(pdict, n=3)
            if len(tops) >= 1: votes['top1'].append(tops[0])
            if len(tops) >= 2: votes['top2'].append(tops[1])
            if len(tops) >= 3: votes['top3'].append(tops[2])
        for k in ('top1','top2','top3'):
            if votes[k]:
                c = Counter(votes[k]).most_common(1)
                if c: 
                    if k=='top1': top1 = c[0][0]; mv_soft[top1] = 1.0
                    if k=='top2' and c[0][0] != top1: top2 = c[0][0]; mv_soft[top2] = 0.8
                    if k=='top3' and c[0][0] not in (top1,top2): top3 = c[0][0]; mv_soft[top3] = 0.6
        tx = [t for t,v in mv_soft.items() if float(v) >= tx_threshold]

        staging_labels.append(survival_rank)
        preds_months.append(risk)
        times_gt.append(gt_survival['survival_month'])
        events.append(gt_survival['if_death'])
        uuid2_tx[u] = tx
        uuid2_pred[u] = risk

    df = pd.DataFrame({"staging": staging_labels,"time": times_gt,"event": events,"predicted": preds_months})
    df.to_csv(out_csv_path, index=False)
    print(f"[process_ours_like_jsonl_with_whitelist] Saved: {out_csv_path} | N={len(df)}")
    return df, uuid2_tx, uuid2_pred

# ================================ 主程序 ================================
if __name__ == "__main__":
    # 原 Ours 路径
    res_path = 'xxx'
    # >>> NEW: SFT 路径（你给的）
    res_path_sft = 'xxx'

    # ==== 在处理 data 之前先构建两个 LLM 的 uuid→tx 映射 ====
    gemini_jsonl = "xxx"
    gpt4o_jsonl  = "xxx"
    uuid2tx_gemini = build_uuid_to_tx_map(gemini_jsonl, threshold=0.5)
    uuid2tx_gpt4o  = build_uuid_to_tx_map(gpt4o_jsonl,  threshold=0.5)

    data = load_jsonl(res_path)
    print(f"Loaded {len(data)} samples from {res_path}")
    fres = []

    your_staging_labels = []
    survival_months_pred = []
    survival_months_gt = []
    death_events = []

    tnm_risk_scores = []
    bclc_risk_scores = []
    cnlc_risk_scores = []

    tnm_death_events = []
    bclc_death_events = []
    cnlc_death_events = []

    tnm_survival_months_gt = []
    bclc_survival_months_gt = []
    cnlc_survival_months_gt = []

    tnm_staging_labels = []
    bclc_staging_labels = []
    cnlc_staging_labels = []

    selected_chunggeng = []

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

    whitelist = []  # >>> NEW: 记录通过 Ours 严格条件的 uuid 作为白名单

    for sample in data:
        sample['source'] = 'seer'
        replies = sample.get('result_ours', [])
        if not isinstance(replies, list) or len(replies) == 0:
            parse_error_survivial += 1; parse_error_treatment += 1; continue

        # —— 以 GT（reward_model.ground_truth）获取三分期用于严格筛选
        s_staging = {
            'bclc': sample['reward_model']['ground_truth']['bclc'],
            'cnlc': sample['reward_model']['ground_truth']['cnlc'],
            'ajcc_tnm': sample['reward_model']['ground_truth']['tnm'],
        }
        staging = get_staging(s_staging)
        bclc = staging['bclc']; cnlc = staging['cnlc']; ajcc_tnm = staging['ajcc_tnm']
        if bclc is None: add_miss(bclc_miss, sample['source'])
        if cnlc is None: add_miss(cnlc_miss, sample['source'])
        if ajcc_tnm is None: add_miss(ajcc_miss, sample['source'])
        if bclc is None or cnlc is None or ajcc_tnm is None:
            continue
        
        s_survival_status = {
            'survival_month': sample['reward_model']['ground_truth']['survival_months'],
            'death': sample['reward_model']['ground_truth']['if_death']
        }
        gt_survival = get_month(s_survival_status)
        if gt_survival is None:
            if 'new_gt_survival' in sample:
                new_gt_survival = get_new_gt_survival(sample)
                if new_gt_survival is not None:
                    gt_survival = new_gt_survival
                else:
                    add_miss(survival_miss, sample['source']); continue
            else:
                add_miss(survival_miss, sample['source']); continue
        if gt_survival['survival_month'] == 0 or gt_survival['survival_month'] < 0:
            survival_0 += 1; continue

        valid_survival_results = []
        valid_treatment_predictions = []
        for reply in replies:
            survival_res = evaluate_survival(gt_survival, reply)
            if survival_res is None:
                parse_error_survivial += 1
                survival_res = {'acc_1yr':0,'acc_3yr':0,'acc_5yr':0,'survival_score':0,'survival_months':6,'risk_score':0,'survival_rank':'C'}
            treatment_res = parse_treatment(reply)
            if (treatment_res is None) or (not isinstance(treatment_res, dict)):
                parse_error_treatment += 1
                treatment_res = {'Surgical_resection': 1, 'Ablation': 0., 'Liver_transplantation': 0., 'Interventional_therapy': 0.,
                                 'Interventional_therapy_plus_ablation': 0., 'Surgical_resection_plus_ablation': 0.,
                                 'Systemic_anti-tumor_therapy': 0., 'Interventional_therapy_plus_systemic_anti-tumor_therapy': 0.,
                                 'Radiotherapy': 0., 'Symptomatic_support': 0., 'Palliative_care': 0.}
            valid_survival_results.append(survival_res)
            valid_treatment_predictions.append(treatment_res)

        if len(valid_survival_results) == 0 or len(valid_treatment_predictions) == 0:
            continue

        months = [r['survival_months'] for r in valid_survival_results if r['survival_months'] is not None]
        if len(months) == 0: continue
        filtered_months = remove_outliers(months)
        final_months = np.mean(filtered_months)

        final_1yr = 1 if final_months > 12 else 0
        final_3yr = 1 if final_months > 36 else 0
        final_5yr = 1 if final_months > 60 else 0
        mapped_month = min((final_months / 100), 1) * 100
        survival_rank = 'D' if mapped_month < 4 else ('C' if mapped_month < 12 else ('B' if mapped_month < 24 else 'A'))
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
            pred_months=final_months, gt_months=survival_months_gt_val, event=if_death_gt,
            gt_flags={'survival_1yr': survival_1yr_gt,'survival_3yr': survival_3yr_gt,'survival_5yr': survival_5yr_gt,'survival_months': survival_months_gt_val}
        )

        final_survival_res = {
            'acc_1yr': acc_1yr,'acc_3yr': acc_3yr,'acc_5yr': acc_5yr,
            'survival_score': final_survival_score,'survival_months': final_months,
            'risk_score': risk,'survival_rank': survival_rank,
        }

        # ===== 获取 GT 的 top3 并构造 tx_actual_all =====
        name_gt_treatment = 'None'; gt_treatment_score = -1.0
        gt_treatment = {}; gt_items = []
        for item in sample['reward_model']['ground_truth']['top3_treatments']:
            t_raw = item.get('treatment'); sc = float(item.get('score', 0.0))
            t_en = cn2en.get(t_raw, t_raw)
            gt_treatment[t_en] = sc; gt_items.append((t_en, sc))
            if sc > gt_treatment_score:
                gt_treatment_score = sc; name_gt_treatment = t_en
        gt_items_sorted = sorted(gt_items, key=lambda x: (-x[1], x[0]))
        tx_actual_all = [t for (t, s) in gt_items_sorted if s >= 0.5]

        # ===== Major voting 生成 ours 的 tx_ours（>=0.5）
        votes = {'top1': [], 'top2': [], 'top3': []}
        def get_top_n_treatments(pred_dict, n=3):
            if not isinstance(pred_dict, dict): return []
            sorted_items = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
            return [item[0] for item in sorted_items[:n]]
        for pred_dict in valid_treatment_predictions:
            tops = get_top_n_treatments(pred_dict, 3)
            if len(tops)>=1: votes['top1'].append(tops[0])
            if len(tops)>=2: votes['top2'].append(tops[1])
            if len(tops)>=3: votes['top3'].append(tops[2])
        major = {}
        for k, w in [('top1',1.0),('top2',0.8),('top3',0.6)]:
            if votes[k]:
                c = Counter(votes[k]).most_common(1)
                if c and c[0][0] not in major: major[c[0][0]] = w
        bt = [t for t, v in major.items() if float(v) >= 0.5]

        your_staging_labels.append(final_survival_res['survival_rank'])
        survival_months_pred.append(final_survival_res['risk_score'])
        survival_months_gt.append(gt_survival['survival_month'])
        death_events.append(gt_survival['if_death'])
        
        if ajcc_tnm is not None:
            tnm_risk_scores.append(ajcc_tnm[0]); tnm_staging_labels.append(ajcc_tnm[1])
            tnm_survival_months_gt.append(gt_survival['survival_month']); tnm_death_events.append(gt_survival['if_death'])
        if bclc is not None:
            bclc_risk_scores.append(bclc[0]); bclc_staging_labels.append(bclc[1])
            bclc_survival_months_gt.append(gt_survival['survival_month']); bclc_death_events.append(gt_survival['if_death'])
        if cnlc is not None:
            cnlc_risk_scores.append(cnlc[0]); cnlc_staging_labels.append(cnlc[1])
            cnlc_survival_months_gt.append(gt_survival['survival_month']); cnlc_death_events.append(gt_survival['if_death'])

        cur_uuid = sample.get("extra_info", {}).get("uuid", sample.get("uuid", ""))
        whitelist.append(cur_uuid)  # >>> NEW: 进入白名单

        fres.append(
            {
                'uuid': cur_uuid,
                'survival_time': gt_survival['survival_month'],
                'predicted_survival': final_survival_res['risk_score'],
                'event': gt_survival['if_death'],
                'ajcc_tnm': ajcc_tnm[1], 'bclc': bclc[1], 'cnlc': cnlc[1],
                'treatment_res': treatment_res,
                'tx_actual': name_gt_treatment,
                'tx_actual_all': tx_actual_all,
                'tx_ours': bt,
                # >>> NEW: 预留并回填
                'tx_ours_sft': [],
                'predicted_survival_sft': None,

                'tx_gemini-2_5-pro': uuid2tx_gemini.get(cur_uuid, []),
                'tx_gpt-4o-2024-08-06': uuid2tx_gpt4o.get(cur_uuid, []),
                'tx_bclc': list(bclc_recommended(bclc[1])) if bclc_recommended(bclc[1]) else '',
                'tx_cnlc': list(cnlc_recommended(cnlc[1])) if cnlc_recommended(cnlc[1]) else '',
            }
        )
        
    # ========================= 并列处理 SFT：严格按 Ours 白名单 ========================= #
    whitelist_set = set(whitelist)
    # 为 SFT 构造 fallback_map（只需 Ours 的 GT & event）
    fallback_map = {row['uuid']: {'survival_time': row['survival_time'], 'event': row['event']} for row in fres}
    df_ours_sft, uuid2tx_sft, uuid2pred_sft = process_ours_like_jsonl_with_whitelist(
        jsonl_path=res_path_sft,
        whitelist_uuids=whitelist_set,
        fallback_map=fallback_map,
        out_csv_path="data/visualization/df_ours_sft_seer.csv",
        tx_threshold=0.5
    )
    # 回填到 fres
    u2row = {r['uuid']: r for r in fres}
    for u in whitelist_set:
        if u in u2row:
            u2row[u]['tx_ours_sft'] = uuid2tx_sft.get(u, [])
            u2row[u]['predicted_survival_sft'] = uuid2pred_sft.get(u, None)

    # ========================= 处理两份外部 LLM 结果并各自落盘 ========================= #
    gemini_path = "xxx"
    gpt4o_path = "xx"
    process_llm_jsonl(gemini_path, "data/visualization/df_gemini-2_5-pro_seer.csv")
    process_llm_jsonl(gpt4o_path, "data/visualization/df_gpt-4o-2024-08-06_seer.csv")

    print(f"Missing survival data for {survival_miss} samples")
    print(f"Missing BCLC staging for {bclc_miss} samples")
    print(f"Missing CNLC staging for {cnlc_miss} samples")
    print(f"Missing AJCC TNM staging for {ajcc_miss} samples")
    print(f"Parse error survival for {parse_error_survivial} samples")
    print(f"Parse error treatment for {parse_error_treatment} samples")

    df_ours = pd.DataFrame({'staging': your_staging_labels,'time': survival_months_gt,'event': death_events,'predicted': survival_months_pred})
    df_tnm  = pd.DataFrame({'staging': tnm_staging_labels,'time': tnm_survival_months_gt,'event': tnm_death_events,'predicted': tnm_risk_scores})
    df_bclc = pd.DataFrame({'staging': bclc_staging_labels,'time': bclc_survival_months_gt,'event': bclc_death_events,'predicted': bclc_risk_scores})
    df_cnlc = pd.DataFrame({'staging': cnlc_staging_labels,'time': cnlc_survival_months_gt,'event': cnlc_death_events,'predicted': cnlc_risk_scores})

    # KM 图（Ours）
    plot_km(df_ours, 'data/visualization/km_overall_seer.svg')
    plot_km_capped(df_ours, 'data/visualization/km_1yr_seer.svg', horizon_months=12)
    plot_km_capped(df_ours, 'data/visualization/km_2yr_seer.svg', horizon_months=24)
    plot_km_capped(df_ours, 'data/visualization/km_3yr_seer.svg', horizon_months=36)
    plot_km_capped(df_ours, 'data/visualization/km_4yr_seer.svg', horizon_months=48)
    plot_km_capped(df_ours, 'data/visualization/km_5yr_seer.svg', horizon_months=60)

    print(f'Total Number of Samples: {len(df_ours)}')
    c_index = concordance_index(df_ours['time'], df_ours['predicted'], df_ours['event'])
    print('Our concordance index: {:.4f}'.format(c_index), '| ', len(df_ours))
    # >>> NEW: Our-SFT 指标
    if len(df_ours_sft) > 0:
        c_index_sft = concordance_index(df_ours_sft['time'], df_ours_sft['predicted'], df_ours_sft['event'])
        print('Our-SFT concordance index: {:.4f}'.format(c_index_sft), '| ', len(df_ours_sft))

    if len(df_tnm) > 0:
        c_index = concordance_index(df_tnm['time'], df_tnm['predicted'], df_tnm['event'])
        print('TNM concordance index: {:.4f}'.format(c_index), '| ', len(df_tnm))
        c_index = concordance_index(df_bclc['time'], df_bclc['predicted'], df_bclc['event'])
        print('BCLC concordance index: {:.4f}'.format(c_index), '| ', len(df_bclc))
        c_index = concordance_index(df_cnlc['time'], df_cnlc['predicted'], df_cnlc['event'])
        print('CNLC concordance index: {:.4f}'.format(c_index), '| ', len(df_cnlc))
    
    # 保存
    df_ours.to_csv('data/visualization/df_ours_seer.csv', index=False)
    df_tnm.to_csv('data/visualization/df_tnm_seer.csv', index=False)
    df_bclc.to_csv('data/visualization/df_bclc_seer.csv', index=False)
    df_cnlc.to_csv('data/visualization/df_cnlc_seer.csv', index=False)
    # >>> NEW: 并列 SFT 的导出已在函数中输出 df_ours_sft_seer.csv
    save_jsonl(fres, "data/visualization/staging_survival_seer.jsonl")
