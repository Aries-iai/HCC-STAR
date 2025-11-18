#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch evaluation for seer / cg / multicenter across models with TTS vs Non-TTS.
GT is loaded from the staging_survival_*.jsonl files:
  * Treatment GT: "tx_actual_all" (list) -- will be normalized to canonical keys
  * Survival time: "survival_time" (months)
  * Death event: "event" (1 = death, 0 = alive)
Only uuids present in the staging file are evaluated (whitelist).
"""

import os, re, ast, json, argparse
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

# -------------------- Treatment label normalization --------------------

import re as _re

_CANONICAL_KEYS = [
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
]

def _norm_token_basic(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("－","-").replace("—","-").replace("–","-")
    s = s.replace("（","(").replace("）",")")
    s = s.lower()
    s = _re.sub(r"\s+", " ", s)
    return s

def _to_snake(s: str) -> str:
    s = _norm_token_basic(s)
    s = _re.sub(r"[\s\-]+", "_", s)
    s = _re.sub(r"[^\w]+", "", s)
    return s

_ALIAS_TO_CANON = {
    # Surgical resection
    "surgical_resection": "Surgical_resection",
    "surgicalresection": "Surgical_resection",
    "surgical": "Surgical_resection",
    "surgery": "Surgical_resection",
    "resection": "Surgical_resection",
    "手术切除": "Surgical_resection",

    # Ablation
    "ablation": "Ablation",
    "消融治疗": "Ablation",

    # Liver transplantation
    "liver_transplantation": "Liver_transplantation",
    "livertransplantation": "Liver_transplantation",
    "liver_transplant": "Liver_transplantation",
    "livertransplant": "Liver_transplantation",
    "liver transplantation": "Liver_transplantation",
    "肝移植": "Liver_transplantation",

    # Interventional therapy
    "interventional_therapy": "Interventional_therapy",
    "interventionaltherapy": "Interventional_therapy",
    "interventional": "Interventional_therapy",
    "interventional therapy": "Interventional_therapy",
    "intervention": "Interventional_therapy",
    "interventions": "Interventional_therapy",
    'TACE': "Interventional_therapy",

    # Interventional therapy + ablation
    "interventional_therapy_plus_ablation": "Interventional_therapy_plus_ablation",
    "interventionaltherapyplusablation": "Interventional_therapy_plus_ablation",
    "interventional_therapy_ablation": "Interventional_therapy_plus_ablation",
    "interventionaltherapyablation": "Interventional_therapy_plus_ablation",
    "interventional therapy plus ablation": "Interventional_therapy_plus_ablation",
    "TACE_plus_ablation": "Interventional_therapy_plus_ablation",

    # Surgical resection + ablation
    "surgical_resection_plus_ablation": "Surgical_resection_plus_ablation",
    "surgicalresectionplusablation": "Surgical_resection_plus_ablation",
    "surgery_plus_ablation": "Surgical_resection_plus_ablation",
    "surgeryablation": "Surgical_resection_plus_ablation",
    "手术切除联合消融治疗": "Surgical_resection_plus_ablation",

    # Systemic anti-tumor therapy (canonical uses hyphen as requested)
    "systemic_anti_tumor_therapy": "Systemic_anti-tumor_therapy",
    "systemic_anti-tumor_therapy": "Systemic_anti-tumor_therapy",
    "systemic anti tumor therapy": "Systemic_anti-tumor_therapy",
    "systemic anti-tumor therapy": "Systemic_anti-tumor_therapy",
    "systemic": "Systemic_anti-tumor_therapy",
    "systemictherapy": "Systemic_anti-tumor_therapy",
    "systemicantitumortherapy": "Systemic_anti-tumor_therapy",

    # Interventional therapy + systemic anti-tumor therapy
    "interventional_therapy_plus_systemic_anti_tumor_therapy": "Interventional_therapy_plus_systemic_anti-tumor_therapy",
    "interventional_therapy_plus_systemic_anti-tumor_therapy": "Interventional_therapy_plus_systemic_anti-tumor_therapy",
    "interventional therapy plus systemic anti tumor therapy": "Interventional_therapy_plus_systemic_anti-tumor_therapy",
    "interventional therapy plus systemic anti-tumor therapy": "Interventional_therapy_plus_systemic_anti-tumor_therapy",
    "interventionaltherapyplussystemicantitumortherapy": "Interventional_therapy_plus_systemic_anti-tumor_therapy",
    "TACE_plus_systemic_anti-tumor_therapy": "Interventional_therapy_plus_systemic_anti-tumor_therapy",

    # Radiotherapy
    "radiotherapy": "Radiotherapy",
    "rt": "Radiotherapy",

    # Symptomatic support
    "symptomatic_support": "Symptomatic_support",
    "symptomaticsupport": "Symptomatic_support",
    "supportivecare": "Symptomatic_support",
    "supportive_care": "Symptomatic_support",

    # Palliative care
    "palliative_care": "Palliative_care",
    "palliativecare": "Palliative_care",
}

def normalize_treatment_label(s: str) -> Optional[str]:
    if s is None:
        return None
    s_norm = _norm_token_basic(str(s))
    if s_norm in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[s_norm]
    s_snake = _to_snake(s_norm)
    return _ALIAS_TO_CANON.get(s_snake)

def normalize_treatment_scores(scores: Dict[str, float]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for k, v in (scores or {}).items():
        canon = normalize_treatment_label(k)
        if not canon:
            continue
        if canon in merged:
            merged[canon] = max(merged[canon], float(v))
        else:
            merged[canon] = float(v)
    return merged

# -------------------- Parsing helpers --------------------

def _safe_to_int(x):
    try:
        if x is None: return None
        return -1*int(float(x))
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

def parse_hard_check_survival(solution_str: str) -> Optional[Dict]:
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
    return {
        "survival_months": _safe_to_int(obj.get("survival_months")),
        "survival_1yr": _safe_to_01_or_none(obj.get("survival_1yr")),
        "survival_3yr": _safe_to_01_or_none(obj.get("survival_3yr")),
        "survival_5yr": _safe_to_01_or_none(obj.get("survival_5yr")),
    }

def parse_treatment(solution_str: str) -> Optional[Dict[str, float]]:
    m = re.search(r"<hard_check_treatment>(.*?)</hard_check_treatment>", solution_str, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    l = raw.find("{"); r = raw.rfind("}")
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
    if not isinstance(obj, dict):
        return None
    if "scores" in obj and isinstance(obj["scores"], dict):
        obj = obj["scores"]
    if not isinstance(obj, dict):
        return None
    cleaned = {}
    for k, v in obj.items():
        try:
            cleaned[str(k)] = float(v)
        except Exception:
            continue
    cleaned = normalize_treatment_scores(cleaned)
    return cleaned if cleaned else None

def majority_vote_bool(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v in (0, 1)]
    if not vals:
        return None
    c = Counter(vals).most_common(2)
    if len(c) == 1 or c[0][1] != c[1][1]:
        return c[0][0]
    return None

def majority_voting_treatment(dicts_list: List[Optional[Dict[str, float]]]) -> Tuple[Optional[Dict[str,float]], Dict[str, Optional[str]]]:
    if not dicts_list:
        return None, {'top1': None, 'top2': None, 'top3': None}

    top_k_lists = {1: [], 2: [], 3: []}
    for d in dicts_list:
        if not isinstance(d, dict):
            continue
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        for k in (1, 2, 3):
            if len(items) >= k:
                top_k_lists[k].append(items[k-1][0])

    def winner(votes):
        if not votes: return None
        c = Counter(votes).most_common(2)
        if len(c) == 1 or c[0][1] != c[1][1]:
            return c[0][0]
        return None

    top1, top2, top3 = winner(top_k_lists[1]), winner(top_k_lists[2]), winner(top_k_lists[3])

    sparse = {}
    if top1: sparse[top1] = 1.0
    if top2 and top2 != top1: sparse[top2] = 0.6
    if top3 and top3 not in (top1, top2): sparse[top3] = 0.3

    return (sparse if sparse else None), {'top1': top1, 'top2': top2, 'top3': top3}

def calculate_topn_hit_with_gt_list(majority_voted_treatments: Dict[str, str], gt_treatment_list: List[str], n: int) -> int:
    if not gt_treatment_list:
        return 0
    top_predictions = []
    for i in range(1, n + 1):
        top_key = f'top{i}'
        if top_key in majority_voted_treatments and majority_voted_treatments[top_key] is not None:
            top_predictions.append(majority_voted_treatments[top_key])
    if not top_predictions:
        return 0
    # gt_set = {str(x).strip() for x in gt_treatment_list if x is not None and str(x).strip() != ""}
    gt_set = [str(gt_treatment_list[0])]
    return 1 if any(tp in gt_set for tp in top_predictions) else 0

# -------------------- C-index calculators --------------------

def cindex_overall(pred_months: List[float], time: List[float], event: List[int]) -> Optional[float]:
    try:
        return float(concordance_index(time, -np.array(pred_months), event))
    except Exception:
        return None

def cindex_at_horizon(pred_months: List[float], time: List[float], event: List[int], horizon_m: int) -> Optional[float]:
    t_capped = np.minimum(np.array(time, dtype=float), float(horizon_m))
    e_within = ((np.array(event, dtype=int) == 1) & (np.array(time, dtype=float) <= float(horizon_m))).astype(int)
    try:
        return float(concordance_index(t_capped, -np.array(pred_months, dtype=float), e_within))
    except Exception:
        return None

# -------------------- IO --------------------

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_uuid_whitelist_and_gt(staging_path: str) -> Tuple[set, Dict[str, Dict]]:
    gt_map = {}
    whitelist = set()
    for rec in load_jsonl(staging_path):
        uid = rec.get("uuid")
        if not uid:
            continue
        whitelist.add(uid)
        tx_list = rec.get("tx_actual_all") or []
        if "tx_actual_all" not in rec:
            tx_list = rec.get("tx_actual") or []
        if isinstance(tx_list, (str,)):
            tx_list = [tx_list]
        # normalize to canonical and drop unknowns
        tx_list = [normalize_treatment_label(x) for x in tx_list]
        tx_list = [x for x in tx_list if x]
        st = rec.get("survival_time")
        ev = rec.get("event")
        try:
            st = float(st) if st is not None else None
        except Exception:
            st = None
        try:
            ev = int(ev) if ev in (0,1) or str(ev) in ("0","1") else None
        except Exception:
            ev = None
        gt_map[uid] = {
            "tx_actual_all": tx_list,
            "survival_time": st,
            "event": ev,
        }
    return whitelist, gt_map

# -------------------- Single job --------------------

def eval_one_dataset_model(
    dataset_key: str,
    model_name: str,
    mode: str,
    pred_path: str,
    staging_path: str,
) -> Dict:
    gt_all_set = []
    pred_all_set = []
    whitelist, gt_map = load_uuid_whitelist_and_gt(staging_path)
    preds = load_jsonl(pred_path)

    y_time, y_event, pred_months = [], [], []
    acc_top1, acc_top2, acc_top3 = [], [], []

    used_surv, used_tx = 0, 0
    in_whitelist = 0

    for rec in preds:
        uid = rec.get("uuid") or rec.get("extra_info").get("uuid")
        if uid not in whitelist:
            continue
        in_whitelist += 1

        gt = gt_map.get(uid, {})
        gt_surv_time = gt.get("survival_time", None)
        gt_event = gt.get("event", None)
        gt_tx_list = gt.get("tx_actual_all", [])

        results_list: List[str] = rec.get("result_ours") or []
        if not isinstance(results_list, list) or len(results_list) == 0:
            continue

        if mode == "nontts":
            results_list = results_list[:1]
        else:
            results_list = results_list[:5]

        # survival aggregation
        surv_month_candidates = []
        for sol in results_list:
            p = parse_hard_check_survival(sol)
            if p and p.get("survival_months") is not None:
                surv_month_candidates.append(p["survival_months"])
        pred_m = float(np.median(surv_month_candidates)) if surv_month_candidates else None

        if pred_m is not None and (gt_surv_time is not None) and (gt_event in (0,1)):
            pred_months.append(pred_m)
            y_time.append(float(gt_surv_time))
            y_event.append(int(gt_event))
            used_surv += 1

        # treatment voting and hit
        tx_dicts = [parse_treatment(sol) for sol in results_list]
        _, voted_tops = majority_voting_treatment(tx_dicts)

        if gt_tx_list:
            for it in voted_tops:
                if 'top' in it:
                    pred_all_set.append(str(voted_tops[it]))
            for it in gt_tx_list:
                gt_all_set.append(it)
            acc_top1.append(calculate_topn_hit_with_gt_list(voted_tops, gt_tx_list, 1))
            acc_top2.append(calculate_topn_hit_with_gt_list(voted_tops, gt_tx_list, 2))
            acc_top3.append(calculate_topn_hit_with_gt_list(voted_tops, gt_tx_list, 3))
            used_tx += 1

    metrics = {
        "survival": {
            "cindex_overall": cindex_overall(pred_months, y_time, y_event) if used_surv > 1 else None,
            "cindex_1yr": cindex_at_horizon(pred_months, y_time, y_event, 12) if used_surv > 1 else None,
            "cindex_3yr": cindex_at_horizon(pred_months, y_time, y_event, 36) if used_surv > 1 else None,
            "cindex_5yr": cindex_at_horizon(pred_months, y_time, y_event, 60) if used_surv > 1 else None,
        },
        "treatment": {
            "top1": float(np.mean(acc_top1)) if acc_top1 else None,
            "top2": float(np.mean(acc_top2)) if acc_top2 else None,
            "top3": float(np.mean(acc_top3)) if acc_top3 else None,
        }
    }

    return {
        "dataset": dataset_key,
        "model": model_name,
        "mode": mode,
        "counts": {
            "all_pred": len(preds),
            "in_whitelist": in_whitelist,
            "used_for_surv": used_surv,
            "used_for_tx": used_tx,
            "pred_all_set": list(set(pred_all_set)),
            "gt_all_set": list(set(gt_all_set)),
        },
        "metrics": metrics
    }

# -------------------- Dataset paths --------------------

def build_paths(model_name: str) -> Dict[str, Tuple[str, str]]:
    return {
        "seer": (
            f"/share/home/cuipeng/results//share/home/cuipeng/data/liver_rl_full_with_tag_v_0915_concat_seer_test_v0916_filter_exclude_0_fixed_all_fixed_seer_{model_name}_5_turns.jsonl",
            "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_seer.jsonl",
        ),
        "cg": (
            f"/share/home/cuipeng/cuipeng_a100/siyan/comined_chunggeng_20250919_full_v0_{model_name}_5_turns.jsonl",
            "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_cg_3.jsonl",
        ),
        "multicenter": (
            f"/share/home/cuipeng/cuipeng_a100/siyan/hcc_external_center_processed_20250920_v1_full_4o_v0_formatted_v4_{model_name}_5_turns.jsonl",
            "/share/home/cuipeng/cuipeng_a100/siyan/visualization_code_ideal/staging_survival_multicenter.jsonl",
        )
    }

def _task(args):
    return eval_one_dataset_model(*args)

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation: TTS vs Non-TTS on seer/cg/multicenter × models (GT from staging files, with treatment label normalization)")
    parser.add_argument("--models", nargs="+", default=["qwen_8b","r1_qwen_32b","r1_qwen_7b","qwq_32b"])
    parser.add_argument("--out_json", default="./eval_summary_tts_vs_nontts_gt.json")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count()//2))
    args = parser.parse_args()

    jobs = []
    for m in args.models:
        paths = build_paths(m)
        for ds, (pred_path, staging_path) in paths.items():
            for mode in ("tts", "nontts"):
                if not os.path.exists(pred_path):
                    print(f"[WARN] missing pred file: {pred_path}")
                    continue
                if not os.path.exists(staging_path):
                    print(f"[WARN] missing staging file: {staging_path}")
                    continue
                jobs.append((ds, m, mode, pred_path, staging_path))

    results = []
    if args.workers > 1 and len(jobs) > 1:
        with Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_task, jobs):
                results.append(r)
    else:
        for j in jobs:
            results.append(_task(j))

    summary = defaultdict(lambda: defaultdict(dict))
    for r in results:
        ds, m, mode = r["dataset"], r["model"], r["mode"]
        summary[ds][m][mode] = {"counts": r["counts"], "metrics": r["metrics"]}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    flat_rows = []
    for ds, dsv in summary.items():
        for m, mv in dsv.items():
            for mode, rv in mv.items():
                sv = rv["metrics"]["survival"]; tv = rv["metrics"]["treatment"]; ct = rv["counts"]
                flat_rows.append({
                    "dataset": ds, "model": m, "mode": mode,
                    "n_pred": ct["all_pred"], "n_in_white": ct["in_whitelist"],
                    "n_surv": ct["used_for_surv"], "n_tx": ct["used_for_tx"],
                    "cindex_overall": sv["cindex_overall"],
                    "cindex_1yr": sv["cindex_1yr"],
                    "cindex_3yr": sv["cindex_3yr"],
                    "cindex_5yr": sv["cindex_5yr"],
                    "top1": tv["top1"], "top2": tv["top2"], "top3": tv["top3"],
                })
    df = pd.DataFrame(flat_rows)
    csv_path = os.path.splitext(args.out_json)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary JSON to: {args.out_json}")
    print(f"Saved flat CSV to: {csv_path}")

if __name__ == "__main__":
    main()
