import re
import math
import json
import ast
try:
    from numpy import exp
except ImportError:
    exp = lambda x: 2.718281828459045 ** x


# ---------- 基础工具 ----------
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

def _max_passed_threshold_months_from_flags(flags):
    """
    给定 flags（可能含 1/0/None），返回“已跨过的最大阈值对应的月数”，无则 None。
    """
    mapping = [("survival_1yr", 12), ("survival_3yr", 36), ("survival_5yr", 60)]
    passed = [m for k, m in mapping if flags.get(k) == 1]
    return max(passed) if passed else None

# ---------- 解析预测 ----------
def parse_hard_check_survival(solution_str):
    """
    抓取 <hard_check_survival>...</hard_check_survival> 并稳健解析为标准 dict:
      {"survival_months": int|None, "survival_1yr":0/1/None, "survival_3yr":..., "survival_5yr":...}
    失败返回 None
    """
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

# ---------- 解析 GT（含 event） ----------
def _extract_gt_survival_with_event(ground_truth):
    """
    允许 ground_truth 是 str/dict；支持两种结构：
      1) 平铺: {"survival_months":..., "survival_1yr":..., ..., "if_death":0/1}
    返回 (gt_dict_all, gt_surv_dict, event)
    """
    if isinstance(ground_truth, str):
        try:
            gt_all = json.loads(ground_truth)
        except Exception:
            return None, None, None
    elif isinstance(ground_truth, dict):
        gt_all = ground_truth
    else:
        return None, None, None

    # event 既可能在外层也可能和 survival 放一起
    event = gt_all.get("if_death", None)
    event = _safe_to_01_or_none(event)  # 0/1/None

    # survival 可能在外层或在 gt_all["survival"]
    base = gt_all.get("survival_months") if isinstance(gt_all.get("survival_months"), dict) else gt_all

    gt_months = _safe_to_int(base.get("survival_months"))
    gt_1yr = _safe_to_01_or_none(base.get("survival_1yr"))
    gt_3yr = _safe_to_01_or_none(base.get("survival_3yr"))
    gt_5yr = _safe_to_01_or_none(base.get("survival_5yr"))

    gt_surv = {
        "survival_months": gt_months,
        "survival_1yr": gt_1yr,
        "survival_3yr": gt_3yr,
        "survival_5yr": gt_5yr,
    }
    return gt_all, gt_surv, event

def _derive_flags_from_event_months(event, months):
    """
    用 (event, months) 派生 1/3/5 年 flags。
    - event=1: 在某阈值前死亡 => 若 months < 12*k 则该阈值标记为 0，否则 1
    - event=0: 删失 => 已跨过的阈值标 1；未跨过的标 None
    - event=None: 全部 None
    """
    flags = {"survival_1yr": None, "survival_3yr": None, "survival_5yr": None}
    thresholds = [("survival_1yr", 12), ("survival_3yr", 36), ("survival_5yr", 60)]

    if months is None or event is None:
        return flags

    if event == 1:
        for k, m in thresholds:
            flags[k] = 0 if months < m else 1
    else:  # event == 0
        for k, m in thresholds:
            flags[k] = 1 if months >= m else None
    return flags

# ---------- 三部分打分 ----------
def _flags_correctness(pred_flags, gt_flags):
    keys = ["survival_1yr", "survival_3yr", "survival_5yr"]
    valid = [k for k in keys if gt_flags.get(k) in (0, 1)]
    if not valid:
        return 1.0
    correct = sum(1 for k in valid if pred_flags.get(k) in (0,1) and pred_flags[k] == gt_flags[k])
    if (correct / len(valid)) == 1.0:
        return 1.0
    else:
        return 0

def _consistency_factor(pred_months, pred_flags, penalty_per_violation=0.25):
    if pred_months is None:
        return 1.0
    factor = 1.0
    for k, m in [("survival_1yr", 12), ("survival_3yr", 36), ("survival_5yr", 60)]:
        v = pred_flags.get(k)
        if v not in (0, 1): 
            continue
        if pred_months >= m and v == 0:
            factor -= penalty_per_violation
        if pred_months < m and v == 1:
            factor -= penalty_per_violation
    return max(0.0, factor)

def _months_score_with_event(pred_months, gt_months, event,
                             gt_flags=None, tau_death=12.0, tau_censor=6.0):
    """
    event=1：对称误差指数衰减；event=0：到 [t_c, ∞) 的单侧距离指数衰减。
    当删失且无 gt_months 时，用 flags 推一个下界；仍无信息则给中性分 0.5。
    """
    if pred_months is None:
        return 0.0

    if event == 1:
        if gt_months is None:
            return 0.0
        diff = abs(pred_months - gt_months)
        return exp(-diff / float(tau_death))

    if event == 0:
        # 优先用 gt_months 作为下界；没有就用 flags 推
        if gt_months is not None:
            t_c = gt_months
        else:
            t_c = _max_passed_threshold_months_from_flags(gt_flags or {})
            if t_c is None:
                # 完全没法得到下界，给中性分
                return 0.5
        d = max(0.0, float(t_c) - float(pred_months))
        return exp(-d / float(tau_censor))

    # event 未知
    return 0.5

# ---------- 主函数：带 event 的生存打分 ----------
def compute_survival_score_with_event(solution_str, ground_truth,
                                      w_flags=0.5, w_consistency=0, w_months=0.5,
                                      tau_death=12.0, tau_censor=6.0):
    """
    survival_score = w_flags*flags_score + w_consistency*consistency + w_months*months_score
    - flags_score：与 GT 的 1/3/5 年标记比对（若 GT 未给，则由 (event, gt_months) 派生）
    - consistency：预测内部一致性（pred_months 与 pred_flags 自洽）
    - months_score：event=1 用对称误差；event=0 用删失下界的单侧距离
    """
    pred = parse_hard_check_survival(solution_str)
    if pred is None:
        return 0.0

    gt_all, gt_flags_raw, event = _extract_gt_survival_with_event(ground_truth)
    if gt_flags_raw is None:
        return 0.0

    # 若 GT flags 缺失，则用 event+gt_months 派生
    need_derived = any(gt_flags_raw.get(k) not in (0,1) for k in ["survival_1yr","survival_3yr","survival_5yr"])
    if need_derived:
        derived = _derive_flags_from_event_months(event, gt_flags_raw.get("survival_months"))
        # 用“给定优先、派生补齐”的策略
        gt_flags = {}
        for k in ["survival_1yr","survival_3yr","survival_5yr"]:
            gt_flags[k] = gt_flags_raw.get(k) if gt_flags_raw.get(k) in (0,1) else derived.get(k)
        # 维持 months 原值
        gt_flags["survival_months"] = gt_flags_raw.get("survival_months")
    else:
        gt_flags = gt_flags_raw

    # 1) flags correctness
    flags_score = _flags_correctness(pred, gt_flags)

    # 2) 预测内部一致性
    # consistency = _consistency_factor(pred.get("survival_months"), pred)

    print(f'Predicted survival month: {pred.get("survival_months")}')
    # 3) 月份误差项（区分 event）
    months_score = _months_score_with_event(
        pred_months=pred.get("survival_months"),
        gt_months=gt_flags.get("survival_months"),
        event=event,
        gt_flags=gt_flags,
        tau_death=tau_death,
        tau_censor=tau_censor
    )

    # score = w_flags*flags_score + w_consistency*consistency + w_months*months_score
    score = w_flags*flags_score + w_months*months_score
    return max(0.0, min(1.0, score))

def compute_process_score(solution_str, ground_truth):
    """
    计算分析过程得分：检查 <ps>, <child_pugh>, <metastasis>, <cancer_thrombus>, 
    <num_tumor>, <tumor_size> 是否与 ground truth 匹配。
    
    参数:
        solution_str (str): 模型生成的响应。
        ground_truth (str): 正确答案（JSON 格式）。
    
    返回:
        float: 分析过程得分（每匹配一项得 1/6 分，最高 1 分）。
    """
    try:
        gt_dict = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        print("decode error hhh")
        return 0.0  # ground truth 格式错误，返回 0 分

    # 提取 ground truth 中的字段
    gt_fields = {
        "ps": str(gt_dict.get("ps", "")),
        "child_pugh": str(gt_dict.get("child_pugh", "")),
        "metastasis": str(gt_dict.get("metastasis", "")),
        "cancer_thrombus": str(gt_dict.get("cancer_thrombus", "")),
        "num_tumor": str(gt_dict.get("num_tumor", "")),
        "tumor_size": str(gt_dict.get("tumor_size", ""))
    }
    
    process_score = 0.0
    fields_to_check = ["ps", "child_pugh", "metastasis", "cancer_thrombus", "num_tumor", "tumor_size"]
    
    for field in fields_to_check:
        pattern = rf"<{field}>(.*?)</{field}>"
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            value = match.group(1).strip()
            print("matched value is {}".format(ValueError))
            if value == gt_fields[field]:
                process_score += 1.0 / 6.0  # 每匹配一项得 1/6 分
        else:
            print("match failed")
    
    return process_score

def compute_format_score(solution_str, ground_truth):
   """
   计算格式得分：检查指定标签是否出现在输出中，包括 <hard_check>。
   除了 "treatment", "treatment_not_recommended" 外，其他标签只能出现一次，多次出现按次数比例扣分。
   
   参数:
       solution_str (str): 模型生成的响应。
       ground_truth (str): 正确答案（未使用，仅为接口一致性）。
   
   返回:
       float: 格式得分（每个标签得 1/11 分，按出现次数比例折扣，最高 1 分）。
   """
   tags_to_check = [
       "ps", "child_pugh", "metastasis", "cancer_thrombus",
       "num_tumor", "tumor_size", "treatment", "treatment_not_recommended", 
       "hard_check", "thinking", "answer"
   ]
   
   # 允许多次出现的标签
   multiple_allowed_tags = {"treatment", "treatment_not_recommended"}
   
   format_score = 0.0
   
   for tag in tags_to_check:
       start_tag = f"<{tag}>"
       end_tag = f"</{tag}>"
       
       # 检查标签是否存在
       if start_tag in solution_str and end_tag in solution_str:
           # 计算出现次数
           start_count = solution_str.count(start_tag)
           end_count = solution_str.count(end_tag)
           
           if tag in multiple_allowed_tags:
               # 对于允许多次出现的标签，只要有配对就得分
               if start_count > 0 and end_count > 0:
                   format_score += 1.0 / 11.0
           else:
               # 对于只能出现一次的标签，按出现次数比例折扣
               if start_count > 0 and end_count > 0:
                   # 取开始和结束标签的最大出现次数作为实际出现次数
                   actual_count = max(start_count, end_count)
                   # 按比例折扣：1 / actual_count
                   discount_factor = 1.0 / actual_count
                   tag_score = (1.0 / 11.0) * discount_factor
                   format_score += tag_score
                   
                   if actual_count > 1:
                       print(f"警告: 标签 <{tag}> 出现{actual_count}次，按比例折扣得分: {tag_score:.4f}")
   
   print(f"格式得分: {format_score:.3f}")
   return format_score

def compute_length_penalty_score(
    solution_str: str,
    max_length: int = 3072,
    min_length: int = 2048,
    smooth_window: int = 128,
    verbose: bool = True,
) -> float:
    """
    计算回复长度惩罚得分（软化阈值版本）。
    逻辑保持原意：
      - x <= min:     0.3 * (x / min)
      - min <= x <= max: x / max
      - x > max:      0.5 * (max / x)
    在 min 与 max 附近使用余弦平滑过渡，避免奖励突变。

    参数:
        solution_str (str): 模型生成的响应。
        max_length (int): 最大长度，默认 3072。
        min_length (int): 最小长度，默认 2048。
        smooth_window (int): 平滑窗口半宽（越大越平滑），默认 128。
        verbose (bool): 是否打印信息，默认 True。

    返回:
        float: 惩罚得分（0.0 ~ 1.0）
    """
    # 基本检查
    if min_length <= 0 or max_length <= 0 or max_length < min_length:
        raise ValueError("min_length 和 max_length 必须为正，且 max_length >= min_length")

    actual_length = len(solution_str)

    def base_low(x: float) -> float:
        return 0.2 * (x / min_length)

    def base_mid(x: float) -> float:
        return x / max_length

    def base_high(x: float) -> float:
        return 0.2 * (max_length / x)

    def smoothstep(t: float) -> float:
        # t ∈ [0, 1]
        return 0.5 - 0.5 * math.cos(math.pi * t)

        if actual_length <= min_length:
            penalty_score = base_low(actual_length)
        elif actual_length <= max_length:
            penalty_score = base_mid(actual_length)
        else:
            penalty_score = base_high(actual_length)
        if verbose:
            print(
                f"回复长度: {actual_length}, 最大长度: {max_length}, 最小长度: {min_length}, "
                f"惩罚得分: {penalty_score:.3f} (硬分段)"
            )
        return float(penalty_score)

    # 软化阈值分段
    if actual_length <= min_length - smooth_window:
        penalty_score = base_low(actual_length)
    elif actual_length < min_length + smooth_window:
        # 低段 -> 中段 过渡
        a = base_low(actual_length)
        b = base_mid(actual_length)
        t = (actual_length - (min_length - smooth_window)) / (2 * smooth_window)  # 0..1
        penalty_score = a * (1 - smoothstep(t)) + b * smoothstep(t)
    elif actual_length <= max_length - smooth_window:
        penalty_score = base_mid(actual_length)
    elif actual_length < max_length + smooth_window:
        # 中段 -> 高段 过渡
        a = base_mid(actual_length)
        b = base_high(actual_length)
        t = (actual_length - (max_length - smooth_window)) / (2 * smooth_window)  # 0..1
        penalty_score = a * (1 - smoothstep(t)) + b * smoothstep(t)
    else:
        penalty_score = base_high(actual_length)

    if verbose:
        print(
            f"回复长度: {actual_length}, 最大长度: {max_length}, 最小长度: {min_length}, "
            f"平滑窗口: {smooth_window}, 惩罚得分: {penalty_score:.3f}"
        )
    return float(penalty_score)



def is_valid_float(value):
    """检查值是否可以转换为有效的浮点数"""
    try:
        float_value = float(value)
        return not (float_value != float_value)  # 排除 NaN
    except (ValueError, TypeError):
        return False



def compute_treatment_score(solution_str, ground_truth):
   """
   计算治疗推荐得分：使用DCG方式比较预测结果 <hard_check> 中的前三个治疗方案与 ground truth 的 top3_treatments。
        
   参数:
       solution_str (str): 模型生成的响应，包含 <hard_check> 标签。
       ground_truth (str): 正确答案（JSON 格式，包含 top3_treatments）。
        
   返回:
       float: 治疗推荐得分（基于nDCG，0.0-1.0）。
   """
   # 解析 ground truth
   try:
       gt_dict = json.loads(ground_truth)
   except (json.JSONDecodeError, TypeError):
       return 0.0  # ground truth 格式错误，返回 0 分

   # 提取 ground truth 中的 top3_treatments
   if "top3_treatments" not in gt_dict:
       return 0.0  # ground truth 中没有 top3_treatments，返回 0 分
       
   gt_treatments = gt_dict["top3_treatments"]
   gt_treatment_names = [t["treatment"] for t in gt_treatments][:3]  # 取前三个治疗方案名称
   print("gt_treatment_names: {}".format(gt_treatment_names))

   # 提取 solution_str 中的 <hard_check> 标签
   hard_check_pattern = r"<hard_check_treatment>(.*?)</hard_check_treatment>"
   hard_check_match = re.search(hard_check_pattern, solution_str, re.DOTALL)
   print("hard_check_match: {}".format(hard_check_match))
   if not hard_check_match:
       print("no hard check")
       return 0.0  # 没有 hard_check 标签，返回 0 分
       
   # 解析 hard_check 中的 scores
   try:
       fixed_hard_check = hard_check_match.group(1).strip().encode().decode('unicode_escape')
       hard_check = json.loads(fixed_hard_check.replace("'", '"'))
       print(f"hard_check: {hard_check}")
   except json.JSONDecodeError:
       print(hard_check_match.group(1).strip())
       return 0.0  # hard_check 格式错误，返回 0 分
       
   if not isinstance(hard_check, dict):
       print(f"错误: hard_check 不是字典，而是 {type(hard_check)}: {hard_check}")
       return 0.0  # 返回默认值以继续运行

   if "scores" not in hard_check:
       print(f"错误: hard_check 中没有 scores")
       return 0.0  # hard_check 中没有 scores，返回 0 分
       
   # 检查 hard_check["scores"] 是否为字典
   if not isinstance(hard_check["scores"], dict):
       print(f"错误: hard_check['scores'] 不是字典，而是 {type(hard_check['scores'])}: {hard_check['scores']}")
       return 0.0  # 返回默认值以继续运行

   pred_treatments = sorted(
       [(k, float(v)) for k, v in hard_check["scores"].items() if v is not None and is_valid_float(v) and float(v) > 0],
       key=lambda x: x[1],
       reverse=True
   )[:3]
   pred_treatment_names = [t[0] for t in pred_treatments]

   # 使用DCG计算得分
   gt_set = set(gt_treatment_names)
   top_k = 3
   
   # 计算DCG
   dcg = 0.0
   for i, pred in enumerate(pred_treatment_names):
       if i >= top_k:
           break
       if pred.strip() in gt_set:
           dcg += 1.0 / math.log2(i + 2)  # i从0开始，+2避免分母为0
   
   # 计算理想DCG
   ideal_dcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(gt_set), top_k))])
   print(f"ideal_dcg: {ideal_dcg}")
   
   # 返回nDCG
   treatment_score = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
   
   return treatment_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    主奖励函数：计算分析过程得分、格式得分和治疗推荐得分的总和。
    
    参数:
        solution_str (str): 模型生成的响应。
        ground_truth (str): 正确答案（JSON 格式）。
        extra_info (dict, optional): 其他元信息。
    
    返回:
        float: 总奖励分数。
    """
    # 确保输入为字符串
    solution_str = str(solution_str).strip()
    if isinstance(ground_truth, dict):
        ground_truth = json.dumps(ground_truth, ensure_ascii=False)
    else:
        ground_truth = str(ground_truth).strip()
    
    print("Ground truth type:", type(ground_truth))
    print("Ground truth content:", ground_truth)
    
    # 计算三个部分的得分
    process_score = compute_process_score(solution_str, ground_truth)
    format_score = compute_format_score(solution_str, ground_truth)
    penalty_score = compute_length_penalty_score(solution_str, max_length=2048, min_length=1536)
    # penalty_score = compute_length_penalty_score(solution_str, max_length=512)
    treatment_score = compute_treatment_score(solution_str, ground_truth)
    survival_score = compute_survival_score_with_event(solution_str, ground_truth)
    print("process_score: {}\n".format(process_score))
    print("format_score: {}\n".format(format_score))
    print("treatment_score: {}\n".format(treatment_score))
    print("survival_score: {}\n".format(survival_score))
    print("response_length: {}\n".format(len(solution_str)))
    #print("penalty_score: {}\n".format(penalty_score))
    # 总得分
    # total_score =  0.3 * (process_score + format_score + penalty_score) + 0.7 * (treatment_score + survival_score)
    total_score = penalty_score + process_score + format_score  + treatment_score + 1.5 * survival_score
    
    return total_score

if __name__ == '__main__':
    solution_str = '<thinking>对于治疗方案的选择:患者是一位40岁的中国籍男性，病理诊断为原发性肝细胞癌（Hepatocellular carcinoma）。根据患者的体力状态评分（<ps>2</ps>）和Child-Pugh评分（<child_pugh>B</child_pugh>），患者的全身状态尚可，但肝功能仍有一定程度的受损。因为患者<metastasis>无</metastasis>远处转移，<cancer_thrombus>无</cancer_thrombus>影像可见血管侵犯，可以进一步根据肿瘤的大小和数量进行评估。患者的肿瘤最大直径为<tumor_size>153</tumor_size>mm（大于5cm），且显示仅有<num_tumor>1</num_tumor>个单发肿瘤。因此，推荐的治疗选项包括<treatment>Surgical resection</treatment>、<treatment>TACE</treatment>、<treatment>Ablation</treatment>、或联合方案如<treatment>TACE plus ablation</treatment>，可能适合进一步评估。结合患者的影像学评估显示肿瘤尚未发生<metastasis>无</metastasis>远处转移，影像学和病理检查<cancer_thrombus>无</cancer_thrombus>血管侵犯。此外，需注意患者是否在后续治疗中存在适应症的变化。\\n对于生存状态的预测:基于患者现有的检查数据和治疗选项建议，虽然患者的体力状态和肝功能有所下降，但总体仍旧处于可接受的治疗窗口内。因<metastasis>无</metastasis>远处转移及<cancer_thrombus>无</cancer_thrombus>明确血管侵犯，患者采取综合或局部治疗方案的效果通常相对较好，预后更倾向于带瘤生存，生存时间会有一定延长。因此，我推测患者可能以较好的生活质量存活较长时间。</thinking><answer>经过思考，最终推荐治疗方案为:<hard_check_treatment>{\\\"scores\\\": {\\\"Surgical_resection\\\": 0.8, \\\"Ablation\\\": 0.6, \\\"Liver_transplantation\\\": 0.5, \\\"TACE\\\": 0.9, \\\"TACE_plus_ablation\\\": 0.7, \\\"Surgical_resection_plus_ablation\\\": 0.6, \\\"Systemic_anti-tumor_therapy\\\": 0.3, \\\"TACE_plus_systemic_anti_tumor_therapy\\\": 0.5, \\\"Radiotherapy\\\": 0.4, \\\"Symptomatic_support\\\": 0.2, \\\"Palliative_care\\\": 0.1}}</hard_check_treatment>\\n最终预测的生存状态为: <hard_check_survival>{\\\"survival_months\\\": 68, \\\"survival_1yr\\\": 1, \\\"survival_3yr\\\": 1, \\\"survival_5yr\\\": 1}</hard_check_survival></answer>'
    ground_truth = {
    "ps": "2",
    "child_pugh": "B",
    "metastasis": "无",
    "cancer_thrombus": "无",
    "num_tumor": "1",
    "tumor_size": "76",
    "survival_months": 49,
    "if_death": 1,
    "survival_1yr": 1,
    "survival_3yr": 1,
    "survival_5yr": 0,
    "top3_treatments": [
    {
    "treatment": "TACE",
    "score": 0.9
    },
    {
    "treatment": "TACE_plus_ablation",
    "score": 0.8
    },
    {
    "treatment": "Symptomatic_support",
    "score": 0.8
    }
    ]
    }
    print(compute_score('', solution_str, ground_truth))
