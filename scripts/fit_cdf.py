#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_ecdf.py  (parallel)

提供 fit_ecdf() 来根据训练集 survival months 拟合经验 CDF（ECDF），
并以多进程方式并行解析 JSONL，大幅加速大文件处理。
"""

import os
import json
import numpy as np
import pickle
import warnings
from typing import Sequence, Optional, Dict, Any, Iterable, Tuple, Optional as Opt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------------- 可配置区 ---------------- #
# 进程数（<=0 表示使用 CPU-1）
NUM_WORKERS = max(1, (cpu_count() or 2) - 1)
# imap_unordered 的块大小
CHUNK_SIZE = 1024
# 是否按你原来的方式转成 int（月数取整）
CAST_TO_INT = True
# ------------------------------------------------------ #

def fit_ecdf(train_times: Sequence[float],
             save_pickle: Optional[str] = None,
             save_npz: Optional[str] = None) -> Dict[str, Any]:
    """
    Fit an empirical CDF from training survival months.

    Args:
        train_times: 1D sequence (list/np.array) of survival months (numeric). NaN 会被忽略（并警告）。
        save_pickle: 可选，若给出路径则把 ECDF 对象以 pickle 保存到该路径。
        save_npz: 可选，若给出路径则把 ECDF 以 npz (numpy) 格式保存到该路径。

    Returns:
        ecdf_obj: dict:
            - 'sorted_times': 1D numpy array of sorted training times (float)
            - 'n': sample size after removing NaNs (int)
    """
    arr = np.asarray(train_times, dtype=float)
    if np.isnan(arr).any():
        warnings.warn("train_times contains NaN values: they will be ignored when fitting ECDF.")
        arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("No valid training times provided after removing NaNs.")
    sorted_times = np.sort(arr)
    n = sorted_times.size
    ecdf_obj = {"sorted_times": sorted_times, "n": int(n)}

    if save_pickle is not None:
        with open(save_pickle, "wb") as f:
            pickle.dump(ecdf_obj, f)

    if save_npz is not None:
        np.savez(save_npz, sorted_times=sorted_times, n=n)

    return ecdf_obj

# -------- 并行解析工具 -------- #
def _parse_survival_from_line(line: str) -> Opt[float]:
    """
    安全解析单行 JSON，返回 survival_months（float 或 None）。
    数据路径：["reward_model"]["ground_truth"]["survival_months"]
    """
    try:
        obj = json.loads(line)
        v = obj["reward_model"]["ground_truth"]["survival_months"]
        x = float(v)
        if CAST_TO_INT:
            x = float(int(x))  # 与原脚本一致：先转 float 再取整，再转回 float
        # 这里如需过滤极端值可加判断，例如 x>=1 才保留
        return x
    except Exception:
        return None

def load_train_times_parallel(jsonl_path: str,
                              num_workers: int = NUM_WORKERS,
                              chunk_size: int = CHUNK_SIZE) -> Tuple[np.ndarray, int]:
    """
    并行解析 JSONL，提取 survival_months。
    返回：(np.array(train_times), skipped_count)
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    # 逐行读取（不一次性构造成 dict 列表，避免占大量内存）
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        return np.array([], dtype=float), 0

    # 单进程 fallback（非常小的文件可以这样）
    if num_workers <= 1:
        vals = []
        skipped = 0
        for line in tqdm(lines, desc="Parsing JSONL (1 worker)"):
            x = _parse_survival_from_line(line)
            if x is None or np.isnan(x):
                skipped += 1
            else:
                vals.append(x)
        return np.asarray(vals, dtype=float), skipped

    # 多进程
    vals = []
    skipped = 0
    with Pool(processes=num_workers) as pool:
        for x in tqdm(pool.imap_unordered(_parse_survival_from_line, lines, chunksize=chunk_size),
                      total=total, desc=f"Parsing JSONL ({num_workers} workers)"):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                skipped += 1
            else:
                vals.append(x)

    return np.asarray(vals, dtype=float), skipped

# -------- 兼容你原来的 API -------- #
def load_jsonl(file_path: str):
    """（保留原函数名以兼容）读取 JSONL -> list[dict]（注意：此函数非并行）。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    # ====== 把这个路径改成你的训练 JSONL 路径 ====== #
    train_data_path = 'xxx'

    # 并行解析 survival_months
    train_times, skipped = load_train_times_parallel(train_data_path, num_workers=NUM_WORKERS, chunk_size=CHUNK_SIZE)
    print(f"[INFO] Loaded {len(train_times)} survival months. Skipped: {skipped}")

    # 如需调试，可恢复你以前的顺序版以做对照（不建议大文件）：
    # train_data = load_jsonl(train_data_path)
    # train_times_seq = []
    # for sample in tqdm(train_data, desc="Sequential sanity-check"):
    #     v = sample["reward_model"]["ground_truth"]["survival_months"]
    #     train_times_seq.append(float(int(float(v))))

    # 训练 ECDF（并保存为 pkl/npz）
    ecdf = fit_ecdf(train_times, save_pickle="ecdf_train.pkl", save_npz="ecdf_train.npz")
    print(f"[OK] ECDF fitted. n={ecdf['n']}  saved to ecdf_train.pkl / ecdf_train.npz")
