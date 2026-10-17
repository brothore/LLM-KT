"""Copied functions from pykt examples/cal_unbalance.py. MIT, copyright 2022 pykt-team.
See third_party/pykt/provenance.json. No upstream runtime imports.
"""
import numpy as np

def calculate_wealth_metrics(wealth_list, alphas=[1.0, 2.0, 3.0]):
    """
    计算平均财富、基尼系数、以及多个 alpha 下的 EAWI 指标。
    EAWI_alpha = average_wealth * (1 - G) ** alpha
    """
    if not wealth_list or len(wealth_list) == 0:
        avg = 0.0
        G = 0.0
        eawi_dict = {f'eawi_alpha_{a}': 0.0 for a in alphas}
        return avg, G, eawi_dict

    wealth = np.array(wealth_list)
    n = len(wealth)
    W = float(np.sum(wealth))
    average_wealth = W / n

    G = gini_coefficient(wealth_list)  # 复用你写好的函数

    eawi_dict = {}
    for alpha in alphas:
        eawi = average_wealth * ((1 - G) ** alpha)
        eawi_dict[f'eawi_alpha_{alpha:.1f}'.replace('.', '')] = float(eawi)
        # 结果键名：eawi_alpha_10, eawi_alpha_20, eawi_alpha_30

    return average_wealth, G, eawi_dict

def gini_coefficient(wealth):
    """
    独立的函数，用于计算基尼系数 (Gini Coefficient)。
   
    参数:
    wealth (list or array-like): 财富列表，每个元素代表一个个体的财富。
   
    返回:
    float: 基尼系数 (0 到 1 之间)。
    """
    if len(wealth) == 0:
        return 0.0
    wealth = np.array(wealth)
    if np.all(wealth == 0):
        return 0.0
    # 确保财富非负（基尼系数假设非负值）
    if np.any(wealth < 0):
        raise ValueError("财富值不能为负数")
    sorted_wealth = np.sort(wealth)
    n = len(wealth)
    index = np.arange(1, n + 1)
    # 标准基尼系数公式
    numerator = np.sum((2 * index - n - 1) * sorted_wealth)
    denominator = n * np.sum(sorted_wealth)
    return numerator / denominator

def sliding_window_collect(trues_series, scores_series, window_size=200):
    """
    对一个学生的完整序列进行滑窗（每个时间步都是一个数据点），
    并将所有窗口的数据点汇集起来。

    Args:
        trues_series (np.array): 学生的真实标签完整序列（每个元素是一个时间步）。
        scores_series (np.array): 学生的预测得分完整序列（每个元素是一个时间步）。
        window_size (int): 滑窗大小，默认为200。

    Returns:
        tuple: (collected_trues, collected_scores)
               collected_trues (list): 汇集后的真实标签数据点。
               collected_scores (list): 汇集后的预测得分数据点。
    """
    if len(trues_series) != len(scores_series):
        raise ValueError("True labels and scores must have the same length.")

    total_len = len(trues_series)
    
    collected_trues = []
    collected_scores = []

    # 如果序列长度小于窗口大小，则只取完整的序列
    if total_len < window_size:
        return trues_series.tolist(), scores_series.tolist()
    
    # 滑窗：从第一个位置开始，每次移动一步
    for start in range(total_len - window_size + 1):
        end = start + window_size
        
        # 收集当前窗口的数据
        collected_trues.extend(trues_series[start:end].tolist())
        collected_scores.extend(scores_series[start:end].tolist())
            
    return collected_trues, collected_scores
