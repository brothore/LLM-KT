import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import sys
import os
ENABLE_SLIDING_WINDOW = True
WINDOW_SIZE = 200
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
def safe_roc_auc(y_true, y_score, dummy_score_strategy='mean'):
    """
    计算 AUC，处理单一类别情况通过添加虚拟数据点。
    :param y_true: 真实标签 (numpy array)
    :param y_score: 预测概率 (numpy array)
    :param dummy_score_strategy: 虚拟点分数策略 ('mean', 'min', 'max')
    :return: AUC 分数
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    unique_labels = np.unique(y_true)
    
    if len(unique_labels) < 2:
        label = unique_labels[0]
        dummy_label = 1 - label  # 相反类别
        # 选择虚拟点的预测分数
        if dummy_score_strategy == 'mean':
            dummy_score = np.mean(y_score)
        elif dummy_score_strategy == 'min':
            dummy_score = np.min(y_score)
        elif dummy_score_strategy == 'max':
            dummy_score = np.max(y_score)
        else:
            raise ValueError("Invalid dummy_score_strategy")
        # 添加虚拟数据点
        y_true = np.append(y_true, dummy_label)
        y_score = np.append(y_score, dummy_score)
    return roc_auc_score(y_true=y_true, y_score=y_score)
# --- 辅助函数：滑窗数据收集 (保持不变) ---
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


# --- 主要解析和计算函数 (逻辑已修改) ---
def parse_and_calculate_aucs_from_file(file_path):
    """
    解析模型预测结果文件（本地路径），计算每个学生的汇集滑窗AUC、
    以及【所有学生所有滑窗序列拼接后】的整体AUC。
    """
    
    # 1. 读取和初步解析数据
    try:
        data = pd.read_csv(file_path, sep='\s+')
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {e}")
    # 在数据准备后添加检查
    print(f"late_trues 的唯一值: {data['late_trues'].unique()}")
    print(f"late_trues 的值计数:\n{data['late_trues'].value_counts()}")

    
    required_cols = ['orirow', 'late_trues', 'late_mean']
    if not all(col in data.columns for col in required_cols):
        data.columns = data.columns.str.strip()
        if not all(col in data.columns for col in required_cols):
             raise ValueError(f"TXT文件缺少必需的列: {required_cols}. 找到的列为: {list(data.columns)}")

    # 2. 数据准备
    data['late_trues'] = pd.to_numeric(data['late_trues'], errors='coerce')
    data['late_mean'] = pd.to_numeric(data['late_mean'], errors='coerce')
    data = data.dropna(subset=['late_trues', 'late_mean'])

    # 3. 按学生ID (orirow) 重组序列、滑窗并计算 AUC
    student_aucs = {}
    
    # *** 新增：存储所有学生的滑窗汇集数据，用于计算整体 AUC ***
    all_window_trues_flat = []
    all_window_scores_flat = []

    # 分组处理每个学生
    for student_uid, group in data.groupby('orirow'):
        student_trues_long = group['late_trues'].values
        student_scores_long = group['late_mean'].values

        # === 根据开关选择：滑窗 or 整序列 ===
        if ENABLE_SLIDING_WINDOW:
            collected_trues, collected_scores = sliding_window_collect(
                student_trues_long, 
                student_scores_long, 
                window_size=WINDOW_SIZE
            )
            # print(f"  [滑窗模式] 学生 {student_uid}: {len(student_trues_long)} → {len(collected_trues)} 个点")
        else:
            collected_trues = student_trues_long.tolist()
            collected_scores = student_scores_long.tolist()
            # print(f"  [整序列模式] 学生 {student_uid}: {len(collected_trues)} 个点")
        # ======================================

        # 加入整体汇集（用于 overall_auc）
        all_window_trues_flat.extend(collected_trues)
        all_window_scores_flat.extend(collected_scores)

        # 计算单个学生AUC
        if len(np.unique(collected_trues)) >= 2 and len(collected_trues) > 0:
            student_final_auc = safe_roc_auc(collected_trues, collected_scores)
        else:
            student_final_auc = 0.5

        student_aucs[student_uid] = student_final_auc

    # === 新增：自动保存 per_student_aucs.json 到 txt 同目录 ===
    mode_suffix = "" if ENABLE_SLIDING_WINDOW else "_no_window"
    json_output_path = file_path.replace('.txt', f'_per_student_aucs{mode_suffix}.json')
    # 转为普通 dict（避免 pandas Int64 问题）
    student_aucs_plain = {str(k): float(v) for k, v in student_aucs.items()}
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(student_aucs_plain, f, indent=2, ensure_ascii=False)
    print(f"已保存每个学生AUC → {json_output_path}")
    # ===========================================================


    # 4. 组装学生AUC DataFrame
    student_auc_df = pd.DataFrame(
        list(student_aucs.items()), 
        columns=['student_uid', 'auc']
    )
    
    # 5. 计算总体和统计结果
    
    # *** 关键修改：整体AUC现在基于所有学生的所有滑窗汇集数据 ***
    if len(np.unique(all_window_trues_flat)) >= 2 and len(all_window_trues_flat) > 0:
        overall_auc = safe_roc_auc(all_window_trues_flat, all_window_scores_flat)
    else:
        # 如果汇集后的数据仍然无法计算 AUC
        overall_auc = 0.5 

    # b. 学生 AUC 统计 (基于每个学生的滑窗汇集 AUC)
    valid_aucs = student_auc_df['auc'].dropna()
    
    if valid_aucs.empty:
        stats = {'mean': 0.5, 'std': 0.0, 'max': 0.5, 'min': 0.5, 'range': 0.0, 'iqr': 0.0}
    else:
        mean_auc = valid_aucs.mean()
        std_auc = valid_aucs.std() if len(valid_aucs) > 1 else 0.0
        max_auc = valid_aucs.max()
        min_auc = valid_aucs.min()
        range_auc = max_auc - min_auc
        denominator = std_auc * range_auc
        
        if denominator == 0:
            # 如果标准差或极差为0 (说明所有学生AUC完全一致)，避免除以零报错
            custom_ratio = 0.0 
            print("警告: 标准差或极差为0，自定义指标设为 0.0")
        else:
            custom_ratio = mean_auc / denominator
        stats = {
            'mean': mean_auc, 
            'std': std_auc, 
            'max': max_auc, 
            'min': min_auc, 
            'range': range_auc,
            'epi': custom_ratio
        }
        # === 新增：在返回前计算基尼系数和 EAWI ===
        valid_aucs_list = valid_aucs.tolist()  # 用于计算财富不平等指标

        average_wealth, gini, eawi_dict = calculate_wealth_metrics(valid_aucs_list, alphas=[1.0, 2.0, 3.0])

        # 更新 stats，加入新指标
        stats.update({
            'gini_coefficient': float(gini),
            'average_auc': float(average_wealth),  # 等价于 mean，但更清晰
            **eawi_dict  # 自动展开 eawi_alpha_10, eawi_alpha_20, eawi_alpha_30
        })
        # IQR: Q3(75%) - Q1(25%)
        q1_auc = np.percentile(valid_aucs_list, 25)
        q3_auc = np.percentile(valid_aucs_list, 75)
        stats.update({
            "iqr": float(q3_auc - q1_auc),
            "q1": float(q1_auc),
            "q3": float(q3_auc)
        })
    # 6. 整合结果字典
    overall_auc_info = {
        'overall_dataset_auc': overall_auc
    }

    # 遍历 stats，把 key 拼接到外层，或者直接合并
    for k, v in stats.items():
        # 方案 A: 加前缀 (推荐，防止命名冲突)
        overall_auc_info[f'student_stats_{k}'] = v 
        # 结果: student_stats_mean, student_stats_gini_coefficient
        
        # 方案 B: 直接合并 (如果确信 key 不会重复)
        # overall_auc_info[k] = v 
        # 结果: mean, gini_coefficient

    return student_auc_df, overall_auc_info

# --- 外层保存函数 (保持不变) ---
def save_auc_results_from_file(input_file_path, student_auc_output_csv=None, overall_stats_output_json=None):
    """
    解析本地TXT文件，并保存学生AUC DataFrame和整体统计结果字典。

    如果 student_auc_output_csv 或 overall_stats_output_json 为 None，
    它们将默认保存在与 input_file_path 相同的目录中，
    文件名分别为 'student_auc_output.csv' 和 'overall_stats_output.json'。
    """
    try:
        # 1. 获取输入文件的基本目录
        base_dir = os.path.dirname(input_file_path)

        # 2. 检查并设置默认的 CSV 输出路径
        if student_auc_output_csv is None:
            # 默认文件名基于原变量名的含义
            default_csv_name = "student_auc_output.csv"
            student_auc_output_csv = os.path.join(base_dir, default_csv_name)
        
        # 3. 检查并设置默认的 JSON 输出路径
        if overall_stats_output_json is None:
            # 默认文件名基于原变量名的含义
            default_json_name = "overall_stats_output.json"
            overall_stats_output_json = os.path.join(base_dir, default_json_name)

        # 4. 解析和计算 AUC (来自您的原始逻辑)
        student_df, overall_info = parse_and_calculate_aucs_from_file(input_file_path)
        
        # 5. 保存学生 AUC DataFrame (来自您的原始逻辑)
        student_df.to_csv(student_auc_output_csv, index=False)
        print(f"✅ 学生AUC结果已保存至: {student_auc_output_csv}")
        print(f"📊 总体统计指标: {json.dumps(overall_info, indent=4, ensure_ascii=False)}")
        # 6. 保存总体统计结果字典 (来自您的原始逻辑)
        with open(overall_stats_output_json, 'w', encoding='utf-8') as f:
            json.dump(overall_info, f, indent=4, ensure_ascii=False, 
                      default=lambda x: round(x, 6) if isinstance(x, (float, np.float_)) else x)
            
        print(f"✅ 总体统计结果已保存至: {overall_stats_output_json}")
        return overall_info
    except Exception as e:
        print(f"❌ 发生错误: {e}")
def main():
    """
    主执行函数，用于从命令行接收输入文件路径。
    """
    # sys.argv 是一个列表，包含所有命令行参数
    # sys.argv[0] 是脚本本身的名称 (例如 "process_auc.py")
    # sys.argv[1] 是第一个参数 (我们期望的 <input_file_path>)
    
    # 检查用户是否提供了正好一个参数（文件路径）
    if len(sys.argv) != 2:
        print("❌ 错误：参数数量不正确。")
        # 打印用法指南
        print(f"用法: python {sys.argv[0]} <input_file_path>")
        sys.exit(1) # 退出程序，并返回一个错误码

    # 获取文件路径
    input_file = sys.argv[1]

    # (推荐) 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误：文件未找到: {input_file}")
        sys.exit(1)

    print(f"--- 正在处理文件: {input_file} ---")
    
    # 调用您的核心函数
    # 由于我们只传入了 input_file，后两个参数将使用默认值 (None)
    # 这将触发您函数中的默认路径逻辑
    save_auc_results_from_file(input_file)
    
    print(f"--- 文件处理完成 ---")

# ---------------------------------------------------------------------------
# [执行入口] - 只有当脚本被直接运行时才调用 main()
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()