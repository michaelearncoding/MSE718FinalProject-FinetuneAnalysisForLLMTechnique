#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# 在import部分之后添加
import os

# 创建所有必要的目录
def ensure_directories():
    dirs = [
        "results",
        "results/figures",
        "results/raw_data",
        "results/model_comparison"
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"确保目录存在: {directory}")

# 脚本开始时调用
ensure_directories()

# 设置中文字体支持 (macOS)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 结果目录
results_dir = "results/model_comparison"

# 定义模型和方法列表
models = ["tinyllama", "phi2", "gemma2b"]
methods = ["base", "full", "lora", "qlora"]
task_map = {
    "hellaswag": "HellaSwag (常识推理)",
    "gsm8k": "GSM8K (数学推理)",
    "mmlu_high_school_computer_science": "MMLU-CS (计算机科学知识)"
}

print("开始分析评估结果...")

# 确保结果目录存在
os.makedirs("results/figures", exist_ok=True)

# 创建数据结构来存储结果
results_data = {}
for task in task_map:
    results_data[task] = pd.DataFrame(index=models, columns=methods)

# 加载所有结果文件
found_results = False
for filename in os.listdir(results_dir):
    if not filename.endswith(".json"):
        continue
    
    # 解析文件名以获取模型和方法
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 2:
        continue
        
    model_name = parts[0]
    method = parts[1]
    
    if model_name not in models or method not in methods:
        continue
    
    # 加载结果文件
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            found_results = True
            
            if "results" not in data:
                print(f"警告: 文件 {filename} 中没有results字段")
                continue
                
            # 提取每个任务的得分
            for task_full in data["results"]:
                task = task_full
                # 标准化MMLU任务名称
                if "mmlu_" in task_full:
                    task = "mmlu_high_school_computer_science"
                
                if task not in task_map:
                    continue
                    
                metrics = data["results"][task_full]
                # 使用主要指标（优先使用acc, 然后是exact_match等）
                for metric in ["acc", "exact_match", "f1"]:
                    if metric in metrics:
                        score = metrics[metric] * 100  # 转换为百分比
                        results_data[task].loc[model_name, method] = score
                        break
                        
        except Exception as e:
            print(f"加载 {filename} 时出错: {e}")

if not found_results:
    print("警告: 未找到任何结果文件。请先运行评估脚本。")
    exit()

# 填充缺失值
for task in task_map:
    results_data[task] = results_data[task].fillna(0)

# 保存结果为CSV
# 保存结果为CSV (改进版)
for task in task_map:
    csv_path = f"results/{task}_comparison.csv"
    try:
        results_data[task].to_csv(csv_path)
        print(f"✓ 成功保存表格数据: {csv_path}")
    except Exception as e:
        print(f"❌ 保存表格数据失败: {csv_path} - 错误: {e}")

print("\n生成可视化图表...")

# 1. 横向比较：每个模型不同微调方法的性能
def plot_horizontal_comparison():
    for task in task_map:
        plt.figure(figsize=(12, 6))
        
        data = results_data[task]
        x = np.arange(len(models))
        width = 0.2
        
        # 定义方法颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 绘制每种方法的柱状图
        for i, method in enumerate(methods):
            offset = width * (i - len(methods)/2 + 0.5)
            method_label = {
                "base": "基础模型",
                "full": "完整微调",
                "lora": "LoRA",
                "qlora": "QLoRA"
            }.get(method, method.capitalize())
            
            bars = plt.bar(x + offset, data[method], width, label=method_label, color=colors[i])
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=0,
                            fontsize=8)
        
        # 图表样式
        plt.title(f'不同微调方法在{task_map[task]}上的性能比较', fontsize=15)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.xlabel('模型', fontsize=12)
        plt.xticks(x, [name.capitalize() for name in models])
        plt.legend(title="微调方法")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, max(100, data.values.max() * 1.1))
        
        plt.tight_layout()

        # plt.savefig(f"results/figures/{task}_horizontal_comparison.png", dpi=300)
        # plt.close()
        # 在plot_horizontal_comparison等函数中的plt.savefig()后添加确认
        try:
            output_path = f"results/figures/{task}_horizontal_comparison.png"
            plt.savefig(output_path, dpi=300)
            print(f"✓ 图表已保存: {output_path}")
        except Exception as e:
            print(f"❌ 图表保存失败: {output_path} - 错误: {e}")

# 2. 纵向比较：不同模型在相同微调方法下的性能
def plot_vertical_comparison():
    for method in methods:
        plt.figure(figsize=(12, 6))
        
        # 准备数据
        task_scores = []
        for task in task_map:
            task_scores.append(results_data[task][method])
        
        data = pd.DataFrame(task_scores, index=task_map.values(), columns=models)
        
        # 转置数据以便按模型分组
        data = data.T
        
        # 绘制分组柱状图
        data.plot(kind='bar', ax=plt.gca())
        
        # 添加数值标签
        for container in plt.gca().containers:
            plt.bar_label(container, fmt='%.1f', fontsize=8)
        
        # 图表样式
        method_label = {
            "base": "基础模型",
            "full": "完整微调",
            "lora": "LoRA",
            "qlora": "QLoRA"
        }.get(method, method.capitalize())
        
        plt.title(f'{method_label}下不同模型的性能比较', fontsize=15)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.xlabel('模型', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, max(100, data.values.max() * 1.1))
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"results/figures/{method}_vertical_comparison.png", dpi=300)
        plt.close()

# 3. 微调前后性能变化
def plot_improvement_heatmap():
    for task in task_map:
        # 计算每个模型微调方法相对于基线的改善
        improvement_data = pd.DataFrame(index=models, columns=methods[1:])  # 排除base
        
        for model in models:
            base_score = results_data[task].loc[model, "base"]
            for method in methods[1:]:
                if method in results_data[task].columns:
                    tuned_score = results_data[task].loc[model, method]
                    # 计算相对改善百分比
                    if base_score > 0:
                        improvement = ((tuned_score - base_score) / base_score) * 100
                    else:
                        improvement = float('nan')
                    improvement_data.loc[model, method] = improvement
        
        # 重命名列以便显示
        improvement_data.columns = [
            "完整微调", "LoRA", "QLoRA"
        ]
        improvement_data.index = [name.capitalize() for name in improvement_data.index]
        
        # 绘制热力图
        plt.figure(figsize=(10, 6))
        
        # 创建自定义颜色映射 - 红色为负值，绿色为正值
        cmap = LinearSegmentedColormap.from_list('RdYlGn', ['#d62728', '#f7f7f7', '#2ca02c'])
        
        # 绘制热力图
        ax = sns.heatmap(improvement_data, annot=True, fmt=".1f", cmap=cmap, center=0,
                   linewidths=.5, cbar_kws={"label": "相对改善 (%)"})
        
        # 设置底部标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        plt.title(f'{task_map[task]}任务上不同微调方法的性能改善', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"results/figures/{task}_improvement_heatmap.png", dpi=300)
        plt.close()

# 4. 微调方法效率对比
def plot_efficiency_comparison():
    # 创建示例效率数据 (实际使用时应替换为真实测量值)
    efficiency_data = pd.DataFrame({
        'method': ["完整微调", "LoRA", "QLoRA"],
        'train_time': [100, 30, 25],  # 相对训练时间 (分钟)
        'memory_usage': [100, 40, 20],  # 内存使用 (%)
        'parameter_count': [100, 1, 0.5]  # 可训练参数比例 (%)
    })
    
    # 绘制效率雷达图
    categories = ['训练时间', '内存使用', '参数数量']
    N = len(categories)
    
    # 转换数据为雷达图格式
    values = []
    for method in efficiency_data['method']:
        method_data = efficiency_data[efficiency_data['method'] == method]
        values.append([
            method_data['train_time'].values[0],
            method_data['memory_usage'].values[0],
            method_data['parameter_count'].values[0]
        ])
    
    # 设置角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 初始化图表
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 绘制每种方法
    for i, method in enumerate(efficiency_data['method']):
        method_values = values[i]
        method_values += method_values[:1]  # 闭合图形
        ax.plot(angles, method_values, linewidth=2, label=method)
        ax.fill(angles, method_values, alpha=0.1)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # 标题和图例
    plt.title('微调方法效率对比 (数值越低越好)', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('results/figures/efficiency_comparison.png', dpi=300)
    plt.close()

# 执行所有可视化
plot_horizontal_comparison()
plot_vertical_comparison()
plot_improvement_heatmap()
# plot_efficiency_comparison()  # 需要实际数据才能使用

print("分析完成！所有数据和图表已保存在 results 目录")