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
            
            bars = plt.bar(x + offset, data[method], width, label=method_label, color=colors[i % len(colors)])
            
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
        
        try:
            output_path = f"results/figures/{method}_vertical_comparison.png"
            plt.savefig(output_path, dpi=300)
            print(f"✓ 图表已保存: {output_path}")
        except Exception as e:
            print(f"❌ 图表保存失败: {output_path} - 错误: {e}")

# 3. 微调前后性能变化
def plot_improvement_heatmap():
    # 检查是否存在基础模型数据
    if "base" not in methods:
        print("警告: 没有基础模型的评估数据，无法创建性能提升热力图")
        return
        
    for task in task_map:
        # 检查是否有基础模型数据
        base_data_available = True
        for model in models:
            if np.isnan(results_data[task].loc[model, "base"]) or results_data[task].loc[model, "base"] == 0:
                base_data_available = False
                break
                
        if not base_data_available:
            print(f"警告: 缺少基础模型数据，无法为{task_map[task]}创建性能提升热力图")
            continue
            
        # 计算每个模型微调方法相对于基线的改善
        improvement_data = pd.DataFrame(index=models, columns=methods[1:])  # 排除base
        
        for model in models:
            base_score = results_data[task].loc[model, "base"]
            for method in methods[1:]:
                if method in results_data[task].columns:
                    tuned_score = results_data[task].loc[model, method]
                    # 确保数值有效并转换为浮点数
                    try:
                        base_score = float(base_score)
                        tuned_score = float(tuned_score)
                        # 计算相对改善百分比
                        if base_score > 0:
                            improvement = ((tuned_score - base_score) / base_score) * 100
                        else:
                            improvement = float('nan')
                        improvement_data.loc[model, method] = improvement
                    except (ValueError, TypeError):
                        improvement_data.loc[model, method] = float('nan')
        
        # 重命名列以便显示
        improvement_data.columns = [
            "完整微调", "LoRA", "QLoRA"
        ]
        improvement_data.index = [name.capitalize() for name in improvement_data.index]
        
        # 转换数据类型
        improvement_data = improvement_data.astype(float)
        
        # 绘制热力图
        plt.figure(figsize=(10, 6))
        
        # 创建自定义颜色映射 - 红色为负值，绿色为正值
        cmap = LinearSegmentedColormap.from_list('RdYlGn', ['#d62728', '#f7f7f7', '#2ca02c'])
        
        ax = sns.heatmap(improvement_data, annot=True, fmt=".1f", cmap=cmap, center=0,
                        cbar_kws={'label': '相对改善 (%)'}, linewidths=0.5)
        
        # 设置标题和标签
        plt.title(f'{task_map[task]}任务上不同微调方法的性能提升', fontsize=15)
        plt.ylabel('模型', fontsize=12)
        plt.xlabel('微调方法', fontsize=12)
        
        plt.tight_layout()
        
        try:
            output_path = f"results/figures/{task}_improvement_heatmap.png"
            plt.savefig(output_path, dpi=300)
            print(f"✓ 图表已保存: {output_path}")
        except Exception as e:
            print(f"❌ 图表保存失败: {output_path} - 错误: {e}")

# 4. 效率比较
def plot_efficiency_comparison():
    # 检查是否有足够的数据进行比较
    method_count = 0
    for method in methods:
        if any(not results_data[task].loc[models[0], method] == 0 for task in task_map):
            method_count += 1
    
    if method_count < 2:
        print("警告: 没有足够的微调方法数据来创建效率比较图")
        print("注意: 此图需要至少两种不同的微调方法数据才能进行比较")
        return
    
    # 创建示例效率数据 (实际使用时应替换为真实测量值)
    plt.figure(figsize=(12, 6))
    
    # 添加明显的标记，指明这是示例数据
    plt.figtext(0.5, 0.01, '注意: 此图使用的是示例数据，不代表实际测量结果', 
                ha='center', color='red', fontsize=12, 
                bbox=dict(facecolor='yellow', alpha=0.2))
    
    # 示例数据 - 在实际使用中替换为真实数据
    methods_display = ["基础模型", "完整微调", "LoRA", "QLoRA"]
    memory_usage = [16, 16, 5, 4]  # GB内存使用
    training_time = [0, 100, 40, 30]  # 相对训练时间 (%)
    inference_speed = [100, 100, 95, 90]  # 相对推理速度 (%)
    
    # 数据不足时使用部分数据
    if method_count < 4:
        methods_display = methods_display[:method_count]
        memory_usage = memory_usage[:method_count]
        training_time = training_time[:method_count]
        inference_speed = inference_speed[:method_count]

    # 绘制效率雷达图
    # 设置雷达图的属性
    categories = ['训练时间', '内存占用', '推理速度']
    N = len(categories)
    
    # 设置雷达图
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    
    # 转换数据为雷达图格式
    values = []
    for method in methods_display:
        method_data = pd.DataFrame({
            'category': categories,
            'value': [training_time[methods_display.index(method)],
                       memory_usage[methods_display.index(method)],
                       100 - memory_usage[methods_display.index(method)]]
        })
        values.append(method_data['value'].values)
    
    # 设置角度
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 设置网格
    plt.xticks(angles[:-1], categories)
    
    # 绘制每种方法
    for i, method in enumerate(methods_display):
        method_values = values[i]
        method_values += method_values[:1]  # 闭合图形
        ax.plot(angles, method_values, linewidth=2, label=method)
        ax.fill(angles, method_values, alpha=0.1)
    
    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('不同微调方法的效率对比 (示例)', fontsize=15)
    
    try:
        output_path = f"results/figures/efficiency_comparison.png"
        plt.savefig(output_path, dpi=300)
        print(f"✓ 图表已保存: {output_path}")
    except Exception as e:
        print(f"❌ 图表保存失败: {output_path} - 错误: {e}")

# 脚本开始时调用
ensure_directories()

# 设置中文字体支持 (macOS)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 结果目录
results_dir = "results/model_comparison"

# 定义模型和方法列表
# 定义理想的模型和方法列表
all_models = ["tinyllama", "phi2", "gemma2b"]
all_methods = ["base", "full", "lora", "qlora"]

# 实际存在的模型和方法列表（将根据文件动态调整）
models = []
methods = []

task_map = {
    "hellaswag": "HellaSwag (常识推理)",
    "gsm8k": "GSM8K (数学推理)",
    "mmlu_high_school_computer_science": "MMLU-CS (计算机科学知识)"
}

print("开始分析评估结果...")

# 确保结果目录存在
os.makedirs("results/figures", exist_ok=True)

# 查找可用的模型和方法
model_map = {
    "tinyllama": "tinyllama",
    "phi": "phi2",
    "phi2": "phi2",
    "gemma": "gemma2b",
    "mistral": "mistral"
}

for filename in os.listdir(results_dir):
    if not filename.endswith(".json"):
        continue
    
    print(f"发现结果文件: {filename}")
    
    # 解析文件名以获取模型和方法
    for model_key in model_map:
        if model_key in filename.lower():
            model_std = model_map[model_key]
            if model_std not in models:
                models.append(model_std)
    
    for method in all_methods:
        if method in filename.lower():
            if method not in methods:
                methods.append(method)

if not models:
    models = ["tinyllama"]  # 默认至少有一个模型
    print("警告: 未检测到任何模型，使用默认模型'tinyllama'")
    
if not methods:
    methods = ["lora"]  # 默认至少有一种方法
    print("警告: 未检测到任何方法，使用默认方法'lora'")
    
# 确保base方法总是在第一位
if "base" in methods:
    methods.remove("base")
    methods = ["base"] + methods

print(f"检测到的模型: {models}")
print(f"检测到的方法: {methods}")

# 创建数据结构来存储结果
results_data = {}
for task in task_map:
    results_data[task] = pd.DataFrame(index=models, columns=methods)

# 加载所有结果文件
found_results = False
for filename in os.listdir(results_dir):
    if not filename.endswith(".json"):
        continue
    
    print(f"发现结果文件: {filename}")
    
    # 解析文件名以获取模型和方法
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 2:
        continue
    
    # 处理形如 tinyllama_1.1b_lora_merged.json 的文件名
    model_name = parts[0]
    # 处理模型名称中可能的版本号
    if len(parts) > 1 and parts[1].replace(".", "").isdigit():
        model_name = parts[0]  # 只使用基本名称，如tinyllama
    
    # 提取方法名
    method = None
    for m in methods:
        if m in filename:
            method = m
            break
    
    # 如果没有找到匹配的方法，跳过
    if method is None:
        # 检查基础模型
        if "base" in filename:
            method = "base"
        else:
            print(f"警告: 无法从 {filename} 确定方法类型")
            continue
    
    # 模型名称规范化映射
    model_map = {
        "tinyllama": "tinyllama",
        "phi": "phi2",
        "phi2": "phi2",
        "gemma": "gemma2b",
        "mistral": "mistral"
    }
    
    # 尝试映射模型名称
    mapped_model = None
    for key, value in model_map.items():
        if key in model_name:
            mapped_model = value
            break
    
    # 使用映射的模型名或原始名
    if mapped_model:
        model_name = mapped_model
    
    print(f"解析结果: 模型={model_name}, 方法={method}")
    
    if model_name not in models:
        print(f"警告: 模型 {model_name} 不在预定义列表中，将被跳过")
        continue
    
    if method not in methods:
        print(f"警告: 方法 {method} 不在预定义列表中，将被跳过")
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

print("\n正在生成水平比较图表...")
try:
    plot_horizontal_comparison()
except Exception as e:
    print(f"生成水平比较图表时出错: {e}")

print("\n正在生成垂直比较图表...")
try:
    plot_vertical_comparison()
except Exception as e:
    print(f"生成垂直比较图表时出错: {e}")

print("\n正在生成性能提升热力图...")
try:
    plot_improvement_heatmap()
except Exception as e:
    print(f"生成性能提升热力图时出错: {e}")

print("\n正在生成效率比较图表...")
try:
    plot_efficiency_comparison()
except Exception as e:
    print(f"生成效率比较图表时出错: {e}")

print("\n分析完成！所有数据和图表已保存在 results 目录")