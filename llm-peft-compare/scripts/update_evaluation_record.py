#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from datetime import datetime

# 配置
RESULTS_DIR = "results/model_comparison"
RECORD_FILE = "models_evaluation_record.md"
MODEL_CONFIGS = {
    "tinyllama_1.1b": {
        "name": "TinyLlama (1.1B)",
        "size": "tiny",
        "section_level": 2
    },
    "phi_2.7b": {
        "name": "Phi-2 (2.7B)",
        "size": "small",
        "section_level": 2
    },
    "mistral_7b": {
        "name": "Mistral (7B)",
        "size": "medium",
        "section_level": 2
    }
}
METHOD_CONFIGS = {
    "base": "基础模型",
    "full": "完整微调",
    "lora": "LoRA",
    "qlora": "QLoRA"
}
TASK_CONFIGS = {
    "mmlu_high_school_computer_science": {
        "name": "MMLU高中计算机科学",
        "metrics": {
            "acc,none": "准确率"
        },
        "highlight": True
    },
    "hellaswag": {
        "name": "HellaSwag",
        "metrics": {
            "acc,none": "准确率",
            "acc_norm,none": "规范化准确率"
        },
        "highlight": False
    },
    "gsm8k": {
        "name": "GSM8K",
        "metrics": {
            "exact_match,flexible-extract": "灵活提取",
            "exact_match,strict-match": "严格匹配"
        },
        "highlight": False
    }
}

def parse_model_name(filename):
    """从结果文件名解析模型名称和方法"""
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 2:
        return None, None
    
    model_id = None
    method = None
    
    # 检查是否包含模型ID
    model_prefixes = list(MODEL_CONFIGS.keys())
    for prefix in model_prefixes:
        for i in range(len(parts) - 1):
            potential_id = "_".join(parts[i:i+2])
            if prefix in potential_id:
                model_id = prefix
                break
        if model_id:
            break
    
    # 检查是否包含方法
    for key in METHOD_CONFIGS.keys():
        if key in filename:
            method = key
            break
    
    return model_id, method

def get_result_value(results, task, metric, stderr=False):
    """从结果中提取特定任务和指标的值"""
    if task not in results:
        return None
    
    task_data = results[task]
    
    # 检查不同格式的指标
    if metric in task_data:
        return task_data[metric]
    
    # 检查是否使用alias作为前缀
    for key in task_data:
        if metric in key:
            if stderr and "stderr" in key:
                return task_data[key]
            elif not stderr and "stderr" not in key:
                return task_data[key]
    
    return None

def format_percentage(value):
    """将值格式化为百分比"""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"

def get_model_section(model_id, method):
    """生成模型章节的标题"""
    model_config = MODEL_CONFIGS.get(model_id)
    if model_config is None:
        model_config = {"name": model_id, "section_level": 2}
    method_name = METHOD_CONFIGS.get(method, method)
    
    section_level = "#" * model_config["section_level"]
    subsection_level = "#" * (model_config["section_level"] + 1)
    
    return f"{subsection_level} {model_config['name']} + {method_name}"

def generate_model_entry(model_id, method, data, date=None):
    """为模型生成完整的评估记录条目"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    method_name = METHOD_CONFIGS.get(method, method)
    section_header = get_model_section(model_id, method)
    
    # 提取结果
    results_table = "| 任务 | 指标 | 结果 | 误差范围 |\n|------|------|------|----------|\n"
    
    # 配置信息
    device = "mps" if "mps" in data.get("config", {}).get("device", "") else "cpu"
    batch_size = data.get("config", {}).get("batch_size", "未知")
    dtype = "float16" if "float16" in data.get("config", {}).get("model_args", "") else "float32"
    
    # 根据模型ID获取模型大小
    model_size = "unknown"
    if model_id in MODEL_CONFIGS:
        model_size = MODEL_CONFIGS[model_id].get("size", "unknown")
    
    # 提取每个任务的结果
    for task_id, task_config in TASK_CONFIGS.items():
        task_alias = task_id
        if "alias" in data.get("results", {}).get(task_id, {}):
            task_alias = data["results"][task_id]["alias"]
        
        for metric_key, metric_name in task_config["metrics"].items():
            value = get_result_value(data.get("results", {}), task_id, metric_key)
            stderr = get_result_value(data.get("results", {}), task_id, metric_key, stderr=True)
            
            formatted_value = format_percentage(value)
            formatted_stderr = f"±{format_percentage(stderr)[:-1]}" if stderr else "N/A"
            
            # 高亮特别重要的指标
            task_display = f"**{task_config['name']}**" if task_config["highlight"] else task_config["name"]
            value_display = f"**{formatted_value}**" if task_config["highlight"] else formatted_value
            
            results_table += f"| {task_display} | {metric_name} | {value_display} | {formatted_stderr} |\n"
    
    # 添加配置详情
    model_info = f"""
- **评估日期**: {date}
- **模型路径**: `/models/{model_id}-instruction-{method}-merged`
- **评估命令**: `./scripts/run_evaluation.sh {model_size} {method} true`
- **设备**: {device.upper()} ({'Apple Silicon' if device.upper() == 'MPS' else 'NVIDIA GPU'})
- **批量大小**: {batch_size}
- **数据类型**: {dtype}
"""
    
    # 合并所有部分
    return f"{section_header}\n\n{results_table}\n{model_info}\n"

def read_results_file(filepath):
    """读取结果文件"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 读取结果文件时出错: {filepath} - {e}")
        return None

def update_record_file(new_entries):
    """更新评估记录文件"""
    try:
        if not os.path.exists(RECORD_FILE):
            print(f"❌ 评估记录文件不存在: {RECORD_FILE}")
            return False
        
        with open(RECORD_FILE, 'r') as f:
            content = f.read()
        
        # 更新每个模型条目
        for model_id, method, entry in new_entries:
            section_header = get_model_section(model_id, method)
            
            # 获取模型名称
            model_name = model_id
            if model_id in MODEL_CONFIGS:
                model_name = MODEL_CONFIGS[model_id].get("name", model_id)
            
            # 获取方法名称
            method_name = method
            if method in METHOD_CONFIGS:
                method_name = METHOD_CONFIGS[method]
            
            # 尝试查找现有的模型章节
            pattern = re.compile(f"(#+\\s+{model_name}.*?\\n)([\\s\\S]*?)(?=#+\\s+|$)")
            model_section_match = pattern.search(content)
            
            if model_section_match:
                # 查找现有的方法子章节
                pattern = re.compile(f"(#+\\s+{model_name}\\s+\\+\\s+{method_name}.*?\\n)([\\s\\S]*?)(?=#+\\s+|$)")
                method_section_match = pattern.search(content)
                
                if method_section_match:
                    # 替换现有章节
                    old_section = method_section_match.group(0)
                    content = content.replace(old_section, entry)
                    print(f"✓ 已更新 {model_id} + {method} 的评估记录")
                else:
                    # 在现有模型章节下添加新的方法章节
                    model_section = model_section_match.group(0)
                    content = content.replace(model_section, model_section + "\n" + entry)
                    print(f"✓ 已添加 {model_id} + {method} 的评估记录")
            else:
                # 没有找到模型章节，添加到文件末尾
                content += f"\n## {model_name}\n\n{entry}"
                print(f"✓ 已添加 {model_id} 的新章节和 {method} 的评估记录")
        
        # 保存更新后的内容
        with open(RECORD_FILE, 'w') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"❌ 更新评估记录文件时出错: {e}")
        return False

def main():
    """主函数，处理所有结果文件并更新评估记录"""
    print("🔍 正在扫描结果文件...")
    
    # 确保目录存在
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ 结果目录不存在: {RESULTS_DIR}")
        return
    
    if not os.path.exists(RECORD_FILE):
        print(f"❌ 评估记录文件不存在: {RECORD_FILE}")
        return
    
    # 读取所有结果文件
    new_entries = []
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue
        
        # 解析模型名称和方法
        model_id, method = parse_model_name(filename)
        if not model_id or not method:
            print(f"⚠️ 无法解析文件名: {filename}")
            continue
        
        # 读取结果文件
        filepath = os.path.join(RESULTS_DIR, filename)
        data = read_results_file(filepath)
        if not data:
            continue
        
        # 生成模型条目
        entry = generate_model_entry(model_id, method, data)
        new_entries.append((model_id, method, entry))
        print(f"✓ 已处理 {model_id} + {method} 的评估结果")
    
    # 更新评估记录文件
    if new_entries:
        if update_record_file(new_entries):
            print("✅ 评估记录已成功更新!")
        else:
            print("❌ 更新评估记录失败")
    else:
        print("⚠️ 没有找到任何结果文件")

if __name__ == "__main__":
    main() 