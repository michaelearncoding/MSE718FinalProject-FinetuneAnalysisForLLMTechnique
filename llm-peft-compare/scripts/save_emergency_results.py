#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紧急结果保存脚本 - 从评估日志中提取结果并保存
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime

def extract_float(text, default=0.0):
    """从文本中提取浮点数"""
    if not text:
        return default
    match = re.search(r"([0-9]+\.[0-9]+)", text)
    if match:
        return float(match.group(1))
    return default

def extract_results_from_log(log_file_path):
    """从日志文件中提取评估结果"""
    if not os.path.exists(log_file_path):
        print(f"❌ 日志文件不存在: {log_file_path}")
        return None
    
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        # 提取表格部分
        table_match = re.search(r"\|(.*?)Tasks(.*?)\n(.*?)\n(.*?)$", log_content, re.MULTILINE | re.DOTALL)
        if not table_match:
            print("❌ 无法从日志中找到结果表格")
            return None
        
        table_text = table_match.group(0)
        
        # 提取任务结果
        results = {
            "hellaswag": {
                "alias": "hellaswag"
            },
            "gsm8k": {
                "alias": "gsm8k"
            },
            "mmlu_high_school_computer_science": {
                "alias": "high_school_computer_science"
            }
        }
        
        # 提取HellaSwag结果
        hellaswag_acc = re.search(r"\|hellaswag\s*\|.*?\|.*?\|.*?\|.*?\|(.*?)\|.*?\|(.*?)\|", table_text)
        hellaswag_norm = re.search(r"\|.*?acc_norm.*?\|(.*?)\|.*?\|(.*?)\|", table_text)
        
        if hellaswag_acc:
            results["hellaswag"]["acc,none"] = extract_float(hellaswag_acc.group(1))
            results["hellaswag"]["acc_stderr,none"] = extract_float(hellaswag_acc.group(2))
        
        if hellaswag_norm:
            results["hellaswag"]["acc_norm,none"] = extract_float(hellaswag_norm.group(1))
            results["hellaswag"]["acc_norm_stderr,none"] = extract_float(hellaswag_norm.group(2))
        
        # 提取GSM8K结果
        gsm8k_flexible = re.search(r"\|gsm8k\s*\|.*?\|flexible-extract\|.*?\|.*?\|(.*?)\|.*?\|(.*?)\|", table_text)
        gsm8k_strict = re.search(r"\|.*?strict-match\|.*?\|.*?\|(.*?)\|.*?\|(.*?)\|", table_text)
        
        if gsm8k_flexible:
            results["gsm8k"]["exact_match,flexible-extract"] = extract_float(gsm8k_flexible.group(1))
            results["gsm8k"]["exact_match_stderr,flexible-extract"] = extract_float(gsm8k_flexible.group(2))
        
        if gsm8k_strict:
            results["gsm8k"]["exact_match,strict-match"] = extract_float(gsm8k_strict.group(1))
            results["gsm8k"]["exact_match_stderr,strict-match"] = extract_float(gsm8k_strict.group(2))
        
        # 提取MMLU结果
        mmlu_match = re.search(r"\|high_school_computer_science\|.*?\|.*?\|.*?\|.*?\|(.*?)\|.*?\|(.*?)\|", table_text)
        
        if mmlu_match:
            results["mmlu_high_school_computer_science"]["acc,none"] = extract_float(mmlu_match.group(1))
            results["mmlu_high_school_computer_science"]["acc_stderr,none"] = extract_float(mmlu_match.group(2))
        
        # 提取模型配置信息
        model_line = re.search(r"hf \(pretrained=(.*?)\)", log_content)
        model_args = model_line.group(1) if model_line else ""
        
        # 提取设备信息
        device = "mps" if "mps" in log_content else "cuda" if "cuda" in log_content else "cpu"
        
        # 提取批量大小
        batch_size_match = re.search(r"batch_size: (\d+)", log_content)
        batch_size = int(batch_size_match.group(1)) if batch_size_match else 8
        
        return {
            "results": results,
            "config": {
                "model": "hf",
                "model_args": model_args,
                "batch_size": batch_size,
                "device": device,
                "no_cache": False,
                "num_fewshot": None,
                "limit": None
            },
            "versions": {
                "gsm8k": 3,
                "hellaswag": 1,
                "mmlu_high_school_computer_science": 1
            }
        }
        
    except Exception as e:
        print(f"❌ 提取结果时出错: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从评估日志中提取结果并保存为JSON')
    parser.add_argument('--log_file', required=True, help='评估日志文件路径')
    parser.add_argument('--output_file', required=True, help='输出结果文件路径')
    parser.add_argument('--create_backup', action='store_true', help='是否创建备份')
    
    args = parser.parse_args()
    
    # 提取结果
    results = extract_results_from_log(args.log_file)
    if not results:
        sys.exit(1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ 结果已保存到: {args.output_file}")
        
        # 创建备份
        if args.create_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{os.path.splitext(args.output_file)[0]}_backup_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ 备份已保存到: {backup_file}")
            
    except Exception as e:
        print(f"❌ 保存结果时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 