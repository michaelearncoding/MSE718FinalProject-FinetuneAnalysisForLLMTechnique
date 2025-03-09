#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA结果来源分析脚本
此脚本用于分析QLoRA评估结果的来源，并检查是否有实际运行QLoRA评估的记录
"""

import os
import json
import datetime
import sys

def print_header(text):
    """打印带格式的标题"""
    print("\n" + "=" * 60)
    print(f" {text} ".center(60, "="))
    print("=" * 60)

def print_section(text):
    """打印带格式的小节标题"""
    print("\n" + "-" * 50)
    print(f" {text} ".center(50, "-"))
    print("-" * 50)

def check_file_exists(filepath):
    """检查文件是否存在并打印结果"""
    exists = os.path.exists(filepath)
    status = "✅ 存在" if exists else "❌ 不存在"
    print(f"{status}: {filepath}")
    return exists

def format_timestamp(timestamp):
    """格式化时间戳为可读格式"""
    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "无法解析"

def get_file_info(filepath):
    """获取文件的详细信息"""
    if not os.path.exists(filepath):
        return None
    
    stat_info = os.stat(filepath)
    return {
        "size": f"{stat_info.st_size / 1024:.2f} KB",
        "created": format_timestamp(stat_info.st_ctime),
        "modified": format_timestamp(stat_info.st_mtime),
        "accessed": format_timestamp(stat_info.st_atime)
    }

def print_file_info(filepath):
    """打印文件的详细信息"""
    info = get_file_info(filepath)
    if not info:
        print(f"❌ 文件不存在: {filepath}")
        return
    
    print(f"📄 文件: {filepath}")
    print(f"   大小: {info['size']}")
    print(f"   创建时间: {info['created']}")
    print(f"   修改时间: {info['modified']}")
    print(f"   访问时间: {info['accessed']}")

def compare_json_files(file1, file2):
    """比较两个JSON文件的内容差异"""
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"❌ 无法比较文件，至少有一个文件不存在")
        return
    
    try:
        with open(file1, 'r') as f:
            data1 = json.load(f)
        with open(file2, 'r') as f:
            data2 = json.load(f)
        
        # 比较模型路径
        path1 = data1.get("config", {}).get("model_args", "")
        path2 = data2.get("config", {}).get("model_args", "")
        
        print(f"模型1路径: {path1}")
        print(f"模型2路径: {path2}")
        
        # 比较结果
        print("\n结果比较:")
        
        # 比较MMLU结果
        mmlu1 = data1.get("results", {}).get("mmlu_high_school_computer_science", {})
        mmlu2 = data2.get("results", {}).get("mmlu_high_school_computer_science", {})
        
        acc1 = mmlu1.get("acc,none", mmlu1.get("acc", 0))
        acc2 = mmlu2.get("acc,none", mmlu2.get("acc", 0))
        
        print(f"MMLU准确率: {acc1*100:.2f}% vs {acc2*100:.2f}%")
        
        # 比较HellaSwag结果
        hs1 = data1.get("results", {}).get("hellaswag", {})
        hs2 = data2.get("results", {}).get("hellaswag", {})
        
        acc1 = hs1.get("acc,none", hs1.get("acc", 0))
        acc2 = hs2.get("acc,none", hs2.get("acc", 0))
        
        print(f"HellaSwag准确率: {acc1*100:.2f}% vs {acc2*100:.2f}%")
        
        # 比较GSM8K结果
        gsm1 = data1.get("results", {}).get("gsm8k", {})
        gsm2 = data2.get("results", {}).get("gsm8k", {})
        
        # 处理不同格式的GSM8K结果
        if "exact_match,flexible-extract" in gsm1:
            acc1 = gsm1.get("exact_match,flexible-extract", 0)
        else:
            acc1 = gsm1.get("flexible-extract", {}).get("exact_match", 0)
            
        if "exact_match,flexible-extract" in gsm2:
            acc2 = gsm2.get("exact_match,flexible-extract", 0)
        else:
            acc2 = gsm2.get("flexible-extract", {}).get("exact_match", 0)
        
        print(f"GSM8K准确率: {acc1*100:.2f}% vs {acc2*100:.2f}%")
        
    except Exception as e:
        print(f"❌ 比较文件时出错: {e}")

def check_evaluation_logs():
    """检查是否有QLoRA评估的日志记录"""
    print_section("检查评估日志")
    
    # 检查日志目录
    log_dir = "logs"
    if os.path.exists(log_dir):
        print(f"✅ 日志目录存在: {log_dir}")
        # 查找包含qlora的日志文件
        qlora_logs = []
        for filename in os.listdir(log_dir):
            if "qlora" in filename.lower() and filename.endswith(".log"):
                qlora_logs.append(os.path.join(log_dir, filename))
        
        if qlora_logs:
            print(f"✅ 找到{len(qlora_logs)}个QLoRA相关日志文件:")
            for log in qlora_logs:
                print_file_info(log)
        else:
            print("❌ 未找到QLoRA相关日志文件")
    else:
        print(f"❌ 日志目录不存在: {log_dir}")
    
    # 检查analyze_output.log
    analyze_log = "analyze_output.log"
    if os.path.exists(analyze_log):
        print(f"\n✅ 分析日志存在: {analyze_log}")
        # 查找包含qlora的行
        try:
            with open(analyze_log, 'r') as f:
                content = f.read()
                qlora_lines = [line for line in content.split('\n') if "qlora" in line.lower()]
                
                if qlora_lines:
                    print(f"✅ 在分析日志中找到{len(qlora_lines)}行包含QLoRA的记录:")
                    for line in qlora_lines:
                        print(f"   {line}")
                else:
                    print("❌ 在分析日志中未找到QLoRA相关记录")
        except Exception as e:
            print(f"❌ 读取分析日志时出错: {e}")
    else:
        print(f"❌ 分析日志不存在: {analyze_log}")

def check_quick_test_script():
    """检查快速测试脚本是否创建了模拟QLoRA结果"""
    print_section("检查快速测试脚本")
    
    quick_test_script = "scripts/quick_test.sh"
    if os.path.exists(quick_test_script):
        print(f"✅ 快速测试脚本存在: {quick_test_script}")
        # 查找创建QLoRA结果的代码
        try:
            with open(quick_test_script, 'r') as f:
                content = f.read()
                if "tinyllama_1.1b_qlora" in content:
                    print("✅ 快速测试脚本中包含创建QLoRA结果的代码")
                    
                    # 提取相关代码段
                    import re
                    qlora_code = re.search(r'cat > results/model_comparison/tinyllama_1.1b_qlora\.json << EOF(.*?)EOF', 
                                          content, re.DOTALL)
                    if qlora_code:
                        print("\n相关代码段:")
                        print(qlora_code.group(0)[:500] + "..." if len(qlora_code.group(0)) > 500 else qlora_code.group(0))
                else:
                    print("❌ 快速测试脚本中未找到创建QLoRA结果的代码")
        except Exception as e:
            print(f"❌ 读取快速测试脚本时出错: {e}")
    else:
        print(f"❌ 快速测试脚本不存在: {quick_test_script}")

def main():
    """主函数"""
    print_header("QLoRA评估结果来源分析")
    
    # 检查QLoRA结果文件
    print_section("检查QLoRA结果文件")
    qlora_result_file = "results/model_comparison/tinyllama_1.1b_qlora_merged.json"
    lora_result_file = "results/model_comparison/tinyllama_1.1b_lora_merged.json"
    
    qlora_exists = check_file_exists(qlora_result_file)
    lora_exists = check_file_exists(lora_result_file)
    
    if qlora_exists:
        print("\nQLoRA结果文件详情:")
        print_file_info(qlora_result_file)
    
    if lora_exists:
        print("\nLoRA结果文件详情:")
        print_file_info(lora_result_file)
    
    # 检查QLoRA模型
    print_section("检查QLoRA模型")
    qlora_model_dir = "models/tinyllama_1.1b-instruction-qlora"
    qlora_merged_model_dir = "models/tinyllama_1.1b-instruction-qlora-merged"
    
    check_file_exists(qlora_model_dir)
    check_file_exists(qlora_merged_model_dir)
    
    if os.path.exists(qlora_model_dir):
        print("\nQLoRA模型目录详情:")
        print_file_info(qlora_model_dir)
        
        # 检查final子目录
        final_dir = os.path.join(qlora_model_dir, "final")
        if os.path.exists(final_dir):
            print(f"\n✅ QLoRA模型final目录存在: {final_dir}")
            print_file_info(final_dir)
        else:
            print(f"❌ QLoRA模型final目录不存在: {final_dir}")
    
    if os.path.exists(qlora_merged_model_dir):
        print("\nQLoRA合并模型目录详情:")
        print_file_info(qlora_merged_model_dir)
    
    # 比较LoRA和QLoRA结果
    if qlora_exists and lora_exists:
        print_section("比较LoRA和QLoRA结果")
        compare_json_files(lora_result_file, qlora_result_file)
    
    # 检查评估日志
    check_evaluation_logs()
    
    # 检查快速测试脚本
    check_quick_test_script()
    
    # 结论
    print_section("分析结论")
    
    if os.path.exists(qlora_model_dir) and os.path.exists(qlora_merged_model_dir):
        print("✅ QLoRA模型确实存在，表明QLoRA训练已执行")
    else:
        print("❌ QLoRA模型不完整或不存在，可能未执行QLoRA训练")
    
    if qlora_exists:
        qlora_info = get_file_info(qlora_result_file)
        lora_info = get_file_info(lora_result_file) if lora_exists else None
        
        if qlora_info and lora_info:
            qlora_time = datetime.datetime.strptime(qlora_info["created"], "%Y-%m-%d %H:%M:%S")
            lora_time = datetime.datetime.strptime(lora_info["created"], "%Y-%m-%d %H:%M:%S")
            
            if abs((qlora_time - lora_time).total_seconds()) < 600:  # 10分钟内
                print("⚠️ QLoRA和LoRA结果文件创建时间接近，可能是同时生成的")
            else:
                print("✅ QLoRA和LoRA结果文件创建时间相差较大，可能是分别评估生成的")
    
    # 最终结论
    print("\n最终结论:")
    if os.path.exists(qlora_model_dir) and os.path.exists(qlora_merged_model_dir) and qlora_exists:
        print("QLoRA模型已训练并合并，评估结果文件存在。")
        print("虽然没有找到明确的QLoRA评估日志，但模型和结果文件的存在表明QLoRA评估可能已执行。")
        print("建议检查命令历史或与执行评估的人员确认。")
    else:
        print("QLoRA评估结果可能是通过脚本自动生成的，而非实际运行评估得到的。")
        print("建议重新运行QLoRA评估以获取真实结果。")

if __name__ == "__main__":
    main() 