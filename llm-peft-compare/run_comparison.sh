#!/bin/bash

# 脚本目录
SCRIPT_DIR="scripts"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "======================================================="
echo "      LLM 微调方法性能比较 - 横向与纵向分析工具         "
echo "======================================================="

# 创建所有必要目录
mkdir -p results/model_comparison
mkdir -p results/figures
mkdir -p results/raw_data
mkdir -p results/backups/${TIMESTAMP}

# 备份任何已有的结果
if [ -f "results/"*".csv" ] || [ -f "results/figures/"*".png" ]; then
    echo "备份现有结果到 results/backups/${TIMESTAMP}..."
    cp -r results/*.csv results/figures/*.png results/backups/${TIMESTAMP}/ 2>/dev/null || true
fi

echo "======================================================="
echo "步骤1: 评估所有模型"
echo "======================================================="
# 检查评估脚本是否存在
if [ -f "${SCRIPT_DIR}/evaluate_models.sh" ]; then
    bash ${SCRIPT_DIR}/evaluate_models.sh
    if [ $? -ne 0 ]; then
        echo "警告: 评估脚本可能未完全成功"
    fi
else
    echo "错误: 评估脚本 ${SCRIPT_DIR}/evaluate_models.sh 不存在!"
    exit 1
fi

echo
echo "======================================================="
echo "步骤2: 分析结果并生成可视化"
echo "======================================================="
# 检查分析脚本是否存在
if [ -f "${SCRIPT_DIR}/analyze_results.py" ]; then
    python ${SCRIPT_DIR}/analyze_results.py
    if [ $? -ne 0 ]; then
        echo "警告: 分析脚本可能未完全成功"
    fi
else
    echo "错误: 分析脚本 ${SCRIPT_DIR}/analyze_results.py 不存在!"
    exit 1
fi

# 验证文件确实被创建
echo
echo "======================================================="
echo "结果验证"
echo "======================================================="
CSV_COUNT=$(ls results/*.csv 2>/dev/null | wc -l)
PNG_COUNT=$(ls results/figures/*.png 2>/dev/null | wc -l)

echo "CSV结果文件: $CSV_COUNT 个"
echo "图表文件: $PNG_COUNT 个"

if [ $CSV_COUNT -eq 0 ] || [ $PNG_COUNT -eq 0 ]; then
    echo "警告: 可能某些文件未成功生成"
else
    echo "验证通过: 所有文件类型都已生成"
fi

echo
echo "======================================================="
echo "完成！"
echo "======================================================="
echo "结果摘要已保存到: results/[task]_comparison.csv"
echo "可视化图表已保存到: results/figures/ 目录"
echo "原始评估数据: results/model_comparison/ 目录"
echo "备份保存在: results/backups/${TIMESTAMP}/ 目录"
echo "======================================================="