#!/bin/bash

# 确保所需目录存在
mkdir -p results

echo "=== 步骤1: 评估所有模型 ==="
bash evaluate_models.sh

echo "=== 步骤2: 分析结果并生成可视化 ==="
python analyze_results.py

echo "=== 完成！ ==="
echo "所有结果和可视化已保存在results目录"