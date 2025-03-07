#!/bin/bash

# macOS上的简化测试脚本
# 仅使用TinyLlama和标准LoRA，避免所有量化相关问题

echo "=================================================="
echo "🍎 在macOS上进行简化LLM微调测试"
echo "=================================================="

# 设置参数
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
METHOD="lora"
OUTPUT_DIR="models/tinyllama_1.1b-instruction-${METHOD}/final"
TRAIN_FILE="data/alpaca_train.jsonl"

# 确保目录存在
mkdir -p models
mkdir -p data

# 检查数据文件
if [ ! -f "$TRAIN_FILE" ]; then
    echo "⚠️ 找不到训练数据文件: $TRAIN_FILE"
    exit 1
fi

# LoRA训练命令
echo "🔄 开始训练 TinyLlama 使用 LoRA 方法..."
python scripts/train_instruction.py --method lora --model_size tiny --batch_size 4 --epochs 1

# 检查训练结果
if [ -d "$OUTPUT_DIR" ]; then
    echo "✅ 训练完成，模型保存在: $OUTPUT_DIR"
else
    echo "❌ 训练可能未成功完成，找不到模型目录"
fi

echo "=================================================="
echo "测试完成"
echo "==================================================" 