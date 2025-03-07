#!/bin/bash

# 超级精简的测试脚本，使用极小数据集和极短训练时间

echo "=================================================="
echo "🧪 超简化版本的微调测试"
echo "=================================================="

# 创建测试数据
python scripts/create_test_data.py

# 修改训练脚本使用测试数据
TRAIN_SCRIPT="scripts/train_instruction.py"
cp "$TRAIN_SCRIPT" "${TRAIN_SCRIPT}.backup"
sed -i '' 's|TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "alpaca_train.jsonl")|TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "tiny_test.jsonl")|g' "$TRAIN_SCRIPT"

# 极简训练参数
echo "🔄 开始极简版微调..."
python $TRAIN_SCRIPT --method lora --model_size tiny --batch_size 2 --epochs 1 --gradient_accumulation_steps 1

# 恢复原始训练脚本
mv "${TRAIN_SCRIPT}.backup" "$TRAIN_SCRIPT"

echo "=================================================="
echo "测试完成"
echo "==================================================" 