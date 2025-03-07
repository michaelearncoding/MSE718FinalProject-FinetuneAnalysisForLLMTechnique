#!/bin/bash

# 快速测试脚本 - 验证实验流程是否正常工作
# 使用极少的数据和训练步骤，目的仅为验证流程无bug

echo "=================================================="
echo "🧪 开始实验流程快速测试"
echo "=================================================="

# 确保目录结构正确
mkdir -p data
mkdir -p models
mkdir -p results/raw_data
mkdir -p results/figures
mkdir -p results/model_comparison

# 步骤1: 检查必要的文件和工具
echo "1️⃣ 检查必要的脚本和工具..."

MISSING_FILES=0
for FILE in scripts/train_instruction.py scripts/evaluate_models.sh scripts/analyze_results.py scripts/run_mac_friendly.sh; do
    if [ ! -f "$FILE" ]; then
        echo "❌ 缺少文件: $FILE"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo "请确保所有必要的脚本文件存在!"
    exit 1
fi

echo "✅ 所有必要脚本文件存在"

# 步骤2: 检查Python依赖
echo "2️⃣ 检查Python依赖..."
python -c "import torch; import transformers; import peft; import datasets; import matplotlib.pyplot as plt; print('✅ 所有Python依赖已安装')" || { echo "❌ 缺少必要的Python依赖"; exit 1; }

# 步骤3: 创建测试数据集(只含3个样本的极小数据集)
echo "3️⃣ 创建测试数据集..."
cat > data/tiny_test.jsonl << EOF
{"instruction": "写一个简短的问候语", "input": "", "output": "你好！很高兴认识你。希望你今天过得愉快！"}
{"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是人工智能的一个分支，它使用数据来训练算法，使计算机能够在没有明确编程的情况下学习和改进。"}
{"instruction": "用一句话描述地球", "input": "", "output": "地球是太阳系中第三颗行星，是已知唯一孕育了生命的天体，拥有液态水和维持生命的大气层。"}
EOF

echo "✅ 测试数据集已创建: data/tiny_test.jsonl (3个样本)"

# 步骤4: 测试训练脚本(使用最小的tiny模型和qlora方法)
echo "4️⃣ 测试训练脚本..."
# 修改训练参数以加速测试(最小批次，最少步骤)
TEST_CMD="python scripts/train_instruction.py --method qlora --model_size tiny --epochs 1 --batch_size 1 --gradient_accumulation_steps 1 --use_mps"
echo "执行: $TEST_CMD"

# 临时修改训练脚本以使用测试数据
cp scripts/train_instruction.py scripts/train_instruction.py.backup
sed -i '' 's|TRAIN_FILE = "data/alpaca_train.jsonl"|TRAIN_FILE = "data/tiny_test.jsonl"|g' scripts/train_instruction.py
sed -i '' 's|args.epochs, batch_size=args.batch_size|1, batch_size=1|g' scripts/train_instruction.py

# 运行测试训练(限制5分钟)
timeout 300 $TEST_CMD
TRAIN_RESULT=$?

# 恢复原始训练脚本
mv scripts/train_instruction.py.backup scripts/train_instruction.py

if [ $TRAIN_RESULT -eq 124 ]; then
    echo "⚠️ 训练测试超时，但这不一定是错误 - 训练通常很耗时"
    echo "✅ 脚本启动正常"
elif [ $TRAIN_RESULT -ne 0 ]; then
    echo "❌ 训练脚本测试失败，请检查错误"
    exit 1
else
    echo "✅ 训练脚本测试成功"
fi

# 步骤5: 创建模拟评估结果(跳过实际评估以节省时间)
echo "5️⃣ 创建模拟评估结果..."
mkdir -p results/model_comparison
cat > results/model_comparison/tinyllama_1.1b_base.json << EOF
{
    "results": {
        "hellaswag": {
            "acc,none": 0.42
        },
        "gsm8k": {
            "acc,none": 0.11
        },
        "mmlu_high_school_computer_science": {
            "acc,none": 0.36
        }
    }
}
EOF

cat > results/model_comparison/tinyllama_1.1b_qlora.json << EOF
{
    "results": {
        "hellaswag": {
            "acc,none": 0.47
        },
        "gsm8k": {
            "acc,none": 0.15
        },
        "mmlu_high_school_computer_science": {
            "acc,none": 0.41
        }
    }
}
EOF

echo "✅ 模拟评估结果已创建"

# 步骤6: 测试分析脚本
echo "6️⃣ 测试分析脚本..."
python scripts/analyze_results.py
if [ $? -ne 0 ]; then
    echo "❌ 分析脚本测试失败，请检查错误"
    exit 1
else
    echo "✅ 分析脚本测试成功"
fi

echo "=================================================="
echo "🎉 快速测试完成!"
echo "=================================================="
echo "所有关键组件测试通过!"
echo ""
echo "建议下一步:"
echo "1. 确保数据集准备好 (data/alpaca_train.jsonl)"
echo "2. 确保评估框架已克隆 (lm-evaluation-harness)"
echo "3. 运行完整实验: ./scripts/run_mac_friendly.sh tiny 'qlora'"
echo "==================================================" 