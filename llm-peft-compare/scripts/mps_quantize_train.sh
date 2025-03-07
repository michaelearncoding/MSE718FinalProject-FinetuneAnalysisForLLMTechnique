#!/bin/bash

# 专为Apple Silicon Mac优化的LLM训练脚本
# 使用PyTorch MPS原生量化代替bitsandbytes

echo "=================================================="
echo "🍎 Apple Silicon Mac优化的LLM微调实验"
echo "=================================================="

# 确保所需目录存在
mkdir -p data
mkdir -p models
mkdir -p results/figures
mkdir -p results/model_comparison
mkdir -p results/raw_data

# 检查是否在Apple Silicon Mac上运行
if [[ $(uname) == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    echo "✅ 运行在Apple Silicon Mac上，将使用MPS加速和原生PyTorch量化"
else
    echo "⚠️ 未检测到Apple Silicon Mac，此脚本优化效果可能有限"
fi

# 参数解析
MODEL_SIZE=${1:-"tiny"}  # tiny, small, medium
RUN_EVALUATION=${2:-"true"}  # true/false

echo "使用模型大小: $MODEL_SIZE"
echo "是否进行评估: $RUN_EVALUATION"

# 检查数据集
if [ ! -f "data/alpaca_train.jsonl" ]; then
    echo "⚠️ 训练数据集不存在 (data/alpaca_train.jsonl)"
    exit 1
fi

# 设置批量大小和步数
BATCH_SIZE=4
GRAD_ACCUM=4

# 根据模型大小优化参数
if [[ "$MODEL_SIZE" == "medium" ]]; then
    BATCH_SIZE=1
    GRAD_ACCUM=8
elif [[ "$MODEL_SIZE" == "small" ]]; then
    BATCH_SIZE=2
    GRAD_ACCUM=4
fi

# 运行训练
echo ""
echo "🔄 开始使用PyTorch MPS量化训练 $MODEL_SIZE 模型..."

# 执行训练命令
CMD="python scripts/train_instruction.py --method qlora --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --model_size $MODEL_SIZE"
echo "执行命令: $CMD"
eval $CMD

# 检查训练结果
if [ $? -ne 0 ]; then
    echo "❌ 训练失败，请检查错误信息"
    exit 1
else
    echo "✅ 训练完成"
fi

# 运行评估
if [ "$RUN_EVALUATION" == "true" ]; then
    echo ""
    echo "🧪 开始评估模型..."
    bash scripts/evaluate_models.sh mps $MODEL_SIZE
    
    # 运行分析
    echo ""
    echo "📊 分析结果..."
    python scripts/analyze_results.py
else
    echo ""
    echo "⏩ 跳过评估阶段"
fi

echo ""
echo "=================================================="
echo "✅ Apple Silicon优化实验完成!"
echo "=================================================="
echo "结果保存在:"
echo "- 模型: models/[model_id]-instruction-qlora/final"
echo "- 评估结果: results/model_comparison/[model_id]_qlora.json"
echo "- 图表: results/figures/"
echo "==================================================" 