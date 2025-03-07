#!/bin/bash

# 针对macOS优化的LLM微调和评估实验
# 自动使用MPS加速和适当的内存优化参数

# 确保所需目录存在
mkdir -p data
mkdir -p models
mkdir -p results

# 检查是否在Apple Silicon Mac上运行
if [[ $(uname) == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    echo "✓ 在Apple Silicon Mac上运行，将使用MPS加速"
    USE_MPS="--use_mps"
    DEVICE="mps"
else
    echo "⚠️ 未检测到Apple Silicon Mac，将使用标准配置"
    USE_MPS=""
    DEVICE="cuda"
fi

# 参数解析
MODEL_SIZE=${1:-"small"}  # tiny, small, medium
METHODS=${2:-"lora,qlora"}  # full,lora,qlora (逗号分隔)
RUN_EVALUATION=${3:-"true"}  # true/false

echo "========================================================"
echo "🚀 开始macOS友好的LLM微调和评估实验"
echo "========================================================"
echo "模型大小: $MODEL_SIZE"
echo "微调方法: $METHODS"
echo "是否进行评估: $RUN_EVALUATION"
echo "========================================================"

# 检查数据文件
if [ ! -f "data/alpaca_train.jsonl" ]; then
    echo "⚠️ 训练数据文件不存在，请确保data/alpaca_train.jsonl文件已准备好"
    exit 1
fi

# 解析要运行的方法
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"

# 运行训练
for METHOD in "${METHOD_ARRAY[@]}"; do
    echo ""
    echo "🔄 正在使用 $METHOD 方法训练 $MODEL_SIZE 模型..."
    
    # 跳过大模型的完整微调
    if [[ "$MODEL_SIZE" == "medium" && "$METHOD" == "full" ]]; then
        echo "⚠️ 跳过7B模型的完整微调 (在macOS上可能导致内存不足)"
        continue
    fi
    
    # 设置批量大小和梯度累积步数
    BATCH_SIZE=4
    GRAD_ACCUM=4
    
    # 针对不同模型大小优化参数
    if [[ "$MODEL_SIZE" == "medium" ]]; then
        BATCH_SIZE=1
        GRAD_ACCUM=8
    elif [[ "$MODEL_SIZE" == "small" ]]; then
        BATCH_SIZE=2
        GRAD_ACCUM=4
    fi
    
    # 运行训练脚本
    CMD="python scripts/train_instruction.py --method $METHOD --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --model_size $MODEL_SIZE $USE_MPS --use_8bit_adam"
    echo "执行命令: $CMD"
    eval $CMD
    
    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        echo "❌ 训练失败，请检查错误信息"
    else
        echo "✓ 训练完成"
    fi
done

# 运行评估
if [ "$RUN_EVALUATION" == "true" ]; then
    echo ""
    echo "🧪 开始模型评估..."
    
    # 运行评估脚本
    bash scripts/evaluate_models.sh $DEVICE $MODEL_SIZE
    
    # 运行分析脚本
    echo ""
    echo "📊 分析结果..."
    python scripts/analyze_results.py
else
    echo ""
    echo "⏩ 跳过评估阶段"
fi

echo ""
echo "========================================================"
echo "✅ 实验运行完成!"
echo "========================================================"
echo "结果保存在:"
echo "- 模型: models/[model_id]-instruction-[method]/final"
echo "- 评估结果: results/model_comparison/[model_id]_[method].json"
echo "- 图表: results/figures/"
echo "========================================================" 