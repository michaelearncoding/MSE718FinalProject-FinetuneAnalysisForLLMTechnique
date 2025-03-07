#!/bin/bash

# 优化版训练脚本 - 使用数据子集和减少训练时间

echo "=================================================="
echo "🚀 开始优化版LLM微调训练（使用5K数据子集）"
echo "=================================================="

# 参数解析
MODEL_SIZE=${1:-"tiny"}  # tiny, small, medium
METHOD=${2:-"qlora"}     # lora, qlora
EPOCHS=${3:-"1"}         # 训练轮数，默认1轮

# 检查是否在Apple Silicon Mac上运行
if [[ $(uname) == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    echo "✓ 在Apple Silicon Mac上运行，将使用MPS加速"
else
    echo "⚠️ 未检测到Apple Silicon Mac，此脚本优化效果可能有限"
fi

# 确保目录存在
mkdir -p data
mkdir -p models
mkdir -p results/figures
mkdir -p results/model_comparison
mkdir -p results/raw_data

# 准备数据子集
if [ ! -f "data/alpaca_train_5k.jsonl" ]; then
    echo "创建5K数据子集..."
    python scripts/create_subset_data.py --input data/alpaca_train.jsonl --output data/alpaca_train_5k.jsonl --size 5000
fi

# 模型映射
if [ "$MODEL_SIZE" == "tiny" ]; then
    MODEL_ID="tinyllama_1.1b"
    # TinyLlama对于小数据集可以使用较大的批量大小
    BATCH_SIZE=16
    GRAD_ACCUM=2
elif [ "$MODEL_SIZE" == "small" ]; then
    MODEL_ID="phi_2.7b"
    BATCH_SIZE=8
    GRAD_ACCUM=2
elif [ "$MODEL_SIZE" == "medium" ]; then
    MODEL_ID="mistral_7b"
    BATCH_SIZE=2
    GRAD_ACCUM=4
else
    echo "错误: 未知的模型大小: $MODEL_SIZE"
    exit 1
fi

# 不同方法的其他优化参数
if [ "$METHOD" == "qlora" ]; then
    # QLoRA可能需要更多累积步骤但可以用更小批量
    GRAD_ACCUM=$((GRAD_ACCUM * 2))
    BATCH_SIZE=$((BATCH_SIZE / 2))
fi

echo "使用配置:"
echo "- 模型: $MODEL_ID"
echo "- 方法: $METHOD"
echo "- 批量大小: $BATCH_SIZE"
echo "- 梯度累积: $GRAD_ACCUM"
echo "- 训练轮数: $EPOCHS"

# 临时修改训练脚本以使用数据子集
TRAIN_SCRIPT="scripts/train_instruction.py"
cp "$TRAIN_SCRIPT" "${TRAIN_SCRIPT}.backup"
sed -i '' 's|TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "alpaca_train.jsonl")|TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "alpaca_train_5k.jsonl")|g' "$TRAIN_SCRIPT"

# 运行训练
echo ""
echo "🔄 开始训练 $MODEL_ID 使用 $METHOD 方法..."

# 执行训练命令
CMD="python $TRAIN_SCRIPT --method $METHOD --model_size $MODEL_SIZE --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --epochs $EPOCHS"
echo "执行命令: $CMD"
eval $CMD

# 恢复原始训练脚本
mv "${TRAIN_SCRIPT}.backup" "$TRAIN_SCRIPT"

# 检查训练结果
MODEL_DIR="models/${MODEL_ID}-instruction-${METHOD}/final"
if [ -d "$MODEL_DIR" ]; then
    echo "✅ 训练完成，模型保存在: $MODEL_DIR"
    
    # 询问是否进行评估
    read -p "是否立即评估模型? (y/n) " RUN_EVAL
    if [[ $RUN_EVAL == "y" || $RUN_EVAL == "Y" ]]; then
        echo "开始评估模型..."
        ./scripts/run_evaluation.sh $MODEL_SIZE $METHOD false
    else
        echo "跳过评估。之后可以使用以下命令评估模型:"
        echo "  ./scripts/run_evaluation.sh $MODEL_SIZE $METHOD false"
    fi
else
    echo "❌ 训练可能未成功完成，找不到模型目录"
fi

echo ""
echo "=================================================="
echo "优化版训练完成!"
echo "=================================================="
echo "如需合并模型，请运行:"
echo "  python scripts/save_merged_model.py --model_size $MODEL_SIZE --method $METHOD"
echo "==================================================" 