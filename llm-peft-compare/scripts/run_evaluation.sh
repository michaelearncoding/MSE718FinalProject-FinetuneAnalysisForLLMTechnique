#!/bin/bash

# 专门用于评估模型性能的脚本

echo "=================================================="
echo "🧪 开始模型评估"
echo "=================================================="

# 参数解析
MODEL_SIZE=${1:-"tiny"}  # tiny, small, medium
METHOD=${2:-"lora"}      # lora, qlora
MERGED=${3:-"false"}     # true (评估合并模型), false (评估适配器)

# 设置设备
if [[ $(uname) == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    DEVICE="mps"
    echo "检测到 Apple Silicon Mac，使用 MPS 加速"
else
    DEVICE="cuda"
    echo "使用 CUDA"
fi

# 模型映射
if [ "$MODEL_SIZE" == "tiny" ]; then
    MODEL_ID="tinyllama_1.1b"
    BATCH_SIZE=8
elif [ "$MODEL_SIZE" == "small" ]; then
    MODEL_ID="phi_2.7b"
    BATCH_SIZE=4
elif [ "$MODEL_SIZE" == "medium" ]; then
    MODEL_ID="mistral_7b"
    BATCH_SIZE=2
else
    echo "错误: 未知的模型大小: $MODEL_SIZE"
    exit 1
fi

echo "评估模型: $MODEL_ID (微调方法: $METHOD)"

# 确保评估框架存在
EVAL_DIR="lm-evaluation-harness"
if [ ! -d "$EVAL_DIR" ]; then
    echo "克隆评估框架..."
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e .
    cd ..
fi

# 确保结果目录存在
mkdir -p results/model_comparison
mkdir -p results/raw_data
mkdir -p results/figures

# 设置评估任务
TASKS="hellaswag,gsm8k,mmlu_high_school_computer_science"
RESULTS_DIR="results/model_comparison"

# 确定模型路径
if [ "$MERGED" == "true" ]; then
    # 使用合并后的完整模型
    MODEL_PATH="models/${MODEL_ID}-instruction-${METHOD}-merged"
    MODEL_TYPE="${MODEL_ID}_${METHOD}_merged"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ 合并模型不存在: $MODEL_PATH"
        echo "请先运行: python scripts/save_merged_model.py --model_size $MODEL_SIZE --method $METHOD"
        exit 1
    fi
else
    # 使用微调适配器（评估脚本会自动加载基础模型和适配器）
    if [ -d "models/${MODEL_ID}-instruction-${METHOD}/final" ]; then
        MODEL_PATH="models/${MODEL_ID}-instruction-${METHOD}/final"
        MODEL_TYPE="${MODEL_ID}_${METHOD}"
    else
        echo "❌ 微调模型不存在: models/${MODEL_ID}-instruction-${METHOD}/final"
        echo "请先完成模型微调"
        exit 1
    fi
fi

echo "评估模型路径: $MODEL_PATH"

# 执行评估
echo "开始评估..."
cd $EVAL_DIR

# 构建评估命令
CMD="python main.py --model hf --model_args pretrained=$MODEL_PATH"

# 针对MacOS的MPS优化
if [ "$DEVICE" == "mps" ]; then
    # 在MPS上使用float16可能加速
    CMD="$CMD,dtype=float16"
    
    # 较大模型可能需要负载优化
    if [[ "$MODEL_SIZE" == "medium" ]]; then
        CMD="$CMD,load_in_4bit=True"
        BATCH_SIZE=1  # 减小批量大小
        echo "⚠️ 大型模型在MPS上使用4bit量化和批量大小=1"
    fi
fi

# 完成命令
CMD="$CMD --tasks $TASKS --device $DEVICE --batch_size $BATCH_SIZE --output_path ../$RESULTS_DIR/${MODEL_TYPE}"

echo "执行命令: $CMD"
eval $CMD

# 检查结果是否成功生成
RESULT_FILE="../$RESULTS_DIR/${MODEL_TYPE}.json"
if [ -f "$RESULT_FILE" ]; then
    echo "✓ 评估结果已保存: $RESULT_FILE"
    # 创建额外备份
    cp "$RESULT_FILE" "../results/raw_data/${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).json"
else
    echo "❌ 警告: 评估结果文件未生成: $RESULT_FILE"
fi

cd ..

# 运行分析脚本
echo ""
echo "📊 分析结果..."
python scripts/analyze_results.py

echo "=================================================="
echo "✅ 评估完成!"
echo "=================================================="
echo "结果保存在:"
echo "- 原始评估结果: $RESULTS_DIR/${MODEL_TYPE}.json"
echo "- 图表: results/figures/"
echo "==================================================" 