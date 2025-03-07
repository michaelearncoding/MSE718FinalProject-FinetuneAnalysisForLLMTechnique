#!/bin/bash

# 设置路径和评估任务
EVAL_DIR="lm-evaluation-harness"
TASKS="hellaswag,gsm8k,mmlu_high_school_computer_science"
RESULTS_DIR="results/model_comparison"

# 确保所需目录存在
mkdir -p results
mkdir -p results/model_comparison
mkdir -p results/figures
mkdir -p results/raw_data
mkdir -p models  # 确保微调模型目录存在

# 检查评估框架是否存在
if [ ! -d "$EVAL_DIR" ]; then
    echo "错误: 评估框架目录 $EVAL_DIR 不存在!"
    echo "请先克隆 lm-evaluation-harness 仓库"
    exit 1
fi

# 根据系统检测默认设备
if [[ $(uname) == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    DEFAULT_DEVICE="mps"
    echo "检测到 Apple Silicon Mac，默认使用 MPS 加速"
else
    DEFAULT_DEVICE="cuda"
    echo "未检测到 Apple Silicon Mac，使用 CUDA"
fi

# 参数解析
USE_DEVICE=${1:-$DEFAULT_DEVICE}
MODEL_SIZE_CLASS=${2:-"small"}  # small, medium, all

echo "使用设备: $USE_DEVICE"
echo "模型大小类别: $MODEL_SIZE_CLASS"

# 基础模型配置 - 格式: "模型路径|模型简称|批量大小|额外参数"
declare -a SMALL_MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0|tinyllama_1.1b|8" 
  "microsoft/phi-1_5|phi_1.5b|8|trust_remote_code=True"
  "microsoft/phi-2|phi_2.7b|4|trust_remote_code=True"
  "EleutherAI/pythia-1.4b|pythia_1.4b|8"
  "EleutherAI/pythia-2.8b|pythia_2.8b|4"
)

declare -a MEDIUM_MODELS=(
  "meta-llama/Llama-2-7b-hf|llama2_7b|2"
  "mistralai/Mistral-7B-v0.1|mistral_7b|2"
)

# 根据用户选择设置模型集
if [ "$MODEL_SIZE_CLASS" == "small" ]; then
    BASE_MODELS=("${SMALL_MODELS[@]}")
    echo "使用小型模型集 (1-3B参数)"
elif [ "$MODEL_SIZE_CLASS" == "medium" ]; then
    BASE_MODELS=("${MEDIUM_MODELS[@]}")
    echo "使用中型模型集 (7B参数) - 注意: 这些模型在MacOS上仅推荐使用QLora方法"
elif [ "$MODEL_SIZE_CLASS" == "all" ]; then
    BASE_MODELS=("${SMALL_MODELS[@]}" "${MEDIUM_MODELS[@]}")
    echo "使用所有模型集 (可能需要较长时间)"
else
    echo "错误: 未知的模型大小类别: $MODEL_SIZE_CLASS"
    exit 1
fi

# 微调方法
declare -a METHODS=("full" "lora" "qlora")

# 评估函数
run_evaluation() {
  local model_path=$1
  local model_name=$2
  local method=$3
  local batch_size=$4
  local extra_args=$5
  
  echo "正在评估: $model_name ($method 方法)"
  
  # 构建评估命令
  cd $EVAL_DIR
  CMD="python main.py --model hf --model_args pretrained=$model_path"
  
  # 添加额外参数（如果有）
  if [ ! -z "$extra_args" ]; then
    CMD="$CMD,$extra_args"
  fi
  
  # 针对MacOS的MPS优化
  if [ "$USE_DEVICE" == "mps" ]; then
    # 在MPS上使用float16可能加速
    CMD="$CMD,dtype=float16"
    
    # 较大模型可能需要负载优化
    if [[ "$model_name" == *"7b"* ]]; then
      CMD="$CMD,load_in_4bit=True"
      batch_size=1  # 减小批量大小
      echo "⚠️ 大型模型 ($model_name) 在MPS上使用4bit量化和批量大小=1"
    fi
  fi
  
  # 完成命令
  CMD="$CMD --tasks $TASKS --device $USE_DEVICE --batch_size $batch_size --output_path ../$RESULTS_DIR/${model_name}_${method}"
  
  echo "执行命令: $CMD"
  eval $CMD
  
  # 检查结果是否成功生成
  RESULT_FILE="../$RESULTS_DIR/${model_name}_${method}.json"
  if [ -f "$RESULT_FILE" ]; then
    echo "✓ 评估结果已保存: $RESULT_FILE"
    # 创建额外备份
    cp "$RESULT_FILE" "../results/raw_data/${model_name}_${method}_$(date +%Y%m%d_%H%M%S).json"
  else
    echo "❌ 警告: 评估结果文件未生成: $RESULT_FILE"
  fi
  
  cd ..
  
  echo "评估完成: $model_name ($method)"
  echo "----------------"
}

echo "==============================================="
echo "开始评估基础模型和微调模型"
echo "==============================================="

# 第1步: 评估所有基础模型
echo "--- 评估基础模型 ---"
for MODEL_INFO in "${BASE_MODELS[@]}"; do
  IFS="|" read -r MODEL_PATH MODEL_NAME BATCH_SIZE EXTRA_ARGS <<< "$MODEL_INFO"
  run_evaluation "$MODEL_PATH" "$MODEL_NAME" "base" "$BATCH_SIZE" "$EXTRA_ARGS"
done

# 第2步: 评估所有微调模型
echo "--- 评估微调模型 ---"
for MODEL_INFO in "${BASE_MODELS[@]}"; do
  IFS="|" read -r BASE_PATH MODEL_NAME BATCH_SIZE EXTRA_ARGS <<< "$MODEL_INFO"
  
  # 对每种微调方法评估
  for METHOD in "${METHODS[@]}"; do
    # 跳过中型模型的完整微调（在MacOS上无法运行）
    if [[ "$MODEL_NAME" == *"7b"* ]] && [[ "$METHOD" == "full" ]]; then
      echo "跳过 $MODEL_NAME 的完整微调评估 (模型太大)"
      continue
    fi
    
    # 构建微调模型路径
    FINETUNED_PATH="models/${MODEL_NAME}-instruction-${METHOD}/final"
    
    # 检查模型目录是否存在
    if [ -d "$FINETUNED_PATH" ]; then
      run_evaluation "$FINETUNED_PATH" "$MODEL_NAME" "$METHOD" "$BATCH_SIZE" "$EXTRA_ARGS"
    else
      echo "模型路径不存在，跳过: $FINETUNED_PATH"
    fi
  done
done

echo "==============================================="
echo "所有评估完成！结果保存在 $RESULTS_DIR"
echo "==============================================="