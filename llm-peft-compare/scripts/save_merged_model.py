#!/usr/bin/env python3
# 将LoRA适配器合并到基础模型，保存完整模型

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 参数设置
parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型并保存")
parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small", "medium"], 
                    help="模型大小: tiny (TinyLlama), small (Phi-2), medium (Mistral/LLaMA-2)")
parser.add_argument("--method", type=str, default="lora", choices=["lora", "qlora"],
                    help="微调方法: lora, qlora")
parser.add_argument("--output_dir", type=str, default=None,
                    help="输出目录，默认为models/[model_id]-instruction-[method]-merged")
args = parser.parse_args()

# 选择模型
model_map = {
    "tiny": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "id": "tinyllama_1.1b"},
    "small": {"name": "microsoft/phi-2", "id": "phi_2.7b"},
    "medium": {"name": "mistralai/Mistral-7B-v0.1", "id": "mistral_7b"}
}

selected_model = model_map[args.model_size]
MODEL_NAME = selected_model["name"]
MODEL_ID = selected_model["id"]

# 设置路径
BASE_MODEL_PATH = MODEL_NAME
ADAPTER_PATH = f"models/{MODEL_ID}-instruction-{args.method}/final"

if args.output_dir:
    OUTPUT_PATH = args.output_dir
else:
    OUTPUT_PATH = f"models/{MODEL_ID}-instruction-{args.method}-merged"

print(f"基础模型: {BASE_MODEL_PATH}")
print(f"适配器路径: {ADAPTER_PATH}")
print(f"输出路径: {OUTPUT_PATH}")

# 检查目录是否存在
if not os.path.exists(ADAPTER_PATH):
    raise ValueError(f"适配器路径不存在: {ADAPTER_PATH}, 请先完成微调")

# 加载模型
print("加载基础模型...")
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 在CPU上加载模型以便于合并
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="cpu",
    torch_dtype=torch.float32  # 使用float32以避免精度问题
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# 加载适配器
print("加载LoRA适配器...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 合并权重
print("合并LoRA权重到基础模型...")
model = model.merge_and_unload()

# 保存合并后的模型
print(f"保存合并后的模型到 {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"✅ 模型合并并保存成功！完整模型位于: {OUTPUT_PATH}")

# 简单测试（可选）
test_texts = [
    "写一个简短的问候语",
    "解释什么是机器学习"
]

print("\n模型测试:")
model = model.to(device)
model.eval()

for text in test_texts:
    print(f"\n输入: {text}")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输出: {response}") 