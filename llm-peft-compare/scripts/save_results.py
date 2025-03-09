#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

# 确保目录存在
os.makedirs("results/model_comparison", exist_ok=True)
os.makedirs("results/raw_data", exist_ok=True)

# TinyLlama LoRA评估结果
tinyllama_lora_results = {
  "results": {
    "gsm8k": {
      "alias": "gsm8k",
      "exact_match,strict-match": 0.022,
      "exact_match_stderr,strict-match": 0.004,
      "exact_match,flexible-extract": 0.0281,
      "exact_match_stderr,flexible-extract": 0.0045
    },
    "hellaswag": {
      "alias": "hellaswag",
      "acc,none": 0.4584,
      "acc_stderr,none": 0.0050,
      "acc_norm,none": 0.5926,
      "acc_norm_stderr,none": 0.0049
    },
    "mmlu_high_school_computer_science": {
      "alias": "high_school_computer_science",
      "acc,none": 0.27,
      "acc_stderr,none": 0.0446
    }
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=/Users/qingdamai/Documents/SideHustle/MSE718FinalProject-MetaAnalysisForLLMTechnique/llm-peft-compare/models/tinyllama_1.1b-instruction-lora-merged,dtype=float16",
    "batch_size": 8,
    "device": "mps",
    "no_cache": False,
    "num_fewshot": None,
    "limit": None
  },
  "versions": {
    "gsm8k": 3,
    "hellaswag": 1,
    "mmlu_high_school_computer_science": 1
  }
}

# 保存TinyLlama LoRA合并模型的结果
output_file = "results/model_comparison/tinyllama_1.1b_lora_merged.json"
with open(output_file, "w") as f:
    json.dump(tinyllama_lora_results, f, indent=2)
    
print(f"✓ 已将TinyLlama LoRA评估结果保存到: {output_file}")

# 同时创建备份
backup_file = f"results/raw_data/tinyllama_1.1b_lora_merged_{os.path.basename(__file__).replace('.py', '')}.json"
with open(backup_file, "w") as f:
    json.dump(tinyllama_lora_results, f, indent=2)

print(f"✓ 备份保存到: {backup_file}") 