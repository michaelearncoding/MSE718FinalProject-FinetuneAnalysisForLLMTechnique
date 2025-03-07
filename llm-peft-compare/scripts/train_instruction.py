import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, choices=["full", "lora", "qlora"], required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--use_8bit_adam", action="store_true", help="记录用户希望使用8bit优化器(仅作为信息)")
parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium"], 
                    help="模型大小: tiny (1B), small (2-3B), medium (7B)")
args = parser.parse_args()

# 根据选择的模型大小设置模型
model_options = {
    "tiny": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "id": "tinyllama_1.1b",
        "max_length": 512
    },
    "small": {
        "name": "microsoft/phi-2",
        "id": "phi_2.7b",
        "max_length": 1024,
        "extra_args": "trust_remote_code=True"
    },
    "medium": {
        "name": "mistralai/Mistral-7B-v0.1",
        "id": "mistral_7b",
        "max_length": 2048
    }
}

selected_model = model_options[args.model_size]
MODEL_NAME = selected_model["name"]
MODEL_ID = selected_model["id"]
MAX_LENGTH = selected_model.get("max_length", 512)
EXTRA_ARGS = selected_model.get("extra_args", "")

# 配置
OUTPUT_DIR = f"models/{MODEL_ID}-instruction-{args.method}"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "alpaca_train.jsonl")
# TRAIN_FILE = "data/alpaca_train.jsonl"
print(f"数据文件路径: {TRAIN_FILE}")

print(f"选择模型: {MODEL_NAME} ({MODEL_ID})")

# 设备配置
device_map = "auto"
if torch.backends.mps.is_available():
    device_map = "mps"
    print("使用MPS加速进行训练")
elif torch.cuda.is_available():
    print("使用CUDA进行训练")
else:
    print("使用CPU进行训练")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 根据设备选择适当的torch数据类型
model_dtype = torch.float16
if device_map == "mps":
    # 在MPS上使用float32可能更稳定
    model_dtype = torch.float32
    print("在MPS设备上使用float32数据类型")

# 加载数据
def load_dataset_from_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # 创建简单的文本格式，避免过度复杂的结构
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item['output']
            
            # 构建最简单的训练文本
            if input_text:
                text = f"{instruction} {input_text} {output}"
            else:
                text = f"{instruction} {output}"
            
            data.append({'text': text})
    
    # 打印数据格式信息
    print(f"加载了 {len(data)} 条训练数据")
    if len(data) > 0:
        print(f"数据样例:\n{data[0]['text'][:200]}...")
    
    return Dataset.from_list(data)

train_dataset = load_dataset_from_jsonl(TRAIN_FILE)
print(f"加载了 {len(train_dataset)} 条训练数据")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 分词函数
def tokenize_function(examples):
    # 简单地对文本进行分词
    result = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length"
    )
    
    # 将input_ids复制为labels
    result["labels"] = result["input_ids"].copy()
    return result

# 分词化数据集
tokenized_datasets = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# 初始化模型和训练配置
if args.method == "full":
    print("执行完整微调...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        device_map=device_map
    )
elif args.method == "lora":
    print("执行LoRA微调...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        device_map=device_map
    )
    # LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
elif args.method == "qlora":
    print("执行QLoRA微调...")
    
    # 检查运行环境
    is_macos = torch.backends.mps.is_available()
    
    if is_macos:
        print("✅ 在Apple Silicon Mac上使用PyTorch原生量化")
        # 使用PyTorch原生量化替代bitsandbytes
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=model_dtype,
            device_map=device_map
        )
        
        # 应用PyTorch的动态量化（适用于MPS）
        try:
            from torch.quantization import quantize_dynamic
            # 量化模型的一部分
            print("应用PyTorch动态量化...")
            modules_to_quantize = ["q_proj", "k_proj", "v_proj", "o_proj"]
            for name, module in model.named_modules():
                if any(q_name in name for q_name in modules_to_quantize):
                    if isinstance(module, torch.nn.Linear):
                        print(f"量化模块: {name}")
                        module = quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as e:
            print(f"量化过程中出现警告 (非致命): {e}")
            print("继续使用标准精度...")
            
        # 使用LoRA配置
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        print("已应用LoRA适配器")
    else:
        try:
            # 标准QLoRA配置，适用于CUDA环境
            print("在CUDA环境中使用bitsandbytes...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_dtype
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map=device_map
            )
            model = prepare_model_for_kbit_training(model)
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
        except ImportError:
            print("⚠️ bitsandbytes不可用，使用标准LoRA")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=model_dtype,
                device_map=device_map
            )
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()

# 训练参数
# 在MPS设备上不使用fp16
use_fp16 = device_map != "mps"  # MPS设备不支持fp16混合精度

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    learning_rate=args.lr,
    fp16=use_fp16,  # 根据设备类型决定是否使用fp16
    optim="adamw_torch",
    remove_unused_columns=False,
)

# 打印精度信息
if device_map == "mps":
    print("⚠️ 在MPS设备上训练，已禁用fp16混合精度")
else:
    print("✓ 使用fp16混合精度训练")

# 打印关于优化器的信息
if args.use_8bit_adam:
    print("注意: 8bit-adam优化器在Trainer中不直接支持，使用标准优化器")

# 数据校对器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # 对齐到8的倍数，提高效率
    return_tensors="pt"
)

# 打印数据校对信息
print(f"使用DataCollatorForLanguageModeling，mlm=False, pad_to_multiple_of=8")
print(f"最大序列长度: {MAX_LENGTH}")

# 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
print(f"模型保存到 {os.path.join(OUTPUT_DIR, 'final')}")

def plot_model_size_impact():
    """分析模型规模对微调效果的影响"""
    
    # 根据模型名称提取模型系列和参数量
    model_info = {
        "llama2_7b": {"family": "LLaMA-2", "size": 7},
        "llama2_13b": {"family": "LLaMA-2", "size": 13},
        "phi_1.5b": {"family": "Phi", "size": 1.5},
        "phi_2.7b": {"family": "Phi", "size": 2.7},
        "tinyllama_1.1b": {"family": "TinyLlama", "size": 1.1},
        "pythia_1.4b": {"family": "Pythia", "size": 1.4},
        "pythia_2.8b": {"family": "Pythia", "size": 2.8}
    }
    
    # 按模型系列分组
    families = {}
    for model in models:
        if model in model_info:
            family = model_info[model]["family"]
            if family not in families:
                families[family] = []
            families[family].append(model)
    
    # 为每个模型系列创建图表
    for family, family_models in families.items():
        if len(family_models) < 2:
            continue  # 跳过只有一个模型的系列
            
        for task in task_map:
            plt.figure(figsize=(12, 8))
            
            # 提取数据
            sizes = [model_info[m]["size"] for m in family_models]
            for method in methods:
                perf_values = [results_data[task].loc[m, method] for m in family_models]
                plt.plot(sizes, perf_values, 'o-', label=method, linewidth=2, markersize=8)
            
            # 图表设置
            plt.title(f'{family}系列模型规模对{task_map[task]}任务的影响', fontsize=15)
            plt.xlabel('参数量 (B)', fontsize=12)
            plt.ylabel('准确率 (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="微调方法")
            
            plt.tight_layout()
            plt.savefig(f"results/figures/{family}_{task}_size_impact.png", dpi=300)
            plt.close()