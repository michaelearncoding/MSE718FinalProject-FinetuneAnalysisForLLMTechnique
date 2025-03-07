import json
import os
from datasets import load_dataset

# 创建数据目录
os.makedirs("data", exist_ok=True)

# 加载Alpaca数据集 - 自我指令数据
dataset = load_dataset("tatsu-lab/alpaca")
print(f"加载了 {len(dataset['train'])} 条训练数据")

# 保存为简单格式以便处理
train_data = []
for item in dataset["train"]:
    # 构建指令格式
    prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input'] if item['input'] else 'N/A'}\n\n### Response:\n"
    response = item["output"]
    
    train_data.append({
        "prompt": prompt,
        "response": response
    })

# 保存为jsonl文件
with open("data/alpaca_train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

print(f"已保存 {len(train_data)} 条训练数据到 data/alpaca_train.jsonl")

# 用于测试的小数据集
with open("data/alpaca_test.jsonl", "w") as f:
    for item in train_data[:100]:  # 只取前100条
        f.write(json.dumps(item) + "\n")

print("数据准备完成！")
