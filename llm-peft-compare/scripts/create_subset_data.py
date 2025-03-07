#!/usr/bin/env python3
# 创建Alpaca数据集的子集

import json
import os
import random
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="创建数据集子集")
parser.add_argument("--input", type=str, default="data/alpaca_train.jsonl", help="输入数据文件路径")
parser.add_argument("--output", type=str, default="data/alpaca_train_5k.jsonl", help="输出数据文件路径")
parser.add_argument("--size", type=int, default=5000, help="子集大小")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# 设置随机种子
random.seed(args.seed)

# 读取原始数据
data = []
try:
    with open(args.input, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的JSON行")
                continue
except FileNotFoundError:
    print(f"错误: 找不到输入文件 {args.input}")
    exit(1)

print(f"读取了 {len(data)} 条数据")

# 创建子集
if len(data) <= args.size:
    subset = data
    print(f"原始数据集大小 ({len(data)}) 小于或等于请求的子集大小 ({args.size})，使用全部数据")
else:
    subset = random.sample(data, args.size)
    print(f"随机选择了 {args.size} 条数据")

# 保存子集
with open(args.output, 'w') as f:
    for item in subset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"子集已保存到 {args.output}") 