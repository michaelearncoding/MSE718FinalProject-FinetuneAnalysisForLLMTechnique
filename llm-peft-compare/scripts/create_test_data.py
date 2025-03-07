#!/usr/bin/env python3
# 创建一个小的测试数据集

import json
import os

# 创建数据目录
os.makedirs("data", exist_ok=True)

# 测试数据
test_data = [
    {
        "instruction": "写一个简短的问候语",
        "input": "",
        "output": "你好！很高兴认识你。希望你今天过得愉快！"
    },
    {
        "instruction": "解释什么是机器学习",
        "input": "",
        "output": "机器学习是人工智能的一个分支，它使用数据来训练算法，使计算机能够在没有明确编程的情况下学习和改进。"
    },
    {
        "instruction": "用一句话描述地球",
        "input": "",
        "output": "地球是太阳系中第三颗行星，是已知唯一孕育了生命的天体，拥有液态水和维持生命的大气层。"
    },
    {
        "instruction": "计算 5 乘以 7 的结果",
        "input": "",
        "output": "5 乘以 7 等于 35。"
    },
    {
        "instruction": "翻译这个句子",
        "input": "The weather is nice today.",
        "output": "今天天气很好。"
    }
]

# 写入测试数据
with open("data/tiny_test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"创建了测试数据集：data/tiny_test.jsonl (包含 {len(test_data)} 条数据)") 