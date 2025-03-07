python -c "
import json

# 读取原始JSON文件（包含数组）
with open('data/alpaca_train.jsonl', 'r') as f:
    data = json.load(f)

# 备份原文件
import shutil
shutil.copy('data/alpaca_train.jsonl', 'data/alpaca_data.json.backup')

# 写入JSONL格式（每行一个对象）
with open('data/alpaca_train.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

print(f'✅ 已成功转换为JSONL格式，包含 {len(data)} 条记录')
"