ai-image-detector
```python

import re
import pandas as pd

# 定义日志文件路径
log_file_path = 'path_to_your_log_file.log'

# 读取日志文件
with open(log_file_path, 'r') as file:
    log_data = file.read()

# 使用正则表达式提取相关信息
pattern = r"DDIM Sampler:\s+(\d+)%\|[^\|]+\| (\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+),\s+([\d\.]+)it/s\]�\[A"
matches = re.findall(pattern, log_data)

# 将提取的数据存储到一个列表中
data = []
for match in matches:
    percent, step, total, elapsed_time, remaining_time, rate = match
    data.append({
        "Progress": int(percent),
        "Step": int(step),
        "Total": int(total),
        "Elapsed Time": elapsed_time,
        "Remaining Time": remaining_time,
        "Rate": float(rate)
    })

# 转换为DataFrame
df = pd.DataFrame(data)

# 打印DataFrame
print(df)

# 保存DataFrame到CSV文件
df.to_csv('extracted_log_data.csv', index=False)