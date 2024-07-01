ai-image-detector
```python

import re
import pandas as pd

# 定义日志文件路径
log_file_path = 'path_to_your_log_file.log'

# 读取日志文件
with open(log_file_path, 'r') as file:
    log_data = file.readlines()

# 使用正则表达式提取Epoch相关信息
pattern = r"Epoch \d+:.*"

# 筛选出相关行并提取信息
matches = [line for line in log_data if re.match(pattern, line)]

# 将提取的数据存储到一个列表中
data = []
epoch_pattern = r"Epoch (\d+):\s+(\d+)%\|[^\|]+\| (\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+),\s+([^\s]+)]"
for match in matches:
    epoch_match = re.match(epoch_pattern, match)
    if epoch_match:
        epoch, percent, step, total, elapsed_time, remaining_time, info = epoch_match.groups()
        data.append({
            "Epoch": int(epoch),
            "Progress": int(percent),
            "Step": int(step),
            "Total": int(total),
            "Elapsed Time": elapsed_time,
            "Remaining Time": remaining_time,
            "Info": info
        })

# 转换为DataFrame
df = pd.DataFrame(data)

# 打印DataFrame
print(df)

# 保存DataFrame到CSV文件
Epoch 9:  38%|███▊      | 655/1706 [11:28<18:24,  1.05s/it, loss=0.0572, v_num=es_2, train/loss_simple_step=0.0442, train/loss_vlb_step=0.000295, train/loss_step=0.0442, global_step=1.6e+4, train/loss_simple_epoch=0.0677, train/loss_vlb_epoch=0.0019, train/loss_epoch=0.0677]
