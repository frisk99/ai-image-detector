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
epoch_pattern = (
    r"Epoch (\d+):\s+(\d+)%\|[^\|]+\|\s+(\d+)/(\d+)\s+\[(\d+:\d+)<(\d+:\d+),\s+([^\s]+)s/it,\s+"
    r"loss=([^\s,]+),\s+v_num=([^\s,]+),\s+train/loss_simple_step=([^\s,]+),\s+"
    r"train/loss_vlb_step=([^\s,]+),\s+train/loss_step=([^\s,]+),\s+global_step=([^\s,]+),\s+"
    r"train/loss_simple_epoch=([^\s,]+),\s+train/loss_vlb_epoch=([^\s,]+),\s+train/loss_epoch=([^\s,]+)]"
)

# 将提取的数据存储到一个列表中
data = []
for line in log_data:
    epoch_match = re.match(epoch_pattern, line)
    if epoch_match:
        groups = epoch_match.groups()
        data.append({
            "Epoch": int(groups[0]),
            "Progress": int(groups[1]),
            "Step": int(groups[2]),
            "Total": int(groups[3]),
            "Elapsed Time": groups[4],
            "Remaining Time": groups[5],
            "Rate": groups[6],
            "Loss": float(groups[7]),
            "Version": groups[8],
            "Loss Simple Step": float(groups[9]),
            "Loss VLB Step": float(groups[10]),
            "Loss Step": float(groups[11]),
            "Global Step": groups[12],
            "Loss Simple Epoch": float(groups[13]),
            "Loss VLB Epoch": float(groups[14]),
            "Loss Epoch": float(groups[15]),
        })

# 转换为DataFrame
df = pd.DataFrame(data)

# 打印DataFrame
print(df)

# 保存DataFrame到CSV文件
df.to_csv('extracted_epoch_log_data.csv', index=False)