ai-image-detector
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('extracted_epoch_log_data.csv')

# 添加行号
df['Line'] = range(1, len(df) + 1)

# 绘制进度条图并保存
plt.figure(figsize=(10, 6))
plt.plot(df['Line'], df['Progress'], marker='o')
plt.xlabel('Line Number')
plt.ylabel('Progress (%)')
plt.title('Progress Over Lines')
plt.grid(True)
plt.savefig('progress_over_lines.png')
plt.close()

# 绘制每种损失类型的图表并保存
loss_types = [
    ('Loss', 'loss_over_lines.png'),
    ('Loss Simple Step', 'loss_simple_step_over_lines.png'),
    ('Loss VLB Step', 'loss_vlb_step_over_lines.png'),
    ('Loss Step', 'loss_step_over_lines.png'),
    ('Loss Simple Epoch', 'loss_simple_epoch_over_lines.png'),
    ('Loss VLB Epoch', 'loss_vlb_epoch_over_lines.png'),
    ('Loss Epoch', 'loss_epoch_over_lines.png')
]

for loss_type, filename in loss_types:
    plt.figure(figsize=(10, 6))
    plt.plot(df['Line'], df[loss_type], marker='o')
    plt.xlabel('Line Number')
    plt.ylabel(loss_type)
    plt.title(f'{loss_type} Over Lines')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# 绘制速率变化图并保存
plt.figure(figsize=(10, 6))
plt.plot(df['Line'], df['Rate'], marker='o')
plt.xlabel('Line Number')
plt.ylabel('Rate (s/it)')
plt.title('Iteration Rate Over Lines')
plt.grid(True)
plt.savefig('iteration_rate_over_lines.png')
plt.close()