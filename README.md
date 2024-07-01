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

# 绘制损失变化图并保存
plt.figure(figsize=(10, 6))
plt.plot(df['Line'], df['Loss'], marker='o', label='Loss')
plt.plot(df['Line'], df['Loss Simple Step'], marker='x', linestyle='--', label='Loss Simple Step')
plt.plot(df['Line'], df['Loss VLB Step'], marker='s', linestyle='-.', label='Loss VLB Step')
plt.plot(df['Line'], df['Loss Step'], marker='^', linestyle=':', label='Loss Step')
plt.plot(df['Line'], df['Loss Simple Epoch'], marker='d', label='Loss Simple Epoch')
plt.plot(df['Line'], df['Loss VLB Epoch'], marker='*', label='Loss VLB Epoch')
plt.plot(df['Line'], df['Loss Epoch'], marker='p', label='Loss Epoch')
plt.xlabel('Line Number')
plt.ylabel('Loss')
plt.title('Loss Over Lines')
plt.legend()
plt.grid(True)
plt.savefig('loss_over_lines.png')
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