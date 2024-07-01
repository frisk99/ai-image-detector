ai-image-detector
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('extracted_epoch_log_data.csv')

# 绘制进度条图
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Progress'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Progress (%)')
plt.title('Epoch Progress Over Time')
plt.grid(True)
plt.show()

# 绘制损失变化图
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], marker='o', label='Loss')
plt.plot(df['Epoch'], df['Loss Simple Step'], marker='x', linestyle='--', label='Loss Simple Step')
plt.plot(df['Epoch'], df['Loss VLB Step'], marker='s', linestyle='-.', label='Loss VLB Step')
plt.plot(df['Epoch'], df['Loss Step'], marker='^', linestyle=':', label='Loss Step')
plt.plot(df['Epoch'], df['Loss Simple Epoch'], marker='d', label='Loss Simple Epoch')
plt.plot(df['Epoch'], df['Loss VLB Epoch'], marker='*', label='Loss VLB Epoch')
plt.plot(df['Epoch'], df['Loss Epoch'], marker='p', label='Loss Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 绘制速率变化图
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Rate'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Rate (s/it)')
plt.title('Iteration Rate Over Epochs')
plt.grid(True)
plt.show()