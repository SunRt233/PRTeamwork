import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(0)

# 定义单元格数量
n = 100

# 定义方格的高度和宽度比例
cell_height = 0.1
cell_width = 1 / n

# 创建一个长方形代表原数据集
colors = plt.cm.GnBu(np.linspace(0.3, 1, n))

# 创建一个长方形
fig, axs = plt.subplots(2, 1, figsize=(10, 4))

# 绘制打乱前的数据集分割
for i in range(n):
    axs[0].barh(0, cell_width, left=i * cell_width, height=cell_height, color=colors[i], edgecolor='black')

# 去掉坐标轴刻度
axs[0].set_xticks([])
axs[0].set_yticks([])

# 添加标题
axs[0].set_title('Data Before Shuffling')

# 打乱数据集
indices = np.arange(n)
np.random.shuffle(indices)

# 绘制打乱后的数据集分割
for i, idx in enumerate(indices):
    axs[1].barh(0, cell_width, left=i * cell_width, height=cell_height, color=colors[idx], edgecolor='black')

# 在打乱后的图上方绘制三个方框
# 假设前60%为训练集，接下来20%为验证集，最后20%为测试集
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

train_end = int(train_ratio * n)
val_end = train_end + int(val_ratio * n)

axs[1].add_patch(plt.Rectangle((0, -cell_height/2), train_ratio, cell_height, fill=False, edgecolor='blue', linewidth=4, linestyle='--', hatch='//'))
axs[1].add_patch(plt.Rectangle((train_ratio, -cell_height/2), val_ratio, cell_height, fill=False, edgecolor='green', linewidth=4, linestyle='--', hatch='//'))
axs[1].add_patch(plt.Rectangle((val_end/n, -cell_height/2), test_ratio, cell_height, fill=False, edgecolor='red', linewidth=4, linestyle='--', hatch='//'))

# 添加图例
handles = [
    plt.Rectangle((0,0),1,1, color='blue'),
    plt.Rectangle((0,0),1,1, color='green'),
    plt.Rectangle((0,0),1,1, color='red')
]
labels = ['Training Set', 'Validation Set', 'Test Set']
axs[1].legend(handles, labels, title="Data Sets", loc='upper right')

# 去掉坐标轴刻度
axs[1].set_xticks([])
axs[1].set_yticks([])

# 添加标题
axs[1].set_title('Data Splitting After Shuffling')

# 显示图形
plt.tight_layout()
plt.show()
