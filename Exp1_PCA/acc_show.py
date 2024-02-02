import matplotlib.pyplot as plt

# 从txt文件读取数据
file_path = 'acc_history.txt'  # 替换为你的文件路径
data = []

with open(file_path, 'r') as file:
    for line in file:
        dimensions, accuracy = map(float, line.strip().split())
        data.append((dimensions, accuracy))

# 将数据分解为两个列表
dimensions, accuracy = zip(*data)

# 绘制曲线图
plt.plot(dimensions, accuracy, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs PCA Dimensions')
plt.xlabel('PCA Target Dimension')
plt.ylabel('Accuracy')
plt.xticks(dimensions)
plt.grid(True)
plt.show()
