import numpy as np
import skfuzzy as fuzz
from sklearn.datasets import load_iris,load_wine
import matplotlib.pyplot as plt

# # 加载Iris数据集
# iris = load_wine()
# X = iris.data.T  # 需要将数据转置，因为 skfuzzy 要求数据以特征为行
# 加载Wine数据集
wine = load_wine()
X = wine.data.T

# 设置模糊聚类的簇数
n_clusters = 3

# 使用FCM算法
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X, n_clusters, m=2, error=0.005, maxiter=1000, init=None
)

# 获取每个样本的隶属度
cluster_membership = np.argmax(u, axis=0)

# 计算 FCI
fci = np.sum((u ** 2) * d)
print(f"Fuzzy c-Means Clustering Index (FCI): {fci}")

# 计算 Xie-Beni 指数
xb_index = np.sum((d**2) / np.min(u, axis=0))
print(f"Xie-Beni Index: {xb_index}")

# 可视化模糊聚类结果
colors = ['b', 'g', 'r']

for j in range(n_clusters):
    plt.scatter(X[0, cluster_membership == j], X[1, cluster_membership == j], color=colors[j], s=30, label=f'Cluster {j + 1}')

plt.scatter(cntr.T[0], cntr.T[1], marker='*', s=200, color='black', label='Cluster Centers')
plt.title('Fuzzy C-Means Clustering on Wine Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()
