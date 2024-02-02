from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import adjusted_rand_score,davies_bouldin_score
import matplotlib.pyplot as plt

# 加载Iris数据集
# iris = load_iris()
# X = iris.data
# y_true = iris.target
wine = load_wine()
X = wine.data
y_true = wine.target

# 初始化K-Means模型，选择聚类数目为3
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(X)

# 计算Adjusted Rand Index
ari = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {ari}")

# 计算Davies-Bouldin Index
dbi = davies_bouldin_score(X, y_pred)
print(f"Davies-Bouldin Index: {dbi}")

# 可视化聚类结果
plt.scatter(X[:, 12], X[:, 8], c=y_pred, cmap='viridis')
plt.title('K-Means Clustering on Wine Dataset')
plt.xlabel('Proline')
plt.ylabel('Proanthocyanins')
plt.show()
