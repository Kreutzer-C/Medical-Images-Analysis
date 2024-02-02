from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载MNIST数据集
mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float64')
y = mnist.target.astype('int64')

# 将数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 计算PCA
pca = PCA(n_components=2)  # 选择要保留的主成分数量
X_pca_train = pca.fit_transform(X_train)

# 可视化降维后的数据
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.colorbar(scatter, label='Digit Label')
plt.title('PCA on MNIST Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
