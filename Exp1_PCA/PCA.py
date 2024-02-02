import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import os,sys
sys.path.append(os.getcwd())
sys.path.append(r'/opt/data/private/Medical Images Analysis/')

from model.Mymodel import MLP
from utils.utils_train import correct_cal

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print(">>>===========device:{}({},{}MB)".format(device,GPU_device.name,GPU_device.total_memory / 1024 ** 2))

# 超参数
epoches = 10
batch_size = 128
pca_components = [1,2,3,4,5,6,7,8,9]
lr = 1e-4

# 加载MNIST数据集
mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float64')
y = mnist.target.astype('int64')
num_classes = y.nunique()

# 将数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

df = pd.DataFrame(columns=['PCA_n', 'ACC'])
for pca_n in pca_components:
    print(f'>>>===Now doing pca_n = {pca_n}===<<<')
    # 计算PCA
    pca = PCA(n_components=pca_n)  # 选择要保留的主成分数量
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.transform(X_test)

    # 神经网络分类
    # 数据集
    train_data_tensor = torch.tensor(X_pca_train).to(device)
    train_label_tensor = torch.Tensor(y_train.values).to(device)
    train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    test_data_tensor = torch.tensor(X_pca_test).to(device)
    test_label_tensor = torch.Tensor(y_test.values).to(device)
    test_dataset = TensorDataset(test_data_tensor, test_label_tensor)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # 模型
    model = MLP(pca_n, num_classes).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    acc_best = 0
    for epoch in tqdm(range(epoches)):
        model.train()
        total_count = 0
        total_correct = 0
        total_loss = 0

        for iteration, (data, label) in enumerate(train_loader):
            data = data.to(torch.float32)
            label = label.to(torch.long)
            total_count += label.size(0)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_func(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            correct_num = correct_cal(output, label)
            total_correct += correct_num

        average_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_count
        print(f"===Training:[Loss:{average_loss} Acc:{accuracy}]")

        # Test
        model.eval()
        test_count = 0
        test_correct = 0
        test_loss = 0
        with torch.no_grad():
            for iteration, (data, label) in enumerate(test_loader):
                data = data.to(torch.float32)
                label = label.to(torch.long)
                test_count += label.size(0)

                output = model(data)

                loss = loss_func(output, label)
                test_loss += loss.item()

                correct_num = correct_cal(output, label)
                test_correct += correct_num

            average_loss_test = test_loss / len(test_loader)
            accuracy_test = test_correct / test_count
            print(f"===Testing:[Loss:{average_loss_test} Acc:{accuracy_test}]")

        if accuracy_test > acc_best:
            acc_best = accuracy_test
            print("===[PCA:{} dim] ACC BEST:{:.4f}===".format(pca_n,acc_best))

    row_index = len(df)
    df.loc[row_index] = [pca_n, acc_best]

df.to_csv('acc_history.txt', sep='\t', index=False)