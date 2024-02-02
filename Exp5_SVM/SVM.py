import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

# 加载训练集和测试集
X_train = np.load('../dataset/train_images.npy')
y_train = np.load('../dataset/train_labels.npy')
X_test = np.load('../dataset/val_images.npy')
y_test = np.load('../dataset/val_labels.npy')

# 数据预处理
# 将图像数据展平成一维数组，以便用于SVM
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# 超参数设置
svm_c = 10
kernel = 'poly'
degrees = [1, 2, 3, 4, 5]
coef0_values = [0.0, 0.1, 0.5, 1.0]


for degree in degrees:
    for coef0 in coef0_values:
        clf = svm.SVC(kernel=kernel, C=svm_c, degree=degree, coef0=coef0)

        clf.fit(X_train_flattened, y_train)

        y_pred = clf.predict(X_test_flattened)

        accuracy = accuracy_score(y_test, y_pred)
        print("===>Kernel:{} SVM_C:{} degree:{} coef0:{} Accuracy: {:.4f}".format(kernel,
                                                                                  svm_c, degree, coef0,
                                                                                  accuracy))
        with open(f'acc_history_{kernel}_{svm_c}.txt', 'a') as file:
            # 在文件中写入三个变量的值
            file.write(f"{degree} {coef0} {accuracy}\n")