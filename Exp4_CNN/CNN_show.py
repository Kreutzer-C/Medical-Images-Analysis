import matplotlib.pyplot as plt

dropout = [0, 0.05, 0.15, 0.3, 0.5]
weight_decay = [1e-4, 3e-4, 5e-4, 1e-3]

# acc = [0.8756, 0.9078, 0.9125, 0.8805, 0.8586]
# f1 = [0.8379, 0.8784, 0.8832, 0.8536, 0.8282]
acc = [0.8680, 0.9213, 0.8375, 0.7836]
f1 = [0.8076, 0.8925, 0.7754, 0.7142]

# plt.plot(dropout, acc, marker='o', linestyle='-', color='b')
# plt.title('Accuracy & F1 Score vs Dropout prob')
# plt.xlabel('Dropout prob')
# plt.ylabel('Accuracy')
# plt.xticks(dropout)
# plt.grid(True)
# plt.show()

# 创建第一个y轴（左侧）
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Weight Decay')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(weight_decay,acc, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个y轴（右侧）
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('F1 Score', color=color)
ax2.plot(weight_decay,f1, color=color, label='F1 Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Accuracy & F1 Score vs Weight Decay')
plt.xticks(weight_decay)
plt.grid(True)
plt.show()