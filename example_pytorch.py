import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# 设定随机种子以便结果可复现
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# 生成目标数据：正弦波形
def generate_sine_data(points=100):
    x = np.linspace(-np.pi, np.pi, points).astype(np.float32)
    y = np.sin(x)
    return x, y
# 准备数据
x_data, y_data = generate_sine_data(points=100)
x_data = x_data.reshape(-1, 1)  # 调整形状以便输入神经网络
y_data = y_data.reshape(-1, 1)

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        # 初始化函数接收输入数据和目标数据，假设它们都是NumPy数组。
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # 返回数据集的长度。
        return len(self.inputs)

    def __getitem__(self, index):
        # 根据索引返回一个样本数据及其标签，同时将NumPy数组转换为PyTorch的Tensor。
        input_data = torch.tensor(self.inputs[index], dtype=torch.float32, requires_grad=False).to(device)    # to device 即传到 gpu 上
        target_data = torch.tensor(self.targets[index], dtype=torch.float32, requires_grad=False).to(device)
        return input_data, target_data

# 实例化数据集
dataset = CustomDataset(x_data, y_data)
# 创建DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


# 设置参数
input_size = 1
hidden_size = 16  # 隐藏层节点数
output_size = 1
# 学习率
learning_rate = 0.01
epochs = 100
# 神经网络会迭代训练很多次。每遍历一次数据集叫做一个epoch，在epoch下每取一个batch都会进行一次训练。


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# 实例化模型、损失函数和优化器
model = Net(input_size, hidden_size, output_size).to(device)
# model = torch.load('model_state.pth')     # 读取
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
for epoch in range(epochs):
    lost_memory = 0
    for inputs, targets in data_loader:
        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)
        lost_memory = loss.item()

        # 反向传播和优化
        optimizer.zero_grad()   # 清空不需要的梯度
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {lost_memory:.4f}')

# 保存模型状态到文件
torch.save(model, 'model_state.pth')

# 加载模型状态
# model = torch.load('model_state.pth')

# 绘制预测结果与真实数据对比
x_tensor = torch.tensor(x_data, dtype=torch.float32)
with torch.no_grad():
    predicted = model(x_tensor).numpy()


'''
# 手动定义权重和偏置，这里初始化为随机值
W1 = torch.randn((input_size, hidden_size), requires_grad=True).to(device)  # requires_grad 标记一个 torch.tensor 是否可训练
b1 = torch.zeros(hidden_size, requires_grad=True).to(device)
W2 = torch.randn((hidden_size, output_size), requires_grad=True).to(device)
b2 = torch.zeros(output_size, requires_grad=True).to(device)

# 训练循环
for epoch in range(epochs):
    lost_memory = 0
    for inputs, targets in data_loader:
        # 手工前向传播
        hidden = torch.tanh(inputs @ W1 + b1)  # 使用 tanh 作为激活函数，@表示矩阵乘法。这个过程也可以扔到函数里，然后封一个类。
        outputs = hidden @ W2 + b2

        # 手工计算损失。只要 loss 是通过含 tensor 型参数的中间式（比如含 W1 它们的 outputs）计算出来的，就可以正常使用 loss.backward()
        # 如果需要在 gpu 上执行，那么 loss 式中涉及的所有 tensor 都必须已转移到了 device=cuda 上。
        loss = torch.mean((outputs - targets) ** 2)  # MSE损失。
        lost_memory = loss.item()
        # 反向传播和优化，仅对 loss 中标注为 requires_grad=True 的 torch.tensor 生效。
        # 它的结果会存在 W1.grad 里。如果需要用推出的梯度式 dL_dW1 计算替代这步，正常算就好，注意算出的结果仍需是 tensor 型就行。
        loss.backward()

        # 手动更新权重。建议对于能定义前向过程与损失函数的问题都定义个 Adam 优化器替换掉这段。
        # Adam 优化器仍然基于梯度下降方法，但它能自适应调学习率，还能用过去的梯度加速收敛。
        with torch.no_grad():   # 加这句的意思是，防止 torch 对块内干的事还要计算梯度。要不然 loss.backward() 里会被叠加上不必要的梯度。
            W1 -= learning_rate * W1.grad   # 如果用的是手动计算，这里用的就不是 W1.grad，而是 dL_dW1 了。
            b1 -= learning_rate * b1.grad
            W2 -= learning_rate * W2.grad
            b2 -= learning_rate * b2.grad
            # 清零梯度，防止梯度累积
            W1.grad.zero_()
            b1.grad.zero_()
            W2.grad.zero_()
            b2.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {lost_memory:.4f}')


# 保存模型状态到文件
model_state = [W1, b1, W2, b2]
torch.save(model_state, 'model_state.pth')

# 加载模型状态
# W1, b1, W2, b2 = torch.load('model_state.pth')

# 绘制预测结果与真实数据对比
x_tensor = torch.tensor(x_data, dtype=torch.float32)
predicted = torch.tanh(x_tensor @ W1 + b1) @ W2 + b2
predicted = predicted.detach().numpy().reshape(-1)
'''


plt.figure(figsize=(10, 5))
plt.plot(x_data, y_data, label='Actual')
plt.plot(x_data, predicted, label='Predicted')
plt.legend()
plt.show()