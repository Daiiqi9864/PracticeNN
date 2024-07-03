import os
import pickle
import numpy as np
np.object = object  # monkey patch solution for compatible problem between tensorflow 2.6.x and numpy 1.2x.x
np.int = int
np.float = float
np.bool = bool
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')     # define a background for pyplot
import matplotlib.pyplot as plt

data_path = r'C:\Users\Daiiqi\train_data.pickle'    # r表示直接用字面意思，不执行\转义符
model_path = r'C:\Users\Daiiqi\train_model.pickle'

# 生成数据
def generate_data(num_points=100):
    x = np.linspace(-np.pi, np.pi, num_points).reshape(num_points, 1)
    y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)  # 添加一些噪声
    return x, y
# 准备数据
x_train, y_train = generate_data()

"""
class FittingModel:
    def __init__(self):
        # 定义模型
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_shape=(1,), activation='tanh'),  # 隐藏层
            tf.keras.layers.Dense(1, activation=None)  # 输出层，这里不使用激活函数因为是回归问题
        ])

        self.optimizer = tf.keras.optimizers.Adam()  # 定义Adam优化器
        self.loss_function = tf.keras.losses.MeanSquaredError()  # 定义损失函数

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:  # 开启梯度记录
            predictions = self.model(inputs, training=True)
            loss = self.loss_function(targets, predictions)  # 用自定义函数计算损失
        gradients = tape.gradient(loss, self.model.trainable_variables)  # 计算梯度
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  # 应用梯度更新权重
        return loss

    def fit(self, x_train, y_train, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in zip(x_train, y_train):
                batch_loss = self.train_step(np.array([x_batch]).reshape(1, 1), np.array([y_batch]).reshape(1, 1))
                total_loss += batch_loss.numpy()
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(x_train)}")

    def predict(self, x_test):
        return self.model(x_test)
"""

# 自定义图
class CustomModel(tf.Module):
    def __init__(self, input_dim, hidden_units):
        super(CustomModel, self).__init__()
        self.w1 = tf.Variable(tf.random.normal([input_dim, hidden_units], dtype=tf.float64), name="w1", trainable=True)  # 隐藏层权重
        self.b1 = tf.Variable(tf.zeros([hidden_units], dtype=tf.float64), name="b1", trainable=True)  # 隐藏层偏置
        self.w2 = tf.Variable(tf.random.normal([hidden_units, 1], dtype=tf.float64), name="w2", trainable=True)  # 输出层权重
        self.b2 = tf.Variable(tf.zeros([1], dtype=tf.float64), name="b2", trainable=True)  # 输出层偏置

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def forward_pass(self, inputs):
        hidden_output = tf.tanh(tf.matmul(inputs, self.w1) + self.b1)  # 隐藏层前向传播
        output = tf.matmul(hidden_output, self.w2) + self.b2  # 输出层前向传播
        return output

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self.forward_pass(inputs)
            loss = self.loss_function(targets, predictions)
        gradients = tape.gradient(loss, [self.w1, self.b1, self.w2, self.b2])  # 计算所有权重和偏置的梯度
        self.optimizer.apply_gradients(zip(gradients, [self.w1, self.b1, self.w2, self.b2]))  # 应用梯度更新
        return loss

    def fit(self, x_train, y_train, epochs=100):
        for epoch in range(epochs):
            total_loss = self.train_step(x_train, y_train)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

    def predict(self, x_test):
        return self.forward_pass(x_test)

    def save_weights(self, checkpoint_path):
        # 保存模型权重到指定路径。
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save(file_prefix=checkpoint_path)

    def load_weights(self, checkpoint_path):
        checkpoint = tf.train.Checkpoint(model=self)
        latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Weights restored from {latest_checkpoint}")
        else:
            print("No checkpoint found to restore from.")


"""
# 初始化模型并训练
model = FittingModel()  # Keras模型初始化
model = tf.keras.models.load_model(model_path)  # Keras模型读取，不初始化
model.fit(x_train, y_train, epochs=100)
"""

# 初始化模型并训练
model = CustomModel(input_dim=1, hidden_units=32)  # 假设输入维度为1，隐藏层单元数为32
model.load_weights("my_model_checkpoint")   # 若之前已保存断点，这里读取。需要先初始化
model.fit(x_train, y_train, epochs=100)


# 计算得到的变量
train_data = [x_train,y_train]  # 这可以是任何可pickle的对象，如列表、字典、numpy数组等

# model.model.save(model_path)  # Keras模型保存
model.save_weights("my_model_checkpoint")   # 自定义模型保存

with open(data_path, 'wb') as f:
    pickle.dump(train_data, f)
'''
with open(data_path, 'rb') as f:
    x_train,y_train = pickle.load(f)
'''

# 预测并绘制结果
x_test = np.linspace(-np.pi, np.pi, 200)
y_pred = model.predict(x_test[:, np.newaxis])
plt.figure(figsize=(12, 6))
plt.plot(x_train, y_train, 'bo', label='Actual Data')
plt.plot(x_test, y_pred, 'r-', label='Predicted Data')
plt.legend()
# plt.savefig('model_loss.png')
plt.show()
