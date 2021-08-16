import tensorflow as tf

out = tf.random.uniform([4, 10])  # 随机模拟网络输出
y = tf.constant([2, 3, 2, 0])  # 随机构造样本真实标签
y = tf.one_hot(y, depth=10)  # one-hot 编码
print(y)
loss = tf.keras.losses.mse(y, out)  # 计算每个样本的 MSE 误差
loss = tf.reduce_mean(loss)  # 平均 MSE,loss 应是标量
print(loss)
print("----------")

# z=wx,模拟获得激活函数的输入 z
z = tf.random.normal([4, 2])
b = tf.zeros([2])  # 创建偏置向量
z = z + b  # 累加上偏置向量
print(z)
print("----------")

x = tf.random.normal([2, 4])  # 2 个样本，特征长度为 4 的张量
w = tf.ones([4, 3])  # 定义 W 张量
b = tf.zeros([3])  # 定义 b 张量
o = x @ w + b  # X@W+b 运算
print(o)
print("----------")
