import tensorflow as tf
import numpy as np

a = tf.convert_to_tensor([1, 2.])
print(a)

b = tf.convert_to_tensor(np.array([[1, 2.], [3, 4]]))
print(b)

a, b = tf.zeros([]), tf.ones([])  # 创建全 0，全 1 的标量
print(a, b)

a, b = tf.zeros([2, 2]), tf.ones([3, 2])  # 创建全 0，全 1 的矩阵
print(a, b)

a = tf.ones([2, 3])  # 创建一个矩阵
b = tf.zeros_like(a)  # 创建一个与 a 形状相同，但是全 0 的新矩阵
print(b)

a = tf.fill([2, 2], 99)  # 创建 2 行 2 列，元素全为 99 的矩阵
print(a)

a = tf.random.normal([2, 2])  # 创建标准正态分布（均值为 0，标准差为 1）的张量
print(a)

b = tf.random.uniform([2, 2], maxval=10)  # 创建采样自[0,10)均匀分布的矩阵
print(b)

c = tf.random.uniform([2, 2], maxval=100, dtype=tf.int32)  # 创建采样自[0,100)均匀分布的整型矩阵
print(c)

a = tf.range(10, delta=2)  # 创建 0~10，步长为 2 的整形序列
print(a)

b = tf.range(1, 10, delta=2)  # 1~10，步长为 2
print(b)
