import tensorflow as tf

a = tf.random.normal([4, 3, 28, 32])
b = tf.random.normal([4, 3, 32, 2])
c = a @ b  # [4,3,28,2]
print(c)

a = tf.random.normal([4, 28, 32])
b = tf.random.normal([32, 16])
# 自动将变量 b 扩展为公共 shape：[4,32,16]，再与变量 a 进行批量形式地矩阵相乘，得到结果的 shape 为[4,28,16]
c = tf.matmul(a, b)
print(c)
