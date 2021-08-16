import tensorflow as tf
import numpy as np

a = tf.constant(123456789, dtype=tf.int16)  # 溢出
b = tf.constant(123456789, dtype=tf.int32)
print(a)
print(b)

print('before:', a.dtype)  # 读取原有张量的数值精度
if a.dtype != tf.float32:  # 如果精度不符合要求，则进行转换
    a = tf.cast(a, tf.float32)  # tf.cast 函数可以完成精度转换
print('after :', a.dtype)  # 打印转换后的精度

a = tf.constant(np.pi, dtype=tf.float16)  # 创建 tf.float16 低精度张量
a = tf.cast(a, tf.double)  # 转换为高精度张量
print(a)

a = tf.constant(123456789, dtype=tf.int32)
a = tf.cast(a, tf.int16)  # 转换为低精度整型
print(a)

a = tf.constant([True, False])
a = tf.cast(a, tf.int32)  # 布尔类型转整型
print(a)

a = tf.constant([-1, 0, 1, 2])
a = tf.cast(a, tf.bool)  # 整型转布尔类型
print(a)
