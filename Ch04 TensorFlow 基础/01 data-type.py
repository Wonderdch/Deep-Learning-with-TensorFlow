import tensorflow as tf

a = 1.2  # python语言方式创建标量
aa = tf.constant(1.2)  # TF方式创建标量
print(type(a), type(aa), tf.is_tensor(aa))

x = tf.constant([1, 2., 3.3])
print(x)  # 打印TF张量的相关信息

x = x.numpy()  # 将TF张量的数据导出为numpy数组格式
print(x)

a = tf.constant([1, 2, 3.])  # 创建3个元素的向量
print(a, a.shape)

a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 创建3维张量
print(a)
print("------- shape -------")
print(a.shape)

a = tf.constant('Hello, Deep Learning.')  # 创建字符串
print(a)

print(tf.strings.lower(a))

a = tf.constant(True)  # 创建布尔类型标量
print(a)

a = tf.constant([True, False])  # 创建布尔类型向量
print(a)

a = tf.constant(True)  # 创建 TF 布尔张量
print("a is True :", a is True)  # TF 布尔类型张量与 python 布尔类型比较
print("a == True :", a == True)  # 仅数值比较
