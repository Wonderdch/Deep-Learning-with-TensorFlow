import tensorflow as tf

a = tf.constant([-1, 0, 1, 2])  # 创建 TF 张量
aa = tf.Variable(a)  # 转换为 Variable 类型
print(aa.name, aa.trainable)  # Variable 类型张量的属性

a = tf.Variable([[1, 2], [3, 4]])  # 直接创建 Variable 张量
print(a)

# 普通张量其实也可以通过 GradientTape.watch() 方法临时加入跟踪梯度信息的列表，从而支持自动求导功能
