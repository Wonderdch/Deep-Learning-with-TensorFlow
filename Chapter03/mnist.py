import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

# 设置GPU使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)

(xs, ys), _ = datasets.mnist.load_data()  # 加载MNIST数据集
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())

batch_size = 32

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs, ys))  # 构建数据集对象
db = db.batch(batch_size).repeat(30)  # 批量训练

# 利用Sequential容器封装3个网络层，前网络层的输出默认作为下一层的输入
model = Sequential([layers.Dense(256, activation='relu'),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(10)])
model.build(input_shape=(4, 28 * 28))
model.summary()

optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()

for step, (x, y) in enumerate(db):

    with tf.GradientTape() as tape:  # 构建梯度记录环境
        # 打平操作，[b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28 * 28))
        # 得到模型输出 output [b, 784] => [b, 10]
        out = model(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # 计算差的平方和，[b, 10]
        loss = tf.square(out - y_onehot)
        # 计算每个样本的平均误差，[b]
        loss = tf.reduce_sum(loss) / x.shape[0]

    acc_meter.update_state(tf.argmax(out, axis=1), y)

    # 计算参数的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 更新网络参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % 2000 == 0:
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
