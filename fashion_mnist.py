import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用 TensorFlow 直接加载 Fashion MNIST 数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 数据归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 添加L2正则化和Dropout
l2_reg = tf.keras.regularizers.l2(0.001)  # L2正则化系数

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),  # 添加L2正则化
    tf.keras.layers.Dropout(0.5),  # Dropout rate: 50%
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),  # 添加另一层的L2正则化
    tf.keras.layers.Dropout(0.5),  # 另一层的Dropout
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试准确率：{test_acc}")

# 显示部分预测结果
predictions = model.predict(x_test)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(f"预测: {predictions[i].argmax()} | 实际: {y_test[i]}")
plt.show()
