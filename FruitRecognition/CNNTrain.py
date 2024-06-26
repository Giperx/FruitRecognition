"""
@File    : CNNTrain.py
@Author  : GiperHsiue
@Time    : 2023/5/29 18:18
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names

# def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
#     # 数据增强
#     train_datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#
#     # 加载训练集
#     train_ds = train_datagen.flow_from_directory(
#         data_dir,
#         target_size=(img_height, img_width),
#         batch_size=batch_size,
#         seed=123,
#         class_mode='categorical')
#
#     # 加载测试集
#     val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         test_data_dir,
#         label_mode='categorical',
#         seed=123,
#         image_size=(img_height, img_width),
#         batch_size=batch_size)
#     class_names = val_ds.class_names
#
#     # 返回处理之后的训练集、验证集和类名
#     return train_ds, val_ds, class_names


# 构建CNN模型
def model_load(IMG_SHAPE=(224, 224, 3), class_num=15):
    # 搭建模型
    model = tf.keras.models.Sequential([
        # 对模型做归一化
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # 卷积层，该卷积层的输出为16
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 卷积层，该卷积层的输出为32
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # 添加池化层
        tf.keras.layers.MaxPooling2D(2, 2),
        # Add another convolution
        # 卷积层，输出为64
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # 转化为一维
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model


# 展示训练过程的曲线
def showAccuracyAndLoss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # 保存图片
    filename = 'results_cnn.png'
    index = 1
    while os.path.isfile(os.path.join('resultsPng', filename)):
        filename = 'results_cnn' + str(index) + '.png'
        index += 1
    filename = os.path.join('resultsPng', filename)
    plt.savefig(filename, dpi=100)


def train(epochs):
    # 开始训练
    begin_time = time()

    # train_ds, val_ds, class_names = data_load("../fruit/train",
    #                                           "../fruit/val", 224, 224, 16)
    train_ds, val_ds, class_names = data_load("../geneFruit/train",
                                              "../fruit/val", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("models/cnn_fv.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")
    # 绘制模型训练过程图
    showAccuracyAndLoss(history)


if __name__ == '__main__':
    train(epochs=15)
