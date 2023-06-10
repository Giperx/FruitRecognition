"""
@File    : geneImage.py
@Author  : GiperHsiue
@Time    : 2023/6/10 22:59
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import shutil

datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转角度范围
    width_shift_range=0.2,  # 随机水平平移范围(相对于图片宽度)
    height_shift_range=0.2,  # 随机竖直平移范围(相对于图片高度)
    shear_range=0.2,  # 随机裁剪
    zoom_range=0.2,  # 随机缩放
    horizontal_flip=True,  # 随机水平翻转
    vertical_flip=True,  # 随机竖直翻转
    fill_mode='nearest')  # 填充模式

# train_dir = '../fruit/train'
train_dir = '../fruit/val'
# save_dir = '../geneFruit/train'
save_dir = '../geneFruit/val'

for subdir in os.listdir(train_dir):
    if not os.path.exists(os.path.join(save_dir, subdir)):
        os.makedirs(os.path.join(save_dir, subdir))
    for file in os.listdir(os.path.join(train_dir, subdir)):
        img = load_img(os.path.join(train_dir, subdir, file))
        x = img_to_array(img)
        # 将图片转化为4D张量(batch_size, height, width, channels)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=os.path.join(save_dir, subdir), save_prefix=file[:-4], save_format='jpg'):
            i += 1
            if i > 3:  # 控制每张图片生成4张新图像
                break
