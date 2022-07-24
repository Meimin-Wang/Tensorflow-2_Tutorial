# @Date 2022/7/24
# @Author Meimin Wang
# @Filename question_5.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn

import tensorflow as tf

x = tf.constant([
    '/path/to/image/1.jpg',
    '/path/to/image/2.jpg',
    '/path/to/image/3.jpg',
    '/path/to/image/4.jpg',
    '/path/to/image/5.jpg',
], dtype=tf.string)

image_names = tf.strings.split(x, '/')  # ragged tensor
for image in image_names:
    print(image[-1].numpy().decode('UTF-8'))