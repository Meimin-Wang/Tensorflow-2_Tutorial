# @Date 2022/7/24
# @Author Meimin Wang
# @Filename question_7.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn
import tensorflow as tf

images = tf.random.normal([10, 100, 100, 3])
mean = tf.reduce_mean(images, axis=[0, 1, 2])
std = tf.math.reduce_std(images, axis=[0, 1, 2])
print(mean)
print(std)