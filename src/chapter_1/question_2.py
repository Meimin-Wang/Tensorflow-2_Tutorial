# @Date 2022/7/24
# @Author Meimin Wang
# @Filename question_2.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn

import tensorflow as tf

x = tf.range(1, 31)

tf.print(x.shape)

x = tf.reshape(x, (1, 3, 2, 5))

tf.print('Transformed:', x.shape)

print(tf.reduce_mean(x, axis=1))