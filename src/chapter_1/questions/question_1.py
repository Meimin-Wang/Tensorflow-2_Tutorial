# @Date 2022/7/24
# @Author Meimin Wang
# @Filename question_1.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn

import tensorflow as tf

x = tf.add_n([
    tf.linalg.diag(tf.ones(i, dtype=tf.int32) * (5-i+1), k=5-i)
    for i in range(1, 6)
])

x += tf.transpose(x) - tf.eye(5, 5, dtype=tf.int32)

tf.print(x)