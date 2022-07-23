# @Date 2022/7/24
# @Author Meimin Wang
# @Filename question_3.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn
import tensorflow as tf
x = tf.constant([[3., 6., 5.], [9., 1., 4.], [1., 2., 2.]])

tf.print('Transpose:', tf.linalg.einsum('ij->ji', x))
tf.print('Trace:', tf.linalg.trace(x))
tf.print('Inverse matrix:', tf.linalg.inv(x))
tf.print('Determination:', tf.linalg.det(x))
tf.print('Eig:', tf.linalg.eigvals(x))
tf.print('Norm:', tf.norm(x))