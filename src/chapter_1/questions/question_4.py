# @Date 2022/7/24
# @Author Meimin Wang
# @Filename question_4.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn

import tensorflow as tf

A = tf.reshape(tf.range(1, 31, dtype=tf.float32), (6, 5))
rows, cols = A.shape

# 4x4 submatrix
for i in range(rows - 4 + 1):
    for j in range(cols - 4 + 1):
        submatrix = A[i:(i+4), j:(j+4)]
        print(f'({i}, {j})', tf.linalg.det(submatrix))

print(tf.reduce_sum(A[1::2, ::2]))

a = tf.gather(A, [1, 3, 4])
b = tf.gather(a, [0, 2, 4], axis=1)
print(b)