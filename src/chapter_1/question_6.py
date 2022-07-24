# @Date 2022/7/24
# @Author Meimin Wang
# @Filename question_6.py
# @Software PyCharm
# @Email blessed@gmail.com
# @Email wangmeimin@cstor.cn
from math import pi, sqrt
from matplotlib import pyplot as plt
import tensorflow as tf

x = tf.linspace(-5, 5, 1000)
y = 1. / sqrt(2 * pi) * tf.exp(-1. / 2 * x ** 2)

plt.plot(x.numpy(), y.numpy())
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Normal Distribution: $f(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}$')
plt.show()