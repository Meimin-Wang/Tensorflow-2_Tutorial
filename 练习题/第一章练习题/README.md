## 第一章：张量（Tensor）练习题

#### 使用TensorFlow实现以下任务：

1. 使用Tensorflow创建一个矩阵$\\boldsymbol A=\\left[\\begin{matrix}1&2&3&4&5\\\\2&1&2&3&4\\\\3&2&1&2&3\\\\4&3&2&1&2\\\\5&4&3&2&1\\end{matrix}\\right]$。
2. 使用Tensorflow创建一个矩阵$\\boldsymbol A=\\left[\\begin{matrix}1&2&3&4&5\\\\6&7&8&9&10\\\\11&12&13&14&15\\\\16&17&18&19&20\\\\21&22&23&24&25\\\\26&27&28&29&30\\end{matrix}\\right]$，输出其形状；将该矩阵变换为一个形状为$1\\times 3\\times 2\\times 5$的张量，并以第二个轴为方向，求其平均值。
3. 使用Tensorflow求矩阵$\\boldsymbol A=\\left[\\begin{matrix}3&6&5\\\\9&1&4\\\\1&2&2\\end{matrix}\\right]$的转置矩阵（transpose matrix），迹(trace)、逆矩阵(inverse matrix)、行列式(determination)、特征值(eigenvalue)，范数(norm)。
4. 设矩阵$\\boldsymbol A=\\left[\\begin{matrix}1&2&3&4&5\\\\6&7&8&9&10\\\\11&12&13&14&15\\\\16&17&18&19&20\\\\21&22&23&24&25\\\\26&27&28&29&30\\end{matrix}\\right]$，（1）求所有$4\\times4$的子矩阵的行列式的值；（2）求矩阵所有的奇数行和偶数列的和；（3）取行$1, 3, 4$行和$0, 2, 4$列的子矩阵。
5. 有5张图片，路径为`/path/to/image/i.jpg`，其中$i=1, 2, 3, 4, 5$。请获取每个图像的名字，即`i.jpg`。
6. 使用TensorFlow绘制定义域在$[-5, 5]$之间的一维标准正态分布（$\\mu=0, \\sigma=1$）的概率密度函数曲线。
7. 通过`tf.random.normal`函数随机生成10张图片，通道为RGB的3通道$c=3$，高$h=100$，宽$w=100$，分别求出每个通道的均值和方差。
