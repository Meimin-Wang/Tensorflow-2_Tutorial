## 2 数据集

数据是神经网络的生产资料，算力是神经网络的生产力，**数据集（Dataset）**是神经网络中的非常重要，也极大地影响了神经网络最后表现的性能。在机器学习领域，有一个著名的定律：

**GIGO: Garbage in Garbage out** ([Wiki](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out))

意思是当你的数据是脏的，那么是不可能得到好的模型的。

TF 2对数据集的构造和TF 1属于继承发展关系，但趋势是向着高级API进行发展。TF 2中与数据集相关的API在`tf.data.xxx`下，此外，TF官方还提供了另一种数据集hub，叫做`Tensorflow Datasets`，可以通过此API框架对著名的数据集进行获取和处理。

> Tip：TF datasets的数据下载之类的是在国内极其不友好，所以一般有些真正的项目中不使用这个框架，一般被用来测试，入门等。

TF datasets数据集列表：https://www.tensorflow.org/datasets/catalog/overview?hl=zh-cn#all_datasets

### 2.1 著名数据集加载

数据集大的分为**图像数据集**、**文本数据集**和**语音数据集**和其他一些数据集，比如**图数据集**等。

需要加载著名的数据集，通常有以下两种方法（MNIST例子）：

- Keras加载

  ```python
  import tensorflow as tf
  
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  print(X_train.shape, y_train.shape)
  print(X_test.shape, y_test.shape)
  ```

  输出：

  ```shell
  (60000, 28, 28) (60000,)
  (10000, 28, 28) (10000,)
  ```

- Tensorflow datasets加载

  ```python
  import tensorflow_datasets as tfds
  
  train_mnist_ds, test_mnist_ds = tfds.load(
      'mnist', data_dir='./mnist',
      as_supervised=True, split=['train', 'test'])
  print(train_mnist_ds)
  print(test_mnist_ds)
  ```

  输出：

  ```shell
  <PrefetchDataset element_spec=(TensorSpec(shape=(28, 28, 1), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>
  <PrefetchDataset element_spec=(TensorSpec(shape=(28, 28, 1), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>
  ```

关于MNIST数据集：MNIST是一个手写数字识别数据集，其中的图片大小为$28\times 28$，如下所示：

![](https://camo.githubusercontent.com/01c057a753e92a9bc70b8c45d62b295431851c09cffadf53106fc0aea7e2843f/687474703a2f2f692e7974696d672e636f6d2f76692f3051493378675875422d512f687164656661756c742e6a7067)

### 3.2 数据集创建与处理

数据集的一般处理流程如下：

![](/Volumes/BlessedWMM/my-git/Tensorflow-2_Tutorial/resources/数据集处理.png)

数据集本质上或者面向开发着应该呈现出一个迭代器（或生成器）的表现，而TF提供了一系列类似于大数据处理流水线的API，包括`map`和`reduce`操作。在经过处理后，我们可以通过以下方式进行遍历数据集：

```python
for examples in dataset:
    // ... do some thing
```

#### 3.2.1 创建数据集

```python
import numpy as np
import tensorflow as tf

# tf.data.Dataset.from_tensors(tensors, name=None)
# - tensors: a tuple of any type elements
ds = tf.data.Dataset.from_tensors(([1, 2, 3], [10, 20, 30], 'ds'))
print(list(ds.as_numpy_iterator()))

# tf.data.Dataset.from_tensor_slices(tensors, name=None)
# - tensors: zip every element
ds = tf.data.Dataset.from_tensor_slices([[1., 2., 3.], [4, 5, 6]], name='ds')
print(list(ds.as_numpy_iterator()))
a = np.array([1, 2, 3, 4, 5])
ds = tf.data.Dataset.from_tensor_slices({'x': a})
print(list(ds.as_numpy_iterator()))

ds = tf.data.Dataset.range(10)
print(list(ds.as_numpy_iterator()))
ds = tf.data.Dataset.range(1, 10, 3)
print(list(ds.as_numpy_iterator()))

# Deprecated
# ds = tf.data.Dataset.from_generator

# tf.data.Dataset.random(seed=None, name=None)
ds = tf.data.Dataset.random().take(10)
print(list(ds.as_numpy_iterator()))

# tf.data.Dataset.list_files like glob.glob
# ds = tf.data.Dataset.list_files('/file/dir/path')
```

输出

```shell
[(array([1, 2, 3], dtype=int32), array([10, 20, 30], dtype=int32), b'ds')]
[array([1., 2., 3.], dtype=float32), array([4., 5., 6.], dtype=float32)]
[{'x': 1}, {'x': 2}, {'x': 3}, {'x': 4}, {'x': 5}]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[1, 4, 7]
[1045308610, 3793825633, 553251631, 1110334695, 3223716891, 1246776994, 3830302915, 4030266722, 1304307785, 3566296323]
```

#### 3.2.2 数据变换（中间算子）

数据集流水线上进行数据变换，比如文件读取，归一化，数据增强，数据拼接，过滤等操作。中间算子指的是调用这些函数后会得到一个新的数据集。

常用的算子有：

- `map`：Map可以将数据集中的每一笔数据通过一个特定的函数进行变换，是非常常用的数据处理手段，例如我们希望输入的图像归一化到$[0,1]$之间，可以如下进行处理：

  ```python
  import numpy as np
  import tensorflow as tf
  
  AUTOTUNE = tf.data.AUTOTUNE
  
  # mock dataset
  X = np.random.randint(low=0, high=256, size=[100, 32, 32, 3], dtype=np.uint8)
  y = np.random.randint(low=0, high=10, size=(100, ), dtype=np.int32)
  
  def normalize(image, label):
      image = tf.cast(image, tf.float32)
      image = image / 255.
      return image, label
  
  ds = tf.data.Dataset.from_tensor_slices((X, y)).map(normalize, num_parallel_calls=AUTOTUNE)
  
  for images, labels in ds.take(1):
      print(images.shape, images.dtype)
      print(labels.shape, labels.dtype)
  ```

  输出：

  ```shell
  (32, 32, 3) <dtype: 'float32'>
  () <dtype: 'int32'>
  ```

  其中`AUTOTUNE=-1`，并且`num_parallel_calls`参数表示使用多少个CPU进行处理数据。`map`还可以经常被用作数据增强。

  **【注意】**尽量不要将所有操作写在一个函数中，然后使用`map`，尽量地拆解成不同的阶段，增加CPU的吞吐量和系统弹性。

- `shuffle`：用于置乱数据，可以更好地为神经网络训练，`shuffle`需要提供一个缓冲区大小，一般设置地比较大，比如`10000`等。

  ```python
  import tensorflow as tf
  
  ds = tf.data.Dataset.range(5).shuffle(10)
  print(list(ds.as_numpy_iterator()))
  ```

  输出：

  ```shell
  [4, 2, 3, 1, 0
  ```

- `repeat`：重复数据集，在训练的过程中通常将“看一遍”数据集称为一个epoch，我们也可以就训练一个epoch，但是数据集可以复制很多个，和多个epoch是一样的效果。

  ```python
  import tensorflow as tf
  
  ds = tf.data.Dataset.range(5).repeat(3)
  print(list(ds.as_numpy_iterator()))
  ```

  输出

  ```shell
  [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
  ```

- `zip`：和Python中的`zip`效果类似，进行两个数据集进行匹配。

- `take`：从数据集中取$n$个样本构成新的数据集。

- `take_while`：有条件地进行取样本构成新的数据集。

- `filter`：过滤数据集，例如，我们要读取一个目录下的所有文件名，过滤掉不是图像的文件名，可以使用`filter`。

- `reduce`：聚合操作。

- `batch`：将数据集变为批数据集(`BatchDataset`）

- ...