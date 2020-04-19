# 目录

[toc]

# 数据集介绍

本次实验所采用的数据集是七月在线课程发布的数据集，主要有训练集和测试集：

| 训练集 | 测试集 |
| :----: | ------ |
|  407M  | 82.9M  |

整体数据集的原始数据分为多个域(field)，每个field下有多个特征属性，统计每个域的大小如下：

```python
[94316, 99781,     6,    23, 34072, 12723, 7, 2]
```

数据集采用**稀疏存储**的方式。为了方便阅读，笔者自己加注了索引编号。

格式采用`label index:value index:value index:value ……`的形式，label表示是否点击，index表示整体filed_size下的索引，value表示对应索引下的值，格式如下：

```python
[1] 0 51 0:1 27:1 28:1 29:1 30:1 31:1 32:1 33:1 34:1 35:1 36:1 37:1 38:1 39:1 40:1 41:1 42:1 43:1 44:1
[2] 0 87 0:1 27:1 28:1 45:1 46:1 47:1 32:1 48:1 49:1 35:1 36:1 50:1 38:1 39:1 40:1 51:1 42:1 52:1 53:1 44:1
[3] 0 33 0:1 27:1 28:1 54:1 55:1 56:1 32:1 57:1 58:1 35:1 36:1 37:1 38:1 39:1 40:1 41:1 42:1 43:1 44:1
[4] 0 65 0:1 27:1 28:1 59:1 60:1 61:1 32:1 62:1 63:1 35:1 36:1 37:1 38:1 39:1 40:1 41:1 42:1 64:1 43:1 44:1
[5] 0 238 0:1 27:1 28:1 65:1 66:1 67:1 32:1 68:1 69:1 70:1 71:1 50:1 38:1 72:1 40:1 51:1 42:1 73:1 74:1 75:1 76:1 43:1 77:1 78:1 79:1 52:1 80:1 81:1 82:1 83:1 84:1 85:1
[6] 0 65 0:1 27:1 28:1 86:1 87:1 88:1 32:1 62:1 63:1 35:1 36:1 37:1 38:1 39:1 40:1 41:1 42:1 43:1 44:1
```

# 稀疏数据处理

为了降低数据存储的复杂度，本实验采用稀疏数据表示，经过coo_matrix函数处理得到如下格式（个人感觉是共现矩阵），如下是第4094和4095条样本的特征数据：

```python
  (4094, 606)	1
  (4094, 1102)	1
  (4094, 4253)	1
  (4094, 5102)	1
  (4094, 5103)	1
  (4094, 5104)	1
  (4095, 0)	1
  (4095, 27)	1
  (4095, 28)	1
  (4095, 30)	1
  (4095, 40)	1
  (4095, 51)	1
  (4095, 70)	1
  (4095, 71)	1
  (4095, 72)	1
  (4095, 122)	1
  (4095, 131)	1
  (4095, 145)	1
  (4095, 151)	1
  (4095, 208)	1
  (4095, 4253)	1
  (4095, 4265)	1
  (4095, 4349)	1
  (4095, 5105)	1
  (4095, 5106)	1
```

所以filed的特征属性维度为：**560869**

# FM 模型代码

```python
class FM(Model):
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,l2_w=0, l2_v=0, random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('v', [input_dim, factor_order], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w'] # [input_dim, 1]
            v = self.vars['v'] # [input_dim, factor_order]
            b = self.vars['b'] # [1]

            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X))) # [n, input_dim] 
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v)) # [n, input_dim] * [input_dim, factor_order] = n*factor_order
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 1), # 按行来求和
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b + p, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
```

根据FM的前向传播公式：
$$
y = w_{0} + \sum^{n}_{i=1}w_{i}x_{i} + \sum^{n-1}_{i=1}\sum^{n}_{j=i+1}w_{ij}x_{i}x_{j}
$$
将最后一项通过矩阵分解转换为如下的公式：
$$
\sum^{n-1}_{i=1}\sum^{n}_{j=i+1}w_{ij}x_{i}x_{j} = \sum^{n-1}_{i=1}\sum^{n}_{j=i+1}<v_{i},v_{j}>x_{i}x_{j} = \frac{1}{2}\sum^{k}_{f=1}((\sum^{n}_{i=1}v_{i,f}x_{i})^{2}-\sum^{n}_{i=1}v^{2}_{i,f}x^{2}_{i})
$$
该实验运行的官方计算图如下：

![graph_run=](/Users/blackzero/Documents/Master/Study/推荐系统/深度推荐指北/FM/images/graph_run=.png)

上图看的不是很清楚，笔者自己画一个核心前向传播图：

![FM前向传播模型计算图](/Users/blackzero/Documents/Master/Study/推荐系统/深度推荐指北/FM/images/FM前向传播模型计算图.jpg)

