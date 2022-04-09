#  计算机视觉-Pj1仓库

本项目基于Numpy对神经网络进行模块化，将网络拆分成'损失函数', '优化器', '传递层'模块，方便自定义。
MNIST文件夹包含了各种数据，可以在[Google网盘](https://drive.google.com/drive/folders/10pMCw9CHptZgAzjbEXeUsAhrkz7Yjk6U?usp=sharing)下载。

## 框架

```powershell
|-- Para_seeking.py
|-- readme.md
|-- report.pdf
|-- router.py
|-- 可视化.ipynb
|-- MNIST
|   |-- param.npz
|   |-- plotinfo.npz
|   |-- record.txt
|   |-- testset.npz
|   |-- trainset.npz
|-- package
|   |-- draw.py
|   |-- net.py
|   |-- parameter.py
|   |-- __init__.py
|   |-- layers
|   |   |-- activation.py
|   |   |-- batchnorm.py
|   |   |-- dropout.py
|   |   |-- layer.py
|   |   |-- linear.py
|   |   |-- __init__.py
|   |-- loss
|   |   |-- celoss.py
|   |   |-- mseloss.py
|   |   |-- __init__.py
|   |-- optim
|   |   |-- momentum.py
|   |   |-- rmsprop.py
|   |   |-- sgd.py
|   |   |-- __init__.py
```

## 文件说明

- MNIST文件夹：包含了样本（trainset.npz, testset.npz)，模型参数（param.npz），参数查找记录（record.txt），可视化信息记录（plotinfo.npz）
- package/layers文件夹：传递层模块，存有规范化层、全连接层以及包括sigmoid、ReLU、Tanh等激活函数
- package/loss文件夹：损失函数模块，实现了均方损失和交叉熵损失
- package/loss文件夹：优化器模块，实现了SGD，Momentum，RMSporp优化器
- package/net.py, parameter.py: 网络和参数的类定义
- package/draw.py: 存放了对训练曲线进行可视化的函数
- router.py: 路由模块，实现了模型参数的上下载，训练函数和测试函数等
- Para_seeking.py：参数查找文件
- 可视化.ipynb：对模型性能及参数进行可视化的文件
- report.pdf：实验报告

## 使用步骤(示例)

### 1.加载模块及设置初始参数

注：请提前从网盘上下载好MNIST文件夹并按框架放置好

```python
import numpy as np
import package
import package.optim as optim
import package.loss as loss
import os
import sys

layers = [
    {'type': 'batchnorm', 'shape': 784, 'requires_grad': False, 'affine': False},
    {'type': 'linear', 'shape': (784, 350)}, 
    {'type': 'batchnorm', 'shape': 350},
    {'type': 'relu'},
    {'type': 'linear', 'shape': (350, 10)}, 
]
loss_fn = loss.CrossEntropyLoss()
net = package.Net(layers)
lr = 0.001
batch_size = 128
optimizer = optim.RMSprpo(net.parameters, lr,penalty=1e-6)
train_file = sys.path[0]+'/MNIST/trainset.npz'
param_file = sys.path[0]+'/MNIST/param.npz'
test_file = sys.path[0]+'/MNIST/testset.npz'
```

### 2.调用训练函数

```python
mn.train(net, loss_fn, train_file, batch_size, optimizer,load_file=None, save_path=param_file, times=4,silent=True)
```

### 3.调用测试函数

```python
# 如果单独测试load_file中需要给定模型参数文件
tacc,tloss=mn.test(net,loss_fn,test_file,load_file=param_file)
# 如果紧跟着训练测试可以不给load_file参数
tacc,tloss=mn.test(net,loss_fn,test_file)
```

注：需要注意net的设置要和训练时一致。

### 4.参数查找

运行Para_seeking.py文件即可。由于进行了两次参数查找，第一次的查找设置在文件中被注释掉了，如需复现第一次查找需解除注释。

### 5.可视化

按顺序运行可视化.ipynb中的代码块即可。
