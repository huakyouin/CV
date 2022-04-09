import numpy as np
import package
import package.optim as optim
import package.loss as loss
import sys
import router as mn
import package.draw as draw
import itertools

## 设置网络
def quick_net(a):
    layers = [
            {'type': 'batchnorm', 'shape': 784, 'requires_grad': False, 'affine': False},
            {'type': 'linear', 'shape': (784, a)},  # 输入->第一层隐含层
            {'type': 'batchnorm', 'shape': a},
            {'type': 'sigmoid'},
            {'type': 'linear', 'shape': (a, 10)}, # 最后一层->输出
        ]
    net = package.Net(layers)
    return net

## 文件路径
train_file = sys.path[0]+'/MNIST/trainset.npz'
param_file = sys.path[0]+'/MNIST/param.npz'
test_file = sys.path[0]+'/MNIST/testset.npz'
record_file = sys.path[0]+'/record.txt'

## 设置损失函数
loss_fn = loss.CrossEntropyLoss()
## 设置批大小
batch_size = 128

'''
# 可变参数范围
nhiddens=[100,128,150,200,300]
lrs=[1e-5,1e-4,1e-3,1e-2]
penaltys=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]

## 开始参数查找
name=[]
recordacc=[]
for i,j,k in itertools.product(nhiddens,lrs,penaltys):
    print('隐含层单元数：%d，学习率：%e，正则化强度：%e'%(i,j,k))
    net=quick_net(i);lr=j;penalty=k
    optimizer = optim.RMSprpo(net.parameters, lr,penalty)
    mn.train(net, loss_fn, train_file, batch_size, optimizer,\
        load_file=None, save_path=None, times=2,silent=True)
    tacc,tloss=mn.test(net,loss_fn,test_file,None)
    name.append([i,j,k])
    recordacc.append(tacc)
    with open(record_file,'a',encoding='utf-8') as f:#使用with open()新建对象f
       f.write('%d,%e,%e,%.2f\n'%(i,j,k,tacc))
'''
nhiddens=[200,300,350,400]
lrs=[1e-3]
penaltys=[5e-8,1e-7,5e-7,1e-6,5e-6]

name=[]
recordacc=[]
for i,j,k in itertools.product(nhiddens,lrs,penaltys):
    print('隐含层单元数：%d，学习率：%e，正则化强度：%e'%(i,j,k))
    net=quick_net(i);lr=j;penalty=k
    optimizer = optim.RMSprpo(net.parameters, lr,penalty)
    mn.train(net, loss_fn, train_file, batch_size, optimizer,\
        load_file=None, save_path=None, times=2,silent=True)
    tacc,tloss=mn.test(net,loss_fn,test_file,None)
    name.append([i,j,k])
    recordacc.append(tacc)
    with open(record_file,'a',encoding='utf-8') as f:#使用with open()新建对象f
       f.write('%d,%e,%e,%.2f\n'%(i,j,k,tacc))

## 输出最优模型
max_v =  max(recordacc) # 返回最大值
max_index = recordacc.index(max_v)# 最大值的索引
max_name=name[max_index]
print('最优参数组合为{:s},精度为{:.2f}%'.format(str(max_name),(max_v*100)))



