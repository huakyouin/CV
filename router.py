import numpy as np
import package
import package.optim as optim
import package.loss as loss
import os
import sys
from tqdm import tqdm


def save(parameters, save_path):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i].data
    np.savez(save_path, **dic)
    
def load(parameters, file):
    params = np.load(file)
    for i in range(len(parameters)):
        parameters[i].data = params[str(i)]

def read_weight(parameters):
    w=[]
    for i in range(len(parameters)):
        w.append(parameters[i].data)
    return w


def load_MNIST(file, transform=False):
    file = np.load(file)
    X = file['X']
    Y = file['Y']
    if transform:
        X = X.reshape(len(X), -1)
    return X, Y


def train(net, loss_fn, train_file, batch_size, optimizer, load_file=None,\
     save_path=None, times=1,go_on=False,silent=False):
    '''
    net--网络结构
    loss_fn--损失函数
    test_file--测试集
    batch_szie--测试的批次
    optimizer--参数优化器
    load_file--训练好的模型参数
    save_path--保存参数路径
    times--轮次
    go_on--是否接着load_file中的参数继续
    '''
    X, Y = load_MNIST(train_file, transform=True)   # 加载训练样本和真实标签
    data_size = X.shape[0]   # 样本数
    ## 如果属于继续训练就加载参数
    if go_on and load_file is not None and os.path.isfile(load_file): load(net.parameters, load_file) 
    ## 根据times轮次设置循环数
    for loop in range(times):  
        i = 0 # i 为样本中该轮已经训练过的数量
        ## 按batch_size分批训练
        while i <= data_size - batch_size:
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            i += batch_size
            ## 前向传播
            output = net.forward(x)  
            ## 根据损失函数计算损失            
            batch_acc, batch_loss = loss_fn(output, y)  
            # print(batch_loss)
            ## 梯度反向传播
            eta = loss_fn.gradient()
            net.backward(eta)
            ## 用优化器进行参数学习
            optimizer.update()
            ## 每50次迭代打印以下训练信息
            if i % 50 == 0 and not silent:
                print("loop: %d, batch: %5d, batch acc: %2.1f, batch loss: %.2f" % \
                    (loop+1, i, batch_acc*100, batch_loss))
    ## 如果传入保存路径，就对参数进行保存
    if save_path is not None: save(net.parameters, save_path)



def test(net, loss_fn, test_file=None, load_file=None,silent=False,get=None):
    '''
    net--网络结构
    loss_fn--损失函数
    test_file--测试集
    load_file--训练好的模型参数
    silent--是否打印结果
    get--直接拿到样本变量
    '''
    if get is not None: X=get[0];Y=get[1]
    if test_file is not None and get is None:X, Y = load_MNIST(test_file, transform=True)   # 加载训练样本和真实标签
    if load_file is not None: load(net.parameters, load_file)
    ## 前向传播
    output = net.forward(X)  
    ## 根据损失函数计算损失            
    acc, loss = loss_fn(output, Y) 
    if not silent: print('测试集准确率:%.2f%%'%(acc*100))
    return acc,loss
    

if __name__ == "__main__": 
    layers = [
        {'type': 'batchnorm', 'shape': 784, 'requires_grad': False, 'affine': False},
        {'type': 'linear', 'shape': (784, 400)},
        {'type': 'batchnorm', 'shape': 400},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (400, 100)},
        {'type': 'batchnorm', 'shape': 100},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (100, 10)}
    ]
    loss_fn = loss.CrossEntropyLoss()
    net = package.Net(layers)
    lr = 0.001
    batch_size = 128
    optimizer = optim.RMSprpo(net.parameters, lr,penalty=1e-6)
    train_file = sys.path[0]+'/MNIST/trainset.npz'
    param_file = sys.path[0]+'/MNIST/param.npz'
    test_file = sys.path[0]+'/MNIST/testset.npz'
    train(net, loss_fn, train_file, batch_size, optimizer, None, save_path=None, times=4)
    test(net, loss_fn, test_file,None)
