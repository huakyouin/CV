import numpy as np
'''
随机梯度下降的参数优化器
'''
class SGD(object):

    def __init__(self, parameters, lr,penalty, decay=0):
        '''
        类包含以下属性：
        parmaeters--需要学习的参数
        lr--学习率
        decay--权重衰减
        penalty--优化时带有的正则项λ系数
        '''
        self.parameters = [p for p in parameters if p.requires_grad]    # 如果有的参数不需要计算梯度就不加进来了
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.penalty =penalty
        self.eps=1e-8

    def update(self):
        if self.decay_rate < 1 and not p.skip_decay:  p.data *= self.decay_rate
        for p in self.parameters:
            grad= p.grad+self.penalty*p.data/(np.linalg.norm(p.data)+self.eps)  #梯度加上正则项
            p.data -= self.lr * grad
        self.lr*=self.decay_rate
            