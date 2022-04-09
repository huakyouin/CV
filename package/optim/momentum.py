import numpy as np
'''
带动量的参数优化器
'''

class Momentum(object):
    def __init__(self, parameters, lr, penalty=0,decay=0, beta=0.9,alpha=1):
        self.lr = lr   # 学习率
        self.decay_rate = 1.0 - decay    # 学习率衰减因子
        self.alpha=alpha        # 学习率衰减因子
        self.beta = beta        # 动量系数 0.9
        self.penalty=penalty    # 正则项系数
        self.parameters = [p for p in parameters if p.requires_grad]
        self.accmulated_grads = [np.zeros(p.data.shape) for p in self.parameters]

    def update(self):
        lr = self.lr
        for p, grad0 in zip(self.parameters, self.accmulated_grads):
            if self.decay_rate < 1 and not p.skip_decay:  p.data *= self.decay_rate
            g_withPenalty= p.grad+self.penalty*p.data/(np.linalg.norm(p.data)+1e-8)
            p.data -= lr * g_withPenalty
            p.data += self.beta* grad0
            np.copyto(grad0,self.beta* grad0 -lr* g_withPenalty) #内存上优于赋值语句
        self.lr*=self.decay_rate