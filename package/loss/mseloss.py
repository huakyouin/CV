import numpy as np

# 均方误差损失函数--学习速度相比交叉熵慢，因此优先使用交叉熵
class MSELoss(object):
    def gradient(self):
        # 返回损失关于输出的梯度
        # \frac{\partial L}{\partial a_{ij}}=\frac{a_{ij}-y_{ij}}{C}
        return self.u *2
    
    def __call__(self, a, y, requires_acc=True):
        '''
        a: 批量的样本输出
        y: 批量的样本真值
        return: 该批样本的平均损失

        输出与真值的shape是一样的，并且都是批量的，单个输出与真值是一维向量
        a.shape = y.shape = (N, C)      N是该批样本的数量，C是单个样本最终输出向量的长度
        '''
        # u_{ij} = a_{ij} - y_{ij}
        self.u = a - y
        # 样本的平均损失
        loss=np.einsum('ij,ij->', self.u, self.u, optimize=True) /(y.size)
        if requires_acc:
            acc = np.argmax(a, axis=-1) == np.argmax(y, axis=-1)
            return acc.mean(), loss
        return loss