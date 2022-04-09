import numpy as np


class RMSprpo(object):
    def __init__(self, parameters, lr, decay=0, beta=0.98, eps=1e-8,penalty=0):
        self.beta = beta  # 平方梯度过去部分权重
        self.eps = eps
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.parameters = [p for p in parameters if p.requires_grad]
        self.accumulated_grads = [np.zeros(p.data.shape) for p in self.parameters] # 累计平方梯度 
        self.penalty=penalty

    def update(self):
        for p, grad in zip(self.parameters, self.accumulated_grads):
            np.copyto(grad, self.beta * grad + (1 - self.beta) * np.power(p.grad, 2))
            g_with_penalty=p.grad+self.penalty*self.penalty*p.data/(np.linalg.norm(p.data)+self.eps)
            p.data -= self.lr * g_with_penalty / (np.sqrt(grad) + self.eps)
        self.lr*=self.decay_rate