import torch

class Accumulator:
    '''在n个变量上累加'''
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def sgd(params, lr, batch_size):
    '''小批量随机梯度下降
    Parameters
    ----------
    params : list
        模型参数列表
    lr : float
        学习率
    batch_size : int
        小批量的大小
    '''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

            