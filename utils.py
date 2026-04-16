import torch
import matplotlib.pyplot as plt

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

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    '''绘制图像列表
    Parameters
    ----------
    imgs : list of torch.Tensor
        图像列表，列表中的每个元素都是一个torch.Tensor
    num_rows : int
        图像网格的行数
    num_cols : int
        图像网格的列数
    titles : list of str, optional
        图像标题列表，默认为None
    scale : float, optional
        图像大小缩放比例，默认为1.5
    '''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.numpy()
            if img.ndim == 3:
                # CHW -> HWC
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                else:
                    img = img.transpose((1, 2, 0))
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_fashion_mnist_labels(labels):
    '''返回Fashion-MNIST数据集的文本标签
    Parameters
    ----------
    labels : torch.Tensor
        数字标签
    '''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]         