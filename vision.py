import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

# print(len(mnist_train), len(mnist_test))
# print(mnist_train[0][0].shape)

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
            # 图片张量转换为numpy数组，并调整维度顺序为（高度，宽度，通道）
            ax.imshow(img.numpy().transpose((1, 2, 0)))
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# X, y = next(iter(torch.utils.data.DataLoader(mnist_train, batch_size=18)))
# show_images(X, 2, 9, titles=get_fashion_mnist_labels(y))
# plt.show()

# batch_size = 256
# train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

num_workers = 1

def load_data_fashion_mnist(batch_size, resize=None):
    '''下载Fashion-MNIST数据集，然后将其加载到内存中
    Parameters
    ----------
    batch_size : int
        小批量的大小
    resize : tuple of int, optional
        图像大小，默认为None，即不调整图像大小
    '''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False))

def test():
    train_iter, test_iter = load_data_fashion_mnist(18)
    X, y = next(iter(train_iter))
    show_images(X, 2, 9, titles=get_fashion_mnist_labels(y))
    plt.show()

# test()

