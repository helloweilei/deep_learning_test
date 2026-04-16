import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import show_images, get_fashion_mnist_labels

# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

# print(len(mnist_train), len(mnist_test))
# print(mnist_train[0][0].shape)

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

