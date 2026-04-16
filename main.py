import torch
import matplotlib.pyplot as plt
import random
from utils import sgd

def main():
    print("Hello from chapter-3!")

def synthetic_data(w, b, num_samples):
    X = torch.normal(0, 1, (num_samples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    '''均方损失
    Parameters
    ----------
    y_hat : torch.Tensor
        预测值
    y : torch.Tensor
        真实值
    '''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def huber_loss(y_hat, y, delta=1.0):
    '''胡伯尔损失
    Parameters
    ----------
    y_hat : torch.Tensor
        预测值
    y : torch.Tensor
        真实值
    '''
    abs_error = torch.abs(y_hat - y.reshape(y_hat.shape))
    return torch.where(abs_error > delta, abs_error - 0.5 * delta, 0.5 * abs_error ** 2 / delta)

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print("features:", features[0], "\nlabel:", labels[0])
    # plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
    # plt.show()

    # for X, y in data_iter(10, features, labels):
    #     print("X:", X, "\ny:", y)
    #     break
    w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.03
    num_epochs = 5
    batch_size = 10
    net = linreg
    loss = huber_loss
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")
