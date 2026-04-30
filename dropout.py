import torch
from torch import nn
from softmax_reg import train_ch3
from vision import load_data_fashion_mnist

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 0:
        return torch.zeros_like(X)
    if dropout == 1:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return X * mask / (1 - dropout)

num_input, num_output, num_hidden1, num_hidden2 = 784, 10, 256, 256

class Net(torch.nn.Module):
    def __init__(self, num_input, num_output, num_hidden1, num_hidden2, is_training = True):
        super(Net, self).__init__()
        self.num_input = num_input
        self.is_training = is_training
        self.line1 = nn.Linear(num_input, num_hidden1)
        self.line2 = nn.Linear(num_hidden1, num_hidden2)
        self.line3 = nn.Linear(num_hidden2, num_output)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.line1(X.reshape((-1, self.num_input))))
        if self.is_training == True:
            H1 = dropout_layer(H1, 0.5)
        H2 = self.relu(self.line2(H1))
        if self.is_training == True:
            H2 = dropout_layer(H2, 0.2)

        return self.line3(H2)

net = Net(num_input, num_output, num_hidden1, num_hidden2)
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduce="none")
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
