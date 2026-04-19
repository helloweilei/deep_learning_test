import torch

from softmax_reg import load_data_fashion_mnist, train_ch3, predict_ch3

num_inputs, num_outputs, num_hiddens = 784, 10, 256
net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_inputs, num_hiddens),
    torch.nn.ReLU(),
    torch.nn.Linear(num_hiddens, num_outputs)
)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

num_epochs, lr = 10, 0.1
loss = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = load_data_fashion_mnist(256)
train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)

predict_ch3(net, test_iter)