import torch
from main import synthetic_data

def load_array(data_arrays, batch_size, is_train=True):
    '''加载数据
    Parameters
    ----------
    data_arrays : list of torch.Tensor
        数据列表，列表中的每个元素都是一个torch.Tensor
    batch_size : int
        小批量的大小
    is_train : bool, optional
        是否为训练数据，默认为True
    '''
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
data_iter = load_array((features, labels), batch_size)

net = torch.nn.Sequential(torch.nn.Linear(2, 1))
loss = torch.nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        train_l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {train_l:f}")

# print(next(iter(data_iter)))