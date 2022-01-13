import torch
import torch.nn as nn
import pandas
from TableData import TableData

data = TableData()
data.load_data("../../data/classificationA/train0.csv", feature_names = ['Outcome','Glucose'], label='Outcome')


class FeatureLayer(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(FeatureLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, out_dim, bias = True)

    def forward(self, data):
        out = self.fc1(data)
        return out


class Model(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.connect = None

    def forward(self, remote):

        self.connect = remote
        connect = torch.cat(tuple(remote[key] for key in remote.keys()), 1)
        return self.fc1(connect)

class ModelLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias = None):
        ctx.save_for_backward(input, wieght, bias)


class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = torch.exp(i)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

exp = Exp()
x1 = torch.randn(100, 20)
x2 = torch.randn(100, 20)
w = 0.01*torch.rand(20, 1)

bias = 0.01*torch.rand(100, 1)
y = torch.mm(x2, w)+torch.mm(x1, w)+bias

Net1 = FeatureLayer(20, 20)
Net2 = FeatureLayer(20, 20)
Net3 = Model(40, 1)
loss = torch.nn.MSELoss()

remote = {}
optim1 = torch.optim.Adam(Net1.parameters(), lr=0.01, weight_decay=1e-4)
optim2 = torch.optim.Adam(Net2.parameters(), lr=0.01, weight_decay=1e-4)
optim3 = torch.optim.Adam(Net3.parameters(), lr=0.01, weight_decay=1e-4)
import pickle
for i in range(200):
    out1 = Net1(x1)
    out2 = Net2(x2)
    r1 = out1.clone().detach().requires_grad_()
    r2 = out2.clone().detach().requires_grad_()
    out11 = pickle.dumps(r1)
    out22 = pickle.dumps(r2)

    out11 = pickle.loads(out11)
    out22 = pickle.loads(out22)
    remote['model1'] = out11
    remote['model2'] = out22

    #meta_output = out1.clone().detach().requires_grad_()
    optim1.zero_grad()
    optim2.zero_grad()
    optim3.zero_grad()
    out3 = Net3(remote)
    loss_val = loss(out3, y)
    loss_val.backward()
    print(loss_val)

    r1 = pickle.dumps(remote['model1'].grad)
    r2 = pickle.dumps(remote['model2'].grad)

    #out1.backward(remote['model1'].grad)
    #out2.backward(remote['model2'].grad)

    r3 = pickle.loads(r1)
    r4 = pickle.loads(r2)
    out1.backward(r3)
    out2.backward(r4)

    optim1.step()
    optim2.step()
    optim3.step()
    print('compute part of the gradient')
