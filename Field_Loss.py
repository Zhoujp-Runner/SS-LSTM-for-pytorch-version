# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 9:58 2022/10/31

import torch
import torch.nn as nn
from torch import optim


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.Sigmoid(),
            nn.Linear(32, 4),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.model(x)
        return out


class FieldLoss(nn.Module):
    def __init__(self):
        super(FieldLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        loss1 = self.mse(output, target)
        g = torch.tensor(1e-3, dtype=torch.float)
        delta_v = output[1] - output[0]
        loss2 = g * torch.exp(-delta_v)
        # print("loss1 ", loss1)
        # print("loss2 ", loss2)
        return loss1 + loss2


if __name__ == '__main__':
    model = TestModel()
    lossfunc = FieldLoss()
    optimizer = optim.Adam(model.parameters())
    input = torch.rand(100, 4)
    target = torch.rand(100, 4)
    loss_sum = 0
    for i in range(100):
        torch.no_grad()
        input_bs = input[i]
        target_bs = input[i]
        out = model(input_bs)
        loss = lossfunc(out, target_bs)
        optimizer.step()
        loss_sum += loss.item()
    print(loss_sum)
