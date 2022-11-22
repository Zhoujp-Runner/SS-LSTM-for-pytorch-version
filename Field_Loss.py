# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 9:58 2022/10/31

import torch
import torch.nn as nn
from torch import optim

import math
import numpy as np


# #################para#################### #
mass = 60  # 行人质量，单位kg
# #################para#################### #


def get_field_energy(ped_ID, frame_ID, source_data):
    pass


def calculate_energy(cur_ped_state, neighbour_state):
    """

    :param cur_ped_state: [coord_x, coord_y, v_x, v_y, angle], angle measured in radians
    :param neighbour_state: [[coord_x, coord_y, v_x, v_y]]
    :return: energy vector : np.ndarray[energy_x, energy_y]
    """
    energy = np.array([0, 0])
    for item in neighbour_state:  # item : [coord_x, coord_y, v_x, v_y]
        speed = math.sqrt(item[2] ** 2 + item[3] ** 2)
        equal_mass = mass * (1.566 * 1e-14 * math.pow(speed, 6.687) + 0.3345)
        x_ = \
            math.cos(cur_ped_state[4]) * (cur_ped_state[0] - item[0]) + \
            math.sin(cur_ped_state[4]) * (cur_ped_state[1] - item[1])
        y_ = \
            - math.sin(cur_ped_state[4]) * (cur_ped_state[0] - item[0]) + \
            math.cos(cur_ped_state[4]) * (cur_ped_state[1] - item[1])
        position_energy = 1 / (x_ ** 2 + y_ ** 2)
        energy_i = equal_mass * position_energy

        # 以目标行人为中心建立局部坐标系,借此来表示向量
        neighbour_coord = [item[0] - cur_ped_state[0], item[1] - cur_ped_state[1]]
        neighbour_angle = math.atan2(neighbour_coord[1], neighbour_coord[0])

        # 这里先不添加负号，正常来说斥力场应该是负方向的，但是目前只需用它来计算大小，所以先不添加
        energy_i_vector = np.array([energy_i * math.cos(neighbour_angle), energy_i * math.sin(neighbour_angle)])
        energy = energy + energy_i_vector
    return energy


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
        print("loss ", loss1 + loss2)
        return loss1 + loss2


if __name__ == '__main__':
    # model = TestModel()
    # lossfunc = FieldLoss()
    # optimizer = optim.Adam(model.parameters())
    # input = torch.randn(100, 4)
    # target = torch.ones(100, 4)
    # loss_sum = 0
    # for i in range(100):
    #     # torch.no_grad()
    #     input_bs = input[i]
    #     target_bs = input[i]
    #     out = model(input_bs)
    #     loss = lossfunc(out, target_bs)
    #     loss.backward()
    #     optimizer.step()
    #     loss_sum += loss.item()
    # print(loss_sum)
    a = np.array([1, 2])
    b = np.array([2, 3])
    print(a + b)
