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
source_data = np.loadtxt("./data/ewap_dataset_full/ewap_dataset/seq_eth/obsmat.txt")
# #################para#################### #


def get_neighbour(ped_ID, obs_len, source_data):
    """
    势场只加入损失函数中，使用的是最后一帧图像的势场,所以要获得最后帧图像的邻居行人
    :param ped_ID:
    :param obs_len:观测帧的长度
    :param source_data: [frame_id, ped_id, coord_x, p_x, p_z, p_y, v_x, v_z, v_y]
    :return:
    """
    # 找到目标行人的最后一帧观测帧
    last_obs_frame = 0
    for item in source_data:
        if item[1] == ped_ID:
            last_obs_frame = item[0]
            print(item[0])
            break
    last_obs_frame = last_obs_frame + (obs_len - 1) * 6
    print(last_obs_frame)

    neighbour = []
    for item in source_data:
        if item[0] == last_obs_frame and item[1] != ped_ID:
            neighbour.append([item[2], item[4], item[5], item[7]])

    # #################test############## #
    # cur = []
    # las = []
    # for item in source_data:
    #     if item[0] == last_obs_frame - 6 and item[1] == ped_ID:
    #         las.append([item[2], item[4], item[5], item[7]])
    #
    # # print(las.__len__())
    # for item in source_data:
    #     if item[0] == last_obs_frame and item[1] == ped_ID:
    #         angle = math.atan2((item[4] - las[0][1]), (item[2] - las[0][0]))
    #         cur = [item[2], item[4], item[5], item[7]]

    return neighbour


def calculate_energy(cur_ped_state, neighbour_state, angle):
    """

    :param cur_ped_state: [coord_x, coord_y, v_x, v_y]
    :param neighbour_state: [[coord_x, coord_y, v_x, v_y]]
    :param angle: current angle measured in radians
    :return: energy vector : np.ndarray[energy_x, energy_y]
    """
    energy = np.array([0, 0])

    if neighbour_state.__len__ == 0:
        return energy

    for item in neighbour_state:  # item : [coord_x, coord_y, v_x, v_y]
        speed = math.sqrt(item[2] ** 2 + item[3] ** 2)
        equal_mass = mass * (1.566 * 1e-14 * math.pow(speed, 6.687) + 0.3345)
        x_ = \
            math.cos(angle) * (cur_ped_state[0] - item[0]) + \
            math.sin(angle) * (cur_ped_state[1] - item[1])
        y_ = \
            - math.sin(angle) * (cur_ped_state[0] - item[0]) + \
            math.cos(angle) * (cur_ped_state[1] - item[1])
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
    def __init__(self, last_input):
        super(FieldLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.last_position = last_input  # 初始化为最后一观测帧的位置

    def forward(self, output, target, ped_id, obs_len):
        # 均方根损失
        loss1 = self.mse(output, target)
        # 求势场大小
        angle = math.atan2((output[1] - self.last_position[1]), (output[0] - self.last_position[0]))
        neighbours = get_neighbour(ped_id, obs_len, source_data)
        field_vecotr = calculate_energy(output, neighbours, angle)
        field_energy = np.linalg.norm(field_vecotr)
        # 总损失
        loss = loss1 + field_energy
        # g = torch.tensor(1e-3, dtype=torch.float)
        # delta_v = output[1] - output[0]
        # loss2 = g * torch.exp(-delta_v)
        self.last_position = output  # update the last_position
        return loss


if __name__ == '__main__':
    model = TestModel()
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
    # a = np.array([1, 2])
    # b = np.array([2, 3])
    # print(a + b)
