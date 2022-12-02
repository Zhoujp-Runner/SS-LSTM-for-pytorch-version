# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 9:58 2022/10/31

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

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
            # print(item[0])
            break
    last_obs_frame = last_obs_frame + (obs_len - 1) * 6
    # print(last_obs_frame)

    neighbour = []
    for item in source_data:
        if item[0] == last_obs_frame and item[1] != ped_ID:
            angle = math.atan2(item[7], item[5])
            neighbour.append([item[2], item[4], item[5], item[7], angle])

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
    neighbour = torch.tensor(neighbour, dtype=torch.float)

    return neighbour


def calculate_energy(ped_state, neighbour_state) -> torch.Tensor:
    """

    :param ped_state: shape:[pred_num, 2] | last dim:[coord_x, coord_y]
    :param neighbour_state: [[coord_x, coord_y, v_x, v_y, angle]]
                            angle: neighbour's current angle of speed measured in radians
    :return: energy vector : np.ndarray[energy_x, energy_y]
    """

    if neighbour_state.__len__ == 0:
        return torch.tensor([0, 0], dtype=torch.float)

    pred_num = len(ped_state)
    energy_for_each_fram = torch.FloatTensor(pred_num)
    index = 0
    for cur_ped_state in ped_state:
        energy = torch.tensor([0, 0], dtype=torch.float)
        for item in neighbour_state:  # item : [coord_x, coord_y, v_x, v_y]
            speed = torch.sqrt(torch.pow(item[2], 2) + torch.pow(item[3], 2))
            # equal_mass = mass * (1.566 * 1e-14 * torch.pow(speed, 6.687) + 0.3345)
            equal_mass = mass
            x_ = \
                torch.cos(item[4]) * (cur_ped_state[0] - item[0]) + \
                torch.sin(item[4]) * (cur_ped_state[1] - item[1])
            y_ = \
                - torch.sin(item[4]) * (cur_ped_state[0] - item[0]) + \
                torch.cos(item[4]) * (cur_ped_state[1] - item[1])

            position_energy = 1 / (x_ ** 6 + y_ ** 6 + 6.0)

            energy_i = equal_mass * position_energy

            # 以目标行人为中心建立局部坐标系,借此来表示向量
            neighbour_coord = [item[0] - cur_ped_state[0], item[1] - cur_ped_state[1]]
            neighbour_angle = torch.atan2(neighbour_coord[1], neighbour_coord[0])

            # 这里先不添加负号，正常来说斥力场应该是负方向的，但是目前只需用它来计算大小，所以先不添加
            energy_i_vector = torch.tensor([energy_i * torch.cos(neighbour_angle), energy_i * torch.sin(neighbour_angle)])
            energy = energy + energy_i_vector
            # print(energy)
        energy_norm = torch.norm(energy)
        # print(energy_norm)
        energy_for_each_fram[index] = energy_norm
        index += 1

    return energy_for_each_fram


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
        # self.last_position = last_input  # 初始化为最后一观测帧的位置

    def forward(self, output, target, ped_id, obs_len):
        """

        :param output: shape[pre_len, batch_size, 2]
        :param target: shape[pre_len, batch_size, 2]
        :param ped_id:
        :param obs_len:
        :return:
        """
        # 均方根损失
        loss1 = self.mse(output, target)
        print("loss1: ", loss1)
        # 求势场大小
        # angle = math.atan2((output[1] - self.last_position[1]), (output[0] - self.last_position[0]))
        output = output.permute(1, 0, 2)
        target = target.permute(1, 0, 2)
        field_energy = 0
        for index in range(len(ped_id)):
            neighbours = get_neighbour(ped_id[index], obs_len, source_data)
            field_vector = calculate_energy(output[index], neighbours)
            field_energy_i = torch.mean(field_vector)  # 一个目标的所有预测点的势场均值
            field_energy += field_energy_i
        field_energy_mean = field_energy / len(ped_id)  # 所有目标的势场均值
        print("loss2: ", field_energy_mean)
        # 总损失
        loss = loss1 + field_energy_mean
        print("total_loss: ", loss)
        # g = torch.tensor(1e-3, dtype=torch.float)
        # delta_v = output[1] - output[0]
        # loss2 = g * torch.exp(-delta_v)
        # self.last_position = output  # update the last_position
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
    # a = torch.tensor(7, dtype=torch.float, requires_grad=True)
    # b = torch.tensor([5], dtype=torch.float, requires_grad=True)
    # c = a + b
    # c = torch.mean(c)
    # print(c)
    a = torch.tensor([[1, 2], [2, 3], [2, 4]], dtype=torch.float)
    b = torch.tensor([[3, 4, 5, 6, 2], [4, 3, 2, 1, 3]], dtype=torch.float)
    e = calculate_energy(a, b)
    print(e)
    print(e.dtype)
    print(torch.norm(e))

