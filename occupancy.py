# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 15:31 2022/9/30


import numpy as np
import math

import torch
from generate_data import PersonDataset


h_matrix = np.loadtxt("./data/ewap_dataset_full/ewap_dataset/seq_eth/H.txt")
filename = "./data/ewap_dataset_full/ewap_dataset/seq_eth/obsmat.txt"
h_inv = np.linalg.inv(h_matrix)


def cal_angle(current_x, current_y, other_x, other_y):
    p0 = [other_x, other_y]
    p1 = [current_x, current_y]
    p2 = [current_x + 0.1, current_y]
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle_degree = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return angle_degree


def get_rectangular_occupancy_map(frame_ID, ped_ID, neighborhood_size, grid_size, data):
    """
    得到矩形占用图，简单来说就是在图上以目标行人为中心，画矩形网格，统计网格中其他行人的个数
    :param frame_ID: 当前帧的ID
    :param ped_ID: 目标行人ID
    :param neighborhood_size: 需要多大的网格框，单位为米
    :param grid_size:每个网格的长度，单位为米
    :param data:输入的数据，原始数据[frame_ID, ped_ID, x, z, y, vx, vz, vy]
    :return:一张矩形占用图
    """
    # 初始化变量
    o_map = np.zeros((int(neighborhood_size / grid_size), int(neighborhood_size / grid_size)))
    ped_list = []
    width_low = 0
    width_high = 0
    height_low = 0
    height_high = 0

    # 在一帧中搜索所有行人
    for i in range(len(data)):
        if data[i][0] == frame_ID:
            ped_list.append([data[i][0], data[i][1], data[i][2], data[i][4]])
    # print(ped_list)
    # print(len(ped_list))
    # [num of ped, [frame_ID, ped_ID, x, y]],
    ped_list = np.reshape(ped_list, [-1, 4])
    # print(ped_list)
    if len(ped_list) == 0:
        print("no pedestrian in this frame!")
    elif len(ped_list) == 1:
        print("only one pedestrian in this frame!")
        return o_map
    else:
        for ped_Index in range(len(ped_list)):
            if ped_list[ped_Index][1] == ped_ID:
                current_x, current_y = ped_list[ped_Index][2], ped_list[ped_Index][3]
                # 源码中的w和h并没有在for循环外声明，出于之前写c的原因，我就在外面加了声明
                width_low, width_high = current_x - neighborhood_size/2, current_x + neighborhood_size/2
                height_low, height_high = current_y - neighborhood_size/2, current_y + neighborhood_size/2
                # print("width: ({}, {}) height: ({}, {})".format(width_low, width_high, height_low, height_high))
        for other_Index in range(len(ped_list)):
            if ped_list[other_Index][1] != ped_ID:
                other_x, other_y = ped_list[other_Index][2], ped_list[other_Index][3]
                if other_x >= width_high or other_x <= width_low or other_y >= height_high or other_y <= height_low:
                    continue
                cell_x = int(np.floor((other_x - width_low) / grid_size))
                cell_y = int(np.floor((other_y - height_low) / grid_size))
                # print((other_x - width_low) / grid_size)
                # print("other_ID: {} x: {} y: {}".format(ped_list[other_Index][1], cell_x, cell_y))

                # 源码中这里是o_map[cell_x, cell_y] += 1
                # 但是放到像素图上，x是横向坐标，代表着列，所以我认为改成y，x更加合理
                o_map[cell_y, cell_x] += 1
    return o_map


def get_circle_occupancy_map(frame_ID, ped_ID, frame_size, neighborhood_radius, grid_radius, grid_angle, data):
    """
    得到圆形占用图，简单来说就是在图上以目标行人为中心，画一个圆形，统计其他行人的个数
    :param frame_ID: 当前帧的ID
    :param ped_ID: 目标行人ID
    :param frame_size: [width, height]
    :param neighborhood_radius: 圆形占用图的半径，单位为米
    :param grid_radius:每个网格的径向长度，单位为米
    :param grid_angle:每个网格的周向角度，单位为pi，能被2整除
    :param data:输入的数据，原始数据[frame_ID, ped_ID, x, z, y, vx, vz, vy]
    :return:一个圆形占用图矩阵
    """
    # 初始化变量
    circle_map = np.zeros((int(neighborhood_radius / grid_radius), int(2 / grid_angle)))
    grid_angle_pi = grid_angle * math.pi
    grid_angle_360 = grid_angle * 180
    ped_list = []
    current_x = 0
    current_y = 0

    width, height = frame_size[0], frame_size[1]
    neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
    grid_bound = grid_radius / (min(width, height) * 1.0)

    # 在一帧中搜索所有行人
    for i in range(len(data)):
        if data[i][0] == frame_ID:
            # coord = [data[i][2], data[i][4], 1]
            # coord_pixel = np.dot(h_inv, coord)
            # coord_pixel /= coord_pixel[-1]
            ped_list.append([data[i][0], data[i][1], data[i][2], data[i][4]])
            # ped_list.append([data[i][0], data[i][1], coord_pixel[0], coord_pixel[1]])
    # print(ped_list)
    # print(len(ped_list))
    # [num of ped, [frame_ID, ped_ID, x, y]],
    ped_list = np.reshape(ped_list, [-1, 4])
    print(ped_list)
    if len(ped_list) == 0:
        print("no pedestrian in this frame!")
    elif len(ped_list) == 1:
        print("only one pedestrian in this frame!")
        return circle_map
    else:
        for ped_Index in range(len(ped_list)):
            if ped_list[ped_Index][1] == ped_ID:
                current_x, current_y = ped_list[ped_Index][2], ped_list[ped_Index][3]
        for other_Index in range(len(ped_list)):
            if ped_list[other_Index][1] != ped_ID:
                other_x, other_y = ped_list[other_Index][2], ped_list[other_Index][3]
                distance = math.sqrt((other_x - current_x)**2 + (other_y - current_y)**2)
                if distance >= neighborhood_radius:
                    continue
                # 计算的占用图矩阵的列数在圆形占用图上是按顺时针递增（参照钟表）
                angle = math.atan2((other_x - current_x), (other_y - current_y))
                if angle < 0:  # 位于目标行人的占用图左半侧
                    angle = 2 * math.pi + angle
                print(angle/math.pi)
                print(angle/grid_angle_pi)
                cell_x = int(np.floor(angle / grid_angle_pi))
                # 占用图矩阵的行数就是由内到外递增
                cell_y = int(np.floor(distance / grid_radius))

                # angle = cal_angle(current_x, current_y, other_x, other_y)
                # if distance >= neighborhood_bound:
                #     continue
                # cell_x = int(np.floor(distance / grid_bound))
                # cell_y = int(np.floor(angle / grid_angle_360))

                print("other_ID: {} x: {} y: {}".format(ped_list[other_Index][1], cell_x, cell_y))

                circle_map[cell_y, cell_x] += 1
    return circle_map


# def get_log_occupancy_map(frame_ID, ped_ID, neighborhood_radius, grid_radius, grid_angle, data):
#     """
#     得到圆形占用图，简单来说就是在图上以目标行人为中心，画一个圆形，统计其他行人的个数
#     :param frame_ID: 当前帧的ID
#     :param ped_ID: 目标行人ID
#     :param neighborhood_radius: 圆形占用图的半径，单位为米
#     :param grid_radius:每个网格的径向长度，单位为米
#     :param grid_angle:每个网格的周向角度，单位为pi，能被2整除
#     :param data:输入的数据，原始数据[frame_ID, ped_ID, x, z, y, vx, vz, vy]
#     :return:一个圆形占用图矩阵
#     """
#     # 初始化变量
#     circle_map = np.zeros((int(neighborhood_radius / grid_radius), int(2 / grid_angle)))
#     grid_angle = grid_angle * math.pi
#     ped_list = []
#     current_x = 0
#     current_y = 0
#
#     # 在一帧中搜索所有行人
#     for i in range(len(data)):
#         if data[i][0] == frame_ID:
#             ped_list.append([data[i][0], data[i][1], data[i][2], data[i][4]])
#     # print(ped_list)
#     # print(len(ped_list))
#     # [num of ped, [frame_ID, ped_ID, x, y]],
#     ped_list = np.reshape(ped_list, [-1, 4])
#     print(ped_list)
#     if len(ped_list) == 0:
#         print("no pedestrian in this frame!")
#     elif len(ped_list) == 1:
#         print("only one pedestrian in this frame!")
#         return circle_map
#     else:
#         for ped_Index in range(len(ped_list)):
#             if ped_list[ped_Index][1] == ped_ID:
#                 current_x, current_y = ped_list[ped_Index][2], ped_list[ped_Index][3]
#         for other_Index in range(len(ped_list)):
#             if ped_list[other_Index][1] != ped_ID:
#                 other_x, other_y = ped_list[other_Index][2], ped_list[other_Index][3]
#                 distance = math.sqrt((other_x - current_x)**2 + (other_y - current_y)**2)
#                 if distance >= neighborhood_radius:
#                     continue
#                 # 计算的占用图矩阵的列数在圆形占用图上是按顺时针递增（参照钟表）
#                 angle = math.atan2((other_x - current_x), (other_y - current_y))
#                 if angle < 0:  # 位于目标行人的占用图左半侧
#                     angle = 2 * math.pi + angle
#                 print(angle/math.pi)
#                 print(angle/grid_angle)
#                 cell_x = int(np.floor(angle / grid_angle))
#                 # 占用图矩阵的行数就是由内到外递增
#                 cell_y = int(np.floor(distance / grid_radius))
#                 print("other_ID: {} x: {} y: {}".format(ped_list[other_Index][1], cell_x, cell_y))
#
#                 circle_map[cell_y, cell_x] += 1
#     return circle_map


# def generate_social_input():
#     source_data = np.loadtxt(filename)
#     social_input = []
#
#     for i in range(len(source_data)):
#         frame_ID


class SocialDataset(PersonDataset):
    def __init__(self, obs_len, pred_len, state='train'):
        super(SocialDataset, self).__init__(obs_len, pred_len, state)
        self.social_rectangular_input = []
        self.social_rectangular_output = []
        self.rectangular_neighborhood_size = 16
        self.rectangular_grid_size = 4
        self.len = len(self.obs_trajs)

        self.generate_social_input_from_obs_trajs()
        self.generate_social_input_from_pred_trajs()
        self.generate_social_data()
        # self.len = len(self.social_rectangular_input)

    def generate_social_input_from_obs_trajs(self):
        """
        通过输入符合obs_len和pred_len的轨迹，得到占用图，计算社会尺度的输入数据
        社会尺度的输入数据：[ped_num, frame_for_each_person, occupancy]
        :return:
        """
        for ped_num in range(len(self.obs_trajs)):
            ped_ID = self.obs_trajs[ped_num][0][0]
            social_rectangular = []
            for frame_num in range(self.obs_len):
                frame_ID = self.obs_trajs[ped_num][frame_num][1]
                rect_occupancy = get_rectangular_occupancy_map(frame_ID, ped_ID,
                                                               self.rectangular_neighborhood_size,
                                                               self.rectangular_grid_size,
                                                               self.source_data)
                rect_occupancy_flatten = np.reshape(rect_occupancy, [-1, ])
                social_rectangular.append(rect_occupancy_flatten)
            social_rectangular = np.reshape(social_rectangular, [self.obs_len, -1])

            self.social_rectangular_input.append(social_rectangular)

        self.social_rectangular_input = np.reshape(self.social_rectangular_input, [-1, self.obs_len, int(self.rectangular_neighborhood_size / self.rectangular_grid_size)**2])

    def generate_social_input_from_pred_trajs(self):
        """
        通过输入符合obs_len和pred_len的轨迹，得到占用图，计算社会尺度的期望输出数据
        社会尺度的输入数据：[ped_num, frame_for_each_person, occupancy]
        :return:
        """
        for ped_num in range(len(self.pred_trajs)):
            ped_ID = self.pred_trajs[ped_num][0][0]
            social_rectangular = []
            for frame_num in range(self.pred_len):
                frame_ID = self.pred_trajs[ped_num][frame_num][1]
                rect_occupancy = get_rectangular_occupancy_map(frame_ID, ped_ID,
                                                               self.rectangular_neighborhood_size,
                                                               self.rectangular_grid_size,
                                                               self.source_data)
                rect_occupancy_flatten = np.reshape(rect_occupancy, [-1, ])
                social_rectangular.append(rect_occupancy_flatten)
            social_rectangular = np.reshape(social_rectangular, [self.pred_len, -1])

            self.social_rectangular_output.append(social_rectangular)

        self.social_rectangular_output = np.reshape(self.social_rectangular_input,
                                                    [-1,
                                                     self.pred_len,
                                                     int(self.rectangular_neighborhood_size / self.rectangular_grid_size)**2])

    def generate_social_data(self):
        """
        生成训练数据
        :return:
        """
        if self.state == 'train':
            self.social_rectangular_input = self.social_rectangular_input[:int(self.len/8)*5]
            self.social_rectangular_output = self.social_rectangular_output[:int(self.len / 8) * 5]
            self.len = len(self.social_rectangular_input)
        elif self.state == 'validation':
            self.social_rectangular_input = self.social_rectangular_input[int(self.len/8)*5:int(self.len/8)*7]
            self.social_rectangular_output = self.social_rectangular_output[int(self.len / 8) * 5:int(self.len / 8) * 7]
            self.len = len(self.social_rectangular_input)

    def __getitem__(self, item):
        x = self.social_rectangular_input[item]
        x = np.reshape(x, [self.obs_len, int(self.rectangular_neighborhood_size / self.rectangular_grid_size)**2])
        y = self.social_rectangular_output[item]
        y = np.reshape(y, [self.pred_len, int(self.rectangular_neighborhood_size / self.rectangular_grid_size)**2])
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y

    def __len__(self):
        return self.len


if __name__ == '__main__':
    # person = PersonDataset(8, 12)
    # data = person.source_data

    # o_map = get_rectangular_occupancy_map(858, 2, 16, 4, data)
    # o_map = np.reshape(o_map, [-1, ])
    # print(o_map)
    # print(o_map)
    # print(math.atan2(1, -math.sqrt(3)))
    # print(math.pi/6)
    # circle_map = get_circle_occupancy_map(858, 2, [640, 480], 8, 2, 0.5, data)
    # circle_map = get_circle_occupancy_map(858, 2, [640, 480], 8, 2, 0.5, data)
    # print(circle_map)
    social = SocialDataset(8, 12, 'train')
    # social.generate_social_input_from_obs_trajs()
    # print(social.social_rectangular_input)
    # social.generate_social_input_from_pred_trajs()
    print(social.__len__())
