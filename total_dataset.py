# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 20:09 2022/10/1

import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

# from generate_data import PersonDataset
# from occupancy import SocialDataset
import occupancy


filename = "./data/ewap_dataset_full/ewap_dataset/seq_eth/obsmat.txt"
video_path = "./data/ewap_dataset_full/ewap_dataset/seq_eth/seq_eth.avi"


class TotalDataset(Dataset):
    def __init__(self, obs_len, pred_len, state='train'):
        super(TotalDataset, self).__init__()
        # ###############Person############## #
        self.filename = filename
        self.ped_num = 0
        self.trajs = []
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.obs_trajs = []
        self.pred_trajs = []
        self.trajs_input = []
        self.expected_output = []
        self.len = 0
        self.state = state
        # ###############Person############## #

        # ###############Social############### #
        self.social_rectangular_input = []
        # self.social_rectangular_output = []
        self.rectangular_neighborhood_size = 16
        self.rectangular_grid_size = 4
        # ###############Social############### #

        # ###############Scene############### #
        self.video_path = video_path
        self.scene_input = []
        self.img_list = []
        self.img_shape = [480, 640, 3]  # [height, width, channels]
        # ###############Scene############### #

        self.source_data = np.loadtxt(self.filename)
        person_id = self.source_data[:, 1]
        self.ped_num = np.size(np.unique(person_id))
        self.get_traj_from_txt()
        self.get_obs_pred()
        self.get_traj_input()
        self.get_expected_output()
        self.len = len(self.trajs_input)

        self.generate_social_input_from_obs_trajs()
        # self.generate_social_input_from_pred_trajs()

        self.get_img_match_to_trajs_frame()

        self.generate_data()

        # print(person_id)
        # print(self.ped_num)
        # print(np.unique(person_id))
        # print(self.source_data)

    def get_traj_from_txt(self):
        """
        将txt文件夹中的数据[frame_ID, person_ID, coord_x, coord_z, coord_y, v_x, v_z, v_y]
        转换成[person_ID, frame_ID, coord_x, coord_y]的格式
        并且,将shape形式从[8908, 4]转换成[ped_unm, frame_for_each_person_num, 4]的形式
        :return:
        """
        for index in range(self.ped_num):
            traj = []
            for i in range(len(self.source_data)):
                if self.source_data[i][1] == index + 1:
                    traj.append([self.source_data[i][1], self.source_data[i][0], self.source_data[i][2], self.source_data[i][4]])
            traj = np.reshape(traj, [-1, 4])
            self.trajs.append(traj)
        # print(self.ped_num/8)
        # print(len(self.trajs))

    def get_obs_pred(self):
        """
        根据观测路径长度以及预测路径长度删选出合适的行人，并将其分别划分到两个列表中
        :return:
        """
        count = 0
        for index in range(len(self.trajs)):
            if len(self.trajs[index]) >= self.obs_len + self.pred_len:
                obs_traj = []
                pred_traj = []
                count += 1
                for i in range(self.obs_len):
                    obs_traj.append(self.trajs[index][i])
                for j in range(self.pred_len):
                    pred_traj.append(self.trajs[index][j + self.obs_len])
                obs_traj = np.reshape(obs_traj, [self.obs_len, 4])
                pred_traj = np.reshape(pred_traj, [self.pred_len, 4])
                self.obs_trajs.append(obs_traj)
                self.pred_trajs.append(pred_traj)
        self.obs_trajs = np.reshape(self.obs_trajs, [count, self.obs_len, 4])
        self.pred_trajs = np.reshape(self.pred_trajs, [count, self.pred_len, 4])

    def get_traj_input(self):
        """
        将观测列表中的最后一维变为[coord_x, coord_y]的形式
        :return:
        """
        for index in range(len(self.obs_trajs)):
            person_in = []
            for i in range(self.obs_len):
                person_in.append([self.obs_trajs[index][i][-2], self.obs_trajs[index][i][-1]])
            person_in = np.reshape(person_in, [self.obs_len, 2])
            self.trajs_input.append(person_in)
        self.trajs_input = np.reshape(self.trajs_input, [len(self.obs_trajs), self.obs_len, 2])
        # print(len(self.trajs_input))
        # return len(self.trajs_input)

    def get_expected_output(self):
        """
        将观测列表和期望输出列表中的最后一维变为[coord_x, coord_y]的形式
        :return:
        """
        for index in range(len(self.pred_trajs)):
            person_out = []
            for i in range(self.pred_len):
                person_out.append([self.pred_trajs[index][i][-2], self.pred_trajs[index][i][-1]])
            person_out = np.reshape(person_out, [self.pred_len, 2])
            self.expected_output.append(person_out)
        self.expected_output = np.reshape(self.expected_output, [len(self.pred_trajs), self.pred_len, 2])

    # def __getitem__(self, idx):
    #     self.get_traj_from_txt()
    #     self.get_obs_pred()
    #     self.get_traj_input()
    #     self.get_expected_output()
    #     return self.trajs_input[idx], self.expected_output[idx]
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
                rect_occupancy = occupancy.get_rectangular_occupancy_map(frame_ID, ped_ID,
                                                                         self.rectangular_neighborhood_size,
                                                                         self.rectangular_grid_size,
                                                                         self.source_data)
                rect_occupancy_flatten = np.reshape(rect_occupancy, [-1, ])
                social_rectangular.append(rect_occupancy_flatten)
            social_rectangular = np.reshape(social_rectangular, [self.obs_len, -1])

            self.social_rectangular_input.append(social_rectangular)

        self.social_rectangular_input = np.reshape(self.social_rectangular_input, [-1, self.obs_len, int(self.rectangular_neighborhood_size / self.rectangular_grid_size)**2])

    def get_img_match_to_trajs_frame(self):
        """
        生成场景尺度的输入 shape:[ped_num, frame_num_for_each_person, img]
        :return:
        """
        video = cv2.VideoCapture(self.video_path)
        frame_index = 0

        while True:
            if not video.isOpened():
                print('video failed to open!')
                break
            ret, img = video.read()
            if img is None:
                print('img is None, video is might ending...')
                break
            self.img_list.append([frame_index, img])
            # print(frame_index)
            frame_index += 1
        # print(len(img_list))
        # print(img_list[804][1])

        # 将数据集中涉及到的帧数给提取出来
        for ped_num in range(len(self.obs_trajs)):
            person_img_list = []
            for frame_num in range(self.obs_len):
                frame_id = int(self.obs_trajs[ped_num][frame_num][1])
                person_img_list.append(self.img_list[frame_id][1])
            self.scene_input.append(person_img_list)

        # self.scene_input = np.reshape(self.scene_input, [-1, self.obs_len, self.img_shape[0], self.img_shape[1], self.img_shape[2]])
        self.scene_input = np.array(self.scene_input)

    # def generate_social_input_from_pred_trajs(self):
    #     """
    #     通过输入符合obs_len和pred_len的轨迹，得到占用图，计算社会尺度的期望输出数据
    #     社会尺度的输入数据：[ped_num, frame_for_each_person, occupancy]
    #     :return:
    #     """
    #     for ped_num in range(len(self.pred_trajs)):
    #         ped_ID = self.pred_trajs[ped_num][0][0]
    #         social_rectangular = []
    #         for frame_num in range(self.pred_len):
    #             frame_ID = self.pred_trajs[ped_num][frame_num][1]
    #             rect_occupancy = occupancy.get_rectangular_occupancy_map(frame_ID, ped_ID,
    #                                                            self.rectangular_neighborhood_size,
    #                                                            self.rectangular_grid_size,
    #                                                            self.source_data)
    #             rect_occupancy_flatten = np.reshape(rect_occupancy, [-1, ])
    #             social_rectangular.append(rect_occupancy_flatten)
    #         social_rectangular = np.reshape(social_rectangular, [self.pred_len, -1])
    #
    #         self.social_rectangular_output.append(social_rectangular)
    #
    #     self.social_rectangular_output = np.reshape(self.social_rectangular_input,
    #                                                 [-1,
    #                                                  self.pred_len,
    #                                                  int(self.rectangular_neighborhood_size / self.rectangular_grid_size)**2])

    def generate_data(self):
        """
        生成训练数据
        :return:
        """
        if self.state == 'train':
            self.trajs_input = self.trajs_input[:int(self.len/8)*5]
            self.expected_output = self.expected_output[:int(self.len / 8) * 5]
            self.social_rectangular_input = self.social_rectangular_input[:int(self.len / 8) * 5]
            self.scene_input = self.scene_input[:int(self.len / 8) * 5]
            self.len = len(self.trajs_input)
        elif self.state == 'validation':
            self.trajs_input = self.trajs_input[int(self.len/8)*5:int(self.len/8)*7]
            self.expected_output = self.expected_output[int(self.len / 8) * 5:int(self.len / 8) * 7]
            self.social_rectangular_input = self.social_rectangular_input[int(self.len / 8) * 5:int(self.len / 8) * 7]
            self.scene_input = self.scene_input[int(self.len / 8) * 5:int(self.len / 8) * 7]
            self.len = len(self.expected_output)
        self.trajs_input = torch.FloatTensor(self.trajs_input)
        self.expected_output = torch.FloatTensor(self.expected_output)
        self.social_rectangular_input = torch.FloatTensor(self.social_rectangular_input)
        self.scene_input = torch.FloatTensor(self.scene_input)

    def __len__(self):
        # input, output = self.generate_data()
        return self.len

    def __getitem__(self, item):
        # input, target = self.generate_data()
        person_input = self.trajs_input[item, :, :]
        social_input = self.social_rectangular_input[item, :, :]
        scene_input = self.scene_input[item]
        scene_input = scene_input.permute(3, 0, 1, 2)
        target = self.expected_output[item, :, :]
        return person_input, social_input, scene_input, target
        # return person_input, social_input,  target


if __name__ == '__main__':
    dataset = TotalDataset(8, 12, 'train')
    x, y, z, n = dataset.__getitem__(0)
    print(z.shape)
    # cv2.imshow('imshow', z)
    # z_np = z[0].numpy()
    # z_np = z_np.astype(np.uint8)
    # print(z_np)
    # cv2.imshow('xxx', z_np)
    # cv2.imshow('zzz', dataset.img_list[804][1])
    # cv2.waitKey(0)
    # cv2.waitKey(0)
    # dataset.get_img_match_to_trajs_frame()
    # print(len(dataset.scene_input))
    # print(dataset.obs_trajs)
    # for i in range(10):
    #     print(i)
