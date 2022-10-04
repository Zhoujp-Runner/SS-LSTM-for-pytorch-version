# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 14:02 2022/9/17

# import cv2
#
# img = cv2.imread("./data/ewap_dataset_full/ewap_dataset/seq_eth/map.png", cv2.IMREAD_COLOR)
# cv2.imshow("img", img)
# cv2.waitKey(0)


import numpy as np
from torch.utils.data import Dataset, DataLoader


filename = "./data/ewap_dataset_full/ewap_dataset/seq_eth/obsmat.txt"


class PersonDataset(Dataset):
    def __init__(self, obs_len, pred_len):
        super(PersonDataset, self).__init__()
        self.filename = filename
        self.ped_num = 500
        self.trajs = []
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.obs_trajs = []
        self.pred_trajs = []
        self.trajs_input = []
        self.expected_output = []
        self.source_data = np.loadtxt(self.filename)
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
                    pred_traj.append(self.trajs[index][j + self.pred_len])
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

    def __getitem__(self, idx):
        self.get_traj_from_txt()
        self.get_obs_pred()
        self.get_traj_input()
        self.get_expected_output()
        return self.trajs_input[idx], self.expected_output[idx]

    def __len__(self):
        return len(self.obs_trajs)


class SocialDataset(PersonDataset):
    def __init__(self, obs_len, pred_len):
        super(SocialDataset, self).__init__(obs_len, pred_len)
        self.social_input = []


if __name__ == '__main__':
    # person = PersonDataset( 1, 1)
    social = SocialDataset(8, 12)
    print(social.expected_output)
    # x, y = person.__getitem__(0)
    # print(x)
    # print(y)
