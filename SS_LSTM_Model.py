# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 21:15 2022/9/17


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.optim import optimizer
from visdom import Visdom

import heapq
from scipy.spatial import distance

# from generate_data import PersonDataset
# from occupancy import SocialDataset
from total_dataset import TotalDataset
from total_dataset import SSLSTMDataset

# #############parameters################# #
observed_frame_num = 8
predicting_frame_num = 12
batch_size = 10
# ######################################## #


# def collate_fn(batch):
#     traj_input, expected_output = zip(*batch)
#     traj_input = torch.LongTensor(traj_input)
#     expected_output = torch.LongTensor(expected_output)
#     return traj_input, expected_output
#
#
# def get_dataset():
#     person_dataset = PersonDataset(observed_frame_num, predicting_frame_num)
#     return person_dataset
#
#
# def get_dataloader(dataset):
#     person_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     return person_dataloader


def calculate_fde(test_label, predicted_output, test_num, show_num):
    """
    计算终点偏移距离
    :param test_label: 测试的标签[ped_num, frame_num, coord]
    :param predicted_output: 预测的输出[ped_num, frame_num, coord]
    :param test_num: 测试数量
    :param show_num: 用于计算最终平均偏移点
    :return:
    """
    total_fde = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        # 计算欧氏距离
        total_fde[i] = distance.euclidean(predicted_result_temp[-1], label_temp[-1])

    # 获取前show_num个最小值
    show_fde = heapq.nsmallest(show_num, total_fde)

    show_fde = np.reshape(show_fde, [show_num, 1])

    return np.average(show_fde)


def calculate_ade(test_label, predicted_output, test_num, predicting_frame_num, show_num):
    """
    计算平均偏移距离
    :param test_label: 测试的标签[ped_num, frame_num, coord]
    :param predicted_output: [ped_num, frame_num, coord]
    :param test_num: 测试数量
    :param predicting_frame_num: 预测的帧数
    :param show_num: 用于计算最终平均偏移点
    :return:
    """
    total_ade = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        ade_temp = 0.0
        for j in range(predicting_frame_num):
            ade_temp += distance.euclidean(predicted_result_temp[j], label_temp[j])
        ade_temp = ade_temp / predicting_frame_num
        total_ade[i] = ade_temp

    show_ade = heapq.nsmallest(show_num, total_ade)

    show_ade = np.reshape(show_ade, [show_num, 1])

    return np.average(show_ade)


def get_data():
    return PersonDataset(observed_frame_num, predicting_frame_num).generate_data()


class PersonModel(nn.Module):
    def __init__(self, hidden_size):
        super(PersonModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=hidden_size),
            nn.ReLU(),
            nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        )

    def forward(self, x):
        input = x.permute(1, 0, 2).float()
        # print(input.dtype)
        out, hidden = self.model(input)
        # print(out.shape)
        # print(hidden.shape)
        return out, hidden


class SocialModel(nn.Module):
    def __init__(self, hidden_size):
        super(SocialModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=16, out_features=hidden_size),
            nn.ReLU(),
            nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        )

    def forward(self, x):
        input = x.permute(1, 0, 2).float()
        out, hidden = self.model(input)
        return out, hidden


class SceneModel(nn.Module):
    def __init__(self, hidden_size):
        super(SceneModel, self).__init__()
        self.cnn = nn.Sequential(
            # [N, C, D, H, W]
            nn.Conv3d(in_channels=3, out_channels=96, kernel_size=(1, 11, 11), stride=(1, 4, 4)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(num_features=96, momentum=0.8),
            nn.Conv3d(in_channels=96, out_channels=256, kernel_size=(1, 5, 5), stride=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(num_features=256, momentum=0.8),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=4)
        self.dense = nn.Sequential(
            nn.Linear(in_features=45056, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=256, hidden_size=hidden_size)

    def forward(self, x):
        x1 = self.cnn(x)
        # print(x1.shape)
        x1 = x1.permute(0, 2, 1, 3, 4)
        x2 = self.flatten(x1)
        x3 = self.dense(x2)
        x3 = x3.permute(1, 0, 2)
        out, hidden = self.gru(x3)
        return out, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=hidden_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, input, hidden_state):
        # [batch_size, 2]==>[1, batch_size, 2]
        input = input.unsqueeze(0)
        out_linear = self.model(input)
        # print(out_linear.shape)
        out, hidden = self.gru(out_linear, hidden_state)
        prediction = self.linear(out.squeeze(0))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder_person = PersonModel(hidden_size=hidden_size)
        self.encoder_social = SocialModel(hidden_size=hidden_size)
        self.encoder_scene = SceneModel(hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size)

    def forward(self, input_person, input_social, input_scene, target):
        """
        Seq2Seq前向传播函数
        :param input_person: person_scale输入[batch_num, frame_num, coord]
        :param input_social:social_scale输入[batch_num, frame_num, occupancy_size]
        :param input_scene:scene_scale输入[batch_num, frame_num, img]
        :param target: 目标输出[batch_num, frame_num, coord]
        :return:
        """
        target = target.permute(1, 0, 2)
        outs = torch.zeros(target.shape)
        out_encoder_person, hidden_person = self.encoder_person(input_person)
        out_encoder_social, hidden_social = self.encoder_social(input_social)
        out_encoder_scene, hidden_scene = self.encoder_scene(input_scene)
        # hidden = torch.concat([hidden_person, hidden_social], dim=0)
        hidden = torch.add(hidden_person, hidden_social)
        hidden = torch.add(hidden, hidden_scene)
        # hidden = torch.add(hidden_person, hidden_social)
        # print(out_encoder.shape)
        # print(hidden_encoder.shape)
        input_person = input_person.permute(1, 0, 2).float()
        input_decoder = input_person[-1, :, :]
        # print(input_decoder.shape)
        for i in range(len(target)):
            out_decoder, hidden = self.decoder(input_decoder, hidden)
            # print(out_decoder.shape)
            outs[i] = out_decoder
            input_decoder = out_decoder
            # print(hidden)
        # print(outs.shape)
        return outs


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train_with_val():
    epochs = 100

    # dataset_person = PersonDataset(observed_frame_num, predicting_frame_num, state='train')
    # dataloader_person = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # dataset_val = TotalDataset(observed_frame_num, predicting_frame_num, state='validation')
    # dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    #
    # dataset = TotalDataset(observed_frame_num, predicting_frame_num, state='train')
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_val = SSLSTMDataset(state='validation')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    dataset = SSLSTMDataset(state='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('dataloader over')

    model = Seq2Seq(128).to(device())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    viz = Visdom(env="SS_LSTM")

    val_loss_epoch = 0
    train_loss_epoch = 0

    for i in range(epochs):
        model.train()
        print("train...")
        train_loss = 0
        for input_person, input_social, input_scene, target in dataloader:
        # for input_person, input_social, target in dataloader:
            # print(target.shape)
            optimizer.zero_grad()

            input_person = input_person.to(device())
            input_social = input_social.to(device())
            input_scene = input_scene.to(device())
            target = target.to(device())

            y_label = target.permute(1, 0, 2)
            # outs = model(input_person, input_social, target)
            outs = model(input_person, input_social, input_scene, target)
            y = outs.to(device())
            loss = criterion(y_label, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print("Realtime loss: {}".format(loss.item()))
        print("train_loss: ", train_loss)
        train_loss_epoch = train_loss/len(dataloader)
        print("Epoch: {} Realtime average loss: {}".format(i, train_loss/len(dataloader)))
        if i % 3 == 0:
            model.eval()
            print("val...")
            val_loss = 0
            with torch.no_grad():
                for in_val_person, in_val_social, in_val_scene, tar_val in dataloader_val:
                # for in_val_person, in_val_social, tar_val in dataloader_val:

                    in_val_person = in_val_person.to(device())
                    in_val_social = in_val_social.to(device())
                    in_val_scene = in_val_scene.to(device())
                    tar_val = tar_val.to(device())

                    y_label_val = tar_val.permute(1, 0, 2)
                    outs_val = model(in_val_person, in_val_social, in_val_scene, tar_val)
                    # outs_val = model(in_val_person, in_val_social, tar_val)
                    y_val = outs_val.to(device())
                    loss_val = criterion(y_label_val, y_val)
                    val_loss += loss_val.item()
                val_loss_epoch = val_loss/len(dataloader_val)
                print("\nVal_loss: ", val_loss/len(dataloader_val))
        x = torch.tensor([i])
        y = torch.tensor([[train_loss_epoch, val_loss_epoch]])
        viz.line(X=x, Y=y, win="Loss_Loss", update='append')


if __name__ == '__main__':
    # person = PersonModel(128)

    # dataset = get_dataset()
    # dataloader = get_dataloader(dataset)

    # input, target = get_data()
    # x = input.permute(1, 0, 2).long()
    # input = input.long()
    # out, hidden = person(input)
    # print(input.shape)
    # print(target.shape)
    # print(x.dtype)

    # decoder = Decoder(128)
    # x = torch.rand(128, 2)
    # y = torch.rand(128, 2)
    # h = torch.rand(1, 128, 128)
    # decoder.forward(x, h)

    # model = Seq2Seq(128)
    # input, target = get_data()
    # model(input, target)
    train_with_val()
    # social = SocialDataset(observed_frame_num, predicting_frame_num)
    # x = torch.rand(2, 8, 16)
    # social_model = SocialModel(128)
    # out, hidden = social_model(x)
    # print(hidden.shape)
    # dataset = SocialDataset(observed_frame_num, predicting_frame_num, state='train')
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # for x, y in dataloader:
    #     print(x.shape)
    #     print(y.shape)
    # x = torch.rand(20, 8, 10)
    # x = x[-1, :, :]
    # print(x.shape)
    # scene = SceneModel(128)
    # x = torch.randn(1, 3, 8, 480, 640)
    # print(scene(x).shape)
    # person = PersonModel(128)
    # x = torch.randn(1, 2)
    # print(person(x))
    # SS_LSTM = Seq2Seq(128)
    # print(SS_LSTM)
    # test_label = np.zeros((1, 1, 2))
    # test_label[0][0][0] = 0
    # test_label[0][0][1] = 1
    # output = np.zeros((1, 1, 2))
    # print(calculate_fde(test_label, output, 1, 1))
