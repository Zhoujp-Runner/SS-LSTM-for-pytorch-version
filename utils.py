# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 14:10 2022/9/17

import numpy as np
import os
import math


def file2matrix(filename):
    data = np.loadtxt(filename)
    # data = np.reshape(data, [-1, 3])
    return data


def get_traj_from_txt(filename, ped_num):
    trajs = []
    for index in range(ped_num):
        data = file2matrix(filename)
        traj = []
        for i in range(len(data)):
            if data[i][1] == index + 1:
                traj.append([data[i][1], data[i][0], data[i][2], data[i][4]])
        traj = np.reshape(traj, [-1, 4])
        trajs.append(traj)
    return trajs


def get_obs_pred(trajs, obs_len, pred_len):
    obs_trajs = []
    pred_trajs = []
    count = 0
    for index in range(len(trajs)):
        if len(trajs[index]) >= obs_len + pred_len:
            obs_traj = []
            pred_traj = []
            count += 1
            for i in range(obs_len):
                obs_traj.append(trajs[index][i])
            for j in range(pred_len):
                pred_traj.append(trajs[index][j + pred_len])
            obs_traj = np.reshape(obs_traj, [obs_len, 4])
            pred_traj = np.reshape(pred_traj, [pred_len, 4])
            obs_trajs.append(obs_traj)
            pred_trajs.append(pred_traj)
    obs_trajs = np.reshape(obs_trajs, [count, obs_len, 4])
    pred_trajs = np.reshape(pred_trajs, [count, pred_len, 4])
    return obs_trajs, pred_trajs


def get_traj_input(obs, obs_len):
    traj_input = []
    for index in range(len(obs)):
        person_in = []
        for i in range(obs_len):
            person_in.append([obs[index][i][-2], obs[index][i][-1]])
        person_in = np.reshape(person_in, [obs_len, 2])
        traj_input.append(person_in)
    traj_input = np.reshape(traj_input, [len(obs), obs_len, 2])
    return traj_input


def get_expected_output(pred, pred_len):
    expected_output = []
    for index in range(len(pred)):
        person_out = []
        for i in range(pred_len):
            person_out.append([pred[index][i][-2], pred[index][i][-1]])
        person_out = np.reshape(person_out, [pred_len, 2])
        expected_output.append(person_out)
    expected_output = np.reshape(expected_output, [len(pred), pred_len, 2])
    return expected_output


if __name__ == '__main__':
    file_test = "./data/ewap_dataset_full/ewap_dataset/seq_eth/obsmat.txt"
    data_test = file2matrix(file_test)
    traje = get_traj_from_txt(file_test, 3)
    obs, pred = get_obs_pred(traje, 2, 6)
    traje_input = get_traj_input(obs, 2)
    expect_output = get_expected_output(pred, 6)
    print(data_test[0])
    print(traje[1][0])
    # print(obs[0][0])
    # print(pred)
    print(traje_input[0])
    print(expect_output[0])
