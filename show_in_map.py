# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 18:26 2022/9/18


import cv2
import torch
from generate_data import PersonDataset
import numpy as np
import visdom


video_path = "./data/ewap_dataset_full/ewap_dataset/seq_eth/seq_eth.avi"
outs_path = "../SSLSTMdata/outs_with_ped_id.npy"
source_path = "./data/ewap_dataset_full/ewap_dataset/seq_eth/obsmat.txt"

h_matrix = np.loadtxt("./data/ewap_dataset_full/ewap_dataset/seq_eth/H.txt")
h_inv = np.linalg.inv(h_matrix)
outs = np.load(outs_path, allow_pickle=True)  # [[ped_id, [prediction_trajectories]]]
source_data = np.loadtxt(source_path)
observed_frame_num = 8
predicting_frame_num = 12


def get_video():
    cap = cv2.VideoCapture(video_path)
    return cap


def draw_point(image, coord_x, coord_y, flag):
    if flag:
        cv2.circle(image, (coord_x, coord_y), 5, (0, 0, 255), -1)
    else:
        cv2.circle(image, (coord_x, coord_y), 5, (0, 255, 0), -1)


def get_prediction_position(index):
    """

    :param index: outs中第几个行人
    :return: ground_truth, predictions : [[ped_id, frame, coord_x, coord_y]]
    """
    # print(outs)
    ped_id = outs[index][0].numpy()
    prediction_positions = outs[index][1]
    # print(prediction_positions)

    # 查找第一帧预测帧
    first_prediction_frame = 0
    first_frame = 0
    for item in source_data:
        if item[1] == ped_id:
            first_frame = item[0]
            print(first_frame)
            first_prediction_frame = item[0] + observed_frame_num * 6  # 6是由采样频率决定的
            break

    count = 0
    ground_truth = []
    predictions = []
    for item in source_data:
        frame = first_prediction_frame + count * 6
        if item[1] == ped_id and item[0] == frame and count < predicting_frame_num:
            ground_truth.append([ped_id, frame, item[2], item[4]])
            predictions.append([ped_id, frame, prediction_positions[count][0], prediction_positions[count][1]])
            count += 1
    ground_truth = np.reshape(ground_truth, [predicting_frame_num, -1])
    predictions = np.reshape(predictions, [predicting_frame_num, -1])

    # print(ped_id)
    # print(prediction_positions)
    print(len(ground_truth))
    print(predictions)
    return ground_truth, predictions


def show():
    video = get_video()
    frame_id = 0
    person = PersonDataset(8, 12)
    data = person.source_data
    gt, pre = get_prediction_position(0)
    ped_id = gt[0][0]
    count = 0
    draw_frame = gt[count][1]

    while video.isOpened():
        ret, img = video.read()
        # print(img.shape)
        if frame_id == draw_frame and count < predicting_frame_num:
            coord_gt = [gt[count][2], gt[count][3], 1]
            coord_pr = [pre[count][2], pre[count][3], 1]
            coord_img_gt = np.dot(h_inv, coord_gt)
            coord_img_pr = np.dot(h_inv, coord_pr)
            coord_img_gt /= coord_img_gt[-1]
            coord_img_pr /= coord_img_pr[-1]
            draw_point(img, int(coord_img_gt[1]), int(coord_img_gt[0]), True)
            draw_point(img, int(coord_img_pr[1]), int(coord_img_pr[0]), False)
            count += 1
            if count < predicting_frame_num:
                draw_frame = gt[count][1]
            print("............ped draw..........")
            print(frame_id)
            cv2.imshow("img", img)
            cv2.waitKey(200)
        # for i in range(len(gt)):
        #     data_frame_id = int(data[i][0])
        #     if frame_id == data_frame_id:
        #         coord = data[i, 2: 5]
        #         coord_ = [coord[0], coord[2], 1]
        #         coord_img = np.dot(h_inv, coord_)
        #         coord_img /= coord_img[-1]
        #         draw_point(img, int(coord_img[1]), int(coord_img[0]))
        #         # print('draw')
        #         cv2.imshow("img", img)
        #         cv2.waitKey(25)
        # print(img)
        frame_id += 1


if __name__ == '__main__':

    # print(np.linalg.inv(h_matrix))
    # ret, img = video.read()
    # img_torch = torch.FloatTensor(img)
    # img_numpy = img_torch.numpy()
    # img_numpy = img_numpy.astype(np.uint8)
    # print(img_numpy.dtype)
    # print('to_numpy: \n', img_numpy, '\n')
    # print('numpy: \n', img)
    # print('to_numpy: \n', type(img_numpy), '\n')
    # print('numpy: \n', type(img))
    # cv2.imshow('xxx', img_numpy)
    # cv2.imshow('zzz', img)
    # print(img.shape)  # [height, width, channels]
    # cv2.waitKey(0)
    # print(type(img))
    # print(img)
    # print("Image size: ", img.shape)
    # coord = data[2, 2: 5]
    # coord_ = [coord[0], coord[2], 1]
    # # print("[x, z, y]: ", np.dot(h_inv, coord_))
    # coord_pixel = np.dot(h_inv, coord_)
    # coord_pixel /= coord_pixel[-1]
    # print(coord_pixel)


    # a = np.array([1, 2, 3, 4])
    # b = np.array([[[1, 2],
    #                [2, 3]],
    #               [[3, 4],
    #                [4, 5]],
    #               [[5, 6],
    #                [6, 7]],
    #               [[7, 8],
    #                [8, 9]]])
    # c = []
    # for i in range(4):
    #     c.append([a[i], b[i]])
    # print(c)
    # c = np.reshape(c, [4, -1])
    # print(c)

    # get_prediction_position(0)
    show()
