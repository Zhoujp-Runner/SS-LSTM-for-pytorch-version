# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 18:26 2022/9/18


import cv2
import torch
from generate_data import PersonDataset
import numpy as np
import visdom


video_path = "./data/ewap_dataset_full/ewap_dataset/seq_eth/seq_eth.avi"
h_matrix = np.loadtxt("./data/ewap_dataset_full/ewap_dataset/seq_eth/H.txt")
h_inv = np.linalg.inv(h_matrix)


def get_video():
    cap = cv2.VideoCapture(video_path)
    return cap


def draw_point(image, coord_x, coord_y):
    cv2.circle(image, (coord_x, coord_y), 5, (0, 0, 255), -1)


if __name__ == '__main__':
    video = get_video()
    frame_id = 0
    person = PersonDataset(8, 12)
    data = person.source_data
    i = 0
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
    while video.isOpened():
        ret, img = video.read()
        # print(img.shape)
        for i in range(len(data)):
            data_frame_id = int(data[i][0])
            if frame_id == data_frame_id:
                coord = data[i, 2: 5]
                coord_ = [coord[0], coord[2], 1]
                coord_img = np.dot(h_inv, coord_)
                coord_img /= coord_img[-1]
                draw_point(img, int(coord_img[1]), int(coord_img[0]))
                # print('draw')
                cv2.imshow("img", img)
                cv2.waitKey(25)
                # print(img)
        frame_id += 1
