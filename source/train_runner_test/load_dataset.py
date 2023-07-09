import pandas as pd
import cv2
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import torch
from torch.utils.data import Dataset, DataLoader

import config


def train_test_split(csv_path, split):
    df_data = pd.read_csv(csv_path, header=None)
    df_data = df_data.dropna()
    len_data = len(df_data)
    valid_split = int(len_data * split)
    train_split = int(len_data - valid_split)
    training_samples = df_data.iloc[:train_split][:]
    valid_samples = df_data.iloc[-valid_split:][:]
    print(f"Training sample instances: {len(training_samples)}")
    print(f"Validation sample instances: {len(valid_samples)}")
    return training_samples, valid_samples


def AugmentFaceKeypointDataset(training_samples, data_path, aug_data_num):
    data_set_list = []
    for data_num in range(training_samples.shape[0]):
        image = cv2.imread(f"{data_path}/{training_samples.iloc[data_num, 0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = image.shape
        image = cv2.resize(image, (config.RESIZE, config.RESIZE))
        # About keypoint
        keypoints = training_samples.iloc[data_num][1:]
        keypoints = np.array(keypoints, dtype="float32")
        keypoints = keypoints.reshape(-1, 2)
        keypoints_per = keypoints * [1 / (orig_w), 1 / (orig_h)]
        data_set_list.append([image, keypoints_per])

        # データ拡張枚数が0枚の場合はデータ拡張の部分をスキップする
        if aug_data_num == 0:
            continue

        landmark_num = len(keypoints)
        kps = KeypointsOnImage(
            [Keypoint(x=keypoints[i][0], y=keypoints[i][1]) for i in range(0,landmark_num)],
            shape=image.shape,
        )

        # About Augment setting
        seq = iaa.Sequential(
            [
                iaa.Affine(
                    rotate=(-80, 80),# 右、左回りに80度回転させる
                    scale={"x": (0.5, 1.2), "y": (0.5, 1.2)},# x軸y軸それぞれずらす
                ),
                iaa.Fliplr(0.5), # 50%の確率で画像を反転させる
            ]
        )
        for aug_count in range(aug_data_num-1):
            # print("データ拡張を行います")
            image_aug, kps_aug = seq(image=image, keypoints=kps)
            keypoints = []
            for i in range(len(kps.keypoints)):
                before = kps.keypoints[i]
                after = kps_aug.keypoints[i]
                keypoints.append([after.x, after.y])
            keypoints = np.array(keypoints, dtype="float32")
            keypoints_per = keypoints * [1 / (orig_w), 1 / (orig_h)]
            image_after = kps_aug.draw_on_image(image_aug, size=0)
            # データ拡張を行った画像をリストに格納する
            data_set_list.append([image_after, keypoints_per])
    print("len(data_set_list)",len(data_set_list))
    return data_set_list


class FaceKeypointDataset(Dataset):
    def __init__(self, dataset, resize):
        self.data = dataset
        self.resize = resize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]
        # 画素値を0~1に変換
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        keypoints = self.data[index][1]
        keypoint_data = {
            "image": torch.tensor(image, dtype=torch.float),
            "keypoints": torch.tensor(keypoints, dtype=torch.float),
        }
        return keypoint_data
