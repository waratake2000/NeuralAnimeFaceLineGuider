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
    # first_column = training_samples.iloc[:, 0]
    data_set_list = []
    # data_num = 1
    for data_num in range(training_samples.shape[0]):
        image = cv2.imread(f"{data_path}/{training_samples.iloc[data_num, 0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = image.shape
        image = cv2.resize(image, (config.RESIZE, config.RESIZE))
        # print(type(image))
        # 画像を表示

        # About keypoint
        keypoints = training_samples.iloc[data_num][1:]
        keypoints = np.array(keypoints, dtype="float32")
        keypoints = keypoints.reshape(-1, 2)
        # keypoints = keypoints * [config.RESIZE / (orig_w), config.RESIZE / (orig_h)]
        keypoints_per = keypoints * [1 / (orig_w), 1 / (orig_h)]
        # print(keypoints)
        # keypoints = keypoints * [config.RESIZE / orig_w, config.RESIZE / orig_h]

        data_set_list.append([image, keypoints_per])

        # データ拡張枚数が0枚の場合はデータ拡張の部分をスキップする
        if aug_data_num == 0:
            continue

        # kps = KeypointsOnImage(
        #     [
        #         Keypoint(x=keypoints[0][0], y=keypoints[0][1]),
        #         Keypoint(x=keypoints[1][0], y=keypoints[1][1]),
        #         Keypoint(x=keypoints[2][0], y=keypoints[2][1]),
        #         Keypoint(x=keypoints[3][0], y=keypoints[3][1]),
        #         Keypoint(x=keypoints[4][0], y=keypoints[4][1]),
        #         Keypoint(x=keypoints[5][0], y=keypoints[5][1]),
        #         Keypoint(x=keypoints[6][0], y=keypoints[6][1]),
        #         Keypoint(x=keypoints[7][0], y=keypoints[7][1]),
        #         Keypoint(x=keypoints[8][0], y=keypoints[8][1]),
        #     ],
        #     shape=image.shape,
        # )

        landmark_num = len(keypoints)
        kps = KeypointsOnImage(
            [Keypoint(x=keypoints[i][0], y=keypoints[i][1]) for i in range(0,landmark_num)],
            shape=image.shape,
        )

        # print("kps",kps)

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
                # print(after.x)
            keypoints = np.array(keypoints, dtype="float32")
            keypoints_per = keypoints * [1 / (orig_w), 1 / (orig_h)]
            # print(keypoints)

            image_after = kps_aug.draw_on_image(image_aug, size=0)

            # データ拡張を行った画像をリストに格納する
            data_set_list.append([image_after, keypoints_per])

            # 描画
            # plt.imshow(image_after)
            # plt.scatter(keypoints[:,0], keypoints[:,1], color='r')
            # plt.show()
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
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # orig_h, orig_w, _ = image.shape
        # image = cv2.resize(image, (self.resize, self.resize))

        # print(orig_h,orig_w)
        # 画素値を0~1に変換
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        # print(image.shape)
        keypoints = self.data[index][1]
        # keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        # ランドマークの座標を画像のサイズで割ることで、割合に変換している
        # keypoints[:, 0] *= (self.resize / orig_w)
        # keypoints[:, 1] *= (self.resize / orig_h)
        # print(keypoints)

        keypoint_data = {
            "image": torch.tensor(image, dtype=torch.float),
            "keypoints": torch.tensor(keypoints, dtype=torch.float),
        }

        return keypoint_data
