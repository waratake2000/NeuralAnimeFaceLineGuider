import pandas as pd
import cv2
import numpy as np
import json

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import torch
from torch.utils.data import Dataset, DataLoader

import config

def cocokeypoints_list_converter(annotation_coco):
    with open(annotation_coco) as f:
        loader = json.load(f)
    dataset_list = []
    for image_metadata in loader["images"]:
        one_aanotation_data = [image_metadata["file_name"]]
        data_id = int(image_metadata["id"])
        for anotation_data in loader["annotations"]:
            if anotation_data['image_id'] == data_id:
                keypoints_annotation = [keypoint for keypoint in anotation_data["keypoints"] if not keypoint == float(2)]
                one_aanotation_data += keypoints_annotation
                dataset_list.append(one_aanotation_data)
    return dataset_list


def train_test_split(annotation_list, split):
    # df_data = pd.read_csv(annotation_list, header=None)
    df_data = pd.DataFrame(annotation_list)
    df_data = df_data.dropna()
    len_data = len(df_data)
    valid_split = int(len_data * split)
    train_split = int(len_data - valid_split)
    training_samples = df_data.iloc[:train_split][:]
    valid_samples = df_data.iloc[-valid_split:][:]
    print(f"Training sample instances: {len(training_samples)}")
    print(f"Validation sample instances: {len(valid_samples)}")
    return training_samples, valid_samples


def AugmentFaceKeypointDataset(training_samples, data_path, aug_data_num, mix_augmentent = (3,5)):

    data_set_list = []
    resize_seq = iaa.Resize({"height": config.RESIZE, "width": config.RESIZE})
    # data_num = 1
    print("shape[0]とは",training_samples.shape[0])

    # print("len type",type(len(training_samples)))
    total_data_count = int(training_samples.shape[0]) * int(aug_data_num)
    print("総データ拡張数: ",total_data_count)
    for data_num in range(training_samples.shape[0]):

        image = cv2.imread(f"{data_path}/{training_samples.iloc[data_num, 0]}")
        image_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ =  image_orig.shape
        # サイズ変更
        image = cv2.resize(image_orig, (512, 512))

        keypoints = training_samples.iloc[data_num][1:]
        keypoints = np.array(keypoints, dtype="float32")
        keypoints = keypoints.reshape(-1, 2)
        keypoints_per = keypoints * [1 / (orig_w), 1 / (orig_h)]


        # print("オリジナル画像")
        if aug_data_num == 0:
            continue

        landmark_num = len(keypoints)
        kps = KeypointsOnImage(
            [Keypoint(x=keypoints[i][0], y=keypoints[i][1]) for i in range(0,landmark_num)],
            shape=image.shape,
        )

        image_resize, kps_resize = resize_seq(image=image, keypoints=kps)
        data_set_list.append([image_resize, keypoints_per])

        seq = iaa.SomeOf(mix_augmentent, [
            iaa.Add((-40, 40), per_channel=0.5),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
            iaa.Cutout(nb_iterations=(10, 70),size=0.05,cval=(0, 255),fill_per_channel=0.5),
            iaa.JpegCompression(compression=(80, 99)),
            iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13)),

            iaa.Grayscale(alpha=(0.5, 1.0)),
            iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
            iaa.Affine(scale={"x": (0.7, 1.1), "y": (0.7, 1.1)}),
            iaa.Affine(rotate=(-45, 45)),

            iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255)),
            iaa.ShearX((-20, 20)),
            iaa.ShearY((-20, 20)),
            iaa.PiecewiseAffine(scale=(0.03, 0.03)),
            iaa.imgcorruptlike.Spatter(severity=2),
            iaa.Superpixels(p_replace=0.1, n_segments=100)
        ],random_order=True)

        for aug_count in range(aug_data_num-1):
            image_aug, kps_aug = seq(image=image_resize, keypoints=kps_resize)
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
            total_data_count -= 1
            print("残りのデータ拡張数: ",total_data_count)
    return data_set_list


class FaceKeypointDataset(Dataset):
    def __init__(self, dataset, resize):
        self.data = dataset
        self.resize = resize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]
        orig_h, orig_w, _ = image.shape
        # image = cv2.resize(image, (self.resize, self.resize))

        # print(orig_h,orig_w)
        # 画素値を0~1に変換
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        # print(image.shape)
        keypoints = self.data[index][1]
        keypoint_data = {
            "image": torch.tensor(image, dtype=torch.float),
            "keypoints": torch.tensor(keypoints, dtype=torch.float),
        }

        return keypoint_data
