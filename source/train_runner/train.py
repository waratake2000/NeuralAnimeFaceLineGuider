import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import argparse
import inspect
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import datetime as dt
import time

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import psutil
import GPUtil as GPU
import csv

from device_info_writer import all_device_info_csv_writer
import config


plt.style.use("ggplot")

def import_class_from_file(dir_path):
    module_name = str(dir_path).replace("./models/", "").replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, dir_path)
    print(spec)

    # モジュールを作成してロードします
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # モジュール内のクラスのリストを取得します
    classes = [member for member in inspect.getmembers(module, inspect.isclass) if member[1].__module__ == module_name]
    if not classes:
        raise Exception(f"No classes found in {module_name}")
    # 最初のクラスを取得します
    class_ = classes[0][1]

    return class_

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
        # print(type(image))
        # 画像を表示

        # About keypoint
        keypoints = training_samples.iloc[data_num][1:]
        keypoints = np.array(keypoints, dtype="float32")
        keypoints = keypoints.reshape(-1, 2)
        keypoints = keypoints * [config.RESIZE / orig_w, config.RESIZE / orig_h]

        data_set_list.append([image, keypoints])

        # データ拡張枚数が0枚の場合はデータ拡張の部分をスキップする
        if aug_data_num == 0:
            continue

        kps = KeypointsOnImage(
            [
                Keypoint(x=keypoints[0][0], y=keypoints[0][1]),
                Keypoint(x=keypoints[1][0], y=keypoints[1][1]),
                Keypoint(x=keypoints[2][0], y=keypoints[2][1]),
                Keypoint(x=keypoints[3][0], y=keypoints[3][1]),
                Keypoint(x=keypoints[4][0], y=keypoints[4][1]),
                Keypoint(x=keypoints[5][0], y=keypoints[5][1]),
                Keypoint(x=keypoints[6][0], y=keypoints[6][1]),
                Keypoint(x=keypoints[7][0], y=keypoints[7][1]),
                Keypoint(x=keypoints[8][0], y=keypoints[8][1]),
            ],
            shape=image.shape,
        )

        # About Augment setting
        seq = iaa.Sequential(
            [
                iaa.ShearX((-30, 30)),
                iaa.Multiply((0.8, 1.3)),  # change brightness, doesn't affect keypoints
                iaa.Affine(
                    rotate=(-50, 50),
                    scale={"x": (0.5, 1.2), "y": (0.5, 1.2)},
                    cval=(10, 255),
                ),
                iaa.Invert(0.05, per_channel=0.5),
                iaa.Fliplr(0.5),
                iaa.RemoveSaturation((0, 0.5)),
                iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                iaa.UniformColorQuantizationToNBits(),
                iaa.AdditiveGaussianNoise(scale=[0, 10]),
            ]
        )

        for aug_count in range(aug_data_num):
            image_aug, kps_aug = seq(image=image, keypoints=kps)
            keypoints = []
            for i in range(len(kps.keypoints)):
                before = kps.keypoints[i]
                after = kps_aug.keypoints[i]
                keypoints.append([after.x, after.y])
                # print(after.x)
            keypoints = np.array(keypoints, dtype="float32")
            # print(keypoints)

            image_after = kps_aug.draw_on_image(image_aug, size=0)

            # データ拡張を行った画像をリストに格納する
            data_set_list.append([image_after, keypoints])

            # 描画
            # plt.imshow(image_after)
            # plt.scatter(keypoints[:,0], keypoints[:,1], color='r')
            # plt.show()
    return data_set_list


class FaceKeypointDataset(Dataset):
    def __init__(self, dataset, resize):
        self.data = dataset
        self.resize = resize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = image.shape
        image = cv2.resize(image, (self.resize, self.resize))

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


def fit(model, dataloader, data, optimizer, criterion):
    # print("training")
    model.train()
    train_running_loss = 0.0
    counter = 0
    num_batches = int(len(data) / dataloader.batch_size)
    for i, data in enumerate(dataloader):
        counter += 1
        image, keypoints = data["image"].to(config.DEVICE), data["keypoints"].to(
            config.DEVICE
        )
        keypoints = keypoints.view(keypoints.size(0), -1)
        # print(keypoints)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


def validate(model, dataloader, data, criterion):
    # print("validate")
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    num_batches = int(len(data) / dataloader.batch_size)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            counter += 1
            image, keypoints = data["image"].to(config.DEVICE), data["keypoints"].to(
                config.DEVICE
            )
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            # if (epoch + 1) % 25 == 0 and i == 0:
    valid_loss = valid_running_loss / counter
    return valid_loss


def record_progress_vram_information(
    file_name, epoch_num, epoch_start_time, lap_time,total_time, train_loss, valid_loss
):
    gpu = GPU.getGPUs()[0]
    mem_info = psutil.virtual_memory()
    used_gpu_memory = str(gpu.memoryUsed)
    free_gpu_memory = str(gpu.memoryFree)
    memory_percentage = str(mem_info.percent)
    current_gpu_power_consumption = str(
        os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader")
        .read()
        .replace("\n", "")
        .replace(" W", "")
    )
    gpu_temperature = str(gpu.temperature)
    with open(str(file_name), "a") as f:
        writer = csv.writer(f)

        # ヘッダがなければ書き込む
        if not os.path.isfile(file_name) or os.stat(file_name).st_size == 0:
            header = [
                "EPOCH_NUM",
                "EPOCH_START_TIME",
                "LAP_TIME",
                "TOTAL_TIME",
                "TRAIN_LOSS",
                "VALID_LOSS",
                "USED_GPU_MOMORY",
                "FREE_GPU_MEMORY",
                "MEMORY_PERCENTAGE",
                "CURRENT_GPU_POWER_CONSUMPTION",
                "GPU_TEMPERATURE",
            ]
            writer.writerow(header)

        write_content = [
            epoch_num,
            epoch_start_time,
            lap_time,
            total_time,
            train_loss,
            valid_loss,
            used_gpu_memory,
            free_gpu_memory,
            memory_percentage,
            current_gpu_power_consumption,
            gpu_temperature,
        ]
        writer.writerow(write_content)
    return write_content

def model_test(model,model_path,dataset_path,image_list,save_image_dir):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for image_name in image_list:
        with torch.no_grad():
            image = cv2.imread(f"{dataset_path}/{image_name}")
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (config.RESIZE, config.RESIZE))
            orig_image = image.copy()
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            outputs = model(image)
            print(outputs)
            outputs = outputs.cpu().detach().numpy()
            outputs = outputs.reshape(-1, 2)
            # plt.subplot(3, 4, i+1)
            plt.imshow(orig_image, cmap='gray')
            for p in range(outputs.shape[0]):
                    plt.plot(outputs[p, 0], outputs[p, 1], 'r.')
                    plt.text(outputs[p, 0], outputs[p, 1], f"{p}")
            plt.axis('off')
            plt.savefig(f"{save_image_dir}/valid_{image_name}")
            plt.show()
            plt.close()

def main():
    start_time = time.time()
    # /root/source/result

    # コマンドライン引数からハイパーパラーメータを取得する
    # ex) python3 commandLIneHikisuu.py --EPOCHS 1 --BATCH_SIZE 2 --LR 0.001 --MODEL_FILE "./CommonCnn.py" --DATA_AUG_FAC 3
    parser = argparse.ArgumentParser(description="このスクリプトはディープラーニングを自動で実行するためのスクリプトです")

    parser.add_argument("--EPOCHS", type=int, help="int: EPOCHS")
    parser.add_argument("--BATCH_SIZE", type=int, help="int: BATCH_SIZE")
    parser.add_argument("--LR", type=float, default=0.0001, help="float: learning rate")
    parser.add_argument("--MODEL_FILE", type=str, help="str: model file path")
    parser.add_argument(
        "--DATA_AUG_FAC",
        type=int,
        default=0,
        help="int: Multiples of the number of images to be expanded",
    )

    args = parser.parse_args()

    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    LR = args.LR
    MODEL_FILE = args.MODEL_FILE
    DATA_AUG_FAC = args.DATA_AUG_FAC

    lap_times = []

    # 記録データを格納するディレクトリの作成
    # 日付と時間を年月日時分秒の形式にフォーマット
    now = datetime.now()
    now_str = str(now.strftime("%Y%m%d%H%M"))
    model_name = str(MODEL_FILE).replace("./models/", "").replace(".py", "")
    info_dir_name = now_str + f"_{model_name}"
    print(info_dir_name)
    # ディレクトリのパスを設定 (現在のスクリプトの位置に作成)
    # 今後記録はこのディレクトリに入れる
    info_dir_path = os.path.join(config.ROOT_PATH, info_dir_name)

    # ディレクトリを作成
    os.makedirs(info_dir_path, exist_ok=True)

    # 指定のディレクトリにマシンの情報を記録する
    all_device_info_csv_writer(f"{info_dir_path}/{now_str}_DeviceInfo.csv")

    # csvファイルを見て、トレーニングデータとバリデーションデータを分ける
    # training_samples,valid_samples = train_test_split(f"{str(config.DATASET_PATH)}/annotations/annotation_from_xml.csv",config.TEST_SPLIT)
    # training_samples,valid_samples = train_test_split(f"/root/dataset/Annotated_High-Resolution_Anime/annotations/annotation_from_xml.csv",config.TEST_SPLIT)
    training_samples, valid_samples = train_test_split(
        str(config.ANNOTATION_DATA),
        config.TEST_SPLIT,
    )

    # テスト用の画像のリストを保存する
    valid_image_names_df = valid_samples.iloc[:, 0]
    valid_image_names = [name for name in valid_image_names_df]

    # データ拡張を行い、numpyで返す
    train_numpy_dataset = AugmentFaceKeypointDataset(
        training_samples, f"{config.DATASET_PATH}/images", DATA_AUG_FAC
    )
    valid_numpy_dataset = AugmentFaceKeypointDataset(
        valid_samples, f"{config.DATASET_PATH}/images", DATA_AUG_FAC
    )

    train_tensor_data = FaceKeypointDataset(train_numpy_dataset, config.RESIZE)
    valid_tensor_data = FaceKeypointDataset(valid_numpy_dataset, config.RESIZE)
    print(train_tensor_data[0])

    print("train_tensor_dataの数",len(train_tensor_data))
    print("valid_tensor_dataの数",len(valid_tensor_data))
    NUM_OF_TRAIN_DATA = len(train_tensor_data)
    NUM_OF_VALID_DATA = len(valid_tensor_data)

    train_loader = DataLoader(train_tensor_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_tensor_data, batch_size=BATCH_SIZE, shuffle=True)

    # モジュール内のクラスを取得
    FaceKeypointModel = import_class_from_file(MODEL_FILE)

    model = FaceKeypointModel().to(config.DEVICE)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []

    loss_per_50epoch = []
    BEST_TRAIN_LOSS = float('inf')
    BEST_VAL_LOSS = float('inf')

    BEST_TRAIN_LOSS_MODEL = ""
    BEST_VALID_LOSS_MODEL = ""

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1} of {EPOCHS}")
        # 開始時刻及びフォーマット
        epoch_start_time = dt.datetime.now()
        formatted_epoch_start_time = epoch_start_time.strftime("%Y-%m-%d %H:%M:%S")

        # lap_time = end_time - start_time

        train_epoch_loss = fit(
            model, train_loader, train_tensor_data, optimizer, criterion
        )
        val_epoch_loss = validate(model, valid_loader, valid_tensor_data, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)

        if epoch % 100 == 0:
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")

        # ラップタイムを計算する
        end_time = dt.datetime.now()
        lap_time = end_time - epoch_start_time
        lap_times.append(lap_time)

        #トータルタイムを計算する
        total_time = time.time() - start_time
        # 秒単位の時間をHH:MM:SSに変換
        elapsed_total_time_hms = time.strftime("%H:%M:%S", time.gmtime(total_time))

        record_progress_vram_information(
            f"{info_dir_path}/{now_str}_LossInfo.csv",
            epoch + 1,
            formatted_epoch_start_time,
            lap_time,
            elapsed_total_time_hms,
            train_epoch_loss,
            val_epoch_loss,
        )

        if (epoch + 1)  % 50 == 0:
            wait_data = f"model_epoch_{epoch + 1}.pth"
            loss_per_50epoch.append([wait_data,train_epoch_loss,val_epoch_loss])

            model_path = f"{info_dir_path}/models/{wait_data}"
            if not os.path.exists(f"{info_dir_path}/models"):
                os.makedirs(f"{info_dir_path}/models")
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                model_path
            )
            if BEST_TRAIN_LOSS > float(train_epoch_loss):
                BEST_TRAIN_LOSS = float(train_epoch_loss)
                BEST_TRAIN_LOSS_MODEL = wait_data

            if BEST_VAL_LOSS > float(val_epoch_loss):
                BEST_VAL_LOSS = float(train_epoch_loss)
                BEST_VALID_LOSS_MODEL = wait_data

            # validationデータをつかってモデルのテストを行う
            save_valid_images_dir = f"{info_dir_path}/valid_images"
            if not os.path.exists(save_valid_images_dir):
                os.makedirs(save_valid_images_dir)

            valid_images_dir = f"epoch_{epoch + 1}"
            valid_images_dir_path = f"{save_valid_images_dir}/{valid_images_dir}"
            if not os.path.exists(valid_images_dir_path):
                os.makedirs(valid_images_dir_path)

            model_test(model,model_path,f"{config.DATASET_PATH}/images",valid_image_names,f"{valid_images_dir_path}")

        # print(type(lap_time))

        # print(lap_times)
        # total_seconds = sum(time.total_seconds() for time in lap_times)
        # laptime_average = total_seconds / len(lap_times)
        # remaining_epochs = EPOCHS - epoch
        # remaining_time = laptime_average * remaining_epochs
        # average_delta = datetime.timedelta(seconds=remaining_time)
        # base_datetime = datetime.datetime(1970, 1, 1)
        # average_datetime = base_datetime + average_delta
        # formatted_datetime = average_datetime.strftime("%Y%m%d%H%M")
        # print(f"学習の終了までの残り時間：{formatted_datetime}")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{info_dir_path}/loss.png")

    elapsed_time = time.time() - start_time
    elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # csvファイルのヘッダの有無を確認
    param_lsit = [
        "MODEL_NAME",
        "EXECUTED_EPOCHS",
        "BATCH_SIZE",
        "LR",
        "NUM_OF_TRAIN_DATA",
        "NUM_OF_VALID_DATA",
        "IMAGE_SIZE",
        "DATA_AUG_FAC",
        "ELAPSED_TIME",
        "BEST_TRAIN_LOSS",
        "BEST_TRAIN_LOSS_MODEL",
        "BEST_VALID_LOSS",
        "BEST_VALID_LOSS_MODEL",
        "RESULT_FILE",

    ]
    # ファイル名
    result_csv = 'result.csv'
    result_csv_path = f"{config.ROOT_PATH}/{result_csv}"

    # ファイルが存在しない場合のみ新たに作成
    if not os.path.exists(result_csv_path):
        with open(result_csv_path, 'w') as f:
            writer = csv.writer(f)
            # ヘッダ行を書き込みます（必要に応じて修正してください）
            writer.writerow(param_lsit)

    paramd_data = [model_name,
                   EPOCHS,
                   BATCH_SIZE,
                   LR,
                   NUM_OF_TRAIN_DATA,
                   NUM_OF_VALID_DATA,
                   config.RESIZE,
                   DATA_AUG_FAC,
                   elapsed_time_hms,
                   BEST_TRAIN_LOSS,
                   BEST_TRAIN_LOSS_MODEL,
                   BEST_VAL_LOSS,
                   BEST_VALID_LOSS_MODEL,
                   info_dir_name]

    with open(result_csv_path, "a") as f:
        writer = csv.writer(f)
        # writer.writerow(param_lsit)
        writer.writerow(paramd_data)


    # # 50エポック毎のモデル及び誤差量
    # loss_per_50epoch_csv = 'loss_per_50epoch.csv'
    # loss_per_50epoch_path = f"{info_dir_path}/{loss_per_50epoch_csv}"
    # loss_per_50epoch_header = ["MODEL_PTH","TRAIN_EPOCH_LOSS","VAL_EPOCH_LOSS"]

    # # ファイルが存在しない場合のみ新たに作成
    # if not os.path.exists(loss_per_50epoch_path):
    #     with open(loss_per_50epoch_path, 'w') as f:
    #         writer = csv.writer(f)
    #         # ヘッダ行を書き込みます（必要に応じて修正してください）
    #         writer.writerow(loss_per_50epoch_header)

    # with open(loss_per_50epoch_path, "a") as f:
    #     writer = csv.writer(f)
    #     # writer.writerow(param_lsit)
    #     writer.writerow(loss_per_50epoch)

    print("DONE TRAINING")


if __name__ == "__main__":
    main()
