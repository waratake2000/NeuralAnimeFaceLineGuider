import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import datetime as dt
import time

import csv

import config
import load_dataset
from device_info_writer import all_device_info_csv_writer
from record_progress_vram_information import record_progress_vram_information
from model_fit_validate import fit
from model_fit_validate import validate
from model_tester import model_test


plt.style.use("ggplot")

def main():
    start_time = time.time()

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
    info_dir_name = now_str + f"_{MODEL_FILE}"
    print("info_dir_name",info_dir_name)
    # ディレクトリのパスを設定 (現在のスクリプトの位置に作成)
    # 今後記録はこのディレクトリに入れる
    info_dir_path = os.path.join(config.ROOT_PATH, info_dir_name)

    # ディレクトリを作成
    os.makedirs(info_dir_path, exist_ok=True)

    # 指定のディレクトリにマシンの情報を記録する
    all_device_info_csv_writer(f"{info_dir_path}/{now_str}_DeviceInfo.csv")

    # csvファイルを見て、トレーニングデータとバリデーションデータを分ける
    training_samples, valid_samples = load_dataset.train_test_split(
        str(config.ANNOTATION_DATA),
        config.TEST_SPLIT,
    )

    # テスト用の画像のリストを保存する
    valid_image_names_df = valid_samples.iloc[:, 0]
    valid_image_names = [name for name in valid_image_names_df]

    # データ拡張を行い、numpyで返す
    train_numpy_dataset = load_dataset.AugmentFaceKeypointDataset(
        training_samples, f"{config.DATASET_PATH}/images", DATA_AUG_FAC
    )
    valid_numpy_dataset = load_dataset.AugmentFaceKeypointDataset(
        valid_samples, f"{config.DATASET_PATH}/images", DATA_AUG_FAC
    )

    train_tensor_data = load_dataset.FaceKeypointDataset(train_numpy_dataset, config.RESIZE)
    valid_tensor_data = load_dataset.FaceKeypointDataset(valid_numpy_dataset, config.RESIZE)
    # print(train_tensor_data[0])

    print("train_tensor_dataの数",len(train_tensor_data))
    print("valid_tensor_dataの数",len(valid_tensor_data))
    NUM_OF_TRAIN_DATA = len(train_tensor_data)
    NUM_OF_VALID_DATA = len(valid_tensor_data)

    train_loader = DataLoader(train_tensor_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_tensor_data, batch_size=BATCH_SIZE, shuffle=True)

    # モジュール内のクラスを取得
    # FaceKeypointModel = import_class_from_file(MODEL_FILE)
    sys.path.append('./models')
    module = importlib.import_module(MODEL_FILE)
    model = module.LandmarkDetector().to(config.DEVICE)
    # model = resnet18().to(config.DEVICE)

    print(model)
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
        # print(lap_time)
        lap_times.append(lap_time)
        print("残り推定学習時間：",lap_time * (EPOCHS - epoch))


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

        model_test_freq = 100
        if (epoch)  % model_test_freq == 0 and epoch != 0:
            wait_data = f"model_epoch_{epoch}.pth"
            loss_per_50epoch.append([wait_data,train_epoch_loss,val_epoch_loss])

            model_path = f"{info_dir_path}/models/{wait_data}"
            if not os.path.exists(f"{info_dir_path}/models"):
                os.makedirs(f"{info_dir_path}/models")
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")
            torch.save(
                {
                    "epoch": epoch,
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

            valid_images_dir = f"epoch_{epoch}"
            valid_images_dir_path = f"{save_valid_images_dir}/{valid_images_dir}"
            if not os.path.exists(valid_images_dir_path):
                os.makedirs(valid_images_dir_path)

            model_test(model,model_path,f"{config.DATASET_PATH}/images",valid_image_names,f"{valid_images_dir_path}")

        write_graph_freq = 30
        if (epoch)  % write_graph_freq == 0 and epoch != 0:
            save_loss_graph_dir = f"{info_dir_path}/loss_graph"
            if not os.path.exists(save_loss_graph_dir):
                os.makedirs(save_loss_graph_dir)
            plt.figure(figsize=(10, 7))
            plt.plot(train_loss, color="orange", label="train loss")
            plt.plot(val_loss, color="red", label="validation loss")

            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            epoch_loss_max = max(max(train_loss[epoch-write_graph_freq:]),max(val_loss[epoch-write_graph_freq:])) * 1.2
            # epoch_loss_min = min(min(train_loss),min(val_loss)) * 0.8
            plt.ylim([0, epoch_loss_max])
            plt.xlim([epoch-int(write_graph_freq*1.2), epoch+5])
            plt.legend()
            plt.savefig(f"{save_loss_graph_dir}/loss_{epoch-write_graph_freq}_{epoch}.png")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim([0, 0.1])
    plt.legend()
    plt.savefig(f"{info_dir_path}/loss_all.png")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0, 0.1])
    plt.legend()
    plt.savefig(f"{info_dir_path}/loss_0.1.png")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0, 0.05])
    plt.legend()
    plt.savefig(f"{info_dir_path}/loss_0.05.png")

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
        "RESULT_FILE"
    ]
    # ファイル名
    result_csv = 'result.csv'
    result_csv_path = f"{config.ROOT_PATH}/{result_csv}"

    # ファイルが存在しない場合のみ新たに作成
    if not os.path.exists(result_csv_path):
        with open(result_csv_path, 'w') as f:
            writer = csv.writer(f)
            # ヘッダ行を書き込みます
            writer.writerow(param_lsit)

    paramd_data = [MODEL_FILE,
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
        writer.writerow(paramd_data)
    print("DONE TRAINING")


if __name__ == "__main__":
    main()
