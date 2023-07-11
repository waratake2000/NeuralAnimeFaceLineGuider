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

import mlflow
from mlflow_func import MlflowWriter
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER,MLFLOW_SOURCE_NAME

import config
import load_dataset
import GPUtil as GPU
import psutil
from model_fit_validate import fit
from model_fit_validate import validate
from model_tester import model_test

plt.style.use("ggplot")

pip_requirements = [
    'torch==1.12.1+cu113'
]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # ex) python3 train.py --EPOCHS 100 --BATCH_SIZE 800 --LEARNING_RATE 0.0001 --MODEL_FILE resnet18 --DATA_AUG_FAC 0
    parser = argparse.ArgumentParser(description="このスクリプトはディープラーニングを自動で実行するためのスクリプトです")

    parser.add_argument("--EPOCHS", type=int, help="int: EPOCHS")
    parser.add_argument("--BATCH_SIZE", type=int, help="int: BATCH_SIZE")
    parser.add_argument("--LEARNING_RATE", type=float, default=0.0001, help="float: learning rate")
    parser.add_argument("--MODEL_FILE", type=str, help="str: model file path")
    parser.add_argument(
        "--DATA_AUG_FAC",
        type=int,
        default=0,
        help="int: Multiples of the number of images to be expanded",
    )
    parser.add_argument("--REPORT", type=bool,default=False, help="str: do you want to report about learning processes?")

    args = parser.parse_args()
    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    LEARNING_RATE = args.LEARNING_RATE
    MODEL_FILE = args.MODEL_FILE
    DATA_AUG_FAC = args.DATA_AUG_FAC
    REPORT = args.REPORT

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

    print("train_tensor_dataの数",len(train_tensor_data))
    print("valid_tensor_dataの数",len(valid_tensor_data))

    train_loader = DataLoader(train_tensor_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_tensor_data, batch_size=BATCH_SIZE, shuffle=True)

    # モジュール内のクラスを取得
    sys.path.append('./models')
    module = importlib.import_module(MODEL_FILE)
    model = module.LandmarkDetector().to(config.DEVICE)
    NUM_OF_PARAMS = count_parameters(model)
    print(f"モデルのパラメータ数: {NUM_OF_PARAMS}")

    # 記録スタート ----------------------------------------------------------------------
    now = datetime.now()
    now_str = str(now.strftime("%Y_%m%d_%H%M"))
    model_run_name = now_str + f"_{MODEL_FILE}"

    tags = {'trial':2,
            MLFLOW_RUN_NAME:str(model_run_name),
            MLFLOW_USER:"nitdev",
            MLFLOW_SOURCE_NAME:"test",
            "MODEL":MODEL_FILE,
            "NUM_OF_PARAMS":NUM_OF_PARAMS,
            "EPOCHS":EPOCHS,
            "BATCH_SIZE":BATCH_SIZE,
            "LEARNING_RATE":LEARNING_RATE,
            "DATA_AUG_FAC":DATA_AUG_FAC,
            "IMAGE_SIZE":config.RESIZE,
    }

    params_dict = {
        "MODEL":MODEL_FILE,
        "NUM_OF_PARAMS":NUM_OF_PARAMS,
        "EPOCHS":EPOCHS,
        "BATCH_SIZE":BATCH_SIZE,
        "LEARNING_RATE":LEARNING_RATE,
        "DATA_AUG_FAC":DATA_AUG_FAC,
        "IMAGE_SIZE":config.RESIZE
    }

    if REPORT:
        mlflow.set_tracking_uri(config.MLRUNS_PATH)
        EXPERIMENT_NAME = str(config.EXPERIMENT_NAME + "_" + config.MODEL_FILE)
    else:
        mlflow.set_tracking_uri("./mlruns/")
        EXPERIMENT_NAME = str("test" + "_" + config.MODEL_FILE)
    print("EXPERIMENT_NAME",EXPERIMENT_NAME)
    writer = MlflowWriter(EXPERIMENT_NAME)
    writer.create_new_run(tags)
    info_dir_path = writer.artifact_uri()
    print("artifact_url",info_dir_path)
    for key,item in params_dict.items():
        writer.log_param(key, item)


    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []

    # for epoch in range(0,int(EPOCHS)+1):
    for epoch in range(0,int(EPOCHS)+1):
        # 2エポック目でgpuの様子を確認する、ついでにパラメータの数も記録しておく
        if epoch == 2:
            gpu = GPU.getGPUs()[0]
            mem_info = psutil.virtual_memory()
            used_gpu_memory = str(gpu.memoryUsed)
            free_gpu_memory = str(gpu.memoryFree)
            memory_percentage = str(mem_info.percent)
            gpu_params_dict = {
                "NUM_OF_PARAMS":NUM_OF_PARAMS,
                "GPU_USED_MEMORY":used_gpu_memory,
                "GPU_FREE_MEMORY":free_gpu_memory,
                "MEMORY_PERCENTAGE":memory_percentage
            }
            for key,item in gpu_params_dict.items():
                writer.log_metric(key, item)

        print(f"Epoch {epoch+1} of {EPOCHS}")
        epoch_start_time = dt.datetime.now()

        train_epoch_loss = fit(
            model, train_loader, train_tensor_data, optimizer, criterion
        )

        val_epoch_loss = validate(model, valid_loader, valid_tensor_data, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print("train_loss",train_epoch_loss)
        print("validation_loss",val_epoch_loss)

        # ラップタイムを計算する
        end_time = dt.datetime.now()
        lap_time = end_time - epoch_start_time
        print("残り推定学習時間：",lap_time * (EPOCHS - epoch))

        writer.log_metric_step("train_loss", train_epoch_loss,step=epoch)
        writer.log_metric_step("validation_loss", val_epoch_loss,step=epoch)
        model_test_freq = 200
        if (epoch)  % model_test_freq == 0 and epoch != 0:
            # 重みパラメータの保存スクリプト
            wait_data = f"model_epoch_{epoch}.pth"

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

            # validationデータをつかってモデルのテスト及び、テストしたvalidation画像の保存を行う
            save_valid_images_dir = f"{info_dir_path}/valid_images"
            if not os.path.exists(save_valid_images_dir):
                os.makedirs(save_valid_images_dir)

            valid_images_dir = f"epoch_{epoch}"
            valid_images_dir_path = f"{save_valid_images_dir}/{valid_images_dir}"
            if not os.path.exists(valid_images_dir_path):
                os.makedirs(valid_images_dir_path)
            model_test(model,model_path,f"{config.DATASET_PATH}/images",valid_image_names,f"{valid_images_dir_path}")
    writer.set_terminated()
    print("DONE TRAINING")


if __name__ == "__main__":
    main()
