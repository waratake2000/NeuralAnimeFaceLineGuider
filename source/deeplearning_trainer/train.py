# nohup python3 train.py --EPOCHS 10000 --BATCH_SIZE 128 --LEARNING_RATE 0.0001 --MODEL_FILE resnet --DATA_AUG_FAC 5 --REPORT True --DEVICE 0 > experiment-20231015-resnet101.log 2>&1 &

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
torch.backends.cudnn.benchmark = True
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
    parser.add_argument("--DEVICE", type=int,default=0, help="str: to use gpu number")

    args = parser.parse_args()
    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    LEARNING_RATE = args.LEARNING_RATE
    MODEL_FILE = args.MODEL_FILE
    DATA_AUG_FAC = args.DATA_AUG_FAC
    REPORT = args.REPORT
    DEVICE = args.DEVICE

    DEVICE = torch.device(f'cuda:{DEVICE}' if torch.cuda.is_available() else 'cpu')

    # coco datasetのjsonファイルを読み込む
    annotation_list = []
    for i in config.ANNOTATION_DATA:
        annotation_list += load_dataset.cocokeypoints_list_converter(str(i))


    print("len(annotation_list)",len(annotation_list))
    print("annotation_list: ")
    print("type(annotation_list): ",type(annotation_list))
    print("len(annotation_list): ",len(annotation_list))

    # for i in annotation_list[0:10]:
    #     print(i)


    # csvファイルを見て、トレーニングデータとバリデーションデータを分ける
    training_samples, valid_samples = load_dataset.train_test_split(
        annotation_list,
        config.TEST_SPLIT,
    )

    # テスト用の画像のリストを保存する
    valid_image_names_df = valid_samples.iloc[:, 0]
    valid_image_names = [name for name in valid_image_names_df]

    # データ拡張を行い、numpyで返す
    # train_numpy_dataset = []
    # valid_numpy_dataset = []
    # print("データ拡張を行い、numpyで返す")
    # for i in config.DATASET_PATH:
    #     print(i)
    #     try:
    #         train_numpy_dataset.append(load_dataset.AugmentFaceKeypointDataset(training_samples, f"{i}", DATA_AUG_FAC))
    #         valid_numpy_dataset.append(load_dataset.AugmentFaceKeypointDataset(valid_samples, f"{i}", DATA_AUG_FAC))
    #     except:
    #         continue

    # /root/dataset/anime_face_landmark_20230912/images/images/AnythingV5Ink_gitv1.5.4_0901_140034_2.jpg
    train_numpy_dataset = load_dataset.AugmentFaceKeypointDataset(training_samples, config.DATASET_PATH, DATA_AUG_FAC)
    valid_numpy_dataset = load_dataset.AugmentFaceKeypointDataset(valid_samples, config.DATASET_PATH, DATA_AUG_FAC)

    train_tensor_data = load_dataset.FaceKeypointDataset(train_numpy_dataset, config.RESIZE)

    valid_tensor_data = load_dataset.FaceKeypointDataset(valid_numpy_dataset, config.RESIZE)

    # print("train_tensor_dataの数",len(train_tensor_data))
    # print("train_tensor_data",train_tensor_data)
    # print("len(train_tensor_data)",len(train_tensor_data))
    # print("valid_tensor_dataの数",len(valid_tensor_data))

    train_loader = DataLoader(train_tensor_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_tensor_data, batch_size=BATCH_SIZE, shuffle=True)

    # モジュール内のクラスを取得
    sys.path.append('./models')
    module = importlib.import_module(MODEL_FILE)
    model = module.LandmarkDetector().to(DEVICE)
    NUM_OF_PARAMS = count_parameters(model)
    print(f"モデルのパラメータ数: {NUM_OF_PARAMS}")

    # return 0

    # 記録スタート ----------------------------------------------------------------------
    now = datetime.now()
    now_str = str(now.strftime("%Y_%m%d_%H%M"))
    mlflow_model_run_name = now_str + f"_{MODEL_FILE}"

    tags = {'trial':2,
            MLFLOW_RUN_NAME:str(mlflow_model_run_name),
            MLFLOW_USER:"gpudev",
            MLFLOW_SOURCE_NAME:"train.py",
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
        EXPERIMENT_NAME = str(config.EXPERIMENT_NAME + "_" + MODEL_FILE)
    else:
        mlflow.set_tracking_uri("./mlruns/")
        EXPERIMENT_NAME = str("test" + "_" + MODEL_FILE)
    print("EXPERIMENT_NAME",EXPERIMENT_NAME)
    writer = MlflowWriter(EXPERIMENT_NAME)
    writer.create_new_run(tags)
    info_dir_path = writer.artifact_uri()
    print("artifact_url",info_dir_path)
    for key,item in params_dict.items():
        writer.log_param(key, item)


    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []

    torch.cuda.empty_cache()

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
            model, train_loader, train_tensor_data, optimizer, criterion ,DEVICE
        )

        val_epoch_loss = validate(model, valid_loader, valid_tensor_data, criterion, DEVICE)
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
        model_test_freq = 500
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
            model_test(model,model_path,config.DATASET_PATH,valid_image_names,f"{valid_images_dir_path}",DEVICE)
    writer.set_terminated()
    print("DONE TRAINING")


if __name__ == "__main__":
    main()
