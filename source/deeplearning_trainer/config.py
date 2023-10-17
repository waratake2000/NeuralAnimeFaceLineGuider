import torch

# nohup python3 train.py --EPOCHS 10000 --BATCH_SIZE 128 --LEARNING_RATE 0.0001 --MODEL_FILE resnet --DATA_AUG_FAC 1 --REPORT True --DEVICE 0 > experiment-2-resnet101.log 2>&1 &

# train.py
# DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# ROOT_PATH = "/root/Cloud/deeplearning_results/output_items"
MLRUNS_PATH = "/root/Cloud/deeplearning_results/mlruns/"
DATASET_PATH = [
    "/root/dataset/anime_face_landmark_20230912/images",
    "/root/dataset/original_dataset/selfie2anime_0-100/images"
    ]
ANNOTATION_DATA = [
    "/root/dataset/anime_face_landmark_20230912/annotations/person_keypoints_Train.json",
    "/root/dataset/original_dataset/selfie2anime_0-100/annotations/person_keypoints_default.json"
    ]
TEST_SPLIT = 0.02
# RESIZE = 128
RESIZE = 256
# EXPERIMENT_NAME = "fate_zero"
EXPERIMENT_NAME = "fate_stay_night"

# auto_trainer.py
# EPOCHS = 5000
# LEARNING_RATE = 0.0001
# MODEL_FILE = "resnet18"
