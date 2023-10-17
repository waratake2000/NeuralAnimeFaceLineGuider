import torch

# train.py
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
ROOT_PATH = "/root/Cloud/deeplearning_results/output_items"
MLRUNS_PATH = "/root/Cloud/deeplearning_results/mlruns/"
DATASET_PATH = "/root/dataset/anime_face_landmark_20230912/images"
ANNOTATION_DATA = "/root/dataset/anime_face_landmark_20230912/annotations/person_keypoints_Train.json"
TEST_SPLIT = 0.02
RESIZE = 128
EXPERIMENT_NAME = "Manga109_Landmark"

# auto_trainer.py
EPOCHS = 10000
LEARNING_RATE = 0.0001
MODEL_FILE = "deepCNN"
