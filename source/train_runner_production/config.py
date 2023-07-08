import torch

# train.py
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_PATH = "/root/Cloud/deeplearning_results/output_items"
MLRUNS_PATH = "/root/Cloud/deeplearning_results/mlruns/"
DATASET_PATH = "/root/dataset/Manga109_landmark_annotated"
ANNOTATION_DATA = "/root/dataset/Manga109_landmark_annotated/Manga109_annotated.csv"
TEST_SPLIT = 0.02
RESIZE = 196
EXPERIMENT_NAME = "Manga109_Landmark"

# auto_trainer.py
EPOCHS = 20000
LEARNING_RATE = 0.0001
MODEL_FILE = "resnet18"
