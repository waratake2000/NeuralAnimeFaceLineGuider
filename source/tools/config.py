import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_PATH = "/root/Cloud/Learning_result"
DATASET_PATH = "/root/dataset/Manga109_landmark_annotated"
ANNOTATION_DATA = "/root/dataset/Manga109_landmark_annotated/Manga109_annotated.csv"
TEST_SPLIT = 0.02
RESIZE = 196
