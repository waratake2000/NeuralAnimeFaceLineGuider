import subprocess
from load_dataset import train_test_split
import config

# 約数を出力する関数
def division_patterns(num):
    remainder_list = []
    for i in range(1, num + 1):
        quotient, _ = divmod(num, i)
        remainder_list.append(quotient)
    remainder_list_sorted = sorted(set(remainder_list), reverse=True)
    return remainder_list_sorted

training_samples, _ = train_test_split(str(config.ANNOTATION_DATA),config.TEST_SPLIT)
training_data_coutns = len(training_samples)

for data_aug_factor in [1,5,10]:
    train_data_len_divisors = division_patterns(int(training_data_coutns * data_aug_factor))
    for batch_size in train_data_len_divisors[0:6*data_aug_factor:data_aug_factor]:
        try:
            # python3 train.py --EPOCHS 2000 --BATCH_SIZE 400 --LEARNING_RATE 0.0001 --MODEL_FILE resnet18 --DATA_AUG_FAC
            command = ["python3", "train.py", "--EPOCHS", "20000", "--BATCH_SIZE", f"{batch_size}", "--LEARNING_RATE", "0.0001", "--MODEL_FILE", "resnet18", "--DATA_AUG_FAC", f"{data_aug_factor}","--REPORT","True"]
            subprocess.run(command)
        except:
            continue
