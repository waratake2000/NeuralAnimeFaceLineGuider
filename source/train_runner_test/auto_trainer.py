import subprocess
from train_test_split import train_test_split
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
train_data_len_divisors = division_patterns(training_data_coutns)
print(train_data_len_divisors)

for num,batch_size in enumerate(train_data_len_divisors[2:]):
    print("num",num)
    # for data_aug_factor in range(0,4):
    try:
<<<<<<< HEAD:source/train_runner/auto_trainer.py
        command = ["python3", "train.py", "--EPOCHS", "10000", "--BATCH_SIZE", f"{batch_size}", "--LR", "0.0001", "--MODEL_FILE", "./models/resnet18.py", "--DATA_AUG_FAC", "0"]
=======
        # python3 train.py --EPOCHS 2000 --BATCH_SIZE 1 --LR 0.0001 --MODEL_FILE ./models/CommonCnn.py --DATA_AUG_FAC 0
        command = ["python3", "train_CommonCNN.py", "--EPOCHS", "2000", "--BATCH_SIZE", f"{batch_size}", "--LR", "0.0001", "--MODEL_FILE", "./models/deepCNN.py", "--DATA_AUG_FAC", "0"]
>>>>>>> baf8fecc72020f3d84b5d2c195d695184baec03e:source/train_runner_test/auto_trainer.py
        subprocess.run(command)
    except:
        continue

