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

for batch_size in train_data_len_divisors[1:]:
    # for data_aug_factor in range(0,4):
    try:
        command = ["python3", "train.py", "--EPOCHS", "20000", "--BATCH_SIZE", f"{batch_size}", "--LR", "0.0001", "--MODEL_FILE", "./models/CommonCnn.py", "--DATA_AUG_FAC", "0"]
        subprocess.run(command)
    except:
        continue

