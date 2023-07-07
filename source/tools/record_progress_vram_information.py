import GPUtil as GPU
import psutil
import os
import csv

def record_progress_vram_information(
    file_name, epoch_num, epoch_start_time, lap_time,total_time, train_loss, valid_loss
):
    gpu = GPU.getGPUs()[0]
    mem_info = psutil.virtual_memory()
    used_gpu_memory = str(gpu.memoryUsed)
    free_gpu_memory = str(gpu.memoryFree)
    memory_percentage = str(mem_info.percent)
    current_gpu_power_consumption = str(
        os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader")
        .read()
        .replace("\n", "")
        .replace(" W", "")
    )
    gpu_temperature = str(gpu.temperature)
    with open(str(file_name), "a") as f:
        writer = csv.writer(f)

        # ヘッダがなければ書き込む
        if not os.path.isfile(file_name) or os.stat(file_name).st_size == 0:
            header = [
                "EPOCH_NUM",
                "EPOCH_START_TIME",
                "LAP_TIME",
                "TOTAL_TIME",
                "TRAIN_LOSS",
                "VALID_LOSS",
                "USED_GPU_MOMORY",
                "FREE_GPU_MEMORY",
                "MEMORY_PERCENTAGE",
                "CURRENT_GPU_POWER_CONSUMPTION",
                "GPU_TEMPERATURE",
            ]
            writer.writerow(header)

        write_content = [
            epoch_num,
            epoch_start_time,
            lap_time,
            total_time,
            train_loss,
            valid_loss,
            used_gpu_memory,
            free_gpu_memory,
            memory_percentage,
            current_gpu_power_consumption,
            gpu_temperature,
        ]
        writer.writerow(write_content)
    return write_content
