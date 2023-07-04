from datetime import datetime
import pytz
import platform
import os
import psutil
import GPUtil as GPU
import csv

def info_csv_writer(info_list,dir_name):
    with open(f"{dir_name}","a") as f:
        writer = csv.writer(f)
        writer.writerows(info_list)

def time_info_csv_writer(dir_name):
    time_title = ["Time Information"]
    utc_list = ["UTC",str(datetime.now(pytz.timezone('UTC')))]
    jst_list = ["JST",str(datetime.now(pytz.timezone('Asia/Tokyo')))]
    info_list = [time_title,utc_list,jst_list]
    info_csv_writer(info_list,dir_name)
    return info_list

def sys_info_csv_writer(dir_name):
    system_title = ["System Information"]
    uname = platform.uname()
    system_list = ["System",str(uname.system)]
    Node_Name_list = ["Node Name",str(uname.node)]
    Release_list = ["Release",str(uname.release)]
    Version_list = ["Version",str(uname.version)]
    info_list = [system_title,system_list,Node_Name_list,Release_list,Version_list]
    info_csv_writer(info_list,dir_name)
    return info_list

def cpu_info_csv_writer(dir_name):
    cpu_title = ["CPU Information"]
    uname = platform.uname()
    cpu_info = os.popen("lscpu | grep \"Model name:\"").read()
    cpu = cpu_info.split("  ")[-1].replace("\n","")
    cpu_name = ["CPU",str(cpu)]
    cpu_processor = ["Processor",str(uname.processor)]
    physical_cores = ["Physical cores", str(psutil.cpu_count(logical=False))]
    total_cores = ["Total cores", str(psutil.cpu_count(logical=True))]
    cpufreq = psutil.cpu_freq()
    max_frequency = ["Max Frequency(Mhz)",f"{cpufreq.max:.2f}"]
    min_frequency = ["Min Frequency(Mhz)",f"{cpufreq.min:.2f}"]
    current_frequency = ["Current Frequency(Mhz)",f"{cpufreq.current:.2f}"]
    total_cpu_usage = ["Total CPU Usage(%)",f"{psutil.cpu_percent()}"]
    info_list = [cpu_title,cpu_name,cpu_processor,physical_cores,total_cores,max_frequency,min_frequency,current_frequency,total_cpu_usage]
    info_csv_writer(info_list,dir_name)
    return info_list

def gpu_info_csv_writer(dir_name):
    gpu = GPU.getGPUs()[0]
    mem_info = psutil.virtual_memory()
    gpu_title = ["GPU Information"]
    gpu_name = ["GPU name", gpu.name]
    nvidia_driver_version = ["NVIDIA Driver Version",str(os.popen("nvidia-smi | grep 'Driver Version' | awk '{ print $3 }'").read().replace("\n",""))]
    cuda_version = ["CUDA Version",str(os.popen("nvidia-smi | grep 'CUDA Version' | awk '{ print $9 }'").read().replace("\n",""))]
    total_gpu_memory = ["Total GPU memory", str(gpu.memoryTotal)]
    used_gpu_memory = ["Used GPU memory", str(gpu.memoryUsed)]
    free_gpu_memory = ["Free GPU memory", str(gpu.memoryFree)]
    memory_percentage = ["Memory percentage", str(mem_info.percent)]
    maximum_gpu_power_consumption = ["Maximum GPU power consumption", str(os.popen("nvidia-smi --query-gpu=power.limit --format=csv,noheader").read().replace("\n",""))]
    current_gpu_power_consumption = ["Current GPU power consumption", str(os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader").read().replace("\n",""))]
    gpu_temperature = ["GPU temperature",str(gpu.temperature)]

    info_list = [gpu_title,gpu_name,nvidia_driver_version,cuda_version,total_gpu_memory,used_gpu_memory,free_gpu_memory,memory_percentage,maximum_gpu_power_consumption,current_gpu_power_consumption,gpu_temperature]
    info_csv_writer(info_list,dir_name)
    return info_list

def all_device_info_csv_writer(dir_name):
    time_info = time_info_csv_writer(dir_name)
    info_csv_writer([""],dir_name)
    sys_info = sys_info_csv_writer(dir_name)
    info_csv_writer([""],dir_name)
    cpu_info = cpu_info_csv_writer(dir_name)
    info_csv_writer([""],dir_name)
    gpu_info = gpu_info_csv_writer(dir_name)
    return [time_info,sys_info,cpu_info,gpu_info]

# all_device_info_csv_writer("./test5.csv")
