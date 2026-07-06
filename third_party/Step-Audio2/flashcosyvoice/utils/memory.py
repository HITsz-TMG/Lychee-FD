import os

import torch
from pynvml import *  # noqa


def get_gpu_memory():
    torch.cuda.synchronize()
    nvmlInit()
    visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
    cuda_device_idx = torch.cuda.current_device()
    cuda_device_idx = visible_device[cuda_device_idx]
    handle = nvmlDeviceGetHandleByIndex(cuda_device_idx)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total
    used_memory = mem_info.used
    free_memory = mem_info.free
    nvmlShutdown()
    return total_memory, used_memory, free_memory
