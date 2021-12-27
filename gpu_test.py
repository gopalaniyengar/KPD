import os
import torch

os.system('nvidia-smi')
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

device = torch.device("cuda")
print(device)
print(torch.cuda.get_device_properties(device))