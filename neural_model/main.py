import time
import torch

print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda.is_available (ROCm usa questa API):", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
