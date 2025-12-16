import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda version (torch):", torch.version.cuda)
    print("gpu count:", torch.cuda.device_count())
    i = 0
    print("using device:", i)
    print("gpu name:", torch.cuda.get_device_name(i))
else:
    print("NO GPU detected by torch. You are on CPU.")
