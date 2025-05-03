import torch

def print_device_info(func):
    def wrapper(*args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device {device.type}")
        return func(*args, **kwargs)
    return wrapper