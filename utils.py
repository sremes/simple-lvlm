import torch


def get_device() -> torch.device:
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device
