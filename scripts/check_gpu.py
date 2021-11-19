"""Test Pytorch is working with gpu."""
import torch


def check_gpu() -> None:
    """Check if can access the gpu."""
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    check_gpu()
