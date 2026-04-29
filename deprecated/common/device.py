"""Device selection helper."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device.

    Priority: CUDA > MPS > CPU.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
