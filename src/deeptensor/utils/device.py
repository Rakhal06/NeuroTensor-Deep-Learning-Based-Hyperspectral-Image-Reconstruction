import torch

def get_device(requested=None):
    """Resolve compute device safely (CUDA or CPU fallback)."""
    if requested and requested != "auto":
        if "cuda" in str(requested).lower() and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
