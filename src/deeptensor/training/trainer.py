"""DeepTensor Unified Training Utilities."""
import torch
import time

class DeepTensorTrainer:
    """Helper class managing optimization loops and closures."""
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def step(self, closure):
        return self.optimizer.step(closure)
