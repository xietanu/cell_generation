"""Module that randomly crops a patch of specified size from an image."""
import torch
import numpy as np


class RandomPatch(torch.nn.Module):
    """Module that randomly crops a patch of specified size from an image."""

    def __init__(self, size: tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Crop a random patch from the input tensor."""
        if in_tensor.shape[2] == self.size[0] and in_tensor.shape[3] == self.size[1]:
            return in_tensor
        top = np.random.randint(0, in_tensor.shape[2] - self.size[0])
        left = np.random.randint(0, in_tensor.shape[3] - self.size[1])
        return in_tensor[:, :, top : top + self.size[0], left : left + self.size[1]]
