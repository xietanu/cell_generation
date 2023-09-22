import torch
import numpy as np


class VoxGridAlphaAdjust(torch.nn.Module):
    """Module to adjust the alpha values of a voxel grid to assist with training."""

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Adjusts the alpha values of a voxel grid to assist with training."""
        adjust_value = np.log2(in_tensor.shape[-1]) + 3

        values = in_tensor[:, :-1, :, :, :]
        alpha = in_tensor[:, -1, :, :, :].unsqueeze(1) ** adjust_value

        return torch.cat((values, alpha), dim=1)
