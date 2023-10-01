import torch
import numpy as np

import nnet


class MaskGenerator(torch.nn.Module):
    def __init__(
        self,
        model_gen: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        background_gen: torch.nn.Module | None = None,
        foreground_gen: torch.nn.Module | None = None,
    ):
        super().__init__()

        self.model_gen = model_gen
        self.mask_from_voxels = nnet.modules.RandomImageFromVoxGrid(device)
        self.background_gen = background_gen
        self.foreground_gen = foreground_gen

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        voxel_grid = self.model_gen(in_tensor)

        background = self.background_gen(in_tensor) if self.background_gen else None

        mask = self.mask_from_voxels(voxel_grid, background=background)
        if self.foreground_gen is not None:
            foreground = self.foreground_gen(in_tensor)

            alpha = foreground[:, -1, :, :] ** 3
            alpha = alpha.unsqueeze(1)
            if foreground.shape[1] == 1:
                mask = mask * (1 - alpha) + alpha
            else:
                value = foreground[:, :-1, :, :]
                mask = mask * (1 - alpha) + alpha * value
        mask = mask * 2 - 1
        return mask
