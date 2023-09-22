import numpy as np
import torch


def get_n_channels(voxel_grid: torch.Tensor | np.ndarray):
    if voxel_grid.ndim == 3:
        return 0
    elif isinstance(voxel_grid, np.ndarray):
        return voxel_grid.shape[-1]
    return voxel_grid.shape[-4]
