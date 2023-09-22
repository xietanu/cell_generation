import torch
import numpy as np

import voxgrid


def get_alpha(voxel_grid: torch.Tensor | np.ndarray):
    """Returns the alpha channel of the voxel model."""
    if voxgrid.get_n_channels(voxel_grid) in [0, 1]:
        return voxel_grid.squeeze()
    elif isinstance(voxel_grid, np.ndarray):
        return voxel_grid[..., -1]
    return voxel_grid[..., -1, :, :, :]


def get_values(voxel_grid: torch.Tensor | np.ndarray):
    """Returns the values of the voxel model."""
    if voxgrid.get_n_channels(voxel_grid) in [0, 1]:
        if isinstance(voxel_grid, np.ndarray):
            return np.ones(voxel_grid.squeeze().shape, dtype=np.float32)
        return torch.ones(voxel_grid.squeeze().shape, dtype=torch.float32)
    elif isinstance(voxel_grid, np.ndarray):
        return voxel_grid[..., :-1]
    return voxel_grid[..., :-1, :, :, :]
