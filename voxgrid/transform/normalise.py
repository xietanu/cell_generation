import numpy as np
import torch


def normalise(voxel_grid: torch.Tensor | np.ndarray):
    """Normalise a voxel grid to the range [0, 1],
    making reasonable assumptions about the range of values.
    """
    if isinstance(voxel_grid, np.ndarray):
        voxel_grid = voxel_grid.astype(np.float32)
    else:
        voxel_grid = voxel_grid.float()

    if voxel_grid.min() >= 0 and voxel_grid.max() <= 1:
        return voxel_grid
    if voxel_grid.min() >= -1 and voxel_grid.max() <= 1:
        return (voxel_grid + 1) / 2
    if voxel_grid.min() >= 0 and voxel_grid.max() <= 255:
        return voxel_grid / 255
    return (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min())
