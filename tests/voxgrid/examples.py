import numpy as np
import torch


def get_example_0c_voxgrid(as_tensor: bool = False):
    if as_tensor:
        voxels = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    else:
        voxels = np.zeros((3, 3, 3), dtype=np.float32)
    voxels[..., 0, 0, 0] = 1
    voxels[..., 0, 1, 0] = 1
    voxels[..., 0, 1, 1] = 1
    voxels[..., 1, 1, 1] = 0.5
    voxels[..., 1, 0, 2] = 1
    return voxels


def get_example_1c_voxgrid(as_tensor: bool = False):
    if as_tensor:
        voxels = torch.zeros((3, 3, 3), dtype=torch.float32)
    else:
        voxels = np.zeros((3, 3, 3), dtype=np.float32)
    voxels[0, 0, 0] = 1
    voxels[0, 1, 0] = 1
    voxels[0, 1, 1] = 1
    voxels[1, 1, 1] = 0.5
    voxels[1, 0, 2] = 1

    if isinstance(voxels, torch.Tensor):
        voxels = voxels.unsqueeze(0)
    else:
        voxels = voxels[..., np.newaxis]
    return voxels

