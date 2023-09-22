import numpy as np
import torch


def numpy_to_torch(voxels: np.ndarray):
    """Converts a numpy array of voxels to a torch tensor."""
    tensor_voxels = torch.from_numpy(voxels).float()

    if len(tensor_voxels.shape) == 3:
        return tensor_voxels.unsqueeze(0)

    if len(tensor_voxels.shape) == 4:
        return tensor_voxels.permute(3, 0, 1, 2)

    if len(tensor_voxels.shape) == 5:
        return tensor_voxels.permute(0, 4, 1, 2, 3)

    raise ValueError(
        "Invalid voxel shape: "
        + str(tensor_voxels.shape)
        + ", expected ([N], H, W, D, [C])"
    )
