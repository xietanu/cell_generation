import torch


def torch_to_numpy(voxels: torch.Tensor):
    """Converts a numpy array of voxels to a torch tensor."""
    numpy_voxels = voxels.detach().cpu()

    if len(numpy_voxels.shape) == 3:
        return numpy_voxels.numpy()

    if len(numpy_voxels.shape) == 4:
        return numpy_voxels.permute(1, 2, 3, 0).squeeze().numpy()

    if len(numpy_voxels.shape) == 5:
        return numpy_voxels.permute(0, 2, 3, 4, 1).numpy()

    raise ValueError(
        "Invalid voxel shape: "
        + str(numpy_voxels.shape)
        + ", expected ([N], [C], H, W, D)"
    )
