import numpy as np
import torch


def is_valid_voxel_grid(
    voxel_grid: np.ndarray | torch.Tensor, raise_error_if_not: bool = False
) -> bool:
    if voxel_grid.max() > 1 or voxel_grid.min() < 0:
        if raise_error_if_not:
            raise ValueError(
                "Voxel grid not normalised: min/max = ["
                + str(voxel_grid.min())
                + ", "
                + str(voxel_grid.max())
                + "], expected [0, 1]"
            )
        return False
    if isinstance(voxel_grid, np.ndarray):
        return is_valid_numpy_voxel_grid(voxel_grid, raise_error_if_not)
    return is_valid_torch_voxel_grid(voxel_grid, raise_error_if_not)


def is_valid_numpy_voxel_grid(
    voxel_grid: np.ndarray, raise_error_if_not: bool = False
) -> bool:
    if voxel_grid.ndim not in (3, 4, 5):
        if raise_error_if_not:
            raise ValueError(
                "Invalid voxel grid shape: "
                + str(voxel_grid.shape)
                + ", expected ([N], H, W, D, [C])"
            )
        return False
    if voxel_grid.ndim == 3:
        return True
    if voxel_grid.shape[-1] not in (1, 2, 4):
        if raise_error_if_not:
            raise ValueError(
                f"Invalid number of channels ({voxel_grid.shape[-1]}),"
                " expected 1, 2 (Value + alpha), or 4 (RGBA)"
            )
        return False
    return True


def is_valid_torch_voxel_grid(
    voxel_grid: torch.Tensor, raise_error_if_not: bool = False
) -> bool:
    if voxel_grid.ndim not in (4, 5):
        if raise_error_if_not:
            raise ValueError(
                "Invalid voxel grid shape: "
                + str(voxel_grid.shape)
                + ", expected ([N], C, H, W, D)"
            )
        return False
    if voxel_grid.shape[-4] not in (1, 2, 4):
        if raise_error_if_not:
            raise ValueError(
                f"Invalid number of channels ({voxel_grid.shape[-1]}),"
                " expected 1, 2 (Value + alpha), or 4 (RGBA)"
            )
        return False
    return True
