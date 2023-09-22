import numpy as np
import torch

import voxgrid


def rotated(
    voxel_grid: np.ndarray | torch.Tensor,
    angles: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray | torch.Tensor:
    """Returns a rotated version of the voxel model."""
    if np.allclose(angles, 0):
        return voxel_grid

    if angles.shape[-1] != 3:
        raise ValueError(
            "Invalid angle shape: " + str(angles.shape) + ", expected ([N], 3)"
        )

    if angles.ndim == 1:
        rotate_matrices = voxgrid.transform.create_rotation_matrix(*angles)
    else:
        rotate_matrices = np.array(
            [
                voxgrid.transform.create_rotation_matrix(*angle_set)
                for angle_set in angles
            ]
        )

    return voxgrid.transform.affine(voxel_grid, rotate_matrices, device=device)


def randomly_rotated(
    voxel_grid: np.ndarray | torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    """Returns a randomly rotated version of the voxel model."""
    if voxel_grid.ndim < 5:
        angles = np.random.uniform(-np.pi, np.pi, 3)
    else:
        angles = np.random.uniform(-np.pi, np.pi, (voxel_grid.shape[0], 3))
    return rotated(
        voxel_grid,
        angles,
        device=device,
    )
