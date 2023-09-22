import numpy as np
import torch

import voxgrid


def affine(
    voxel_grids: torch.Tensor | np.ndarray,
    affine_matrices: torch.Tensor | np.ndarray,
    *,
    return_torch: bool | None = None,
    mode: str = "bilinear",
    padding_mode: str = "border",
    device: torch.device = torch.device("cpu"),
):
    """Applies an affine transform to a set of voxels.

    Defaults to returning a torch tensor if the input is a torch tensor,
    and a numpy array if the input is a numpy array. This can be overridden
    by setting `return_torch` to `True` or `False`.
    """
    if return_torch is None:
        return_torch = isinstance(voxel_grids, torch.Tensor)

    out_dims = voxel_grids.ndim

    voxgrid.is_valid_voxel_grid(voxel_grids, raise_error_if_not=True)

    if isinstance(voxel_grids, np.ndarray):
        voxel_grids = voxgrid.convert.numpy_to_torch(voxel_grids)

    while voxel_grids.ndim < 5:
        voxel_grids = voxel_grids.unsqueeze(0)

    if isinstance(affine_matrices, np.ndarray):
        affine_matrices = torch.from_numpy(affine_matrices).float()

    if affine_matrices.shape[-2:] == (4, 4):
        affine_matrices = affine_matrices[..., :3, :4]
    elif affine_matrices.shape[-2:] != (3, 4):
        raise ValueError(
            "Invalid affine matrix shape: "
            + str(affine_matrices.shape)
            + ", expected ([N], 3, 4)"
        )
    if affine_matrices.ndim == 2:
        affine_matrices = affine_matrices.unsqueeze(0)
    elif affine_matrices.ndim != 3:
        raise ValueError(
            "Invalid affine matrix shape: "
            + str(affine_matrices.shape)
            + ", expected ([N], 3, 4)"
        )

    affine_grids = torch.nn.functional.affine_grid(
        affine_matrices,
        list(voxel_grids.shape),
        align_corners=False,
    )

    affine_grids = affine_grids.to(device)
    voxel_grids = voxel_grids.to(device)

    views = torch.nn.functional.grid_sample(
        voxel_grids,
        affine_grids,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )

    views = views.clamp(0, 1)

    while views.ndim > out_dims:
        views = views[0]

    if return_torch:
        return views
    return voxgrid.convert.torch_to_numpy(views)
