import torch
import numpy as np

import voxgrid


def create_image(
    voxel_grid: torch.Tensor | np.ndarray,
    device: torch.device = torch.device("cpu"),
    background: torch.Tensor | np.ndarray | None = None,
) -> np.ndarray | torch.Tensor:
    """Returns a 2D image of the voxel model."""

    if isinstance(voxel_grid, np.ndarray):
        if isinstance(background, torch.Tensor):
            raise ValueError(
                "Cannot use torch tensor as background for numpy voxel grid."
            )
        return create_image_from_numpy(voxel_grid, background=background)

    if isinstance(voxel_grid, torch.Tensor):
        if isinstance(background, np.ndarray):
            raise ValueError(
                "Cannot use numpy array as background for torch voxel grid."
            )
        return create_image_from_torch(voxel_grid, device=device, background=background)


def create_image_from_numpy(
    voxel_grid: np.ndarray, background: np.ndarray | None = None
):
    n_channels = voxgrid.get_n_channels(voxel_grid)
    n_images = voxel_grid.shape[0] if voxel_grid.ndim > 4 else 1

    if background is None:
        image = np.zeros(
            (
                n_images,
                voxel_grid.shape[0],
                voxel_grid.shape[1],
                max(n_channels - 1, 1),
            ),
            dtype=np.float32,
        )
    else:
        image = background.copy()

    alphas = voxgrid.get_alpha(voxel_grid)[..., np.newaxis]
    values = voxgrid.get_values(voxel_grid)

    if values.ndim < alphas.ndim:
        values = values[..., np.newaxis]

    if values.ndim < 5:
        values = values[np.newaxis, ...]
        alphas = alphas[np.newaxis, ...]

    depth = alphas.shape[-2]

    for level in reversed(range(depth)):
        alpha = alphas[..., level, :]
        value = values[..., level, :]
        image = image * (1 - alpha) + alpha * value

    return image.squeeze()


def create_image_from_torch(
    voxel_grid: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    background: torch.Tensor | None = None,
):
    n_channels = voxgrid.get_n_channels(voxel_grid)
    n_images = voxel_grid.shape[0] if voxel_grid.ndim > 4 else 1

    if background is None:
        image = torch.zeros(
            (
                n_images,
                max(n_channels - 1, 1),
                voxel_grid.shape[2],
                voxel_grid.shape[3],
            ),
            dtype=torch.float32,
            device=device,
        )
    else:
        image = background.to(device)

    alphas = voxgrid.get_alpha(voxel_grid).unsqueeze(-4)  # type: ignore
    values = voxgrid.get_values(voxel_grid).to(device)  # type: ignore

    if values.ndim < alphas.ndim:
        values = values.unsqueeze(-4)  # type:

    if values.ndim < 5:
        values = values.unsqueeze(0)
        alphas = alphas.unsqueeze(0)

    for level in range(voxel_grid.shape[-1]):
        alpha = alphas[..., level]
        value = values[..., level]
        image = image * (1 - alpha) + alpha * value

    while image.ndim > voxel_grid.ndim - 1:
        image = image.squeeze(0)

    return image
