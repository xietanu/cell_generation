import torch

import voxel.transform_torch


def apply_transform(
    voxel_object: torch.Tensor,
    transform_matrix: torch.Tensor,
    centred: bool = False,
    bg_col: float = 0.0,
) -> torch.Tensor:
    """
    Rotate a 3D object around the z-axis.
    """
    voxel_object = voxel_object.cuda()

    if voxel_object.ndim != 3:
        raise ValueError("Object must be 3D")

    if transform_matrix.shape != (4, 4):
        raise ValueError("Transform matrix must be 4x4")

    if centred:
        transform_matrix = voxel.transform_torch.centre_transform_matrix(
            transform_matrix, voxel_object.shape
        )

    target_coords = _create_target_coords(voxel_object)

    transformed_coords = _create_transformed_coords(
        transform_matrix.to(float), target_coords.to(float)
    )

    mask = _create_in_bounds_mask(voxel_object, transformed_coords)

    transformed_coords = transformed_coords[:, mask]
    target_coords = target_coords[:, mask]

    transformed_object = (
        torch.ones(voxel_object.shape, dtype=voxel_object.dtype).cuda() * bg_col
    )

    transformed_object[
        target_coords[0, :],
        target_coords[1, :],
        target_coords[2, :],
    ] = voxel_object[
        transformed_coords[0, :],
        transformed_coords[1, :],
        transformed_coords[2, :],
    ]

    return transformed_object


def _create_in_bounds_mask(voxel_object, transformed_coords):
    mask = torch.logical_and(
        torch.logical_and(
            torch.logical_and(
                transformed_coords[0, :, :] >= 0,
                transformed_coords[0, :, :] < voxel_object.shape[0],
            ),
            torch.logical_and(
                transformed_coords[1, :, :] >= 0,
                transformed_coords[1, :, :] < voxel_object.shape[1],
            ),
        ),
        torch.logical_and(
            transformed_coords[2, :, :] >= 0,
            transformed_coords[2, :, :] < voxel_object.shape[2],
        ),
    ).cuda()

    return mask


def _create_transformed_coords(
    transform_matrix: torch.Tensor, target_coords: torch.Tensor
):
    transformed_coords = (
        (torch.linalg.inv(transform_matrix) @ target_coords.reshape((4, -1)))
        .reshape(target_coords.shape)
        .round()
        .to(int)
    ).cuda()

    return transformed_coords


def _create_target_coords(voxel_object: torch.Tensor):
    target_coords_x, target_coords_y, target_coords_z = torch.meshgrid(
        torch.arange(0, voxel_object.shape[0]),
        torch.arange(0, voxel_object.shape[1]),
        torch.arange(0, voxel_object.shape[2]),
    )
    target_coords = torch.stack(
        (
            target_coords_x.cuda(),
            target_coords_y.cuda(),
            target_coords_z.cuda(),
            torch.ones(voxel_object.shape).cuda(),
        ),
        dim=0,
    ).to(int)

    return target_coords
