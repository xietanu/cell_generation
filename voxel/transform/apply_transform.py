import numpy as np

import voxel.transform


def apply_transform(
    voxel_object: np.ndarray, transform_matrix: np.ndarray, centred: bool = False
) -> np.ndarray:
    """
    Rotate a 3D object around the z-axis.
    """
    if voxel_object.ndim != 3:
        raise ValueError("Object must be 3D")

    if transform_matrix.shape != (4, 4):
        raise ValueError("Transform matrix must be 4x4")

    if centred:
        transform_matrix = voxel.transform.centre_transform_matrix(
            transform_matrix, voxel_object.shape
        )

    target_coords = _create_target_coords(voxel_object)

    transformed_coords = _create_transformed_coords(transform_matrix, target_coords)

    mask = _create_in_bounds_mask(voxel_object, transformed_coords)

    transformed_coords = transformed_coords[:, mask]
    target_coords = target_coords[:, mask]

    transformed_object = np.zeros(voxel_object.shape, dtype=voxel_object.dtype)

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
    mask = np.logical_and(
        np.logical_and(
            np.logical_and(
                transformed_coords[0, :, :] >= 0,
                transformed_coords[0, :, :] < voxel_object.shape[0],
            ),
            np.logical_and(
                transformed_coords[1, :, :] >= 0,
                transformed_coords[1, :, :] < voxel_object.shape[1],
            ),
        ),
        np.logical_and(
            transformed_coords[2, :, :] >= 0,
            transformed_coords[2, :, :] < voxel_object.shape[2],
        ),
    )

    return mask


def _create_transformed_coords(transform_matrix, target_coords):
    transformed_coords = (
        (np.linalg.inv(transform_matrix) @ target_coords.reshape((4, -1)))
        .reshape(target_coords.shape)
        .round(0)
        .astype(int)
    )

    return transformed_coords


def _create_target_coords(voxel_object):
    target_coords = np.indices(voxel_object.shape)
    target_coords = np.vstack(
        (target_coords, np.ones((1, *voxel_object.shape)))
    ).astype(int)

    return target_coords
