import numpy as np


def create_background_mask(voxel_object: np.ndarray) -> np.ndarray:
    """
    Create a background mask from a 3D object.
    """
    if voxel_object.ndim != 3:
        raise ValueError("Object must be 3D")

    background_mask = np.max(voxel_object, axis=2) == 0

    return background_mask
