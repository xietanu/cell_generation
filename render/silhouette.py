import numpy as np


def silhouette(voxel_object: np.ndarray) -> np.ndarray:
    """
    Create a silhouette from a 3D object.
    """
    if voxel_object.ndim != 3:
        raise ValueError("Object must be 3D")

    return np.max(voxel_object, axis=2)
