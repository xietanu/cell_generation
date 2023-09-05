import numpy as np

import render.utils


def depth_map(voxel_object: np.ndarray) -> np.ndarray:
    """
    Create a depth map from a 3D object.
    """
    if voxel_object.ndim != 3:
        raise ValueError("Object must be 3D")

    depth = np.argmax(voxel_object, axis=2)
    
    depth = np.max(depth) - depth

    depth[render.utils.create_background_mask(voxel_object)] = 0

    return depth
