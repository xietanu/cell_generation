import itertools

import numpy as np


def sphere(r: int) -> np.ndarray:
    """
    Create a sphere of size `size` with space around for rotation.
    """
    border = r // 2

    space_size = 2 * r + 2 * border

    voxel_sphere = np.zeros((space_size, space_size, space_size), dtype=np.uint8)

    for x, y, z in itertools.product(range(space_size), repeat=3):
        if (x - r - border) ** 2 + (y - r - border) ** 2 + (
            z - r - border
        ) ** 2 <= r**2:
            voxel_sphere[x, y, z] = 255

    return voxel_sphere
