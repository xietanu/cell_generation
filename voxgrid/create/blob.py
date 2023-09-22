import voxgrid

import numpy as np


def blob(
    space_size: tuple[int, int, int], n_spheres: int, sphere_size_range: tuple[int, int]
) -> np.ndarray:
    """
    Create a blob made of several spheres
    """
    voxel_space = np.zeros(space_size, dtype=np.uint8)

    for _ in range(n_spheres):
        r = np.random.randint(*sphere_size_range)
        x = np.random.randint(2 * r, space_size[0] - r * 2)
        y = np.random.randint(2 * r, space_size[1] - r * 2)
        z = np.random.randint(2 * r, space_size[2] - r * 2)

        voxel_space[x - r : x + r, y - r : y + r, z - r : z + r] = np.maximum(
            voxel_space[x - r : x + r, y - r : y + r, z - r : z + r],
            voxgrid.create.sphere(r, border=0),
        )

    return voxel_space
