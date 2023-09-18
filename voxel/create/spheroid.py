import itertools

import numpy as np

import voxel


def spheroid(length_multi, r, space_size) -> voxel.TranspVoxel:
    """
    Create a sphere of size `size` with space around for rotation.
    """
    voxel_sphere = np.zeros(space_size, dtype=np.uint8)

    for x, y, z in itertools.product(
        range(space_size[0]), range(space_size[1]), range(space_size[2])
    ):
        if (x - space_size[0] // 2) ** 2 + (y - space_size[1] // 2) ** 2 + (
            (z - space_size[2] // 2) * 1 / length_multi
        ) ** 2 <= r**2:
            voxel_sphere[x, y, z] = 1

    return voxel.TranspVoxel(voxel_sphere)
