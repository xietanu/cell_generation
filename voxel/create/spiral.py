"""Create a cuboid with given side lengths inside of a larger space."""
import numpy as np

import cv2
import voxel


def spiral(
    thickness_r: float, width_r: float, space_size: tuple[int, int, int]
) -> voxel.TranspVoxel:
    """Create a cuboid with given side lengths inside of a larger space."""
    top = space_size[2] // 4
    bottom = top * 3 + 1

    voxels = np.zeros(space_size, dtype=np.uint8)

    for i, layer_index in enumerate(range(top, bottom)):
        x_centre = space_size[0] // 2 + int(
            np.sin(1.25 * i / (bottom - top) * 2 * np.pi) * -width_r
        )
        y_centre = space_size[1] // 2 + int(
            np.cos(1.25 * i / (bottom - top) * 2 * np.pi) * width_r
        )

        voxels[:, :, layer_index] = cv2.circle(
            np.zeros(space_size[:2], dtype=np.uint8),
            (x_centre, y_centre),
            thickness_r,
            1,
            -1,
        )

    return voxel.TranspVoxel(voxels)
