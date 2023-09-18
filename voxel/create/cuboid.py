"""Create a cuboid with given side lengths inside of a larger space."""
import numpy as np

import voxel


def cuboid(
    sides: tuple[int, int, int], space_size: tuple[int, int, int]
) -> voxel.TranspVoxel:
    """Create a cuboid with given side lengths inside of a larger space."""
    height, width, depth = sides
    space_height, space_width, space_depth = space_size

    if space_height < height or space_width < width or space_depth < depth:
        raise ValueError("Space must be larger than cuboid")

    top_edge = (space_height - height) // 2
    bottom_edge = top_edge + height
    left_edge = (space_width - width) // 2
    right_edge = left_edge + width
    front_edge = (space_depth - depth) // 2
    back_edge = front_edge + depth

    voxel_cuboid = np.zeros(space_size, dtype=np.uint8)
    voxel_cuboid[top_edge:bottom_edge, left_edge:right_edge, front_edge:back_edge] = 1

    return voxel.TranspVoxel(voxel_cuboid)
