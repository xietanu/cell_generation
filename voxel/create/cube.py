import numpy as np


def cube(size: int) -> np.ndarray:
    """
    Create a cube of size `size` with space around for rotation.
    """
    border = size // 2

    space_size = size + 2 * border

    voxel_cube = np.zeros((space_size, space_size, space_size), dtype=np.uint8)
    voxel_cube[border:-border, border:-border, border:-border] = 255

    return voxel_cube


"""Create a cuboid with given side lengths inside of a larger space."""
import numpy as np

import voxel


def cuboid(side_length: int, space_size: tuple[int, int, int]) -> voxel.TranspVoxel:
    """Create a cuboid with given side lengths inside of a larger space."""
    if (side_length - 4) % 3 != 0:
        raise ValueError("Side length must be 4 + 3n")

    space_height, space_width, space_depth = space_size

    top_edge = (space_height - side_length) // 2
    bottom_edge = top_edge + side_length
    left_edge = (space_width - side_length) // 2
    right_edge = left_edge + side_length
    front_edge = (space_depth - side_length) // 2
    back_edge = front_edge + side_length

    voxel_cuboid = np.zeros(space_size, dtype=np.uint8)
    voxel_cuboid[top_edge:bottom_edge, left_edge:right_edge, front_edge:back_edge] = 1

    return voxel.TranspVoxel(voxel_cuboid)
