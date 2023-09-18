import numpy as np

import voxel

RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
ORANGE = (1, 0.6, 0.2, 1)
WHITE = (1, 1, 1, 1)
YELLOW = (1, 1, 0, 1)
BLACK = (0, 0, 0, 1)


def rubiks(side_length: int, space_size: tuple[int, int, int]) -> voxel.ColourVoxel:
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

    voxel_cuboid = np.zeros((*space_size, 4), dtype=np.float32)

    voxel_cuboid[top_edge:bottom_edge, left_edge:right_edge, front_edge, :] = RED
    voxel_cuboid[top_edge:bottom_edge, left_edge:right_edge, back_edge, :] = ORANGE
    voxel_cuboid[top_edge:bottom_edge, left_edge, front_edge:back_edge, :] = YELLOW
    voxel_cuboid[top_edge:bottom_edge, right_edge, front_edge:back_edge, :] = WHITE
    voxel_cuboid[top_edge, left_edge:right_edge, front_edge:back_edge, :] = BLUE
    voxel_cuboid[bottom_edge, left_edge:right_edge, front_edge:back_edge, :] = GREEN
    voxel_cuboid[top_edge, left_edge, front_edge:back_edge, :] = BLACK
    voxel_cuboid[top_edge, right_edge, front_edge:back_edge, :] = BLACK
    voxel_cuboid[bottom_edge, left_edge, front_edge:back_edge, :] = BLACK
    voxel_cuboid[bottom_edge, right_edge, front_edge:back_edge, :] = BLACK
    voxel_cuboid[top_edge:bottom_edge, left_edge, front_edge, :] = BLACK
    voxel_cuboid[top_edge:bottom_edge, right_edge, front_edge, :] = BLACK
    voxel_cuboid[top_edge:bottom_edge, left_edge, back_edge, :] = BLACK
    voxel_cuboid[top_edge:bottom_edge, right_edge, back_edge, :] = BLACK
    voxel_cuboid[top_edge, left_edge:right_edge, back_edge, :] = BLACK
    voxel_cuboid[bottom_edge, left_edge:right_edge, front_edge, :] = BLACK
    voxel_cuboid[top_edge, left_edge:right_edge, front_edge, :] = BLACK
    voxel_cuboid[bottom_edge, left_edge:right_edge, back_edge, :] = BLACK

    return voxel.ColourVoxel(voxel_cuboid)
