import numpy as np

import voxgrid

RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
ORANGE = (1, 0.6, 0.2, 1)
WHITE = (1, 1, 1, 1)
YELLOW = (1, 1, 0, 1)
BLACK = (0, 0, 0, 1)


def rubiks(side_length: int, space_size: tuple[int, int, int]) -> voxgrid.VoxGrid:
    """Create a cuboid with given side lengths inside of a larger space."""
    space_height, space_width, space_depth = space_size

    top_edge = (space_height - side_length) // 2
    bottom_edge = top_edge + side_length
    left_edge = (space_width - side_length) // 2
    right_edge = left_edge + side_length
    front_edge = (space_depth - side_length) // 2
    back_edge = front_edge + side_length

    voxel_cuboid = np.zeros((*space_size, 4), dtype=np.float32)

    voxel_cuboid[
        top_edge:bottom_edge,
        left_edge:right_edge,
        front_edge:back_edge,
    ] = full_rubiks((side_length, side_length, side_length))

    voxel_cuboid[
        top_edge + 1 : bottom_edge - 1,
        left_edge + 1 : right_edge - 1,
        front_edge + 1 : back_edge - 1,
    ] = full_rubiks((side_length - 2, side_length - 2, side_length - 2))

    voxel_cuboid[
        top_edge + 2 : bottom_edge - 2,
        left_edge + 2 : right_edge - 2,
        front_edge + 2 : back_edge - 2,
    ] = full_rubiks((side_length - 4, side_length - 4, side_length - 4))

    return voxgrid.VoxGrid(voxel_cuboid)


def full_rubiks(space_size):
    voxel_cuboid = np.zeros((*space_size, 4), dtype=np.float32)

    voxel_cuboid[0, :, :, :] = WHITE
    voxel_cuboid[-1, :, :, :] = YELLOW
    voxel_cuboid[:, 0, :, :] = ORANGE
    voxel_cuboid[:, -1, :, :] = RED
    voxel_cuboid[:, :, 0, :] = GREEN
    voxel_cuboid[:, :, -1, :] = BLUE

    voxel_cuboid[0, 0, :, :] = WHITE
    voxel_cuboid[-1, 0, :, :] = WHITE
    voxel_cuboid[0, -1, :, :] = WHITE
    voxel_cuboid[-1, -1, :, :] = WHITE

    voxel_cuboid[0, :, 0, :] = WHITE
    voxel_cuboid[-1, :, 0, :] = WHITE
    voxel_cuboid[0, :, -1, :] = WHITE
    voxel_cuboid[-1, :, -1, :] = WHITE

    voxel_cuboid[:, 0, 0, :] = WHITE
    voxel_cuboid[:, -1, 0, :] = WHITE
    voxel_cuboid[:, 0, -1, :] = WHITE
    voxel_cuboid[:, -1, -1, :] = WHITE

    return voxel_cuboid
