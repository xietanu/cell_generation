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
