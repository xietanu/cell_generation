"""Create a cuboid with given side lengths inside of a larger space."""
import numpy as np

import voxgrid


def pyramid(
    sides: tuple[int, int, int], space_size: tuple[int, int, int]
) -> voxgrid.SolidVoxel:
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

    back_lr = np.random.randint(0, right_edge - left_edge)
    back_tb = np.random.randint(0, bottom_edge - top_edge)

    voxel_cuboid = np.zeros(space_size, dtype=np.float32)

    for i, layer in enumerate(range(front_edge, back_edge)):
        percent = i / (back_edge - front_edge)
        cur_back_lr = int(back_lr * percent)
        width_lr = int((right_edge - left_edge) * (1 - percent))
        cur_back_tb = int(back_tb * percent)
        height_tb = int((bottom_edge - top_edge) * (1 - percent))

        voxel_cuboid[
            top_edge + cur_back_tb : top_edge + cur_back_tb + height_tb,
            left_edge + cur_back_lr : left_edge + cur_back_lr + width_lr,
            layer,
        ] = 1

    return voxgrid.SolidVoxel(voxel_cuboid)
