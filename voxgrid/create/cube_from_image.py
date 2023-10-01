import numpy as np
import cv2

import voxgrid


def cube_from_image(
    image_path: str, space_size: tuple[int, int, int], grey: bool = False
) -> voxgrid.VoxGrid:
    """Create a cuboid with given side lengths inside of a larger space."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grey else cv2.IMREAD_COLOR)

    image = image.astype(np.float32) / 255

    if grey:
        image = image[..., np.newaxis]

    if image.shape[0] // 6 != image.shape[1]:
        raise ValueError("Image must be 6 square faces stacked vertically.")

    side_length = image.shape[1]
    space_height, space_width, space_depth = space_size

    top_edge = (space_height - side_length) // 2
    bottom_edge = top_edge + side_length
    left_edge = (space_width - side_length) // 2
    right_edge = left_edge + side_length
    front_edge = (space_depth - side_length) // 2
    back_edge = front_edge + side_length

    channels = 2 if grey else 4

    voxel_cuboid = np.zeros((*space_size, channels), dtype=np.float32)

    voxel_cuboid[
        top_edge:bottom_edge,
        left_edge:right_edge,
        front_edge:back_edge,
    ] = image_cube(image, channels)

    return voxgrid.VoxGrid(voxel_cuboid)


def image_cube(image: np.ndarray, channels: int) -> np.ndarray:
    side_length = image.shape[1]

    voxel_cube = np.zeros(
        (side_length, side_length, side_length, channels), dtype=np.float32
    )

    voxel_cube[:, :, :, -1] = 1

    image_sides = [
        image[i * side_length : i * side_length + side_length, :] for i in range(6)
    ]

    voxel_cube[0, :, :, : channels - 1] = image_sides[0]
    voxel_cube[-1, :, :, : channels - 1] = image_sides[1]
    voxel_cube[:, 0, :, : channels - 1] = image_sides[2]
    voxel_cube[:, -1, :, : channels - 1] = image_sides[3]
    voxel_cube[:, :, 0, : channels - 1] = image_sides[4]
    voxel_cube[:, :, -1, : channels - 1] = image_sides[5]

    return voxel_cube
