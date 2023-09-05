import numpy as np


def centre_transform_matrix(
    transform_matrix: np.ndarray, object_shape: np.ndarray | tuple[int, int, int]
) -> np.ndarray:
    if len(object_shape) != 3:
        raise ValueError("Object must be 3D")

    centre = np.array(object_shape) / 2 - 0.5  # type: ignore

    centring_matrix = np.array(
        [
            [1, 0, 0, -centre[0]],
            [0, 1, 0, -centre[1]],
            [0, 0, 1, -centre[2]],
            [0, 0, 0, 1],
        ]
    )

    reverse_matrix = np.array(
        [[1, 0, 0, centre[0]], [0, 1, 0, centre[1]], [0, 0, 1, centre[2]], [0, 0, 0, 1]]
    )

    return reverse_matrix @ transform_matrix @ centring_matrix
