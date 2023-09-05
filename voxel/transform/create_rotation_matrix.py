import numpy as np


def create_rotation_matrix(
    x_angle: float, y_angle: float, z_angle: float
) -> np.ndarray:
    """
    Create a rotation matrix around the x-axis.
    """
    rotation_matrix = (
        create_z_rotation_matrix(z_angle)
        @ create_y_rotation_matrix(y_angle)
        @ create_x_rotation_matrix(x_angle)
    )

    return rotation_matrix


def create_x_rotation_matrix(angle: float) -> np.ndarray:
    """
    Create a rotation matrix around the x-axis.
    """
    rotation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )

    return rotation_matrix


def create_y_rotation_matrix(angle: float) -> np.ndarray:
    """
    Create a rotation matrix around the y-axis.
    """
    rotation_matrix = np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )

    return rotation_matrix


def create_z_rotation_matrix(angle: float) -> np.ndarray:
    """
    Create a rotation matrix around the z-axis.
    """
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    return rotation_matrix
