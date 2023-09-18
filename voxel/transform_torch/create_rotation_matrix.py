import torch


def create_rotation_matrix(
    angles: torch.Tensor,
) -> torch.Tensor:
    """
    Create a rotation matrix around the x-axis.
    """
    rotation_matrix = (
        create_z_rotation_matrix(angles[2])
        @ create_y_rotation_matrix(angles[1])
        @ create_x_rotation_matrix(angles[0])
    )

    return rotation_matrix


def create_x_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix around the x-axis.
    """
    rotation_matrix = torch.eye(4).cuda()
    rotation_matrix[1:3, 1:3] = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )

    return rotation_matrix


def create_y_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix around the y-axis.
    """
    rotation_matrix = torch.eye(4).cuda()
    rotation_matrix[0, 0] = torch.cos(angle)
    rotation_matrix[0, 2] = torch.sin(angle)
    rotation_matrix[2, 0] = -torch.sin(angle)
    rotation_matrix[2, 2] = torch.cos(angle)

    return rotation_matrix


def create_z_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix around the z-axis.
    """
    rotation_matrix = torch.eye(4).cuda()
    rotation_matrix[0:2, 0:2] = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )

    return rotation_matrix
