import torch


def centre_transform_matrix(
    transform_matrix: torch.Tensor, object_shape: tuple[int, int, int]
) -> torch.Tensor:
    if len(object_shape) != 3:
        raise ValueError("Object must be 3D")

    centre = torch.Tensor([*object_shape]) / 2 - 0.5  # type: ignore

    centring_matrix = torch.eye(4).cuda()
    centring_matrix[0:3, 3] = -centre

    reverse_matrix = torch.eye(4).cuda()
    reverse_matrix[0:3, 3] = centre

    return reverse_matrix @ transform_matrix @ centring_matrix
