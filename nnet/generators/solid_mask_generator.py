"""Generates a voxel representation of a 3D object from a latent vector."""
import torch

import numpy as np

import voxel


class SolidMaskGenerator(torch.nn.Module):
    """Generates a voxel representation of a 3D object from a latent vector.

    Outputs either as a Voxel model or a mask from a random viewpoint (default).
    """

    def __init__(
        self,
        voxel_generator: torch.nn.Module,
        *,
        device="cpu",
    ):
        super().__init__()
        self.voxel_generator = voxel_generator
        self.device = device

    def cpu(self):
        """Sets the model to use the CPU."""
        super().cpu()
        self.device = "cpu"

    def cuda(self, device=None):
        """Sets the model to use the GPU."""
        super().cuda(device=device)
        self.device = "cuda"

    def forward(
        self,
        encoded: torch.Tensor,
    ):
        """Processes a latent vector through the model."""
        angles = torch.rand(encoded.shape[0], 3, device=self.device) * np.pi * 2 - np.pi

        voxels = self.voxel_generator(encoded)

        views = []

        for voxel_set, angle_set in zip(voxels, angles):
            rot_mat = voxel.transform_torch.create_rotation_matrix(angle_set)
            views.append(
                voxel.transform_torch.apply_transform(
                    voxel_set.squeeze(), rot_mat, centred=True
                ).unsqueeze(0)
            )

        views = torch.stack(views)

        masks = torch.max(views, dim=4)[0]

        masks = masks * 2 - 1

        return masks
