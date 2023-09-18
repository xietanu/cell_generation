"""Generates a voxel representation of a 3D object from a latent vector."""
import torch

import numpy as np

import voxel


class TransMaskGenerator(torch.nn.Module):
    """Generates a voxel representation of a 3D object from a latent vector.

    Outputs either as a Voxel model or a mask from a random viewpoint (default).
    """

    def __init__(
        self,
        voxel_generator: torch.nn.Module,
        output_img_shape: tuple[int, int],
        *,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.voxel_generator = voxel_generator
        self.alpha_adjust = np.log2(output_img_shape[0]) + 2
        self.output_image_shape = output_img_shape

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
        encoded,
    ):
        """Processes a latent vector through the model."""

        angles = torch.rand(encoded.shape[0], 3, device=self.device) * np.pi * 2 - np.pi

        voxels = self.voxel_generator(encoded)

        voxels = voxels**self.alpha_adjust

        views = []

        for voxel_set, angle_set in zip(voxels, angles):
            rot_mat = voxel.transform_torch.create_rotation_matrix(angle_set)
            views.append(
                voxel.transform_torch.apply_transform(
                    voxel_set.squeeze(), rot_mat, centred=True
                ).unsqueeze(0)
            )

        views = torch.stack(views)

        images = torch.zeros(
            (views.shape[0], 1, *self.output_image_shape),
            device=self.device,
            dtype=torch.float32,
        )

        for level in reversed(range(views.shape[4])):
            alpha = views[:, :, :, :, level]
            # print("Alpha:", alpha.shape)
            # print("Image:", image.shape)
            # print("Level:", rotated[:3, :, :, level].shape)
            images = images * (1 - alpha) + alpha

        images = 2 * images - 1

        return images
