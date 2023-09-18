"""Generates a voxel representation of a 3D object from a latent vector."""
import torch

import numpy as np

import voxel
import nnet


class ColourVoxelGenerator(torch.nn.Module):
    """Generates a voxel representation of a 3D object from a latent vector.

    Outputs either as a Voxel model or a mask from a random viewpoint (default).
    """

    def __init__(
        self,
        latent_size: int,
        base_channels: int,
        space_side_length: int,
        *,
        device="cuda",
        dropout: float = 0.0,
        initial_side: int = 4,
        output_voxels: bool = False,
    ):
        super().__init__()
        if space_side_length & (space_side_length - 1) != 0:
            raise ValueError("space_side_length must be a power of 2")

        self.output_voxel_shape = (
            space_side_length,
            space_side_length,
            space_side_length,
        )
        self.output_image_shape = (3, space_side_length, space_side_length)

        self._output_voxels = output_voxels

        self.device = device

        height, width, depth = (initial_side, initial_side, initial_side)

        cur_channels = (base_channels * space_side_length) // initial_side

        self.decoder = torch.nn.ModuleList(
            [
                torch.nn.Linear(latent_size, cur_channels * height * width * depth),
                torch.nn.LeakyReLU(),
                torch.nn.Unflatten(1, (cur_channels, height, width, depth)),
            ]
        )

        while (
            height < space_side_length
            and width < space_side_length
            and depth < space_side_length
        ):
            next_channels = max(cur_channels // 2, 1)

            self.decoder.append(
                torch.nn.Sequential(
                    torch.nn.Conv3d(cur_channels, cur_channels, 3, padding=1),
                    torch.nn.BatchNorm3d(cur_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout3d(dropout),
                    # torch.nn.ConvTranspose3d(
                    #     cur_channels,
                    #     next_channels,
                    #     4,
                    #     stride=2,
                    #     padding=1,
                    #     output_padding=0,
                    # ),
                    # torch.nn.BatchNorm3d(next_channels),
                    # torch.nn.LeakyReLU(),
                    # torch.nn.Dropout3d(dropout),
                    torch.nn.Upsample(scale_factor=2, mode="trilinear"),
                    torch.nn.ReflectionPad3d(1),
                    torch.nn.Conv3d(
                        cur_channels, next_channels, kernel_size=3, stride=1, padding=0
                    ),
                    torch.nn.BatchNorm3d(next_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout3d(dropout),
                )
            )

            cur_channels = next_channels

            height *= 2
            width *= 2
            depth *= 2

        self.output = torch.nn.Sequential(
            nnet.blocks.resnet.create_resnet_block_simple_3d(
                cur_channels,
                cur_channels,
                kernel_size=5,
                activation=torch.nn.LeakyReLU,
            ),
            # torch.nn.Conv3d(cur_channels, cur_channels, 3, padding=1),
            # torch.nn.BatchNorm3d(cur_channels),
            # torch.nn.LeakyReLU(),
            # torch.nn.Dropout3d(dropout),
            torch.nn.Conv3d(cur_channels, 4, 1),
            torch.nn.Sigmoid(),
        )

        self.decoder = torch.nn.Sequential(*self.decoder)

    def forward(
        self,
        encoded,
        *,
        return_voxels: bool | None = None,
        return_mask: bool | None = None,
    ):
        """Processes a latent vector through the model."""
        if return_voxels is None and return_mask is None:
            return_voxels = self._output_voxels
        elif return_voxels is None:
            return_voxels = not return_mask
        elif return_mask and return_voxels:
            raise ValueError("Can only return one of voxels or image")
        elif not return_mask and not return_voxels:
            raise ValueError("Must return one of voxels or image")

        angles = torch.rand(encoded.shape[0], 3, device=self.device) * np.pi * 2 - np.pi

        voxels = self.decoder(encoded)

        voxels = self.output(voxels)

        if return_voxels:
            return voxels

        images = []

        for voxel_set, angle_set in zip(voxels, angles):
            rot_mat = voxel.transform_torch.create_rotation_matrix(angle_set)
            channels = []
            for i in range(4):
                channels.append(
                    voxel.transform_torch.apply_transform(
                        voxel_set[i, :, :, :], rot_mat, centred=True
                    )
                )
            rotated = torch.stack(channels, dim=0)

            image = torch.zeros(self.output_image_shape, device=self.device)

            for level in reversed(range(rotated.shape[3])):
                alpha = rotated[3, :, :, level].unsqueeze(0)
                # print("Alpha:", alpha.shape)
                # print("Image:", image.shape)
                # print("Level:", rotated[:3, :, :, level].shape)
                image = image * (1 - alpha) + alpha * rotated[:3, :, :, level]

            image = 2 * image - 1

            images.append(image)

        return torch.stack(images, dim=0)

    def output_voxels(self):
        """Set the model to output voxels instead of masks."""
        self._output_voxels = True

    def output_masks(self):
        """Set the model to output masks instead of voxels."""
        self._output_voxels = False
