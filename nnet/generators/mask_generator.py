"""Module that generates an image from a latent vector using a voxel generator."""
import torch

import nnet


class MaskGenerator(torch.nn.Module):
    """Module that generates an image from a latent vector using a voxel generator."""

    def __init__(
        self,
        model_gen: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        background_gen: torch.nn.Module | None = None,
        foreground_gen: torch.nn.Module | None = None,
    ):
        super().__init__()

        self.model_gen = model_gen
        self.mask_from_voxels = nnet.modules.RandomImageFromVoxGrid(device)
        self.background_gen = background_gen
        self.foreground_gen = foreground_gen

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """se a latent vector to generate a voxel model, and from that an image.

        If a background generator is specified, it is used to generate a background.
        Similarly, if a foreground generator is specified, it is used to generate a
        foreground.

        Parameters
        ----------
        in_tensor : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Image generated from the latent vector.
        """
        voxel_grid = self.model_gen(in_tensor)

        background = self.background_gen(in_tensor) if self.background_gen else None

        mask = self.mask_from_voxels(voxel_grid, background=background)
        if self.foreground_gen is not None:
            foreground = self.foreground_gen(in_tensor)

            alpha = foreground[:, -1, :, :]  # ** 3
            alpha = alpha.unsqueeze(1)
            if foreground.shape[1] == 1:
                mask = mask * (1 - alpha) + alpha
            else:
                value = foreground[:, :-1, :, :]
                mask = mask * (1 - alpha) + alpha * value
        mask = mask * 2 - 1
        return mask
