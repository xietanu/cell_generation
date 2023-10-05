import torch

import voxgrid


class RandomImageFromVoxGrid(torch.nn.Module):
    """Creates images from random angles from the voxel grids supplied.

    By default these are layered over a black background, but a background image
    tensor can be supplied.
    """

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.device = device

    def cpu(self) -> None:
        """Sets the model to use the CPU."""
        super().cpu()
        self.device = torch.device("cpu")

    def cuda(self, device: torch.device = torch.device("cuda")) -> None:
        """Sets the model to use the GPU."""
        super().cuda(device=device)
        self.device = device

    def forward(
        self, in_tensor: torch.Tensor, background: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Creates an image of the input voxel grid tensor from a random angle.

        Parameters
        ----------
        in_tensor : torch.Tensor
            Input voxel grid tensor. Should be of shape (N, C, H, W, D).
        background : torch.Tensor | None, optional
            Background image tensor for the voxel grid to be layered over.
            Should be of shape (C', H, W, D), where C' is 1 for 1 or 2 channel voxel grids
            and 3 for 4 channel voxel grids.
            By default None, will use a flat black background.

        Returns
        -------
        torch.Tensor
            The output image tensor.
        """

        rotated = voxgrid.transform.randomly_rotated(in_tensor, device=self.device)

        return voxgrid.convert.create_image(rotated, device=self.device, background=background)  # type: ignore
