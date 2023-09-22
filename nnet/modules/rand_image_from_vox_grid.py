import torch

import voxgrid


class RandomImageFromVoxGrid(torch.nn.Module):
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

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        rotated = voxgrid.transform.randomly_rotated(in_tensor, device=self.device)

        return voxgrid.convert.create_image(rotated, device=self.device)  # type: ignore
