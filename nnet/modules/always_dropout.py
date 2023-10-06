import torch


class AlwaysDropout(torch.nn.Module):
    """Module that always performs a convolutional dropout."""

    def __init__(self, probability: float = 0.1, device: str | torch.device = "cuda"):
        super().__init__()
        self.probability = probability
        self.device = device

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Adjusts the alpha values of a voxel grid to assist with training."""
        mask = torch.bernoulli(
            torch.ones((in_tensor.shape[0], in_tensor.shape[1]) + (1,) * len(in_tensor.shape[2:]))
            * (1 - self.probability),
        ).to(self.device)

        return in_tensor * mask
