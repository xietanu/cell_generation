import torch


class AlwaysDropout(torch.nn.Module):
    """Module that alwazs performs a convolutional dropout."""

    def __init__(self, probability: float = 0.1):
        super().__init__()
        self.probability = probability

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Adjusts the alpha values of a voxel grid to assist with training."""
        mask = torch.bernoulli(
            torch.ones((1, in_tensor.shape[1]) + (1,) * len(in_tensor.shape[2:]))
            * (1 - self.probability)
        )

        return in_tensor * mask
