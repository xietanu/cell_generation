import torch

import numpy as np


class ConvToLinear(torch.nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        out_dim: int,
        *,
        activation: type[torch.nn.Module] = torch.nn.GELU
    ):
        super().__init__()

        self.main = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(in_shape, dtype=int), out_dim),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
