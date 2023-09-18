import torch
import numpy as np


class LinearToConv(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: tuple[int, ...],
        *,
        activation: type[torch.nn.Module] = torch.nn.GELU,
    ):
        super().__init__()
        self.in_dim = in_dim

        self.main = torch.nn.Sequential(
            torch.nn.Linear(in_dim, np.prod(out_shape, dtype=int)),
            activation(),
            torch.nn.Unflatten(1, out_shape),
        )

        self.main = torch.nn.Sequential(*self.main)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
