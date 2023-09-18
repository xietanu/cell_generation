from typing import Protocol

import torch


class BlockFactory(Protocol):
    def __call__(self, in_dim: int, out_dim: int, **kwargs) -> torch.nn.Module:  # type: ignore
        """Returns a block with the given input and output dimensions."""


class ResizeBlockFactory(Protocol):
    def __call__(self, in_dim: int, out_dim: int, *, factor: int = 2, **kwargs) -> torch.nn.Module:  # type: ignore
        """Returns a block with the given input and output dimensions."""
