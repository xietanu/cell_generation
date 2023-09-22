"""Protocol for meta-image classes."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import matplotlib.figure


class MetaImage(Protocol):
    """Protocol for meta-image classes."""

    title: str | None

    @property
    def shape(self):
        """Returns the shape of the image."""

    def as_array(self) -> np.ndarray:  # type: ignore
        """Returns the image as a numpy array."""

    def as_tensor(self) -> torch.Tensor:  # type: ignore
        """Returns the image as a PyTorch tensor."""

    def plot(
        self, figsize: tuple[float, float] | None = None
    ) -> matplotlib.figure.Figure:  # type: ignore
        """Plots the image as a matplotlib figure."""

    def plot_as_subfigure(
        self,
        fig,
        subplot_shape: tuple[int, int],
        index: int,
        **kwargs,
    ):
        """Plots the image as a matplotlib subfigure."""

    def copy(self) -> MetaImage:  # type: ignore
        """Returns a copy of the class."""
