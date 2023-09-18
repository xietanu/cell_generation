"""Mask image, with range of -1 to 1."""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.figure

import protocols


class ColourImage(protocols.MetaImage):
    """Mask image, with range of -1 to 1."""

    title: str | None

    def __init__(self, data: np.ndarray, *, title: str | None = None):
        if data.ndim != 3:
            raise ValueError("Mask must be 3D")
        if data.max() - data.min() == 0:
            data[:, :] = -1
        else:
            data = (
                2 * (data.astype(np.float32) - data.min()) / (data.max() - data.min())
                - 1
            )
        self.array = data.astype(np.float32)
        self.title = title

    @property
    def shape(self):
        """Returns the shape of the mask."""
        return self.array.shape

    def as_array(self) -> np.ndarray:
        """Returns the mask as a numpy array."""
        return self.array.copy()

    def as_tensor(self) -> torch.Tensor:
        """Returns the mask as a PyTorch tensor."""
        return torch.from_numpy(self.array).permute(2, 0, 1)

    @classmethod
    def from_array(cls, array, *, title: str | None = None) -> ColourImage:
        """Creates a mask from a numpy array."""
        return cls(array, title=title)

    @classmethod
    def from_tensor(cls, tensor, *, title: str | None = None) -> ColourImage:
        """Creates a mask from a PyTorch tensor."""
        return cls(
            tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy(), title=title
        )

    def plot(
        self, figsize: tuple[float, float] | None = None
    ) -> matplotlib.figure.Figure:
        """Plots the mask as a matplotlib figure."""
        fig = plt.figure(figsize=figsize)
        fig.gca().imshow(self.as_array() / 2 + 0.5)
        if self.title is not None:
            fig.suptitle(self.title)

        return fig

    def plot_as_subfigure(
        self,
        fig,
        subplot_shape: tuple[int, int],
        index: int,
        **kwargs,
    ):
        """Plots the mask as a matplotlib subfigure."""
        ax = fig.add_subplot(subplot_shape[1], subplot_shape[0], index + 1)
        ax.imshow(self.array / 2 + 0.5, **kwargs)
        if self.title is not None:
            ax.set_title(self.title)

    def copy(self) -> ColourImage:
        """Returns a copy of the mask."""
        return ColourImage(self.array.copy())