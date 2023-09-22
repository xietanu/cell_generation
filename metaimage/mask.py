"""Mask image, with range of -1 to 1."""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.figure

import protocols


class Mask(protocols.MetaImage):
    """Mask image, with range of -1 to 1."""

    title: str | None

    def __init__(self, data: np.ndarray | torch.Tensor, *, title: str | None = None):
        if isinstance(data, torch.Tensor):
            numpy_data = data.detach().cpu().squeeze().numpy()
        else:
            numpy_data = data.copy()

        if numpy_data.ndim != 2:
            raise ValueError("Mask must be 2D")

        if numpy_data.max() - numpy_data.min() == 0:
            numpy_data[:, :] = -1
        else:
            numpy_data = (
                2
                * (numpy_data.astype(np.float32) - numpy_data.min())
                / (numpy_data.max() - numpy_data.min())
                - 1
            )
        self.array = numpy_data.astype(np.float32)
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
        return torch.from_numpy(self.array[np.newaxis, :, :])

    def plot(
        self, figsize: tuple[float, float] | None = None
    ) -> matplotlib.figure.Figure:
        """Plots the mask as a matplotlib figure."""
        fig = plt.figure(figsize=figsize)
        fig.gca().imshow(self.as_array(), cmap="gray", vmin=-1, vmax=1)
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
        axes = fig.add_subplot(subplot_shape[1], subplot_shape[0], index + 1)
        axes.imshow(self.array, cmap="gray", vmin=-1, vmax=1, **kwargs)
        if self.title is not None:
            axes.set_title(self.title)

    def copy(self) -> Mask:
        """Returns a copy of the mask."""
        return Mask(self.array.copy())
