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

        if numpy_data.ndim not in (2, 3):
            raise ValueError("Mask must be 2D (greyscale) or 3D (Grey + alpha)")
        if numpy_data.ndim == 3 and numpy_data.shape[2] != 2:
            raise ValueError("3D mask must be grey + alpha (2 colour channels)")

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

        array = np.zeros(
            (self.array.shape[0], self.array.shape[1], 4), dtype=np.float32
        )

        if self.array.ndim == 2:
            array[:, :, :3] = self.array[:, :, np.newaxis]
            array[:, :, 3] = 1
        else:
            array[:, :, :3] = self.array[:, :, 0, np.newaxis]
            array[:, :, 3] = self.array[:, :, 1]

        fig.gca().imshow(array / 2 + 0.5)
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

        array = np.zeros(
            (self.array.shape[0], self.array.shape[1], 4), dtype=np.float32
        )

        if self.array.ndim == 2:
            array[:, :, :3] = self.array[:, :, np.newaxis]
            array[:, :, 3] = 1
        else:
            array[:, :, :3] = self.array[:, :, 0, np.newaxis]
            array[:, :, 3] = self.array[:, :, 1]

        axes.imshow(array / 2 + 0.5, **kwargs)
        if self.title is not None:
            axes.set_title(self.title)

    def copy(self) -> Mask:
        """Returns a copy of the mask."""
        return Mask(self.array.copy())
