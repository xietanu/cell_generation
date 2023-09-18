"""Simple voxel class with a single channel."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import torch

import protocols
import voxel
import metaimage


class ColourVoxel(protocols.Voxel):
    """Colour voxel class with a single channel."""

    title: str | None

    def __init__(self, data: np.ndarray, *, title: str | None = None) -> None:
        if data.ndim != 4:
            raise ValueError(f"data must be 4-dimensional ({data.shape})")
        if data.shape[3] != 4:
            raise ValueError(f"data must have 4 channels ({data.shape})")

        data = data.astype(np.float32)
        self.array = (data - data.min()) / (data.max() - data.min())
        self.title = title

    @property
    def shape(self):
        """Returns the shape of the voxel model."""
        return self.array.shape

    def as_array(self) -> np.ndarray:
        """Returns the voxel model as a numpy array."""
        return self.array.copy()

    def as_tensor(self) -> torch.Tensor:
        """Returns the voxel model as a PyTorch tensor."""
        return torch.from_numpy(self.array.copy()).permute(3, 0, 1, 2)

    @classmethod
    def from_array(cls, array, *, title: str | None = None) -> ColourVoxel:
        """Creates a SimpleVoxel from a numpy array."""
        return cls(array, title=title)

    @classmethod
    def from_tensor(cls, tensor, *, title: str | None = None) -> ColourVoxel:
        """Creates a SimpleVoxel from a PyTorch tensor."""
        return cls(
            tensor.detach().squeeze().cpu().permute(1, 2, 3, 0).numpy(), title=title
        )

    def create_mask(
        self, x_angle: float = 0.0, y_angle: float = 0.0, z_angle: float = 0.0
    ) -> metaimage.Mask:
        """Creates a mask from the voxel model from a given viewpoint."""
        simple = voxel.TranspVoxel(self.array[:, :, :, 3])

        return simple.create_mask(x_angle, y_angle, z_angle)

    def create_image(
        self, x_angle: float = 0.0, y_angle: float = 0.0, z_angle: float = 0.0
    ) -> metaimage.ColourImage:
        """Creates a mask from the voxel model from a given viewpoint."""
        if x_angle != 0.0 or y_angle != 0.0 or z_angle != 0.0:
            rotated = self.rotated(x_angle, y_angle, z_angle)
        else:
            rotated = self

        image = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.float32)

        for level in reversed(range(self.shape[2])):
            alpha = rotated.array[:, :, level, 3, np.newaxis]
            image = image * (1 - alpha) + rotated.array[:, :, level, :3] * alpha

        return metaimage.ColourImage.from_array(image)

    def rotated(self, x_angle: float, y_angle: float, z_angle: float) -> ColourVoxel:
        """Returns a rotated version of the voxel model."""

        array = self.array.copy()

        for channel in range(self.array.shape[3]):
            array[:, :, :, channel] = (
                voxel.TranspVoxel(array[:, :, :, channel])
                .rotated(x_angle, y_angle, z_angle)
                .as_array()
            )

        return ColourVoxel(array, title=self.title)

    def plot(
        self, figsize: tuple[float, float] | None = None
    ) -> matplotlib.figure.Figure:
        """Plots the voxel model as a matplotlib figure."""
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection="3d")
        ax.voxels(
            self.array[:, :, :, 3] >= 0.5,
            edgecolor="none",
            facecolors=self.array[:, :, :],
            shade=True,
        )
        if self.title is not None:
            fig.suptitle(self.title)

        return fig

    def plot_as_subfigure(
        self,
        fig,
        subplot_shape: tuple[int, int],
        index: int,
        *,
        angle_adjust: tuple[float, float] = (35, 25),
        **_,
    ):
        """Plots the voxel model as a subfigure with a given angle."""
        ax = fig.add_subplot(
            subplot_shape[1], subplot_shape[0], index + 1, projection="3d"
        )
        ax.voxels(
            self.array[:, :, :, 3] >= 0.5,
            edgecolor="none",
            facecolors=self.array[:, :, :],
            shade=True,
        )
        ax.view_init(*angle_adjust)
        if self.title is not None:
            ax.set_title(self.title)

    def copy(self) -> ColourVoxel:
        return ColourVoxel(self.array)
