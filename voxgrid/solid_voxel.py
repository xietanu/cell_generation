"""Simple voxel class with a single channel."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import torch

import protocols
import voxgrid
import metaimage

RED = (1, 0, 0)


class SolidVoxel(protocols.Voxel):
    """Simple voxel class with a single channel."""

    title: str | None

    def __init__(self, data: np.ndarray, *, title: str | None = None) -> None:
        if data.ndim != 3:
            raise ValueError(f"data must be 3-dimensional ({data.shape})")
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
        return torch.from_numpy(self.array.copy()[np.newaxis, ...])

    @classmethod
    def from_array(cls, array, *, title: str | None = None) -> SolidVoxel:
        """Creates a SimpleVoxel from a numpy array."""
        return cls(array, title=title)

    @classmethod
    def from_tensor(cls, tensor, *, title: str | None = None) -> SolidVoxel:
        """Creates a SimpleVoxel from a PyTorch tensor."""
        return cls(tensor.detach().squeeze().cpu().numpy(), title=title)

    def create_image(
        self, x_angle: float = 0.0, y_angle: float = 0.0, z_angle: float = 0.0
    ) -> metaimage.Mask:
        """Creates a mask from the voxel model from a given viewpoint."""
        if x_angle != 0.0 or y_angle != 0.0 or z_angle != 0.0:
            rotated = self.rotated(x_angle, y_angle, z_angle)
        else:
            rotated = self

        mask = np.max(rotated.array, axis=2)

        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        return metaimage.Mask.from_array(mask)

    def rotated(self, x_angle: float, y_angle: float, z_angle: float) -> SolidVoxel:
        """Returns a rotated version of the voxel model."""
        rotate_matrix = voxgrid.transform.create_rotation_matrix(
            x_angle, y_angle, z_angle
        )
        return SolidVoxel(
            voxgrid.transform.apply_transform(self.array, rotate_matrix, centred=True)
        )

    def plot(
        self,
        figsize: tuple[float, float] | None = None,
        *,
        angle_adjust: tuple[float, float] = (35, 25),
    ) -> matplotlib.figure.Figure:
        """Plots the voxel model as a matplotlib figure."""
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection="3d")

        array = self.as_array()

        ax.voxels(array >= 0.5, edgecolor="none", facecolors=RED, shade=True)  # type: ignore
        ax.view_init(*angle_adjust)  # type: ignore

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

        array = self.as_array()

        ax.voxels(array >= 0.5, edgecolor="none", facecolors=RED, shade=True)
        ax.view_init(*angle_adjust)

        if self.title is not None:
            ax.set_title(self.title)

    def copy(self) -> SolidVoxel:
        return SolidVoxel(self.array)
