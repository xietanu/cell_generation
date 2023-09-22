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


class TranspVoxel(protocols.Voxel):
    """Simple voxel class with a single channel."""

    title: str | None

    def __init__(self, data: np.ndarray, *, title: str | None = None) -> None:
        if data.ndim != 3:
            raise ValueError(f"data must be 3-dimensional ({data.shape})")
        data = data.astype(np.float32)
        self.array = (data - data.min()) / (data.max() - data.min())
        self.alpha_adjust = np.log2(self.array.shape[0]) + 2
        self.array = self.array**self.alpha_adjust
        self.array = np.clip(self.array, 0, 1)

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
    def from_array(cls, array, *, title: str | None = None) -> TranspVoxel:
        """Creates a SimpleVoxel from a numpy array."""
        return cls(array, title=title)

    @classmethod
    def from_tensor(cls, tensor, *, title: str | None = None) -> TranspVoxel:
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

        image = np.zeros((self.shape[0], self.shape[1]), dtype=np.float32)

        for level in reversed(range(self.shape[2])):
            alpha = rotated.array[:, :, level]
            image = image * (1 - alpha) + alpha

        return metaimage.Mask.from_array(image)

    def rotated(self, x_angle: float, y_angle: float, z_angle: float) -> TranspVoxel:
        """Returns a rotated version of the voxel model."""
        rotate_matrix = voxgrid.transform.create_rotation_matrix(
            x_angle, y_angle, z_angle
        )[np.newaxis, :3, :4]

        affine_grid = torch.nn.functional.affine_grid(
            torch.from_numpy(rotate_matrix),
            [1, 1, *self.shape],
            align_corners=False,
        ).float()

        view = torch.nn.functional.grid_sample(
            self.as_tensor().float()[np.newaxis, ...],
            affine_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return TranspVoxel.from_tensor(view[0, 0, ...])

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

        colours = np.zeros(array.shape + (4,), dtype=np.float32)
        colours[..., 3] = array
        colours[..., :3] = RED

        ax.voxels(  # type: ignore
            array > 0.01, edgecolor="none", facecolors=colours[:, :, :], shade=True
        )
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

        colours = np.zeros(array.shape + (4,), dtype=np.float32)
        colours[..., 3] = array
        colours[..., :3] = RED

        ax.voxels(
            array > 0.25, edgecolor="none", facecolors=colours[:, :, :], shade=True
        )
        ax.view_init(*angle_adjust)

        if self.title is not None:
            ax.set_title(self.title)

    def copy(self) -> TranspVoxel:
        return TranspVoxel(self.array)
