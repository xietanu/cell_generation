"""Simple voxel class with a single channel."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import torch

import protocols
import voxgrid
import metaimage

BLUE = (0.03, 0.57, 0.82)


class VoxGrid(protocols.Voxel):
    """Simple voxel class with a single channel."""

    title: str | None

    def __init__(
        self, data: np.ndarray | torch.Tensor, *, title: str | None = None
    ) -> None:
        if data.ndim == 5:
            raise ValueError(
                "VoxGrid is for storing single voxel grids, "
                f"this appears to be a batch of voxel grids ({data.shape})"
            )

        normalised = voxgrid.transform.normalise(data)

        if isinstance(normalised, torch.Tensor):
            normalised = voxgrid.convert.torch_to_numpy(normalised).copy()

        voxgrid.is_valid_numpy_voxel_grid(normalised, raise_error_if_not=True)

        if normalised.ndim != 4:
            normalised = normalised[..., np.newaxis]

        self.array = normalised
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
        return voxgrid.convert.numpy_to_torch(self.array)

    def create_image(self) -> metaimage.Mask:
        """Creates an image from the voxel model."""
        return metaimage.Mask(voxgrid.convert.create_image(self.array))  # type: ignore

    def rotated(
        self, x_angle: float = 0, y_angle: float = 0, z_angle: float = 0
    ) -> VoxGrid:
        """Returns a rotated version of the voxel model."""
        return VoxGrid(
            voxgrid.transform.rotated(self.array, np.array([x_angle, y_angle, z_angle]))
        )

    def plot(
        self,
        figsize: tuple[float, float] | None = None,
        *,
        angle_adjust: tuple[float, float] = (35, 25),
    ) -> matplotlib.figure.Figure:
        """Plots the voxel model as a matplotlib figure."""
        fig = plt.figure(figsize=figsize)
        axes = plt.axes(projection="3d")

        self._plot_on_axes(axes, angle_adjust=angle_adjust)

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
        axes = fig.add_subplot(
            subplot_shape[1], subplot_shape[0], index + 1, projection="3d"
        )

        self._plot_on_axes(axes, angle_adjust=angle_adjust)

        if self.title is not None:
            axes.set_title(self.title)

    def _plot_on_axes(self, axes, angle_adjust: tuple[float, float] = (35, 25)):
        array = self.as_array()

        colours = np.zeros(array.shape[:3] + (4,), dtype=np.float32)
        colours[..., 3] = array[:, :, :, -1]
        if array.shape[-1] == 1:
            colours[..., :3] = BLUE
        else:
            colours[..., :3] = array[:, :, :, :-1]

        axes.voxels(  # type: ignore
            array[:, :, :, -1] >= 0.25,
            edgecolor="none",
            facecolors=colours[:, :, :],
            shade=True,
        )
        axes.view_init(*angle_adjust)  # type: ignore

    def copy(self) -> VoxGrid:
        return VoxGrid(self.array)
