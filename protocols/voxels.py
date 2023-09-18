from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import matplotlib.figure

import protocols


class Voxel(Protocol):
    title: str | None

    @property
    def shape(self):
        """Returns the shape of the voxel model."""

    def as_array(self) -> np.ndarray:  # type: ignore
        """Returns the voxel model as a numpy array."""

    def as_tensor(self) -> torch.Tensor:  # type: ignore
        """Returns the voxel model as a PyTorch tensor."""

    @classmethod
    def from_array(cls, array, *, title: str | None = None) -> Voxel:  # type: ignore
        """Creates a Voxel model from a numpy array."""

    @classmethod
    def from_tensor(cls, tensor, *, title: str | None = None) -> Voxel:  # type: ignore
        """Creates a Voxel model from a PyTorch tensor."""

    def create_mask(self, x_angle: float, y_angle: float, z_angle: float) -> protocols.MetaImage:  # type: ignore
        """Creates a mask from the voxel model from a given viewpoint."""

    def rotated(self, x_angle: float, y_angle: float, z_angle: float) -> Voxel:  # type: ignore
        """Returns a rotated version of the voxel model."""

    def plot(
        self, figsize: tuple[float, float] | None = None
    ) -> matplotlib.figure.Figure:  # type: ignore
        """Plots the voxel model as a matplotlib figure."""

    def plot_as_subfigure(
        self,
        fig,
        subplot_shape: tuple[int, int],
        index: int,
        **kwargs,
    ):
        """Plots the voxel model as a matplotlib subfigure."""

    def copy(self) -> Voxel:  # type: ignore
        """Returns a copy of the class."""
