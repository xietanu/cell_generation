"""Plotting functions for displaying multiple images in a grid."""
from typing import Iterable

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import torch

import protocols
import voxel
import metaimage

TitledItem = tuple[str, protocols.Plotable | np.ndarray | torch.Tensor | None]


def grid(
    items: Iterable[protocols.Plotable | np.ndarray | TitledItem | torch.Tensor | None]
    | Iterable[
        Iterable[protocols.Plotable | np.ndarray | TitledItem | torch.Tensor | None]
    ],
    figsize: tuple[float, float] = (12, 12),
) -> matplotlib.figure.Figure:
    """Plots the items in a grid."""
    item_grid: list[list[protocols.Plotable | None]] = [
        (
            [convert_to_plotable(subitem) for subitem in item]  # type: ignore
            if is_non_numpy_iterable(item)
            else [convert_to_plotable(item)]  # type: ignore
        )
        for item in items
    ]

    row_length = max(len(row) for row in item_grid)
    column_length = len(item_grid)

    figure = plt.figure(figsize=figsize)
    for i, row in enumerate(item_grid):
        for j, item in enumerate(row):
            if item is not None:
                item.plot_as_subfigure(
                    figure,
                    (row_length, column_length),
                    i * row_length + j,
                )

    return figure


def is_non_numpy_iterable(obj):
    """Returns whether an object is an iterable that is not a numpy array."""
    return isinstance(obj, Iterable) and not isinstance(obj, np.ndarray)


def convert_to_plotable(
    obj: np.ndarray | torch.Tensor | protocols.Plotable | None,
) -> protocols.Plotable | None:
    """Converts an object to a plotable object."""
    if isinstance(obj, tuple):
        title, plotable = obj
        plotable = convert_to_plotable(plotable)
        if plotable is not None:
            plotable.title = title
        return plotable
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            return metaimage.Mask.from_array(obj)
        if obj.ndim == 3:
            return voxel.TranspVoxel.from_array(obj)
        raise ValueError("Plotable array must be 2D or 3D")
    if isinstance(obj, torch.Tensor):
        if obj.squeeze().ndim == 2:
            return metaimage.Mask.from_tensor(obj)
        if obj.squeeze().ndim == 3:
            return voxel.TranspVoxel.from_tensor(obj)
        raise ValueError("Plotable tensor must be 2D or 3D")
    return obj
