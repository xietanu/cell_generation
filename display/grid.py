"""Plotting functions for displaying multiple images in a grid."""
from typing import Iterable

import matplotlib.figure
import matplotlib.pyplot as plt

import protocols


TitledItem = tuple[str, protocols.Plotable]


def grid(
    items: Iterable[protocols.Plotable | TitledItem | None]
    | Iterable[Iterable[protocols.Plotable | TitledItem | None]],
    figsize: tuple[float, float] = (12, 12),
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Plots the items in a grid."""
    item_grid: list[list[protocols.Plotable | TitledItem | None]] = [
        (
            [subitem for subitem in item]  # type: ignore
            if isinstance(item, Iterable)
            else [item]  # type: ignore
        )
        for item in items
    ]

    row_length = max(len(row) for row in item_grid)
    column_length = len(item_grid)

    figure = plt.figure(figsize=figsize)
    for i, row in enumerate(item_grid):
        for j, item in enumerate(row):
            if item is None:
                continue
            if isinstance(item, tuple):
                title = item[0]
                item = item[1].copy()
                item.title = title
            item.plot_as_subfigure(
                figure,
                (row_length, column_length),
                i * row_length + j,
            )
    if title:
        figure.suptitle(title)
        
    figure.tight_layout()

    return figure
