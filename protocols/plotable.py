from __future__ import annotations
from typing import Protocol

import matplotlib.figure


class Plotable(Protocol):
    title: str | None

    def plot(
        self, figsize: tuple[float, float] | None = None
    ) -> matplotlib.figure.Figure:  # type: ignore
        """Plots the class as a matplotlib figure."""

    def plot_as_subfigure(
        self,
        fig,
        subplot_shape: tuple[int, int],
        index: int,
        **kwargs,
    ):
        """Plots the class as a matplotlib subfigure."""

    def copy(self) -> Plotable:  # type: ignore
        """Returns a copy of the class."""
