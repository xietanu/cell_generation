import torch
import numpy as np

import protocols


class ResizeStack(torch.nn.Module):
    """Stack of downsampling blocks."""

    def __init__(
        self,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        layer_factory: protocols.BlockFactory,
        resize_factory: protocols.ResizeBlockFactory,
        *,
        layer_factories_per_downsample: int | None = None,
        total_layers: int | None = None,
    ):
        super().__init__()

        if out_shape[1] > in_shape[1]:
            bigger = out_shape
            smaller = in_shape
            downsample = False
        else:
            bigger = in_shape
            smaller = out_shape
            downsample = True

        scale_factor = smaller[0] // bigger[0]

        if not all(
            bigger[i] // scale_factor == smaller[i] for i in range(1, len(in_shape))
        ):
            raise ValueError("Inconsistent scaling factors not supported")

        self.layer_schedule = create_layer_schedule(
            smaller[0] // bigger[0],
            total_layers=total_layers,
            layer_factories_per_downsample=layer_factories_per_downsample,
            big_to_small=out_shape[0] > in_shape[0],
        )

        self.main = torch.nn.ModuleList()

        cur_channels = in_shape[0]
        cur_w = in_shape[1]

        for factor, n_layers in self.layer_schedule:
            if downsample:
                next_channels = cur_channels * factor
                next_w = cur_w // factor
            else:
                next_channels = cur_channels // factor
                next_w = cur_w * factor
            if n_layers > 1:
                for _ in range(n_layers - 1):
                    self.main.append(
                        layer_factory(cur_channels, cur_channels),
                    )
            if n_layers > 0:
                self.main.append(
                    layer_factory(cur_channels, next_channels),
                )
                self.main.append(
                    resize_factory(next_channels, next_channels, factor=factor),
                )
            else:
                self.main.append(
                    resize_factory(cur_channels, next_channels, factor=factor),
                )
            cur_channels = next_channels

        self.main = torch.nn.Sequential(*self.main)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def create_layer_schedule(
    shrink_scale: int,
    layer_factories_per_downsample: int | None = None,
    total_layers: int | None = None,
    big_to_small: bool = False,
) -> list[tuple[int, int]]:
    factors = prime_factors(shrink_scale)

    if big_to_small:
        factors = factors[::-1]

    if layer_factories_per_downsample is None and total_layers is None:
        return [(factor, 1) for factor in factors]
    if layer_factories_per_downsample is not None and total_layers is not None:
        raise ValueError(
            "Cannot specify both layer_factories_per_downsample and total_layers"
        )
    if layer_factories_per_downsample is not None:
        return [(factor, layer_factories_per_downsample) for factor in factors]

    if total_layers is not None:
        schedule = [[factor, total_layers // len(factors)] for factor in factors]
        schedule[len(schedule) // 2][1] += total_layers % len(factors)
        schedule = [(factor, layers) for factor, layers in schedule]
        return schedule

    raise ValueError("Unreachable")


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
