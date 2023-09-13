"""Bottleneck resnet block."""
from __future__ import annotations
import torch.nn

import nnet.blocks.resnet


def create_resnet_bottleneck(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    activation: type[torch.nn.Module] = torch.nn.GELU,
    stochastic_depth_rate: float = 0.0,
) -> nnet.blocks.resnet.ResnetBlock:
    """Initialize the bottleneck resnet block."""

    layers = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            padding=0,
        ),
        torch.nn.BatchNorm2d(num_features=out_channels // 4),
        activation(),
        torch.nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels // 4,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        ),
        torch.nn.BatchNorm2d(num_features=out_channels // 4),
        activation(),
        torch.nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        ),
        torch.nn.BatchNorm2d(num_features=out_channels),
    )

    residual_transform = None

    if in_channels != out_channels:
        residual_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
        )

    return nnet.blocks.resnet.ResnetBlock(
        layers=layers,
        residual_transform=residual_transform,
        activation=activation,
        stochastic_depth_rate=stochastic_depth_rate,
    )
