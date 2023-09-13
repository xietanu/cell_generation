"""Implements a U-Net.

As outlined in this paper: https://arxiv.org/pdf/1505.04597v1.pdf
"""
import nnet.blocks
import torch.nn


class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        kernel_size=3,
        first_layer: bool = True,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        if first_layer:
            base_channels = in_channels
        else:
            base_channels = in_channels * 2

        self.in_layers = nnet.blocks.resnet.create_resnet_block_simple(
            in_channels,
            base_channels,
            kernel_size=kernel_size,
            stochastic_depth_rate=stochastic_depth_rate,
        )
        self.down = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        if depth > 1:
            self.inner = UNet(
                depth - 1,
                base_channels,
                first_layer=False,
                kernel_size=kernel_size,
                stochastic_depth_rate=stochastic_depth_rate,
            )
        else:
            self.inner = nnet.blocks.unet.Bottom(
                base_channels,
                kernel_size=kernel_size,
                stochastic_depth_rate=stochastic_depth_rate,
            )

        self.up = torch.nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )

        self.out_layers = nnet.blocks.resnet.create_resnet_block_simple(
            base_channels * 2,
            base_channels,
            kernel_size=kernel_size,
            stochastic_depth_rate=stochastic_depth_rate,
        )

    def forward(self, x):
        """Forward pass of the U-Net."""
        skip = self.in_layers(x)
        inner = self.up(self.inner(self.down(skip)))
        out = self.out_layers(
            torch.cat([inner, skip], dim=1)  # pylint: disable=no-member
        )
        return out
