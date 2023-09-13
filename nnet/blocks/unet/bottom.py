"""Implements the bottom level of a U-Net.

As outlined in this paper: https://arxiv.org/pdf/1505.04597v1.pdf
"""
import torch.nn
import nnet.blocks.resnet


class Bottom(torch.nn.Module):
    """Implements the bottom lever of a U-Net."""

    def __init__(self, in_channels: int, kernel_size=3, stochastic_depth_rate=0.0):
        """Initializes the bottom level."""
        super().__init__()
        self.layers = nnet.blocks.resnet.create_resnet_block_simple(
            in_channels,
            in_channels * 2,
            kernel_size=kernel_size,
            stochastic_depth_rate=stochastic_depth_rate,
        )

    def forward(self, x):
        """Forward pass of the bottom level."""
        out = self.layers(x)
        return out
