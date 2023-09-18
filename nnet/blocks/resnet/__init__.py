"""Module for ResNet block implementation."""
from nnet.blocks.resnet.resnet_block import ResnetBlock
from nnet.blocks.resnet.simple import (
    create_resnet_block_simple,
    create_resnet_2d_factory,
)
from nnet.blocks.resnet.bottleneck import create_resnet_bottleneck
from nnet.blocks.resnet.simple3d import (
    create_resnet_block_simple_3d,
    create_resnet_3d_factory,
)
