# nnet

nnet contains building blocks for neural network architectures. This includes new modules for specific aspects of the generationa architecture as well as blocks of layers to simplify the creation of more complex architectures.

## nnet.blocks

Contains classes and functions to create multi-layer blocks, typically to an established format such as a ResNet block.

### nnet.blocks.resnet

Allows for quick and simple creation of ResNet blocks with useful configuration.

#### nnet.blocks.resnet.ResNetBlock

A parent class for ResNet blocks, typically not used directly in code. It provides the general structure that there are a collection of layers with a bypass that adds the residual to the output layer. If a different number of output layers to input is specificed, this is handled by a 1x1 conv layer.

This is primarily intended for basic ResNet blocks, as outlined in [the original paper](https://arxiv.org/abs/1512.03385). For more complex setups, such as ResNext, a new implementation would likely be preferable.

#### nnet.blocks.resnet.create_resnet_block_simple

Creates a basic 2 layer ResNet block for 2D convolutions. Along with numbers of input/output channels, kernel size, etc., a dropout rate for channels and a [stochastic depth dropout](https://arxiv.org/abs/1603.09382) rate can be specified for use during training.

#### nnet.blocks.resnet.create_resnet_block_simple_3d

As above, for 3D convolutions. Will likely be combined into a single function with an option to reduce code duplication.

#### nnet.blocks.resnet.create_resnet_2d_factory

Creates a partially configured factory function, for creating 2D resnet blocks. This is used to help automate the creation of larger networks.

The kernel size, activation function and dropout are specified at this stage, requiring just input and output channels to be supplied to create a resnet block.

#### nnet.blocks.resnet.create_resnet_3d_factory

As above, for 3D convolutions. Will likely be combined into a single function with an option to reduce code duplication.

#### nnet.blocks.resnet.create_resnet_bottleneck

Creates a 3 layer bottleneck ResNet block for 2D convolutions. Works similarly to the 2 layer implementation.

### nnet.blocks.conv

Contains functions to create convolutional layer blocks. Primarily for ease, to bundle up convolutional layers with activation, dropout and batch normalization.

#### nnet.blocks.conv.BasicConvBlock

A basic convolutional block, including a convolutional layer, batch normalization, activation and dropout layers. Whether each is used can be configured, with standard parameters assumed.



## nnet.modules


### nnet.modules.VoxGridAlphaAdjust

Adjusts the alpha channel of a voxel grid, such that when a view is created from the top, all the voxels having an alpha of 0.5 will result in a 50% transparency in the resultant image. This is done by taking the alpha channel to a power - for a 32x32x32 grid, this is 5.44.

This is done to address a problem that inhibits learning - when initialised, the resulting image typically has visibility into only the first few layers of the voxel grid, with information behind this completely lost. This makes it very hard for the model to learn to form shapes, as changes to the colour or alpha value of the vast majority of voxels has no impact early on. Making this adjustment removes this problem, and the model converges much faster and better accordingly.

Additionally, this solution ensures there is a smooth transition throughout, and that all values from 0 to 1 remain accessible.