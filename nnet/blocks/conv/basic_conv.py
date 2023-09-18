"""Class and function for basic convolutional blocks."""
import torch


class BasicConvBlock(torch.nn.Module):
    """Basic convolutional block with activation, dropout and batch norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        activation: type[torch.nn.Module] = torch.nn.GELU,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        use_3d: bool = False,
        padding: int | None = None,
    ):
        super().__init__()

        if use_3d:
            conv_type = torch.nn.Conv3d
            batch_norm = torch.nn.BatchNorm3d
            dropout_type = torch.nn.Dropout3d
        else:
            conv_type = torch.nn.Conv2d
            batch_norm = torch.nn.BatchNorm2d
            dropout_type = torch.nn.Dropout2d

        padding = padding or (kernel_size - stride + 1) // 2

        layers = torch.nn.ModuleList(
            [
                conv_type(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            ]
        )

        if use_batch_norm:
            layers.append(batch_norm(num_features=out_channels))

        layers.append(activation())

        if dropout > 0.0:
            layers.append(dropout_type(dropout))

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(in_tensor)


def create_basic_conv_factory(
    kernel_size: int = 3,
    *,
    use_3d: bool = False,
    stride: int = 1,
    activation: type[torch.nn.Module] = torch.nn.GELU,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
):
    """Create a basic convolutional block factory."""

    def create_block(in_dim: int, out_dim: int, **kwargs) -> torch.nn.Module:
        """Create a basic convolutional block."""
        custom_kernel_size = kwargs.get("kernel_size", kernel_size)
        custom_stride = kwargs.get("stride", stride)

        return BasicConvBlock(
            in_dim,
            out_dim,
            kernel_size=custom_kernel_size,
            stride=custom_stride,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_3d=use_3d,
        )

    return create_block


def create_strided_downsample_factory(
    *,
    use_3d: bool = False,
    activation: type[torch.nn.Module] = torch.nn.GELU,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
):
    """Create a strided convolution block factory for downsampling."""

    def create_block(
        in_dim: int, out_dim: int, *, factor: int = 2, **kwargs
    ) -> torch.nn.Module:
        """Create a strided convolution block for downsampling."""
        kernel_size = kwargs.get("kernerl_size", factor + 2)

        stride = factor

        return BasicConvBlock(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_3d=use_3d,
        )

    return create_block
