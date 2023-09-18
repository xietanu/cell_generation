import torch


class TransposeConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | None = None,
        stride: int = 2,
        activation: type[torch.nn.Module] = torch.nn.GELU,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        use_3d: bool = False,
        padding: int | None = None,
    ):
        super().__init__()

        if kernel_size is None:
            kernel_size = stride + 2

        if use_3d:
            conv_type = torch.nn.ConvTranspose3d
            batch_norm = torch.nn.BatchNorm3d
            dropout_type = torch.nn.Dropout3d
        else:
            conv_type = torch.nn.ConvTranspose2d
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def create_transpose_conv_factory(
    *,
    use_3d: bool = False,
    activation: type[torch.nn.Module] = torch.nn.GELU,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
):
    def create_block(
        in_dim: int, out_dim: int, *, factor: int = 2, **kwargs
    ) -> torch.nn.Module:
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        else:
            kernel_size = None

        stride = factor

        return TransposeConvBlock(
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
