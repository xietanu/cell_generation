"""Torch module for an upsampling block."""
import torch


class UpsampleBlock(torch.nn.Module):
    """Torch module for an upsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        scale_factor: int = 2,
        activation: type[torch.nn.Module] = torch.nn.GELU,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        use_3d: bool = False,
        mode: str = "nearest",
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

        padding = padding or kernel_size // 2

        layers = torch.nn.ModuleList(
            [
                torch.nn.Upsample(scale_factor=scale_factor, mode=mode),
                conv_type(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
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


def create_upsample_block_factory(
    kernel_size: int = 3,
    *,
    use_3d: bool = False,
    activation: type[torch.nn.Module] = torch.nn.GELU,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
):
    """Create a factory for upsampling blocks."""

    def create_block(
        in_dim: int, out_dim: int, *, factor: int = 2, **kwargs
    ) -> torch.nn.Module:
        """Create an upsampling block."""
        custom_kernel_size = kwargs.get("kernel_size", kernel_size)

        return UpsampleBlock(
            in_dim,
            out_dim,
            scale_factor=factor,
            kernel_size=custom_kernel_size,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_3d=use_3d,
        )

    return create_block
