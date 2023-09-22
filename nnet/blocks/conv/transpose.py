"""Torch module for a transpose convolutional block."""
import torch


class TransposeConvBlock(torch.nn.Module):
    """Torch module for a transpose convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | None = None,
        stride: int = 2,
        activation: type[torch.nn.Module] = torch.nn.GELU,
        use_batch_norm: bool = True,
        dropout: float | torch.nn.Module = 0.0,
        use_3d: bool = False,
        padding: int | None = None,
    ):
        super().__init__()

        if kernel_size is None:
            kernel_size = stride + 2

        if use_3d:
            conv_type = torch.nn.ConvTranspose3d
            batch_norm = torch.nn.BatchNorm3d
            dropout_type = (
                torch.nn.Dropout3d(dropout) if isinstance(dropout, float) else dropout
            )
        else:
            conv_type = torch.nn.ConvTranspose2d
            batch_norm = torch.nn.BatchNorm2d
            dropout_type = (
                torch.nn.Dropout2d(dropout) if isinstance(dropout, float) else dropout
            )

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

        if isinstance(dropout, torch.nn.Module) or dropout > 0.0:
            layers.append(dropout_type)  # type: ignore

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(in_tensor)


def create_transpose_conv_factory(
    *,
    use_3d: bool = False,
    activation: type[torch.nn.Module] = torch.nn.GELU,
    use_batch_norm: bool = True,
    dropout: float | torch.nn.Module = 0.0,
):
    def create_block(
        in_dim: int, out_dim: int, *, factor: int = 2, **kwargs
    ) -> torch.nn.Module:
        kernel_size = kwargs.get("kernel_size", factor + 2)

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
