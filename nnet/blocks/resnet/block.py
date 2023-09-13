"""Base class for ResNet blocks."""
import torch


class ResnetBlock(torch.nn.Module):
    """Base class for ResNet blocks."""

    def __init__(
        self,
        layers: torch.nn.Sequential,
        residual_transform: torch.nn.Sequential | None = None,
        activation: type[torch.nn.Module] = torch.nn.GELU,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.layers = layers
        self.residual_transform = residual_transform
        self.activation = activation
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet block."""
        identity = in_tensor
        if (
            self.stochastic_depth_rate > 0.0
            and self.training
            and torch.rand(1) < self.stochastic_depth_rate
        ):
            return (
                self.residual_transform(identity)
                if self.residual_transform is not None
                else identity
            )
        tensor = self.layers(in_tensor)
        if self.residual_transform is not None:
            identity = self.residual_transform(identity)
        tensor += identity
        return self.activation()(tensor)
