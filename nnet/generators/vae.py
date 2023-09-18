"""Variational Autoencoder (VAE) module."""
import torch


class VAE(torch.nn.Module):
    """Variational Autoencoder (VAE) module."""

    def __init__(
        self,
        encoder,
        decoder,
        latent_size=12,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.encoder = encoder
        self.decoder = decoder

        self.fc_mu = torch.nn.Linear(latent_size, latent_size)
        self.fc_var = torch.nn.Linear(latent_size, latent_size)

    def forward(
        self, in_tensor, *, encoder_kwargs=None, decoder_kwargs=None, encode_only=False
    ):
        """Forward pass."""
        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        encoded = self.encoder(in_tensor, **encoder_kwargs)

        if isinstance(encoded, tuple):
            decoder_kwargs["additional"] = encoded[1:]
            encoded = encoded[0]

        mean = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        distribution = torch.distributions.Normal(mean, std)
        sampled = distribution.rsample()

        if encode_only:
            return sampled, mean, log_var

        decoded = self.decoder(
            sampled,
            **decoder_kwargs,
        )

        return decoded, mean, log_var


def vae_loss_function(output, target):
    """Loss function for a VAE."""
    value, mean, log_var = output
    if isinstance(value, tuple):
        value = value[0]
    repo_loss = torch.nn.L1Loss()(value, target[0])
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return repo_loss + kld
