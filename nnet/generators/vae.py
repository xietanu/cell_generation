import torch


class VAE(torch.nn.Module):
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
        self, x, *, encoder_kwargs=None, decoder_kwargs=None, encode_only=False
    ):
        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        encoded = self.encoder(x, **encoder_kwargs)

        if isinstance(encoded, tuple):
            decoder_kwargs["additional"] = encoded[1:]
            encoded = encoded[0]

        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        sampled = q.rsample()

        if encode_only:
            return sampled, mu, log_var

        decoded = self.decoder(
            sampled,
            **decoder_kwargs,
        )

        return decoded, mu, log_var


def vae_loss_function(output, target):
    value, mu, log_var = output
    if isinstance(value, tuple):
        value = value[0]
    repo_loss = torch.nn.L1Loss()(value, target[0])
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return repo_loss + kld
