"""Neural network model definitions and objective for VAE/CVAE training."""

import torch
import torch.nn as nn


class MultiCurrencyVAE(nn.Module):
    """
        Encoder 1:  7  → 7   Tanh
        Encoder 2:  7  → 4   None  (μ and logvar)
        Sampler:    4  → 2
        Decoder 1:  2  → 4   Tanh
        Decoder 2:  4  → 7   Tanh
        Decoder 3:  7  → 7   Sigmoid
    """

    def __init__(self, input_dim=7, latent_dim=2):
        """Initialize unconditional VAE encoder/decoder architecture."""
        super(MultiCurrencyVAE, self).__init__()

        self.input_dim = input_dim   # N = 7
        self.latent_dim = latent_dim  # K = 2

        # ── Encoder ───────────────────────────────
        self.encoder = nn.Sequential(
            # Encoder 1: 7 → 7, Tanh
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            # Encoder 2: 7 → 4, None
            nn.Linear(input_dim, latent_dim * 2),
            # [μ₁, μ₂, logvar₁, logvar₂]
        )

        # ── Decoder ───────────────────────────────
        self.decoder = nn.Sequential(
            # Decoder 1: 2 → 4, Tanh
            nn.Linear(latent_dim, 4),
            nn.Tanh(),
            # Decoder 2: 4 → 7, Tanh
            nn.Linear(4, input_dim),
            nn.Tanh(),
            # Decoder 3: 7 → 7, Sigmoid
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """
        inputs: x (batch, 7)
        outputs: mu (batch, 2), logvar (batch, 2)
        """
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        z = μ + σ·ω, ω ~ N(0,1)
        """
        if self.training:
            # σ = exp(logvar/2)
            std = torch.exp(0.5 * logvar)
            # ω ~ N(0,1)
            omega = torch.randn_like(std)
            # z = μ + σ·ω
            return mu + std * omega
        else:
            return mu

    def decode(self, z):
        """
        input: z (batch, 2)
        output: S' (batch, 7)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        input: x  (batch, 7)
        output:
            x_recon (batch, 7)
            mu      (batch, 2)
            logvar  (batch, 2)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_latent(self, x):
        """
        z = μ
        input: x (batch, 7)
        output: z (batch, 2)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


class MultiCurrencyCVAE(nn.Module):
    """
    Table 3: Multi-Currency CVAE
      Encoder: [swap_rates(7), one_hot_currency(C)] = 7+C
      Decoder: [latent_z(2),   one_hot_currency(C)] = 2+C

      Encoder 1: (7+C) → 9,   Tanh
      Encoder 2:  9    → 4,   None  (output μ, logvar)
      Sampler:    4    → 2
      Decoder 1: (2+C) → 7,   Tanh
      Decoder 2:  7    → 7,   Tanh
      Decoder 3:  7    → 7,   Sigmoid
    """

    def __init__(self, input_dim=7, latent_dim=2,
                 n_currencies=3):
        """Initialize conditional VAE using one-hot currency conditioning."""
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_currencies = n_currencies

        # ── Encoder ───────────────────────────────
        # input: rates + one-hot currency
        enc_input_dim = input_dim + n_currencies  # 7 + 3 = 10

        self.encoder = nn.Sequential(
            # Encoder 1: (7+C) → 9, Tanh
            nn.Linear(enc_input_dim, 9),
            nn.Tanh(),
            # Encoder 2: 9 → 4, None
            nn.Linear(9, latent_dim * 2),
        )

        # ── Decoder ───────────────────────────────
        # input: latent z + one-hot currency
        dec_input_dim = latent_dim + n_currencies  # 2 + 3 = 5

        self.decoder = nn.Sequential(
            # Decoder 1: (2+C) → 7, Tanh
            nn.Linear(dec_input_dim, input_dim),
            nn.Tanh(),
            # Decoder 2: 7 → 7, Tanh
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            # Decoder 3: 7 → 7, Sigmoid
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def _one_hot(self, labels):
        """
        labels: (batch,) LongTensor
        return: (batch, C) FloatTensor,one-hot
        """
        one_hot = torch.zeros(
            labels.size(0), self.n_currencies,
            device=labels.device
        )
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return one_hot

    def encode(self, x, labels):
        """
        x:      (batch, 7)
        labels: (batch,)
        return:   mu (batch, 2), logvar (batch, 2)
        """
        one_hot = self._one_hot(labels)
        x_cond = torch.cat([x, one_hot], dim=1)  # (batch, 7+C)
        h = self.encoder(x_cond)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            omega = torch.randn_like(std)
            return mu + std * omega
        return mu

    def decode(self, z, labels):
        """
        z:      (batch, 2)
        labels: (batch,)
        return:   x_recon (batch, 7)
        """
        one_hot = self._one_hot(labels)
        z_cond = torch.cat([z, one_hot], dim=1)  # (batch, 2+C)
        return self.decoder(z_cond)

    def forward(self, x, labels):
        """
        x:      (batch, 7)
        labels: (batch,)
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

    def get_latent(self, x, labels):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x, labels)
        return mu


def vae_loss(x, x_recon, mu, logvar, beta=1e-7, N=7):
    """
    Compute reconstruction + KL divergence objective.

    D_VAE = (1/N) * D_L2 + β * D_KLD

    Args:
        x:           (batch, 7)
        x_recon:     (batch, 7)
        mu:          (batch, 2)
        logvar:      (batch, 2)
        beta:    KLD     ( 1e-7)
        N:       dim x (= 7)

    Returns:
        total_loss
        recon_loss
        kld_loss
    """

    # ──  D_L2 ─────────────────────────────
    #  D_L2 = Σ(S'_n - S_n)²
    # avg batch
    recon_loss = torch.sum((x_recon - x) ** 2, dim=1).mean()

    # ── KLD loss D_KLD ────────────────────────────
    #  D_KLD = (1/2) Σ(σ²_k + μ²_k - 1 - ln(σ²_k))
    # logvar = ln(σ²) σ² = exp(logvar)
    kld_loss = 0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1 - logvar,
        dim=1
    ).mean()

    # ── total loss ────────────────────────────────────
    #  D_VAE = (1/N) * D_L2 + β * D_KLD
    total_loss = (1.0 / N) * recon_loss + beta * kld_loss

    return total_loss, recon_loss, kld_loss
