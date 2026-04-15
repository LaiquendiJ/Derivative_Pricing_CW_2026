import torch
import torch.nn as nn


class MultiCurrencyVAE(nn.Module):
    """
        Encoder 1:  7  → 7   Tanh
        Encoder 2:  7  → 4   None  (μ 和 logvar)
        Sampler:    4  → 2
        Decoder 1:  2  → 4   Tanh
        Decoder 2:  4  → 7   Tanh
        Decoder 3:  7  → 7   Sigmoid
    """

    def __init__(self, input_dim=7, latent_dim=2):
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


def vae_loss(x, x_recon, mu, logvar, beta=1e-7, N=7):
    """
    VAE loss

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
