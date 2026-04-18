"""Training loops and RMSE evaluation utilities for VAE/CVAE models."""

import torch
import torch.optim as optim
import numpy as np
from src.model import vae_loss
S_MIN, S_MAX = -0.02, 0.08


def train_vae(model, train_loader, test_loader,
              n_epochs=500, lr=1e-3, beta=1e-7,
              device='cpu', verbose=True):
    """

    Train the unconditional VAE and track train/test losses per epoch.

    Args:
        model:        MultiCurrencyVAE
        train_loader:  DataLoader
        test_loader:   DataLoader
        n_epochs
        lr:           learning rate
        beta:         KLD weight(1e-7)
        device:       'cpu' / 'cuda'
        verbose

    Returns:
        history: epoch-wise loss curves for total/reconstruction/KLD.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_total': [], 'train_recon': [], 'train_kld': [],
        'test_total':  [], 'test_recon':  [], 'test_kld':  []
    }

    for epoch in range(n_epochs):

        # ── train ──────────────────────────────
        model.train()
        train_losses = {'total': [], 'recon': [], 'kld': []}

        for x_batch, _ in train_loader:
            # Label is unused in vanilla VAE training.
            x_batch = x_batch.to(device)

            optimizer.zero_grad()

            x_recon, mu, logvar = model(x_batch)

            total, recon, kld = vae_loss(
                x_batch, x_recon, mu, logvar, beta=beta
            )

            total.backward()
            optimizer.step()

            train_losses['total'].append(total.item())
            train_losses['recon'].append(recon.item())
            train_losses['kld'].append(kld.item())

        # ── test ──────────────────────────────
        model.eval()
        test_losses = {'total': [], 'recon': [], 'kld': []}

        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                x_recon, mu, logvar = model(x_batch)
                total, recon, kld = vae_loss(
                    x_batch, x_recon, mu, logvar, beta=beta
                )
                test_losses['total'].append(total.item())
                test_losses['recon'].append(recon.item())
                test_losses['kld'].append(kld.item())

        for key in ['total', 'recon', 'kld']:
            history[f'train_{key}'].append(
                np.mean(train_losses[key])
            )
            history[f'test_{key}'].append(
                np.mean(test_losses[key])
            )

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch+1:4d}/{n_epochs}] "
                f"Train Loss: {history['train_total'][-1]:.6f} "
                f"(Recon: {history['train_recon'][-1]:.6f}, "
                f"KLD: {history['train_kld'][-1]:.6f}) | "
                f"Test Loss: {history['test_total'][-1]:.6f}"
            )

    return history


def train_cvae(model, train_loader, test_loader,
               n_epochs=500, lr=1e-3, beta=1e-7,
               device='cpu', verbose=True):
    """Train the conditional VAE with currency labels as conditioning input."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {
        'train_total': [], 'train_recon': [], 'train_kld': [],
        'test_total':  [], 'test_recon':  [], 'test_kld':  []
    }

    for epoch in range(n_epochs):

        # ── train ──────────────────────────────────
        model.train()
        train_losses = {'total': [], 'recon': [], 'kld': []}

        for x_batch, labels in train_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # CVAE forward
            x_recon, mu, logvar = model(x_batch, labels)
            total, recon, kld = vae_loss(
                x_batch, x_recon, mu, logvar, beta=beta
            )
            total.backward()
            optimizer.step()

            train_losses['total'].append(total.item())
            train_losses['recon'].append(recon.item())
            train_losses['kld'].append(kld.item())

        # ── test ──────────────────────────────────
        model.eval()
        test_losses = {'total': [], 'recon': [], 'kld': []}

        with torch.no_grad():
            for x_batch, labels in test_loader:
                x_batch = x_batch.to(device)
                labels = labels.to(device)

                x_recon, mu, logvar = model(x_batch, labels)
                total, recon, kld = vae_loss(
                    x_batch, x_recon, mu, logvar, beta=beta
                )
                test_losses['total'].append(total.item())
                test_losses['recon'].append(recon.item())
                test_losses['kld'].append(kld.item())

        for key in ['total', 'recon', 'kld']:
            history[f'train_{key}'].append(
                np.mean(train_losses[key])
            )
            history[f'test_{key}'].append(
                np.mean(test_losses[key])
            )

        if verbose and (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1:4d}/{n_epochs}] "
                f"Train: {history['train_total'][-1]:.6f} "
                f"(R:{history['train_recon'][-1]:.4f} "
                f"K:{history['train_kld'][-1]:.6f}) | "
                f"Test: {history['test_total'][-1]:.6f}"
            )

    return history


def compute_rmse_vae(model, dataset, device='cpu'):
    """
    Compute per-sample RMSE in basis points for VAE reconstructions.

    Args:
        model
        dataset: SwapRateDataset

    Returns:
        rmse_bp: array of sample-wise RMSE values in basis points.
    """
    model.eval()
    all_rmse = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)  # (1, 7)

            x_recon, _, _ = model(x)

            # [0,1]
            x_orig = x.cpu().numpy()[0] * (S_MAX - S_MIN) + S_MIN
            x_rec = x_recon.cpu().numpy()[0] * (S_MAX - S_MIN) + S_MIN

            # Convert to basis points where 1 bp = 0.01%.
            rmse = np.sqrt(np.mean((x_rec - x_orig) ** 2))
            rmse_bp = rmse * 10000  # bp

            all_rmse.append(rmse_bp)

    return np.array(all_rmse)


def compute_rmse_cvae(model, dataset, device='cpu'):
    """Compute per-sample RMSE in basis points for CVAE reconstructions."""

    model.eval()
    all_rmse = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, label = dataset[i]
            x = x.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)

            # CVAE
            mu, _ = model.encode(x, label)
            x_recon = model.decode(mu, label)

            x_orig = x.cpu().numpy()[0] * (S_MAX - S_MIN) + S_MIN
            x_rec = x_recon.cpu().numpy()[0] * (S_MAX - S_MIN) + S_MIN

            rmse = np.sqrt(np.mean((x_rec - x_orig) ** 2))
            rmse_bp = rmse * 10000
            all_rmse.append(rmse_bp)

    return np.array(all_rmse)
