"""Plotting helpers for model diagnostics and latent-space interpretation."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Ellipse
from scipy.interpolate import PchipInterpolator

MATURITIES = [2, 3, 5, 10, 15, 20, 30]
S_MIN, S_MAX = -0.02, 0.08
x_smooth = np.linspace(2, 30, 500)
currency_color = {'USD': 'green', 'GBP': 'orange', 'EUR': 'blue'}


def plot_training_history(history, title):
    """
    Plot train/test curves for total, reconstruction, and KL losses.
    """
    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, loss_type in zip(
        axes,
        ['total', 'recon', 'kld'],
        ['Total Loss', 'Reconstruction Loss', 'KLD Loss']
    ):
        ax.plot(history[f'train_{key}'], label='Train', linewidth=1)
        ax.plot(history[f'test_{key}'],  label='Test',  linewidth=1)
        ax.set_title(f'{title}-{loss_type}', fontsize=15)
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig/training_history.png', dpi=150)
    plt.show()


def plot_rmse_distribution(rmse_dict, title='RMSE Distribution'):
    """
    Plot histogram-based RMSE density comparison across methods/splits.

    Args:
        rmse_dict: {'Method Name': rmse_array,...}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['green', 'orange', 'blue', 'red']

    for (name, rmse), color in zip(rmse_dict.items(), colors):
        ax.hist(rmse, bins=50, alpha=0.5,
                label=name, color=color,
                density=True, edgecolor='none')

    ax.set_xlabel('Swap Rate RMSE (bp)', fontsize=15)
    ax.set_ylabel('Probability Density', fontsize=15)
    # ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig(f'fig/{title}.png', dpi=150)
    plt.show()


def plot_rmse_by_currency(per_ccy_rmse):
    """Compare in-sample RMSE distributions across currencies."""
    for ccy, rmse in per_ccy_rmse.items():
        plt.hist(
            rmse,
            bins=30,
            alpha=0.5,
            label=ccy,
            density=True
        )

    plt.xlabel("RMSE (bp)", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.title("In-sample RMSE Distribution by Currency", fontsize=15)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('fig/rmse_across_currencies.png', dpi=150)
    plt.show()


def plot_world_map(model, datasets, currencies, device='cpu'):
    """
    Visualize latent points and covariance ellipses per currency.
    """
    model.eval()

    _, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Panel (a) ─────────────────────
    ax = axes[0]
    all_z = []
    all_labels = []

    for ccy_idx, (ccy, dataset) in enumerate(
        zip(currencies, datasets)
    ):
        z_list = []
        with torch.no_grad():
            for i in range(len(dataset)):
                x, _ = dataset[i]
                x = x.unsqueeze(0).to(device)
                mu, _ = model.encode(x)
                z_list.append(mu.cpu().numpy()[0])

        z_arr = np.array(z_list)
        all_z.append(z_arr)
        all_labels.extend([ccy_idx] * len(z_arr))

        ax.scatter(z_arr[:, 0], z_arr[:, 1],
                   s=5,
                   color=currency_color[ccy],
                   label=ccy)
    for ccy_idx, (ccy, z_arr) in enumerate(
        zip(currencies, all_z)
    ):
        mean = np.mean(z_arr, axis=0)
        cov = np.cov(z_arr.T)

        _draw_ellipse(ax, mean, cov,
                      color=currency_color[ccy],
                      label=ccy, n_std=2.0)

        ax.annotate(ccy, mean,
                    fontsize=8,
                    ha='center', va='center',
                    color=currency_color[ccy])
    ax.set_xlabel('$z_1$', fontsize=15)
    ax.set_ylabel('$z_2$', fontsize=15)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title('(a) Latent Space - All Observations', fontsize=15)
    ax.legend(markerscale=5, loc='upper left',
              fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    # ── Panel (b): 2-sigma ───────────────────
    ax = axes[1]

    for ccy_idx, (ccy, z_arr) in enumerate(
        zip(currencies, all_z)
    ):
        mean = np.mean(z_arr, axis=0)
        cov = np.cov(z_arr.T)

        _draw_ellipse(ax, mean, cov,
                      color=currency_color[ccy],
                      label=ccy, n_std=2.0)

        ax.annotate(ccy, mean,
                    fontsize=8,
                    ha='center', va='center',
                    color=currency_color[ccy])

    ax.set_xlabel('$z_1$', fontsize=15)
    ax.set_ylabel('$z_2$', fontsize=15)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title('(b) 2-Sigma Ellipses by Currency', fontsize=15)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('fig/world_map.png', dpi=150)
    plt.show()


def _draw_ellipse(ax, mean, cov, color, label, n_std=1.5):
    """
    Draw a covariance ellipse centered at mean for 2D latent points.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(
        np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    )

    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=mean, width=width, height=height,
        angle=angle,
        edgecolor=color, facecolor='none',
        linewidth=1.5, label=label
    )
    ax.add_patch(ellipse)


def plot_reconstruction_vae_cvae(
        vae_model, cvae_model,
        datasets, currencies,
        n_samples=300, device='cpu'):
    """Overlay historical and reconstructed curves for VAE vs CVAE."""

    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    x_smooth = np.linspace(2, 30, 500)

    vae_model.eval()
    cvae_model.eval()

    with torch.no_grad():
        for i, (dataset, currency_name) in enumerate(zip(datasets,
                                                         currencies)):

            indices = np.random.choice(
                len(dataset), min(n_samples, len(dataset)),
                replace=False
            )

            titles = ['Historical', 'VAE', 'CVAE']
            for j in range(3):
                ax = axes[i, j]
                ax.set_title(f'{currency_name} - {titles[j]}', fontsize=15)
                ax.set_xlabel('Maturity (years)', fontsize=15)
                ax.set_ylabel('Swap Rate (%)', fontsize=15)
                ax.set_xlim(2, 30)
                ax.grid(True, alpha=0.3)
            for idx in indices:
                x, label = dataset[idx]

                x_t = x.unsqueeze(0).to(device)
                label_t = label.unsqueeze(0).to(device)

                # ===== VAE =====
                mu_vae, _ = vae_model.encode(x_t)
                x_vae = vae_model.decode(mu_vae)

                # ===== CVAE =====
                mu_cvae, _ = cvae_model.encode(x_t, label_t)
                x_cvae = cvae_model.decode(mu_cvae, label_t)

                x_orig = (x.numpy() * (S_MAX - S_MIN) + S_MIN) * 100
                x_vae = (x_vae.cpu().numpy()[0] *
                         (S_MAX - S_MIN) + S_MIN) * 100
                x_cvae = (x_cvae.cpu().numpy()[0]
                          * (S_MAX - S_MIN) + S_MIN) * 100

                f_org = PchipInterpolator(MATURITIES, x_orig)
                f_vae = PchipInterpolator(MATURITIES, x_vae)
                f_cvae = PchipInterpolator(MATURITIES, x_cvae)

                axes[i, 0].plot(x_smooth, f_org(x_smooth),
                                alpha=0.5, linewidth=1.2)
                axes[i, 1].plot(x_smooth, f_vae(x_smooth),
                                alpha=0.5, linewidth=1.2)
                axes[i, 2].plot(x_smooth, f_cvae(x_smooth),
                                alpha=0.5, linewidth=1.2)

    plt.tight_layout()
    plt.savefig('fig/reconstruction_vae_vs_cvae.png', dpi=150)
    plt.show()


def plot_ellipse_decoding_multi(model, datasets, currencies,
                                all_z_background,
                                n_points=60, device='cpu'):
    """Decode points sampled along latent ellipses into yield curves."""

    model.eval()
    n_rows = len(datasets)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_points))

    x_smooth = np.linspace(min(MATURITIES), max(MATURITIES), 100)

    with torch.no_grad():
        for i, (dataset, currency_name) in enumerate(zip(datasets,
                                                         currencies)):

            # ===== latent =====
            z_list = []
            for j in range(len(dataset)):
                x, _ = dataset[j]
                x = x.unsqueeze(0).to(device)
                mu, _ = model.encode(x)
                z_list.append(mu.cpu().numpy()[0])

            z_arr = np.array(z_list)
            mean = np.mean(z_arr, axis=0)
            cov = np.cov(z_arr.T)

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # ===== axes =====
            ax_latent = axes[i, 0]
            ax_curves = axes[i, 1]

            # ===== latent plot =====
            ax_latent.scatter(all_z_background[:, 0], all_z_background[:, 1],
                              s=2, color='lightgray', alpha=0.5)

            ax_latent.scatter(z_arr[:, 0], z_arr[:, 1],
                              s=5, color='#0077b6', alpha=0.7)

            ax_latent.set_title(f'{currency_name} - Latent Space', fontsize=15)
            ax_latent.set_xlabel('$z_1$', fontsize=15)
            ax_latent.set_ylabel('$z_2$', fontsize=15)
            ax_latent.set_xlim(-4, 4)
            ax_latent.set_ylim(-4, 4)

            # ===== curves plot =====
            ax_curves.set_title(
                f'{currency_name} - Decoded Curves', fontsize=15)
            ax_curves.set_xlabel('Maturity (years)', fontsize=15)
            ax_curves.set_ylabel('Swap Rate (%)', fontsize=15)
            ax_curves.set_xlim(2, 30)
            ax_curves.set_ylim(-2, 9)

            ellipse_points = []

            for angle, color in zip(angles, colors):

                point_local = 2.0 * np.array([
                    np.sqrt(np.maximum(eigenvalues[0], 1e-9)) * np.cos(angle),
                    np.sqrt(np.maximum(eigenvalues[1], 1e-9)) * np.sin(angle)
                ])

                z_point = mean + eigenvectors @ point_local
                ellipse_points.append(z_point)

                # latent
                ax_latent.scatter(z_point[0], z_point[1],
                                  color=color, s=15, edgecolors='black',
                                  linewidth=0.5)

                # decode
                z_tensor = torch.tensor(
                    z_point, dtype=torch.float32
                ).unsqueeze(0).to(device)

                decoded_x = model.decode(z_tensor).cpu().numpy()[0]
                curve_pct = (decoded_x * (S_MAX - S_MIN) + S_MIN) * 100

                f = PchipInterpolator(MATURITIES, curve_pct)
                ax_curves.plot(x_smooth, f(x_smooth),
                               color=color, linewidth=1.2, alpha=0.8)

            ellipse_points = np.array(ellipse_points)
            ax_latent.plot(
                np.append(ellipse_points[:, 0], ellipse_points[0, 0]),
                np.append(ellipse_points[:, 1], ellipse_points[0, 1]),
                color='black', linestyle='--', linewidth=1
            )

    plt.tight_layout()
    plt.savefig('fig/ellipse_decoding_all.png', dpi=150)
    plt.show()
