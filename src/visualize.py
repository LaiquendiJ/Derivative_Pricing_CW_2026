import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Ellipse
# from sklearn.decomposition import PCA

MATURITIES = [2, 3, 5, 10, 15, 20, 30]
S_MIN, S_MAX = 0.02, 0.06


def plot_training_history(history):
    """
    loss function
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, title in zip(
        axes,
        ['total', 'recon', 'kld'],
        ['Total Loss', 'Reconstruction Loss', 'KLD Loss']
    ):
        ax.plot(history[f'train_{key}'], label='Train', linewidth=1)
        ax.plot(history[f'test_{key}'],  label='Test',  linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def plot_rmse_distribution(rmse_dict, title='RMSE Distribution'):
    """
    Figure 9, 10, 11

    Args:
        rmse_dict: {'Method Name': rmse_array,...}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['green', 'orange', 'blue', 'red']

    for (name, rmse), color in zip(rmse_dict.items(), colors):
        ax.hist(rmse, bins=50, alpha=0.5,
                label=name, color=color,
                density=True, edgecolor='none')

    ax.set_xlabel('Swap Rate RMSE (bp)')
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig('rmse_distribution.png', dpi=150)
    plt.show()


def plot_world_map(model, datasets, currencies, device='cpu'):
    """
    Figure 13
    """
    model.eval()

    colors = plt.cm.tab10(np.linspace(0, 1, len(currencies)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

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
                   s=2, alpha=0.3,
                   color=colors[ccy_idx],
                   label=ccy)

    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    ax.set_title('(a) Latent Space - All Observations')
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
                      color=colors[ccy_idx],
                      label=ccy, n_std=2.0)

        ax.annotate(ccy, mean,
                    fontsize=8,
                    ha='center', va='center',
                    color=colors[ccy_idx])

    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    ax.set_title('(b) 2-Sigma Ellipses by Currency')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('world_map.png', dpi=150)
    plt.show()


def _draw_ellipse(ax, mean, cov, color, label, n_std=2.0):
    """
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


def plot_reconstruction(model, dataset, currency_name,
                        n_samples=50, device='cpu'):
    """
    Figure 14
    """
    model.eval()

    indices = np.random.choice(
        len(dataset), min(n_samples, len(dataset)),
        replace=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, title in zip(axes, ['Historical', 'Reconstructed']):
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Swap Rate (%)')
        ax.set_title(f'{currency_name} - {title}')
        ax.set_xticks(MATURITIES)
        ax.grid(True, alpha=0.3)

    with torch.no_grad():
        for idx in indices:
            x, _ = dataset[idx]
            x_tensor = x.unsqueeze(0).to(device)

            x_recon, _, _ = model(x_tensor)

            x_orig = x.numpy() * (S_MAX - S_MIN) + S_MIN
            x_rec = x_recon.cpu().numpy()[0] * (S_MAX - S_MIN) + S_MIN

            x_orig_pct = x_orig * 100
            x_rec_pct = x_rec * 100

            axes[0].plot(MATURITIES, x_orig_pct,
                         alpha=0.2, linewidth=0.5, color='blue')
            axes[1].plot(MATURITIES, x_rec_pct,
                         alpha=0.2, linewidth=0.5, color='orange')

    plt.tight_layout()
    plt.savefig(f'reconstruction_{currency_name}.png', dpi=150)
    plt.show()


def plot_ellipse_decoding(model, dataset, currency_name,
                          n_points=20, device='cpu'):
    """
    Figure 15
    """
    model.eval()

    z_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
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

    # sample points
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_points))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_latent = axes[0]
    ax_latent.scatter(z_arr[:, 0], z_arr[:, 1],
                      s=2, alpha=0.3, color='blue',
                      label=currency_name)
    ax_latent.set_xlabel('$z_1$')
    ax_latent.set_ylabel('$z_2$')
    ax_latent.set_title(f'{currency_name} - Latent Space')

    ax_curves = axes[1]
    ax_curves.set_xlabel('Maturity (years)')
    ax_curves.set_ylabel('Swap Rate (%)')
    ax_curves.set_title(f'{currency_name} - Decoded Curves')
    ax_curves.set_xticks(MATURITIES)
    ax_curves.grid(True, alpha=0.3)

    with torch.no_grad():
        for angle, color in zip(angles, colors):
            point_local = 2.0 * np.array([
                np.sqrt(eigenvalues[0]) * np.cos(angle),
                np.sqrt(eigenvalues[1]) * np.sin(angle)
            ])
            z_point = mean + eigenvectors @ point_local

            ax_latent.scatter(z_point[0], z_point[1],
                              s=50, color=color, zorder=5)

            z_tensor = torch.tensor(
                z_point, dtype=torch.float32
            ).unsqueeze(0).to(device)

            curve = model.decode(z_tensor).cpu().numpy()[0]
            curve_pct = (curve * (S_MAX - S_MIN) + S_MIN) * 100

            ax_curves.plot(MATURITIES, curve_pct,
                           color=color, linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(f'ellipse_decoding_{currency_name}.png', dpi=150)
    plt.show()
