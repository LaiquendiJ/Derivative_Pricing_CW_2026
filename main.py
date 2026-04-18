"""Entry point for training/evaluating VAE and CVAE models on swap-rate curves."""

import torch
import numpy as np

from src.data_loader import (load_all_currencies,
                             get_dataloaders,
                             SwapRateDataset)
from src.model import MultiCurrencyVAE, MultiCurrencyCVAE
from src.train import (train_vae, train_cvae,
                       compute_rmse_vae, compute_rmse_cvae)
from src.visualize import (plot_training_history,
                           plot_rmse_distribution,
                           plot_world_map,
                           plot_reconstruction_vae_cvae,
                           plot_ellipse_decoding_multi,
                           plot_rmse_by_currency)

CONFIG = {
    'currencies':  ['GBP', 'EUR', 'USD'],
    'split_date':  '2024-01-01',
    'start_date':  '2009-01-01',
    'batch_size':  64,
    'n_epochs':    3000,
    'lr':          1e-3,
    'beta':        1e-7,
    'latent_dim':  2,
    'input_dim':   7,
    'n_currencies': 3,
    'device':      'cuda' if torch.cuda.is_available() else 'cpu',
    'seed':        42,
}

# Notes from an earlier experiment setup/results kept for reference:
# - split_date: 2024-01-01
# - start_date: 2015-01-01
# - batch_size: 64
# - n_epochs: 1000
# - lr: 1e-3
# - beta: 1e-7
# - VAE mean RMSE: 5.65bp (in), 14.46bp (out)
# - CVAE mean RMSE: 2.85bp (in), 7.06bp (out)


def run_vae(dfs, train_loader, test_loader,
            train_dataset, test_dataset):
    """Train the unconditional VAE and report/save reconstruction metrics."""
    print("\n" + "="*50)
    print("Training Multi-Currency VAE")
    print("="*50)

    model = MultiCurrencyVAE(
        input_dim=CONFIG['input_dim'],
        latent_dim=CONFIG['latent_dim']
    )

    history = train_vae(
        model, train_loader, test_loader,
        n_epochs=CONFIG['n_epochs'],
        lr=CONFIG['lr'],
        beta=CONFIG['beta'],
        device=CONFIG['device'],
        verbose=True
    )

    train_rmse = compute_rmse_vae(model, train_dataset, CONFIG['device'])
    test_rmse = compute_rmse_vae(model, test_dataset,  CONFIG['device'])

    print(f"\nVAE In-sample  RMSE: "
          f"mean={train_rmse.mean():.2f} bp, "
          f"median={np.median(train_rmse):.2f} bp")
    print(f"VAE Out-sample RMSE: "
          f"mean={test_rmse.mean():.2f} bp, "
          f"median={np.median(test_rmse):.2f} bp")

    torch.save(model.state_dict(), 'vae_model.pth')
    return model, history, train_rmse, test_rmse


def run_cvae(dfs, train_loader, test_loader,
             train_dataset, test_dataset):
    """Train the conditional VAE and report/save reconstruction metrics."""
    print("\n" + "="*50)
    print("Training Multi-Currency CVAE")
    print("="*50)

    model = MultiCurrencyCVAE(
        input_dim=CONFIG['input_dim'],
        latent_dim=CONFIG['latent_dim'],
        n_currencies=CONFIG['n_currencies']
    )

    history = train_cvae(
        model, train_loader, test_loader,
        n_epochs=CONFIG['n_epochs'],
        lr=CONFIG['lr'],
        beta=CONFIG['beta'],
        device=CONFIG['device'],
        verbose=True
    )

    train_rmse = compute_rmse_cvae(
        model, train_dataset, CONFIG['device']
    )
    test_rmse = compute_rmse_cvae(
        model, test_dataset, CONFIG['device']
    )

    print(f"\nCVAE In-sample  RMSE: "
          f"mean={train_rmse.mean():.2f} bp, "
          f"median={np.median(train_rmse):.2f} bp")
    print(f"CVAE Out-sample RMSE: "
          f"mean={test_rmse.mean():.2f} bp, "
          f"median={np.median(test_rmse):.2f} bp")

    torch.save(model.state_dict(), 'cvae_model.pth')
    return model, history, train_rmse, test_rmse


def main():
    """Run end-to-end data loading, training, evaluation, and plotting."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print("=" * 50)
    print("Multi-Currency VAE & CVAE for Yield Curves")
    print("=" * 50)

    # ── 1. ───────────────────────────────
    print("\n[1] Loading Bloomberg data...")
    dfs = load_all_currencies(start_date=CONFIG['start_date'])

    # ── 2. DataLoader ─────────────────────────────
    print("\n[2] Creating dataloaders...")
    train_loader, test_loader, train_dataset, test_dataset = \
        get_dataloaders(
            dfs,
            currencies=CONFIG['currencies'],
            batch_size=CONFIG['batch_size'],
            split_date=CONFIG['split_date']
        )

    # ── 3. VAE ───────────────────────────────
    vae_model, vae_history, vae_train_rmse, vae_test_rmse = \
        run_vae(dfs, train_loader, test_loader,
                train_dataset, test_dataset)

    # ── 4. CVAE ──────────────────────────────
    cvae_model, cvae_history, cvae_train_rmse, cvae_test_rmse = \
        run_cvae(dfs, train_loader, test_loader,
                 train_dataset, test_dataset)

    # ── 5. RMSE ──────────────────────────────
    print("\n" + "="*50)
    print("RMSE Comparison")
    print("="*50)
    print(f"{'Method':<10} {'In-sample':>12} {'Out-sample':>12}")
    print("-" * 36)
    print(f"{'VAE':<10} "
          f"{vae_train_rmse.mean():>10.2f}bp "
          f"{vae_test_rmse.mean():>10.2f}bp")
    print(f"{'CVAE':<10} "
          f"{cvae_train_rmse.mean():>10.2f}bp "
          f"{cvae_test_rmse.mean():>10.2f}bp")

    # ── 6. Visualization ─────────────────────────────────
    print("\n[6] Generating plots...")

    plot_training_history(vae_history,  title='VAE')
    plot_training_history(cvae_history, title='CVAE')

    # RMSE comparison plots for in/out sample and VAE vs CVAE.
    plot_rmse_distribution(
        {
            'VAE in-sample':    vae_train_rmse,
            'VAE out-of-sample':   vae_test_rmse,
        },
        title='VAE_in_out_of_sample_RMSE'
    )
    plot_rmse_distribution(
        {
            'VAE':    vae_train_rmse,
            'CVAE':   cvae_train_rmse,
        },
        title='VAE_CVAE_RMSE'
    )
    per_ccy_rmse = {}

    for ccy in CONFIG['currencies']:
        dataset = SwapRateDataset(
            dfs,
            currencies=[ccy],
            train=True,
            split_date=CONFIG['split_date']
        )

        rmse = compute_rmse_vae(vae_model, dataset, CONFIG['device'])
        per_ccy_rmse[ccy] = rmse

    plot_rmse_by_currency(per_ccy_rmse)

    per_ccy_train = [
        SwapRateDataset(
            dfs, currencies=[ccy],
            train=True, split_date=CONFIG['split_date']
        )
        for ccy in CONFIG['currencies']
    ]

    plot_world_map(
        vae_model, per_ccy_train,
        CONFIG['currencies'], CONFIG['device']
    )

    plot_reconstruction_vae_cvae(
        vae_model,
        cvae_model,
        per_ccy_train[:3],
        CONFIG['currencies'][:3],
        device=CONFIG['device']
    )

    all_z_list = []
    vae_model.eval()
    with torch.no_grad():
        for dataset in per_ccy_train:
            for i in range(len(dataset)):
                x, _ = dataset[i]
                mu, _ = vae_model.encode(x.unsqueeze(0).to(CONFIG['device']))
                all_z_list.append(mu.cpu().numpy()[0])
    all_z_background = np.array(all_z_list)

    plot_ellipse_decoding_multi(
        vae_model,
        per_ccy_train[:3],
        CONFIG['currencies'][:3],
        all_z_background,
        device=CONFIG['device']
    )
    torch.save(vae_model.state_dict(), 'model/vae_model.pth')
    torch.save(cvae_model.state_dict(), 'model/cvae_model.pth')

    print("\nDone!")


if __name__ == '__main__':
    main()
