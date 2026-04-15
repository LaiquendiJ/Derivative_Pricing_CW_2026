import torch
import numpy as np

from src.data_loader import (load_all_currencies,
                             get_dataloaders,
                             SwapRateDataset)

from src.model import MultiCurrencyVAE
from src.train import train_vae, compute_rmse
from src.visualize import (plot_training_history,
                           plot_rmse_distribution,
                           plot_world_map,
                           plot_reconstruction,
                           plot_ellipse_decoding)

# ── setup ──────────────────────────────────────────
CONFIG = {
    'currencies':  ['GBP', 'EUR', 'USD'],
    'split_date':  '2024-07-01',
    'start_date':  '2023-01-30',
    'batch_size':  32,
    'n_epochs':    500,
    'lr':          1e-3,
    'beta':        1e-7,
    'latent_dim':  2,
    'input_dim':   7,
    'device':      'cuda' if torch.cuda.is_available() else 'cpu',
    'seed':        42,
}


def main():

    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print("=" * 50)
    print("Multi-Currency VAE for Yield Curves")
    print("=" * 50)
    print(f"Device:     {CONFIG['device']}")
    print(f"Currencies: {CONFIG['currencies']}")
    print(f"Split date: {CONFIG['split_date']}")

    print("\n[1] Loading Bloomberg data...")
    dfs = load_all_currencies(start_date=CONFIG['start_date'])

    print("\n[2] Creating dataloaders...")
    train_loader, test_loader, train_dataset, test_dataset = \
        get_dataloaders(
            dfs,
            currencies=CONFIG['currencies'],
            batch_size=CONFIG['batch_size'],
            split_date=CONFIG['split_date']
        )

    print("\n[3] Initializing model...")
    model = MultiCurrencyVAE(
        input_dim=CONFIG['input_dim'],
        latent_dim=CONFIG['latent_dim']
    )
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")

    print("\n[4] Training VAE...")
    history = train_vae(
        model, train_loader, test_loader,
        n_epochs=CONFIG['n_epochs'],
        lr=CONFIG['lr'],
        beta=CONFIG['beta'],
        device=CONFIG['device'],
        verbose=True
    )

    print("\n[5] Computing RMSE...")
    train_rmse = compute_rmse(
        model, train_dataset, CONFIG['device']
    )
    test_rmse = compute_rmse(
        model, test_dataset, CONFIG['device']
    )

    print(f"In-sample  RMSE: "
          f"mean = {train_rmse.mean():.2f} bp, "
          f"median = {np.median(train_rmse):.2f} bp")
    print(f"Out-sample RMSE: "
          f"mean = {test_rmse.mean():.2f} bp, "
          f"median = {np.median(test_rmse):.2f} bp")

    # ── Visualization ─────────────────────────────────
    print("\n[6] Generating plots...")

    plot_training_history(history)

    # RMSE
    plot_rmse_distribution(
        {
            'In-sample':     train_rmse,
            'Out-of-sample': test_rmse
        },
        title='In-sample vs Out-of-sample RMSE'
    )

    per_ccy_train = [
        SwapRateDataset(
            dfs,
            currencies=[ccy],
            train=True,
            split_date=CONFIG['split_date']
        )
        for ccy in CONFIG['currencies']
    ]

    plot_world_map(
        model, per_ccy_train,
        CONFIG['currencies'],
        CONFIG['device']
    )

    for ccy, dataset in zip(CONFIG['currencies'], per_ccy_train):
        plot_reconstruction(
            model, dataset, ccy,
            device=CONFIG['device']
        )
        plot_ellipse_decoding(
            model, dataset, ccy,
            device=CONFIG['device']
        )

    torch.save(model.state_dict(), 'vae_model.pth')
    print("\nModel saved to vae_model.pth")
    print("Done!")


if __name__ == '__main__':
    main()
