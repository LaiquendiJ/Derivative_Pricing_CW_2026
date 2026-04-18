"""Data loading and dataset utilities for multi-currency swap-rate curves."""

import pandas as pd
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

S_MIN, S_MAX = -0.02, 0.08
MATURITIES = [2, 3, 5, 10, 15, 20, 30]
TARGET_TENORS = ['2Y', '3Y', '5Y', '10Y', '15Y', '20Y', '30Y']

FILE_PATH = "Bloomberg - Historical Data v2026-04-16.xlsx"

CURRENCIES = ['GBP', 'EUR', 'USD']

SHEET_NAMES = {
    'GBP': 'gbp ois results',
    'EUR': 'eur estr results',
    'USD': 'usd sofr results'
}

TICKER_MAPPING = {
    'gbp ois results': ['BPSWS',  ' Curncy'],
    'eur estr results': ['EESWE', ' Curncy'],
    'usd sofr results': ['USOSFR', ' Curncy']
}

TABLE_MAPPING = {}
for sheet, (prefix, suffix) in TICKER_MAPPING.items():
    TABLE_MAPPING[sheet] = [
        prefix + t + suffix
        for t in ['2', '3', '5', '10', '15', '20', '30']
    ]


def find_ticker_col(raw_df, ticker):
    """Locate the column index containing a ticker string in row 0."""
    for col in range(raw_df.shape[1]):
        cell = str(raw_df.iloc[0, col]).strip()
        if ticker.lower() in cell.lower():
            return col
    return None


def get_data(sheet_name, start_date="2023-01-30"):
    """Load one currency sheet, align tenor columns, and filter by date."""

    TARGET_TENORS = ['Date', '2Y', '3Y', '5Y', '10Y', '15Y', '20Y', '30Y']
    result_df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)
    result_df.columns = TARGET_TENORS
    result_df = result_df.dropna(subset=TARGET_TENORS)
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    result_df = result_df[result_df['Date'] >= start_date]
    result_df = result_df.reset_index(drop=True)
    return result_df


def load_all_currencies(start_date="2023-01-30"):
    """
    Return a dict with one cleaned DataFrame per configured currency.
    """
    dfs = {}
    for ccy in CURRENCIES:
        sheet = SHEET_NAMES[ccy]
        df = get_data(sheet, start_date)
        dfs[ccy] = df
        print(f"{ccy}: {len(df)} observations "
              f"({df['Date'].min().date()} to "
              f"{df['Date'].max().date()})")
    return dfs


# ── Dataset ──────────────
class SwapRateDataset(Dataset):
    """
    PyTorch dataset of normalized swap rates with currency labels.
    """

    def __init__(self, dfs, currencies=CURRENCIES,
                 train=True, split_date='2024-01-01'):
        """Build train/test split and cache normalized arrays in memory."""
        self.currencies = currencies
        self.split_date = pd.Timestamp(split_date)
        self.train = train
        self.all_currencies = CURRENCIES
        self.data, self.labels, self.dates = \
            self._prepare(dfs)

    def _prepare(self, dfs):
        """Stack selected currencies and create labels for model conditioning."""
        all_rates = []
        all_labels = []
        all_dates = []

        for ccy in self.currencies:
            df = dfs[ccy].copy()
            ccy_idx = self.all_currencies.index(ccy)
            if self.train:
                df = df[df['Date'] <= self.split_date]
            else:
                df = df[df['Date'] > self.split_date]

            if len(df) == 0:
                print(f"Warning: No data for {ccy} "
                      f"in {'train' if self.train else 'test'} set")
                continue

            rates_pct = df[TARGET_TENORS].values  # (N, 7)

            rates = rates_pct / 100.0

            # Normalize rates into [0, 1] range expected by Sigmoid decoder.
            rates_norm = (rates - S_MIN) / (S_MAX - S_MIN)
            rates_norm = np.clip(rates_norm, 0.0, 1.0)

            all_rates.append(rates_norm.astype(np.float32))
            all_labels.append(
                np.full(len(rates_norm), ccy_idx, dtype=np.int64)
            )
            all_dates.extend(df['Date'].tolist())

            split_str = 'train' if self.train else 'test'
            print(f"  {ccy} ({split_str}): {len(rates_norm)} obs")

        if len(all_rates) == 0:
            raise ValueError("No data loaded!")

        data = np.vstack(all_rates)
        labels = np.concatenate(all_labels)

        return data, labels, all_dates

    def __len__(self):
        """Number of observations in this split."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return one normalized curve and its integer currency label."""
        return (
            torch.tensor(self.data[idx],   dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

    def get_rates_original(self, idx):
        """Inverse-transform one normalized sample back to percentage rates."""

        rates_norm = self.data[idx]
        rates = rates_norm * (S_MAX - S_MIN) + S_MIN


        return rates * 100.0  



def get_dataloaders(dfs, currencies=CURRENCIES,
                    batch_size=64, split_date='2024-01-01'):
    """Create train/test datasets and DataLoaders with a date split."""
    print("\nPreparing training set:")
    train_dataset = SwapRateDataset(
        dfs, currencies, train=True,  split_date=split_date
    )

    print("\nPreparing test set:")
    test_dataset = SwapRateDataset(
        dfs, currencies, train=False, split_date=split_date
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False
    )

    print(f"\nTotal train samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    return train_loader, test_loader, train_dataset, test_dataset
