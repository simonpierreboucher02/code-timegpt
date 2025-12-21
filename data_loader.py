import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset


class SP500StockDataset(Dataset):
    """
    Dataset for loading S&P 500 stock prices from Yahoo Finance.
    Implements the window sampling strategy described in the paper.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        context_length_range: Tuple[int, int] = (96, 512),
        forecast_horizon_range: Tuple[int, int] = (24, 96),
        use_returns: bool = True,
        cache_data: bool = True
    ):
        """
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for data download (YYYY-MM-DD)
            end_date: End date for data download (YYYY-MM-DD)
            context_length_range: (min_L, max_L) for randomized context length
            forecast_horizon_range: (min_H, max_H) for randomized forecast horizon
            use_returns: If True, use log returns instead of raw prices
            cache_data: If True, cache downloaded data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.L_min, self.L_max = context_length_range
        self.H_min, self.H_max = forecast_horizon_range
        self.use_returns = use_returns
        self.cache_data = cache_data

        # Download data for all tickers
        print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
        self.data = self._download_data()
        self.series_list = self._prepare_series()

        # Compute valid sampling ranges
        self._compute_sampling_ranges()

        print(f"Loaded {len(self.series_list)} series with total {self.total_samples} valid samples")

    def _download_data(self) -> Dict[str, pd.DataFrame]:
        """Download stock data from Yahoo Finance"""
        data_dict = {}

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)

                if len(df) < self.L_max + self.H_max:
                    print(f"Skipping {ticker}: insufficient data ({len(df)} days)")
                    continue

                # Use adjusted close price
                prices = df['Close'].values

                if self.use_returns:
                    # Compute log returns
                    returns = np.diff(np.log(prices))
                    values = returns
                else:
                    values = prices

                # Create time features
                dates = df.index

                data_dict[ticker] = {
                    'values': values,
                    'dates': dates[1:] if self.use_returns else dates,  # Align with returns
                    'ticker': ticker
                }

            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue

        return data_dict

    def _prepare_series(self) -> List[Dict]:
        """Prepare list of series with metadata"""
        series_list = []

        for i, (ticker, data) in enumerate(self.data.items()):
            series_list.append({
                'series_id': i,
                'ticker': ticker,
                'values': data['values'],
                'dates': data['dates'],
                'T': len(data['values'])
            })

        return series_list

    def _compute_sampling_ranges(self):
        """Compute valid anchor times for each series"""
        for series in self.series_list:
            T = series['T']
            # Valid anchor times: t ∈ {L_max, ..., T - H_max}
            series['valid_anchors'] = list(range(self.L_max, T - self.H_max + 1))

        # Compute total number of possible samples
        self.total_samples = sum(len(s['valid_anchors']) for s in self.series_list)

    def __len__(self) -> int:
        """Return number of possible samples"""
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample a training window according to the paper's sampling strategy.

        Returns:
            Dictionary containing:
                - y_ctx: context values (L,)
                - y_fut: future values (H,)
                - dates_ctx: context dates for time features
                - dates_fut: future dates for time features
                - series_id: series identifier
                - mu: normalization mean
                - sigma: normalization std
        """
        # Sample a series uniformly
        series_idx = np.random.randint(len(self.series_list))
        series = self.series_list[series_idx]

        # Sample an anchor time uniformly from valid anchors
        t = np.random.choice(series['valid_anchors'])

        # Sample context length L and horizon H
        L = np.random.randint(self.L_min, self.L_max + 1)
        H = np.random.randint(self.H_min, self.H_max + 1)

        # Ensure we have enough data
        while t - L < 0 or t + H > series['T']:
            t = np.random.choice(series['valid_anchors'])

        # Extract context and future windows
        y_ctx = series['values'][t - L:t]
        y_fut = series['values'][t:t + H]

        dates_ctx = series['dates'][t - L:t]
        dates_fut = series['dates'][t:t + H]

        # Local normalization (Eq. 2-3 in paper)
        mu = np.mean(y_ctx)
        sigma = np.std(y_ctx) + 1e-8  # epsilon for numerical stability

        # Normalize both context and future
        y_ctx_norm = (y_ctx - mu) / sigma
        y_fut_norm = (y_fut - mu) / sigma

        # Extract time features
        time_features_ctx = self._extract_time_features(dates_ctx)
        time_features_fut = self._extract_time_features(dates_fut)

        return {
            'y_ctx': torch.FloatTensor(y_ctx_norm),
            'y_fut': torch.FloatTensor(y_fut_norm),
            'time_features_ctx': torch.FloatTensor(time_features_ctx),
            'time_features_fut': torch.FloatTensor(time_features_fut),
            'series_id': series['series_id'],
            'mu': torch.FloatTensor([mu]),
            'sigma': torch.FloatTensor([sigma]),
            'L': L,
            'H': H
        }

    def _extract_time_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract time features from dates.
        Features: day_of_week, day_of_month, month, day_of_year (normalized)
        Plus Fourier features for periodicity
        """
        features = []

        for date in dates:
            # Basic calendar features (normalized to [0, 1])
            day_of_week = date.dayofweek / 6.0
            day_of_month = date.day / 31.0
            month = date.month / 12.0
            day_of_year = date.dayofyear / 365.0

            # Fourier features for weekly periodicity
            freq_week = 2 * np.pi * date.dayofweek / 7.0
            sin_week = np.sin(freq_week)
            cos_week = np.cos(freq_week)

            # Fourier features for annual periodicity
            freq_year = 2 * np.pi * date.dayofyear / 365.0
            sin_year = np.sin(freq_year)
            cos_year = np.cos(freq_year)

            feat = [
                day_of_week, day_of_month, month, day_of_year,
                sin_week, cos_week, sin_year, cos_year
            ]
            features.append(feat)

        return np.array(features, dtype=np.float32)


def get_sp500_tickers(num_tickers: Optional[int] = None) -> List[str]:
    """
    Get list of S&P 500 tickers.

    Args:
        num_tickers: If specified, return only first N tickers

    Returns:
        List of ticker symbols
    """
    # Popular S&P 500 stocks for testing
    # In practice, you'd want to fetch the full list from Wikipedia or another source
    sp500_sample = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
        'ABBV', 'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'TMO', 'PEP',
        'CSCO', 'ACN', 'LLY', 'ADBE', 'NKE', 'WMT', 'NFLX', 'ABT', 'CRM',
        'DHR', 'VZ', 'CMCSA', 'INTC', 'ORCL', 'AMD', 'TXN', 'NEE', 'BMY',
        'PM', 'UPS', 'RTX', 'QCOM', 'HON', 'UNP', 'IBM', 'LOW', 'AMGN',
        'BA', 'CAT', 'GE', 'SBUX', 'MDT', 'AXP', 'GS', 'ISRG', 'BLK',
        'MMM', 'DE', 'ADI', 'TJX', 'CVS', 'LMT', 'SYK', 'GILD', 'AMT',
        'MO', 'C', 'ADP', 'BKNG', 'ZTS', 'MDLZ', 'PLD', 'TMUS', 'CB',
        'SO', 'DUK', 'CI', 'REGN', 'NOW', 'SLB', 'CL', 'SCHW', 'MMC',
        'BDX', 'EOG', 'CSX', 'ITW', 'USB', 'PNC', 'ETN', 'NOC', 'HUM',
        'BSX', 'WM', 'AON', 'MU', 'CME', 'TGT', 'COP', 'FISV', 'APD'
    ]

    if num_tickers:
        return sp500_sample[:num_tickers]
    return sp500_sample


def collate_variable_length(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Find max lengths in batch
    max_L = max(item['L'] for item in batch)
    max_H = max(item['H'] for item in batch)

    batch_size = len(batch)
    time_feat_dim = batch[0]['time_features_ctx'].shape[-1]

    # Initialize padded tensors
    y_ctx_padded = torch.zeros(batch_size, max_L)
    y_fut_padded = torch.zeros(batch_size, max_H)
    time_features_ctx_padded = torch.zeros(batch_size, max_L, time_feat_dim)
    time_features_fut_padded = torch.zeros(batch_size, max_H, time_feat_dim)

    # Create attention masks
    ctx_mask = torch.zeros(batch_size, max_L, dtype=torch.bool)
    fut_mask = torch.zeros(batch_size, max_H, dtype=torch.bool)

    series_ids = []
    mus = []
    sigmas = []
    actual_L = []
    actual_H = []

    for i, item in enumerate(batch):
        L = item['L']
        H = item['H']

        y_ctx_padded[i, :L] = item['y_ctx']
        y_fut_padded[i, :H] = item['y_fut']
        time_features_ctx_padded[i, :L] = item['time_features_ctx']
        time_features_fut_padded[i, :H] = item['time_features_fut']

        ctx_mask[i, :L] = True
        fut_mask[i, :H] = True

        series_ids.append(item['series_id'])
        mus.append(item['mu'])
        sigmas.append(item['sigma'])
        actual_L.append(L)
        actual_H.append(H)

    return {
        'y_ctx': y_ctx_padded,
        'y_fut': y_fut_padded,
        'time_features_ctx': time_features_ctx_padded,
        'time_features_fut': time_features_fut_padded,
        'ctx_mask': ctx_mask,
        'fut_mask': fut_mask,
        'series_id': torch.LongTensor(series_ids),
        'mu': torch.stack(mus),
        'sigma': torch.stack(sigmas),
        'actual_L': torch.LongTensor(actual_L),
        'actual_H': torch.LongTensor(actual_H)
    }
