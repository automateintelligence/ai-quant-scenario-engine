"""Technical indicator calculations for price paths."""

from __future__ import annotations

import numpy as np


def compute_rsi(
    price_paths: np.ndarray,
    period: int = 14,
    fillna: bool = True
) -> np.ndarray:
    """
    Compute RSI (Relative Strength Index) for price paths.

    RSI = 100 - (100 / (1 + RS))
    where RS = average gain / average loss over the period

    Args:
        price_paths: Price array of shape [n_paths, n_steps]
        period: RSI period (default: 14)
        fillna: If True, fill NaN values at start with 50 (neutral)

    Returns:
        RSI array of shape [n_paths, n_steps] with values 0-100
    """
    prices = np.asarray(price_paths, dtype=float)
    n_paths, n_steps = prices.shape

    if period <= 0:
        raise ValueError("RSI period must be positive")
    if period >= n_steps:
        # Not enough data for RSI calculation
        if fillna:
            return np.full_like(prices, 50.0, dtype=float)
        return np.full_like(prices, np.nan, dtype=float)

    # Calculate price changes
    deltas = np.diff(prices, axis=1)  # shape: [n_paths, n_steps-1]

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initialize RSI array
    rsi = np.zeros_like(prices, dtype=float)

    # For each path, compute RSI
    for i in range(n_paths):
        path_gains = gains[i]
        path_losses = losses[i]

        # First average: simple average over period
        avg_gain = np.mean(path_gains[:period])
        avg_loss = np.mean(path_losses[:period])

        # Compute RSI for first period
        if avg_loss == 0:
            rsi[i, period] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i, period] = 100.0 - (100.0 / (1.0 + rs))

        # Smoothed average for subsequent periods (Wilder's smoothing)
        for j in range(period + 1, n_steps):
            avg_gain = (avg_gain * (period - 1) + path_gains[j - 1]) / period
            avg_loss = (avg_loss * (period - 1) + path_losses[j - 1]) / period

            if avg_loss == 0:
                rsi[i, j] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi[i, j] = 100.0 - (100.0 / (1.0 + rs))

    # Handle initial NaN values
    if fillna:
        rsi[:, :period] = 50.0  # Neutral RSI for warmup period
    else:
        rsi[:, :period] = np.nan

    return rsi


def compute_sma(
    price_paths: np.ndarray,
    period: int,
    fillna: bool = True
) -> np.ndarray:
    """
    Compute Simple Moving Average for price paths.

    Args:
        price_paths: Price array of shape [n_paths, n_steps]
        period: SMA period
        fillna: If True, fill initial NaN values with price

    Returns:
        SMA array of shape [n_paths, n_steps]
    """
    prices = np.asarray(price_paths, dtype=float)
    n_paths, n_steps = prices.shape

    if period <= 0:
        raise ValueError("SMA period must be positive")
    if period > n_steps:
        period = n_steps

    kernel = np.ones(period) / period
    sma = np.zeros_like(prices, dtype=float)

    for i in range(n_paths):
        conv = np.convolve(prices[i], kernel, mode="same")
        sma[i] = conv

    if fillna:
        # Fill initial values with actual price
        for i in range(min(period - 1, n_steps)):
            sma[:, i] = prices[:, i]

    return sma


def compute_ema(
    price_paths: np.ndarray,
    period: int,
    fillna: bool = True
) -> np.ndarray:
    """
    Compute Exponential Moving Average for price paths.

    Args:
        price_paths: Price array of shape [n_paths, n_steps]
        period: EMA period
        fillna: If True, use SMA for initial value

    Returns:
        EMA array of shape [n_paths, n_steps]
    """
    prices = np.asarray(price_paths, dtype=float)
    n_paths, n_steps = prices.shape

    if period <= 0:
        raise ValueError("EMA period must be positive")

    alpha = 2.0 / (period + 1.0)
    ema = np.zeros_like(prices, dtype=float)

    for i in range(n_paths):
        if fillna:
            # Initialize with SMA
            ema[i, 0] = np.mean(prices[i, :min(period, n_steps)])
        else:
            ema[i, 0] = prices[i, 0]

        for j in range(1, n_steps):
            ema[i, j] = alpha * prices[i, j] + (1 - alpha) * ema[i, j - 1]

    return ema


def compute_bollinger_bands(
    price_paths: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
    fillna: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Bollinger Bands for price paths.

    Args:
        price_paths: Price array of shape [n_paths, n_steps]
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        fillna: If True, fill initial NaN values

    Returns:
        Tuple of (upper_band, middle_band, lower_band), each [n_paths, n_steps]
    """
    prices = np.asarray(price_paths, dtype=float)
    n_paths, n_steps = prices.shape

    if period <= 0:
        raise ValueError("Bollinger period must be positive")
    if period > n_steps:
        period = n_steps

    middle = compute_sma(prices, period, fillna=fillna)

    # Compute rolling standard deviation
    std = np.zeros_like(prices, dtype=float)
    for i in range(n_paths):
        for j in range(period - 1, n_steps):
            window = prices[i, max(0, j - period + 1):j + 1]
            std[i, j] = np.std(window, ddof=1) if len(window) > 1 else 0.0

    if fillna:
        # Fill initial std with overall std
        for i in range(min(period - 1, n_steps)):
            std[:, i] = np.std(prices[:, :min(period, n_steps)], axis=1, keepdims=False)

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return upper, middle, lower


def compute_atr(
    price_paths: np.ndarray,
    high_paths: np.ndarray | None = None,
    low_paths: np.ndarray | None = None,
    period: int = 14,
    fillna: bool = True
) -> np.ndarray:
    """
    Compute Average True Range for price paths.

    If high/low paths not provided, uses price for all ranges.

    Args:
        price_paths: Close price array [n_paths, n_steps]
        high_paths: High price array [n_paths, n_steps] (optional)
        low_paths: Low price array [n_paths, n_steps] (optional)
        period: ATR period (default: 14)
        fillna: If True, fill initial values

    Returns:
        ATR array of shape [n_paths, n_steps]
    """
    prices = np.asarray(price_paths, dtype=float)
    n_paths, n_steps = prices.shape

    if high_paths is None:
        high_paths = prices
    if low_paths is None:
        low_paths = prices

    high = np.asarray(high_paths, dtype=float)
    low = np.asarray(low_paths, dtype=float)

    if period <= 0:
        raise ValueError("ATR period must be positive")

    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr = np.zeros_like(prices, dtype=float)

    for i in range(n_paths):
        tr[i, 0] = high[i, 0] - low[i, 0]
        for j in range(1, n_steps):
            hl = high[i, j] - low[i, j]
            hc = abs(high[i, j] - prices[i, j - 1])
            lc = abs(low[i, j] - prices[i, j - 1])
            tr[i, j] = max(hl, hc, lc)

    # ATR is smoothed average of TR
    atr = np.zeros_like(prices, dtype=float)
    for i in range(n_paths):
        # Initial ATR
        atr[i, period - 1] = np.mean(tr[i, :period])

        # Wilder's smoothing
        for j in range(period, n_steps):
            atr[i, j] = (atr[i, j - 1] * (period - 1) + tr[i, j]) / period

    if fillna:
        for i in range(min(period - 1, n_steps)):
            atr[:, i] = atr[:, period - 1]

    return atr


def compute_macd(
    price_paths: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    fillna: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Args:
        price_paths: Price array [n_paths, n_steps]
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        fillna: If True, fill initial values

    Returns:
        Tuple of (macd_line, signal_line, histogram), each [n_paths, n_steps]
    """
    fast_ema = compute_ema(price_paths, fast_period, fillna=fillna)
    slow_ema = compute_ema(price_paths, slow_period, fillna=fillna)

    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, signal_period, fillna=fillna)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_stochastic(
    price_paths: np.ndarray,
    high_paths: np.ndarray | None = None,
    low_paths: np.ndarray | None = None,
    k_period: int = 14,
    d_period: int = 3,
    fillna: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Stochastic Oscillator (%K and %D).

    Args:
        price_paths: Close price array [n_paths, n_steps]
        high_paths: High price array [n_paths, n_steps] (optional)
        low_paths: Low price array [n_paths, n_steps] (optional)
        k_period: %K period (default: 14)
        d_period: %D smoothing period (default: 3)
        fillna: If True, fill initial values

    Returns:
        Tuple of (percent_k, percent_d), each [n_paths, n_steps]
    """
    prices = np.asarray(price_paths, dtype=float)
    n_paths, n_steps = prices.shape

    if high_paths is None:
        high_paths = prices
    if low_paths is None:
        low_paths = prices

    high = np.asarray(high_paths, dtype=float)
    low = np.asarray(low_paths, dtype=float)

    percent_k = np.zeros_like(prices, dtype=float)

    for i in range(n_paths):
        for j in range(k_period - 1, n_steps):
            window_high = np.max(high[i, max(0, j - k_period + 1):j + 1])
            window_low = np.min(low[i, max(0, j - k_period + 1):j + 1])

            if window_high - window_low == 0:
                percent_k[i, j] = 50.0
            else:
                percent_k[i, j] = 100.0 * (prices[i, j] - window_low) / (window_high - window_low)

    if fillna:
        for i in range(min(k_period - 1, n_steps)):
            percent_k[:, i] = 50.0

    # %D is SMA of %K
    percent_d = compute_sma(percent_k, d_period, fillna=fillna)

    return percent_k, percent_d


def compute_all_features(
    price_paths: np.ndarray,
    high_paths: np.ndarray | None = None,
    low_paths: np.ndarray | None = None,
    rsi_period: int = 14,
    sma_periods: list[int] | None = None,
    fillna: bool = True
) -> dict[str, np.ndarray]:
    """
    Compute all common technical indicators for price paths.

    Args:
        price_paths: Close price array [n_paths, n_steps]
        high_paths: High price array [n_paths, n_steps] (optional)
        low_paths: Low price array [n_paths, n_steps] (optional)
        rsi_period: RSI period (default: 14)
        sma_periods: List of SMA periods to compute (default: [20, 50, 200])
        fillna: If True, fill initial NaN values

    Returns:
        Dictionary of feature name -> array [n_paths, n_steps]
    """
    if sma_periods is None:
        sma_periods = [20, 50, 200]

    features = {}

    # RSI
    features["rsi"] = compute_rsi(price_paths, period=rsi_period, fillna=fillna)

    # SMAs
    for period in sma_periods:
        features[f"sma_{period}"] = compute_sma(price_paths, period=period, fillna=fillna)

    # EMAs
    features["ema_12"] = compute_ema(price_paths, period=12, fillna=fillna)
    features["ema_26"] = compute_ema(price_paths, period=26, fillna=fillna)

    # MACD
    macd_line, signal_line, histogram = compute_macd(price_paths, fillna=fillna)
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_histogram"] = histogram

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(price_paths, fillna=fillna)
    features["bb_upper"] = bb_upper
    features["bb_middle"] = bb_middle
    features["bb_lower"] = bb_lower

    # ATR
    features["atr"] = compute_atr(
        price_paths, high_paths=high_paths, low_paths=low_paths, fillna=fillna
    )

    # Stochastic
    stoch_k, stoch_d = compute_stochastic(
        price_paths, high_paths=high_paths, low_paths=low_paths, fillna=fillna
    )
    features["stochastic_k"] = stoch_k
    features["stochastic_d"] = stoch_d

    return features


__all__ = [
    "compute_rsi",
    "compute_sma",
    "compute_ema",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_macd",
    "compute_stochastic",
    "compute_all_features",
]
