"""
Mathematical utilities for Market Decoder
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import statistics
from scipy import stats

def calculate_sma(data: List[float], period: int) -> List[float]:
    """
    Calculate Simple Moving Average
    
    Args:
        data: List of values
        period: SMA period
        
    Returns:
        List of SMA values
    """
    if len(data) < period:
        return [None] * len(data)
    
    sma_values = []
    for i in range(len(data)):
        if i < period - 1:
            sma_values.append(None)
        else:
            sma = sum(data[i-period+1:i+1]) / period
            sma_values.append(sma)
    
    return sma_values

def calculate_ema(data: List[float], period: int, smoothing: float = 2.0) -> List[float]:
    """
    Calculate Exponential Moving Average
    
    Args:
        data: List of values
        period: EMA period
        smoothing: Smoothing factor
        
    Returns:
        List of EMA values
    """
    if len(data) < period:
        return [None] * len(data)
    
    ema_values = []
    multiplier = smoothing / (period + 1)
    
    # First EMA is SMA
    sma = sum(data[:period]) / period
    ema_values.extend([None] * (period - 1))
    ema_values.append(sma)
    
    # Calculate subsequent EMAs
    for i in range(period, len(data)):
        ema = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        ema_values.append(ema)
    
    return ema_values

def calculate_rsi(data: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index
    
    Args:
        data: List of prices
        period: RSI period
        
    Returns:
        List of RSI values
    """
    if len(data) < period + 1:
        return [None] * len(data)
    
    deltas = np.diff(data)
    seed = deltas[:period+1]
    
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(data)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
    
    for i in range(period, len(data)):
        delta = deltas[i-1]
        
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        rs = up / down if down != 0 else 0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    
    return rsi.tolist()

def calculate_macd(data: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: List of prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Dictionary with MACD, signal, and histogram
    """
    if len(data) < slow_period + signal_period:
        empty = [None] * len(data)
        return {'macd': empty, 'signal': empty, 'histogram': empty}
    
    # Calculate EMAs
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    # Calculate MACD line
    macd_line = []
    for i in range(len(data)):
        if fast_ema[i] is None or slow_ema[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(fast_ema[i] - slow_ema[i])
    
    # Calculate signal line (EMA of MACD)
    macd_valid = [v for v in macd_line if v is not None]
    if len(macd_valid) >= signal_period:
        signal_line = calculate_ema(macd_valid, signal_period)
        # Pad with None values to match original length
        none_count = len(data) - len(signal_line)
        signal_line = [None] * none_count + signal_line
    else:
        signal_line = [None] * len(data)
    
    # Calculate histogram
    histogram = []
    for i in range(len(data)):
        if macd_line[i] is None or signal_line[i] is None:
            histogram.append(None)
        else:
            histogram.append(macd_line[i] - signal_line[i])
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data: List[float], period: int = 20, num_std: float = 2.0) -> Dict[str, List[float]]:
    """
    Calculate Bollinger Bands
    
    Args:
        data: List of prices
        period: SMA period
        num_std: Number of standard deviations
        
    Returns:
        Dictionary with upper, middle, and lower bands
    """
    if len(data) < period:
        empty = [None] * len(data)
        return {'upper': empty, 'middle': empty, 'lower': empty}
    
    sma = calculate_sma(data, period)
    
    upper_band = []
    lower_band = []
    
    for i in range(len(data)):
        if i < period - 1:
            upper_band.append(None)
            lower_band.append(None)
        else:
            # Calculate standard deviation
            window = data[i-period+1:i+1]
            std_dev = statistics.stdev(window) if len(window) > 1 else 0
            
            upper = sma[i] + (num_std * std_dev)
            lower = sma[i] - (num_std * std_dev)
            
            upper_band.append(upper)
            lower_band.append(lower)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
    """
    Calculate Average True Range
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of closing prices
        period: ATR period
        
    Returns:
        List of ATR values
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input lists must have the same length")
    
    if len(high) < period + 1:
        return [None] * len(high)
    
    # Calculate True Range
    tr = []
    for i in range(len(high)):
        if i == 0:
            tr.append(high[i] - low[i])
        else:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr.append(max(tr1, tr2, tr3))
    
    # Calculate ATR
    atr = []
    for i in range(len(tr)):
        if i < period - 1:
            atr.append(None)
        elif i == period - 1:
            # First ATR is average of first period TR values
            atr.append(sum(tr[:period]) / period)
        else:
            # Wilder's smoothing method
            atr_val = (atr[i-1] * (period - 1) + tr[i]) / period
            atr.append(atr_val)
    
    return atr

def calculate_percentile(data: List[float], value: float) -> float:
    """
    Calculate percentile rank of a value in a dataset
    
    Args:
        data: List of values
        value: Value to find percentile for
        
    Returns:
        Percentile rank (0-100)
    """
    if not data:
        return 50.0
    
    # Count values less than or equal to the given value
    count = sum(1 for x in data if x <= value)
    
    # Calculate percentile
    percentile = (count / len(data)) * 100
    
    return percentile

def calculate_z_score(data: List[float], value: float) -> float:
    """
    Calculate Z-score of a value relative to a dataset
    
    Args:
        data: List of values
        value: Value to calculate Z-score for
        
    Returns:
        Z-score
    """
    if not data or len(data) < 2:
        return 0.0
    
    mean = statistics.mean(data)
    std_dev = statistics.stdev(data) if len(data) > 1 else 0
    
    if std_dev == 0:
        return 0.0
    
    return (value - mean) / std_dev

def calculate_correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate correlation coefficient between two datasets
    
    Args:
        x: First dataset
        y: Second dataset
        
    Returns:
        Correlation coefficient
    """
    if len(x) != len(y):
        raise ValueError("Datasets must have the same length")
    
    if len(x) < 2:
        return 0.0
    
    return np.corrcoef(x, y)[0, 1]

def calculate_regression(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Calculate linear regression parameters
    
    Args:
        x: Independent variable
        y: Dependent variable
        
    Returns:
        Dictionary with slope, intercept, and R-squared
    """
    if len(x) != len(y):
        raise ValueError("Datasets must have the same length")
    
    if len(x) < 2:
        return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0}
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err
    }

def calculate_volatility(data: List[float], annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns)
    
    Args:
        data: List of prices
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility as decimal
    """
    if len(data) < 2:
        return 0.0
    
    # Calculate returns
    returns = []
    for i in range(1, len(data)):
        if data[i-1] != 0:
            ret = (data[i] - data[i-1]) / data[i-1]
            returns.append(ret)
    
    if not returns:
        return 0.0
    
    # Calculate standard deviation
    volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(252)  # Trading days in a year
    
    return volatility

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.05, annualize: bool = True) -> float:
    """
    Calculate Sharpe Ratio
    
    Args:
        returns: List of returns
        risk_free_rate: Annual risk-free rate
        annualize: Whether the returns are annualized
        
    Returns:
        Sharpe Ratio
    """
    if not returns:
        return 0.0
    
    # Calculate excess returns
    if annualize:
        daily_rf = risk_free_rate / 252
    else:
        daily_rf = risk_free_rate
    
    excess_returns = [r - daily_rf for r in returns]
    
    # Calculate mean and standard deviation
    mean_excess = statistics.mean(excess_returns) if excess_returns else 0.0
    std_excess = statistics.stdev(excess_returns) if len(excess_returns) > 1 else 0.0
    
    if std_excess == 0:
        return 0.0
    
    # Calculate Sharpe Ratio
    if annualize:
        sharpe = (mean_excess * 252) / (std_excess * np.sqrt(252))
    else:
        sharpe = mean_excess / std_excess
    
    return sharpe

def calculate_max_drawdown(prices: List[float]) -> Dict[str, float]:
    """
    Calculate maximum drawdown
    
    Args:
        prices: List of prices
        
    Returns:
        Dictionary with max drawdown and related metrics
    """
    if len(prices) < 2:
        return {'max_drawdown': 0.0, 'drawdown_period': 0, 'recovery_period': 0}
    
    # Calculate cumulative returns
    cumulative = []
    peak = prices[0]
    max_dd = 0.0
    dd_start = 0
    dd_end = 0
    recovery = 0
    
    for i, price in enumerate(prices):
        if price > peak:
            peak = price
        
        drawdown = (peak - price) / peak if peak > 0 else 0.0
        
        if drawdown > max_dd:
            max_dd = drawdown
            dd_end = i
        
        cumulative.append(drawdown)
    
    # Find drawdown start (peak before drawdown)
    for i in range(dd_end, -1, -1):
        if prices[i] >= peak:
            dd_start = i
            break
    
    # Calculate recovery period
    recovery_price = prices[dd_end]
    for i in range(dd_end, len(prices)):
        if prices[i] >= peak:
            recovery = i - dd_end
            break
    
    return {
        'max_drawdown': max_dd,
        'drawdown_period': dd_end - dd_start,
        'recovery_period': recovery,
        'drawdown_start': dd_start,
        'drawdown_end': dd_end
    }

def calculate_value_at_risk(returns: List[float], confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk
    
    Args:
        returns: List of returns
        confidence_level: Confidence level (0.95 for 95%)
        
    Returns:
        VaR as decimal
    """
    if not returns:
        return 0.0
    
    # Sort returns
    sorted_returns = sorted(returns)
    
    # Calculate index for VaR
    index = int((1 - confidence_level) * len(sorted_returns))
    
    if index >= len(sorted_returns):
        index = len(sorted_returns) - 1
    
    var = sorted_returns[index]
    
    # VaR is negative for losses
    return abs(var) if var < 0 else 0.0

def calculate_expected_shortfall(returns: List[float], confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR)
    
    Args:
        returns: List of returns
        confidence_level: Confidence level
        
    Returns:
        Expected Shortfall as decimal
    """
    if not returns:
        return 0.0
    
    # Calculate VaR first
    var = calculate_value_at_risk(returns, confidence_level)
    
    # Find returns worse than VaR
    bad_returns = [r for r in returns if r <= -var]
    
    if not bad_returns:
        return var
    
    # Calculate average of bad returns
    es = statistics.mean(bad_returns)
    
    return abs(es) if es < 0 else var

def calculate_skewness(data: List[float]) -> float:
    """
    Calculate skewness of a dataset
    
    Args:
        data: List of values
        
    Returns:
        Skewness
    """
    if len(data) < 3:
        return 0.0
    
    return stats.skew(data)

def calculate_kurtosis(data: List[float]) -> float:
    """
    Calculate kurtosis of a dataset
    
    Args:
        data: List of values
        
    Returns:
        Kurtosis
    """
    if len(data) < 4:
        return 0.0
    
    return stats.kurtosis(data)

def calculate_confidence_interval(data: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval
    
    Args:
        data: List of values
        confidence_level: Confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) < 2:
        return (0.0, 0.0)
    
    mean = statistics.mean(data)
    std_err = stats.sem(data)
    
    # Calculate critical value
    if len(data) < 30:
        # Use t-distribution for small samples
        df = len(data) - 1
        critical_value = stats.t.ppf((1 + confidence_level) / 2, df)
    else:
        # Use normal distribution for large samples
        critical_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    margin = critical_value * std_err
    
    return (mean - margin, mean + margin)

def normalize_data(data: List[float]) -> List[float]:
    """
    Normalize data to range [0, 1]
    
    Args:
        data: List of values
        
    Returns:
        Normalized data
    """
    if not data:
        return []
    
    min_val = min(data)
    max_val = max(data)
    
    if max_val == min_val:
        return [0.5] * len(data)
    
    return [(x - min_val) / (max_val - min_val) for x in data]

def standardize_data(data: List[float]) -> List[float]:
    """
    Standardize data to have mean 0 and standard deviation 1
    
    Args:
        data: List of values
        
    Returns:
        Standardized data
    """
    if len(data) < 2:
        return [0.0] * len(data)
    
    mean = statistics.mean(data)
    std = statistics.stdev(data) if len(data) > 1 else 1.0
    
    if std == 0:
        return [0.0] * len(data)
    
    return [(x - mean) / std for x in data]

def calculate_rolling_statistic(data: List[float], window: int, statistic: str = 'mean') -> List[float]:
    """
    Calculate rolling statistic
    
    Args:
        data: List of values
        window: Rolling window size
        statistic: Statistic to calculate ('mean', 'std', 'min', 'max', 'median')
        
    Returns:
        List of rolling statistics
    """
    if len(data) < window:
        return [None] * len(data)
    
    result = []
    
    for i in range(len(data)):
        if i < window - 1:
            result.append(None)
        else:
            window_data = data[i-window+1:i+1]
            
            if statistic == 'mean':
                value = statistics.mean(window_data)
            elif statistic == 'std':
                value = statistics.stdev(window_data) if len(window_data) > 1 else 0.0
            elif statistic == 'min':
                value = min(window_data)
            elif statistic == 'max':
                value = max(window_data)
            elif statistic == 'median':
                value = statistics.median(window_data)
            else:
                raise ValueError(f"Unknown statistic: {statistic}")
            
            result.append(value)
    
    return result

def detect_outliers(data: List[float], method: str = 'zscore', threshold: float = 3.0) -> List[bool]:
    """
    Detect outliers in a dataset
    
    Args:
        data: List of values
        method: Detection method ('zscore' or 'iqr')
        threshold: Detection threshold
        
    Returns:
        List of booleans indicating outliers
    """
    if len(data) < 2:
        return [False] * len(data)
    
    outliers = []
    
    if method == 'zscore':
        # Z-score method
        mean = statistics.mean(data)
        std = statistics.stdev(data) if len(data) > 1 else 1.0
        
        if std == 0:
            return [False] * len(data)
        
        for value in data:
            zscore = abs((value - mean) / std)
            outliers.append(zscore > threshold)
    
    elif method == 'iqr':
        # IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return [False] * len(data)
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        for value in data:
            outliers.append(value < lower_bound or value > upper_bound)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers

def calculate_entropy(data: List[float], bins: int = 10) -> float:
    """
    Calculate entropy of a dataset
    
    Args:
        data: List of values
        bins: Number of bins for histogram
        
    Returns:
        Entropy
    """
    if not data:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(data, bins=bins, density=True)
    
    # Remove zero probabilities
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy

def calculate_mutual_information(x: List[float], y: List[float], bins: int = 10) -> float:
    """
    Calculate mutual information between two datasets
    
    Args:
        x: First dataset
        y: Second dataset
        bins: Number of bins for histogram
        
    Returns:
        Mutual information
    """
    if len(x) != len(y):
        raise ValueError("Datasets must have the same length")
    
    if len(x) < 2:
        return 0.0
    
    # Create 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    
    # Calculate marginal distributions
    hist_x, _ = np.histogram(x, bins=x_edges, density=True)
    hist_y, _ = np.histogram(y, bins=y_edges, density=True)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_2d[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_x[i] * hist_y[j]))
    
    return mi