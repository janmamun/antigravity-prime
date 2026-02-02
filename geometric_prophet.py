import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

def calculate_hurst(ts):
    """
    Calculate Hurst Exponent to determine trend persistence.
    """
    try:
        if len(ts) < 30: return 0.5
        lags = range(2, 20)
        tau = []
        for lag in lags:
            diff = np.subtract(ts[lag:], ts[:-lag])
            s = np.std(diff)
            tau.append(s if s > 1e-9 else 1e-9)
            
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0] * 2.0)
    except:
        return 0.5

def calculate_fourier_prediction(ts):
    """
    Use Fast Fourier Transform to find dominant cycles and predict next move.
    """
    try:
        if len(ts) < 50: return 0
        yf = fft(ts)
        xf = fftfreq(len(ts), 1)
        
        # Filter noise - keep only dominant frequencies
        indices = np.argsort(np.abs(yf))[-8:] # Top 8 frequencies
        t_next = len(ts) + 1
        reconstructed = 0
        for i in indices:
            if i == 0: continue # Skip DC component
            amp = np.abs(yf[i]) / len(ts)
            phase = np.angle(yf[i])
            freq = xf[i]
            reconstructed += amp * np.cos(2 * np.pi * freq * t_next + phase)
        
        return np.sign(reconstructed)
    except:
        return 0

def calculate_fractal_dimension(ts):
    """
    Calculate Box-Counting Dimension (approximation).
    """
    try:
        if len(ts) < 20: return 1.5
        diffs = np.diff(ts)
        total_abs_diff = np.sum(np.abs(diffs))
        if total_abs_diff < 1e-9: return 1.0
        
        return 1.0 + (float(np.log(total_abs_diff)) / np.log(len(ts)))
    except:
        return 1.5

def get_geometric_signal(df):
    """
    Combines math indicators into a single prediction signal.
    """
    prices = df['close'].values
    if len(prices) < 50: return "WAIT", 0
    
    hurst = calculate_hurst(prices)
    fourier = calculate_fourier_prediction(prices)
    fractal = calculate_fractal_dimension(prices)
    
    # Logic: 
    # If Mean Reverting (H < 0.45) AND Fourier says Reverse -> STRONG FADE
    # If Trending (H > 0.55) AND Fourier says Follow -> STRONG TREND
    
    last_move = np.sign(prices[-1] - prices[-2])
    
    score = 0
    if fourier != 0 and last_move != 0:
        if hurst < 0.4: # Strong Mean Reversion
            if fourier != last_move:
                score = 65 # High conviction reversal
        elif hurst > 0.6: # Strong Trend
            if fourier == last_move:
                score = 75 # High conviction continuation
            
    signal = "WAIT"
    if score >= 60:
        signal = "BUY" if fourier > 0 else "SELL"
    
    return signal, score, {"hurst": hurst, "fourier": fourier, "fractal": fractal}

def check_cross_correlation(symbol_a_prices, symbol_b_prices):
    """
    Check if two charts are in sync mathematically.
    """
    if len(symbol_a_prices) != len(symbol_b_prices):
        min_len = min(len(symbol_a_prices), len(symbol_b_prices))
        symbol_a_prices = symbol_a_prices[-min_len:]
        symbol_b_prices = symbol_b_prices[-min_len:]
        
    correlation = np.corrcoef(symbol_a_prices, symbol_b_prices)[0,1]
    return correlation
