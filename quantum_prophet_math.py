import numpy as np
import pandas as pd
from scipy.stats import entropy

def calculate_hurst(ts):
    """Calculates Hurst Exponent. H > 0.5 = Trending, H < 0.5 = Mean Reverting."""
    try:
        if len(ts) < 20: return 0.5
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

def calculate_lyapunov_exponent(ts):
    """
    Measures Chaos. 
    Positive = System is chaotic and unpredictable.
    Negative = System is stable and predictable (The Code is operative).
    """
    try:
        if len(ts) < 20: return 0
        diffs = np.abs(np.diff(ts))
        if np.all(diffs < 1e-9): return -0.5 # Perfectly stable
        
        ratios = diffs / (np.roll(diffs, 1) + 1e-9)
        # Filter for valid logarithmic ratios to prevent RuntimeWarnings
        valid_ratios = ratios[(ratios > 1e-6) & (ratios < 1e6)]
        
        if len(valid_ratios) < 5: return 0
        lyapunov = np.mean(np.log(valid_ratios))
        return float(lyapunov)
    except: return 0

def calculate_hurst_velocity(ts):
    """
    Measures dH/dt (Rate of change of the Hurst Exponent).
    If price is rising but Hurst is falling rapidly, it is a 'Momentum Vacuum' (EXHAUSTION).
    """
    try:
        if len(ts) < 60: return 0
        chunk_size = min(30, len(ts) // 2)
        h1 = calculate_hurst(ts[-chunk_size*2:-chunk_size])
        h2 = calculate_hurst(ts[-chunk_size:])
        return float(h2 - h1)
    except: return 0

def calculate_volatility_adjusted_entropy(ts):
    """
    Scales Entropy by Volatility. 
    If Volatility is low but Entropy is high, a 'Hidden Spot' of accumulation is detected.
    """
    try:
        ent = calculate_shannon_entropy(ts)
        vol = np.std(np.diff(ts))
        return ent / (vol + 1e-9)
    except: return 0

def calculate_entropy_surge(ts):
    """
    Measures the rate of change in Shannon Entropy.
    A surge in entropy during a price move indicates a 'Fake Out'.
    """
    try:
        if len(ts) < 40: return 0
        h1 = calculate_shannon_entropy(ts[:-10])
        h2 = calculate_shannon_entropy(ts[-10:])
        return h2 - h1
    except: return 0

def calculate_z_score_deviation(ts):
    """
    Measures how far price has deviated from the OU Fair Value.
    Z > 2.0 = Overstretched (High Reversal Probability).
    """
    try:
        mu = calculate_ornstein_uhlenbeck(ts)
        std = np.std(ts)
        return (ts[-1] - mu) / (std + 1e-9)
    except: return 0

def calculate_shannon_entropy(ts):
    """
    Measures Uncertainty.
    Low = High information density (High conviction zone).
    High = Noise (Randomness).
    """
    try:
        # Bin the price changes to create a probability distribution
        diffs = np.diff(ts)
        if np.all(diffs < 1e-9): return 0
        
        hist, _ = np.histogram(diffs, bins=10)
        prob_dist = hist / (hist.sum() + 1e-9)
        return float(entropy(prob_dist))
    except: return 0

def calculate_ornstein_uhlenbeck(ts):
    """
    Mean Reversion Drift (The 'Gravity' of fair value).
    Returns 'mu' (The long-term mean price the market is forced to return to).
    """
    try:
        if len(ts) < 20: return ts[-1]
        x = ts[:-1]
        y = ts[1:]
        # Simple linear regression: y = a*x + b
        poly = np.polyfit(x, y, 1)
        a, b = poly[0], poly[1]
        # mu = b / (1 - a)
        mu = b / (1 - a + 1e-9)
        return mu
    except: return ts[-1]

def get_quantum_signal(df):
    """
    The 'Grand Code' Signal.
    Combines Entropy, Chaos, and Drift.
    """
    # Phase 88: Case-Insensitive Column Handling
    close_col = 'close' if 'close' in df.columns else 'Close'
    if close_col not in df.columns:
        return "WAIT", 0, {}
        
    prices = df[close_col].values
    current_price = prices[-1]
    
    lyapunov = calculate_lyapunov_exponent(prices)
    shannon = calculate_shannon_entropy(prices)
    fair_value = calculate_ornstein_uhlenbeck(prices)
    
    # Logic: 
    # 1. Stability Check: If Lyapunov is negative, the trend is 'Locked' (Predictable).
    # 2. Conviction Check: If Shannon is low, the signal is pure.
    # 3. Direction: Gap between Current and Fair Value.
    
    predictability = 1.0 / (1.0 + np.exp(lyapunov)) # Sigmoid to map predictability [0, 1]
    drift_gap = (fair_value - current_price) / (current_price + 1e-9)
    entropy_surge = calculate_entropy_surge(prices)
    z_score = calculate_z_score_deviation(prices)
    h_velocity = calculate_hurst_velocity(prices)
    vol_ent = calculate_volatility_adjusted_entropy(prices)
    
    # THE CODE: CRACKING EXHAUSTION
    # 1. THE MOMENTUM VACUUM: If H is falling while price is moving -> REVERSE.
    # 2. THE CHAOS PEAK: If Entropy is high and Z-Score is extreme -> REVERSE.
    
    score = 0
    signal = "WAIT"
    
    # Detection of Trend Exhaustion (The 'Code' for the top/bottom)
    if h_velocity < -0.15 and abs(z_score) > 1.5:
        # Trend is dying + price is stretched = REVERSAL
        signal = "SELL" if z_score > 0 else "BUY"
        score = 95
    elif entropy_surge > 0.4 and abs(z_score) > 1.0:
        # Chaos is rising during a move = FAKE OUT
        signal = "SELL" if (prices[-1] > prices[-5]) else "BUY"
        score = 85
    else:
        # Standard follow logic
        if lyapunov < 0 and shannon < 1.4:
            if abs(drift_gap) > 0.015:
                signal = "BUY" if drift_gap > 0 else "SELL"
                score = 60
            
    return signal, score, {"lyapunov": lyapunov, "shannon": shannon, "fair_value": fair_value, "gap": drift_gap, "h_vel": h_velocity, "z_score": z_score}

if __name__ == "__main__":
    # Test on synthetic stable vs random
    stable = np.sin(np.linspace(0, 10, 100)) + 10
    random = np.cumsum(np.random.randn(100)) + 10
    
    s_sig, s_score, s_m = get_quantum_signal(pd.DataFrame({'close': stable}))
    r_sig, r_score, r_m = get_quantum_signal(pd.DataFrame({'close': random}))
    
    print(f"Stable Output: {s_sig}, Score: {s_score}, Metrics: {s_m}")
    print(f"Random Output: {r_sig}, Score: {r_score}, Metrics: {r_m}")
