import numpy as np
import pandas as pd
import ccxt

def calculate_hurst(ts):
    """Calculates Hurst Exponent. H > 0.5 = Trending, H < 0.5 = Mean Reverting."""
    try:
        if len(ts) < 30: return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

def detect_hidden_voids(df):
    """Detects Liquidity Voids (Fair Value Gaps). These are 'Hidden Spots' likely to be filled."""
    voids = []
    # Loop through candles to find gaps between (n-1).high and (n+1).low
    for i in range(1, len(df) - 1):
        prev_h = df['high'].iloc[i-1]
        next_l = df['low'].iloc[i+1]
        
        # Bullish Void (Price moved up too fast)
        if next_l > prev_h:
            voids.append({'type': 'BULL_VOID', 'top': next_l, 'bottom': prev_h, 'index': i})
            
        # Bearish Void (Price moved down too fast)
        prev_l = df['low'].iloc[i-1]
        next_h = df['high'].iloc[i+1]
        if next_h < prev_l:
            voids.append({'type': 'BEAR_VOID', 'top': prev_l, 'bottom': next_h, 'index': i})
            
    return voids

def calculate_liquidity_gravity(df, current_price, window=50):
    """Identifies the 'Gravity Point' - the price level with the highest volume cluster."""
    recent_df = df.tail(window)
    # Binning price levels to find volume clusters
    bins = 20
    hist, edges = np.histogram(recent_df['close'], bins=bins, weights=recent_df['volume'])
    max_bin_idx = np.argmax(hist)
    gravity_price = (edges[max_bin_idx] + edges[max_bin_idx+1]) / 2
    
    # Gravitational Pull: F = Vol / distance^2
    dist = abs(current_price - gravity_price) / current_price
    if dist == 0: dist = 0.0001
    pull = sum(hist) / (dist ** 2)
    
    return gravity_price, pull

def get_meme_prophecy(df):
    """
    Combines Voids, Gravity, and Hurst into a predictive signal.
    """
    prices = df['close'].values
    current_price = prices[-1]
    
    hurst = calculate_hurst(prices)
    voids = detect_hidden_voids(df)
    gravity_price, gravity_pull = calculate_liquidity_gravity(df, current_price)
    
    # Check if we are near a "Hidden Spot" (Void)
    in_void = False
    for v in voids[-5:]: # Check last 5 voids
        if v['bottom'] <= current_price <= v['top']:
            in_void = True
            break
            
    # Formula: If Trending (H > 0.55) AND Price is above Gravity -> PUMP LIKELY
    # If in a Void -> VOLATILITY IMMINENT
    
    score = 0
    if hurst > 0.55: score += 30 # Trend Persistence
    if current_price > gravity_price: score += 20 # Above Gravity
    if in_void: score += 20 # In "Hidden Spot" (Volatility Zone)
    
    msg = f"Hurst: {hurst:.3f} | Gravity: {gravity_price:.4f} | InVoid: {in_void}"
    
    signal = "WAIT"
    if score >= 50:
        signal = "BUY" if current_price > gravity_price else "SELL"
        
    return signal, score, msg

if __name__ == "__main__":
    # Internal Test
    data = {'high': [10, 15, 30, 32, 35], 'low': [8, 12, 28, 30, 32], 'close': [9, 14, 29, 31, 34], 'volume': [100, 200, 500, 300, 400]}
    df = pd.DataFrame(data)
    v = detect_hidden_voids(df)
    print(f"Test Voids: {v}")
