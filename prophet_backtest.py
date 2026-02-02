import ccxt
import pandas as pd
import numpy as np
from geometric_prophet import calculate_hurst, calculate_fourier_prediction, calculate_fractal_dimension, get_geometric_signal, check_cross_correlation
import time

def backtest_prophet_advanced(symbol='BTC/USDT'):
    print(f"Prophet Research: Advanced Backtesting on {symbol}...")
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    results = []
    window = 100
    for i in range(window, len(df) - 1):
        test_df = df.iloc[i-window:i]
        actual_next_move = np.sign(df.iloc[i+1]['close'] - df.iloc[i]['close'])
        
        signal, score, metrics = get_geometric_signal(test_df)
        
        if signal != "WAIT":
            pred = 1 if signal == "BUY" else -1
            results.append({
                'actual': actual_next_move,
                'pred': pred,
                'score': score,
                'h_val': metrics['hurst']
            })
    
    if not results:
        print(f"No math conviction signals found for {symbol} with current thresholds.")
        return 0, 0
        
    res_df = pd.DataFrame(results)
    acc = (res_df['pred'] == res_df['actual']).mean()
    
    print(f"RESULTS FOR {symbol}:")
    print(f" - Conviction Math Accuracy: {acc*100:.2f}% ({len(results)} signals)")
    print(f" - Avg Hurst: {res_df['h_val'].mean():.4f}")
    
    return acc, len(results)

def analyze_cross_correlation(sym1='BTC/USDT', sym2='PEPE/USDT'):
    print(f"Prophet Research: Analyzing Mathematical Correlation between {sym1} and {sym2}...")
    ex = ccxt.binance()
    ohlcv1 = ex.fetch_ohlcv(sym1, timeframe='1h', limit=100)
    ohlcv2 = ex.fetch_ohlcv(sym2, timeframe='1h', limit=100)
    
    p1 = [x[4] for x in ohlcv1]
    p2 = [x[4] for x in ohlcv2]
    
    corr = check_cross_correlation(p1, p2)
    print(f"Cross-Correlation (1h): {corr:.4f}")
    
    h1 = calculate_hurst(p1)
    h2 = calculate_hurst(p2)
    print(f"Hurst Sync: {sym1}={h1:.4f}, {sym2}={h2:.4f}")

if __name__ == "__main__":
    backtest_prophet_advanced('BTC/USDT')
    backtest_prophet_advanced('DOGE/USDT')
    backtest_prophet_advanced('PEPE/USDT')
    analyze_cross_correlation('BTC/USDT', 'PEPE/USDT')
    analyze_cross_correlation('BTC/USDT', 'DOGE/USDT')
