import ccxt
import pandas as pd
import numpy as np
from geometric_prophet import calculate_hurst, calculate_fourier_prediction, get_geometric_signal
import time

def backtest_prophet_advanced(symbol='BTC/USDT', timeframe='1h'):
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=500)
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
                'h_val': metrics['hurst'],
                'f_val': metrics['fourier']
            })
    
    if not results:
        print(f"| {symbol:10} | N/A (Low Conviction) | N/A |")
        return
        
    res_df = pd.DataFrame(results)
    acc = (res_df['pred'] == res_df['actual']).mean()
    avg_h = res_df['h_val'].mean()
    
    print(f"| {symbol:10} | {acc*100:6.2f}% | {len(results):4} | {avg_h:7.4f} |")

if __name__ == "__main__":
    print("-" * 55)
    print(f"| {'Symbol':10} | {'Accuracy':8} | {'Signals':8} | {'Avg Hurst':10} |")
    print("-" * 55)
    coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'PEPE/USDT', 'SHIB/USDT', 'BONK/USDT', 'WIF/USDT', 'XRP/USDT', 'ADA/USDT']
    for coin in coins:
        try:
            backtest_prophet_advanced(coin)
        except:
            pass
    print("-" * 55)
