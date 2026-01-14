import pandas as pd
import numpy as np
from trading_bot_v17 import UltimateV17Bot

def test_v10_logic():
    bot = UltimateV17Bot()
    
    # Mock data for analysis
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='H'),
        'Open': np.random.uniform(100, 110, 200),
        'High': np.random.uniform(110, 120, 200),
        'Low': np.random.uniform(90, 100, 200),
        'Close': np.random.uniform(100, 110, 200),
        'Volume': np.random.uniform(1000, 5000, 200)
    }
    df = pd.DataFrame(data)
    
    print("--- TESTING V10 NEURAL SHADOW & SPOOF DEFENSE ---")
    symbol = "BTC/USDT"
    
    # 1. Test Persistence (Spoof Defense)
    # Fill persistence store with "flickering" data
    bot.persistence_store[symbol] = {
        'bids': [100, 20, 100, 20, 100],
        'asks': [100, 100, 100, 100, 100],
        'times': [1, 2, 3, 4, 5]
    }
    
    analysis = bot.analyze_snapshot(df, symbol)
    
    print(f"Signal: {analysis['signal']}")
    print(f"Score: {analysis['score']}")
    print(f"Reasons: {analysis['reasons']}")
    
    spoof_detected = any("INSTITUTIONAL INTEGRITY" in r for r in analysis['reasons'])
    print(f"Spoof Detected: {spoof_detected}")
    
    # 2. Test Rebalancer
    mock_pos = [
        {'symbol': 'WINNER/USDT', 'unrealizedPnl': 5.0, 'notional': 100.0},
        {'symbol': 'LOSER/USDT', 'unrealizedPnl': -0.1, 'notional': 100.0}
    ]
    rebal = bot.rebalance_active_alpha(mock_pos)
    print(f"Rebalance Actions: {rebal}")

if __name__ == "__main__":
    test_v10_logic()
