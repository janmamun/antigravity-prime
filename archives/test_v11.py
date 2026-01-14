import pandas as pd
import numpy as np
from trading_bot_v17 import UltimateV17Bot

def test_v11_logic():
    bot = UltimateV17Bot()
    
    # 1. Test TP Grid Calculation
    entry = 900.0
    side = "BUY"
    atr = 10.0
    grid = bot.calculate_tp_grid(entry, side, atr)
    
    print("--- TESTING V11 TP GRID ---")
    for i, g in enumerate(grid):
        print(f"Tier {i+1}: Target ${g['target']:.2f} | Size: {g['size_pct']*100}%")
    
    assert grid[0]['target'] == 908.0 # entry + 0.8 * 10
    assert grid[2]['target'] == 930.0 # entry + 3.0 * 10
    
    # 2. Test Trailing SL
    # Current SL is 880, price is 910. Trailing should move SL up.
    new_sl = bot.trail_liquidity_sl("BNB/USDT", 910.0, 880.0, "BUY")
    print(f"Trailing SL Move: $880 -> ${new_sl:.2f}")
    assert new_sl > 880.0

    print("--- V11 SYSTEM LOGIC VERIFIED ---")

if __name__ == "__main__":
    test_v11_logic()
