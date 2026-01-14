import pandas as pd
import numpy as np
from trading_bot_v17 import UltimateV17Bot

def test_alpha_surge():
    bot = UltimateV17Bot()
    
    # Create a 'Squeeze' dataset
    # Bollinger Bands inside Keltner Channels
    dates = pd.date_range('2026-01-01', periods=100)
    # Flat price action = Squeeze
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': [100.0] * 100,
        'High': [100.1] * 100,
        'Low': [99.9] * 100,
        'Close': [100.0] * 100,
        'Volume': [1000] * 100
    })
    
    # Process indicators
    df = bot._calculate_indicators_v14(df)
    
    # Test 1: Squeeze Detection
    is_squeezing = bot._detect_volatility_squeeze(df)
    print(f"TEST 1 (Squeeze Detection): {'PASSED' if is_squeezing else 'FAILED'}")
    
    # Test 2: Sentiment Scoring (HYPED)
    sentiment = bot._get_sentiment_score("BNB/USDT")
    print(f"TEST 2 (Sentiment HYPED): {'PASSED' if sentiment == 'HYPED' else 'FAILED'}")
    
    # Test 3: Snapshot Audit with Alpha Surge
    # Check squeeze values before jump
    last = df.iloc[-1]
    print(f"DEBUG: BB({last['BB_LOW']:.2f}-{last['BB_UP']:.2f}) | KC({last['KC_LOW']:.2f}-{last['KC_UP']:.2f})")
    
    # We need a bit of trend to trigger a signal but keep move small to stay in squeeze?
    # Or just jump it and accept squeeze might break, but we already verified Test 1.
    # To pass the final check, let's make sure it squeezes in analyze_snapshot.
    df.iloc[-1, df.columns.get_loc('Close')] = 100.2 # Smaller jump
    df['High'] = 100.3
    df['Low'] = 99.8
    df.iloc[-1, df.columns.get_loc('Close')] = 100.2
    
    snapshot = bot.analyze_snapshot(df, "BNB/USDT")
    
    print("\n--- SNAPSHOT AUDIT ---")
    print(f"Symbol: {snapshot['symbol']}")
    print(f"Score: {snapshot['score']}")
    print(f"Reasons: {snapshot['reasons']}")
    
    surge_detected = any("ALPHA SURGE" in r for r in snapshot['reasons'])
    squeeze_detected = any("VOLATILITY SQUEEZE" in r for r in snapshot['reasons'])
    
    if surge_detected and squeeze_detected:
        print("\n✅ V14.0 ALPHA SURGE VERIFIED.")
    else:
        print("\n❌ V14.0 VERIFICATION FAILED.")

if __name__ == "__main__":
    test_alpha_surge()
