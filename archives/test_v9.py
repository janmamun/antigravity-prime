import pandas as pd
from trading_bot_v17 import UltimateV17Bot
import ccxt

def test_v9_logic():
    bot = UltimateV17Bot()
    exch = ccxt.binance({'options': {'defaultType': 'future'}})
    
    print("--- TESTING SOVEREIGN V9.0 ALPHA STRIKE ---")
    symbol = "BTC/USDT"
    ohlcv = exch.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    analysis = bot.analyze_snapshot(df, symbol)
    
    print(f"Symbol: {analysis['symbol']}")
    print(f"Signal: {analysis['signal']}")
    print(f"Score: {analysis['score']}")
    print(f"Regime: {analysis['regime']}")
    print("Reasons:")
    for r in analysis.get('reasons', []):
        print(f" - {r}")

if __name__ == "__main__":
    test_v9_logic()
