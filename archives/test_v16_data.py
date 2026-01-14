
import ccxt
import time

def test_institutional_data():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    symbol = 'BTC/USDT:USDT'
    print(f"--- FETCHING DATA FOR {symbol} ---")
    
    try:
        # 1. Open Interest History
        oi_history = exchange.fetch_open_interest_history(symbol, timeframe='5m', limit=5)
        print("OI History (Last 5 periods):")
        for h in oi_history:
            print(f"  Time: {h['datetime']} | OI: {h['openInterestAmount']}")
        
        # 2. Large Trades (Whales)
        trades = exchange.fetch_trades(symbol, limit=50)
        large_trades = [t for t in trades if float(t['amount']) * float(t['price']) > 50000]
        print(f"Found {len(large_trades)} whale trades (> $50k) in last 50 trades.")
        for wt in large_trades[:3]:
            print(f"  Whale: {wt['side'].upper()} {wt['amount']} @ {wt['price']}")
            
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    test_institutional_data()
