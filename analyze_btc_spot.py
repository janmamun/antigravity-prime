import ccxt
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

def analyze_btc_history():
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    proxy = os.getenv("BINANCE_PROXY")
    
    config = {
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    }
    
    if proxy:
        config['proxies'] = {'http': proxy, 'https': proxy}
        
    exchange = ccxt.binance(config)
    symbol = 'BTC/USDT'
    
    try:
        print(f"Fetching trade history for {symbol}...")
        # Get trades for the last 30 days
        since = exchange.parse8601((datetime.now() - timedelta(days=30)).isoformat())
        all_trades = []
        
        while since < exchange.milliseconds():
            trades = exchange.fetch_my_trades(symbol, since)
            if not trades:
                break
            all_trades.extend(trades)
            since = trades[-1]['timestamp'] + 1
            if len(trades) < 500: # Assuming 500 is the limit
                break
        
        if not all_trades:
            print("No trades found.")
            return

        df = pd.DataFrame(all_trades)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Simple analysis
        buys = df[df['side'] == 'buy']
        sells = df[df['side'] == 'sell']
        
        print(f"\nTotal Trades: {len(df)}")
        print(f"Total Buys: {len(buys)}")
        print(f"Total Sells: {len(sells)}")
        
        # Calculate Total Cost and Revenue
        total_cost = (buys['amount'] * buys['price']).sum()
        total_revenue = (sells['amount'] * sells['price']).sum()
        
        buy_avg = buys['price'].mean()
        sell_avg = sells['price'].mean()
        
        print(f"Average Buy Price: {buy_avg:.2f}")
        print(f"Average Sell Price: {sell_avg:.2f}")
        print(f"Total USDT Spent: {total_cost:.2f}")
        print(f"Total USDT Received: {total_revenue:.2f}")
        
        # Save to CSV for deeper analysis
        df.to_csv('btc_spot_history.csv', index=False)
        print("\nFull history saved to btc_spot_history.csv")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_btc_history()
