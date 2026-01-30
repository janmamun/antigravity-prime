import ccxt
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import json

def continuous_monitor():
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    proxy = os.getenv("BINANCE_PROXY")
    
    config = {
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    }
    
    if proxy:
        config['proxies'] = {'http': proxy, 'https': proxy}
        
    exchange = ccxt.binance(config)
    log_file = "live_monitoring.log"
    
    print(f"Starting 1-minute live monitoring. Logging to {log_file}...")
    
    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            balance = exchange.fetch_balance()
            usdt_total = balance.get('total', {}).get('USDT', 0)
            
            positions = exchange.fapiPrivateV2GetAccount().get('positions', [])
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            
            log_entry = f"[{now}] Total USDT: {usdt_total} | Active Positions: {len(active_positions)}\n"
            for p in active_positions:
                log_entry += f"  - {p['symbol']}: {p['positionAmt']} @ {p['entryPrice']} (PnL: {p['unrealizedProfit']})\n"
            
            with open(log_file, "a") as f:
                f.write(log_entry)
            
            print(log_entry.strip())
            
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now()}] Error: {e}\n")
            print(f"Error: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    continuous_monitor()
