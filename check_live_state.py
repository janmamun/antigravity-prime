import ccxt
import os
from dotenv import load_dotenv
import json

def check_live_state():
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
    
    try:
        print("--- BINANCE LIVE STATE ---")
        balance = exchange.fetch_balance()
        usdt_total = balance.get('total', {}).get('USDT', 0)
        usdt_free = balance.get('free', {}).get('USDT', 0)
        print(f"USDT Total: {usdt_total}")
        print(f"USDT Free: {usdt_free}")
        
        positions = exchange.fapiPrivateV2GetAccount().get('positions', [])
        active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        
        print(f"\nActive Positions: {len(active_positions)}")
        for p in active_positions:
            print(f"Symbol: {p['symbol']}, Size: {p['positionAmt']}, Entry: {p['entryPrice']}, PnL: {p['unrealizedProfit']}")
            
        open_orders = exchange.fetch_open_orders()
        print(f"\nOpen Orders: {len(open_orders)}")
        for o in open_orders:
            print(f"Symbol: {o['symbol']}, Side: {o['side']}, Type: {o['type']}, Price: {o['price']}, StopPrice: {o.get('stopPrice')}")
            
    except Exception as e:
        print(f"Error fetching live state: {e}")

if __name__ == "__main__":
    check_live_state()
