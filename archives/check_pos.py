import ccxt
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def check_live_positions():
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
    balance = exchange.fetch_balance()
    positions = balance['info']['positions']
    
    active = [p for p in positions if float(p['positionAmt']) != 0]
    
    print("--- ACTIVE POSITIONS ---")
    if not active:
        print("No active positions.")
    else:
        for p in active:
            side = "LONG" if float(p['positionAmt']) > 0 else "SHORT"
            qty = p['positionAmt']
            # Unified keys in CCXT for Binance are 'entryPrice', 'unrealizedProfit'
            # But let's check the 'info' sub-dict which is the raw Binance response
            info = p['info']
            pnl = p.get('unrealizedProfit', info.get('unrealizedProfit'))
            entry = p.get('entryPrice', info.get('entryPrice'))
            
            print(f"Symbol: {p['symbol']} | Side: {side} | Qty: {qty} | Entry: {entry} | PnL: {pnl} USDT")

if __name__ == "__main__":
    check_live_positions()
