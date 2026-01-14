import ccxt
import os
from dotenv import load_dotenv
import json

load_dotenv()

def check_raw_positions():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    proxy = os.getenv("BINANCE_PROXY")
    
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    if proxy:
        exchange.proxies = {'http': proxy, 'https': proxy}
        
    # Fetch all active positions with non-zero balance
    positions = exchange.fetch_positions()
    active = [p for p in positions if float(p.get('contracts', 0)) != 0]
    
    print("--- ACTIVE POSITIONS (RAW) ---")
    if not active:
        print("No active positions.")
    else:
        for p in active:
            # CCXT Unified format
            symbol = p['symbol']
            side = p['side']
            contracts = p['contracts']
            entry = p['entryPrice']
            pnl = p['unrealizedPnl']
            print(f"Symbol: {symbol} | Side: {side} | Qty: {contracts} | Entry: {entry} | PnL: {pnl} USDT")

if __name__ == "__main__":
    check_raw_positions()
