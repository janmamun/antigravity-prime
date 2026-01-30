import ccxt, os, json
from dotenv import load_dotenv
load_dotenv()
try:
    ex = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'options': {'defaultType': 'future'}
    })
    bal = ex.fetch_balance()
    pos = [p for p in bal.get('info', {}).get('positions', []) if float(p.get('positionAmt', 0)) != 0]
    for p in pos:
        symbol = p['symbol']
        raw_amt = float(p['positionAmt'])
        abs_amt = abs(raw_amt)
        side = 'sell' if raw_amt > 0 else 'buy'
        
        # Only terminate Alts (Not BTC/ETH)
        if 'BTC' not in symbol and 'ETH' not in symbol:
            print(f"🚨 [CLEAN SWEEP] Terminating {symbol} | Amt: {raw_amt}")
            ex.create_market_order(symbol, side, abs_amt)
            # Cancel all open orders for this symbol
            ex.cancel_all_orders(symbol)
    print("✅ Clean Sweep Complete.")
except Exception as e:
    print(f"Error: {e}")
