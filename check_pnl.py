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
        amt = float(p['positionAmt'])
        entry = float(p.get('entryPrice', 0))
        # Fetch current ticker
        ticker = ex.fetch_ticker(symbol)
        last = float(ticker['last'])
        # Simple PnL calc (ignoring fees for quick audit)
        pnl = (last - entry) * amt if amt > 0 else (entry - last) * abs(amt)
        pnl_pct = (pnl / (entry * abs(amt))) * 100 if entry != 0 else 0
        print(f"Symbol: {symbol}, Side: {'LONG' if amt > 0 else 'SHORT'}, PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
except Exception as e:
    print(f"Error: {e}")
