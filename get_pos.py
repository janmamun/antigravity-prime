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
        print(f"Symbol: {p['symbol']}, Amt: {p['positionAmt']}, Entry: {p.get('entryPrice', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")
