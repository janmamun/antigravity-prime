import ccxt, os, json
from dotenv import load_dotenv
load_dotenv()
ex = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET'),
    'options': {'defaultType': 'future'}
})
bal = ex.fetch_balance()
positions = [p for p in bal.get('info', {}).get('positions', []) if float(p.get('positionAmt', 0)) != 0]
for p in positions:
    print(f"Symbol: {p['symbol']}, Amt: {p['positionAmt']}, Entry: {p['entryPrice']}")
