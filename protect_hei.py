import os, ccxt, json
from dotenv import load_dotenv
load_dotenv()

def protect_hei():
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'options': {'defaultType': 'future', 'warnOnFetchOpenOrdersWithoutSymbol': False}
    })
    
    symbol = 'HEI/USDT'
    # Fetch current position to get exact qty
    pos = [p for p in exchange.fetch_positions() if p['symbol'] == 'HEI/USDT:USDT'][0]
    qty = abs(float(pos['contracts']))
    entry = float(pos['entryPrice'])
    
    sl_price = 0.1398 
    tp_price = 0.1550
    
    print(f"Position: {symbol} | Qty: {qty} | Entry: {entry}")
    print(f"Placing SL at {sl_price} and TP at {tp_price}...")
    
    try:
        sl_res = exchange.create_order(symbol, 'STOP_MARKET', 'sell', qty, params={'stopPrice': sl_price, 'reduceOnly': True})
        print(f"SL Order Result: {sl_res.get('id')}")
        tp_res = exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', qty, params={'stopPrice': tp_price, 'reduceOnly': True})
        print(f"TP Order Result: {tp_res.get('id')}")
    except Exception as e:
        print(f"Protection Error: {e}")

if __name__ == '__main__':
    protect_hei()
