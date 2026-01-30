import os, ccxt, json
from dotenv import load_dotenv
load_dotenv()

def protect_axs():
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'options': {'defaultType': 'future', 'warnOnFetchOpenOrdersWithoutSymbol': False}
    })
    
    symbol = 'AXS/USDT'
    qty = 184.0
    sl_price = 2.302
    tp_price = 2.407
    
    print(f"Attempting to place SL for {symbol} at {sl_price}...")
    try:
        sl_res = exchange.create_order(symbol, 'STOP_MARKET', 'sell', qty, params={'stopPrice': sl_price, 'reduceOnly': True})
        print(f"SL Order Result: {json.dumps(sl_res, indent=2)}")
    except Exception as e:
        print(f"SL Order Failed: {e}")

    print(f"Attempting to place TP for {symbol} at {tp_price}...")
    try:
        tp_res = exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', qty, params={'stopPrice': tp_price, 'reduceOnly': True})
        print(f"TP Order Result: {json.dumps(tp_res, indent=2)}")
    except Exception as e:
        print(f"TP Order Failed: {e}")

    # Fetch recent trades for the account
    print("\nRecent Account Trades:")
    trades = exchange.fetch_my_trades(limit=10)
    for t in trades:
        print(f"Trade: {t['symbol']} | {t['side']} | PnL: {t['info'].get('realizedPnl')} | QuoteQty: {t['info'].get('quoteQty')}")

if __name__ == '__main__':
    protect_axs()
