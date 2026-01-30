import os, ccxt, json
from dotenv import load_dotenv
load_dotenv()

def verify():
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'options': {'defaultType': 'future', 'warnOnFetchOpenOrdersWithoutSymbol': False}
    })
    
    # 1. Check Balance
    bal = exchange.fetch_balance()
    print(f"Total Balance: {bal['total']['USDT']}")
    
    # 2. Check Positions
    pos = exchange.fetch_positions()
    active_pos = [p for p in pos if float(p.get('contracts', 0)) != 0]
    print(f"Active Positions: {[(p['symbol'], p['contracts'], p['unrealizedPnl']) for p in active_pos]}")
    
    # 3. Check All Open Orders
    orders = exchange.fetch_open_orders()
    print(f"Open Orders Count: {len(orders)}")
    for o in orders:
        print(f"Order: {o['symbol']} | {o['type']} | {o['side']} | StopPrice: {o.get('stopPrice')}")

if __name__ == '__main__':
    verify()
