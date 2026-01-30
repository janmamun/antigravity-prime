
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET")
proxy = os.getenv("BINANCE_PROXY")

exchange_config = {
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
}
if proxy:
    exchange_config['proxies'] = {'http': proxy, 'https': proxy}

exchange = ccxt.binance(exchange_config)

try:
    print("\n--- Futures Account V2 ---")
    account = exchange.fapiPrivateV2GetAccount()
    print(f"Total Wallet Balance: {account.get('totalWalletBalance')}")
    print(f"Total Margin Balance (Equity): {account.get('totalMarginBalance')}")
    print(f"Total Unrealized Profit: {account.get('totalUnrealizedProfit')}")
    
    print("\n--- Assets Detail ---")
    for asset in account.get('assets', []):
        if float(asset['walletBalance']) != 0:
            print(f"{asset['asset']}: Wallet: {asset['walletBalance']} | Equity: {asset['marginBalance']}")

    print("\n--- Spot Balance (Non-Zero) ---")
    spot_exch = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    if proxy:
        spot_exch.proxies = {'http': proxy, 'https': proxy}
    
    spot_bal = spot_exch.fetch_balance()
    for asset, total in spot_bal.get('total', {}).items():
        if total > 0:
            print(f"{asset}: {total}")
    
except Exception as e:
    print(f"Error: {e}")
