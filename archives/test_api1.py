import ccxt
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    # Try alternate cluster
    ex = ccxt.binance({
        'options': {'defaultType': 'future'},
        'urls': {
            'api': {
                'public': 'https://fapi.binance.com', # Standard for futures
                'private': 'https://fapi.binance.com'
            }
        }
    })
    # Try api1 backup for public
    ex.urls['api']['public'] = 'https://api1.binance.com'
    print(f"Testing api1.binance.com...")
    try:
        t = ex.fetch_ticker('BTC/USDT')
        print(f"✅ Success on api1! Price: {t['last']}")
    except Exception as e:
        print(f"❌ api1 failed: {e}")

if __name__ == '__main__':
    main()
