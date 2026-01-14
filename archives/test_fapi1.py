import ccxt
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    ex = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'options': {'defaultType': 'future'}
    })
    # Manual override to fapi1
    ex.urls['api']['public'] = 'https://fapi1.binance.com/fapi/v1'
    ex.urls['api']['private'] = 'https://fapi1.binance.com/fapi/v1'
    
    print("Testing fapi1.binance.com balance check...")
    try:
        b = ex.fetch_balance()
        print(f"✅ Success on fapi1! Balance: {b['total']['USDT']}")
    except Exception as e:
        print(f"❌ fapi1 failed: {e}")

if __name__ == '__main__':
    main()
