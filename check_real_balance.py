import ccxt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_balance():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET')
    
    if not api_key or not api_secret:
        print("Error: API credentials not found in env.")
        return

    # Use synchronous client for simplicity
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'options': {'defaultType': 'future'},
        'enableRateLimit': True
    })

    try:
        # Fetch balance
        balance = exchange.fetch_balance()
        
        info = balance['info']
        
        print("\n--- Live Binance Futures Balance ---")
        
        # Try to parse from raw info if available (most reliable for Equity)
        if 'totalWalletBalance' in info:
             wallet_bal = float(info['totalWalletBalance'])
             unrealized_pnl = float(info['totalUnrealizedProfit'])
             total_margin_balance = float(info['totalMarginBalance']) # This is usually Equity
             
             print(f"💰 Wallet Balance: ${wallet_bal:.2f}")
             print(f"📈 Unrealized PnL: ${unrealized_pnl:.2f}")
             print(f"💎 Net Equity:     ${total_margin_balance:.2f}")
        else:
            # Fallback
            usdt_balance = balance.get('USDT', {})
            print(f"💰 Usdt Total: {usdt_balance.get('total', 'N/A')}")
            print(f"Raw Info: {info}")

    except Exception as e:
        print(f"Error fetching balance: {e}")

if __name__ == "__main__":
    fetch_balance()
