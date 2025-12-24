
import ccxt
import pandas as pd
import time

class BinanceClient:
    def __init__(self, api_key=None, secret_key=None):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Futures trading
            }
        })
        self.public_mode = not (api_key and secret_key)
        
    def fetch_ohlcv(self, symbol, timeframe='1d', limit=100):
        """Fetch candle data"""
        try:
            # Map symbol format (e.g. BTC-USD to BTC/USDT)
            # Our bot uses yahoo format 'BTC-USD', ccxt uses 'BTC/USDT'
            clean_symbol = symbol.replace('-USD', '/USDT')
            
            ohlcv = self.exchange.fetch_ohlcv(clean_symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
            
    def get_current_price(self, symbol):
        clean_symbol = symbol.replace('-USD', '/USDT')
        try:
            ticker = self.exchange.fetch_ticker(clean_symbol)
            return ticker['last']
        except:
            return 0.0
