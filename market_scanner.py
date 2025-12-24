
import ccxt
import pandas as pd
import csv
import os
from datetime import datetime
from trading_bot_v17 import UltimateV17Bot

class MarketScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.bot = UltimateV17Bot()
        self.log_file = "scan_logs.csv"
        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Symbol", "Status", "Score", "Signal", "Regime", "Funding", "OI_Mom", "Whale_Bias", "Reasons"])

    def log_scan(self, symbol, status, score, signal, reasons):
        try:
            file_exists = os.path.exists(self.log_file)
            is_empty = file_exists and os.path.getsize(self.log_file) == 0
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists or is_empty:
                    writer.writerow(["Timestamp", "Symbol", "Status", "Score", "Signal", "Regime", "Funding", "OI_Mom", "Whale_Bias", "Reasons"])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol, status, score, signal, 
                    reasons.get('regime', 'N/A'),
                    reasons.get('funding', 0),
                    reasons.get('oi_mom', 0),
                    reasons.get('whale_bias', 'NEUTRAL'),
                    "; ".join(reasons.get('reasons', []))
                ])
        except Exception as e:
            print(f"Log Error: {e}")

    def get_recent_logs(self, limit=50):
        try:
            if not os.path.exists(self.log_file): return pd.DataFrame()
            df = pd.read_csv(self.log_file)
            return df.tail(limit).sort_index(ascending=False)
        except:
            return pd.DataFrame()

    def get_top_volume_coins(self, limit=10):
        """Fetch Top coins by 24h Volume"""
        try:
            tickers = self.exchange.fetch_tickers()
            valid = []
            for s, t in tickers.items():
                if '/USDT' in s and float(t['quoteVolume'] or 0) > 0:
                    valid.append(t)
            
            if not valid: raise Exception("No valid tickers")
                
            sorted_tickers = sorted(valid, key=lambda x: float(x['quoteVolume']), reverse=True)
            return [t['symbol'] for t in sorted_tickers[:limit]]
            
        except Exception as e:
            print(f"Scanner Note: Using Major Liquid Pairs (Auto-Fallback)")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'BNB/USDT', 'TRX/USDT', 'LINK/USDT', 'AVAX/USDT']

    def scan_market(self):
        """Scan top coins and return the best Trade Setup"""
        top_coins = self.get_top_volume_coins(limit=100)
        opportunities = []
        
        for symbol in top_coins:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=300)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                analysis = self.bot.analyze_snapshot(df, symbol)
                
                if analysis and analysis.get('signal') != 'ERROR':
                    is_valid = analysis['signal'] != 'WAIT'
                    status = "ACCEPTED" if is_valid else "REJECTED"
                    
                    self.log_scan(
                        symbol, status, analysis['score'], 
                        analysis['signal'], analysis
                    )
                    
                    if is_valid:
                        opportunities.append(analysis)
                else:
                    # Analysis returned None or Error
                    err_reasons = analysis.get('reasons', ["Unknown Error"]) if analysis else ["Analysis Failed"]
                    self.log_scan(symbol, "ERROR", 0, "N/A", {"reasons": err_reasons, "regime": "ERROR"})
                    
            except Exception as e:
                self.log_scan(symbol, "ERROR", 0, "N/A", {"reasons": [f"Exception: {str(e)}"], "regime": "ERROR"})
                continue
                
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        return opportunities
