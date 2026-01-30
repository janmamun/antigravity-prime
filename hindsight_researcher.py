
import asyncio
import ccxt
import pandas as pd
import json
import os
from datetime import datetime, timedelta

class HindsightResearcher:
    """
    Sovereign Hindsight Machine: 
    Analyzes historical chart data to identify 'Breakout Signatures' and successful strategies.
    """
    def __init__(self, exchange=None):
        self.exchange = exchange or ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        self.signatures_file = "breakout_signatures.json"
        self.load_signatures()

    def load_signatures(self):
        if os.path.exists(self.signatures_file):
            with open(self.signatures_file, 'r') as f:
                self.signatures = json.load(f)
        else:
            self.signatures = []

    def save_signatures(self):
        with open(self.signatures_file, 'w') as f:
            json.dump(self.signatures, f, indent=4)

    async def fetch_historical_data(self, symbol, timeframe='1h', days=30):
        """Fetch extensive historical data for research"""
        since = self.exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, timeframe, since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def identify_breakouts(self, df, window=14, threshold=0.05):
        """Find massive price increases (>5% in short windows)"""
        df['returns'] = df['Close'].pct_change(window)
        breakouts = df[df['returns'] > threshold].copy()
        return breakouts

    def extract_signature(self, df, breakout_idx):
        """Extract the technical state BEFORE the breakout"""
        # Look at the state 3 bars before the breakout
        pre_idx = max(0, breakout_idx - 3)
        segment = df.iloc[max(0, pre_idx-20):pre_idx]
        
        if len(segment) < 20: return None
        
        # Calculate key metrics (simplified for prototype)
        close = segment['Close']
        vol = segment['Volume']
        
        # Get the timestamp from the breakout row
        ts = df.index[breakout_idx]
        ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
        
        signature = {
            "avg_vol_surge": (vol.iloc[-1] / vol.mean()) if vol.mean() > 0 else 1,
            "trend_slope": (close.iloc[-1] - close.iloc[0]) / close.iloc[0],
            "volatility": segment['returns'].std() if 'returns' in segment else 0,
            "timestamp": ts_str
        }
        return signature

    async def perform_research(self, symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LPTUSDT', 'LINKUSDT', 'AVAXUSDT']):
        print(f"🔭 [HINDSIGHT] Starting historical research on {len(symbols)} assets...")
        for symbol in symbols:
            try:
                df = await self.fetch_historical_data(symbol, timeframe='1h', days=60)
                breakouts = self.identify_breakouts(df, window=4, threshold=0.03) # 3% in 4h
                
                for idx in breakouts.index:
                    # Find numerical index
                    num_idx = df.index.get_loc(idx)
                    sig = self.extract_signature(df, num_idx)
                    if sig:
                        sig['symbol'] = symbol
                        self.signatures.append(sig)
                
                # Keep only unique/recent signatures
                self.signatures = self.signatures[-200:]
                self.save_signatures()
                print(f"✅ [HINDSIGHT] Processed {symbol}: Found {len(breakouts)} potential signatures.")
            except Exception as e:
                print(f"⚠️ [HINDSIGHT] Research failed for {symbol}: {e}")

if __name__ == "__main__":
    # Test Run
    async def test():
        researcher = HindsightResearcher()
        await researcher.perform_research()
        print(f"Total Signatures Extracted: {len(researcher.signatures)}")
    
    asyncio.run(test())
