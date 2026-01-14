
import asyncio
import ccxt
import pandas as pd
import csv
import os
import random
from datetime import datetime
from trading_bot_v17 import UltimateV17Bot

class MarketScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        })
        # Phase 27: Spot Data Fallback (Bypassing Futures IP Ban)
        self.spot_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        })
        self.bot = UltimateV17Bot()
        self.bot.proxy_mgr = self.bot.proxy_mgr # Placeholder to show intent, will be handled by passing instance if needed, but for now we just need the bot's proxy_mgr to be the same.
        # Ensure the bot's exchange also uses the proxies
        self.log_file = "scan_logs.csv"

        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Symbol", "Status", "Score", "Signal", "Regime", "Funding", "OI_Mom", "Whale_Bias", "Narrative", "Reasons"])

    def log_scan(self, symbol, status, score, signal, reasons):
        try:
            file_exists = os.path.exists(self.log_file)
            is_empty = file_exists and os.path.getsize(self.log_file) == 0
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists or is_empty:
                    writer.writerow(["Timestamp", "Symbol", "Status", "Score", "Signal", "Regime", "Funding", "OI_Mom", "Whale_Bias", "Narrative", "Reasons"])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol, status, score, signal, 
                    reasons.get('regime', 'N/A'),
                    reasons.get('funding', 0),
                    reasons.get('oi_mom', 0),
                    reasons.get('whale_bias', 'NEUTRAL'),
                    reasons.get('narrative', 'N/A'),
                    "; ".join(reasons.get('reasons', []))
                ])
        except Exception as e:
            print(f"Log Error: {e}")

    def get_recent_logs(self, limit=50):
        try:
            if not os.path.exists(self.log_file): return pd.DataFrame()
            # keep_default_na=False prevents "N/A" from being converted to NaN
            df = pd.read_csv(self.log_file, keep_default_na=False)
            return df.tail(limit).sort_index(ascending=False)
        except:
            return pd.DataFrame()

    async def _fetch_with_proxy(self, exchange_config, method_name, *args, **kwargs):
        """Helper to fetch data using proxy rotation by creating fresh instances"""
        # If live, skip the direct connection as we know it's blocked for Futures
        proxy_attempts = [None] + [self.bot.proxy_mgr.get_proxy() for _ in range(3)]
        
        # Check defaults
        is_spot = True
        if isinstance(exchange_config, dict):
            is_spot = exchange_config.get('options', {}).get('defaultType') == 'spot'
        else:
            try:
                is_spot = getattr(exchange_config, 'options', {}).get('defaultType') == 'spot'
            except:
                is_spot = True
        
        for current_proxy in proxy_attempts:
            try:
                temp_exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot' if is_spot else 'future'},
                    'headers': {'User-Agent': 'Mozilla/5.0'},
                    'timeout': 10000,
                    'proxies': current_proxy if current_proxy else {}
                })
                
                method = getattr(temp_exchange, method_name)
                # Fetch markets if needed
                return await asyncio.to_thread(method, *args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                retryable = ["418", "429", "1003", "451", "404", "permission denied", "timed out", "connection reset", "network error", "remote end closed", "max retries exceeded"]
                if any(code in err_msg for code in retryable):
                    if current_proxy: self.bot.proxy_mgr.report_failure(current_proxy)
                    await asyncio.sleep(1)
                    continue
                else:
                    # Nuclear Fallback: Direct Mirror if all failed
                    if current_proxy == proxy_attempts[-1]:
                         try:
                             mirror_x = ccxt.binance({
                                 'enableRateLimit': True,
                                 'options': {'defaultType': 'spot' if is_spot else 'future'},
                                 'urls': {'api': {'public': 'https://api1.binance.com/api'}},
                                 'timeout': 10000
                             })
                             m = getattr(mirror_x, method_name)
                             return m(*args, **kwargs)
                         except: pass
                    raise e
        raise Exception(f"All proxies exhausted for {method_name}")

    async def get_top_volume_coins(self, limit=10):
        """Fetch Top coins by 24h Volume"""
        try:
            # Use spot config for volume checks with proxy fallback
            spot_config = {'options': {'defaultType': 'spot'}}
            tickers = await self._fetch_with_proxy(spot_config, 'fetch_tickers')

            valid = []
            for s, t in tickers.items():
                if '/USDT' in s and float(t['quoteVolume'] or 0) > 0:
                    symbol_base = s.split('/')[0]
                    if symbol_base in ['USDC', 'FDUSD', 'TUSD', 'USTC', 'DAI', 'EUR', 'GBP', 'USD1']:
                        continue
                    valid.append(t)
            
            if not valid: raise Exception("No valid tickers")
                
            sorted_tickers = sorted(valid, key=lambda x: float(x['quoteVolume']), reverse=True)
            return [t['symbol'] for t in sorted_tickers[:limit]]
            
        except Exception as e:
            print(f"Scanner Note: Proxy-Fetch Failed, using Major Pairs fallback: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'BNB/USDT', 'TRX/USDT', 'LINK/USDT', 'AVAX/USDT']

    async def get_alpha_strike_candidates(self, limit=10):
        """Fetch Low-Cap Moonshots (24h Gainer + Volume Spike)"""
        try:
            spot_config = {'options': {'defaultType': 'spot'}}
            tickers = await self._fetch_with_proxy(spot_config, 'fetch_tickers')

            candidates = []
            for s, t in tickers.items():
                if '/USDT' not in s: continue
                base = s.split('/')[0]
                if base in ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'BNB', 'USDC', 'FDUSD', 'TUSD']:
                    continue
                
                pct_change = float(t['percentage'] or 0)
                # Phase 33 & 35: Top Gainers (>5%) with volume filter
                if pct_change > 5:
                    candidates.append({
                        'symbol': s,
                        'pct': pct_change,
                        'vol': float(t['quoteVolume'] or 0)
                    })
            
            # Sort by percentage change
            sorted_alpha = sorted(candidates, key=lambda x: x['pct'], reverse=True)
            res_symbols = [c['symbol'] for c in sorted_alpha[:limit]]
            print(f"🔥 [ALPHA SCANNER] Found {len(candidates)} candidates > 5%. Top: {res_symbols}")
            return res_symbols
        except Exception as e:
            print(f"Alpha Scanner Note: {e}")
            return []

    async def _analyze_symbol(self, symbol, alpha_mode=False):
        """Helper for parallel scanning of a single symbol"""
        try:
            # Use spot_config to fetch OHLCV safely via proxy
            spot_config = {'options': {'defaultType': 'spot'}}
            ohlcv = await self._fetch_with_proxy(spot_config, 'fetch_ohlcv', symbol, timeframe='1h', limit=300)
            
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Phase 33: Alpha Strike Injection
            analysis = self.bot.analyze_snapshot(df, symbol, alpha_mode=alpha_mode)
            
            if analysis and analysis.get('signal') != 'ERROR':
                is_valid = analysis['signal'] != 'WAIT'
                status = "ACCEPTED" if is_valid else "REJECTED"
                
                # Phase 33: Alpha Narrative Tag
                if alpha_mode:
                    analysis['narrative'] = f"[ALPHA STRIKE] {analysis.get('narrative', '')}"
                
                self.log_scan(
                    symbol, status, analysis['score'], 
                    analysis['signal'], analysis
                )
                
                if is_valid:
                    return analysis
            else:
                err_reasons = analysis.get('reasons', ["Unknown Error"]) if analysis else ["Analysis Failed"]
                self.log_scan(symbol, "ERROR", 0, "N/A", {"reasons": err_reasons, "regime": "ERROR"})
                
        except Exception as e:
            print(f"❌ [SCANNER ERROR] {symbol}: {e}")
            self.log_scan(symbol, "ERROR", 0, "N/A", {"reasons": [f"Exception: {str(e)}"], "regime": "ERROR"})
        return None

    async def scan_market(self):
        """Scan top coins and Alpha Moonshots"""
        top_coins = await self.get_top_volume_coins(limit=12)
        alpha_coins = await self.get_alpha_strike_candidates(limit=8)
        
        # Unique list: prioritize Alpha mode
        tasks = []
        seen = set()
        
        for sym in alpha_coins:
            if sym not in seen:
                tasks.append(self._analyze_symbol(sym, alpha_mode=True))
                seen.add(sym)

        for sym in top_coins:
            if sym not in seen:
                tasks.append(self._analyze_symbol(sym, alpha_mode=False))
                seen.add(sym)

        results = await asyncio.gather(*tasks)
        opportunities = [r for r in results if r is not None]
        # Prioritize Alpha results if scores are close
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        return opportunities
