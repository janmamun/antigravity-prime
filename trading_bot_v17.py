import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
import os
import json
import ccxt
from dotenv import load_dotenv
from neural_oracle import NeuralOracle
from sentiment_scanner import SentimentScanner
from proxy_manager import ProxyManager

from news_syndicate import NewsSyndicate
from telegram_bridge import TelegramBridge

class NewsSentimentScanner:
    """Sovereign V17.0 News Intelligence (Gemini-Powered)"""
    def __init__(self):
        self.syndicate = NewsSyndicate()
        
    def fetch_news_bias(self, symbol):
        """Fetch real news sentiment from Gemini Syndicate"""
        res = self.syndicate.get_market_sentiment(symbol)
        return {
            "score": res.get("sentiment_score", 0),
            "narrative": res.get("narrative", "Neutral market conditions.")
        }

warnings.filterwarnings('ignore')

class UltimateV17Bot:
    def __init__(self, initial_capital=100):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.safeguard_vault = 0
        self.hedge_active = False # Phase 8.0: Crash Shield
        self.load_safeguard()
        self.trades = []
        self.daily_equity = []
        
        self.config_file = "bot_config.json"
        self.load_config()

        # Phase 26: Memory-based API caching
        self._cache = {} 
        self._cache_ttl = 60 # 1 minute
        
        # Phase 17: State Persistence for Hindsight
        self.hindsight_file = "hindsight_data.json"
        self.load_hindsight()
        
        # Super-Intelligence Modules
        self.oracle = NeuralOracle()
        self.sentiment = SentimentScanner()
        
        # Phase 23: Live Mode Initialization
        load_dotenv()
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_SECRET")
        self.is_live = os.getenv("LIVE_MODE", "false").lower() == "true"
        
        # CCXT Binance for Funding Rates & Data
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }
        
        # Phase 25: Geo-Restriction Fix (Proxy Support)
        proxy = os.getenv("BINANCE_PROXY")
        if proxy:
            exchange_config['proxies'] = {
                'http': proxy,
                'https': proxy
            }
            print(f"🌐 [SYSTEM] Proxy detected: {proxy[:15]}...")
        
        # Phase 25: Guardian 2.0 Safeguards
        self.max_daily_loss_pct = 5.0  # Stop all trading if down 5% today
        self.emergency_stop = False
        self.session_start_equity = self.capital
        
        # Phase 23: The Sovereign Ascension (AI-Ownership Mode)
        self.win_rate_history = [] # Rolling 10 trades [1, 0, 1...]
        self.cortex_multiplier = 1.0 # 0.8 to 1.5 based on performance
        self.dynamic_barrier = 55.0 # Phase 40: Aggressive Trading Mode
        self.liquidity_memory = {} # {symbol: {bids: [], asks: []}}
        
        if self.is_live and self.api_key and self.api_secret:
            try:
                exchange_config.update({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'timeout': 5000, # 5 second timeout
                })
                self.exchange = ccxt.binance(exchange_config)
                # Skip fetch_balance during init to avoid hanging the UI
                print(f"🚀 [SYSTEM] ENTRANCE TO LIVE TRADE PROTOCOL: {self.api_key[:5]}... STANDBY")
            except Exception as e:
                print(f"⚠️ [CORE] INITIAL CONNECTION BLOCKED (IP BAN): {e}")
                # Keep is_live=True to allow ProxyManager to attempt the bypass during execution
                self.exchange = ccxt.binance(exchange_config)
        else:
            self.exchange = ccxt.binance(exchange_config)
            print("🪐 [SYSTEM] Simulation Mode Active. No Live Keys Detected.")
        
        # Phase 27: Market Analytics Exchange (Spot Fallback + Strict Timeout)
        market_config = {'enableRateLimit': True, 'timeout': 3000, 'options': {'defaultType': 'spot'}}
        if self.is_live and self.api_key and self.api_secret:
            market_config.update({'apiKey': self.api_key, 'secret': self.api_secret})
        
        self.market_exch = ccxt.binance(market_config) 
        
        # God-Mode: Proxy Bypass Layer (Dynamic Sentinel - Skip Init for speed)
        self.proxy_mgr = ProxyManager(skip_init=True) 
        
        # Phase 17: Sentiment Syndicate (Real-Time News Fusion)
        self.news_scanner = NewsSentimentScanner()
        
        # Phase 19: Telegram Alert Syndicate
        self.telegram = TelegramBridge()

        # Phase 10.0: Institutional Spoofing Defense
        self.persistence_store = {} # {symbol: {'bids': [history], 'asks': [history], 'timestamps': [...]}}

    def load_config(self):
        # Phase 38: Use cache to avoid hammering disk on every scan
        now = time.time()
        if hasattr(self, '_config_cache_time') and (now - self._config_cache_time) < 300: # 5 min cache
            return self.config

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                    self._config_cache_time = now
            except:
                self.config = {
                    "rsi_oversold": 35,
                    "rsi_overbought": 70,
                    "min_score": 45,
                    "risk_factor": 0.1,
                    "leverage": 5,
                    "dca_enabled": True,
                    "dca_max_entries": 3,
                    "max_open_positions": 6,
                    "hedge_ratio": 0.5, # Phase 8.0
                    "crash_threshold": -0.02, # Phase 8.0: -2% in 5m
                    "compounding_ratio": 0.15 # Phase 10.0: Aggressive growth allocation
                }
        else:
            self.config = {
                "rsi_oversold": 35,
                "rsi_overbought": 70,
                "min_score": 45,
                "risk_factor": 0.1,
                "leverage": 5,
                "dca_enabled": True,
                "dca_max_entries": 3,
                "max_open_positions": 6,
                "hedge_ratio": 0.5, # Phase 8.0
                "crash_threshold": -0.02, # Phase 8.0: -2% in 5m
                "compounding_ratio": 0.15 # Phase 10.0: Aggressive growth allocation
            }
        
        # Phase 28: Global Toxic Asset Blacklist
        self.blacklist_file = "blacklist.json"
        self.blacklist = self.load_blacklist()
        return self.config

    def load_blacklist(self):
        # Phase 38: Use cache for blacklist
        now = time.time()
        if hasattr(self, '_blacklist_cache_time') and (now - self._blacklist_cache_time) < 300:
            return self.blacklist

        if os.path.exists(self.blacklist_file):
            try:
                with open(self.blacklist_file, 'r') as f:
                    self.blacklist = json.load(f)
                    self._blacklist_cache_time = now
            except: self.blacklist = []
        else: self.blacklist = []
        return self.blacklist

    def load_hindsight(self):
        if os.path.exists(self.hindsight_file):
            with open(self.hindsight_file, 'r') as f:
                self.hindsight = json.load(f)
        else:
            self.hindsight = {} # {symbol: [list of previous predictions]}

    def save_hindsight(self):
        with open(self.hindsight_file, 'w') as f:
            json.dump(self.hindsight, f, indent=4)

    def load_safeguard(self):
        """Phase 6.0: Load the perpetual PnL vault balance from sim_state.json or config"""
        if os.path.exists("sim_state.json"):
            try:
                with open("sim_state.json", 'r') as f:
                    state = json.load(f)
                    self.safeguard_vault = state.get("safeguard_vault", 0)
            except: self.safeguard_vault = 0
        else:
            self.safeguard_vault = 0

    def save_safeguard(self):
        """Phase 6.0: Update sim_state with the current vault balance"""
        if os.path.exists("sim_state.json"):
            try:
                with open("sim_state.json", 'r') as f:
                    state = json.load(f)
                state["safeguard_vault"] = self.safeguard_vault
                with open("sim_state.json", 'w') as f:
                    json.dump(state, f, indent=4)
            except: pass

    def _calculate_atr(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_indicators_v14(self, df):
        """Phase 14.2: Technical indicators for Volatility Squeeze detection"""
        # Bollinger Bands
        period = 20
        df['MA20'] = df['Close'].rolling(window=period).mean()
        df['STD20'] = df['Close'].rolling(window=period).std()
        df['BB_UP'] = df['MA20'] + (2 * df['STD20'])
        df['BB_LOW'] = df['MA20'] - (2 * df['STD20'])
        
        # Keltner Channels
        atr = self._calculate_atr(df, period=20)
        df['KC_UP'] = df['MA20'] + (1.5 * atr)
        df['KC_LOW'] = df['MA20'] - (1.5 * atr)
        
        return df

    def _calculate_adx(self, df, period=14):
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        tr = self._calculate_atr(df, period=1)
        plus_di = 100 * (plus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-6))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-6))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        adx = dx.rolling(period).mean()
        return adx

    def get_funding_rate(self, symbol):
        try:
            # Funding is only on futures. If banned, return 0.0001
            funding = self.exchange.fetch_funding_rate(symbol)
            return funding['fundingRate']
        except:
            return 0.0001

    def _calculate_oi_momentum(self, symbol):
        # Move to self.market_exch or handle error (OI is futures only)
        try:
            cache_key = f"oi_{symbol}"
            now = time.time()
            if cache_key in self._cache and (now - self._cache[cache_key]['time']) < self._cache_ttl:
                return self._cache[cache_key]['val']

            # OI is futures-only. This will still fail if fapi is banned.
            oi_history = self.exchange.fetch_open_interest_history(symbol, timeframe='5m', limit=10)
            if not oi_history or len(oi_history) < 2: return 0.0
            latest_oi = float(oi_history[-1]['openInterestAmount'])
            prev_oi_avg = np.mean([float(h['openInterestAmount']) for h in oi_history[:-1]])
            res = (latest_oi / prev_oi_avg) - 1
            
            self._cache[cache_key] = {'val': res, 'time': now}
            return res
        except:
            return 0.0

    def _get_trade_based_metrics(self, symbol):
        """Phase 26/6.0: Consolidate trade history calls to save API weight & track Block Trades"""
        try:
            cache_key = f"trades_{symbol}"
            now = time.time()
            if cache_key in self._cache and (now - self._cache[cache_key]['time']) < self._cache_ttl:
                return self._cache[cache_key]['val']

            # Use Spot Exchange for trades (Not banned!)
            trades = self.market_exch.fetch_trades(symbol, limit=200)
            
            # Whale Bias & Block Trades (>$100k)
            whales = [t for t in trades if (float(t['amount']) * float(t['price'])) >= 100000]
            block_trades = len(whales)
            block_volume = sum(float(t['amount']) * float(t['price']) for t in whales)
            
            whale_bias = "NEUTRAL"
            if whales:
                buy_vol = sum(float(t['amount']) for t in whales if t['side'] == 'buy')
                sell_vol = sum(float(t['amount']) for t in whales if t['side'] == 'sell')
                if buy_vol > sell_vol * 1.5: whale_bias = "BULLISH"
                elif sell_vol > buy_vol * 1.5: whale_bias = "BEARISH"
            
            # CVD
            buys = sum(float(t['amount']) for t in trades if t['side'] == 'buy')
            sells = sum(float(t['amount']) for t in trades if t['side'] == 'sell')
            cvd = (buys - sells) / (buys + sells + 1e-6)
            
            res = {"whale_bias": whale_bias, "cvd": cvd, "block_trades": block_trades, "block_volume": block_volume}
            self._cache[cache_key] = {'val': res, 'time': now}
            return res
        except:
            return {"whale_bias": "NEUTRAL", "cvd": 0.0, "block_trades": 0}

    def analyze_snapshot(self, df, symbol="UNKNOWN", fetch_callback=None, alpha_mode=False):

        # Phase 38: Use optimized cache-bound reloads
        self.blacklist = self.load_blacklist()
        self.load_config()

        if symbol in self.blacklist:
            return {"symbol": symbol, "signal": "SKIP", "score": 0, "regime": "BLACKLIST", "reasons": ["GLOBAL BLACKLIST: TOXIC ASSET"]}

        score = 0
        reasons = []

        if len(df) < 50:
            return {"symbol": symbol, "signal": "ERROR", "score": 0, "regime": "ERROR", "reasons": ["Insufficient Data"]}

        # 1. Indicators & ATR (Dynamic Periods)
        ma_fast = self.config.get('ma_fast', 19)
        ma_slow = self.config.get('ma_slow', 40)
        
        df['MA40'] = df['Close'].rolling(ma_slow).mean()
        df['MA19'] = df['Close'].rolling(ma_fast).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['ATR'] = self._calculate_atr(df)
        df = self._calculate_indicators_v14(df)
        df['ADX'] = self._calculate_adx(df)
        
        current_price = df['Close'].iloc[-1]
        adr = df['ATR'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        
        # 3. Regime Detection (Moved Up for Phase 9.0)
        regime = "NEUTRAL"
        if adx > 25: regime = "TREND"
        elif adx < 20: regime = "RANGE"
        
        # 2. Sovereign V2.0 Intelligence (Oracle & News)
        sentiment_data = self.sentiment.analyze_sentiment(df)
        news_data = self.news_scanner.fetch_news_bias(symbol)
        news_bias = news_data.get("score", 0)
        narrative = news_data.get("narrative", "N/A")
        
        # Phase 10: Inject Orderflow Data for Neural Shadow
        trade_metrics = self._get_trade_based_metrics(symbol)
        liq_info = self._get_liquidity_depth(symbol)
        
        df['OBI'] = liq_info.get('imbalance', 0.0)
        df['CVD_Delta'] = trade_metrics.get('cvd', 0.0)
        
        oracle_data = self.oracle.predict(df, symbol)
        orderflow_shadow = self.oracle.predict_orderflow(df, symbol)
        
        # ML Confidence Filter (Optimized for V9.0: Neural Decoupling)
        ml_confidence = oracle_data['confidence_score'] if oracle_data else 0
        conf_req = 10 if regime == "RANGE" else 25 # Lowered for more alpha
        
        neural_silence = ml_confidence < conf_req
        if neural_silence:
            reasons.append(f"NEURAL SILENCE: AI Uncertain ({ml_confidence:.1f}/{conf_req}). Technical Override Active.")


        # 2.5 Macro-Sentiment Pulse (Phase 7.0)
        macro_bias = self.config.get('macro_bias', 'NEUTRAL')
        if macro_bias == "RISK-OFF":
            score -= 10
            reasons.append("MACRO BIAS: RISK-OFF (Preservation Mode)")
        elif macro_bias == "RISK-ON":
            score += 10
            reasons.append("MACRO BIAS: RISK-ON (Aggression Boost)")


        # 4. Multi-Timeframe Confluence (Scalp vs Trend)
        sma20 = df['Close'].rolling(20).mean().iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        
        is_uptrend = current_price > sma50
        is_scalp_bullish = current_price > sma20
        
        if is_uptrend and is_scalp_bullish:
            score += 25
            reasons.append("MTF Confluence: BULLISH (1h/5m Match)")
        elif not is_uptrend and not is_scalp_bullish:
            # Phase 9.2: Neutralize Trend Penalty on Washouts
            if regime == "RANGE" and df['RSI'].iloc[-1] < 35:
                score -= 5 # Reduced penalty for washout
                reasons.append("MTF BEARISH: neutralized by Range-Washout setup")
            else:
                score -= 25
                reasons.append("MTF Confluence: BEARISH (1h/5m Match)")
        else:
            # Phase 9.1: Reduce penalty in ranges (Neutrality is normal)
            penalty = 5 if regime == "RANGE" else 15
            score -= penalty
            reasons.append(f"Trend Divergence: Low Impact in {regime} (-{penalty})")

        # 2.7 Phase 17: Sentiment Syndicate Fusion (Phase 21: Deep Aggression)
        # We now double the impact of the sentiment syndicate for aggressive frontal-entry
        score += news_bias * 1.5 
        if abs(news_bias) > 20:
             score += (news_bias * 0.5) # Extra bonus for extreme narrative shifts
             reasons.append(f"Narrative Fusion: {narrative} (+{news_bias * 1.5:.1f})")
        else:
             reasons.append(f"News Sentiment: {narrative} ({news_bias:.1f})")

        # 6. Sovereign Edge Nodes (RESTORED)
        # A. Magic 19/40 Cross
        curr_19 = df['MA19'].iloc[-1]
        curr_40 = df['MA40'].iloc[-1]
        prev_19 = df['MA19'].iloc[-2]
        prev_40 = df['MA40'].iloc[-2]
        
        if prev_19 < prev_40 and curr_19 > curr_40:
            score += 30
            reasons.append("MAGIC CROSS: Golden (19/40)")
        elif prev_19 > prev_40 and curr_19 < curr_40:
            score -= 30
            reasons.append("MAGIC CROSS: Death (19/40)")

        # B. Liquidity Sweep
        sweep = self._detect_liquidity_sweep(df)
        if sweep == "BULLISH_SWEEP":
            score += 35
            reasons.append("LIQUIDITY SWEEP: Bullish PDL Grab")
        elif sweep == "BEARISH_SWEEP":
            score -= 35
            reasons.append("LIQUIDITY SWEEP: Bearish PDH Grab")

        # C. Volatility Climax
        climax = self._detect_volatility_climax(df)
        if climax == "CLIMAX":
            score *= 0.5 # Neutralize score on exhaustion
            reasons.append("VOLATILITY CLIMAX: Exhaustion Detected (Neutralizing)")

        # D. Sovereign Memory Booster
        memory_boost = self._get_memory_booster(symbol)
        if memory_boost != 0:
            score += memory_boost
            reasons.append(f"Memory Booster: Active (+{memory_boost})")

        # 6. Technical Edges (Legacy & Enhanced)
        rsi_os = self.config.get('rsi_oversold', 30)
        rsi_ob = self.config.get('rsi_overbought', 70)
        
        if df['RSI'].iloc[-1] < rsi_os:
            boost = 35 if regime == "RANGE" else 20
            score += boost
            reasons.append(f"RSI Oversold (<{rsi_os}) | RANGE BOOST (+{boost})")
        elif df['RSI'].iloc[-1] > rsi_ob:
            boost = 35 if regime == "RANGE" else 20
            score -= boost
            reasons.append(f"RSI Overbought (>{rsi_ob}) | RANGE BOOST (+{boost})")

        # Phase 5.0: Liquidity & Institutional Imbalance
        trade_metrics = self._get_trade_based_metrics(symbol)
        cvd_delta = trade_metrics['cvd']
        liq_info = self._get_liquidity_depth(symbol)
        liq_depth = liq_info['depth']
        imbalance = liq_info['imbalance']
        
        if cvd_delta > 0.05:
            score += 20
            reasons.append(f"Bullish CVD Delta (+{cvd_delta*100:.1f}%)")
        
        if liq_depth < 1.0: # Filter for extremely thin books
            score -= 40 
            reasons.append(f"CRITICAL: Low Liquidity ({liq_depth:.1f}x)")
            
        if imbalance > 0.3: # Strong bid support
            score += 15
            reasons.append(f"INSTITUTIONAL: Strong Bid Support ({imbalance*100:.1f}%)")
        elif imbalance < -0.3: # Heavy ask wall
            score -= 25
            reasons.append(f"INSTITUTIONAL: Heavy Ask Wall Resistance ({imbalance*100:.1f}%)")

        # Phase 7.0: Basket Correlation Guard (Diversified Alpha)
        if symbol != "BTC/USDT":
            portfolio_correlation = self._calculate_basket_correlation(df, symbol)
            if portfolio_correlation > 0.60:
                score -= 30
                reasons.append(f"BASKET CORRELATION GUARD: High Portfolio Beta ({portfolio_correlation:.2f})")
            elif portfolio_correlation < 0.30:
                score += 15
                reasons.append(f"BASKET ALPHA: Diversified Asset ({portfolio_correlation:.2f})")

        # Phase 8.0: Liquidity Sniper (Orderbook Depth)
        # Check depth for BOTH sides to inform the score before signal is locked
        buy_depth = self._get_orderbook_depth(symbol, "BUY", fetch_callback=fetch_callback)
        sell_depth = self._get_orderbook_depth(symbol, "SELL", fetch_callback=fetch_callback)

        
        if buy_depth > 1.8:
            score += 20
            reasons.append(f"LIQUIDITY SNIPER: Massive Buy Wall ({buy_depth:.1f}x)")
        elif sell_depth > 1.8:
            score -= 20
            reasons.append(f"LIQUIDITY RISK: Massive Sell Wall ({sell_depth:.1f}x)")

        # Phase 8.0: Recursive Performance Constraints
        constraints = self._get_learned_constraints()
        if symbol in constraints:
            c = constraints[symbol]
            if time.time() < c.get('expiry', 0):
                boost_req = c.get('min_score_boost', 15)
                score -= boost_req
                reasons.append(f"RECURSIVE PROBATION: {c.get('reason')} (-{boost_req} pts)")
        exhaustion = self._detect_whale_exhaustion(symbol)
        if exhaustion:
            score -= 30
            reasons.append("WHALE EXHAUSTION: High CVD Volatility detected")

        # Phase 10.0: Neural Shadow (Orderflow Prediction)
        if orderflow_shadow == "ACCUMULATION_SHADOW":
            score += 25
            reasons.append("NEURAL SHADOW: Institutional Accumulation Predicted (+25)")
        elif orderflow_shadow == "DISTRIBUTION_SHADOW":
            score -= 25
            reasons.append("NEURAL SHADOW: Institutional Distribution Predicted (-25)")

        # Phase 10.0: Institutional Spoofing Defense (Liquidity Persistence)
        persistence_trust = self._check_liquidity_persistence(symbol, liq_info['depth'] * (1+imbalance), liq_info['depth'] * (1-imbalance), fetch_callback=fetch_callback)

        if persistence_trust < 1.0:
            score -= 35
            reasons.append("INSTITUTIONAL INTEGRITY: Flickering/Spoofed Walls detected (-35)")

        # Hindsight Memory
        if symbol in self.hindsight:
            recent_fails = [h for h in self.hindsight[symbol][-5:] if h['result'] == 'LOSS']
            if len(recent_fails) >= 2:
                score -= 15
                reasons.append("Hindsight Penalty (Recent Logic Failure)")

        # Phase 5.0: Capital Compounding Milestone Check
        milestone_bonus = 0
        if self.is_live:
            live_bal = self.get_live_balance()

            if live_bal > 120: # +20% Milestone
                milestone_bonus = 10
                reasons.append(f"COMPOUNDING MILESTONE: Active (+20% ROI Bonus)")
            elif live_bal > 140: # +40% Milestone
                milestone_bonus = 20
                
        score += milestone_bonus

        # Phase 29: Systemic Failure Protection (Global)
        if hasattr(self, 'trades') and len(self.trades) >= 3:
            recent_trades = self.trades[-3:]
            if all(t.get('PnL', 0) < 0 for t in recent_trades):
                score -= 20
                reasons.append("SYSTEMIC FAILURE: 3 Sequential Losses Detected (Tightening)")

        # Phase 7.0: Funding Harvest Protocol
        funding_rate = self.get_funding_rate(symbol)
        if funding_rate < -0.0005:
            strategy = "SHORT SQUEEZE YIELD"
        elif funding_rate > 0.0005:
            strategy = "LONG EXHAUSTION HARVEST"
        else:
            strategy = "INSTITUTIONAL TREND SCALP"
            if regime == "RANGE":
                strategy = "GRID REVERSION MATRIX"

        # Phase 15: Institutional Alpha Pulse (Tidal Shift & Liquidity Vacuum)
        vacuum_detected = self._detect_liquidity_vacuum(symbol, fetch_callback=fetch_callback)

        if vacuum_detected:
            score += 20
            reasons.append("LIQUIDITY VACUUM: Low-resistance zone detected (+20)")
        
        tidal_roc = self._calculate_tidal_shift(symbol)
        if tidal_roc > 0:
            score += 15
            reasons.append(f"TIDAL SHIFT: Institutional Flow Accelerating ({tidal_roc:.1f}% RoC)")
        elif tidal_roc < -50:
            score -= 10
            reasons.append("TIDAL SHIFT: Institutional Flow Receding")

        # 10. Phase 14 enhancements: Squeeze Sniper
        is_squeezing = self._detect_volatility_squeeze(df)
        if is_squeezing:
            score += 10
            reasons.append("VOLATILITY SQUEEZE: Explosion imminent (+10 bonus)")

        # Phase 33: Alpha Strike Momentum Injection
        if alpha_mode:
            # A. 24h Breakout Detection
            day_high = df['High'].tail(24).max()
            if current_price >= day_high * 0.99:
                score += 30
                reasons.append("ALPHA BREAKOUT: Price at/near 24h High (+30)")
            
            # B. Volume Spike Confirmation
            last_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].tail(20).mean()
            if last_vol > avg_vol * 2.0:
                score += 15
                reasons.append("ALPHA VOLUME: Massive 1h Spike detected (+15)")
            
            # C. Institutional Front-Run (CVD)
            if cvd_delta > 0.15:
                score += 20
                reasons.append("ALPHA CVD: Strong Front-Running detected (+20)")

        # Phase 40: Momentum Breakout Hunter (For ALL trades, not just Alpha)
        day_high = df['High'].tail(24).max()
        day_low = df['Low'].tail(24).min()
        last_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].tail(20).mean()
        
        # Bullish Breakout: Price breaks 24h high with volume
        if current_price >= day_high * 0.995 and last_vol > avg_vol * 1.5:
            score += 40
            reasons.append("🚀 BREAKOUT HUNTER: 24h High Break + Volume Confirmation (+40)")
        # Bearish Breakdown: Price breaks 24h low with volume
        elif current_price <= day_low * 1.005 and last_vol > avg_vol * 1.5:
            score -= 40
            reasons.append("📉 BREAKDOWN HUNTER: 24h Low Break + Volume Confirmation (-40)")

        # 7. Final Decision & Dynamic Barrier (Phase 23: Sovereign Ascension)
        signal = "WAIT"
        
        # Self-Evolving Barrier: Moves between 55 and 85 based on Win Rate
        min_score = self.dynamic_barrier
        
        # Phase 40: Enhanced Alpha Strike Barrier
        if alpha_mode:
            min_score = 45.0 # Ultra-aggressive entry for Moonshots
            reasons.append(f"ALPHA MODE: Ultra-Low Entry Barrier {min_score} for Moonshot Seizing.")
        
        if regime == "TREND":
            min_score -= 5 # Slightly lower for trending markets
        
        # Neural Decoupling: If AI is silent, we need HIGHER technical conviction
        if neural_silence and not alpha_mode: # Alpha Mode relies on Alpha Technicals
            penalty = 5 if self.config.get('risk_factor') == "AGGRESSIVE" else 15
            min_score += penalty 
            reasons.append(f"NEURAL DECOUPLING: Score requirement increased to {min_score} (Penalty: +{penalty})")

        # Phase 29: Sentiment Aggressor & Quota-Hit Resilience
        if abs(news_bias) > 30:
            min_score -= 20 
            reasons.append(f"SENTIMENT AGGRESSOR: EXPLOSIVE NARRATIVE. Lowering Entry Barrier to {min_score}")
        elif abs(news_bias) > 12:
            min_score -= 10
            reasons.append(f"SENTIMENT AGGRESSOR: News Momentum detected. Lowering Entry Barrier to {min_score}")
        elif news_bias == 0 and "Quota" in narrative:
            # QUOTA HIT FALLBACK: If we can't get news, we lower the barrier slightly to allow technicals to lead
            min_score -= 5
            reasons.append(f"QUOTA FALLBACK: LLM Silhouette active. Lowering barrier to {min_score} for technical priority.")

        # Phase 40: Enhanced Funding Rate Harvester
        if score >= min_score and funding_rate < -0.0005:
            bonus = 25 if funding_rate < -0.001 else 15
            score += bonus
            reasons.append(f"💰 FUNDING HARVEST: Longing with {funding_rate*100:.3f}% funding (+{bonus})")
        elif score <= -min_score and funding_rate > 0.0005:
            bonus = 25 if funding_rate > 0.001 else 15
            score += bonus
            reasons.append(f"💰 FUNDING HARVEST: Shorting with {funding_rate*100:.3f}% funding (+{bonus})")

        # Phase 23: Liquidity Trap Sniper
        if score >= min_score or score <= -min_score:
            if self._detect_liquidity_trap(symbol):
                score *= 0.5 # Slash conviction if a trap is suspected
                reasons.append("⚠️ [CORTEX] LIQUIDITY TRAP SUSPECTED: Slicing conviction score by 50%")

        # Final Signal Assignment
        if score >= min_score: 
            signal = "BUY"
            tp = current_price + (adr * 2.5)
            sl = current_price - (adr * 1.5)
        elif score <= -min_score: 
            signal = "SELL"
            tp = current_price - (adr * 2.5)
            sl = current_price + (adr * 1.5)
        else:
            tp = current_price + (adr * 2.5)
            sl = current_price - (adr * 1.5)
        
        # Phase 11.1: Adaptive TP Grid (Restored)
        tp_grid = self.calculate_tp_grid(current_price, signal, adr)

        # Phase 15: Turbo Scaling Flag
        conviction_threshold = self.config.get('conviction_threshold', 95)
        turbo_scaling = score >= conviction_threshold and is_squeezing
        if turbo_scaling:
            reasons.append(f"MISSION CRITICAL: Conviction {score} reached. TURBO SCALING ENGAGED.")

        # Save prediction for Hindsight
        self._record_prediction(symbol, signal, score, current_price)

        return {
            "symbol": symbol, "signal": signal, "score": score,
            "price": current_price, "current_price": current_price,
            "tp": tp, "sl": sl, "tp_grid": tp_grid, "atr": adr, "regime": regime,
            "strategy": strategy, "turbo_scaling": turbo_scaling, "alpha_mode": alpha_mode,
            "funding": funding_rate, "oi_mom": 0.0, "tidal_roc": tidal_roc,
            "whale_bias": "NEUTRAL", "narrative": narrative, "reasons": reasons
        }

    def check_safeguards(self):
        """Guardian 2.0: Prevent catastrophic loss"""
        print("DEBUG: check_safeguards initiated")
        current_equity = self.get_live_balance()
        print(f"DEBUG: current_equity for safeguards: {current_equity}")
        
        # Security: If equity is 0 and we were banned, it's a false alarm
        if current_equity == 0 and self.session_start_equity > 0:
            print("⚠️ [GUARDIAN] Equity reported as 0. Suspected API Block. Safeguard Bypassed.")
            return True
            
        drawdown = (self.session_start_equity - current_equity) / (self.session_start_equity + 1e-6) * 100
        print(f"DEBUG: Drawdown calculated: {drawdown:.2f}%")
        
        if drawdown >= self.max_daily_loss_pct:
            print(f"🚨 [GUARDIAN] EMERGENCY STOP: Daily Loss Limit Reached ({drawdown:.2f}% | Equity: {current_equity})")
            self.emergency_stop = True
            return False
        return True

    def get_live_balance(self):
        """Phase 23/ God-Mode: Fetch real USDT Futures balance with Direct-First Proxy Fallback"""
        if not self.is_live: return self.capital
        
        # 1. Try Direct first
        self.exchange.proxies = {}
        try:
            balance = self.exchange.fetch_balance()
            # Log full balance keys for debugging
            total_bal = balance.get('total', {})
            usdt_bal = float(total_bal.get('USDT', 0))
            
            print(f"💰 [SYSTEM] Net Wallet Integrity: USDT: {usdt_bal:.2f} | Total Keys: {list(total_bal.keys())[:5]}")
            
            self.capital = usdt_bal
            return usdt_bal
        except Exception as e:
            err_str = str(e)
            if "418" not in err_str and "429" not in err_str:
                print(f"⚠️ [CORE] Direct Balance Fetch Failed (API Error): {err_str[:50]}")
                return self.capital
            print("❌ [CORE] Direct Balance Fetch Blocked (IP BAN). Switching to Proxy...")

        # 2. Try with proxy if direct fails with rate limit
        proxy = self.proxy_mgr.get_proxy()
        if proxy:
            self.exchange.proxies = proxy
            try:
                balance = self.exchange.fetch_balance()
                usdt_bal = float(balance.get('total', {}).get('USDT', 0))
                self.proxy_mgr.report_success(proxy)
                self.capital = usdt_bal
                return usdt_bal
            except Exception as e:
                print(f"⚠️ [CORE] Proxy Balance Fetch Failed: {str(e)[:50]}")
                self.proxy_mgr.report_failure(proxy)
        return self.capital

    def get_active_positions(self):
        """Phase 22/26: Returns live positions from Binance with Liquidation data (Resilient) - Uses V2 API"""
        if not self.is_live: return []
        
        proxy_attempts = [None] + [self.proxy_mgr.get_proxy() for _ in range(2)]
        
        for current_proxy in proxy_attempts:
            try:
                self.exchange.proxies = current_proxy if current_proxy else {}
                # Use V2 API endpoint instead of V1 (V1 returns 404)
                account = self.exchange.fapiPrivateV2GetAccount()
                positions = account.get('positions', [])
                active = []
                for p in positions:
                    if float(p.get('positionAmt', 0)) != 0:
                        active.append({
                            'symbol': p.get('symbol', 'UNKNOWN'),
                            'size': float(p.get('positionAmt', 0)),
                            'side': 'BUY' if float(p.get('positionAmt', 0)) > 0 else 'SELL',
                            'entry': float(p.get('entryPrice', 0)),
                            'mark_price': float(p.get('markPrice', 0)),
                            'liquidation_price': float(p.get('liquidationPrice', 0)),
                            'unrealized_pnl': float(p.get('unrealizedProfit', 0)),
                            'leverage': float(p.get('leverage', 1))
                        })
                return active
            except Exception as e:
                err = str(e).lower()
                if "418" in err or "429" in err or "451" in err or "ip" in err:
                    if current_proxy: self.proxy_mgr.report_failure(current_proxy)
                    continue
                print(f"❌ [GOD-MODE] Position Risk Error: {e}")
                break
        return []


    def get_compounded_risk(self):
        """Phase 10.2: Auto-Scaling Risk based on growth from initial capital"""
        ratio = self.config.get('compounding_ratio', 0.15)
        growth = self.capital - self.initial_capital
        
        # Handle string or float risk factor
        rf = self.config.get('risk_factor', 0.1)
        if rf == "AGGRESSIVE": current_risk = self.config.get('risk_factor_pct', 0.15)
        elif rf == "DEFENSIVE": current_risk = self.config.get('risk_factor_pct', 0.05)
        else: current_risk = float(rf) if isinstance(rf, (int, float)) else 0.1
        
        if growth > 0:
            extra_risk = (growth * ratio) / (self.capital + 1e-6)
            base_risk = min(current_risk + extra_risk, 0.25) # Cap base risk at 25%
            return base_risk
        return current_risk

    def rebalance_active_alpha(self, current_positions):
        """Phase 10.1: Dynamic Alpha Rebalancer
        Shifts exposure from 'Static' or 'Bleeding' assets to 'Trending' winners.
        This returns actionable rebalance directives.
        """
        if not current_positions or len(current_positions) < 2:
            return []

        leaders = []
        deadweight = []
        
        for pos in current_positions:
            pnl_pct = float(pos.get('unrealizedPnl', 0)) / (abs(float(pos.get('notional', 1))) + 1e-6) * 100
            
            if pnl_pct > 2.0:
                leaders.append(pos)
            elif -1.5 < pnl_pct < 0.5:
                deadweight.append(pos)
        
        actions = []
        if leaders and deadweight:
            target_deadweight = min(deadweight, key=lambda x: float(x.get('unrealizedPnl', 0)))
            target_leader = max(leaders, key=lambda x: float(x.get('unrealizedPnl', 0)))
            
            actions.append({
                "action": "REBALANCE_ALPHA",
                "trim_symbol": target_deadweight['symbol'],
                "boost_symbol": target_leader['symbol'],
                "reason": f"Sovereign Rebalancer: Shifting weight to {target_leader['symbol']} (+{pnl_pct:.1f}%)"
            })
            
        return actions

    def calculate_tp_grid(self, entry_price, side, atr):
        """Phase 11.1 / 26: Adaptive TP Grids 2.0
        Returns a 3-tier exit strategy that tightens during high volatility.
        """
        direction = 1 if side == "BUY" else -1
        
        # Phase 26: Tightening factor based on ATR/Price ratio (High Vol = Tighter TP)
        vol_ratio = atr / entry_price
        tighten = 0.85 if vol_ratio > 0.02 else 1.0 # Tighten by 15% if 1h ATR > 2%
        
        # 30/40/30 split across 3 targets
        grid = [
            {"target": entry_price + (atr * 0.8 * direction * tighten), "size_pct": 0.30},
            {"target": entry_price + (atr * 1.5 * direction * tighten), "size_pct": 0.40},
            {"target": entry_price + (atr * 3.0 * direction * tighten), "size_pct": 0.30}
        ]
        return grid

    def _get_current_atr(self, symbol):
        """Helper to get current ATR for a symbol without full analysis"""
        try:
            ohlcv = self.market_exch.fetch_ohlcv(symbol, timeframe='1h', limit=30)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['ATR'] = self._calculate_atr(df)
            return df['ATR'].iloc[-1]
        except: return 0.0

    def trail_liquidity_sl(self, symbol, current_price, current_sl, side, alpha_mode=False):
        """Phase 11.2: Trailing Profit Walls
        Moves SL behind the most persistent and massive orderbook walls.
        """
        try:
            atr = self._get_current_atr(symbol)
            if atr == 0: return current_sl
            
            # Phase 33: Tighter Trail for Alpha Moonshots
            multiplier = 0.5 if alpha_mode else 1.2
            
            if side == "BUY":
                potential_sl = current_price - (atr * multiplier)
                if potential_sl > current_sl:
                    return potential_sl
            elif side == "SELL":
                potential_sl = current_price + (atr * multiplier)
                if potential_sl < current_sl:
                    return potential_sl
        except: pass
        return current_sl


    def _detect_volatility_squeeze(self, df):
        """Phase 14.2: Volatility Expansion Sniper
        Detects if BB is inside KC (Bollinger Bands inside Keltner Channels)
        """
        try:
            last = df.iloc[-1]
            if last['BB_UP'] < last['KC_UP'] and last['BB_LOW'] > last['KC_LOW']:
                return True
        except: pass
        return False

    def _detect_liquidity_vacuum(self, symbol, fetch_callback=None):
        """Phase 15: Liquidity Vacuum Detection
        Identifies price ranges with significantly lower depth than average.
        """
        try:
            if fetch_callback:
                ob = fetch_callback(self.market_exch, 'fetch_order_book', symbol, limit=50)
            else:
                ob = self.market_exch.fetch_order_book(symbol, limit=50)
            bids = ob['bids']
            asks = ob['asks']

            
            # Calculate average depth per level
            avg_bid_depth = np.mean([b[1] for b in bids])
            avg_ask_depth = np.mean([a[1] for a in asks])
            
            # Check for "thin" spots near the spread
            near_bids = bids[:5]
            near_asks = asks[:5]
            
            bid_thin = any(b[1] < avg_bid_depth * 0.2 for b in near_bids)
            ask_thin = any(a[1] < avg_ask_depth * 0.2 for a in near_asks)
            
            return bid_thin or ask_thin
        except:
            return False

    def _calculate_tidal_shift(self, symbol):
        """Phase 15: Institutional Tidal Shift
        Tracks the volume of block trades (>$100k) over rolling windows to calculate RoC.
        """
        try:
            now = time.time()
            if not hasattr(self, 'tidal_store'):
                self.tidal_store = {} # {symbol: [(timestamp, volume), ...]}
            
            if symbol not in self.tidal_store:
                self.tidal_store[symbol] = []
            
            # 1. Fetch current block volume
            metrics = self._get_trade_based_metrics(symbol)
            current_vol = metrics.get('block_volume', 0)
            
            # 2. Store sample
            self.tidal_store[symbol].append((now, current_vol))
            
            # 3. Cleanup samples older than 10 minutes
            self.tidal_store[symbol] = [s for s in self.tidal_store[symbol] if (now - s[0]) < 600]
            
            if len(self.tidal_store[symbol]) < 5: return 0.0
            
            # 4. Calculate RoC between now and 5 minutes ago
            old_samples = [s for s in self.tidal_store[symbol] if (now - s[0]) > 300]
            if not old_samples: return 0.0
            
            old_vol = old_samples[-1][1]
            if old_vol == 0: return 100.0 if current_vol > 0 else 0.0
            
            roc = ((current_vol - old_vol) / old_vol) * 100
            return roc
        except:
            return 0.0

    def execute_live_order(self, symbol, side, amount_usd, price, tp, sl, turbo_scaling=False):
        """Phase 23/25: Execute real order on Binance Futures with Precision rounding"""
        print(f"🚀 [CORTEX_LIVE_V27] LIVE EXECUTION ATTEMPT: {symbol} | side={side} | price={price} | tp={tp} | sl={sl}")
        if not self.is_live: return {"status": "ERROR", "msg": "Not in Live Mode"}
        
        # Guardian 2.0 Check & Balance Sync
        if self.emergency_stop or not self.check_safeguards():
            return {"status": "ERROR", "msg": "GUARDIAN EMERGENCY STOP ACTIVE"}
            
        # Dynamic Margin Scaling: Phase 23 Sovereign Compounding (Kelly-Based)
        equity = self.get_live_balance()
        dynamic_amount = self._calculate_kelly_allocation(equity)
        
        # Phase 15: Turbo Scaling boost (Stacks with Kelly Evolution)
        if turbo_scaling:
            turbo_factor = self.config.get('turbo_scaling_factor', 1.5)
            dynamic_amount *= turbo_factor
            print(f"🔥 [TURBO] Scaling detected! Boosting Kelly allocation by {turbo_factor}x")
        else:
            # If not turbo, we still use the AI-optimized dynamic amount
            pass
        
        # Phase 6.0: Portfolio Guardian (Correlation Risk)
        if symbol != "BTC/USDT":
            # Fetch latest data for correlation
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)
                df_temp = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df_temp.set_index('timestamp', inplace=True)
                correlation = self._calculate_correlation(df_temp, "BTC/USDT")
                
                if correlation > 0.90:
                    print(f"🛡️ [GUARDIAN] High correlation detected ({correlation:.2f}). Halving position size.")
                    dynamic_amount *= 0.5
            except Exception as e:
                 print(f"⚠️ [GUARDIAN] Correlation check failed: {e}. Proceeding with caution.")
        
        if dynamic_amount < 110: dynamic_amount = 110 # Increased to 110 to satisfy $100 notional minimum on some pairs
        
        print(f"💰 [GOD-MODE] Final Allocation: ${dynamic_amount:.2f} (Equity: ${equity:.2f})")
        
        # God-Mode: Multi-Proxy Execution Loop (Direct First, then Proxy Fallback)
        proxy_attempts = [None] + [self.proxy_mgr.get_proxy() for _ in range(3)]
        
        print(f"DEBUG: Starting proxy loop with {len(proxy_attempts)} attempts")
        for current_proxy in proxy_attempts:
            if current_proxy:
                self.exchange.proxies = current_proxy
                print(f"📡 [GOD-MODE] Attempting execution via Proxy: {current_proxy['http']}")
            else:
                self.exchange.proxies = {} # Reset to direct
                print("📡 [GOD-MODE] Attempting Direct Execution...")

            try:
                # 1. Position Count Safeguard (Updated for V7.0 Scaling)
                print("DEBUG: Fetching balance info for positions")
                balance_info = self.exchange.fetch_balance()
                print("DEBUG: Balance info fetched successfully")
                open_positions = balance_info.get('info', {}).get('positions', [])
                active_pos_data = [p for p in open_positions if float(p.get('positionAmt', 0)) != 0]
                active_count = len(active_pos_data)
                max_pos = self.config.get("max_open_positions", 6)
                
                if active_count >= max_pos:
                    return {"status": "ERROR", "msg": f"MAX POSITIONS REACHED ({active_count}/{max_pos})"}

                # Phase 7.0: Portfolio Basket Correlation Lock
                if active_count > 1:
                    active_symbols = [p['symbol'] for p in active_pos_data]
                    if not self._verify_basket_sync(symbol, active_symbols):
                         return {"status": "ERROR", "msg": "BASKET SYNC DENIED: Correlated Over-Exposure"}

                # 2. Precision Prep & Dynamic Lev (Phase 15: Leverage Boost)
                base_lev = int(getattr(self, 'dynamic_leverage', self.config.get("leverage", 1)))
                if turbo_scaling:
                    max_turbo_lev = self.config.get('max_leverage_turbo', 3)
                    leverage = min(base_lev * 2, max_turbo_lev)
                    print(f"⚡ [TURBO] Boosting leverage from {base_lev}x to {leverage}x")
                else:
                    leverage = base_lev
                
                self.exchange.set_leverage(leverage, symbol)
                self.exchange.load_markets()
                
                raw_qty = (dynamic_amount * leverage) / price
                qty = float(self.exchange.amount_to_precision(symbol, raw_qty))
                tp_price = float(self.exchange.price_to_precision(symbol, tp))
                sl_price = float(self.exchange.price_to_precision(symbol, sl))
                
                print(f"📡 [GOD-MODE] Final Specs: {side} {qty} {symbol} @ {price} | TP: {tp_price} SL: {sl_price} | Lev: {leverage}x")
                
                # 3. Main Order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='MARKET',
                    side='buy' if side == "BUY" else 'sell',
                    amount=qty
                )
                
                # Success Logic
                if current_proxy: self.proxy_mgr.report_success(current_proxy)
                
                # 4. TP/SL
                try:
                    # CRITICAL: reduceOnly prevents phantom positions when position is closed externally
                    self.exchange.create_order(symbol, 'STOP_MARKET', 'sell' if side == "BUY" else 'buy', qty, params={'stopPrice': sl_price, 'reduceOnly': True})
                    self.exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell' if side == "BUY" else 'buy', qty, params={'stopPrice': tp_price, 'reduceOnly': True})
                    
                    # Phase 19: Telegram Notification
                    if self.telegram.is_active:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        loop.create_task(self.telegram.notify_trade(
                            symbol=symbol, side=side, qty=qty, 
                            price=price, tp=tp_price, sl=sl_price
                        ))
                except Exception as e:
                    print(f"⚠️ [TRADE CAP] TP/SL Orders failed: {e}")
                
                # Audit Trail for Sovereign Audit System
                audit_log = {
                    "symbol": symbol, "side": side, "qty": qty, "price": price,
                    "tp": tp, "sl": sl, "leverage": leverage, "audit_tag": "GOD-MODE",
                    "timestamp": datetime.now().isoformat()
                }
                self._save_audit(audit_log)
                
                print(f"✅ [GOD-MODE] {side} {symbol} Executed | Bypass: SUCCESS")
                return {"status": "SUCCESS", "id": order.get('id')}

            except Exception as e:
                err_msg = str(e)
                print(f"DEBUG: execute_live_order error: {err_msg}")
                if "418" in err_msg or "429" in err_msg or "1003" in err_msg:
                    print(f"❌ [GOD-MODE] Execution Blocked (" + err_msg + "). Switching Proxy...")
                    if current_proxy: self.proxy_mgr.report_failure(current_proxy)
                    time.sleep(1.0)
                    continue
                else:
                    return {"status": "ERROR", "msg": "Live Execution Failed: " + err_msg}
        
        return {"status": "ERROR", "msg": "ALL PROXIES EXHAUSTED. EXECUTION IMPOSSIBLE."}

    def update_live_stop_loss(self, symbol, side, new_sl):
        """Phase 11.2: Sync Trailing SL to Binance by canceling old stop and creating new one"""
        if not self.is_live: return
        try:
            # 1. Cancel previous stop loss orders for this symbol
            orders = self.exchange.fetch_open_orders(symbol)
            for o in orders:
                if o['type'] == 'STOP_MARKET':
                    self.exchange.cancel_order(o['id'], symbol)
            
            # 2. Get current position size for the stop order
            pos = self.get_active_positions()
            active = [p for p in pos if p['symbol'] == symbol]
            if not active: return
            qty = abs(active[0]['size'])
            
            # 3. Create new stop loss
            sl_price = float(self.exchange.price_to_precision(symbol, new_sl))
            res = self.exchange.create_order(
                symbol, 'STOP_MARKET', 
                'sell' if side == "BUY" else 'buy', 
                qty, 
                params={'stopPrice': sl_price}
            )
            print(f"📡 [GOD-MODE] SL UPDATED: {symbol} to ${sl_price}")
            return res
        except Exception as e:
            print(f"❌ [GOD-MODE] SL UPDATE ERROR: {e}")
            return None

    def sync_hindsight(self, trade_history):
        """Phase 20: Resolve PENDING predictions using actual simulator history"""
        updated = False
        for sym, history in self.hindsight.items():
            for entry in history:
                if entry.get('result') == 'PENDING':
                    # Find matching trade in history
                    # We match based on Symbol and proximity in time (within 12h of prediction)
                    pred_time = datetime.fromisoformat(entry['timestamp'])
                    
                    for trade in trade_history:
                        if trade['Symbol'] == sym:
                            trade_time = datetime.fromisoformat(trade['Time'])
                            if abs((trade_time - pred_time).total_seconds()) < 43200: # 12h window
                                entry['result'] = 'PROFIT' if trade['PnL'] > 0 else 'LOSS'
                                entry['actual_pnl'] = trade['PnL']
                                updated = True
                                break
        if updated:
            self.save_hindsight()

    def _record_prediction(self, symbol, signal, score, price):
        if signal == "WAIT": return
        if symbol not in self.hindsight: self.hindsight[symbol] = []
        self.hindsight[symbol].append({
            "timestamp": datetime.now().isoformat(),
            "signal": signal, "score": score, "price": price,
            "result": "PENDING"
        })
        # Keep only last 50
        self.hindsight[symbol] = self.hindsight[symbol][-50:]
        self.save_hindsight()

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))


    def _calculate_correlation(self, df, base_symbol="BTC/USDT"):
        """Phase 6.0: Pearson Correlation with Market Leader"""
        try:
            cache_key = f"corr_{df.index[-1]}_{base_symbol}"
            if cache_key in self._cache: return self._cache[cache_key]['val']

            # Fetch base data
            base_ohlcv = self.market_exch.fetch_ohlcv(base_symbol, timeframe='1h', limit=50)
            base_df = pd.DataFrame(base_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Align and correlate
            combined = pd.DataFrame({
                'asset': df['Close'].tail(50).values,
                'base': base_df['Close'].tail(50).values
            })
            corr = combined['asset'].corr(combined['base'])
            
            self._cache[cache_key] = {'val': corr, 'time': time.time()}
            return corr
        except: return 0.5

    def _calculate_basket_correlation(self, df, symbol):
        """Phase 7.0: Mean Correlation against the active portfolio basket"""
        try:
            if not self.is_live: return 0.2 # Neutral in simulation
            
            open_positions = self.exchange.fetch_balance().get('info', {}).get('positions', [])
            active_symbols = [p['symbol'] for p in open_positions if float(p.get('positionAmt', 0)) != 0 and p['symbol'] != symbol]
            
            if not active_symbols:
                # If no other positions, fallback to BTC correlation as beta-risk check
                return self._calculate_correlation(df, "BTC/USDT")
            
            correlations = []
            for s in active_symbols:
                correlations.append(self._calculate_correlation(df, s))
            
            return np.mean(correlations)
        except:
            return 0.5

    def _verify_basket_sync(self, symbol, active_symbols):
        """Phase 7.0: Safety check during execution to prevent correlated clusters"""
        try:
            if not active_symbols: return True
            # Fetch data for current symbol to check against active ones
            ohlcv = self.market_exch.fetch_ohlcv(symbol, timeframe='1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('timestamp', inplace=True)
            
            corr = self._calculate_basket_correlation(df, symbol)
            return corr < 0.85 # Tighter limit during hard-execution
        except:
            return True

    def _get_orderbook_depth(self, symbol, signal, fetch_callback=None):
        """Phase 8.0: Analyze top 20 levels for institutional Bid/Ask walls"""
        try:
            if fetch_callback:
                orderbook = fetch_callback(self.market_exch, 'fetch_order_book', symbol, limit=20)
            else:
                orderbook = self.market_exch.fetch_order_book(symbol, limit=20)

            bids = orderbook['bids']
            asks = orderbook['asks']
            
            # Sum volume within 1% of mid price
            mid_price = (bids[0][0] + asks[0][0]) / 2
            bid_vol = sum([b[1] for b in bids if b[0] >= mid_price * 0.99])
            ask_vol = sum([a[1] for a in asks if a[0] <= mid_price * 1.01])
            
            if signal == "BUY" or signal == "LONG":
                return bid_vol / (ask_vol + 1e-9)
            elif signal == "SELL" or signal == "SHORT":
                return ask_vol / (bid_vol + 1e-9)
            return 1.0
        except:
            return 1.0

    def _detect_whale_exhaustion(self, symbol):
        """Phase 6.0: Detect Climax using CVD Volatility"""
        try:
            metrics = self._get_trade_based_metrics(symbol)
            if metrics['block_trades'] > 100 and abs(metrics['cvd']) > 0.3:
                return True
            return False
        except:
            return False

    def check_global_volatility(self, btc_df):
        """Phase 8.0: Detect flash drops in BTC to trigger Crash Shield"""
        if btc_df.empty: return False
        
        # Calculate 5-minute percent change (assuming 1m or 5m data)
        lookback = 5
        if len(btc_df) < lookback: return False
        
        start_price = btc_df['Close'].iloc[-lookback]
        end_price = btc_df['Close'].iloc[-1]
        pct_change = (end_price / start_price) - 1
        
        threshold = self.config.get("crash_threshold", -0.02)
        if pct_change <= threshold:
            print(f"🚨 [CRASH SHIELD] Triggered! BTC drop detected: {pct_change:.2%}")
            return True
        return False

    def execute_delta_hedge(self, active_positions):
        """Phase 8.0: Open a market short on BTC to hedge alt-portfolio downside"""
        if self.hedge_active: return {"status": "SKIPPED", "msg": "Hedge already active"}
        
        try:
            total_alt_value = sum([float(p['positionAmt']) * float(p['entryPrice']) for p in active_positions if p['symbol'] != "BTC/USDT"])
            if total_alt_value <= 0: return {"status": "SKIPPED", "msg": "No alt exposure to hedge"}
            
            hedge_ratio = self.config.get("hedge_ratio", 0.5)
            hedge_amount = total_alt_value * hedge_ratio
            
            btc_price = self.market_exch.fetch_ticker("BTC/USDT")['last']
            btc_qty = hedge_amount / btc_price
            
            print(f"🛡️ [CRASH SHIELD] Executing BTC Short Hedge: ${hedge_amount:.2f} ({btc_qty:.4f} BTC)")
            
            # Using market order for hedge speed
            order = self.exchange.create_order(
                symbol="BTC/USDT",
                type='MARKET',
                side='sell',
                amount=self.exchange.amount_to_precision("BTC/USDT", btc_qty)
            )
            
            self.hedge_active = True
            return {"status": "SUCCESS", "order": order}
        except Exception as e:
            return {"status": "ERROR", "msg": str(e)}

    def _detect_liquidity_sweep(self, df):
        """Phase 21: PDH/PDL Sweep Detection"""
        if len(df) < 24: return "NONE"
        
        # Simplified PDH/PDL (last 24 bars)
        pdh = df['High'].iloc[-25:-1].max()
        pdl = df['Low'].iloc[-25:-1].min()
        
        curr_high = df['High'].iloc[-1]
        curr_low = df['Low'].iloc[-1]
        curr_close = df['Close'].iloc[-1]
        
        # Bearish Sweep: Price went above PDH but closed below it
        if curr_high > pdh and curr_close < pdh:
            return "BEARISH_SWEEP"
        
        # Bullish Sweep: Price went below PDL but closed above it
        if curr_low < pdl and curr_close > pdl:
            return "BULLISH_SWEEP"
            
        return "NONE"

    def _get_liquidity_depth(self, symbol):
        """Phase 5.0: Institutional Order Book Analysis"""
        try:
            # Use market_exch (Spot) to avoid rate limits/bans
            ob = self.market_exch.fetch_order_book(symbol, limit=20)
            bids = sum(float(b[1]) for b in ob['bids'][:15])
            asks = sum(float(a[1]) for a in ob['asks'][:15])
            
            if (bids + asks) == 0: return {"depth": 0, "imbalance": 0}
            
            imbalance = (bids - asks) / (bids + asks)
            depth_score = (bids + asks) / 2
            
            return {"depth": depth_score, "imbalance": imbalance}
        except: 
            return {"depth": 5.0, "imbalance": 0.0}

    def _detect_volatility_climax(self, df):
        """God-Mode: Detect Volume/Volatility Exhaustion"""
        if len(df) < 20: return "NONE"
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['Volume'].iloc[-1]
        
        avg_range = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        curr_range = df['High'].iloc[-1] - df['Low'].iloc[-1]
        
        if curr_vol > avg_vol * 3 and curr_range > avg_range * 2.5:
            return "CLIMAX"
        return "NONE"

    def _get_memory_booster(self, symbol):
        """Phase 22: Retrieve memory for symbol-specific performance"""
        try:
            if os.path.exists("sovereign_knowledge.json"):
                with open("sovereign_knowledge.json", 'r') as f:
                    knowledge = json.load(f)
                
                insights = knowledge.get('insights', [])
                if not insights: return 0
                
                latest = insights[-1]
                if symbol in latest.get('top_performing_assets', []):
                    return 15 # Significant boost for winning assets
            return 0
        except: return 0

    def _get_learned_constraints(self):
        """Phase 8.0: Load auto-generated constraints from the Evolution Sentinel"""
        try:
            if os.path.exists("learned_constraints.json"):
                with open("learned_constraints.json", 'r') as f:
                    return json.load(f)
        except: return {}

    # --- Phase 34: Sovereign Harvest (Spot Optimization) ---
    def harvest_spot_profits(self):
        """Monitor Spot BTC and harvest profits into Futures capital"""
        if not self.is_live: return {"status": "SKIPPED", "msg": "Harvest only active in Live Mode"}

        try:
            # 1. Fetch Spot BTC Balance
            bal = self.market_exch.fetch_balance()
            btc_total = bal['total'].get('BTC', 0)
            if btc_total < 0.001: return {"status": "SKIPPED", "msg": "Insufficient BTC to harvest"}

            # 2. Check Price Spike (Current vs 24h Avg)
            ohlcv = self.market_exch.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=24)
            avg_24h = np.mean([o[4] for o in ohlcv])
            ticker = self.market_exch.fetch_ticker('BTC/USDT')
            curr_price = ticker['last']

            # Trigger: 2% Spike
            if curr_price > avg_24h * 1.02:
                harvest_qty = btc_total * 0.01 # Sell 1% (Conservative Harvest)
                print(f"💰 [HARVEST] BTC Spike Detected! Selling {harvest_qty:.6f} BTC at ${curr_price:.2f}")
                
                # Execute Spot Market Sell
                order = self.market_exch.create_market_sell_order('BTC/USDT', harvest_qty)
                
                # 3. Transfer USDT from Spot to Futures
                # ccxt transfer method: from, to, currency, amount
                time.sleep(2) # Wait for settlement
                usdt_gain = float(order.get('cost', 0))
                if usdt_gain > 5:
                    print(f"🚀 [HARVEST] Transferring ${usdt_gain:.2f} to Futures account...")
                    self.market_exch.transfer('USDT', usdt_gain, 'spot', 'future')
                    
                return {"status": "SUCCESS", "msg": f"Harvested ${usdt_gain:.2f} from BTC spike."}
            
            return {"status": "IDLE", "msg": "No significant spike detected."}
        except Exception as e:
            return {"status": "ERROR", "msg": str(e)}

    def manage_dual_investment(self):
        """Recommend Sell-High targets for Binance Earn"""
        try:
            ticker = self.market_exch.fetch_ticker('BTC/USDT')
            curr_price = ticker['last']
            target = curr_price * 1.03
            return {
                "recommendation": f"Dual Investment 'Sell High' @ ${target:,.0f}",
                "expected_yield": "approx. 40-90% APR",
                "current_price": curr_price
            }
        except: return {}

    def daily_capital_sentinel(self, base_usd=10000):
        """Phase 37: Secures daily profits by maintaining a constant capital baseline"""
        if not self.is_live: return {"status": "SKIPPED", "msg": "Daily Harvest only active in Live Mode"}

        try:
            # 1. Fetch current BTC position and price
            bal = self.market_exch.fetch_balance()
            btc_amt = bal['total'].get('BTC', 0)
            
            ticker = self.market_exch.fetch_ticker('BTC/USDT')
            price = ticker['last']
            
            current_value = btc_amt * price
            buffer = 50 # Minimum $50 profit to trigger a harvest
            
            if current_value > base_usd + buffer:
                surplus_usd = current_value - base_usd
                qty_to_sell = surplus_usd / price
                
                print(f"💰 [DAILY HARVEST] Surplus Detected: ${surplus_usd:.2f}. Selling {qty_to_sell:.6f} BTC...")
                
                # 2. Execute Market Sell on Spot
                order = self.market_exch.create_market_sell_order('BTC/USDT', qty_to_sell)
                
                return {
                    "status": "SUCCESS", 
                    "msg": f"Locked in ${surplus_usd:.2f} profit. New BTC Value: ${base_usd:.2f}",
                    "usdt_gain": surplus_usd
                }
            
            return {"status": "IDLE", "msg": f"Capital at ${current_value:.2f} (under ${base_usd + buffer} threshold)."}
        except Exception as e:
            return {"status": "ERROR", "msg": str(e)}

    def _check_liquidity_persistence(self, symbol, current_bids, current_asks, fetch_callback=None):

        """Phase 10.0: Anti-Spoofing Engine
        Tracks if the liquidity walls are stable or 'flickering'
        """
        now = time.time()
        if symbol not in self.persistence_store:
            self.persistence_store[symbol] = {'bids': [], 'asks': [], 'times': []}
        
        store = self.persistence_store[symbol]
        store['bids'].append(current_bids)
        store['asks'].append(current_asks)
        store['times'].append(now)
        
        # Keep last 10 samples
        if len(store['times']) > 10:
            store['bids'].pop(0)
            store['asks'].pop(0)
            store['times'].pop(0)
            
        if len(store['times']) < 3: return 1.0 # Not enough history
        
        # Calculate Coefficient of Variation (Stability)
        bid_cv = np.std(store['bids']) / (np.mean(store['bids']) + 1e-6)
        ask_cv = np.std(store['asks']) / (np.mean(store['asks']) + 1e-6)
        
        # If instability is high (>50%), it's a flickering 'Fake Wall'
        if bid_cv > 0.50 or ask_cv > 0.50:
            return 0.3 # 70% reduction in trust for this wall
            
        return 1.0

    def _update_cortex_evolution(self, win):
        """Phase 23: Update win rate history and adjust convictin thresholds"""
        self.win_rate_history.append(1 if win else 0)
        self.win_rate_history = self.win_rate_history[-20:] # Last 20 trades
        
        if len(self.win_rate_history) >= 5:
            wr = sum(self.win_rate_history) / len(self.win_rate_history)
            # Adjust multiplier: 0.8 (loss streak) to 1.5 (win streak)
            self.cortex_multiplier = 1.0 + (wr - 0.5) * 1.0
            self.cortex_multiplier = max(0.7, min(1.8, self.cortex_multiplier))
            
            # Adjust dynamic barrier: 85 (defensive) to 55 (aggressive / selective)
            self.dynamic_barrier = 80 - (wr * 30) # High WR = Lower Barrier
            self.dynamic_barrier = max(55.0, min(85.0, self.dynamic_barrier))
            
            print(f"🧠 [CORTEX_EVOLUTION] Win Rate: {wr:.2%} | Multiplier: {self.cortex_multiplier:.2f} | Barrier: {self.dynamic_barrier:.1f}")

    def _calculate_kelly_allocation(self, equity):
        """Phase 23: Exponential compounding using simplified Kelly Criterion"""
        if len(self.win_rate_history) < 5:
            return 100.0 # Standard fixed size until data is gathered
            
        wr = sum(self.win_rate_history) / len(self.win_rate_history)
        win_size = 1.5 # Avg win is 1.5x avg loss (Target R:R)
        
        # Kelly % = (bp - q) / b (where b=odds, p=win_prob, q=loss_prob)
        kelly_fraction = (win_size * wr - (1 - wr)) / win_size
        kelly_fraction = max(0.02, min(0.25, kelly_fraction)) # Cap at 25% of equity per trade
        
        allocation = equity * kelly_fraction * self.cortex_multiplier
        # Safely cap at 10x current session start to prevent accidental over-leveraging
        allocation = min(allocation, self.session_start_equity * 5) 
        
        if allocation < 11: allocation = 11
        return allocation

    def _detect_liquidity_trap(self, symbol, fetch_callback=None):
        """Phase 23: Detect institutional 'Spoofing' by tracking wall instability"""
        try:
            if fetch_callback:
                ob = fetch_callback({'options': {'defaultType': 'future'}}, 'fetch_order_book', symbol, limit=20)
            else:
                ob = self.exchange.fetch_order_book(symbol, limit=20)
            bids = ob['bids']
            asks = ob['asks']
            
            # Wall Instability Check (already has a helper, let's enhance it)
            stability = self._check_liquidity_persistence(symbol, sum([b[1] for b in bids]), sum([a[1] for a in asks]))
            
            if stability < 0.5:
                print(f"⚠️ [CORTEX_INTEL] Liquidity Trap Detected in {symbol}. Instability: {stability:.2f}")
                return True
            return False
        except: return False

    def check_liquidation_guard(self, active_positions):
        """Phase 26: Emergency exit if mark price is near liquidation price"""
        if not self.is_live or not active_positions: return
        
        for pos in active_positions:
            mark = pos['mark_price']
            liq = pos['liquidation_price']
            if liq == 0: continue
            
            # Distance to liquidation
            dist = abs((mark - liq) / mark)
            if dist < 0.10: # 10% safety margin
                print(f"🚨 [LIQUIDATION_GUARD] Warning: {pos['symbol']} distance to liq: {dist:.2%}. TRIGGERING EMERGENCY EXIT.")
                self.emergency_market_close(pos['symbol'])

    def emergency_market_close(self, symbol):
        """Phase 26: Market close a specific position and cancel its SL/TP"""
        try:
            # 1. Fetch position to get exact qty
            pos = self.get_active_positions()
            active = [p for p in pos if p['symbol'] == symbol]
            if not active: return
            qty = abs(active[0]['size'])
            side = 'sell' if active[0]['size'] > 0 else 'buy'
            
            print(f"🛡️ [GOD-MODE] EMERGENCY CLOSE: Closing {qty} {symbol} via Market Order")
            
            # 2. Cancel ALL orders for this symbol first
            self.exchange.cancel_all_orders(symbol)
            
            # 3. Market Close Order
            self.exchange.create_order(symbol, 'MARKET', side, qty, params={'reduceOnly': True})
            
            # 4. Notify Telegram
            if self.telegram.is_active:
                import asyncio
                loop = asyncio.get_event_loop()
                loop.create_task(self.telegram.send_message(f"🚨 **EMERGENCY EXIT**: {symbol} closed by Liquidation Guard at market price."))
            
            return True
        except Exception as e:
            print(f"❌ [GOD-MODE] Emergency Close Failed for {symbol}: {e}")
            return False

    def _save_audit(self, audit_data):
        """Phase 25: Save granular trade metadata for Sovereign Audit System"""
        try:
            audit_file = "sovereign_audit.json"
            audits = []
            if os.path.exists(audit_file):
                with open(audit_file, 'r') as f:
                    audits = json.load(f)
            
            audits.append(audit_data)
            audits = audits[-500:]
            
            with open(audit_file, 'w') as f:
                json.dump(audits, f, indent=4)
        except Exception as e:
            print(f"Failed to log Sovereign Audit: {e}")

if __name__ == "__main__":
    bot = UltimateV17Bot()
    print("V17 Sovereign Bot Active.")
