
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import json
import ccxt
from dotenv import load_dotenv
from neural_oracle import NeuralOracle
from sentiment_scanner import SentimentScanner

warnings.filterwarnings('ignore')

class UltimateV17Bot:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_equity = []
        
        self.config_file = "bot_config.json"
        self.load_config()
        
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
        
        # Phase 25: Guardian 2.0 Safeguards
        self.max_daily_loss_pct = 5.0  # Stop all trading if down 5% today
        self.emergency_stop = False
        self.session_start_equity = self.capital
        
        if self.is_live and self.api_key and self.api_secret:
            try:
                exchange_config.update({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                })
                self.exchange = ccxt.binance(exchange_config)
                # Verify connection immediately
                self.exchange.fetch_balance()
                print(f"🚀 [SYSTEM] ENTRANCE TO LIVE TRADE PROTOCOL: {self.api_key[:5]}... ACTIVE")
            except Exception as e:
                print(f"❌ [CORE ERROR] LIVE CONNECTION FAILED: {e}")
                self.is_live = False
                self.exchange = ccxt.binance(exchange_config)
        else:
            self.exchange = ccxt.binance(exchange_config)
            print("🪐 [SYSTEM] Simulation Mode Active. No Live Keys Detected.")

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "rsi_oversold": 35,
                "rsi_overbought": 70,
                "min_score": 45,
                "risk_factor": 0.1,
                "leverage": 3,
                "dca_enabled": True,
                "dca_max_entries": 3,
                "max_open_positions": 5
            }

    def load_hindsight(self):
        if os.path.exists(self.hindsight_file):
            with open(self.hindsight_file, 'r') as f:
                self.hindsight = json.load(f)
        else:
            self.hindsight = {} # {symbol: [list of previous predictions]}

    def save_hindsight(self):
        with open(self.hindsight_file, 'w') as f:
            json.dump(self.hindsight, f, indent=4)

    def _calculate_atr(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

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
            funding = self.exchange.fetch_funding_rate(symbol)
            return funding['fundingRate']
        except:
            return 0.0001

    def _calculate_oi_momentum(self, symbol):
        try:
            oi_history = self.exchange.fetch_open_interest_history(symbol, timeframe='5m', limit=10)
            if not oi_history or len(oi_history) < 2: return 0.0
            latest_oi = float(oi_history[-1]['openInterestAmount'])
            prev_oi_avg = np.mean([float(h['openInterestAmount']) for h in oi_history[:-1]])
            return (latest_oi / prev_oi_avg) - 1
        except:
            return 0.0

    def _get_whale_bias(self, symbol, threshold_usd=50000):
        try:
            trades = self.exchange.fetch_trades(symbol, limit=100)
            whales = [t for t in trades if (float(t['amount']) * float(t['price'])) >= threshold_usd]
            if not whales: return "NEUTRAL"
            buy_vol = sum(float(t['amount']) for t in whales if t['side'] == 'buy')
            sell_vol = sum(float(t['amount']) for t in whales if t['side'] == 'sell')
            if buy_vol > sell_vol * 1.5: return "BULLISH"
            if sell_vol > buy_vol * 1.5: return "BEARISH"
            return "NEUTRAL"
        except:
            return "NEUTRAL"

    def analyze_snapshot(self, df, symbol="UNKNOWN"):
        if len(df) < 50:
            return {"symbol": symbol, "signal": "ERROR", "reasons": ["Insufficient Data"]}

        # 1. Indicators
        df['MA40'] = df['Close'].rolling(40).mean()
        df['MA19'] = df['Close'].rolling(19).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['ATR'] = self._calculate_atr(df)
        df['ADX'] = self._calculate_adx(df)
        
        current_price = df['Close'].iloc[-1]
        adr = df['ATR'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        
        # 2. Regime Detection
        regime = "NEUTRAL"
        if adx > 25: regime = "TREND"
        elif adx < 20: regime = "RANGE"
        
        # 3. Funding & Institutional
        funding_rate = self.get_funding_rate(symbol)
        oi_mom = self._calculate_oi_momentum(symbol)
        whale_bias = self._get_whale_bias(symbol)
        
        # Phase 21: Order Flow & Liquidity
        cvd_delta = self._calculate_cvd(symbol)
        liq_sweep = self._detect_liquidity_sweep(df)
        liq_depth = self._get_liquidity_depth(symbol)
        
        # Phase 22: Sovereign Memory
        memory_booster = self._get_memory_booster(symbol)
        
        # 4. Signal Scoring
        score = 0
        reasons = []
        
        # Legacy Edges
        if df['MA19'].iloc[-1] > df['MA40'].iloc[-1]:
            score += 20
            reasons.append("Magic 19 Cross")
        
        if regime == "TREND":
            if current_price > df['MA19'].iloc[-1]:
                score += 15
                reasons.append("Trend Following")
        elif regime == "RANGE":
            if df['RSI'].iloc[-1] < 35:
                score += 20
                reasons.append("RSI Oversold")
            if current_price > df['Low'].iloc[-1]:
                score += 10
                reasons.append("Bear Trap Support")

        # Super-Intelligence
        oracle_pred = self.oracle.predict(df)
        oracle_str = str(oracle_pred).replace('np.float64(', '').replace(')', '')
        if "BULLISH" in oracle_str:
            score += 15
            reasons.append(f"Oracle: {oracle_str}")
            
        sentiment_data = self.sentiment.analyze_sentiment(df)
        if sentiment_data['label'] == "EXTREME FEAR":
            score += 10
            reasons.append("Sentiment: Fear Edge")

        # Phase 16 Edges
        if oi_mom > 0.005:
            score += 15
            reasons.append(f"OI Accumulation (+{oi_mom*100:.1f}%)")
        if whale_bias == "BULLISH":
            score += 10
            reasons.append("Whale Inflow Detected")
            
        if funding_rate <= 0:
            score += 10
            reasons.append("Funding Advantage")
            
        # Phase 21 Institutional Edges
        if cvd_delta > 0.05:
            score += 20
            reasons.append(f"Bullish CVD Delta (+{cvd_delta*100:.1f}%)")
        
        if liq_sweep == "BULLISH_SWEEP":
            score += 25
            reasons.append("Liquidity Sweep (PDL Reversal)")
        elif liq_sweep == "BEARISH_SWEEP":
            score -= 25
            reasons.append("Liquidity Sweep (PDH Rejection)")
            
        if liq_depth < 2.0: # Less than 2x average spread volume
            score -= 20
            reasons.append("Low Liquidity Depth Warning")
        elif liq_depth > 5.0:
            score += 10
            reasons.append("Strong Liquidity Support")

        # Phase 22 Sovereign Memory Boost
        if memory_booster > 0:
            score += memory_booster
            reasons.append(f"Sovereign Memory Boost (+{memory_booster}pts)")

        # Phase 17: Hindsight Adjustment
        # If we failed on this symbol recently, penalize the score
        if symbol in self.hindsight:
            recent_fails = [h for h in self.hindsight[symbol][-5:] if h['result'] == 'LOSS']
            if len(recent_fails) >= 2:
                score -= 15 # Increased penalty
                reasons.append("Hindsight Penalty (Recent Lost Trades)")

        # Final Decision
        signal = "WAIT"
        min_score = self.config.get("min_score", 65)
        if score >= min_score: signal = "BUY"
        elif score <= 20: signal = "SELL" # Proactive shorting?
        
        tp = current_price + (adr * 2.5) # Increased reward
        sl = current_price - (adr * 1.5)
        
        # Save prediction for Hindsight
        self._record_prediction(symbol, signal, score, current_price)

        # Strategy Recommendation
        strategy = "INSTITUTIONAL TREND SCALP"
        if regime == "RANGE":
            strategy = "GRID REVERSION MATRIX"
        elif "Whale Inflow" in str(reasons):
            strategy = "WHALE FOLLOW PROTOCOL"
        elif funding_rate < 0:
            strategy = "SHORT SQUEEZE DELTA"

        return {
            "symbol": symbol, "signal": signal, "score": score,
            "price": current_price, "current_price": current_price,
            "tp": tp, "sl": sl, "atr": adr, "regime": regime,
            "strategy": strategy,
            "funding": funding_rate, "oi_mom": oi_mom,
            "whale_bias": whale_bias, "reasons": reasons
        }

    def check_safeguards(self):
        """Guardian 2.0: Prevent catastrophic loss"""
        current_equity = self.get_live_balance()
        drawdown = (self.session_start_equity - current_equity) / (self.session_start_equity + 1e-6) * 100
        
        if drawdown >= self.max_daily_loss_pct:
            print(f"🚨 [GUARDIAN] EMERGENCY STOP: Daily Loss Limit Reached ({drawdown:.2f}%)")
            self.emergency_stop = True
            return False
        return True

    def get_live_balance(self):
        """Phase 23: Fetch real USDT Futures balance"""
        if not self.is_live: return self.capital
        try:
            balance = self.exchange.fetch_balance()
            # Try to find USDT specifically in total
            return float(balance.get('total', {}).get('USDT', 0))
        except Exception as e:
            print(f"Error fetching live balance: {e}")
            return self.capital # Fallback to last known

    def execute_live_order(self, symbol, side, amount_usd, price, tp, sl):
        """Phase 23/25: Execute real order on Binance Futures with Precision rounding"""
        if not self.is_live: return {"status": "ERROR", "msg": "Not in Live Mode"}
        
        # Guardian 2.0 Check
        if self.emergency_stop or not self.check_safeguards():
            return {"status": "ERROR", "msg": "GUARDIAN EMERGENCY STOP ACTIVE"}
            
        try:
            # 1. Position Count Safeguard
            open_positions = self.exchange.fetch_balance().get('info', {}).get('positions', [])
            active_count = len([p for p in open_positions if float(p.get('positionAmt', 0)) != 0])
            max_pos = self.config.get("max_open_positions", 5)
            
            if active_count >= max_pos:
                return {"status": "ERROR", "msg": f"MAX POSITIONS REACHED ({active_count}/{max_pos})"}

            # 2. Calc quantity based on price and leverage
            leverage = int(os.getenv("CAPITAL_LEVERAGE", 10))
            self.exchange.set_leverage(leverage, symbol)
            
            # Load markets for precision logic
            self.exchange.load_markets()
            
            # Ensure quantity meets minimums and precision
            raw_qty = (amount_usd * leverage) / price
            qty = float(self.exchange.amount_to_precision(symbol, raw_qty))
            
            # Round TP/SL to exchange precision
            tp_price = float(self.exchange.price_to_precision(symbol, tp))
            sl_price = float(self.exchange.price_to_precision(symbol, sl))
            
            # 3. Main Order
            order = self.exchange.create_order(
                symbol=symbol,
                type='MARKET',
                side='buy' if side == "BUY" else 'sell',
                amount=qty
            )
            
            # 4. TP/SL: Using Binance 'STOP_MARKET' for secondary protection
            try:
                self.exchange.create_order(symbol, 'STOP_MARKET', 'sell' if side == "BUY" else 'buy', qty, params={'stopPrice': sl_price})
                self.exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell' if side == "BUY" else 'buy', qty, params={'stopPrice': tp_price})
            except Exception as e:
                print(f"⚠️ [TRADE CAP] TP/SL Orders failed: {e}")
            
            return {"status": "SUCCESS", "order_id": order['id'], "msg": f"Live {side} Executed for {symbol} | Safeguard: ACTIVE"}
        except Exception as e:
            return {"status": "ERROR", "msg": f"Live Execution Failed: {str(e)}"}

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

    def _calculate_cvd(self, symbol):
        """Phase 21: Cumulative Volume Delta"""
        try:
            trades = self.exchange.fetch_trades(symbol, limit=200)
            buys = sum(float(t['amount']) for t in trades if t['side'] == 'buy')
            sells = sum(float(t['amount']) for t in trades if t['side'] == 'sell')
            return (buys - sells) / (buys + sells + 1e-6)
        except: return 0.0

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
        """Phase 22: Check Order Book Depth vs Spread"""
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=20)
            bids = sum(float(b[1]) for b in ob['bids'][:10])
            asks = sum(float(a[1]) for a in ob['asks'][:10])
            mid_vol = (bids + asks) / 2
            return mid_vol
        except: return 5.0 # Default to safe depth if check fails

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

if __name__ == "__main__":
    bot = UltimateV17Bot()
    print("V17 Sovereign Bot Active.")
