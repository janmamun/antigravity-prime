
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import ccxt
import os
import json
from neural_oracle import NeuralOracle
from sentiment_scanner import SentimentScanner

warnings.filterwarnings('ignore')

class UltimateV15Bot:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_equity = []
        
        self.config_file = "bot_config.json"
        self.load_config()
        
        # Super-Intelligence Modules
        self.oracle = NeuralOracle()
        self.sentiment = SentimentScanner()
        
        # CCXT Binance for Funding Rates & Data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "rsi_oversold": 35,
                "rsi_overbought": 70,
                "min_score": 65,
                "risk_factor": 0.1,  # 10% equity risk
                "leverage": 3,
                "dca_enabled": True,
                "dca_max_entries": 3
            }

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

        tr = self._calculate_atr(df, period=1) # TR is max(H-L, H-Cp, L-Cp)
        # Simplified TR for DM
        plus_di = 100 * (plus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-6))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-6))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        adx = dx.rolling(period).mean()
        return adx

    def get_funding_rate(self, symbol):
        """Fetch real-time funding rate from Binance"""
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return funding['fundingRate']
        except:
            return 0.0001 # Default neutral

    def analyze_snapshot(self, df, symbol="UNKNOWN"):
        """
        Phase 15 Core Signal Engine
        Returns a decision dictionary.
        """
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
        
        # 3. Funding Intelligence
        funding_rate = self.get_funding_rate(symbol)
        funding_bias = 0
        if funding_rate <= 0.0001: funding_bias = 10 # Longs are cheap
        if funding_rate < -0.0001: funding_bias = 20 # Shorts are paying longs (Extreme)
        
        # 4. Signal Scoring
        score = 0
        reasons = []
        
        # Magic MA Edge
        if df['MA19'].iloc[-1] > df['MA40'].iloc[-1]:
            score += 20
            reasons.append("Magic 19 Cross")
        
        # Trend / Range Logic
        if regime == "TREND":
            if current_price > df['MA19'].iloc[-1]:
                score += 15
                reasons.append("Trend Following (Above MA19)")
        elif regime == "RANGE":
            if df['RSI'].iloc[-1] < 35:
                score += 20
                reasons.append("Range Low (RSI Oversold)")
            if df['Low'].iloc[-1] < df['Low'].iloc[-2] and current_price > df['Low'].iloc[-1]:
                score += 10
                reasons.append("Bear Trap Detected")

        # Core Technicals
        if df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1] * 2:
            score += 10
            reasons.append("Volume Surge")
        if current_price > df['SMA200'].iloc[-1]:
            score += 10
            reasons.append("Above SMA200")
            
        # 5. Super-Intelligence Boost
        oracle_pred = self.oracle.predict(df)
        if oracle_pred == "BULLISH":
            score += 10
            reasons.append("Neural Oracle: Bullish")
        elif oracle_pred == "STRONG_BULLISH":
            score += 15
            reasons.append("Neural Oracle: Strong Bullish")
            
        sentiment_data = self.sentiment.analyze_sentiment(df)
        if sentiment_data['label'] == "EXTREME FEAR":
            score += 10
            reasons.append("Sentiment: Extreme Fear (Contrarian Buy)")
            
        # Funding Bonus
        score += funding_bias
        if funding_bias > 0: reasons.append(f"Funding Advantage (+{funding_bias})")
        
        # Final Decision
        signal = "WAIT"
        min_score = self.config.get("min_score", 65)
        
        if score >= min_score:
            signal = "BUY"
        
        # Position Detail
        tp = current_price + (adr * 2)
        sl = current_price - (adr * 1.5) # Dynamic ATR SL
        
        # Hard Caps
        if (tp/current_price - 1) > 0.20: tp = current_price * 1.20
        if (1 - sl/current_price) > 0.08: sl = current_price * 0.92

        return {
            "symbol": symbol,
            "signal": signal,
            "score": score,
            "price": current_price,
            "current_price": current_price,
            "entry": current_price,
            "tp": tp,
            "sl": sl,
            "atr": adr,
            "regime": regime,
            "funding": funding_rate,
            "reasons": reasons
        }

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    bot = UltimateV15Bot()
    print("V15 Bot Initialized. Infinite Intelligence Active.")
