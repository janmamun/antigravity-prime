
import pandas as pd
import numpy as np
import ta

class GodModeStrategy:
    def __init__(self):
        pass

    def analyze(self, df):
        """
        Run the 'God Mode' Analysis on a single dataframe.
        Expects: Open, High, Low, Close, Volume
        Returns: Dict with Signal, TP, SL, Confidence, Reasoning
        """
        if len(df) < 50:
            return None

        # 1. Feature Engineering
        # ----------------------
        close = df['Close']
        
        # Magic Numbers
        ma40 = close.rolling(40).mean()
        ma19 = close.rolling(19).mean()
        ma7 = close.rolling(7).mean()
        
        # Volatility & Momentum
        atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], close, window=14).average_true_range()
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd = ta.trend.MACD(close)
        
        # Current Values
        current_price = close.iloc[-1]
        cur_ma40 = ma40.iloc[-1]
        cur_ma19 = ma19.iloc[-1]
        cur_atr = atr.iloc[-1]
        cur_rsi = rsi.iloc[-1]
        
        # 2. Logic Core (The "Unbeatable" Rules)
        # --------------------------------------
        score = 0
        reasons = []
        
        # Trend Filter (Magic 40 - The "Trial" Period)
        trend = "NEUTRAL"
        if current_price > cur_ma40:
            score += 30
            trend = "BULLISH"
            reasons.append("Price > Magic 40 Trend")
        else:
            score -= 30
            trend = "BEARISH"
            reasons.append("Price < Magic 40 Trend")
            
        # Momentum Filter (Magic 19 - The "Miracle")
        if current_price > cur_ma19:
            score += 20
            reasons.append("Price > Magic 19 Momentum")
        else:
            score -= 20
            
        # RSI Optimization
        if cur_rsi < 30:
            score += 15
            reasons.append("RSI Oversold (Bounce Likely)")
        elif cur_rsi > 70:
            score -= 15
            reasons.append("RSI Overbought (Pallback Likely)")
            
        # Golden Cross check (7 crossing 19)
        prev_ma7 = ma7.iloc[-2]
        prev_ma19 = ma19.iloc[-2]
        if ma7.iloc[-1] > ma19.iloc[-1] and prev_ma7 <= prev_ma19:
            score += 25
            reasons.append("God Cross (7/19 MA)")
            
        # 3. Decision & Trade Management
        # ------------------------------
        signal = "WAIT"
        tp = 0.0
        sl = 0.0
        
        # Decision Thresholds
        if score >= 60:
            signal = "BUY"
            # Smart TP/SL using ATR (Volatility)
            # Stop Loss = 2x ATR below price (Give it room to breathe)
            # Take Profit = 4x ATR above price (2:1 Reward ratio)
            sl = current_price - (cur_atr * 2.0)
            tp = current_price + (cur_atr * 4.0)
            
        elif score <= -60:
            signal = "SELL"
            sl = current_price + (cur_atr * 2.0)
            tp = current_price - (cur_atr * 4.0)
            
        return {
            "signal": signal,
            "score": score,
            "current_price": current_price,
            "entry": current_price,
            "tp": tp,
            "sl": sl,
            "atr": cur_atr,
            "trend": trend,
            "reasons": reasons
        }
