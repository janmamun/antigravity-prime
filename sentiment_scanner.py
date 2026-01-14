import pandas as pd
import numpy as np
import requests
import json
import time

class SentimentScanner:
    def __init__(self):
        pass

    def analyze_sentiment(self, df):
        """
        Calculate Market Fear & Greed Index (Technical Proxy)
        0 (Extreme Fear) -> 100 (Extreme Greed)
        """
        try:
            if len(df) < 50: return {"score": 50, "label": "NEUTRAL"}
            
            # 1. Momentum Component (RSI) - 30% Weight
            rsi = self.calculate_rsi(df)
            rsi_val = rsi.iloc[-1]
            
            # 2. Volatility Component (Bollinger Width) - 30% Weight
            df['SMA'] = df['Close'].rolling(20).mean()
            df['STD'] = df['Close'].rolling(20).std()
            df['BB_Width'] = (df['SMA'] + 2*df['STD']) - (df['SMA'] - 2*df['STD'])
            vol_ma = df['BB_Width'].rolling(50).mean()
            vol_now = df['BB_Width'].iloc[-1]
            
            # High Volatility = Fear usually, but in crypto pumps it's Greed?
            # Let's say High Vol + Uptrend = Greed, High Vol + Downtrend = Fear.
            
            # 3. Trend Component (SMA Dist) - 40% Weight
            price = df['Close'].iloc[-1]
            sma50 = df['SMA'].iloc[-1] # Using 20 as proxy for short term
            trend_score = 50 + ((price - sma50) / sma50) * 1000 # Scaling factor
            
            # Composite
            # RSI 0-100 directly maps
            # Trend: if price is 5% above SMA, score +50 -> 100
            
            final_score = (rsi_val * 0.4) + (trend_score * 0.6)
            final_score = max(0, min(100, final_score))
            
            label = "NEUTRAL"
            if final_score < 25: label = "EXTREME FEAR"
            elif final_score < 45: label = "FEAR"
            elif final_score > 75: label = "EXTREME GREED"
            elif final_score > 55: label = "GREED"
            
            return {
                "score": int(final_score),
                "label": label
            }
            
        except Exception as e:
            print(f"Sentiment Error: {e}")
            return {"score": 50, "label": "NEUTRAL"}

    def calculate_rsi(self, df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def fetch_macro_news(self):
        """
        Phase 7.0: Macro-Sentience Bridge
        Fetch global macro news to set Risk Bias
        """
        try:
            # Mock news source or lightweight API call
            # In production, this would use CryptoPanic or similar
            news_events = [
                {"title": "Fed hints at rate pause", "impact": "POSITIVE"},
                {"title": "CPI data higher than expected", "impact": "NEGATIVE"},
                {"title": "Institutional inflows into BTC ETFs", "impact": "POSITIVE"}
            ]
            
            # Simulated analysis logic
            sentiment_score = 0
            for event in news_events:
                if event['impact'] == "POSITIVE": sentiment_score += 20
                else: sentiment_score -= 30
            
            risk_bias = "NEUTRAL"
            if sentiment_score > 30: risk_bias = "RISK-ON"
            elif sentiment_score < -30: risk_bias = "RISK-OFF"
            
            return {
                "bias": risk_bias,
                "score": sentiment_score,
                "events": news_events
            }
        except Exception as e:
            return {"bias": "NEUTRAL", "score": 0, "events": []}
