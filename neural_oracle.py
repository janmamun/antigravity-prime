
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class NeuralOracle:
    def __init__(self):
        self.model_file = "neural_oracle.joblib"
        self.scaler_file = "neural_scaler.joblib"
        self.models = {} # {symbol: model}
        self.scalers = {} # {symbol: scaler}
        
    def _prepare_features(self, df):
        """Feature Engineering 2.0: Multi-Dimensional Sentiment & Flow"""
        df = df.copy()
        
        # Returns & Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Vol_20'] = df['Returns'].rolling(20).std()
        
        # Volume Profile (Z-Score)
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Z'] = (df['Volume'] - df['Vol_SMA']) / (df['Volume'].rolling(20).std() + 1e-6)
        
        # Momentum
        df['RSI'] = self.calculate_rsi(df)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Dist_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # Phase 10: Orderflow Features
        if 'OBI' not in df.columns: df['OBI'] = 0.0
        if 'CVD_Delta' not in df.columns: df['CVD_Delta'] = 0.0
        
        # Target: Next Candle Return
        df['Target'] = df['Returns'].shift(-1)
        
        df.dropna(inplace=True)
        return df
        
    def calculate_rsi(self, df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def train(self, df, symbol):
        """Train the Random Forest Model for a specific symbol"""
        try:
            data = self._prepare_features(df)
            if len(data) < 100: return False 
            
            feature_cols = ['RSI', 'Vol_20', 'Volume_Z', 'Dist_SMA50', 'OBI', 'CVD_Delta']
            X = data[feature_cols]
            y = data['Target']
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, shuffle=False)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            return True
        except Exception as e:
            print(f"Oracle Training Error ({symbol}): {e}")
            return False

    def predict(self, df_current, symbol="BTC/USDT"):
        """Predict Next Return with Confidence Intensity"""
        if symbol not in self.models:
            if len(df_current) > 150: self.train(df_current, symbol)
            else: return None
        
        try:
            data = self._prepare_features(df_current)
            if data.empty: return None
            
            feature_cols = ['RSI', 'Vol_20', 'Volume_Z', 'Dist_SMA50', 'OBI', 'CVD_Delta']
            current_features = data[feature_cols].iloc[[-1]]
            
            # Confidence Decay (Bypass if data is older than 1 hour in crypto time)
            # For simplicity, we assume row index -1 is 'now'
            
            current_scaled = self.scalers[symbol].transform(current_features)
            predicted_return = self.models[symbol].predict(current_scaled)[0]
            
            # Score Intensity: Based on predicted move vs volatility
            vol = data['Vol_20'].iloc[-1]
            intensity = predicted_return / (vol + 1e-6)
            
            confidence = "NEUTRAL"
            if predicted_return > 0.003: confidence = "BULLISH"
            elif predicted_return < -0.003: confidence = "BEARISH"
            
            # Absolute confidence score (0-100)
            conf_score = min(abs(intensity) * 20, 100)
            
            return {
                "predicted_change_pct": predicted_return * 100,
                "confidence": confidence,
                "confidence_score": conf_score,
                "intensity": intensity
            }
        except Exception as e:
            print(f"Oracle Prediction Error: {e}")
            return None

    def predict_orderflow(self, df_current, symbol):
        """Phase 10: Predict if the next 5m window is an Accumulation or Distribution shadow"""
        prediction = self.predict(df_current, symbol)
        if not prediction: return "STAGNANT"
        
        intensity = prediction.get('intensity', 0)
        if intensity > 1.5: return "ACCUMULATION_SHADOW"
        elif intensity < -1.5: return "DISTRIBUTION_SHADOW"
        return "STAGNANT"
