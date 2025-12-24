
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
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _prepare_features(self, df):
        """Feature Engineering for ML"""
        df = df.copy()
        
        # Technical Features
        df['Returns'] = df['Close'].pct_change()
        df['Vol_20'] = df['Returns'].rolling(20).std()
        df['RSI'] = self.calculate_rsi(df)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Dist_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # Lag Features (Past 3 candles)
        for lag in range(1, 4):
            df[f'Return_Lag{lag}'] = df['Returns'].shift(lag)
            
        # Target: Next Candle Close (Shifted back)
        df['Target'] = df['Close'].shift(-1)
        
        df.dropna(inplace=True)
        return df
        
    def calculate_rsi(self, df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train(self, df):
        """Train the Random Forest Model"""
        try:
            data = self._prepare_features(df)
            
            if len(data) < 100:
                return False # Not enough data
            
            # Features vs Target
            feature_cols = ['RSI', 'Vol_20', 'Dist_SMA50', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3']
            X = data[feature_cols]
            y = data['Target']
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train RF
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Oracle Training Error: {e}")
            return False

    def predict(self, df_current):
        """Predict Next Close Price"""
        if not self.is_trained:
            # Try to train on fly if history passed
            if len(df_current) > 150:
                self.train(df_current)
            else:
                return None
        
        try:
            # Prepare latest single row
            data = self._prepare_features(df_current)
            if data.empty: return None
            
            feature_cols = ['RSI', 'Vol_20', 'Dist_SMA50', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3']
            current_features = data[feature_cols].iloc[[-1]] # Last row
            
            # Scale
            current_scaled = self.scaler.transform(current_features)
            
            # Predict
            predicted_price = self.model.predict(current_scaled)[0]
            current_price = df_current['Close'].iloc[-1]
            
            predicted_change = (predicted_price - current_price) / current_price
            
            confidence = "NEUTRAL"
            if predicted_change > 0.005: confidence = "BULLISH" # +0.5%
            elif predicted_change < -0.005: confidence = "BEARISH" # -0.5%
            
            return {
                "predicted_price": predicted_price,
                "predicted_change_pct": predicted_change * 100,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Oracle Prediction Error: {e}")
            return None
