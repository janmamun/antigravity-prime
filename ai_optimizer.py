
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

class AITradingOptimizer:
    def __init__(self, trades_csv='v8_ultimate_trades.csv'):
        self.trades_csv = trades_csv
        try:
            self.df = pd.read_csv(self.trades_csv)
            # Filter for completed trades and ensure columns exist
            signal_cols = [c for c in self.df.columns if c.startswith('sig_')]
            if not signal_cols:
                print("⚠️ No signal columns found in trade log. Run backtest with updated bot first.")
                self.df = pd.DataFrame() # Empty
        except FileNotFoundError:
            print(f"⚠️ {trades_csv} not found.")
            self.df = pd.DataFrame()

    def analyze_performance(self):
        """Analyze which signals correlate with high PnL"""
        if self.df.empty:
            return {}
        
        # Prepare Data
        feature_cols = [c for c in self.df.columns if c.startswith('sig_')]
        target_col = 'pnl_pct'
        
        # Drop rows with NaN
        data = self.df.dropna(subset=feature_cols + [target_col])
        
        if len(data) < 10:
            return {"status": "Not enough data to learn (need >10 trades)"}
            
        X = data[feature_cols]
        y = data[target_col]
        
        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get Importances
        importances = model.feature_importances_
        feature_importance_dict = dict(zip(feature_cols, importances))
        
        # Sort
        sorted_features = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_features

    def get_optimization_suggestions(self):
        """Suggest weight adjustments based on AI analysis"""
        importances = self.analyze_performance()
        if not importances or "status" in importances:
            return []
            
        suggestions = []
        for feature, importance in importances.items():
            clean_name = feature.replace('sig_', '').upper()
            
            # Logic: If importance is high, suggest boost. If low, suggest output.
            # (In a real systemic loop, we would check correlation too, not just importance)
            # Simplified: Importance = Influence. Only boost if Average PnL for that signal activation is positive.
            
            # Check avg PnL when this signal was active (>0)
            avg_pnl_when_active = self.df[self.df[feature] > 0]['pnl_pct'].mean()
            
            action = "MAINTAIN" 
            if importance > 0.15 and avg_pnl_when_active > 0:
                action = "BOOST WEIGHT 🚀"
            elif importance > 0.15 and avg_pnl_when_active < 0:
                action = "REDUCE WEIGHT (False Signals) 🔻"
            elif importance < 0.05:
                action = "IGNORE (Low Impact) 💤"
                
            suggestions.append({
                'Signal': clean_name,
                'Importance': f"{importance:.1%}",
                'Avg PnL When Active': f"{avg_pnl_when_active:.2f}%",
                'AI Recommendation': action
            })
            
        return suggestions

if __name__ == "__main__":
    ai = AITradingOptimizer()
    suggestions = ai.get_optimization_suggestions()
    print("🧠 AI BRAIN RECOMMENDATIONS:")
    for s in suggestions:
        print(s)
