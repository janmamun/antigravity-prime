import ccxt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from geometric_prophet import calculate_hurst, calculate_fourier_prediction
from meme_prophet_math import calculate_liquidity_gravity, detect_hidden_voids

def run_50_trade_audit_sensitive():
    print("-" * 120)
    print(f"| {'Time':10} | {'Symbol':10} | {'Side':5} | {'PnL':7} | {'Hurst':7} | {'Fourier':7} | {'Gravity Dist'} | {'Signal Sync'} |")
    print("-" * 120)
    
    with open('sim_state.json', 'r') as f:
        state = json.load(f)
    
    history = state.get('history', [])
    if not history: return

    ex = ccxt.binance()
    
    total_audited = 0
    total_sync = 0
    total_friction = 0
    
    # Audit last 50 trades
    for trade in history[-50:]:
        symbol = trade['Symbol']
        side = trade['Side']
        pnl = trade['PnL']
        trade_time_str = trade['Time']
        
        try:
            dt = datetime.fromisoformat(trade_time_str)
            since = int(dt.timestamp() * 1000) - (150 * 3600 * 1000)
            
            ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=150)
            if not ohlcv: continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_pre = df[df['timestamp'] < int(dt.timestamp() * 1000)].tail(100)
            
            if len(df_pre) < 30: continue
            
            prices = df_pre['close'].values
            h = calculate_hurst(prices)
            f = calculate_fourier_prediction(prices) # 1 for Up, -1 for Down
            
            curr_price = prices[-1]
            grav_p, grav_pull = calculate_liquidity_gravity(df_pre, curr_price)
            grav_dist = (curr_price - grav_p) / grav_p
            
            # Prediction Logic:
            # If H > 0.5 (Trending): Prediction follows last move
            # If H < 0.5 (Mean Reverting): Prediction fades last move
            last_move = np.sign(prices[-1] - prices[-2])
            math_pred = f # Using Fourier as primary directional bias
            
            sync_status = "Neutral"
            if math_pred == 1 and side == "BUY": sync_status = "✅ SYNC"
            elif math_pred == -1 and side == "SELL": sync_status = "✅ SYNC"
            elif math_pred != 0: sync_status = "❌ FRICTION"
            
            if sync_status == "✅ SYNC": total_sync += 1
            elif sync_status == "❌ FRICTION": total_friction += 1
            total_audited += 1
            
            print(f"| {trade_time_str[11:19]:10} | {symbol:10} | {side:5} | {pnl:7.2f} | {h:7.3f} | {f:7.1f} | {grav_dist:12.4f} | {sync_status:12} |")
            
        except Exception as e:
            pass
            
    print("-" * 120)
    print(f"Prophet Multi-Factor Audit Summary:")
    print(f" - Trades Audited: {total_audited}")
    print(f" - Mathematical SYNC (Math agreed with Trade): {total_sync}")
    print(f" - Mathematical FRICTION (Math suggested opposite): {total_friction}")
    if total_audited > 0:
        print(f" - Overall Math/Trade Alignment: {(total_sync/total_audited)*100:.2f}%")
    print("-" * 120)

if __name__ == "__main__":
    run_50_trade_audit_sensitive()
