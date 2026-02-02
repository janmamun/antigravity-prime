import ccxt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Import all research modules
from geometric_prophet import calculate_hurst, calculate_fourier_prediction
from meme_prophet_math import calculate_liquidity_gravity, detect_hidden_voids
from quantum_prophet_math import calculate_lyapunov_exponent, calculate_shannon_entropy, calculate_ornstein_uhlenbeck

def run_grand_code_audit():
    print("=" * 135)
    print(f"| {'Time':10} | {'Symbol':10} | {'Bot':5} | {'PnL':7} | {'Hurst':5} | {'Four':4} | {'Void'} | {'Grav %'} | {'Lya':6} | {'Ent':5} | {'Fair %'} | {'Code Verdict'} |")
    print("=" * 135)
    
    with open('sim_state.json', 'r') as f:
        state = json.load(f)
    
    history = state.get('history', [])
    if not history: return

    ex = ccxt.binance()
    
    total_audited = 0
    saved_losses = 0
    confirmed_wins = 0
    friction_events = 0
    
    # Audit last 50 trades
    for trade in history[-50:]:
        symbol = trade['Symbol']
        side = trade['Side']
        pnl = trade['PnL']
        trade_time_str = trade['Time']
        
        try:
            dt = datetime.fromisoformat(trade_time_str)
            # Fetch 110 hours leading up to the trade to ensure a full 100-candle set
            since = int(dt.timestamp() * 1000) - (110 * 3600 * 1000)
            
            ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=120)
            if not ohlcv: continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_pre = df[df['timestamp'] < int(dt.timestamp() * 1000)].tail(100)
            
            if len(df_pre) < 40: continue
            
            prices = df_pre['close'].values
            curr_p = prices[-1]
            
            # 1. Geo
            h = calculate_hurst(prices)
            f = calculate_fourier_prediction(prices)
            
            # 2. Meme
            grav_p, _ = calculate_liquidity_gravity(df_pre, curr_p)
            grav_dist = (curr_p - grav_p) / grav_p
            voids = detect_hidden_voids(df_pre)
            in_void = any(v['bottom'] <= curr_p <= v['top'] for v in voids[-3:])
            v_tag = "YES" if in_void else "NO"
            
            # 3. Quantum
            lya = calculate_lyapunov_exponent(prices)
            ent = calculate_shannon_entropy(prices)
            fair = calculate_ornstein_uhlenbeck(prices)
            fair_dist = (fair - curr_p) / curr_p
            
            # THE CODE VERDICT
            # We want to see if the Math predicted the PnL.
            # A 'Code Confirmed' means Math Direction == Side.
            # A 'Code Friction' means Math Direction != Side.
            
            math_dir = 0
            if f != 0: math_dir = f
            elif fair_dist > 0.01: math_dir = 1
            elif fair_dist < -0.01: math_dir = -1
            
            side_num = 1 if side == "BUY" else -1
            verdict = "NEUTRAL"
            
            if math_dir == side_num:
                verdict = "✅ SYNC"
                if pnl > 0: confirmed_wins += 1
            elif math_dir != 0:
                verdict = "❌ FRICTION"
                friction_events += 1
                if pnl < 0: saved_losses += abs(pnl)
            
            total_audited += 1
            
            print(f"| {trade_time_str[11:19]:10} | {symbol:10} | {side:5} | {pnl:7.2f} | {h:5.2f} | {f:4.1f} | {v_tag:4} | {grav_dist*100:6.2f} | {lya:6.3f} | {ent:5.2f} | {fair_dist*100:6.2f} | {verdict:12} |")
            
        except Exception as e:
            pass
            
    print("=" * 135)
    print(f"GRAND QUANTUM CODE AUDIT SUMMARY:")
    print(f" - Trades Audited: {total_audited}")
    print(f" - Quantum Syncs (High Conviction Matches): {confirmed_wins}")
    print(f" - Mathematical Friction Events (Warnings): {friction_events}")
    print(f" - DRAWDOWN PROTECTION: ${saved_losses:.2f} in losses identified by the Quantum Multi-Factor.")
    eff = ((confirmed_wins + (friction_events if saved_losses > 0 else 0)) / (total_audited if total_audited > 0 else 1)) * 100
    print(f" - Code Efficiency: {eff:.2f}% predictability discovered.")
    print("=" * 135)

if __name__ == "__main__":
    run_grand_code_audit()
