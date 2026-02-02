import ccxt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from geometric_prophet import get_geometric_signal

def run_hindsight_audit():
    print("-" * 80)
    print(f"| {'Time':20} | {'Symbol':10} | {'Bot Side':8} | {'PnL':8} | {'Prophet Prediction':18} |")
    print("-" * 80)
    
    # Load past trades
    with open('sim_state.json', 'r') as f:
        state = json.load(f)
    
    pnl_history = state.get('history', [])
    if not pnl_history:
        print("No trades found in history.")
        return

    ex = ccxt.binance()
    
    matches = 0
    contradictions = 0
    total_signals = 0
    
    # Take the last 15 trades for audit
    for trade in pnl_history[-15:]:
        symbol = trade['Symbol']
        side = trade['Side']
        pnl = trade['PnL']
        trade_time_str = trade['Time']
        
        try:
            # Parse timestamp to ms for ccxt
            # Format: 2026-01-30T12:35:24.147476
            dt = datetime.fromisoformat(trade_time_str)
            since = int(dt.timestamp() * 1000) - (100 * 3600 * 1000) # 100 hours before
            
            ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=101)
            if not ohlcv: continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Filter to only data BEFORE the trade entry
            df = df[df['timestamp'] < int(dt.timestamp() * 1000)]
            
            if len(df) < 50: continue
            
            signal, score, metrics = get_geometric_signal(df)
            
            prophet_note = "N/A (Neutral)"
            if signal != "WAIT":
                total_signals += 1
                prophet_note = f"{signal} (Score: {score})"
                
                # Check if Prophet matched the Bot's trade
                if signal == side:
                    matches += 1
                    prophet_note = f"✅ {signal}"
                else:
                    contradictions += 1
                    prophet_note = f"❌ {signal}"
            
            print(f"| {trade_time_str[:19]:20} | {symbol:10} | {side:8} | {pnl:8.2f} | {prophet_note:18} |")
            
        except Exception as e:
            # print(f"Error auditing {symbol}: {e}")
            pass
            
    print("-" * 80)
    print(f"Prophet Correlation Summary:")
    print(f" - Total Trades Audited: 15")
    print(f" - Prophet Signals Issued: {total_signals}")
    print(f" - Matches (Bot matched Math): {matches}")
    print(f" - Contradictions (Math warned against Bot): {contradictions}")
    if total_signals > 0:
        print(f" - Predictive Sync: {(matches/total_signals)*100:.2f}%")
    print("-" * 80)

if __name__ == "__main__":
    run_hindsight_audit()
