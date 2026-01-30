import json, os, pandas as pd
from datetime import datetime

def analyze():
    # 1. Audit sovereign_audit.json (Realized Trades)
    trades = []
    if os.path.exists('sovereign_audit.json'):
        with open('sovereign_audit.json', 'r') as f:
            trades = json.load(f)
    
    if not trades:
        print("No trades found in audit.")
        return

    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # We need to find realized PnL. The audit usually stores entries. 
    # Let's look for "exit" logs or calculate from balance if possible.
    # But since we want "Strategy Success", let's look at the win/loss distribution.
    
    print("--- SYMBOL PERFORMANCE (ALL TIME) ---")
    if 'symbol' in df.columns:
        symbol_counts = df['symbol'].value_counts()
        print(symbol_counts.head(10))

    # 2. Check strategy_memory.json
    if os.path.exists('strategy_memory.json'):
        print("\n--- STRATEGY MEMORY WIN RATES ---")
        with open('strategy_memory.json', 'r') as f:
            mem = json.load(f)
            perf = mem.get('performance', {})
            sorted_perf = sorted(perf.items(), key=lambda x: (x[1]['wins'] / (x[1]['wins'] + x[1]['losses'])) if (x[1]['wins'] + x[1]['losses']) > 0 else 0, reverse=True)
            for sym, data in sorted_perf[:10]:
                total = data['wins'] + data['losses']
                wr = (data['wins'] / total * 100) if total > 0 else 0
                print(f"{sym}: {wr:.1f}% WR ({total} trades)")

    # 3. Time Series Analysis (When did we peak?)
    # Based on our previous knowledge, we hit  and then .
    # Let's find the dates of those peaks.
    print("\n--- EQUITY PEAK TRACER ---")
    # Search logs for "Total USDT" and dates
    os.system('grep "Total USDT" monitoring.log | head -n 5')
    os.system('grep "Total USDT" monitoring.log | tail -n 5')
    
analyze()
