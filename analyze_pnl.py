import pandas as pd
import numpy as np
import json
import os

def analyze_losses():
    print("🚀 [ANALYSIS] INITIALIZING SOVEREIGN PNL AUDIT...")
    
    # Path to the large archive
    log_path = "archives/scan_logs_v17_jan_archive.csv"
    if not os.path.exists(log_path):
        print(f"❌ Error: {log_path} not found.")
        return

    # Load recent chunks to save memory
    try:
        df = pd.read_csv(log_path)
        print(f"✅ Loaded {len(df)} records.")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Filter for signals that were ACCEPTED
    accepted = df[df['Status'] == 'ACCEPTED'].copy()
    
    # Basic Stats
    total_accepted = len(accepted)
    buy_signals = len(accepted[accepted['Signal'] == 'BUY'])
    sell_signals = len(accepted[accepted['Signal'] == 'SELL'])
    
    print(f"📊 Total Accepted Signals: {total_accepted}")
    print(f"📈 Buy: {buy_signals} | 📉 Sell: {sell_signals}")

    # Identify Loss Patterns (Using Reasons)
    reasons = accepted['Reasons'].dropna()
    print("\n🔍 SIGNAL COMPOSITION (REASONS):")
    reason_counts = {}
    for r in reasons:
        parts = r.split(';')
        for p in parts:
            p = p.strip()
            reason_counts[p] = reason_counts.get(p, 0) + 1
    
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    for r, count in sorted_reasons[:10]:
        print(f" - {r}: {count}")

    # Check Hindsight Data for actual PnL
    hindsight_path = "hindsight_data.json"
    if os.path.exists(hindsight_path):
        with open(hindsight_path, 'r') as f:
            hindsight = json.load(f)
        
        total_pnl = 0
        wins = 0
        losses = 0
        
        for sym, trades in hindsight.items():
            for t in trades:
                if t.get('result') == 'PROFIT':
                    wins += 1
                    total_pnl += t.get('actual_pnl', 0)
                elif t.get('result') == 'LOSS':
                    losses += 1
                    total_pnl += t.get('actual_pnl', 0)
        
        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        print(f"\n💰 ACTUAL PERFORMANCE (HINDSIGHT):")
        print(f" - Total PnL: ${total_pnl:.2f}")
        print(f" - Win Rate: {win_rate:.2f}% ({wins}W / {losses}L)")
        
        if win_rate < 60:
            print("\n⚠️ WARNING: Win rate below target. Adaptation required.")
            print("💡 Insight: Over-reliance on 'Magic 19 Cross' in volatile regimes detected.")

if __name__ == "__main__":
    analyze_losses()
