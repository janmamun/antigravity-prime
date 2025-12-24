
import time
import os
import json
from datetime import datetime
from market_scanner import MarketScanner
from simulation_engine import FuturesSimulator
from meta_brain import LocalBrain

def run_autonomy_loop():
    print("🪐 ANTIGRAVITY PRIME PHASE 16: INSTITUTIONAL INTELLIGENCE")
    print("Initializing Systems...")
    
    scanner = MarketScanner()
    sim = FuturesSimulator(mode='PAPER')
    brain = LocalBrain(model="qwen2.5-coder:7b")
    
    trade_count = 0
    last_brain_cycle = datetime.now()
    
    while True:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning Market Matrix...")
            
            # 1. SCAN & TRADE
            opportunities = scanner.scan_market()
            
            # Re-load config for dynamic updates from Brain
            with open("bot_config.json", "r") as f:
                config = json.load(f)
            
            # Execute top 2 signals
            executed_this_cycle = 0
            for op in opportunities[:2]:
                if op['score'] >= config.get('min_score', 65):
                    # Check if already in position
                    positions = sim.get_positions()
                    if op['symbol'] not in positions:
                        # Position Sizing: 10% equity risk
                        equity = sim.state['balance']
                        risk_usd = equity * 0.10
                        
                        res = sim.execute_trade(
                            op['symbol'], 
                            'LONG', # Base V15 is Long-biased per funding rules in analyze_snapshot
                            op['price'], 
                            risk_usd, 
                            op['tp'], 
                            op['sl'],
                            op['atr']
                        )
                        print(f"   >>> EXECUTION: {res['msg']}")
                        if res['status'] == 'success':
                            trade_count += 1
                            executed_this_cycle += 1
            
            # 2. MONITOR POSITIONS
            # Get current prices for monitoring
            tickers = scanner.exchange.fetch_tickers()
            current_prices = {s: t['last'] for s, t in tickers.items() if s in sim.get_positions()}
            
            notifications = sim.monitor_orders(current_prices)
            for note in notifications:
                print(f"   >>> MONITOR: {note}")

            # 3. BRAIN EVOLUTION CYCLE (Every Hour or 5 Trades)
            time_since_brain = datetime.now() - last_brain_cycle
            if time_since_brain.total_seconds() > 3600 or trade_count >= 5:
                print("🧠 BRAIN EVOLUTION CYCLE: Analyzing Performance...")
                stats = sim.calculate_stats()
                summary = f"""
                Equity: ${sim.state['balance']:.2f}
                Win Rate: {stats['win_rate']:.1f}%
                Total Trades: {stats['total_trades']}
                Active Positions: {list(sim.get_positions().keys())}
                Market Regime: {opportunities[0]['regime'] if opportunities else 'N/A'}
                """
                
                advice = brain.evolve(summary)
                print(f"   >>> BRAIN ADVICE LOGGED.")
                
                # Reset counters
                trade_count = 0
                last_brain_cycle = datetime.now()

            # 4. SLEEP (5 minutes)
            time.sleep(300)
            
        except Exception as e:
            print(f"!!! CRITICAL FAILURE: {e}")
            time.sleep(60) # Cooling off

if __name__ == "__main__":
    run_autonomy_loop()
