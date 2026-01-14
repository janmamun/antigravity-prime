
import time
import json
import os
from datetime import datetime
from market_scanner import MarketScanner
from simulation_engine import FuturesSimulator
from meta_brain import LocalBrain, MetaBrain

HEARTBEAT_FILE = "guardian_heartbeat.json"

def log_sovereign(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    with open("bot_output.log", "a") as f:
        f.write(f"[{timestamp}] [{level}] {msg}\n")

def update_heartbeat():
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            json.dump({"last_heartbeat": time.time(), "status": "ALIVE"}, f)
    except:
        pass

def run_v17_loop():
    log_sovereign("🪐 ANTIGRAVITY PRIME PHASE 17: SOVEREIGN INTELLIGENCE", "SYSTEM")
    log_sovereign("Initializing Guardian Protocols...", "SYSTEM")
    
    scanner = MarketScanner()
    brain = LocalBrain()
    meta = MetaBrain()
    sim = FuturesSimulator() # For direct execution handle
    
    log_sovereign("System Sovereign. Monitoring Market Matrix.", "SYSTEM")
    
    trade_count = 0
    
    while True:
        try:
            update_heartbeat()
            
            # --- 0. LIVE STATUS CHECK ---
            is_live = scanner.bot.is_live
            live_balance = scanner.bot.get_live_balance() if is_live else sim.state['balance']
            
            # --- 1. MONITOR ACTIVE POSITIONS ---
            if is_live:
                # Real-time positions would be fetched here
                # For now, we continue tracking via logs and sim for hybrid audit
                pass
            
            positions = sim.get_positions()
            if positions:
                log_sovereign(f"Monitoring {len(positions)} active positions...", "EXECUTOR")
                # Fetch current prices for active positions
                active_symbols = list(positions.keys())
                try:
                    tickers = scanner.exchange.fetch_tickers(active_symbols)
                    prices = {s: t['last'] for s, t in tickers.items() if t['last']}
                    
                    notifications = sim.monitor_orders(prices)
                    for note in notifications:
                        log_sovereign(note, "EXECUTOR")
                    
                    # Phase 20: Hindsight Sync - Resolve PENDING predictions
                    scanner.bot.sync_hindsight(sim.state['history'])
                except Exception as pe:
                    log_sovereign(f"Price Update Error: {pe}", "EXECUTOR")

            # --- 2. MARKET SCAN ---
            log_sovereign("Scanning Market Matrix (Top 15)...", "SCANNER")
            opportunities = scanner.scan_market()
            
            if opportunities:
                top_trade = opportunities[0]
                log_sovereign(f"Top Opportunity: {top_trade['symbol']} | Score: {top_trade['score']}", "SCANNER")
                
                # Minimum score for overnight auto-trade: 75 (Higher for safety)
                if top_trade['score'] >= 75:
                    log_sovereign(f"EXECUTION SIGNAL: {top_trade['signal']} {top_trade['symbol']}", "EXECUTOR")
                    
                    # Ensure we have the absolute latest price
                    latest_ticker = scanner.exchange.fetch_ticker(top_trade['symbol'])
                    curr_price = latest_ticker['last']
                    
                    pos_size = 100 # Multi-coin strategy: smaller size per trade
                    
                    if is_live:
                        # Allocate 10% of live balance for each trade (Micro-Cap Mode)
                        live_pos_size = live_balance * 0.1
                        res = scanner.bot.execute_live_order(
                            top_trade['symbol'], 
                            top_trade['signal'], 
                            live_pos_size, 
                            curr_price, 
                            top_trade['tp'], 
                            top_trade['sl']
                        )
                        log_sovereign(f"LIVE EXECUTION: {res.get('msg')}", "SYSTEM")
                        
                        # We still mirror in sim for dashboard analytics
                        sim.execute_trade(
                            top_trade['symbol'], 
                            top_trade['signal'], 
                            curr_price, 
                            100, # Base unit for analytics 
                            top_trade['tp'], 
                            top_trade['sl'],
                            atr=top_trade.get('atr', 0)
                        )
                    else:
                        res = sim.execute_trade(
                            top_trade['symbol'], 
                            top_trade['signal'], 
                            curr_price, 
                            pos_size, 
                            top_trade['tp'], 
                            top_trade['sl'],
                            atr=top_trade.get('atr', 0)
                        )
                    log_sovereign(res.get('msg', 'Trade Attempt Complete'), "EXECUTOR")
                    trade_count += 1
            else:
                log_sovereign("No high-confidence signals. Standing by.", "SCANNER")

            # --- 3. BRAIN CYCLE ---
            if trade_count >= 5:
                log_sovereign("Commencing Brain Evolution Cycle...", "BRAIN")
                perf_summary = meta.run_self_audit()
                brain.evolve(perf_summary)
                trade_count = 0
                log_sovereign("Evolution Cycle Complete.", "BRAIN")
            
            time.sleep(120) # 2 minute heartbeat for tighter monitoring
            
        except Exception as e:
            log_sovereign(f"CRITICAL ERROR in Loop: {e}", "GUARDIAN")
            time.sleep(10)

if __name__ == "__main__":
    run_v17_loop()
