import os
import json
import time
import pandas as pd
from datetime import datetime
from meta_brain import MetaBrain, LocalBrain
import subprocess
from simulation_engine import FuturesSimulator
import numpy as np
import ccxt
from sentiment_scanner import SentimentScanner

class EvolutionarySentinel:
    def __init__(self):
        self.meta = MetaBrain()
        self.brain = LocalBrain()
        self.state_file = "sim_state.json"
        self.config_file = "bot_config.json"
        self.bridge_log = "neural_bridge.log"
        self.blacklist_file = "blacklist.json"
        self.constraints_file = "learned_constraints.json" # Phase 8.0
        self.sentiment = SentimentScanner() # Phase 7.0
        
        if not os.path.exists(self.constraints_file):
            with open(self.constraints_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.blacklist_file):
            with open(self.blacklist_file, 'w') as f:
                json.dump([], f)

    def log_to_bridge(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] 🦅 [EVOLUTION] {msg}\n"
        with open(self.bridge_log, "a") as f:
            f.write(entry)
        print(entry.strip())

    def get_blacklist(self):
        with open(self.blacklist_file, 'r') as f:
            return json.load(f)

    def save_blacklist(self, blacklist):
        with open(self.blacklist_file, 'w') as f:
            json.dump(list(set(blacklist)), f, indent=4)

    def audit_trade_deaths(self):
        """Find symbols that are killing the PnL and blacklist them"""
        if not os.path.exists(self.state_file): return
        
        with open(self.state_file, 'r') as f:
            state = json.load(f)
            history = state.get("history", [])
            
        if not history: return
        
        df = pd.DataFrame(history)
        # Identify assets with > 5% drawdown in a single trade
        # Or assets with < 20% win-rate over 5+ trades
        deaths = []
        
        # 1. High Impact Losses
        # PnL in sim_state is absolute. If starting balance is 10k, -50 is 0.5%. 
        # For a 100 USDT live start, -5 is 5%.
        threshold = 50 if state.get("initial_balance", 10000) > 1000 else 5
        
        high_losses = df[df['PnL'] < -threshold]
        for sym in high_losses['Symbol'].unique():
            deaths.append(sym)
            self.log_to_bridge(f"CRITICAL LOSS detected on {sym} (< -{threshold}). Adding to toxic assets.")

        # 2. Sequential Failures
        for sym in df['Symbol'].unique():
            sym_df = df[df['Symbol'] == sym].tail(5)
            if len(sym_df) >= 3:
                wins = len(sym_df[sym_df['PnL'] > 0])
                if wins == 0:
                    deaths.append(sym)
                    self.log_to_bridge(f"REPEATED FAILURE on {sym} (0/{len(sym_df)} Win Rate). Blacklisting.")

        current_blacklist = self.get_blacklist()
        current_blacklist.extend(deaths)
        self.save_blacklist(current_blacklist)

    def debrief_losses(self):
        """Phase 8.0: Analyze failed trades and auto-generate constraints"""
        if not os.path.exists(self.state_file): return
        
        with open(self.state_file, 'r') as f:
            history = json.load(f).get("history", [])
        
        if len(history) < 10: return
        
        df = pd.DataFrame(history)
        symbols = df['Symbol'].unique()
        
        constraints = {}
        if os.path.exists(self.constraints_file):
            with open(self.constraints_file, 'r') as f:
                constraints = json.load(f)
        
        for sym in symbols:
            sym_df = df[df['Symbol'] == sym]
            if len(sym_df) < 3: continue
            
            win_rate = len(sym_df[sym_df['PnL'] > 0]) / len(sym_df)
            
            if win_rate < 0.33:
                # Poor performance detected - Generate a constraint
                constraints[sym] = {
                    "min_score_boost": 10, # Require higher score for this asset
                    "reason": f"Recursive Hindsight: Low Win Rate ({win_rate*100:.1f}%)",
                    "expiry": time.time() + (86400 * 3) # 3 day probation
                }
                self.log_to_bridge(f"RECURSIVE PATCH: Applying Probation to {sym} (WR: {win_rate*100:.1f}%)")
        
        with open(self.constraints_file, 'w') as f:
            json.dump(constraints, f, indent=4)

    def optimize_params(self):
        """Heuristic-based parameter tuning"""
        res = self.meta.run_self_audit()
        self.log_to_bridge(res)
        
        # Deep optimization based on recent volatility
        config = self.meta.load_config()
        # If win rate is low, increase rsi_oversold (buy lower)
        # and decrease leverage
        history = self.meta.load_history()
        if not history: return
        
        df = pd.DataFrame(history)
        win_rate = len(df[df['PnL'] > 0]) / len(df)
        
        if win_rate < 0.55:
            config['rsi_oversold'] = max(20, config.get('rsi_oversold', 35) - 2)
            config['leverage'] = max(1, config.get('leverage', 5) - 1)
            self.log_to_bridge("WIN RATE BELOW THRESHOLD. Entering DEFENSIVE mode. Buying deeper, reducing leverage.")
        elif win_rate > 0.70:
            config['rsi_oversold'] = min(40, config.get('rsi_oversold', 35) + 1)
            config['leverage'] = min(20, config.get('leverage', 5) + 1)
            self.log_to_bridge("WIN RATE SUPERIOR. Entering AGGRESSIVE mode ($$$). Scaling leverage.")
            
        self.meta.save_config(config)

    def run_ab_tuning(self):
        """Phase 6.0: Real Background A/B Testing using FuturesSimulator"""
        if not os.path.exists(self.config_file): return
        
        self.log_to_bridge("Initiating Alpha-Tuner A/B Backtesting...")
        
        # 1. Prepare Data (Fetch recent data for top 3 coins)
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        data_map = {}
        try:
            exch = ccxt.binance({'options': {'defaultType': 'spot'}})
            for sym in symbols:
                ohlcv = exch.fetch_ohlcv(sym, timeframe='1h', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                data_map[sym] = df
        except Exception as e:
            self.log_to_bridge(f"Alpha-Tuner Data Error: {e}")
            return

        # 2. Define Variations
        variations = [
            {"name": "CONSERVATIVE", "rsi_oversold": 30, "rsi_overbought": 70, "min_score": 75},
            {"name": "BALANCED", "rsi_oversold": 35, "rsi_overbought": 65, "min_score": 65},
            {"name": "AGGRESSIVE", "rsi_oversold": 40, "rsi_overbought": 60, "min_score": 55}
        ]
        
        results = []
        from trading_bot_v17 import UltimateV17Bot # Import inside to avoid circular deps if any
        
        for var in variations:
            # Setup Sim with variation
            sim_bot = UltimateV17Bot()
            sim_bot.config.update(var)
            simulator = FuturesSimulator(initial_balance=1000, leverage=10)
            
            # Simulate
            for sym, df in data_map.items():
                for i in range(50, len(df)):
                    snapshot = df.iloc[:i]
                    analysis = sim_bot.analyze_snapshot(snapshot, sym)
                    if analysis['signal'] != 'WAIT':
                        simulator.execute_trade(
                            sym, analysis['signal'], analysis['price'], 
                            100, analysis['tp'], analysis['sl'], analysis['atr']
                        )
                    # Simple monitor (no live price stream here, just use current data row)
                    current_prices = {sym: df.iloc[i]['Close']}
                    simulator.monitor_orders(current_prices)

            stats = simulator.calculate_stats()
            # Score = (Total PnL / Max Drawdown) * Win Rate
            # Simplified Score for now: PnL + WinRate
            score = stats['total_pnl'] + (stats['win_rate'] * 0.5)
            results.append({"name": var['name'], "score": score, "config": var, "stats": stats})
            self.log_to_bridge(f"Variation {var['name']} -> PnL: ${stats['total_pnl']:.2f}, WR: {stats['win_rate']:.1f}%")

        # 3. Promote Best
        best = max(results, key=lambda x: x['score'])
        self.log_to_bridge(f"ALPHA-TUNER: {best['name']} strategy dominant (Recovery Score: {best['score']:.1f})")
        
        current_config = self.meta.load_config()
        # Merge if better than 0 or significant improvement
        if best['score'] > 0:
            promotion_data = best['config'].copy()
            name = promotion_data.pop('name')
            current_config.update(promotion_data)
            self.meta.save_config(current_config)
            self.log_to_bridge(f"STRATEGY PROMOTED: {name} moves to Live Production.")

    def run_cycle(self):
        self.log_to_bridge("Starting Evolutionary Cycle V4.0...")
        
        # 1. Audit & Blacklist
        self.audit_trade_deaths()
        
        # 2. Optimize Config & Run A/B Tuning
        self.debrief_losses() # Phase 8.0
        self.optimize_params()
        self.run_ab_tuning()
        
        # 3. Request LLM Analysis
        history = self.meta.load_history()
        # Find latest balance from sim_state.json if available
        current_bal = "N/A"
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                current_bal = json.load(f).get("balance", "N/A")
        
        performance_summary = f"Current Balance: {current_bal} | Blacklisted: {self.get_blacklist()} | Win Rate: {self.meta.get_win_rate()}%"
        advice = self.brain.evolve(performance_summary)
        self.log_to_bridge(f"Neural Advice Received: {advice[:200]}...")
        
        # 4. Strategy Patching (Experimental)
        if "REDUCE RISK" in advice.upper():
            config = self.meta.load_config()
            config['risk_factor_pct'] = max(0.02, config.get('risk_factor_pct', 0.1) * 0.8)
            self.meta.save_config(config)
            self.log_to_bridge("Applied LLM-driven Risk Reduction.")

        # 5. Macro-Sentiment Sync (Phase 7.0)
        macro_data = self.sentiment.fetch_macro_news()
        self.log_to_bridge(f"Macro-Sentience Pulse: {macro_data['bias']} (Score: {macro_data['score']})")
        
        # Apply Macro Bias to Global Config
        config = self.meta.load_config()
        config['macro_bias'] = macro_data['bias']
        
        if macro_data['bias'] == "RISK-OFF":
            config['risk_factor_pct'] = max(0.02, config.get('risk_factor_pct', 0.1) * 0.7)
            config['max_open_positions'] = 3
            self.log_to_bridge("GLOBAL RISK-OFF detected. Retracting exposure, tightening limits.")
        elif macro_data['bias'] == "RISK-ON":
            config['risk_factor_pct'] = min(0.2, config.get('risk_factor_pct', 0.1) * 1.3)
            config['max_open_positions'] = 8
            self.log_to_bridge("GLOBAL RISK-ON detected. Scaling up alpha-seeking parameters.")
            
        self.meta.save_config(config)

        # 6. Indicator Sensitivity Patches (Phase 5.0)
        try:
            if "indicator_patches" in advice:
                import re
                # Look for JSON block in the advice
                match = re.search(r'\{.*"indicator_patches".*\}', advice, re.DOTALL)
                if match:
                    try:
                        patch_data = json.loads(match.group())
                        patches = patch_data.get("indicator_patches", {})
                        if patches:
                            config = self.meta.load_config()
                            config.update(patches)
                            self.meta.save_config(config)
                            self.log_to_bridge(f"Applied Neural Hindsight Patches: {list(patches.keys())}")
                    except json.JSONDecodeError:
                        pass 
        except Exception as e:
            self.log_to_bridge(f"Sentinel Patching Error: {e}")
        
        self.log_to_bridge("Cycle Complete. Standing by for next sync (1h).")

    def run(self):
        while True:
            try:
                self.log_to_bridge("Starting heartbeat sync...")
                self.run_cycle()
            except Exception as e:
                 self.log_to_bridge(f"SENTINEL ERROR: {e}")
            time.sleep(600) # Reduce sleep from 1h to 10m for better dashboard status

if __name__ == "__main__":
    import sys
    sentinel = EvolutionarySentinel()
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        sentinel.run_cycle()
    else:
        sentinel.run()
