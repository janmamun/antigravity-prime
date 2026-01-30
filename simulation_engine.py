
import pandas as pd
import json
import os
import ccxt
import random
from datetime import datetime, timedelta

class FuturesSimulator:
    def __init__(self, mode='PAPER', api_key=None, api_secret=None, initial_balance=10000, leverage=3):
        print("🛡️ [SIM] Phase 75.4 Shielded Engine Initialized.")
        self.mode = mode
        self.state_file = "sim_state.json"
        self.leverage = leverage
        self.exchange = None
        
        # Load or Init State
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
            except:
                self.state = self._get_default_state(initial_balance)
        else:
            self.state = self._get_default_state(initial_balance)
            
        # Self-Healing: Ensure all keys exist
        default_state = self._get_default_state(initial_balance)
        for key, val in default_state.items():
            if key not in self.state:
                self.state[key] = val

    def _get_default_state(self, initial_balance):
        return {
            "balance": initial_balance,
            "initial_balance": initial_balance,
            "safeguard_vault": 0,
            "total_funding_fees": 0,
            "positions": {}, 
            "history": [],
            "strategy_performance": {}, 
            "last_funding_check": datetime.now().isoformat()
        }

    def get_positions(self):
        return self.state.get("positions", {})

    def monitor_orders(self, current_prices):
        notifications = []
        positions = self.state.get("positions", {})
        symbols = list(positions.keys())
        
        for symbol in symbols:
            try:
                price = current_prices.get(symbol)
                if price is None:
                    continue
                
                pos = positions[symbol]
                side = pos.get('side', 'LONG')
                is_long = side in ["LONG", "BUY"]
                tp = float(pos.get('tp') or 0)
                sl = float(pos.get('sl') or 0)
                
                triggered = False
                reason = ""
                
                if is_long:
                    if tp > 0 and price >= tp: triggered, reason = True, "TAKE PROFIT 🎯"
                    elif sl > 0 and price <= sl: triggered, reason = True, "STOP LOSS 🛑"
                else:
                    if tp > 0 and price <= tp: triggered, reason = True, "TAKE PROFIT 🎯"
                    elif sl > 0 and price >= sl: triggered, reason = True, "STOP LOSS 🛑"
                
                if triggered:
                    pnl = self.close_position(symbol, price)
                    notifications.append(f"{symbol}: {reason} Triggered at ${price:.2f} (PnL: ${pnl:.2f})")
            except Exception as e:
                # Shielded against any secondary errors (NoneType, etc)
                continue
                
        return notifications

    def execute_trade(self, symbol, side, price, amount_usd, tp=0, sl=0, atr=0, alpha_mode=False, strategy_id="default"):
        margin_req = amount_usd / self.leverage
        if margin_req > self.state.get("balance", 0):
            return {"status": "error", "msg": "Insufficient Margin"}
            
        if symbol in self.state["positions"]:
            return {"status": "error", "msg": "Position already open"}
            
        self.state["positions"][symbol] = {
            "entry_price": price,
            "size": (margin_req * self.leverage) / price,
            "side": side,
            "tp": tp,
            "sl": sl,
            "atr": atr,
            "alpha_mode": alpha_mode,
            "strategy_id": strategy_id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_state()
        return {"status": "success", "msg": f"Opened {side} {symbol} @ ${price:.2f}"}

    def close_position(self, symbol, price):
        if symbol not in self.state["positions"]: return 0
        pos = self.state["positions"][symbol]
        quantity = pos['size']
        entry_price = pos['entry_price']
        side = pos['side']
        strat_id = pos.get('strategy_id', 'default')
        
        if side in ['LONG', 'BUY']: pnl = (price - entry_price) * quantity
        else: pnl = (entry_price - price) * quantity
            
        self.state["balance"] = self.state.get("balance", 0) + pnl
        
        # Strategy Tracking
        perf = self.state.get("strategy_performance", {})
        perf[strat_id] = perf.get(strat_id, 0) + pnl
        self.state["strategy_performance"] = perf

        self.state["history"].append({
            "Symbol": symbol, "PnL": pnl, "Side": side, "Time": datetime.now().isoformat()
        })
        
        del self.state["positions"][symbol]
        self._save_state()
        return pnl

    def calculate_stats(self, lookback=100):
        history = self.state.get("history", [])
        if not history: return {"win_rate": 0, "total_trades": 0}
        trades = history[-lookback:]
        wins = sum(1 for t in trades if t.get('PnL', 0) > 0)
        return {"win_rate": (wins/len(trades))*100, "total_trades": len(history)}

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=4)
        except: pass
