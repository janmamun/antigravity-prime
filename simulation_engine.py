
import pandas as pd
import json
import os
import ccxt
import random
from datetime import datetime, timedelta

class FuturesSimulator:
    def __init__(self, mode='PAPER', api_key=None, api_secret=None, initial_balance=10000, leverage=3):
        self.mode = mode
        self.state_file = "sim_state.json"
        self.leverage = leverage
        self.exchange = None
        
        # Live Trading Setup
        if self.mode == 'LIVE' and api_key and api_secret:
            try:
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
                print("⚡ LIVE TRADING ENGINE ARMED")
            except Exception as e:
                print(f"Failed to connect to Exchange: {e}")
                self.mode = 'PAPER' # Fallback
        
        # Load or Init State
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                "balance": initial_balance,
                "initial_balance": initial_balance,
                "positions": {}, 
                "history": [],
                "last_funding_check": datetime.now().isoformat()
            }

    def get_positions(self):
        return self.state["positions"]

    def _apply_funding_fees(self, current_prices):
        """Deduct funding fees from balance every 8 hours"""
        now = datetime.now()
        last_check = datetime.fromisoformat(self.state.get("last_funding_check", now.isoformat()))
        
        if now - last_check >= timedelta(hours=8):
            total_fee = 0
            for symbol, pos in self.state["positions"].items():
                if symbol not in current_prices: continue
                # Realistically, we should fetch actual funding rate
                # For simulation, we assume 0.01% standard
                funding_rate = 0.0001 
                position_value = pos['size'] * current_prices[symbol]
                fee = position_value * funding_rate
                # Fee is paid by LONG if positive, by SHORT if negative
                if pos['side'] == 'LONG': total_fee += fee
                else: total_fee -= fee
            
            self.state["balance"] -= total_fee
            self.state["last_funding_check"] = now.isoformat()
            self._save_state()

    def get_portfolio_status(self, current_prices):
        self._apply_funding_fees(current_prices)
        equity = self.state["balance"]
        unrealized_pnl = 0
        margin_used = 0
        positions_summary = []
        
        for symbol, pos in self.state["positions"].items():
            if symbol not in current_prices: continue
            
            cp = current_prices[symbol]
            ep = pos['entry_price']
            sz = pos['size']
            side = pos['side']
            
            pnl = (cp - ep) * sz if side in ["LONG", "BUY"] else (ep - cp) * sz
            margin = (ep * sz) / self.leverage
            
            unrealized_pnl += pnl
            margin_used += margin
            
            positions_summary.append({
                "Symbol": symbol, "Side": side, "Size": sz,
                "Entry": ep, "Mark": cp, "PnL ($)": pnl, 
                "PnL (%)": (pnl/margin)*100 if margin else 0
            })
            
        equity += unrealized_pnl
        return {
            "equity": equity, "margin_used": margin_used, 
            "free_margin": equity - margin_used,
            "unrealized_pnl": unrealized_pnl,
            "positions": positions_summary
        }

    def monitor_orders(self, current_prices):
        notifications = []
        symbols = list(self.state["positions"].keys())
        
        for symbol in symbols:
            if symbol not in current_prices: continue
            
            price = current_prices[symbol]
            pos = self.state["positions"][symbol]
            side = pos['side']
            is_long = side in ["LONG", "BUY"]
            tp = pos.get('tp', 0)
            sl = pos.get('sl', 0)
            atr = pos.get('atr', 0)
            partial_taken = pos.get('partial_taken', False)
            
            triggered = False
            reason = ""
            
            # --- Trailing Stop Logic ---
            if atr > 0:
                activation_price = pos['entry_price'] + (atr * 1.5) if is_long else pos['entry_price'] - (atr * 1.5)
                is_active = (price >= activation_price) if is_long else (price <= activation_price)
                
                if is_active:
                    new_sl = price - (atr * 1.0) if is_long else price + (atr * 1.0)
                    # Only move SL in our favor
                    if is_long and new_sl > sl: 
                        pos['sl'] = new_sl
                    elif not is_long and (sl == 0 or new_sl < sl):
                        pos['sl'] = new_sl

            # --- Partial Profit Logic (Close 50% at 1.5x ATR) ---
            if not partial_taken and atr > 0:
                p_tp = pos['entry_price'] + (atr * 1.5) if is_long else pos['entry_price'] - (atr * 1.5)
                hit_p = (price >= p_tp) if is_long else (price <= p_tp)
                
                if hit_p:
                    self.close_partial(symbol, price, 0.5)
                    notifications.append(f"{symbol}: Partial Profit Taken (50%) at ${price:.2f}")

            # --- Hard TP/SL ---
            if is_long:
                if tp > 0 and price >= tp: triggered, reason = True, "TAKE PROFIT 🎯"
                elif sl > 0 and price <= sl: triggered, reason = True, "STOP LOSS 🛑"
            else:
                if tp > 0 and price <= tp: triggered, reason = True, "TAKE PROFIT 🎯"
                elif sl > 0 and price >= sl: triggered, reason = True, "STOP LOSS 🛑"
            
            if triggered:
                pnl = self.close_position(symbol, price)
                notifications.append(f"{symbol}: {reason} Triggered at ${price:.2f} (PnL: ${pnl:.2f})")
                
        return notifications

    def execute_trade(self, symbol, side, price, amount_usd, tp=0, sl=0, atr=0):
        # Apply slippage for PAPER mode
        if self.mode == 'PAPER':
            slippage = random.uniform(0.0001, 0.0005)
            price = price * (1 + slippage) if side == 'LONG' else price * (1 - slippage)

        margin_req = amount_usd / self.leverage
        if margin_req > self.state["balance"]:
            return {"status": "error", "msg": "Insufficient Margin"}
            
        # DCA Mode: Support multiple entries by averaging
        if symbol in self.state["positions"]:
            pos = self.state["positions"][symbol]
            new_size = pos['size'] + ((margin_req * self.leverage) / price)
            new_entry = ((pos['entry_price'] * pos['size']) + (price * ((margin_req * self.leverage) / price))) / new_size
            pos['entry_price'] = new_entry
            pos['size'] = new_size
            pos['tp'] = tp # Update TP/SL if needed
            pos['sl'] = sl
            self._save_state()
            return {"status": "success", "msg": f"DCA Entry for {symbol} @ ${price:.2f}"}
            
        self.state["positions"][symbol] = {
            "entry_price": price,
            "size": (margin_req * self.leverage) / price,
            "side": side,
            "tp": tp,
            "sl": sl,
            "atr": atr,
            "partial_taken": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self._save_state()
        return {"status": "success", "msg": f"Opened {side} {symbol} @ ${price:.2f}"}

    def close_partial(self, symbol, price, percent):
        if symbol not in self.state["positions"]: return
        pos = self.state["positions"][symbol]
        close_size = pos['size'] * percent
        
        # Calculate PnL for the closed part
        if pos['side'] in ['LONG', 'BUY']: pnl = (price - pos['entry_price']) * close_size
        else: pnl = (pos['entry_price'] - price) * close_size
        
        self.state["balance"] += pnl
        pos['size'] -= close_size
        pos['partial_taken'] = True
        self._save_state()

    def close_position(self, symbol, price):
        if symbol not in self.state["positions"]: return 0
        pos = self.state["positions"][symbol]
        quantity = pos['size']
        entry_price = pos['entry_price']
        side = pos['side']
        
        if side in ['LONG', 'BUY']: pnl = (price - entry_price) * quantity
        else: pnl = (entry_price - price) * quantity
            
        self.state["balance"] += pnl
        self.state["history"].append({
            "Symbol": symbol, "Entry": entry_price, "Exit": price,
            "PnL": pnl, "Side": side, "Time": datetime.now().isoformat()
        })
        
        del self.state["positions"][symbol]
        self._save_state()
        return pnl

    def calculate_stats(self):
        """Calculate Win Rate and Stats for Dashboard"""
        trades = self.state.get("history", [])
        if not trades:
            return {"win_rate": 0.0, "total_pnl": 0.0, "total_trades": 0}
            
        wins = sum(1 for t in trades if t.get('PnL', 0) > 0)
        total = len(trades)
        win_rate = (wins / total) * 100 if total > 0 else 0
        total_pnl = sum(t.get('PnL', 0) for t in trades)
        
        return {
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_trades": total
        }

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)
