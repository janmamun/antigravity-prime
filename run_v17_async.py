
import asyncio
import json
import os
import time
import pandas as pd
import numpy as np
import ccxt.pro
from datetime import datetime
from market_scanner import MarketScanner
from simulation_engine import FuturesSimulator
from meta_brain import LocalBrain, MetaBrain

# Constants
HEARTBEAT_FILE = "guardian_heartbeat_async.json"
TOP_LIMIT = 20 

def log_sovereign(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {msg}"
    print(formatted_msg)
    with open("bot_output_async.log", "a") as f:
        f.write(formatted_msg + "\n")

def update_heartbeat():
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            json.dump({"last_heartbeat": time.time(), "status": "ALIVE_ASYNC"}, f)
    except:
        pass

class AsyncSovereignEngine:
    def __init__(self):
        self.scanner = MarketScanner()
        self.brain = LocalBrain()
        self.meta = MetaBrain()
        self.sim = FuturesSimulator()
        
        self.spot_ws = ccxt.pro.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        })
        
        self.price_buffer = {} 
        self.is_running = True
        self.trade_count = 0
        self.last_rebalance = 0 
        self.last_live_symbols = set() 
        self.proxy_mgr = self.scanner.bot.proxy_mgr 


    async def update_heartbeat_loop(self):
        while self.is_running:
            update_heartbeat()
            if self.scanner.bot.telegram.is_active:
                now = time.time()
                if not hasattr(self, 'last_telegram_heartbeat'):
                    self.last_telegram_heartbeat = 0
                
                if now - self.last_telegram_heartbeat > 3600: 
                    try:
                        is_live = self.scanner.bot.is_live
                        if is_live:
                            equity = await self._engine_fetch_with_proxy(self.scanner.bot, 'get_live_balance')
                            raw_pos = await self._engine_fetch_with_proxy(self.scanner.bot, 'fetch_positions')
                            active_count = len([p for p in raw_pos if float(p.get('contracts', 0)) != 0])
                        else:
                            status = self.sim.get_portfolio_status(self.price_buffer)
                            equity = status['equity']
                            active_count = len(status['positions'])
                            
                        await self.scanner.bot.telegram.send_heartbeat(equity, active_count)
                        self.last_telegram_heartbeat = now
                    except Exception as e:
                        log_sovereign(f"Telegram Heartbeat Error: {e}", "ERROR")
            
            await asyncio.sleep(10)

    async def watch_prices(self):
        log_sovereign("🛰️ WS Sentinel active", "SYSTEM")
        while self.is_running:
            try:
                top_symbols = await self.scanner.get_top_volume_coins(limit=10) 
                active_symbols = list(self.sim.get_positions().keys())
                symbols = list(set(top_symbols + active_symbols))
                
                for symbol in symbols:
                    try:
                        # Try WS first
                        ticker = await asyncio.wait_for(self.spot_ws.watch_ticker(symbol), timeout=5.0)
                        self.price_buffer[symbol] = ticker['last']
                    except Exception:
                        # Fallback to REST via engine proxy loop
                        try:
                            ticker = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_ticker', symbol)
                            self.price_buffer[symbol] = ticker['last']
                        except:
                            pass
                    await asyncio.sleep(0.1)
                await asyncio.sleep(1)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                log_sovereign(f"WS/Sentinel Loop Error: {e}\n{error_trace}", "ERROR")
                await asyncio.sleep(5)

    async def _engine_fetch_with_proxy(self, obj, method_name, *args, **kwargs):
        import operator
        proxy_attempts = [None] + [self.proxy_mgr.get_proxy() for _ in range(3)]
        exchange = obj if hasattr(obj, 'proxies') else getattr(obj, 'exchange', None)
        
        for current_proxy in proxy_attempts:
            if exchange:
                exchange.proxies = current_proxy if current_proxy else {}

            try:
                # Resolve nested methods (e.g., 'market_exch.fetch_order_book')
                method = operator.attrgetter(method_name)(obj)
                if asyncio.iscoroutinefunction(method):
                    return await method(*args, **kwargs)
                else:
                    return await asyncio.to_thread(method, *args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                if any(code in err_msg for code in ["418", "429", "1003", "451", "404", "permission denied"]):
                    if current_proxy and exchange: self.proxy_mgr.report_failure(current_proxy)
                    await asyncio.sleep(1)
                    continue
                else:
                    raise e
        raise Exception(f"Engine Proxy Exhausted for {method_name}")

    async def run_logic_cycle(self):
        log_sovereign("🚀 ASYNC SOVEREIGN CYCLE ACTIVE", "SYSTEM")
        
        while self.is_running:
            try:
                is_live = self.scanner.bot.is_live
                if is_live:
                    try:
                        live_positions = await self._engine_fetch_with_proxy(self.scanner.bot, 'get_active_positions')
                        curr_live_symbols = {p['symbol'] for p in live_positions}
                        
                        closed_symbols = self.last_live_symbols - curr_live_symbols
                        for sym in closed_symbols:
                            log_sovereign(f"⚔️ [STEEL_EXIT] {sym} Closed. Cleaning up stale orders...", "SYSTEM")
                            
                            # CRITICAL FIX: Cancel all pending orders for this symbol to prevent phantom trades
                            try:
                                open_orders = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_open_orders', sym)
                                if open_orders:
                                    log_sovereign(f"🧹 [ORDER_CLEANUP] Cancelling {len(open_orders)} stale orders for {sym}", "SYSTEM")
                                    for order in open_orders:
                                        try:
                                            await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'cancel_order', order['id'], sym)
                                        except:
                                            pass
                            except Exception as e:
                                log_sovereign(f"⚠️ [ORDER_CLEANUP] Failed to cancel orders for {sym}: {e}", "ERROR")
                            
                            await self._engine_fetch_with_proxy(self.scanner.bot, 'get_live_balance')
                            try:
                                incomes = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fapiPrivateGetIncome', {'symbol': sym.replace('/', ''), 'limit': 1})
                                if incomes:
                                    pnl = float(incomes[0]['income'])
                                    self.scanner.bot._update_cortex_evolution(pnl > 0)
                                    
                                    # ALERT: Notify on significant losses
                                    if pnl < -10:
                                        log_sovereign(f"🚨 [LOSS ALERT] {sym} closed with ${pnl:.2f} loss!", "CRITICAL")
                                        # macOS notification
                                        import subprocess
                                        subprocess.run(['osascript', '-e', f'display notification "Loss ${pnl:.2f} on {sym}" with title "⚠️ Trading Alert"'], capture_output=True)
                            except: pass
                        self.last_live_symbols = curr_live_symbols

                        if live_positions:
                            log_sovereign(f"Monitoring {len(live_positions)} LIVE positions...", "EXECUTOR")
                            await self._engine_fetch_with_proxy(self.scanner.bot, 'check_liquidation_guard', live_positions)
                            
                            for pos in live_positions:
                                sym = pos['symbol']
                                if sym in self.price_buffer:
                                    curr_price = self.price_buffer[sym]
                                    initial_sl = pos.get('sl', pos['entry'])
                                    new_sl = await self._engine_fetch_with_proxy(self.scanner.bot, 'trail_liquidity_sl', sym, curr_price, initial_sl, pos['side'])
                                    if new_sl != initial_sl:
                                        log_sovereign(f"🛡️ LIVE TRAILING SL: Moving {sym} Protection to ${new_sl:.2f}", "SYSTEM")
                                        await self._engine_fetch_with_proxy(self.scanner.bot, 'update_live_stop_loss', sym, pos['side'], new_sl)
                    except Exception as e:
                        log_sovereign(f"Live Monitoring Error: {e}", "ERROR")

                sim_positions = self.sim.get_positions()
                if sim_positions:
                    log_sovereign(f"Monitoring {len(sim_positions)} SIM positions...", "EXECUTOR")
                    sim_prices = {s: self.price_buffer.get(s) for s in sim_positions.keys() if s in self.price_buffer}
                    if sim_prices:
                        notifications = self.sim.monitor_orders(sim_prices)
                        for note in notifications: 
                            log_sovereign(note, "EXECUTOR")
                            if "EXIT" in note:
                                self.scanner.bot._update_cortex_evolution("PROFIT" in note)
                        
                        for sym, pos in sim_positions.items():
                            curr_price = sim_prices.get(sym)
                            if curr_price:
                                # Phase 33: Tighter trail for Alpha
                                alpha_mode = pos.get('alpha_mode', False)
                                new_sl = await asyncio.to_thread(self.scanner.bot.trail_liquidity_sl, sym, curr_price, pos['sl'], pos['side'], alpha_mode=alpha_mode)
                                if new_sl != pos['sl']:
                                    log_sovereign(f"🛡️ SIM TRAILING SL: Moving {sym} Protection to ${new_sl:.2f} {'[ALPHA]' if alpha_mode else ''}", "SYSTEM")
                                    pos['sl'] = new_sl

                log_sovereign(f"Scanning Market Matrix (Top {TOP_LIMIT})...", "SCANNER")
                opportunities = await self.scanner.scan_market()
                
                if opportunities:
                    log_sovereign(f"🔍 [ENGINE DEBUG] Scanner returned {len(opportunities)} opportunities.", "DEBUG")
                    for top_trade in opportunities:
                        if top_trade['signal'] in ["BUY", "SELL", "LONG", "SHORT"]:
                            log_sovereign(f"SOVEREIGN SIGNAL: {top_trade['symbol']} | {top_trade['signal']} | SCORE: {top_trade['score']}", "EXECUTOR")
                            curr_price = self.price_buffer.get(top_trade['symbol'], top_trade.get('price'))
                            
                            if curr_price is not None:
                                try:
                                    ob = await self._engine_fetch_with_proxy(self.scanner.bot, 'market_exch.fetch_order_book', top_trade['symbol'], limit=5)
                                    best_bid = ob['bids'][0][0]
                                    best_ask = ob['asks'][0][0]
                                    spread = (best_ask - best_bid) / best_bid
                                    
                                    # Phase 32: Dynamic Slippage Logic
                                    score = top_trade.get('score', 0)
                                    max_allowed_spread = 0.0015 # Default 0.15%
                                    if score > 90: max_allowed_spread = 0.0025 # 0.25% for A+ setups
                                    elif score > 80: max_allowed_spread = 0.0020 # 0.20% for A setups
                                    
                                    if spread > max_allowed_spread: 
                                        log_sovereign(f"🛡️ SLIPPAGE GUARD: Spread {spread:.4f} > Limit {max_allowed_spread:.4f} for {top_trade['symbol']} (Score: {score})", "SYSTEM")
                                        continue
                                except Exception as e:
                                    log_sovereign(f"⚠️ [ENGINE] Orderbook Error: {e}", "DEBUG")
                                    pass

                                if is_live:
                                    # Phase 35: Margin Guard
                                    try:
                                        bal = await asyncio.to_thread(self.scanner.bot.exchange.fetch_balance)
                                        avail_usdt = float(bal['free'].get('USDT', 0))
                                        
                                        # Phase 40: Dynamic Position Sizing
                                        score = top_trade.get('score', 0)
                                        alpha_mode = top_trade.get('alpha_mode', False)
                                        
                                        if alpha_mode:
                                            # Alpha trades: $25-50 based on score
                                            amount = 50 if score > 70 else 25
                                        else:
                                            # Standard trades: $50-150 based on conviction
                                            if score > 90: amount = 150
                                            elif score > 75: amount = 100
                                            elif score > 60: amount = 75
                                            else: amount = 50
                                        
                                        if avail_usdt < amount:
                                            log_sovereign(f"🛡️ [MARGIN GUARD] Insufficient USDT ({avail_usdt:.2f} < {amount}). Skipping {top_trade['symbol']}", "SYSTEM")
                                            continue
                                            
                                    except Exception as e:
                                        log_sovereign(f"⚠️ [MARGIN GUARD] Balance Check Failed: {e}", "DEBUG")
                                        amount = 25 # Fallback to minimum
                                    
                                    # Phase 40: Symbol Normalization (PEPE -> 1000PEPE)
                                    exec_symbol = top_trade['symbol']
                                    if exec_symbol == 'PEPE/USDT':
                                        exec_symbol = '1000PEPE/USDT:USDT'
                                    elif '/USDT' in exec_symbol and ':USDT' not in exec_symbol:
                                        exec_symbol = exec_symbol + ':USDT'
                                    
                                    log_sovereign(f"🚀 LIVE EXECUTION: {exec_symbol} | {top_trade['signal']} | Allocation: ${amount} (Score: {score})", "SYSTEM")
                                    res = await asyncio.to_thread(
                                        self.scanner.bot.execute_live_order,
                                        exec_symbol, top_trade['signal'], 
                                        amount, curr_price, top_trade['tp'], top_trade['sl'],
                                        turbo_scaling=top_trade.get('turbo_scaling', False)
                                    )
                                    log_sovereign(f"LIVE RESULT: {res.get('msg')}", "SYSTEM")
                                
                                # Simulation with same dynamic sizing
                                score = top_trade.get('score', 0)
                                alpha_mode = top_trade.get('alpha_mode', False)
                                if alpha_mode:
                                    amount = 50 if score > 70 else 25
                                else:
                                    if score > 90: amount = 150
                                    elif score > 75: amount = 100
                                    elif score > 60: amount = 75
                                    else: amount = 50
                                self.sim.execute_trade(top_trade['symbol'], top_trade['signal'], curr_price, amount, top_trade['tp'], top_trade['sl'], alpha_mode=alpha_mode)
                                self.trade_count += 1

                        else:
                            # Skip neutral/wait signals
                            continue

                await asyncio.sleep(30)

            except Exception:
                import traceback
                error_trace = traceback.format_exc()
                log_sovereign(f"🛡️ [UNIVERSAL SHIELD] Recovered from Cycle Crash:\n{error_trace}", "CRITICAL")
                await asyncio.sleep(20) # Breather

    async def _harvest_worker(self):
        """Phase 34: Sovereign Harvest Background Worker (Every 4 Hours)"""
        log_sovereign("🌾 Harvest Sentinel active", "SYSTEM")
        while self.is_running:
            try:
                # Synchronous CCXT calls wrapped in to_thread
                res = await asyncio.to_thread(self.scanner.bot.harvest_spot_profits)
                if res.get('status') == 'SUCCESS':
                    log_sovereign(f"💰 [HARVEST SUCCESS] {res.get('msg')}", "SYSTEM")
                elif res.get('status') == 'ERROR':
                     log_sovereign(f"⚠️ [HARVEST ERROR] {res.get('msg')}", "ERROR")
                
                # Also log yield recommendation
                yield_info = await asyncio.to_thread(self.scanner.bot.manage_dual_investment)
                if yield_info and yield_info.get('recommendation'):
                    log_sovereign(f"💡 [YIELD TIP] {yield_info.get('recommendation')} | {yield_info.get('expected_yield')}", "SYSTEM")
                
            except Exception as e:
                log_sovereign(f"Harvest Worker Exception: {e}", "ERROR")
            
            await asyncio.sleep(4 * 3600) # 4 hours

    async def _daily_harvest_worker(self):
        """Phase 37: Daily Automated Harvest (Every 24 Hours)"""
        log_sovereign("🗓️ Daily Profit Lock Sentinel active", "SYSTEM")
        while self.is_running:
            try:
                # Check for surplus appreciation above $10k baseline
                res = await asyncio.to_thread(self.scanner.bot.daily_capital_sentinel, base_usd=10000)
                if res.get('status') == 'SUCCESS':
                    log_sovereign(f"💰 [DAILY HARVEST SUCCESS] {res.get('msg')}", "SYSTEM")
                elif res.get('status') == 'ERROR':
                     log_sovereign(f"⚠️ [DAILY HARVEST ERROR] {res.get('msg')}", "ERROR")
                else:
                    log_sovereign(f"📡 [DAILY HARVEST] {res.get('msg')}", "DEBUG")
                
            except Exception as e:
                log_sovereign(f"Daily Harvest Worker Exception: {e}", "ERROR")
            
            # Check once every 24 hours
            await asyncio.sleep(24 * 3600)

    async def start(self):
        while self.is_running:
            try:
                self.heartbeat_task = asyncio.create_task(self.update_heartbeat_loop())
                self.watch_task = asyncio.create_task(self.watch_prices())
                self.logic_task = asyncio.create_task(self.run_logic_cycle())
                self.harvest_task = asyncio.create_task(self._harvest_worker()) # Phase 34
                self.daily_lock_task = asyncio.create_task(self._daily_harvest_worker()) # Phase 37
                
                await asyncio.gather(self.heartbeat_task, self.watch_task, self.logic_task, self.harvest_task, self.daily_lock_task)
            except Exception as e:
                log_sovereign(f"WATCHDOG: Engine Crash Detected ({e}). Restarting in 30s...", "CRITICAL")
                await asyncio.sleep(30)

if __name__ == "__main__":
    engine = AsyncSovereignEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        log_sovereign("Sovereign Protocol Terminated by User.", "SYSTEM")
    except Exception as e:
        log_sovereign(f"FATAL SYSTEM ERROR: {e}", "CRITICAL")
