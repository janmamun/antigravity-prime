
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
from geometric_prophet import get_geometric_signal

# Constants
HEARTBEAT_FILE = "guardian_heartbeat_async.json"
TOP_LIMIT = 20

# Profit-Locking Thresholds
PROFIT_LOCK_PCT = 2.5    # Aggressive Offensive: +2.5% ROI
BREAKEVEN_PCT = 2.0      # Exact Golden Era: +2% ROI
MAX_STOP_LOSS_PCT = 3.0

# Daily Loss Circuit Breaker
DAILY_LOSS_LIMIT_PCT = 5.0  # Stop trading if down 5% for the day
SESSION_EQUITY_FILE = "session_start_equity.json"  # Persist across restarts
SESSION_PEAK_FILE = "session_peak_equity.json"

# Phase 48: Intelligence Upgrade
TRAILING_DRAWDOWN_HALT_PCT = 10.0  # Looser for Hyper-Growth
WINNING_COOL_DOWN_SECONDS = 3600  # 1-hour cooldown

# Phase 49: The Iron Gate
LOSS_COOL_DOWN_SECONDS = 1800    # 30-min cooldown after loss (Aggressive)

# Phase 79: Stale Anchor Protection
MAX_POSITION_AGE_SECONDS = 72 * 3600  # 72 hours

# Phase 69: Hyper-Optimization
SILENT_MODE = True              # Reduce IO/Logging overhead
LAST_DIAGNOSTIC_LOG = 0
DIAGNOSTIC_COOLDOWN = 300       # Log diagnostics every 5 minutes only

# Phase 51: The Sovereign Harvest 2.0
HARVEST_1_USD = 5.0            # Exact Golden Era Trigger
HARVEST_1_RATIO = 0.5          # Sell 50%
GIVEBACK_THRESHOLD = 0.25      # Exit rest if 25% of peak profit is lost
HARVEST_STATE_FILE = "harvest_state.json"

from news_syndicate import NewsSyndicate
from telegram_bridge import TelegramBridge

def get_harvest_state():
    try:
        if os.path.exists(HARVEST_STATE_FILE):
            with open(HARVEST_STATE_FILE, 'r') as f:
                return json.load(f)
    except: pass
    return {"partial_harvests": {}, "peak_profits": {}}

def save_harvest_state(partial_harvests, peak_profits):
    try:
        with open(HARVEST_STATE_FILE, 'w') as f:
            json.dump({"partial_harvests": partial_harvests, "peak_profits": peak_profits}, f)
    except: pass

_h_state = get_harvest_state()
PARTIAL_HARVEST_STORE = _h_state.get("partial_harvests", {})
PEAK_PROFIT_STORE = _h_state.get("peak_profits", {})

def get_session_start_equity():
    """Load persisted session equity or return None"""
    try:
        if os.path.exists(SESSION_EQUITY_FILE):
            with open(SESSION_EQUITY_FILE, 'r') as f:
                data = json.load(f)
                # Check if it's from today (reset daily)
                stored_date = data.get('date')
                today = datetime.now().strftime('%Y-%m-%d')
                if stored_date == today:
                    return data.get('equity')
    except:
        pass
    return None

def set_session_start_equity(equity):
    """Persist session equity for today"""
    try:
        data = {'equity': equity, 'date': datetime.now().strftime('%Y-%m-%d')}
        with open(SESSION_EQUITY_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def get_session_peak_equity():
    try:
        if os.path.exists(SESSION_PEAK_FILE):
            with open(SESSION_PEAK_FILE, 'r') as f:
                data = json.load(f)
                if data.get('date') == datetime.now().strftime('%Y-%m-%d'):
                    return data.get('peak')
    except Exception as e:
        if not SILENT_MODE: log_sovereign(f"Peak Equity Read Error: {e}", "DEBUG")
        pass
    return None

def set_session_peak_equity(peak):
    try:
        data = {'peak': peak, 'date': datetime.now().strftime('%Y-%m-%d')}
        with open(SESSION_PEAK_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        log_sovereign(f"Peak Equity Write Error: {e}", "ERROR")
        pass

SESSION_START_EQUITY = get_session_start_equity()  # Load from file on startup
SESSION_PEAK_EQUITY = get_session_peak_equity()

def sync_financial_baseline(current_equity):
    """Force reset peak if it's significantly higher than reality (Phantom Peak Fix)"""
    global SESSION_PEAK_EQUITY, SESSION_START_EQUITY
    
    # If peak is >5% higher than start (and we are at start), or if peak is None
    if SESSION_PEAK_EQUITY is None:
        SESSION_PEAK_EQUITY = current_equity
        set_session_peak_equity(current_equity)
        
    # Phase 75: Hard Baseline Override
    # If the user has manually reset files to 267.28, but we see 306.95 in memory
    # We prioritize the session files + real-time current balance
    if SESSION_PEAK_EQUITY > current_equity * 1.10: # 10% discrepancy
        log_sovereign(f"⚠️ [FINANCIAL] Phantom Peak detected (${SESSION_PEAK_EQUITY}). Resetting to current: ${current_equity}", "CRITICAL")
        SESSION_PEAK_EQUITY = current_equity
        set_session_peak_equity(current_equity)



# Symbol format conversion helpers
def binance_to_ccxt(symbol):
    """Convert Binance format (GUNUSDT) to CCXT format (GUN/USDT)"""
    if symbol.endswith('USDT'):
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol

def ccxt_to_binance(symbol):
    """Convert CCXT format (GUN/USDT:USDT) to Binance format (GUNUSDT)"""
    return symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace("/", "")

# Position SL tracking store (symbol -> stop_loss price)
POSITION_SL_STORE = {}

def log_sovereign(msg, level="INFO"):
    global LAST_DIAGNOSTIC_LOG
    # Phase 69: Silent Mode filtering for diagnostic logs
    if SILENT_MODE and level in ["EXECUTOR", "SCANNER", "DEBUG"]:
        if time.time() - LAST_DIAGNOSTIC_LOG < DIAGNOSTIC_COOLDOWN:
            return
        LAST_DIAGNOSTIC_LOG = time.time()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {msg}"
    print(formatted_msg)
    try:
        with open("bot_output_async.log", "a") as f:
            f.write(formatted_msg + "\n")
    except Exception: pass



def update_heartbeat():
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            json.dump({"last_heartbeat": time.time(), "status": "ALIVE_ASYNC"}, f)
    except Exception as e:
        pass

# Phase 53: Instance Guard (Atomic Singleton Lock)
LOCK_FILE = "bot.lock"

def acquire_lock():
    try:
        # Atomic file creation with O_EXCL prevents race conditions
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        # Check if the existing PID is actually running
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            if not os.path.exists(f"/proc/{pid}") and not os.kill(pid, 0) is None: 
                pass # This logic might vary by OS, simplified for Mac
        except (ProcessLookupError, ValueError, OSError):
            # Stale lock - process is dead
            try:
                os.remove(LOCK_FILE)
                return acquire_lock() # Try again
            except Exception: pass
        return False
    except Exception:
        return False

def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass

class AsyncSovereignEngine:
    def __init__(self):
        self.scanner = MarketScanner()
        self.brain = LocalBrain()
        self.meta = MetaBrain()
        self.sim = FuturesSimulator()
        self.telegram = TelegramBridge()
        
        self.spot_ws = ccxt.pro.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        })
        
        self.price_buffer = {} 
        self.symbol_regimes = {} # symbol -> regime status
        self.is_running = True
        self.trade_count = 0
        self.last_rebalance = 0 
        self.last_live_symbols = set() 
        self.proxy_mgr = self.scanner.bot.proxy_mgr 
        self.task_restart_count = 0 
        self.max_restarts_per_hour = 10 
        self.restart_timestamps = [] 
        self.pos_cooldowns = {}

        # Phase 64: Uptime Watchdog
        self.last_pulse_time = time.time()
        
        # AI-Guardian Kill-Switch State
        self.kill_switch_file = "kill_switch.json"
        
        # Send Start Notification


    async def safe_task(self, coro, name="Unknown"):
        """Wrapper to catch and log anomalies in background tasks"""
        try:
            return await coro
        except asyncio.CancelledError:
            log_sovereign(f"⚠️ [TASK] Task '{name}' cancelled.", "DEBUG")
            raise
        except Exception as e:
            log_sovereign(f"❌ [TASK CRITICAL] Task '{name}' failed: {e}", "ERROR")
            import traceback
            log_sovereign(traceback.format_exc(), "DEBUG")
            return None

    # ==== Phase 54: Atomic Shield Helper ====
    async def place_atomic_trade(self, ccxt_sym_full, side, amount, entry_price, tp_price=None, sl_price=None, reduce_only=False, alpha_mode=False):
        """Place a market order and immediately attach stop‑loss / take‑profit.
        Phase 83: Atomic Shield - Emergency close if SL protection fails.
        """
        try:
            # Market entry
            entry_order = await asyncio.to_thread(
                self.scanner.bot.exchange.create_order,
                ccxt_sym_full,
                'MARKET',
                side,
                amount,
                None,
                {'reduceOnly': reduce_only}
            )
            log_sovereign(f"[ATOMIC] {side} {ccxt_sym_full} qty={amount} price≈{entry_price:.4f} (reduceOnly={reduce_only})", "SYSTEM")
            
            # Attach SL (Priority 1)
            if sl_price is not None:
                try:
                    await self._engine_fetch_with_proxy(
                        self.scanner.bot,
                        'update_live_stop_loss',
                        ccxt_sym_full,
                        side,
                        sl_price
                    )
                    log_sovereign(f"[ATOMIC] SL set at {sl_price:.4f}", "SYSTEM")
                except Exception as e:
                    # Phase 83: Atomic Shield - Emergency Market Close on SL Failure
                    log_sovereign(f"🚨 [ATOMIC SHIELD] Failed to set SL for {ccxt_sym_full}: {e}. CLOSING NAKED POSITION IMMEDIATELY!", "CRITICAL")
                    close_side = 'sell' if side.upper() == 'BUY' else 'buy'
                    try:
                        await asyncio.to_thread(
                            self.scanner.bot.exchange.create_order,
                            ccxt_sym_full,
                            'MARKET',
                            close_side,
                            amount,
                            None,
                            {'reduceOnly': True}
                        )
                        log_sovereign(f"🛡️ [ATOMIC SHIELD] Emergency close successful for {ccxt_sym_full}.", "CRITICAL")
                    except Exception as fatal_e:
                        log_sovereign(f"💀 [FATAL] ATOMIC SHIELD FAILED TO CLOSE {ccxt_sym_full}: {fatal_e}", "ERROR")
                    return None # Signal failure
            
            # Attach TP (Priority 2)
            if tp_price is not None:
                try:
                    await asyncio.to_thread(
                        self.scanner.bot.exchange.create_order,
                        ccxt_sym_full,
                        'TAKE_PROFIT_MARKET',
                        'sell' if side.upper() == 'BUY' else 'buy',
                        amount,
                        None,
                        {'reduceOnly': True, 'stopPrice': tp_price}
                    )
                    log_sovereign(f"[ATOMIC] TP set at {tp_price:.4f}", "SYSTEM")
                except Exception as e:
                    log_sovereign(f"[ATOMIC] Failed to set TP: {e}", "ERROR")
            
            return entry_order
        except Exception as e:
            log_sovereign(f"❌ [ATOMIC] Trade Entry Failed: {e}", "ERROR")
            return None

    # ==== Phase 55: Liquidity Sentinel Helper ====
    async def check_liquidity(self, ccxt_sym_full, desired_qty, max_slippage_pct=0.002, depth=20, alpha_mode=False):
        """Phase 85.2: Verify depth + Natural Spread Guard before trading."""
        try:
            if not desired_qty or abs(desired_qty) < 1e-9: return True # Nothing to fill
            
            ob = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_order_book', ccxt_sym_full, limit=depth)
            if not ob or not ob['bids'] or not ob['asks'] or len(ob['bids']) == 0 or len(ob['asks']) == 0: 
                log_sovereign(f"🛡️ [LIQUIDITY] {ccxt_sym_full} rejected: Empty Orderbook detected.", "SYSTEM")
                return False
            
            mid_price = (ob['bids'][0][0] + ob['asks'][0][0]) / 2
            
            # 1. Natural Spread Guard
            natural_spread = (ob['asks'][0][0] - ob['bids'][0][0]) / mid_price
            if natural_spread > 0.003: # 0.3% limit
                log_sovereign(f"🛡️ [SPREAD GUARD] {ccxt_sym_full} rejected: Natural spread too high ({natural_spread*100:.2f}% > 0.3%).", "SYSTEM")
                return False

            # 2. Volume-Weighted Slippage Check
            side = 'asks' if desired_qty > 0 else 'bids'
            price_levels = ob.get(side, [])
            qty_to_fill = abs(desired_qty)
            
            total_vol = 0
            weighted_sum = 0
            for price, amount in price_levels:
                fill = min(amount, qty_to_fill - total_vol)
                weighted_sum += price * fill
                total_vol += fill
                if total_vol >= qty_to_fill: break
            
            if total_vol < qty_to_fill:
                log_sovereign(f"🛡️ [LIQUIDITY] {ccxt_sym_full} rejected: Insufficient book depth for {qty_to_fill} units.", "SYSTEM")
                return False
            
            avg_exec_price = weighted_sum / qty_to_fill
            slippage = abs(avg_exec_price - mid_price) / mid_price
            
            if slippage > max_slippage_pct:
                log_sovereign(f"🛡️ [LIQUIDITY] {ccxt_sym_full} rejected: Est. Slippage {slippage*100:.2f}% > {max_slippage_pct*100:.2f}% limit.", "CRITICAL")
                return False
                
            return True
        except Exception as e:
            log_sovereign(f"[LIQUIDITY] Check failed: {e}", "ERROR")
        return False

    async def emergency_close_all(self):
        """Phase 88: AI-Guardian Kill-Switch - Immediate full portfolio liquidation"""
        log_sovereign("🚨 [AI-GUARDIAN] KILL-SWITCH ENGAGED. LIQUIDATING ALL POSITIONS!", "CRITICAL")
        try:
            positions = await asyncio.to_thread(self.scanner.bot.get_active_positions)
            for pos in positions:
                symbol = pos['symbol']
                ccxt_symbol = binance_to_ccxt(symbol)
                side = 'sell' if pos['side'] == 'BUY' else 'buy'
                size = abs(pos['size'])
                
                log_sovereign(f"🛡️ [KILL-SWITCH] Closing {symbol} ({size})...", "SYSTEM")
                await asyncio.to_thread(
                    self.scanner.bot.exchange.create_order,
                    ccxt_symbol,
                    'MARKET',
                    side,
                    size,
                    None,
                    {'reduceOnly': True}
                )
            
            # Cancel all open orders
            log_sovereign("🛡️ [KILL-SWITCH] Purging all open orders...", "SYSTEM")
            await asyncio.to_thread(self.scanner.bot.exchange.fapiPrivateDeleteAllOpenOrders)
            
            log_sovereign("✅ [AI-GUARDIAN] Liquidation Complete. System Halted.", "CRITICAL")
        except Exception as e:
            log_sovereign(f"💀 [KILL-SWITCH FAILURE] Error during liquidation: {e}", "ERROR")

    async def _kill_switch_sentinel(self):
        """Monitor for kill_switch.json to trigger emergency shutdown"""
        while self.is_running:
            if os.path.exists(self.kill_switch_file):
                try:
                    with open(self.kill_switch_file, 'r') as f:
                        trigger = json.load(f)
                    if trigger.get("active", False):
                        await self.emergency_close_all()
                        # Move to .bak to prevent loop
                        os.rename(self.kill_switch_file, self.kill_switch_file + ".bak")
                        # self.is_running = False # Optionally stop engine
                except Exception as e:
                    log_sovereign(f"Kill-Switch Error: {e}", "DEBUG")
            await asyncio.sleep(5) # Frequent check

    async def update_heartbeat_loop(self):
        while self.is_running:
            update_heartbeat()
            # Telegram Heartbeat (Disabled)
            # if self.scanner.bot.telegram.is_active:
            #     now = time.time()
            #     if not hasattr(self, 'last_telegram_heartbeat'):
            #         self.last_telegram_heartbeat = 0
            #     
            #     if now - self.last_telegram_heartbeat > 3600: 
            #         try:
            #             is_live = self.scanner.bot.is_live
            #             if is_live:
            #                 equity = await self._engine_fetch_with_proxy(self.scanner.bot, 'get_live_balance')
            #                 raw_pos = await self._engine_fetch_with_proxy(self.scanner.bot, 'fetch_positions')
            #                 active_count = len([p for p in raw_pos if float(p.get('contracts', 0)) != 0])
            #             else:
            #                 status = self.sim.get_portfolio_status(self.price_buffer)
            #                 equity = status['equity']
            #                 active_count = len(status['positions'])
            #                 
            #             await self.scanner.bot.telegram.send_heartbeat(equity, active_count)
            #             self.last_telegram_heartbeat = now
            #         except Exception as e:
            #             log_sovereign(f"Telegram Heartbeat Error: {e}", "ERROR")
            
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
                    except (asyncio.TimeoutError, Exception) as e:
                        err_str = str(e).lower()
                        # Special handling for WebSocket disconnects & "Closing Transport" race conditions
                        if "closing transport" in err_str or "connection closed" in err_str or "connection_reset" in err_str or "clientconnectionreseterror" in err_str:
                            log_sovereign(f"🔌 [WS SENTINEL] Nuclear Reset active for {symbol} (Transport Corruption Detected)", "DEBUG")
                            try:
                                # Force clean shutdown of corrupted transport
                                if self.spot_ws:
                                    # Attempt to close the underlying transport gracefully if possible
                                    try:
                                        await asyncio.wait_for(self.spot_ws.close(), timeout=1.0)
                                    except Exception: pass
                            except Exception: pass
                            
                            # Phase 67: Core Re-initialization (Nuclear Option)
                            # Re-initialize the WS object if it's stuck in "closing" state
                            try:
                                await asyncio.sleep(1.0) # Wait for socket cleanup
                                self.spot_ws = ccxt.pro.binance({
                                    'enableRateLimit': True,
                                    'options': {'defaultType': 'spot'},
                                    'headers': {
                                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                                    }
                                })
                            except Exception: pass
                        
                        # Fallback to REST via engine proxy loop
                        try:
                            ticker = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_ticker', symbol)
                            self.price_buffer[symbol] = ticker['last']
                        except Exception:
                            pass
                    await asyncio.sleep(0.1)
                # Phase 64: Update Uptime Pulse
                self.last_pulse_time = time.time()
                
                await asyncio.sleep(60) 
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
        global SESSION_START_EQUITY
        log_sovereign("🚀 ASYNC SOVEREIGN CYCLE ACTIVE", "SYSTEM")
        
        # Phase 49: Atomic Startup Cleanup (The Clean Sweep)
        try:
            log_sovereign("🧹 [CLEAN SWEEP] Initializing Global Order Purge...", "SYSTEM")
            active_pos = await self._engine_fetch_with_proxy(self.scanner.bot, 'get_active_positions')
            active_symbols = {p['symbol'] for p in active_pos}
            
            # Fetch ALL open orders (with warning suppressed)
            self.scanner.bot.exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
            all_open_orders = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_open_orders')
            
            for order in all_open_orders:
                if order['symbol'] not in active_symbols:
                    log_sovereign(f"🧹 [CLEAN SWEEP] Purging ghost order {order['id']} for {order['symbol']}.", "SYSTEM")
                    await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'cancel_order', order['id'], order['symbol'])
        except Exception as e:
            log_sovereign(f"⚠️ [CLEAN SWEEP] Startup purge failed: {e}", "DEBUG")

        while self.is_running:
            # Phase 64: Socket Watchdog Pulse
            self.last_pulse_time = time.time()

            try:
                is_live = self.scanner.bot.is_live
                if is_live:
                    # CIRCUIT BREAKER: Check daily loss limit
                    try:
                        current_equity = await self._engine_fetch_with_proxy(self.scanner.bot, 'get_live_balance')
                        
                        # Phase 75: Financial Baseline Sync
                        sync_financial_baseline(current_equity)
                        
                        if SESSION_START_EQUITY is None:
                            globals()['SESSION_START_EQUITY'] = current_equity
                            set_session_start_equity(current_equity)  # Persist to file
                            log_sovereign(f"📊 [SESSION] Starting equity: ${SESSION_START_EQUITY:.2f}", "SYSTEM")
                        
                        if SESSION_START_EQUITY > 0:
                            drawdown_pct = ((SESSION_START_EQUITY - current_equity) / SESSION_START_EQUITY) * 100
                            if drawdown_pct >= DAILY_LOSS_LIMIT_PCT:
                                log_sovereign(f"🚨 [CIRCUIT BREAKER] Daily loss limit hit! Drawdown: {drawdown_pct:.1f}% >= {DAILY_LOSS_LIMIT_PCT}%. Skipping New Scans.", "CRITICAL")
                                if self.telegram.is_active:
                                    asyncio.create_task(self.telegram.send_message(f"🚨 *CIRCUIT BREAKER ACTIVATED*\nDaily loss limit hit: {drawdown_pct:.1f}% >= {DAILY_LOSS_LIMIT_PCT}%. Skipping new scans."))
                                # Skip to monitoring instead of fully continuing
                                goto_monitoring = True
                            else:
                                goto_monitoring = False
                        else:
                            goto_monitoring = False
                    except Exception as e:
                        log_sovereign(f"⚠️ [CIRCUIT BREAKER] Balance check failed: {e}", "DEBUG")
                        goto_monitoring = False
                    
                    # Phase 48: Trailing Session Protection
                    try:
                        global SESSION_PEAK_EQUITY
                        if SESSION_PEAK_EQUITY is None or current_equity > SESSION_PEAK_EQUITY:
                            globals()['SESSION_PEAK_EQUITY'] = current_equity
                            save_session_peak_equity(current_equity) # Persist it
                            log_sovereign(f"🏔️ [SESSION] New Peak Equity: ${SESSION_PEAK_EQUITY:.2f}", "SYSTEM")
                        
                        if SESSION_PEAK_EQUITY > 0:
                            trailing_drop = ((SESSION_PEAK_EQUITY - current_equity) / SESSION_PEAK_EQUITY) * 100
                            if trailing_drop >= TRAILING_DRAWDOWN_HALT_PCT:
                                log_sovereign(f"🚨 [TRAILING PROTECTION] Peak Drawdown hit! Drop: {trailing_drop:.1f}% from ${SESSION_PEAK_EQUITY:.2f}. Skipping New Scans.", "CRITICAL")
                                if self.telegram.is_active:
                                    asyncio.create_task(self.telegram.send_message(f"🚨 *TRAILING PROTECTION ACTIVATED*\nEquity dropped {trailing_drop:.1f}% from peak ${SESSION_PEAK_EQUITY:.2f}. Skipping new scans."))
                                goto_monitoring = True
                    except Exception as e:
                        if not SILENT_MODE: log_sovereign(f"Trailing Protection Check Error: {e}", "DEBUG")
                    
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
                                        except Exception:
                                            pass
                            except Exception as e:
                                log_sovereign(f"⚠️ [ORDER_CLEANUP] Failed to cancel orders for {sym}: {e}", "ERROR")
                            
                            await self._engine_fetch_with_proxy(self.scanner.bot, 'get_live_balance')
                            try:
                                incomes = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fapiPrivateGetIncome', {'symbol': sym.replace('/', ''), 'limit': 5})
                                if incomes:
                                    pnl = float(incomes[0]['income'])
                                    self.scanner.bot._update_cortex_evolution(pnl > 0)
                                    
                                    # Phase 45: Sovereign Memory Persistence
                                    try:
                                        ccxt_sym = sym if '/' in sym else f"{sym[:len(sym)-4]}/{sym[len(sym)-4:]}"
                                        # Phase 45/49: Update persistent memory per symbol (Safe PnL Source)
                                        # Filter to only get REALIZED_PNL
                                        for inc in incomes:
                                            if inc.get('incomeType') == 'REALIZED_PNL':
                                                pnl_val = float(inc.get('income', 0))
                                                win = pnl_val > 0
                                                self.scanner.bot._update_strategy_memory(ccxt_sym, win)
                                                
                                                # Phase 49: Loss Cool-Down (1-hour lock on failure)
                                                if not win:
                                                    log_sovereign(f"🧊 [LOSS COOLDOWN] Locking {ccxt_sym} for {LOSS_COOL_DOWN_SECONDS}s after failure.", "SYSTEM")
                                                    self.pos_cooldowns[ccxt_sym] = time.time() + LOSS_COOL_DOWN_SECONDS
                                                break
                                    except Exception: pass
                                    
                                    # ALERT: Notify on significant losses
                                    if pnl < -10:
                                        log_sovereign(f"🚨 [LOSS ALERT] {sym} closed with ${pnl:.2f} loss!", "CRITICAL")
                                        # macOS notification
                                        import subprocess
                                        subprocess.run(['osascript', '-e', f'display notification "Loss ${pnl:.2f} on {sym}" with title "⚠️ Trading Alert"'], capture_output=True)
                                        if self.telegram.is_active:
                                            asyncio.create_task(self.telegram.send_message(f"🚨 *LOSS ALERT*: {sym} closed with ${pnl:.2f} loss!"))
                                    
                                    # Phase 51: Reset Harvest States
                                    binance_sym = sym.replace('/', '')
                                    if binance_sym in PARTIAL_HARVEST_STORE:
                                        del PARTIAL_HARVEST_STORE[binance_sym]
                                    if binance_sym in PEAK_PROFIT_STORE:
                                        del PEAK_PROFIT_STORE[binance_sym]
                                    save_harvest_state(PARTIAL_HARVEST_STORE, PEAK_PROFIT_STORE)
                                    
                            except Exception as e:
                                log_sovereign(f"Symbol Cleanup Error: {e}", "ERROR")
                        self.last_live_symbols = curr_live_symbols

                        if live_positions:
                            # Phase 84: Sovereign State Reconciliation (Cleanup)
                            current_live_bi_symbols = set([p['symbol'] for p in live_positions])
                            stale_harvests = [s for s in PARTIAL_HARVEST_STORE.keys() if s not in current_live_bi_symbols]
                            stale_peaks = [s for s in PEAK_PROFIT_STORE.keys() if s not in current_live_bi_symbols]
                            
                            if stale_harvests or stale_peaks:
                                for s in stale_harvests: 
                                    if s in PARTIAL_HARVEST_STORE: del PARTIAL_HARVEST_STORE[s]
                                for s in stale_peaks: 
                                    if s in PEAK_PROFIT_STORE: del PEAK_PROFIT_STORE[s]
                                save_harvest_state(PARTIAL_HARVEST_STORE, PEAK_PROFIT_STORE)
                                log_sovereign(f"🧹 [STATE SYNC] Purged {len(stale_harvests)} stale harvest states.", "SYSTEM")

                            log_sovereign(f"Monitoring {len(live_positions)} LIVE positions...", "EXECUTOR")
                            await self._engine_fetch_with_proxy(self.scanner.bot, 'check_liquidation_guard', live_positions)
                            
                            for pos in live_positions:
                                binance_sym = pos['symbol']  # "GUNUSDT" format
                                ccxt_sym = binance_to_ccxt(binance_sym)  # "GUN/USDT" format
                                ccxt_sym_full = ccxt_sym + ":USDT"  # "GUN/USDT:USDT" for futures
                                
                                # Get current price from buffer (uses CCXT format)
                                curr_price = self.price_buffer.get(ccxt_sym)
                                if not curr_price:
                                    continue
                                
                                # Calculate ROI
                                entry = pos['entry']
                                size = abs(pos['size'])
                                leverage = pos.get('leverage', 1)
                                margin = (size * entry) / leverage if leverage > 0 else size * entry
                                unrealized_pnl = pos.get('unrealized_pnl', 0)
                                roi = (unrealized_pnl / margin) * 100 if margin > 0 else 0
                                
                                # Phase 79: Time-Stop Enforcement (Live)
                                try:
                                    update_time = int(pos.get('updateTime', 0)) / 1000 # ms to s
                                    age_seconds = time.time() - update_time
                                    if age_seconds > MAX_POSITION_AGE_SECONDS:
                                        log_sovereign(f"⏳ [TIME STOP] {binance_sym} open for {age_seconds/3600:.1f}h. Closing stale anchor.", "CRITICAL")
                                        close_side = 'sell' if pos['side'] == 'BUY' else 'buy'
                                        await self.place_atomic_trade(ccxt_sym_full, close_side, size, entry, reduce_only=True)
                                        continue
                                except Exception as e:
                                    if not SILENT_MODE: log_sovereign(f"Time Stop Error: {e}", "DEBUG")
                                                                # 🎯 PROFIT LOCK: Auto-close at +10% (Legacy, kept for higher ROI targets if needed)
                                if roi >= PROFIT_LOCK_PCT:
                                    try:
                                        log_sovereign(f"🎯 [PROFIT LOCK] {binance_sym} at +{roi:.1f}%! Closing for ${unrealized_pnl:.2f} profit!", "CRITICAL")
                                        close_side = 'sell' if pos['side'] == 'BUY' else 'buy'
                                        await self.place_atomic_trade(ccxt_sym_full, close_side, size, entry, sl_price=entry*0.97, reduce_only=True)
                                        # macOS notification for profit
                                        import subprocess
                                        subprocess.run(['osascript', '-e', f'display notification "Locked ${unrealized_pnl:.2f} profit on {binance_sym}" with title "🎯 Profit Secured!"'], capture_output=True)
                                        if self.telegram.is_active:
                                            asyncio.create_task(self.telegram.send_message(f"🎯 *PROFIT LOCK*: {binance_sym} closed at +{roi:.1f}% for ${unrealized_pnl:.2f} profit!"))
                                        
                                        # Phase 48: Winning Cool-Down Protocol
                                        log_sovereign(f"🧊 [COOL-DOWN] Locking {ccxt_sym} for {WINNING_COOL_DOWN_SECONDS}s after Profit Lock.", "SYSTEM")
                                        self.pos_cooldowns[ccxt_sym] = time.time() + WINNING_COOL_DOWN_SECONDS
                                        continue
                                    except Exception as e:
                                        log_sovereign(f"❌ [PROFIT LOCK] Failed to close {binance_sym}: {e}", "ERROR")

                                # 💎 SOVEREIGN HARVEST (Phase 51: Multi-Stage Exit)
                                if unrealized_pnl >= HARVEST_1_USD and not PARTIAL_HARVEST_STORE.get(binance_sym):
                                    # Phase 53: Adaptive Harvest Switch
                                    symbol_regime = self.symbol_regimes.get(binance_sym, "NEUTRAL")
                                    is_hunter_mode = (symbol_regime == "SOVEREIGN_TREND")
                                    
                                    if is_hunter_mode:
                                        log_sovereign(f"🏹 [HUNTER MODE] {binance_sym} in SOVEREIGN_TREND. Skipping harvest to capture moonshot. Locking SL to breakeven.", "CRITICAL")
                                        # Skip selling, just protect the capital
                                        # Atomic Breakeven Lock
                                        breakeven_sl = entry * 1.005
                                        log_sovereign(f"🛡️ [HUNTER LOCK] Moving SL to Breakeven ${breakeven_sl:.4f}", "SYSTEM")
                                        await self._engine_fetch_with_proxy(self.scanner.bot, 'update_live_stop_loss', ccxt_sym_full, pos['side'], breakeven_sl)
                                        PARTIAL_HARVEST_STORE[binance_sym] = True # Mark as "handled"
                                        save_harvest_state(PARTIAL_HARVEST_STORE, PEAK_PROFIT_STORE)
                                        if self.telegram.is_active:
                                            asyncio.create_task(self.telegram.send_message(f"🏹 *HUNTER MODE*: {binance_sym} in SOVEREIGN_TREND. SL moved to breakeven to capture moonshot."))
                                    else:
                                        # Farmer Mode (Standard Harvesting)
                                        try:
                                            log_sovereign(f"🌾 [HARVEST_50] {binance_sym} hit ${unrealized_pnl:.2f}. Selling {HARVEST_1_RATIO*100}% to bank win.", "CRITICAL")
                                            harvest_qty = size * HARVEST_1_RATIO
                                            close_side = 'sell' if pos['side'] == 'BUY' else 'buy'
                                            await self.place_atomic_trade(ccxt_sym_full, close_side, harvest_qty, entry, sl_price=entry*0.97, reduce_only=True)
                                            PARTIAL_HARVEST_STORE[binance_sym] = True
                                            save_harvest_state(PARTIAL_HARVEST_STORE, PEAK_PROFIT_STORE)
                                            
                                            # Atomic Breakeven Lock
                                            breakeven_sl = entry * 1.005
                                            log_sovereign(f"🛡️ [HARVEST LOCK] Moving SL to Breakeven ${breakeven_sl:.4f}", "SYSTEM")
                                            await self._engine_fetch_with_proxy(self.scanner.bot, 'update_live_stop_loss', ccxt_sym_full, pos['side'], breakeven_sl)
                                            POSITION_SL_STORE[binance_sym] = breakeven_sl
                                            
                                            import subprocess
                                            subprocess.run(['osascript', '-e', f'display notification "Harvested $5 profit on {binance_sym}. Remainder is now Risk-Free." with title "🌾 Harvest Successful"'], capture_output=True)
                                            if self.telegram.is_active:
                                                asyncio.create_task(self.telegram.send_message(f"🌾 *SOVEREIGN HARVEST*: {binance_sym} hit ${unrealized_pnl:.2f}. Sold {HARVEST_1_RATIO*100}% and moved SL to breakeven."))
                                        except Exception as e:
                                            log_sovereign(f"❌ [HARVEST] Failed: {e}", "ERROR")
                                    continue # Let the next cycle handle the remaining position

                                # 🏦 THE PROFIT VAULT (Dynamic Giveback Guard)
                                if unrealized_pnl > 0:
                                    peak = PEAK_PROFIT_STORE.get(binance_sym, 0)
                                    if unrealized_pnl > peak:
                                        PEAK_PROFIT_STORE[binance_sym] = unrealized_pnl
                                        save_harvest_state(PARTIAL_HARVEST_STORE, PEAK_PROFIT_STORE)
                                    
                                    # If we have a decent profit, don't let it vanish (Giveback threshold)
                                    if peak >= 15.0: # Only engage vault above $15 to allow breathing room
                                        allowed_pnl = peak * (1 - GIVEBACK_THRESHOLD)
                                        if unrealized_pnl < allowed_pnl:
                                            log_sovereign(f"🏦 [PROFIT VAULT] {binance_sym} profit fell from ${peak:.2f} to ${unrealized_pnl:.2f} (>{GIVEBACK_THRESHOLD*100}% drop). Executing Panic Harvest.", "CRITICAL")
                                            try:
                                                close_side = 'sell' if pos['side'] == 'BUY' else 'buy'
                                                await self.place_atomic_trade(ccxt_sym_full, close_side, size, entry, sl_price=entry*0.97)
                                                if self.telegram.is_active:
                                                    asyncio.create_task(self.telegram.send_message(f"🏦 *PROFIT VAULT*: {binance_sym} profit fell from ${peak:.2f} to ${unrealized_pnl:.2f}. Executing panic harvest."))
                                                continue
                                            except Exception as e:
                                                log_sovereign(f"❌ [VAULT EXIT] Failed: {e}", "ERROR")
                                
                                # Get tracked SL or use entry as fallback
                                initial_sl = POSITION_SL_STORE.get(binance_sym, entry * 0.97)  # Default 3% below entry
                                
                                # 🛡️ BREAKEVEN LOCK: Move SL to breakeven at +5%
                                if roi >= BREAKEVEN_PCT:
                                    breakeven_sl = entry * 1.005  # 0.5% above entry to cover fees
                                    if initial_sl < breakeven_sl:
                                        log_sovereign(f"🛡️ [BREAKEVEN] Moving {binance_sym} SL to ${breakeven_sl:.4f} (was ${initial_sl:.4f})", "SYSTEM")
                                        try:
                                            await self._engine_fetch_with_proxy(self.scanner.bot, 'update_live_stop_loss', ccxt_sym_full, pos['side'], breakeven_sl)
                                            POSITION_SL_STORE[binance_sym] = breakeven_sl
                                            initial_sl = breakeven_sl
                                        except Exception as e:
                                            log_sovereign(f"❌ [BREAKEVEN] Failed: {e}", "ERROR")
                                
                                # Standard trailing SL logic
                                new_sl = await self._engine_fetch_with_proxy(self.scanner.bot, 'trail_liquidity_sl', ccxt_sym, curr_price, initial_sl, pos['side'])
                                if new_sl != initial_sl and new_sl > initial_sl:
                                    log_sovereign(f"🛡️ LIVE TRAILING SL: Moving {binance_sym} Protection to ${new_sl:.4f}", "SYSTEM")
                                    try:
                                        await self._engine_fetch_with_proxy(self.scanner.bot, 'update_live_stop_loss', ccxt_sym_full, pos['side'], new_sl)
                                        POSITION_SL_STORE[binance_sym] = new_sl
                                    except Exception as e:
                                        log_sovereign(f"❌ [TRAILING SL] Failed: {e}", "ERROR")
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
                                
                                # Phase 79: Time-Stop Enforcement (Sim)
                                try:
                                    # Sim positions have %Y-%m-%d %H:%M:%S format
                                    entry_time = datetime.strptime(pos['time'], "%Y-%m-%d %H:%M:%S")
                                    age_seconds = (datetime.now() - entry_time).total_seconds()
                                    if age_seconds > MAX_POSITION_AGE_SECONDS:
                                        log_sovereign(f"⏳ [SIM TIME STOP] {sym} open for {age_seconds/3600:.1f}h. Harvesting stale sim.", "EXECUTOR")
                                        self.sim.close_position(sym, curr_price)
                                except Exception as e:
                                    if not SILENT_MODE: log_sovereign(f"Sim Time Stop Error: {e}", "DEBUG")

                if goto_monitoring:
                    await asyncio.sleep(60) # Still run the loop but slow down
                    continue

                log_sovereign(f"Scanning Market Matrix (Top {TOP_LIMIT})...", "SCANNER")
                opportunities = await self.scanner.scan_market()
                
                # Phase 48: Filter Cooldowns
                now = time.time()
                self.pos_cooldowns = {k: v for k, v in self.pos_cooldowns.items() if v > now}
                
                if opportunities:
                    for opp in opportunities:
                        self.symbol_regimes[opp['symbol']] = opp.get('regime', 'NEUTRAL')
                        self.symbol_regimes[ccxt_to_binance(opp['symbol'])] = opp.get('regime', 'NEUTRAL')
                    
                    active_cooldowns = list(self.pos_cooldowns.keys())
                    opportunities = [o for o in opportunities if o['symbol'] not in active_cooldowns]
                    
                    log_sovereign(f"🔍 [ENGINE DEBUG] Scanner returned {len(opportunities)} opportunities (filtered).", "DEBUG")
                    for top_trade in opportunities:
                        if top_trade['signal'] in ["BUY", "SELL", "LONG", "SHORT"]:
                            score = top_trade.get('score', 0)
                            min_conviction = self.scanner.bot.config.get('min_score', 85)
                            
                            if abs(score) < min_conviction:
                                log_sovereign(f"🛡️ [GATEKEEPER] Rejecting {top_trade['symbol']} | Score {score} < Required {min_conviction}", "SYSTEM")
                                continue
                                
                            log_sovereign(f"SOVEREIGN SIGNAL: {top_trade['symbol']} | {top_trade['signal']} | SCORE: {score}", "EXECUTOR")
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
                                    # Phase 41: MILLIONAIRE RISK DISCIPLINE
                                    try:
                                        bal = await asyncio.to_thread(self.scanner.bot.exchange.fetch_balance)
                                        avail_usdt = float(bal['free'].get('USDT', 0))
                                        total_equity = float(bal['total'].get('USDT', 0))
                                        
                                        # 1. LIQUIDITY GUARD: $10M 24h Volume
                                        try:
                                            ticker = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_ticker', top_trade['symbol'])
                                            vol_24h = float(ticker.get('quoteVolume', 0))
                                            # Phase 81: Relaxed liquidity for alpha moonshots
                                            default_min = self.scanner.bot.config.get('min_volume_24h', 10000000)
                                            min_vol = 1000000 if self.scanner.bot.config.get('risk_factor') == "AGGRESSIVE" else default_min
                                            if vol_24h < min_vol:
                                                log_sovereign(f"🛡️ [LIQUIDITY GUARD] {top_trade['symbol']} volume ${vol_24h:,.0f} < ${min_vol:,.0f}. Skipping.", "SYSTEM")
                                                continue
                                        except Exception: pass

                                        # 2. TREND FILTER: Must be above 4h EMA for BUY
                                        try:
                                            ohlcv_4h = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_ohlcv', top_trade['symbol'], timeframe='4h', limit=50)
                                            if ohlcv_4h:
                                                df_4h = pd.DataFrame(ohlcv_4h, columns=['t','o','h','l','c','v'])
                                                ema_4h = df_4h['c'].ewm(span=21).mean().iloc[-1]
                                                # Phase 81: Alpha Surge Bypass synced with Aggressive threshold (55)
                                                bypass_threshold = 55 if self.scanner.bot.config.get('risk_factor') == "AGGRESSIVE" else 110
                                                if score >= bypass_threshold:
                                                    log_sovereign(f"⚡ [ALPHA SURGE] {top_trade['symbol']} Score {score} >= {bypass_threshold}. Bypassing Trend Filter.", "CRITICAL")
                                                else:
                                                    if top_trade['signal'] == "BUY" and curr_price < ema_4h:
                                                        log_sovereign(f"🛡️ [TREND FILTER] {top_trade['symbol']} below 4h EMA (${ema_4h:.4f}). Skipping BUY.", "SYSTEM")
                                                        continue
                                                    elif top_trade['signal'] == "SELL" and curr_price > ema_4h:
                                                        log_sovereign(f"🛡️ [TREND FILTER] {top_trade['symbol']} above 4h EMA (${ema_4h:.4f}). Skipping SELL.", "SYSTEM")
                                                        continue
                                        except: pass

                                        # 3. 1% RISK ENFORCEMENT
                                        # Use max_loss_per_trade_pct from config
                                        max_risk_pct = self.scanner.bot.config.get('max_loss_per_trade_pct', 1.0)
                                        max_loss_allowed = total_equity * (max_risk_pct / 100)
                                        
                                        # Estimate potential loss
                                        entry_price = float(curr_price)
                                        stop_price = float(top_trade['sl'])
                                        leverage = self.scanner.bot.config.get('leverage', 2)
                                        
                                        # Allocation planning (Adjusted for Golden Era Risk)
                                        score = top_trade.get('score', 0)
                                        alpha_mode = top_trade.get('alpha_mode', False)
                                        if score > 120: amount = 40
                                        elif score > 100: amount = 30
                                        else: amount = 20
                                        
                                        # Calculate actual risk in USD
                                        price_diff_pct = abs(entry_price - stop_price) / entry_price
                                        potential_loss = amount * price_diff_pct * leverage
                                        
                                        if potential_loss > max_loss_allowed:
                                            # Scale down amount to fit risk
                                            amount = (max_loss_allowed / (price_diff_pct * leverage))
                                            log_sovereign(f"⚖️ [RISK SCALING] Scaling down {top_trade['symbol']} allocation to ${amount:.2f} to maintain {max_risk_pct}% risk (${max_loss_allowed:.2f})", "SYSTEM")
                                            
                                        if amount < 11: # Phase 84: Strict $11 Notional Guard
                                            log_sovereign(f"🛡️ [NOTIONAL GUARD] {top_trade['symbol']} allocation too small (${amount:.2f} < $11). Skipping.", "SYSTEM")
                                            continue

                                        if avail_usdt < amount:
                                            log_sovereign(f"🛡️ [MARGIN GUARD] Insufficient USDT ({avail_usdt:.2f} < {amount}). Skipping {top_trade['symbol']}", "SYSTEM")
                                            continue
                                            
                                    except Exception as e:
                                        log_sovereign(f"⚠️ [MILLIONAIRE GUARD] Checks Failed: {e}", "DEBUG")
                                        amount = 10 # Safer fallback
                                    
                                    # Phase 40: Symbol Normalization (PEPE -> 1000PEPE)
                                    exec_symbol = top_trade['symbol']
                                    if exec_symbol == 'PEPE/USDT':
                                        exec_symbol = '1000PEPE/USDT:USDT'
                                    elif '/USDT' in exec_symbol and ':USDT' not in exec_symbol:
                                        exec_symbol = exec_symbol + ':USDT'
                                    
                                    log_sovereign(f"🚀 LIVE EXECUTION: {exec_symbol} | {top_trade['signal']} | Allocation: ${amount} (Score: {score})", "SYSTEM")
                                    
                                    # Phase 62: Adaptive Liquidity Sentinel (Permission Boost)
                                    if self.scanner.bot.config.get('liquidity_sentinel', True):
                                        # Base slippage from config
                                        base_slip = self.scanner.bot.config.get('max_slippage_pct', 0.0035)
                                        
                                        # Adaptive Permission Scaling
                                        # Phase 85: Meme Safeguard (Adaptive Slippage for Alphas)
                                        if alpha_mode:
                                            max_slip = 0.008  # 0.8% for Memes
                                            log_sovereign(f"🚀 [MEME SAFEGUARD] Granting 0.8% Slippage Permission for {exec_symbol}", "SYSTEM")
                                        elif score > 115:
                                            max_slip = 0.012  # God-Mode: 1.2% permission for Moonshots
                                            log_sovereign(f"⚡ [PERMISSION_BOOST] Score {score} > 115. Granting 1.2% Slippage Permission.", "SYSTEM")
                                        elif score > 100:
                                            max_slip = 0.008  # Predator-Mode: 0.8% permission
                                            log_sovereign(f"🦅 [PREDATOR_BOOST] Score {score} > 100. Granting 0.8% Slippage Permission.", "SYSTEM")
                                        else:
                                            max_slip = base_slip

                                        raw_qty = (amount * self.scanner.bot.config.get('leverage', 2)) / curr_price
                                        depth = self.scanner.bot.config.get('liquidity_check_depth', 20)
                                        
                                        is_liquid = await self.check_liquidity(exec_symbol, raw_qty, max_slip, depth)
                                        if not is_liquid:
                                            log_sovereign(f"🛡️ [LIQUIDITY SENTINEL] Aborting {exec_symbol}: Insufficient orderbook depth (Max Slip: {max_slip*100:.2f}%).", "CRITICAL")
                                            continue
                                        log_sovereign(f"✅ [LIQUIDITY SENTINEL] Depth verified for {exec_symbol} (Limit: {max_slip*100:.2f}%).", "SYSTEM")

                                    # Phase 54: Atomic Shield (Native SL/TP)
                                    if self.scanner.bot.config.get('atomic_shield', True):
                                        # Precision Prep
                                        self.scanner.bot.exchange.load_markets()
                                        raw_qty = (amount * self.scanner.bot.config.get('leverage', 2)) / curr_price
                                        qty = float(self.scanner.bot.exchange.amount_to_precision(exec_symbol, raw_qty))
                                        tp_price = float(self.scanner.bot.exchange.price_to_precision(exec_symbol, top_trade['tp']))
                                        sl_price = float(self.scanner.bot.exchange.price_to_precision(exec_symbol, top_trade['sl']))
                                        side = 'buy' if top_trade['signal'] == "BUY" else 'sell'
                                        
                                        # Phase 72: Explicit Leverage Sync
                                        leverage = self.scanner.bot.config.get('leverage', 2)
                                        try:
                                            await asyncio.to_thread(self.scanner.bot.exchange.set_leverage, int(leverage), exec_symbol)
                                            log_sovereign(f"⚖️ [LEVERAGE SYNC] {exec_symbol} forced to {leverage}x", "SYSTEM")
                                        except Exception as e:
                                            log_sovereign(f"⚠️ [LEVERAGE SYNC] Failed for {exec_symbol}: {e}", "DEBUG")
                                        
                                        try:
                                            # Unified Atomic Entry
                                            print(f"DEBUG: Executing Atomic Shield Entry for {exec_symbol}")
                                            order = await asyncio.to_thread(
                                                self.scanner.bot.exchange.create_order,
                                                symbol=exec_symbol,
                                                type='market',
                                                side=side,
                                                amount=qty,
                                                params={
                                                    'stopLoss': sl_price,
                                                    'takeProfit': tp_price
                                                }
                                            )
                                            log_sovereign(f"✅ [ATOMIC SHIELD] Trade executed with native SL/TP: {order['id']}", "SYSTEM")
                                            
                                            # Phase 64: Telegram Syndicate Alert
                                            if self.telegram.is_active:
                                                asyncio.create_task(self.telegram.notify_trade(exec_symbol, side.upper(), qty, curr_price, tp_price, sl_price))
                                                
                                        except Exception as e:
                                            log_sovereign(f"❌ [ATOMIC SHIELD FAILURE] {e}", "CRITICAL")
                                            # Telegram Alert for Failure
                                            if self.telegram.is_active:
                                                asyncio.create_task(self.telegram.send_message(f"⚠️ *ATOMIC ENTRY FAILED*: {exec_symbol}\nError: {str(e)[:50]}"))
                                
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
                                strat_id = top_trade.get('strategy_id', 'default')
                                self.sim.execute_trade(top_trade['symbol'], top_trade['signal'], curr_price, amount, top_trade['tp'], top_trade['sl'], alpha_mode=alpha_mode, strategy_id=strat_id)
                                self.trade_count += 1

                        else:
                            # Skip neutral/wait signals
                            continue

                await asyncio.sleep(60)

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
                # BREAK: Sleep at the start to avoid hammering on bot restart
                await asyncio.sleep(4 * 3600)  # 4 hours
                
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

    async def _daily_harvest_worker(self):
        """Phase 37: Daily Automated Harvest (Every 24 Hours)"""
        log_sovereign("🗓️ Daily Profit Lock Sentinel active", "SYSTEM")
        while self.is_running:
            try:
                # BREAK: Sleep at the start to avoid hammering on bot restart
                await asyncio.sleep(24 * 3600)  # 24 hours
                
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

    async def _evolution_worker(self):
        """Phase 75: Sovereign Evolution & Hindsight Research (Periodic)"""
        log_sovereign("🧠 [EVOLUTION] Research & Development loop active.", "SYSTEM")
        while self.is_running:
            try:
                # 1. Hindsight Research Cycle (Every 12 Hours)
                # We do this at the start to build knowledge immediately
                await self.scanner.bot.learn_from_hindsight()
                
                # 2. Strategy Promotion Audit (Every 6 Hours)
                # Wait 6 hours between audits
                for _ in range(6):
                    await asyncio.sleep(3600)
                    
                    # Audit SIM performance
                    sim_state = self.sim.state
                    strat_perf = sim_state.get("strategy_performance", {})
                    
                    for strat_id, pnl in strat_perf.items():
                        if pnl > (self.scanner.bot.capital * 0.05): # Over 5% account growth in SIM
                            log_sovereign(f"🏆 [EVOLUTION] SIM Strategy '{strat_id}' reached promotion threshold (PnL: ${pnl:.2f}).", "SYSTEM")
                            
                            # Example tactical upgrade (Scaling up Alpha)
                            if strat_id == "alpha_strike":
                                upgrade_settings = {
                                    "min_score": 65, # Lower barrier for LIVE since it's proven
                                    "profit_lock_pct": 7.5 # Increase greed slightly
                                }
                                self.scanner.bot.promote_sim_strategy(strat_id, upgrade_settings)
                                
                                # Reset SIM perf for this ID to avoid double promotion
                                self.sim.state["strategy_performance"][strat_id] = 0
                                self.sim._save_state()

                # Long sleep before the next full research cycle
                await asyncio.sleep(6 * 3600)
                
            except Exception as e:
                log_sovereign(f"Evolution Worker Exception: {e}", "ERROR")
                await asyncio.sleep(3600)

    async def _prophet_research_worker(self):
        """Phase 86: The Geometric Prophet. Scans market geometry every 10 mins."""
        log_sovereign("🔮 [PROPHET] Geometric Research Agent active.", "SYSTEM")
        while self.is_running:
            try:
                # 1. Fetch Candidates (Memes + Top Gainers)
                candidates = await self.scanner.get_alpha_strike_candidates(limit=15)
                # 2. Perform Mathematical Analysis
                for symbol in candidates:
                    try:
                        # Fetch OHLCV (1h for regime detection, 5m for fine-tuning)
                        ohlcv = await self._engine_fetch_with_proxy(self.scanner.bot.exchange, 'fetch_ohlcv', symbol, timeframe='1h', limit=100)
                        if not ohlcv: continue
                        
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        signal, score, metrics = get_geometric_signal(df)
                        
                        # Store prediction in bot's memory
                        self.scanner.bot.prophet_predictions[symbol] = {
                            "signal": signal,
                            "score": 25, # Fixed bonus weight for Math confirmation
                            "metrics": metrics
                        }
                    except Exception as sym_e:
                        log_sovereign(f"Prophet Error on {symbol}: {sym_e}", "DEBUG")
                
                log_sovereign(f"🔮 [PROPHET] Research cycle complete. Predictions updated for {len(candidates)} symbols.", "SYSTEM")
                await asyncio.sleep(600) # 10 minutes
            except Exception as e:
                log_sovereign(f"Prophet Worker Exception: {e}", "ERROR")
                await asyncio.sleep(60)

    async def start(self):
        # Initialize background tasks
        self.heartbeat_task = asyncio.create_task(self.safe_task(self.update_heartbeat_loop(), "Heartbeat"))
        self.watch_task = asyncio.create_task(self.safe_task(self.watch_prices(), "MarketWatch"))
        self.logic_task = asyncio.create_task(self.safe_task(self.run_logic_cycle(), "LogicCycle"))
        self.harvest_task = asyncio.create_task(self.safe_task(self._harvest_worker(), "HarvestWorker")) # Phase 34
        self.daily_lock_task = asyncio.create_task(self.safe_task(self._daily_harvest_worker(), "DailyLock")) # Phase 37
        self.evolution_task = asyncio.create_task(self.safe_task(self._evolution_worker(), "EvolutionWorker")) # Phase 75
        self.prophet_task = asyncio.create_task(self.safe_task(self._prophet_research_worker(), "ProphetResearch")) # Phase 86
        self.kill_switch_task = asyncio.create_task(self.safe_task(self._kill_switch_sentinel(), "KillSwitch")) # Phase 88
        
        # Send Start Notification (Moved from __init__)
        if self.telegram.is_active:
             asyncio.create_task(self.safe_task(self.telegram.send_message("🚀 *SOVEREIGN ENGINE STARTING* (Phase 64 - Fortress Resilience)"), "StartupMsg"))
        
        while self.is_running:
            try:
                # Wait for any task to complete/fail
                done, pending = await asyncio.wait(
                    [self.heartbeat_task, self.watch_task, self.logic_task, self.harvest_task, self.daily_lock_task, self.evolution_task, self.prophet_task, self.kill_switch_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    try:
                        await task
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        log_sovereign(f"ENGINE TASK FAILURE: {e}\n{tb}", "CRITICAL")
                
                # Phase 64: Zombie Socket Watchdog (Force Restart if Pulse Stops)
                now = time.time()
                if now - self.last_pulse_time > 300: # 5 Minutes
                    log_sovereign(f"💀 [WATCHDOG] Zombie Socket detected (Pulse: {now - self.last_pulse_time:.1f}s ago). Force Restarting Engine...", "CRITICAL")
                    if self.telegram.is_active:
                         asyncio.create_task(self.safe_task(self.telegram.send_message("💀 *ZOMBIE SOCKET DETECTED*: Pulse was lost for > 5 minutes. Force restarting the engine..."), "ZombieAlert"))
                    # Break to trigger a full system restart
                    break

                # Check restart rate limiting
                now = time.time()
                self.restart_timestamps = [t for t in self.restart_timestamps if now - t < 3600]  # Keep last hour
                
                if len(self.restart_timestamps) >= self.max_restarts_per_hour:
                    log_sovereign(f"🚨 [RESTART LIMIT] Too many restarts ({len(self.restart_timestamps)}/hour). Cooling down for 5 minutes.", "CRITICAL")
                    await asyncio.sleep(300)  # 5 minute cooldown
                    self.restart_timestamps = []  # Reset after cooldown
                    continue
                
                # Restart failed tasks if still running
                if self.is_running:
                    restarted_any = False
                    if self.heartbeat_task in done: 
                        self.heartbeat_task = asyncio.create_task(self.update_heartbeat_loop())
                        restarted_any = True
                    if self.watch_task in done: 
                        self.watch_task = asyncio.create_task(self.watch_prices())
                        restarted_any = True
                    if self.logic_task in done: 
                        self.logic_task = asyncio.create_task(self.run_logic_cycle())
                        restarted_any = True
                    if self.evolution_task in done:
                        self.evolution_task = asyncio.create_task(self._evolution_worker())
                        restarted_any = True
                    if self.harvest_task in done: 
                        self.harvest_task = asyncio.create_task(self._harvest_worker())
                        restarted_any = True
                    if self.daily_lock_task in done: 
                        self.daily_lock_task = asyncio.create_task(self._daily_harvest_worker())
                        restarted_any = True
                    
                    if self.prophet_task in done:
                        self.prophet_task = asyncio.create_task(self._prophet_research_worker())
                        restarted_any = True
                    
                    if restarted_any:
                        self.restart_timestamps.append(now)
                        log_sovereign(f"🔄 [ENGINE] Task restarted. Restart count this hour: {len(self.restart_timestamps)}", "DEBUG")
                    
                    await asyncio.sleep(30)  # Increased cooldown between checks
            except Exception as e:
                log_sovereign(f"WATCHDOG CRITICAL: {e}", "CRITICAL")
                await asyncio.sleep(30)

if __name__ == "__main__":
    if not acquire_lock():
        exit(1)
        
    engine = AsyncSovereignEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        log_sovereign("Sovereign Protocol Terminated by User.", "SYSTEM")
    except Exception as e:
        log_sovereign(f"FATAL SYSTEM ERROR: {e}", "CRITICAL")
    finally:
        release_lock()
