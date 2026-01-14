#!/usr/bin/env python3
"""
SOVEREIGN POSITION WATCHER - Live Position Monitor & Auto-Close
Monitors open positions and closes them at profit targets or stops
"""

import ccxt
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
PROFIT_TARGET_PCT = 1.5   # Close at +1.5% profit
STOP_LOSS_PCT = -3.0      # Close at -3% loss  
CHECK_INTERVAL = 30       # Check every 30 seconds
MAX_RUNTIME_MIN = 60      # Run for max 60 minutes

def get_exchange():
    return ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 10000
    })

def get_positions(exchange):
    """Get active positions using V2 API"""
    account = exchange.fapiPrivateV2GetAccount()
    positions = account.get('positions', [])
    active = []
    for p in positions:
        if float(p.get('positionAmt', 0)) != 0:
            active.append({
                'symbol': p.get('symbol', 'UNKNOWN'),
                'size': float(p.get('positionAmt', 0)),
                'entry': float(p.get('entryPrice', 0)),
                'mark': float(p.get('markPrice', 0)),
                'pnl': float(p.get('unrealizedProfit', 0)),
                'leverage': float(p.get('leverage', 1))
            })
    return active

def close_position(exchange, symbol, size):
    """Close a position by placing an opposite order"""
    side = 'sell' if size > 0 else 'buy'
    amount = abs(size)
    
    # Normalize symbol for CCXT (e.g., DOGEUSDT -> DOGE/USDT:USDT)
    base = symbol.replace('USDT', '')
    ccxt_symbol = f"{base}/USDT:USDT"
    
    print(f"🔴 CLOSING: {ccxt_symbol} | Side: {side.upper()} | Amount: {amount}")
    
    try:
        order = exchange.create_market_order(ccxt_symbol, side, amount, params={'reduceOnly': True})
        print(f"✅ CLOSED: Order ID {order['id']}")
        return True
    except Exception as e:
        print(f"❌ Close failed: {e}")
        return False

def main():
    exchange = get_exchange()
    start_time = time.time()
    
    print("=" * 60)
    print("🦅 SOVEREIGN POSITION WATCHER - LIVE MONITOR")
    print(f"   Profit Target: +{PROFIT_TARGET_PCT}% | Stop Loss: {STOP_LOSS_PCT}%")
    print(f"   Check Interval: {CHECK_INTERVAL}s | Max Runtime: {MAX_RUNTIME_MIN}min")
    print("=" * 60)
    print()
    
    closed_symbols = set()
    
    while True:
        elapsed_min = (time.time() - start_time) / 60
        
        if elapsed_min >= MAX_RUNTIME_MIN:
            print(f"\n⏱️ Max runtime reached ({MAX_RUNTIME_MIN}min). Exiting...")
            break
        
        try:
            positions = get_positions(exchange)
            
            if not positions:
                print(f"\n🎯 All positions closed! Exiting watcher...")
                break
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Active Positions:")
            
            for pos in positions:
                sym = pos['symbol']
                if sym in closed_symbols:
                    continue
                    
                size = pos['size']
                entry = pos['entry']
                pnl = pos['pnl']
                leverage = pos['leverage']
                
                # Calculate ROI
                notional = abs(size * entry)
                margin = notional / leverage if leverage > 0 else notional
                roi = (pnl / margin) * 100 if margin > 0 else 0
                
                # Status emoji
                if roi >= PROFIT_TARGET_PCT:
                    status = "🎯 TARGET HIT"
                elif roi <= STOP_LOSS_PCT:
                    status = "🛑 STOP HIT"
                elif roi > 0:
                    status = "✅ PROFIT"
                else:
                    status = "⏳ WAITING"
                
                print(f"  {sym:12} | PnL: ${pnl:+.2f} ({roi:+.2f}%) | {status}")
                
                # Auto-close logic
                if roi >= PROFIT_TARGET_PCT:
                    print(f"  ➡️ Auto-closing {sym} at profit target!")
                    if close_position(exchange, sym, size):
                        closed_symbols.add(sym)
                        
                elif roi <= STOP_LOSS_PCT:
                    print(f"  ➡️ Auto-closing {sym} at stop loss!")
                    if close_position(exchange, sym, size):
                        closed_symbols.add(sym)
            
            # Check if all closed
            remaining = [p for p in positions if p['symbol'] not in closed_symbols]
            if not remaining:
                print(f"\n🎯 All positions closed successfully!")
                break
                
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(CHECK_INTERVAL)
    
    print("\n" + "=" * 60)
    print("🦅 WATCHER SESSION ENDED")
    print("=" * 60)

if __name__ == "__main__":
    main()
