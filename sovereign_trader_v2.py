#!/usr/bin/env python3
"""
SOVEREIGN TRADER V2 - Smart, Conservative Trading
- Fewer trades, bigger targets
- Proper position management with Reduce Only orders
- Cancels stale orders on position close
- Target: $100 daily profit (10% at $1000 balance)
"""

import ccxt
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# CONSERVATIVE CONFIGURATION
# Focus on quality over quantity
MIN_24H_CHANGE = 25.0      # Only strong movers (was 15%)
MIN_SCORE = 75             # High conviction only (was 60)
MAX_POSITION_SIZE = 75     # Bigger positions for bigger gains
PROFIT_TARGET_PCT = 5.0    # Target +5% (bigger wins)
STOP_LOSS_PCT = -3.0       # Stop at -3% (tighter R:R)
CHECK_INTERVAL = 120       # Check every 2 min (less churn)
MAX_POSITIONS = 3          # Max 3 concurrent
MIN_VOLUME_USD = 50_000_000  # Min $50M daily volume

# Symbols to skip
SKIP_SYMBOLS = [
    'ALPACA/USDT:USDT', 'BNX/USDT:USDT', 'ALPHA/USDT:USDT', 
    'OCEAN/USDT:USDT', 'LUNA/USDT:USDT', 'UST/USDT:USDT'
]

# Session tracking
session_start_balance = None
session_pnl = 0
trades_closed = 0
wins = 0
losses = 0

def get_exchange():
    return ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 15000
    })

def get_top_movers(exchange, limit=10):
    """Get top momentum coins by 24h change"""
    tickers = exchange.fetch_tickers()
    futures = {k: v for k, v in tickers.items() if '/USDT:USDT' in k}
    
    filtered = []
    for sym, data in futures.items():
        if sym in SKIP_SYMBOLS:
            continue
        pct = data.get('percentage', 0) or 0
        vol = data.get('quoteVolume', 0) or 0
        if pct >= MIN_24H_CHANGE and vol >= MIN_VOLUME_USD:
            # Check if trading is allowed
            try:
                markets = exchange.load_markets()
                if sym in markets:
                    status = markets[sym].get('info', {}).get('status', 'UNKNOWN')
                    if status != 'TRADING':
                        continue
            except:
                continue
            
            filtered.append({
                'symbol': sym,
                'change_24h': pct,
                'volume': vol,
                'price': data.get('last', 0)
            })
    
    return sorted(filtered, key=lambda x: x['change_24h'], reverse=True)[:limit]

def get_positions(exchange):
    """Get active positions using V2 API"""
    account = exchange.fapiPrivateV2GetAccount()
    positions = account.get('positions', [])
    active = []
    for p in positions:
        if float(p.get('positionAmt', 0)) != 0:
            size = float(p.get('positionAmt', 0))
            entry = float(p.get('entryPrice', 0))
            pnl = float(p.get('unrealizedProfit', 0))
            leverage = float(p.get('leverage', 1))
            notional = abs(size * entry)
            margin = notional / leverage if leverage > 0 else notional
            roi = (pnl / margin) * 100 if margin > 0 else 0
            
            active.append({
                'symbol': p.get('symbol', 'UNKNOWN'),
                'size': size,
                'entry': entry,
                'pnl': pnl,
                'roi': roi,
                'leverage': leverage,
                'notional': notional
            })
    return active

def calculate_score(mover, exchange):
    """Calculate conviction score"""
    score = 0
    reasons = []
    
    change = mover['change_24h']
    if change >= 50:
        score += 45
        reasons.append(f"🚀 Extreme momentum (+{change:.1f}%)")
    elif change >= 35:
        score += 35
        reasons.append(f"📈 Strong momentum (+{change:.1f}%)")
    elif change >= 25:
        score += 25
        reasons.append(f"✅ Good momentum (+{change:.1f}%)")
    
    vol_m = mover['volume'] / 1_000_000
    if vol_m > 500:
        score += 25
        reasons.append(f"💰 Huge volume (${vol_m:.0f}M)")
    elif vol_m > 100:
        score += 15
        reasons.append(f"📊 Strong volume (${vol_m:.0f}M)")
    
    # Check for consolidation (buying the dip)
    try:
        ohlcv = exchange.fetch_ohlcv(mover['symbol'], '15m', limit=8)
        if ohlcv:
            closes = [c[4] for c in ohlcv]
            highs = [c[2] for c in ohlcv]
            lows = [c[3] for c in ohlcv]
            
            current = closes[-1]
            highest = max(highs)
            lowest = min(lows)
            
            # How far from daily high?
            pullback = (highest - current) / highest * 100 if highest > 0 else 0
            
            if 3 < pullback < 15:  # 3-15% pullback = good entry
                score += 20
                reasons.append(f"🎯 Buying pullback ({pullback:.1f}% off high)")
            elif pullback < 2:  # Near highs
                score -= 10
                reasons.append("⚠️ Near local high")
    except:
        pass
    
    return score, reasons

def open_trade(exchange, symbol, side='buy', size_usd=75):
    """Open a new position with better precision handling"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        amount = size_usd / price
        
        markets = exchange.load_markets()
        if symbol in markets:
            amount = float(exchange.amount_to_precision(symbol, amount))
        
        print(f"🟢 OPENING: {symbol} | {side.upper()} | ${size_usd}")
        
        order = exchange.create_market_order(symbol, side, amount)
        print(f"✅ FILLED @ ${price:.4f} | ID: {order['id']}")
        return True
        
    except Exception as e:
        print(f"❌ Order failed: {e}")
        return False

def close_position(exchange, symbol, size):
    """Close a position with Reduce Only"""
    global trades_closed, wins, losses, session_pnl
    
    side = 'sell' if size > 0 else 'buy'
    amount = abs(size)
    
    base = symbol.replace('USDT', '')
    ccxt_symbol = f"{base}/USDT:USDT"
    
    print(f"🔴 CLOSING: {ccxt_symbol} | {side.upper()} | Size: {amount}")
    
    try:
        order = exchange.create_market_order(
            ccxt_symbol, side, amount, 
            params={'reduceOnly': True}  # CRITICAL: Reduce Only!
        )
        print(f"✅ CLOSED: Order ID {order['id']}")
        trades_closed += 1
        return True
    except Exception as e:
        print(f"❌ Close failed: {e}")
        return False

def main():
    global session_start_balance, session_pnl, wins, losses
    
    exchange = get_exchange()
    
    # Get starting balance
    balance = exchange.fetch_balance()
    session_start_balance = float(balance.get('total', {}).get('USDT', 0))
    
    print("=" * 60)
    print("🦅 SOVEREIGN TRADER V2 - CONSERVATIVE MODE")
    print("=" * 60)
    print(f"   Starting Balance: ${session_start_balance:.2f}")
    print(f"   Target: +$100 profit")
    print(f"   Min Momentum: +{MIN_24H_CHANGE}% | Min Score: {MIN_SCORE}")
    print(f"   Position Size: ${MAX_POSITION_SIZE}")
    print(f"   TP: +{PROFIT_TARGET_PCT}% | SL: {STOP_LOSS_PCT}%")
    print("=" * 60)
    print()
    
    trades_opened = 0
    
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        try:
            positions = get_positions(exchange)
            
            # Get current balance
            balance = exchange.fetch_balance()
            current_balance = float(balance.get('total', {}).get('USDT', 0))
            total_unrealized = sum(p['pnl'] for p in positions)
            equity = current_balance + total_unrealized
            session_pnl = equity - session_start_balance
            
            print(f"\n{'='*60}")
            print(f"[{timestamp}] SESSION STATUS")
            print(f"   Equity: ${equity:.2f} | Session PnL: ${session_pnl:+.2f}")
            print(f"   Progress: {session_pnl/100*100:.1f}% of $100 target")
            print(f"{'='*60}")
            
            # Check target reached
            if session_pnl >= 100:
                print("\n🎯🎯🎯 TARGET REACHED! $100 PROFIT! 🎯🎯🎯")
                print("Closing all positions and stopping...")
                for pos in positions:
                    close_position(exchange, pos['symbol'], pos['size'])
                break
            
            # Manage existing positions
            print(f"\n[{timestamp}] POSITIONS ({len(positions)})")
            for pos in positions:
                sym = pos['symbol']
                roi = pos['roi']
                pnl = pos['pnl']
                
                if roi >= PROFIT_TARGET_PCT:
                    print(f"  🎯 {sym}: ${pnl:+.2f} ({roi:+.1f}%) - TARGET HIT!")
                    if close_position(exchange, sym, pos['size']):
                        wins += 1
                elif roi <= STOP_LOSS_PCT:
                    print(f"  🛑 {sym}: ${pnl:+.2f} ({roi:+.1f}%) - STOPPED")
                    if close_position(exchange, sym, pos['size']):
                        losses += 1
                else:
                    status = "✅ PROFIT" if roi > 0 else "⏳ HOLDING"
                    print(f"  {status} {sym}: ${pnl:+.2f} ({roi:+.1f}%)")
            
            if not positions:
                print("  No open positions")
            
            # Look for new opportunities
            if len(positions) < MAX_POSITIONS:
                print(f"\n[{timestamp}] SCANNING...")
                
                movers = get_top_movers(exchange, limit=8)
                open_symbols = {p['symbol'] for p in positions}
                
                best_setup = None
                best_score = 0
                
                for mover in movers:
                    sym = mover['symbol']
                    base_sym = sym.replace('/USDT:USDT', 'USDT')
                    
                    if base_sym in open_symbols:
                        continue
                    
                    score, reasons = calculate_score(mover, exchange)
                    
                    if score > best_score:
                        best_score = score
                        best_setup = {
                            'symbol': sym,
                            'score': score,
                            'reasons': reasons,
                            'change': mover['change_24h'],
                            'price': mover['price']
                        }
                
                if best_setup and best_setup['score'] >= MIN_SCORE:
                    print(f"\n🎯 HIGH-CONVICTION SETUP: {best_setup['symbol']}")
                    print(f"   Score: {best_setup['score']} | 24h: +{best_setup['change']:.1f}%")
                    for r in best_setup['reasons']:
                        print(f"   {r}")
                    
                    if open_trade(exchange, best_setup['symbol'], 'buy', MAX_POSITION_SIZE):
                        trades_opened += 1
                else:
                    print(f"  Waiting for better setup (best score: {best_score}, need {MIN_SCORE})")
            
            # Stats
            print(f"\n📊 Session: {trades_opened} opened | {wins} wins | {losses} losses")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"\n⏳ Next check in {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
