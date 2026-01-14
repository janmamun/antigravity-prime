#!/usr/bin/env python3
"""
SOVEREIGN LIVE TRADER - Autonomous Trade Finder & Executor
Scans market, finds high-conviction setups, and executes trades automatically
"""

import ccxt
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration (FEE-ADJUSTED + AGGRESSIVE MODE)
# Binance Futures fees: 0.04% taker x2 = 0.08% round trip
# TARGET: $100 profit today
MIN_24H_CHANGE = 15.0      # Only consider coins up >15% in 24h
MIN_SCORE = 60             # Lowered for more opportunities
MAX_POSITION_SIZE = 50     # Increased to $50 per trade
PROFIT_TARGET_PCT = 2.5    # Target +2.5% (nets ~2.42% after fees)
STOP_LOSS_PCT = -2.0       # Stop at -2.0% (nets ~-2.08% after fees)
CHECK_INTERVAL = 45        # Faster scanning every 45 seconds
MAX_POSITIONS = 5          # Allow 5 concurrent positions

# Symbols to skip (SETTLING/delisted/problematic)
SKIP_SYMBOLS = [
    'ALPACA/USDT:USDT', 'BNX/USDT:USDT', 'ALPHA/USDT:USDT', 
    'OCEAN/USDT:USDT', 'LUNA/USDT:USDT', 'UST/USDT:USDT'
]

def get_exchange():
    return ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 10000
    })

def get_top_movers(exchange, limit=20):
    """Get top momentum coins by 24h change"""
    tickers = exchange.fetch_tickers()
    futures = {k: v for k, v in tickers.items() if '/USDT:USDT' in k}
    
    # Filter by minimum momentum and volume
    filtered = []
    for sym, data in futures.items():
        pct = data.get('percentage', 0) or 0
        vol = data.get('quoteVolume', 0) or 0
        if pct >= MIN_24H_CHANGE and vol > 10_000_000:  # >$10M volume
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
            mark = float(p.get('markPrice', 0))
            pnl = float(p.get('unrealizedProfit', 0))
            leverage = float(p.get('leverage', 1))
            notional = abs(size * entry)
            margin = notional / leverage if leverage > 0 else notional
            roi = (pnl / margin) * 100 if margin > 0 else 0
            
            active.append({
                'symbol': p.get('symbol', 'UNKNOWN'),
                'size': size,
                'entry': entry,
                'mark': mark,
                'pnl': pnl,
                'roi': roi,
                'leverage': leverage
            })
    return active

def calculate_score(mover, exchange):
    """Calculate conviction score for a coin"""
    score = 0
    reasons = []
    
    # 1. Base momentum score
    change = mover['change_24h']
    if change >= 50:
        score += 40
        reasons.append(f"🚀 Extreme momentum (+{change:.1f}%)")
    elif change >= 30:
        score += 30
        reasons.append(f"📈 Strong momentum (+{change:.1f}%)")
    elif change >= 15:
        score += 20
        reasons.append(f"✅ Good momentum (+{change:.1f}%)")
    
    # 2. Volume bonus
    vol_m = mover['volume'] / 1_000_000
    if vol_m > 200:
        score += 20
        reasons.append(f"💰 Massive volume (${vol_m:.0f}M)")
    elif vol_m > 50:
        score += 10
        reasons.append(f"📊 Good volume (${vol_m:.0f}M)")
    
    # 3. Volatility check (prefer not already extended)
    try:
        ohlcv = exchange.fetch_ohlcv(mover['symbol'], '15m', limit=4)
        if ohlcv:
            last_4_candles = [c[4] for c in ohlcv]  # Close prices
            recent_range = (max(last_4_candles) - min(last_4_candles)) / min(last_4_candles) * 100
            if recent_range < 3:  # Consolidating
                score += 15
                reasons.append("🎯 Consolidating (good entry)")
            elif recent_range > 8:  # Extended
                score -= 10
                reasons.append("⚠️ Extended range")
    except:
        pass
    
    return score, reasons

def open_trade(exchange, symbol, side='buy', size_usd=25):
    """Open a new position"""
    try:
        # Get current price
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        
        # Calculate amount
        amount = size_usd / price
        
        # Get precision using exchange method (safer)
        markets = exchange.load_markets()
        if symbol in markets:
            amount = exchange.amount_to_precision(symbol, amount)
            amount = float(amount)  # Convert back to float for order
        
        print(f"🟢 OPENING: {symbol} | Side: {side.upper()} | Amount: {amount} | ~${size_usd}")
        
        order = exchange.create_market_order(symbol, side, amount)
        print(f"✅ ORDER FILLED: ID {order['id']}")
        return True
        
    except Exception as e:
        print(f"❌ Order failed: {e}")
        return False


def close_position(exchange, symbol, size):
    """Close a position"""
    side = 'sell' if size > 0 else 'buy'
    amount = abs(size)
    
    # Normalize symbol
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
    
    print("=" * 60)
    print("🦅 SOVEREIGN LIVE TRADER - AUTONOMOUS MODE")
    print(f"   Min Momentum: +{MIN_24H_CHANGE}% | Min Score: {MIN_SCORE}")
    print(f"   Max Position: ${MAX_POSITION_SIZE} | Max Positions: {MAX_POSITIONS}")
    print(f"   TP: +{PROFIT_TARGET_PCT}% | SL: {STOP_LOSS_PCT}%")
    print("=" * 60)
    print()
    
    trades_opened = 0
    
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        try:
            # 1. Check existing positions first
            positions = get_positions(exchange)
            
            print(f"\n[{timestamp}] === POSITION CHECK ===")
            if positions:
                for pos in positions:
                    sym = pos['symbol']
                    roi = pos['roi']
                    pnl = pos['pnl']
                    
                    status = "✅ PROFIT" if roi > 0 else "⏳ HOLDING"
                    if roi >= PROFIT_TARGET_PCT:
                        status = "🎯 TARGET!"
                        print(f"  {sym}: ${pnl:+.2f} ({roi:+.2f}%) - {status}")
                        close_position(exchange, sym, pos['size'])
                    elif roi <= STOP_LOSS_PCT:
                        status = "🛑 STOPPED"
                        print(f"  {sym}: ${pnl:+.2f} ({roi:+.2f}%) - {status}")
                        close_position(exchange, sym, pos['size'])
                    else:
                        print(f"  {sym}: ${pnl:+.2f} ({roi:+.2f}%) - {status}")
            else:
                print("  No open positions")
            
            # 2. Look for new opportunities if we have room
            if len(positions) < MAX_POSITIONS:
                print(f"\n[{timestamp}] === SCANNING OPPORTUNITIES ===")
                
                movers = get_top_movers(exchange, limit=10)
                
                # Skip symbols we already have
                open_symbols = {p['symbol'] for p in positions}
                
                best_setup = None
                best_score = 0
                
                for mover in movers:
                    sym = mover['symbol']
                    base_sym = sym.replace('/USDT:USDT', 'USDT')
                    
                    # Skip problematic symbols
                    if sym in SKIP_SYMBOLS:
                        continue
                    
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
                    print(f"\n🎯 BEST SETUP FOUND: {best_setup['symbol']}")
                    print(f"   Score: {best_setup['score']} | 24h: +{best_setup['change']:.1f}%")
                    for r in best_setup['reasons']:
                        print(f"   {r}")
                    
                    # Execute trade
                    if open_trade(exchange, best_setup['symbol'], 'buy', MAX_POSITION_SIZE):
                        trades_opened += 1
                        print(f"\n📊 Total trades opened this session: {trades_opened}")
                else:
                    print(f"  No high-conviction setups found (best score: {best_score})")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"\n⏳ Next scan in {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
