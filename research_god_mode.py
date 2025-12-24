
from market_scanner import MarketScanner
import pandas as pd
import time

def run_god_mode_test():
    print("🧙‍♂️ INITIATING GOD MODE HISTORICAL VERIFICATION")
    print("===============================================")
    
    scanner = MarketScanner() # Public mode
    
    # 1. Fetch Top Coins Live
    print("📡 Detecting Market Leaders (Top Volume)...")
    top_coins = scanner.get_top_volume_coins(limit=10)
    print(f"   Targets Acquired: {top_coins}")
    
    print("\n⚔️ RUNNING SIMULATION ON TARGETS")
    print("   Strategy: Magic 40 Trend + 2xATR Dynamic Stops")
    print("-" * 60)
    
    total_wins = 0
    total_trades = 0
    
    results_summary = []
    
    for symbol in top_coins:
        print(f"   Analyzing {symbol}...", end=" ")
        try:
            # Fetch deeper history for backtest (1000 candles)
            ohlcv = scanner.exchange.fetch_ohlcv(symbol, '1d', limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Simple vector backtest of the strategy logic
            # Using the same logic as GodModeStrategy but iterated
            df['ma40'] = df['Close'].rolling(40).mean()
            df['atr'] = (df['High'] - df['Low']).rolling(14).mean() # Simplified ATR
            
            # Entry: Close > MA40
            # Exit: Trailing Stop? Or Fixed Target? 
            # Let's test the specific request: "predicts TP/SL"
            # We simulate taking the trade when Signal is generated and hitting TP or SL
            
            balance = 1000
            coin_trades = 0
            coin_wins = 0
            
            for i in range(50, len(df)-1):
                price = df['Close'].iloc[i]
                ma = df['ma40'].iloc[i]
                atr = df['atr'].iloc[i]
                
                # Signal Generation
                if price > ma and df['Close'].iloc[i-1] <= df['ma40'].iloc[i-1]:
                    # CROSSOVER BUY
                    entry = price
                    tp = entry + (atr * 4) # 4x Reward
                    sl = entry - (atr * 2) # 2x Risk
                    
                    # Check future candles for result
                    outcome = "PENDING"
                    for j in range(i+1, min(i+30, len(df))):
                        future_high = df['High'].iloc[j]
                        future_low = df['Low'].iloc[j]
                        
                        if future_low <= sl:
                            outcome = "LOSS"
                            break
                        if future_high >= tp:
                            outcome = "WIN"
                            break
                            
                    if outcome == "WIN":
                        coin_wins += 1
                        total_wins += 1
                    if outcome != "PENDING":
                        coin_trades += 1
                        total_trades += 1
                        
            win_rate = (coin_wins / coin_trades * 100) if coin_trades > 0 else 0
            print(f" Win Rate: {win_rate:.1f}% ({coin_wins}/{coin_trades})")
            results_summary.append({'Symbol': symbol, 'WinRate': win_rate, 'Trades': coin_trades})
            
        except Exception as e:
            print(f"Error: {e}")

    print("\n📊 GOD MODE REPORT CARD")
    print("======================")
    avg_win_rate = sum(r['WinRate'] for r in results_summary) / len(results_summary) if results_summary else 0
    print(f"GLOBAL WIN RATE: {avg_win_rate:.2f}%")
    
    if avg_win_rate > 50:
        print("✅ STATUS: STRATEGY VALIDATED (Positive Expectancy with 2:1 Ratio)")
    else:
        print("⚠️ STATUS: OPTIMIZATION NEEDED")

if __name__ == "__main__":
    run_god_mode_test()
