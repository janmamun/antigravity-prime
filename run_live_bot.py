
import time
from datetime import datetime
from trading_bot_v8 import UltimateV8Bot
import sys

def main():
    print("🧙‍♂️ STARTING MAGICIAN BOT LIVE DASHBOARD")
    print("========================================")
    print("Press Ctrl+C to stop")
    print("")
    
    bot = UltimateV8Bot(initial_capital=10000)
    
    try:
        # Loop forever (or until user stops)
        # For this demo environment, we loop 3 times with short sleep
        # In real life, while True: time.sleep(60)
        
        loop_count = 0
        max_loops = 1 # Just run once for the user demo
        
        while loop_count < max_loops:
            print(f"\n⏰ SCAN TIME: {datetime.now().strftime('%H:%M:%S')}")
            bot.live_trading_mode(assets=['BTC-USD', 'ETH-USD', 'SOL-USD'])
            
            print("\n⏳ Waiting for next scan...")
            loop_count += 1
            # time.sleep(60) # 1 minute wait
            
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")

if __name__ == "__main__":
    main()
