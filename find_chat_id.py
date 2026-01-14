import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def get_chat_id():
    if not TOKEN or ":" not in TOKEN:
        print("❌ Error: TELEGRAM_BOT_TOKEN not found or invalid in .env")
        return

    print(f"📡 Polling Telegram API for updates (Token: {TOKEN[:10]}...)")
    print("👉 ACTION REQUIRED: Open your Telegram bot and send it ANY message now!")
    
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    
    start_time = time.time()
    while time.time() - start_time < 60: # Poll for 60 seconds
        try:
            resp = requests.get(url).json()
            if resp.get("ok") and resp.get("result"):
                # Get the last message
                last_update = resp["result"][-1]
                chat_id = last_update["message"]["chat"]["id"]
                first_name = last_update["message"]["chat"].get("first_name", "Commander")
                print(f"\n✅ FOUND CHAT ID: {chat_id}")
                print(f"👤 USER: {first_name}")
                print("\n📋 Next Steps:")
                print(f"1. Add 'TELEGRAM_CHAT_ID={chat_id}' to your .env file.")
                print("2. Restart the bot via 'pm2 restart sovereign-bot'.")
                return
        except Exception as e:
            print(f"⚠️ Error polling: {e}")
        
        time.sleep(2)
        print(".", end="", flush=True)

    print("\n⌛ Timeout: No messages received. Try again and make sure you've messaged the bot.")

if __name__ == "__main__":
    get_chat_id()
