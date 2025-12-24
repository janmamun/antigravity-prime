
import time
import json
import os
import subprocess
import signal

HEARTBEAT_FILE = "guardian_heartbeat.json"
BOT_SCRIPT = "run_v17.py"

def is_bot_alive():
    if not os.path.exists(HEARTBEAT_FILE):
        return False
    try:
        with open(HEARTBEAT_FILE, "r") as f:
            data = json.load(f)
            last_hb = data.get("last_heartbeat", 0)
            if time.time() - last_hb < 600: # 10 minutes
                return True
    except:
        pass
    return False

def start_bot():
    print(f"[{time.ctime()}] Guardian: Starting sovereign loop...")
    return subprocess.Popen(["./venv/bin/python", "-u", BOT_SCRIPT], 
                            stdout=open("bot_output.log", "a"), 
                            stderr=subprocess.STDOUT)

def main():
    print(f"[{time.ctime()}] Guardian Protocol Alpha Initialized.")
    bot_process = None
    
    if not is_bot_alive():
        bot_process = start_bot()
    
    while True:
        time.sleep(60) # Check every minute
        if not is_bot_alive():
            print(f"[{time.ctime()}] Guardian: Bot stall detected or not running. Restarting...")
            if bot_process:
                try:
                    os.kill(bot_process.pid, signal.SIGTERM)
                except:
                    pass
            bot_process = start_bot()
        else:
            # Check if process is still physically running (in case heartbeat file is lying)
            if bot_process and bot_process.poll() is not None:
                print(f"[{time.ctime()}] Guardian: Process died but heartbeat was recent. Restarting...")
                bot_process = start_bot()

if __name__ == "__main__":
    main()
