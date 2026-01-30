import json
import os
import time
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import ccxt
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Sovereign Nexus API", version="1.0.0")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER_FILE = os.path.join(BASE_DIR, "evolution_ledger.json")
EQUITY_FILE = os.path.join(BASE_DIR, "session_start_equity.json")
LOG_FILE = os.path.join(BASE_DIR, "bot_output_async.log")
HEARTBEAT_FILE = os.path.join(BASE_DIR, "guardian_heartbeat_async.json")

def read_json_safe(path):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None

@app.get("/api/v1/health")
async def health_check():
    hb = read_json_safe(HEARTBEAT_FILE) or {"status": "OFFLINE", "last_heartbeat": 0}
    pulse_age = time.time() - hb.get("last_heartbeat", 0)
    return {
        "status": "ACTIVE" if pulse_age < 300 else "STALLED",
        "pulse_seconds_ago": round(pulse_age, 1),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/equity")
async def get_equity():
    # Phase 75: Financial Synchronization Fix
    equity_data = read_json_safe(EQUITY_FILE) or {"equity": 259.75, "date": "N/A"}
    current_bal = equity_data.get("equity", 259.75)  # Use baseline as first fallback
    
    # Scrape logs for latest Integrity Pulse
    try:
        if os.path.exists(LOG_FILE):
            import re
            with open(LOG_FILE, 'r') as f:
                # Read last 200 lines to find the most recent pulse
                lines = f.readlines()[-200:]
                for line in reversed(lines):
                    # Fixed Regex for broader integrity matching
                    match = re.search(r"Balance Integrity: USDT:\s*([\d\.]+)", line)
                    if not match:
                        match = re.search(r"Integrity: USDT:\s*([\d\.]+)", line)
                    
                    if match:
                        current_bal = float(match.group(1))
                        break
    except: pass

    return {
        "baseline": equity_data.get("equity", 0),
        "current": current_bal,
        "target": 500.00,
        "is_approaching_milestone": current_bal > 295.0
    }

@app.get("/api/v1/ledger")
async def get_ledger():
    ledger = read_json_safe(LEDGER_FILE) or []
    return ledger[-10:] # Return last 10 events

@app.get("/api/v1/positions")
async def get_positions():
    try:
        ex = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'options': {'defaultType': 'future'}
        })
        bal = await asyncio.to_thread(ex.fetch_balance)
        pos = [p for p in bal.get('info', {}).get('positions', []) if float(p.get('positionAmt', 0)) != 0]
        
        formatted = []
        for p in pos:
            formatted.append({
                "symbol": p['symbol'],
                "side": "BUY" if float(p['positionAmt']) > 0 else "SELL",
                "size": abs(float(p['positionAmt'])),
                "entry": float(p.get('entryPrice', 0)),
                "mark": float(p.get('markPrice', 0)),
                "pnl": float(p.get('unrealizedProfit', 0))
            })
        return formatted
    except Exception as e:
        print(f"Positions Fetch Error: {e}")
        return []

@app.get("/api/v1/stats")
async def get_stats():
    stats = {"win_rate": 0, "total_trades": 0, "wins": 0, "losses": 0}
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                # Scan a larger window for accurate session stats
                lines = f.readlines()[-5000:] 
                wins = 0
                losses = 0
                for line in lines:
                    if "[PROFIT LOCK]" in line: wins += 1
                    if "STOP LOSS" in line: losses += 1
                
                total = wins + losses
                if total > 0:
                    stats["win_rate"] = (wins / total) * 100
                    stats["total_trades"] = total
                    stats["wins"] = wins
                    stats["losses"] = losses
    except: pass
    return stats

@app.get("/api/v1/signals")
async def get_signals():
    # Parse last 20 signals from log
    signals = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()[-200:]
                for line in lines:
                    if "SOVEREIGN SIGNAL" in line or "LIVE EXECUTION" in line:
                        signals.append(line.strip())
    except: pass
    return signals[-15:]

# Serve static files for the UI
if os.path.exists(os.path.join(BASE_DIR, "nexus_ui")):
    app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "nexus_ui"), html=True), name="ui")
