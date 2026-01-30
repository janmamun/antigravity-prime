import json, os
from datetime import datetime

ledger_path = 'evolution_ledger.json'
if os.path.exists(ledger_path):
    with open(ledger_path, 'r') as f:
        history = json.load(f)
else:
    history = []

entry = {
    "timestamp": datetime.now().isoformat(),
    "phase": "Phase 63: New Day Pursuit (Jan 25)",
    "status": "ACTIVE",
    "changes": {
        "daily_reset": True,
        "compounding_active": True,
        "mode": "Sovereign Predator"
    },
    "equity_start_of_day": 268.5, # Anticipated based on last night's lock
    "goal": "",
    "strategy_note": "Resuming with 15% compounding and adaptive slippage. The morning hunt is targeting volatility in high-cap L1s and proven alpha assets."
}

history.append(entry)

with open(ledger_path, 'w') as f:
    json.dump(history, f, indent=4)
print("✅ Evolution Ledger Updated: Jan 25 Morning Reset.")
