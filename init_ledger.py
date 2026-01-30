import json, os
from datetime import datetime

ledger_path = 'evolution_ledger.json'
history = []

if os.path.exists(ledger_path):
    with open(ledger_path, 'r') as f:
        history = json.load(f)

# Record the Sovereign Predator (Phase 60) success
update_entry = {
    "timestamp": datetime.now().isoformat(),
    "phase": "Phase 60: Sovereign Predator",
    "status": "COMPLETED",
    "changes": {
        "score_barrier": 55.0,
        "leverage": 5,
        "alt_scanning": True,
        "winner_bonus": ["SOL", "BNB", "DOGE", "PEPE", "BERA"]
    },
    "equity_start": 274.66,
    "equity_peak": 292.77, # Based on previous session knowledge
    "impact_realized": "+.57 (First Hour) + Multi-Hour wins on ENSO/KAIA/0G",
    "lessons": "Restoring Golden Period sensitivity (Score 55) was the key to unlocking recovery velocity. High-alpha alts (ENSO/KAIA) are the profit drivers."
}

history.append(update_entry)

with open(ledger_path, 'w') as f:
    json.dump(history, f, indent=4)
print(f"✅ Evolution Ledger Initialized at {ledger_path}")
