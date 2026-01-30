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
    "phase": "Phase 62: Overnight Performance Audit (Jan 25)",
    "status": "SUCCESS",
    "changes": {
        "permission_boost_impact": "19 LIVE entries granted adaptive slippage",
        "major_wins": ["ENSO +75.8%", "KAIA +67.1%", "EUL +52.4%", "ENSO +48.8%"]
    },
    "equity_current": 296.77,
    "delta_overnight": "+.35",
    "lessons": "Phase 62 (Permission Boost) has successfully bridged the gap. By allowing 1.2% slippage on God-Tier signals (Score > 115), the LIVE engine captured the exact same moonshots that were previously only hitting in SIM."
}

history.append(entry)

with open(ledger_path, 'w') as f:
    json.dump(history, f, indent=4)
print("✅ Evolution Ledger Updated with Overnight Success Report.")
