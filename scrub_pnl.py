
import json
import os

STATE_FILE = "sim_state.json"

def scrub_pnl():
    if not os.path.exists(STATE_FILE):
        print("No sim_state.json found.")
        return

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    if "history" not in state:
        print("No history found in state.")
        return

    print(f"Scrubbing {len(state['history'])} history records...")
    fixed_count = 0
    
    for trade in state["history"]:
        entry = trade["Entry"]
        exit = trade["Exit"]
        side = trade.get("Side", "BUY")
        old_pnl = trade["PnL"]
        
        # Determine expected profit/loss sign based on price move
        is_win = (exit > entry) if side in ["BUY", "LONG"] else (exit < entry)
        expected_sign = 1 if is_win else -1
        
        # Check if current PnL sign is wrong
        if (old_pnl > 0 and expected_sign < 0) or (old_pnl < 0 and expected_sign > 0):
            # We fix the PnL while preserving the absolute magnitude (if possible)
            # Actually, a better fix is to recalculate if we had the size,
            # but since we only have PnL, we just flip the sign.
            trade["PnL"] = -old_pnl
            print(f"Fixed {trade['Symbol']} {side}: {entry} -> {exit} | PnL {old_pnl} -> {trade['PnL']}")
            fixed_count += 1
            
    if fixed_count > 0:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
        print(f"Audit Complete. Fixed {fixed_count} corrupted trade records.")
    else:
        print("No errors found in data matrix.")

if __name__ == "__main__":
    scrub_pnl()
