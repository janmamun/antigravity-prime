#!/bin/bash
# SOVEREIGN TRADING SYSTEM - ONE-CLICK LAUNCHER
# Run this script to start everything: ./start_sovereign.sh

cd "$(dirname "$0")"
source venv/bin/activate

echo "=============================================="
echo "🦅 SOVEREIGN TRADING SYSTEM"
echo "=============================================="
echo ""

# Check if already running
if pgrep -f "run_v17_async" > /dev/null; then
    echo "⚠️  Trading bot is already running!"
    BOT_STATUS="✅ RUNNING"
else
    BOT_STATUS="❌ STOPPED"
fi

if pgrep -f "streamlit.*dashboard_antigravity" > /dev/null; then
    echo "⚠️  Dashboard is already running!"
    DASH_STATUS="✅ RUNNING"
else
    DASH_STATUS="❌ STOPPED"
fi

echo ""
echo "Current Status:"
echo "  Trading Bot: $BOT_STATUS"
echo "  Dashboard: $DASH_STATUS"
echo ""
echo "Options:"
echo "  1) Start Everything"
echo "  2) Start Trading Bot Only"
echo "  3) Start Dashboard Only"
echo "  4) Stop Everything"
echo "  5) View Live Status"
echo "  6) Open Dashboard in Browser"
echo "  q) Quit"
echo ""
read -p "Choose option: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Trading Bot..."
        pkill -f "run_v17_async" 2>/dev/null
        sleep 1
        PYTHONUNBUFFERED=1 nohup python3 run_v17_async.py > bot_output_async.log 2>&1 &
        echo "   Bot started with PID: $!"
        
        echo ""
        echo "📊 Starting Dashboard..."
        pkill -f "streamlit.*dashboard_antigravity" 2>/dev/null
        sleep 1
        nohup streamlit run dashboard_antigravity.py --server.port 8501 --server.headless true > dashboard.log 2>&1 &
        echo "   Dashboard started with PID: $!"
        
        sleep 3
        echo ""
        echo "🌐 Opening Dashboard in browser..."
        open "http://localhost:8501"
        
        echo ""
        echo "✅ System is now live!"
        echo "   Dashboard: http://localhost:8501"
        ;;
    2)
        echo ""
        echo "🚀 Starting Trading Bot..."
        pkill -f "run_v17_async" 2>/dev/null
        sleep 1
        PYTHONUNBUFFERED=1 nohup python3 run_v17_async.py > bot_output_async.log 2>&1 &
        echo "   Bot started with PID: $!"
        echo "✅ Trading bot is now live!"
        ;;
    3)
        echo ""
        echo "📊 Starting Dashboard..."
        pkill -f "streamlit.*dashboard_antigravity" 2>/dev/null
        sleep 1
        nohup streamlit run dashboard_antigravity.py --server.port 8501 --server.headless true > dashboard.log 2>&1 &
        echo "   Dashboard started with PID: $!"
        sleep 2
        open "http://localhost:8501"
        echo "✅ Dashboard is now live at http://localhost:8501"
        ;;
    4)
        echo ""
        echo "🛑 Stopping all processes..."
        pkill -f "run_v17_async" 2>/dev/null
        pkill -f "streamlit.*dashboard_antigravity" 2>/dev/null
        pkill -f "sovereign_trader" 2>/dev/null
        pkill -f "live_trader" 2>/dev/null
        echo "✅ All trading processes stopped."
        ;;
    5)
        echo ""
        echo "=== LIVE STATUS ==="
        python3 -c "
import ccxt
import os
from dotenv import load_dotenv
load_dotenv()
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})
try:
    balance = exchange.fetch_balance()
    usdt = float(balance.get('total', {}).get('USDT', 0))
    account = exchange.fapiPrivateV2GetAccount()
    positions = [p for p in account.get('positions', []) if float(p.get('positionAmt', 0)) != 0]
    total_pnl = sum(float(p.get('unrealizedProfit', 0)) for p in positions)
    print(f'💰 Balance: \${usdt:.2f}')
    print(f'📊 Open Positions: {len(positions)}')
    print(f'📈 Unrealized PnL: \${total_pnl:+.2f}')
    print(f'💵 Equity: \${usdt + total_pnl:.2f}')
    if positions:
        print()
        for p in positions:
            s = p.get('symbol', 'N/A')
            pnl = float(p.get('unrealizedProfit', 0))
            icon = '✅' if pnl >= 0 else '❌'
            print(f'  {s}: {icon} \${pnl:+.2f}')
except Exception as e:
    print(f'Error: {e}')
"
        ;;
    6)
        echo "🌐 Opening Dashboard..."
        open "http://localhost:8501"
        ;;
    q|Q)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo ""
echo "Press any key to exit..."
read -n 1
