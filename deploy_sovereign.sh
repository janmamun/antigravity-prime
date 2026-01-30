#!/延
# Phase 64: Deployment Forge (Sovereign Engine Auto-Provisioner)

echo "🏹 [SOVEREIGN FORGE] Initializing Deployment Protocol..."

# 1. Update & Dependencies
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git htop

# 2. Virtual Environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual Environment Created."
fi

source venv/bin/activate

# 3. Python Requirements
pip install pandas numpy ccxt python-dotenv httpx
echo "✅ Python Dependencies Installed."

# 4. PM2 (Watchdog) Install
if ! command -v pm2 &> /dev/null; then
    sudo apt-get install -y nodejs npm
    sudo npm install -g pm2
    echo "✅ PM2 Sentinel Installed."
fi

# 5. Verification
echo "🛡️ [SOVEREIGN FORGE] Verifying API Credentials..."
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file missing. Please upload your BINANCE_API_KEY and SECRET before starting."
else
    echo "✅ .env file detected."
fi

# 6. Launch Sequence
echo "🚀 [SOVEREIGN FORGE] Readiness Level: 100%. To launch, run:"
echo "pm2 start run_v17_async.py --name sovereign --interpreter venv/bin/python3"
