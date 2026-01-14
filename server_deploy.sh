#!/bin/bash

# Sovereign V13.0 - Server Deploy Script
# Optimized for Ubuntu 22.04 LTS

echo "🚀 Initiating Sovereign Server Setup..."

# 1. Update & Install Base Dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl nodejs npm build-essential

# 2. Install PM2 (Process Manager)
sudo npm install -g pm2

# 3. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 4. Install ML & Trading Dependencies
pip install --upgrade pip
pip install pandas numpy ccxt scikit-learn streamlit joblib python-dotenv matplotlib requests httpx

# 5. Setup PM2 for Persistence
pm2 startup
sudo env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u $USER --hp $HOME

echo "✅ Sovereign Dependencies Installed."
echo "⚠️  CRITICAL: Please copy your .env file to $(pwd)/.env"
echo "👉 To start the bot, run: pm2 start ecosystem.config.js"
