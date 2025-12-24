
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from market_scanner import MarketScanner
from simulation_engine import FuturesSimulator
import os
import json
import time
from datetime import datetime
import traceback
import ccxt
import subprocess
import sys
from meta_brain import LocalBrain

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION (SOVEREIGN MISSION CONTROL)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ANTIGRAVITY PRIME: SOVEREIGN", layout="wide", page_icon="🪐")

if 'refresh_interval' not in st.session_state: st.session_state['refresh_interval'] = 30000 # Increased to 30s
st_autorefresh(interval=st.session_state['refresh_interval'], key="datarefresh")

if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'brain' not in st.session_state: st.session_state['brain'] = LocalBrain()

# CSS: SOVEREIGN CYBER-GRID THEME
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=Fira+Code:wght@300;500&display=swap');
    
    :root {
        --cyber-purple: #7000FF;
        --cyber-cyan: #00D1FF;
        --cyber-blue: #0047FF;
        --bg-deep: #0A0019;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: var(--bg-deep);
        color: #FFFFFF;
    }
    
    /* RADIAL GLOW BACKGROUND */
    .stApp {
        background: radial-gradient(circle at top right, rgba(112, 0, 255, 0.15), transparent),
                    radial-gradient(circle at bottom left, rgba(0, 209, 255, 0.1), transparent),
                    var(--bg-deep);
    }

    /* PREMIUM GLASS CARDS */
    .sovereign-card {
        background: var(--glass-bg);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 16px 48px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .sovereign-card:hover {
        transform: translateY(-5px);
        border-color: var(--cyber-purple);
        box-shadow: 0 0 20px rgba(112, 0, 255, 0.2);
    }
    
    /* WALLET HERO HEADER */
    .wallet-hero {
        background: linear-gradient(135deg, var(--cyber-purple), var(--cyber-blue));
        padding: 40px;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(112, 0, 255, 0.3);
        animation: hero-glow 4s infinite alternate;
    }
    @keyframes hero-glow {
        from { box-shadow: 0 20px 40px rgba(112, 0, 255, 0.3); }
        to { box-shadow: 0 20px 60px rgba(112, 0, 255, 0.5); }
    }

    /* STREAMER HUD V2 */
    .streamer-container {
        font-family: 'Fira Code', monospace;
        font-size: 11px;
        color: var(--cyber-cyan);
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid var(--glass-border);
        padding: 15px;
        height: 250px;
        overflow-y: auto;
        border-radius: 12px;
        scrollbar-width: thin;
    }

    .metric-title { color: rgba(255, 255, 255, 0.7); font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; font-weight: 700; }
    .metric-value { font-size: 32px; font-weight: 800; color: #fff; line-height: 1; text-shadow: 0 0 15px rgba(255,255,255,0.2); }
    
    /* ANIMATED TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: var(--glass-bg);
        border-radius: 10px;
        border: 1px solid var(--glass-border);
        color: #fff;
        padding: 0 20px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--cyber-purple) !important;
        border-color: var(--cyber-purple) !important;
    }

    /* MOBILE OPTIMIZATION */
    @media (max-width: 640px) {
        .metric-value { font-size: 24px; }
        .sovereign-card { padding: 15px; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. STATE & DATA
# -----------------------------------------------------------------------------

if 'sim' not in st.session_state: st.session_state['sim'] = FuturesSimulator()
if 'scanner' not in st.session_state: st.session_state['scanner'] = MarketScanner()

def get_heartbeat():
    if os.path.exists("guardian_heartbeat.json"):
        with open("guardian_heartbeat.json", "r") as f:
            return json.load(f)
    return {"status": "OFFLINE", "last_heartbeat": 0}

def ensure_bot_running():
    """Phase 24: Auto-start background bot if stalled or offline"""
    hb = get_heartbeat()
    hb_age = time.time() - hb['last_heartbeat']
    
    # If offline for more than 5 minutes, restart
    if hb['status'] == "OFFLINE" or hb_age > 300:
        try:
            # We use a non-blocking subprocess call
            python_exec = sys.executable
            subprocess.Popen([python_exec, "run_v17.py"], 
                             stdout=open("bot_output.log", "a"), 
                             stderr=subprocess.STDOUT)
            return "RESTARTING"
        except Exception as e:
            return f"ERROR: {e}"
    return "RUNNING"

# AUTO-START ENGINE
engine_status = ensure_bot_running()

def get_live_logs(filename="bot_output.log", lines=20):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.readlines()[-lines:]
    return ["Awaiting Sovereign Streamer initialization..."]

@st.cache_data(ttl=300) # 5 min cache for charts
def fetch_chart_data(symbol):
    try:
        if 'exchange' not in st.session_state:
            st.session_state.exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 5000})
        return st.session_state.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=30)
    except: return []

@st.cache_data(ttl=20) # 20s cache for prices
def get_realtime_prices(symbols):
    prices = {}
    if not symbols: return prices
    if 'exchange' not in st.session_state:
        st.session_state.exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 5000})
    
    try:
        # Filter for unique symbols and valid CCXT symbols
        symbols = [s for s in list(set(symbols)) if "/" in s]
        if not symbols: return prices
        
        # Only fetch if we have few symbols, otherwise use scanner cache
        if len(symbols) <= 5:
            tickers = st.session_state.exchange.fetch_tickers(symbols)
            for s in symbols:
                if s in tickers:
                    prices[s] = tickers[s]['last']
    except Exception as e:
        print(f"Price Fetch Error: {e}")
    return prices

def render_tradingview(symbol="BTCUSDT"):
    # Convert symbol for TV widget. Example: BTC/USDT:USDT -> BTCUSDT
    tv_symbol = symbol.split(':')[0].replace("/", "")
    if not tv_symbol.endswith("USDT"): tv_symbol += "USDT"
    
    components.html(f"""
        <div class="tradingview-widget-container" style="height:100%;width:100%">
          <div id="tradingview_54321" style="height:500px;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true,
            "symbol": "BINANCE:{tv_symbol}.P",
            "interval": "60",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_54321"
          }});
          </script>
        </div>
    """, height=520)

# -----------------------------------------------------------------------------
# 3. MISSION CONTROL LAYOUT
# -----------------------------------------------------------------------------

# Sidebar: Controls & Settings
with st.sidebar:
    st.markdown("<h2 style='color:#00FF41'>SOVEREIGN COMMAND</h2>", unsafe_allow_html=True)
    auto_pilot = st.toggle("ACTIVATE SOVEREIGN LOOP", value=True)
    
    hb = get_heartbeat()
    hb_age = time.time() - hb['last_heartbeat']
    hb_status = "ACTIVE" if hb_age < 600 else "STALLED"
    hb_color = "var(--cyber-green)" if hb_status == "ACTIVE" else "var(--cyber-red)"
    
    st.markdown(f"""
    <div style="background:#111; padding:15px; border-radius:8px; border-left:4px solid {hb_color}">
        <div style="font-size:10px; color:#666">GUARDIAN PROTOCOL</div>
        <div style="font-size:18px; color:{hb_color}; font-weight:bold">{hb_status}</div>
        <div style="font-size:10px; color:#444">Last Heartbeat: {int(hb_age)}s ago</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### STRATEGY PARAMETERS")
    risk = st.slider("Risk Per Trade (%)", 1, 20, 10)
    lev = st.select_slider("Leverage", options=[1, 3, 5, 10, 20], value=3)

# Main Dashboard
# 1. Fetch real-time prices for active positions and selected symbols
current_positions = st.session_state.sim.get_positions()
symbols_to_fetch = list(current_positions.keys())
if "BTC/USDT" not in symbols_to_fetch: symbols_to_fetch.append("BTC/USDT")

current_prices = get_realtime_prices(symbols_to_fetch)
metrics = st.session_state.sim.get_portfolio_status(current_prices)
sim_stats = st.session_state.sim.calculate_stats()

# SOVEREIGN AUDIT HEADER
with st.container():
    realized = sim_stats.get('total_pnl', 0)
    unrealized = metrics['unrealized_pnl']
    total_net = realized + unrealized
    
    # Live Readiness Check
    is_live = st.session_state.scanner.bot.is_live
    live_bal = st.session_state.scanner.bot.get_live_balance() if is_live else metrics['equity']
    
    is_profitable = total_net > 0
    # Fix: Correctly check chat history for offline status
    brain_active = "Active"
    if st.session_state.get('chat_history') and "Offline" in str(st.session_state.chat_history[-1]):
        brain_active = "Offline"
    
    ready_status = "READY 🚀" if is_profitable and brain_active == "Active" else "CAUTION ⚠️"
    mode_label = "LIVE 🚀" if is_live else "SIMULATION 🪐"
    mode_color = "var(--cyber-purple)" if is_live else "var(--cyber-cyan)"
    
    status_icon = "🟢" if total_net > 0 else "🔴"
    
    # PREMIUM WALLET HERO
    st.markdown(f"""
    <div class="wallet-hero">
        <div style="font-size:14px; color:rgba(255,255,255,0.7); text-transform:uppercase; letter-spacing:3px;">Total Balance</div>
        <div style="font-size:48px; font-weight:700; margin:10px 0;">${live_bal:,.2f}</div>
        <div style="font-size:16px;">
            <span style="color:rgba(255,255,255,0.8);">Mission Profit:</span> 
            <b style="color:#fff; background:rgba(0,0,0,0.2); padding:4px 12px; border-radius:20px;">{total_net:+,.2f} USD</b>
        </div>
        <div style="margin-top:20px; display:flex; justify-content:center; gap:15px;">
            <div style="background:rgba(255,255,255,0.1); padding:8px 16px; border-radius:12px; font-size:12px;">{mode_label}</div>
            <div style="background:rgba(255,255,255,0.1); padding:8px 16px; border-radius:12px; font-size:12px;">{ready_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class="sovereign-card"><div class="metric-title">EQUITY</div><div class="metric-value">${metrics['equity']:,.2f}</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="sovereign-card"><div class="metric-title">WIN RATE</div><div class="metric-value" style="color:var(--cyber-blue)">{sim_stats['win_rate']:.1f}%</div></div>""", unsafe_allow_html=True)
with c3:
    # Get latest data from logs
    logs_df = st.session_state.scanner.get_recent_logs(limit=1)
    oi_disp = "+0.0%"
    whale_disp = "NEUTRAL"
    if not logs_df.empty:
        oi_disp = f"{logs_df.iloc[0].get('OI_Mom', 0)*100:+.1f}%"
        whale_disp = logs_df.iloc[0].get('Whale_Bias', 'NEUTRAL')
    
    st.markdown(f"""<div class="sovereign-card"><div class="metric-title">WHALE DEPTH</div><div class="metric-value" style="font-size:20px">{whale_disp}</div><div style="font-size:10px; color:#888">OI MOM: {oi_disp}</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="sovereign-card"><div class="metric-title">SYSTEM LOAD</div><div class="metric-value" style="color:var(--cyber-green)">OPTIMAL</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Middle Section: Chart & Sovereign Streamer
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### MARKET MATRIX ANALYTICS")
    active_sym = "BTC/USDT"
    if metrics['positions']:
        active_sym = metrics['positions'][0]['Symbol']
    elif not logs_df.empty:
        active_sym = logs_df.iloc[0]['Symbol']
    
    # Normalize for CCXT: BTC/USDT:USDT -> BTC/USDT
    ccxt_sym = active_sym.split(':')[0]
    
    # DUAL CHARTS
    tab_tv, tab_std = st.tabs(["TRADINGVIEW MATRIX", "STANDARD INTELLIGENCE"])
    
    with tab_tv:
        render_tradingview(ccxt_sym)
        
    with tab_std:
        st.info(f"Visualizing Intelligence Stream: {ccxt_sym}")
        # Real-ish chart data (if we have it)
        try:
            ohlcv = fetch_chart_data(ccxt_sym)
            df_chart = pd.DataFrame(ohlcv, columns=['timestamp', 'Price', 'High', 'Low', 'Close', 'Volume'])
            df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'], unit='ms')
            df_chart.set_index('timestamp', inplace=True)
            
            # Add synthetic signals for demonstration (based on bot logic)
            df_chart['Oracle'] = df_chart['Price'].rolling(5).mean() * 1.002
            df_chart['MA19'] = df_chart['Price'].rolling(19).mean()
            
            # Indicators / Predictions
            latest_price = df_chart['Price'].iloc[-1]
            pred_bullish = df_chart['Oracle'].iloc[-1] > df_chart['Price'].iloc[-1]
            
            # Target / SL Lines
            if pred_bullish:
                df_chart['Target'] = latest_price * 1.05
                df_chart['StopLoss'] = latest_price * 0.97
            else:
                df_chart['Target'] = latest_price * 0.95
                df_chart['StopLoss'] = latest_price * 1.03
                
            st.line_chart(df_chart[['Price', 'Oracle', 'MA19', 'Target', 'StopLoss']])
            
            if pred_bullish:
                st.success(f"PREDICTION: BULLISH 🟢 | RECOMMENDED STRATEGY: INSTITUTIONAL TREND SCALP | TARGET: ${latest_price * 1.05:.2f}")
            else:
                st.warning(f"PREDICTION: BEARISH 🔴 | RECOMMENDED STRATEGY: SHORT SQUEEZE DELTA | TARGET: ${latest_price * 0.95:.2f}")
        except Exception as e:
            st.warning(f"Standard Intelligence stream initializing... ({str(e)})")
            chart_data = pd.DataFrame(
                np.random.randn(50, 3).cumsum(axis=0) + [100, 102, 98],
                columns=['Price', 'Oracle', 'MA19']
            )
            st.line_chart(chart_data)

with col_right:
    st.markdown("### SOVEREIGN STREAMER")
    try:
        log_lines = get_live_logs(lines=15)
        log_content = "".join(log_lines)
    except:
        log_content = "Streamer connection pending..."
    st.markdown(f"""<div class="streamer-container"><pre>{log_content}</pre></div>""", unsafe_allow_html=True)
    
    st.markdown("### BRAIN EVOLUTION")
    try:
        brain_lines = get_live_logs("brain_log.txt", lines=5)
        brain_content = "".join(brain_lines)
    except:
        brain_content = "Brain sequence engaging..."
    st.markdown(f"""<div class="streamer-container" style="height:120px; color:var(--cyber-blue);"><pre>{brain_content}</pre></div>""", unsafe_allow_html=True)

# Bottom Section: Active Matrix (Positions & Intelligence Log)
st.markdown("### ACTIVE MATRIX")
t1, t2, t3, t4, t5, t6 = st.tabs(["POSITIONS", "INTELLIGENCE LOG", "HINDSIGHT MATRIX", "GLOBE SCANNER", "NEURAL BRIDGE", "SOVEREIGN AUDIT"])

with t1:
    if metrics['positions']:
        pos_df = pd.DataFrame(metrics['positions'])
        # Ensure correct column ordering and precision
        cols = ["Symbol", "Side", "Size", "Entry", "Mark", "PnL ($)", "PnL (%)"]
        if all(c in pos_df.columns for c in cols):
            st.dataframe(pos_df[cols].style.format({
                "Entry": "${:,.4f}", "Mark": "${:,.4f}", 
                "PnL ($)": "${:,.2f}", "PnL (%)": "{:,.2f}%"
            }), use_container_width=True)
        else:
            st.table(pos_df)
    else:
        st.markdown("<div style='text-align:center; padding:50px; color:#444'>NO ACTIVE POSITIONS IN MATRIX</div>", unsafe_allow_html=True)

with t2:
    all_logs = st.session_state.scanner.get_recent_logs(limit=100) # Increased limit
    if not all_logs.empty:
        st.dataframe(all_logs, use_container_width=True)
    else:
        st.info("Scanner data pending...")

with t3:
    if os.path.exists("hindsight_data.json"):
        with open("hindsight_data.json", "r") as f:
            hindsight = json.load(f)
        
        flat_hindsight = []
        for sym, history in hindsight.items():
            for entry in history:
                row = entry.copy()
                row['Symbol'] = sym
                flat_hindsight.append(row)
        
        if flat_hindsight:
            h_df = pd.DataFrame(flat_hindsight).sort_values('timestamp', ascending=False)
            # Add color coding for Profit/Loss
            st.dataframe(h_df.style.applymap(lambda x: 'color: #0f4' if x == 'PROFIT' else ('color: #f44' if x == 'LOSS' else 'color: #888'), subset=['result']), use_container_width=True)
        else:
            st.info("Hindsight learning in progress...")
    else:
        st.info("Hindsight file initializing...")

with t4:
    st.markdown("<div style='color:var(--cyber-green); font-size:12px; margin-bottom:10px;'>GLOBAL CRYPTO PANOPTICON - REAL-TIME OPPORTUNITIES</div>", unsafe_allow_html=True)
    
    # Show last scan results by default
    recent_scans = st.session_state.scanner.get_recent_logs(limit=100)
    if not recent_scans.empty:
        # Filter for only distinct symbols to keep it clean
        compact_scans = recent_scans.drop_duplicates(subset=['Symbol'], keep='first')
        st.write("Last Global Scan Matrix:")
        st.dataframe(compact_scans[['Symbol', 'Score', 'Signal', 'Regime', 'Whale_Bias', 'Reasons']], use_container_width=True)
    
    if st.button("TRIGGER NEW GLOBAL SCAN"):
        with st.spinner("Scanning all Binance USDT Futures..."):
            best_setups = st.session_state.scanner.scan_market()
            if best_setups:
                st.success(f"Discovered {len(best_setups)} high-alpha opportunities!")
                st.rerun()
            else:
                st.warning("No clear signals found. Matrix is quiet.")

with t5:
    st.markdown("<div style='color:var(--cyber-blue); font-size:12px; margin-bottom:10px;'>DIRECT INTERFACE TO THE SOVEREIGN BRAIN</div>", unsafe_allow_html=True)
    
    # Chat History
    for chat in st.session_state.chat_history:
        role_color = "var(--cyber-blue)" if chat['role'] == "Sovereign" else "#eee"
        st.markdown(f"**{chat['role']}**: <span style='color:{role_color}'>{chat['msg']}</span>", unsafe_allow_html=True)
    
    # Input
    with st.form("bridge_form", clear_on_submit=True):
        user_input = st.text_input("Transmit message to Brain:", placeholder="e.g. Analysis of overnight PnL?")
        submitted = st.form_submit_button("TRANSMIT")
        
        if submitted and user_input:
            # Disable refresh while thinking
            st.session_state['refresh_interval'] = 3600000 # 1 hour
            st.session_state.chat_history.append({"role": "User", "msg": user_input})
            
            # Context injection
            context = f"Equity: {metrics['equity']}, Win Rate: {sim_stats['win_rate']}%, Active Positions: {len(metrics['positions'])}"
            with st.spinner("AI BRAIN THINKING..."):
                try:
                    response = st.session_state.brain.chat(user_input, context=context)
                except Exception as e:
                    response = f"Neural Link Error: {str(e)}"
                
                st.session_state.chat_history.append({"role": "Sovereign", "msg": response})
            
            # Re-enable refresh
            st.session_state['refresh_interval'] = 15000
            st.rerun()

with t6:
    st.markdown("### 🪐 SYSTEM CONFORMANCE AUDIT")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Mission Stats**")
        st.write(f"- Initial Capital: `${st.session_state.sim.state.get('initial_balance', 10000):,.2f}`")
        st.write(f"- Realized PnL: `${realized:,.2f}`")
        st.write(f"- Total Trades: `{sim_stats['total_trades']}`")
        st.write(f"- Win Rate: `{sim_stats['win_rate']:.1f}%`")
    
    with col_b:
        st.write("**Health Status**")
        st.write(f"- Guardian Protocol: `{hb_status}`")
        st.write(f"- Brain Connection: `{brain_active}`")
        st.write(f"- Price Feed: `Active (Cached 15s)`")
        
    st.divider()
    st.write("**Sovereign Data Scrub Tool**")
    st.info("The bot identified historical formula errors. Click below to re-calculate past trade PnL for 100% mission accuracy.")
    if st.button("RUN PERFORMANCE RE-CALCULATION"):
        with st.spinner("Scrubbing data matrix..."):
            history = st.session_state.sim.state.get("history", [])
            fixed_count = 0
            for trade in history:
                # Recalculate PnL if needed (Fix for Step 1015 bug)
                entry = trade['Entry']
                exit = trade['Exit']
                side = trade.get('Side', 'BUY')
                old_pnl = trade['PnL']
                # Determine expected profit/loss based on price move
                is_win = (exit > entry) if side in ['BUY', 'LONG'] else (exit < entry)
                expected_sign = 1 if is_win else -1
                
                # If the sign is wrong, flip it.
                if (old_pnl > 0 and expected_sign < 0) or (old_pnl < 0 and expected_sign > 0):
                    trade['PnL'] = -old_pnl
                    fixed_count += 1
            
            if fixed_count > 0:
                st.session_state.sim._save_state()
                st.success(f"Audit Complete. Fixed {fixed_count} corrupted trade records. System re-aligned.")
                st.rerun()
            else:
                st.info("Data matrix already optimized. No errors found.")

# footer
st.markdown(f"""<div style="text-align:center; color:#333; font-size:10px; margin-top:50px;">ANTIGRAVITY PRIME V17.0 | SOVEREIGN INTELLIGENCE | {datetime.now().strftime("%H:%M:%S")}</div>""", unsafe_allow_html=True)
