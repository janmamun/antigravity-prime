
import streamlit as st
import pandas as pd
import time
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from market_scanner import MarketScanner
from simulation_engine import FuturesSimulator
from llm_brain import LLMBrain
import toml
import os
import importlib

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="GOD MODE TERMINAL", layout="wide", page_icon="🦅")

# CSS: PROFESSIONAL DARK THEME
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0e11; /* Binance Dark */
        color: #eaecef;
    }
    
    /* TABLES */
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    
    /* STATUS INDICATORS */
    .status-accepted { color: #00E676; font-weight: bold; }
    .status-rejected { color: #FF1744; font-weight: bold; }
    
    /* AUTO-PILOT SWITCH */
    .auto-pilot-on {
        color: #00E676;
        font-weight: 900;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. STATE & UTILS
# -----------------------------------------------------------------------------

SECRETS_PATH = ".streamlit/secrets.toml"

def save_api_key(key):
    try:
        os.makedirs(".streamlit", exist_ok=True)
        current = {}
        if os.path.exists(SECRETS_PATH): current = toml.load(SECRETS_PATH)
        current['GEMINI_API_KEY'] = key
        with open(SECRETS_PATH, 'w') as f: toml.dump(current, f)
        st.toast("🔑 API Key Saved Securely!", icon="✅")
    except Exception as e: st.error(f"Failed to save key: {e}")

# Initialize State
if 'sim' not in st.session_state: st.session_state['sim'] = FuturesSimulator()
if 'scanner' not in st.session_state: st.session_state['scanner'] = MarketScanner()

# Self-Healing
if not hasattr(st.session_state.sim, 'get_positions'):
     del st.session_state['sim']
     importlib.reload(importlib.import_module('simulation_engine'))
     st.session_state['sim'] = FuturesSimulator()

# -----------------------------------------------------------------------------
# 3. TRADINGVIEW WIDGET
# -----------------------------------------------------------------------------

def render_tradingview_widget(symbol="BINANCE:BTCUSDT.P"):
    # Professional TradingView Widget
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_123456"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 600,
        "symbol": "{symbol}",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "details": true,
        "hotlist": true,
        "calendar": true,
        "container_id": "tradingview_123456"
      }}
      );
      </script>
    </div>
    """
    components.html(html, height=600)

# -----------------------------------------------------------------------------
# 4. SIDEBAR & AUTO-PILOT
# -----------------------------------------------------------------------------

with st.sidebar:
    st.image("https://cryptologos.cc/logos/solana-sol-logo.png", width=40)
    st.title("GOD MODE V10")
    st.markdown("`QUANTUM EDITION`")
    st.markdown("---")
    
    # EXECUTION MODE
    st.subheader("⚙️ ENGINE MODE")
    mode = st.radio("EXECUTION LAYER", ["PAPER SIMULATION", "LIVE REAL MONEY"], index=0)
    
    # Load Keys
    try: 
        binance_key = st.secrets["BINANCE_API_KEY"]
        binance_secret = st.secrets["BINANCE_SECRET"]
    except:
        binance_key = ""
        binance_secret = ""
        
    if mode == "LIVE REAL MONEY":
        st.error("⚠️ REAL MONEY TRADING ACTIVE")
        st.markdown("**WARNING**: The bot will execute REAL trades on your Binance Futures account. Ensure you have USDT collateral.")
        
        # Check integrity
        if not binance_key or not binance_secret:
            st.warning("BINANCE KEYS MISSING IN SECRETS")
            binance_key = st.text_input("Binance API Key", type="password")
            binance_secret = st.text_input("Binance Secret", type="password")
            
    # Re-Init Simulator if Mode Changed
    sim_mode = 'LIVE' if mode == "LIVE REAL MONEY" else 'PAPER'
    if st.session_state.sim.mode != sim_mode:
        st.session_state['sim'] = FuturesSimulator(mode=sim_mode, api_key=binance_key, api_secret=binance_secret)
        st.toast(f"Switched to {sim_mode} Mode", icon="🔄")
    
    st.markdown("---")
    st.subheader("🤖 AUTO-PILOT")
    auto_pilot = st.toggle("ACTIVATE SYSTEM", key="autopilot")
    
    if auto_pilot:
        st.markdown("<p class='auto-pilot-on'>⚡ HUNTING ACTIVE</p>", unsafe_allow_html=True)
        refresh_rate = 5000 
    else:
        st.markdown("🔴 SYSTEM STANDBY")
        refresh_rate = 30000 
    
    st_autorefresh(interval=refresh_rate, limit=None, key="facer")
    
    st.markdown("---")
    try: existing_key = st.secrets["GEMINI_API_KEY"]
    except: existing_key = ""
    api_key_input = st.text_input("Gemini API Key", value=existing_key, type="password")
    if api_key_input and api_key_input != existing_key: save_api_key(api_key_input)
    
    oracle = LLMBrain(api_key_input) if api_key_input else None

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD (PROFESSIONAL LAYOUT)
# -----------------------------------------------------------------------------

# --- METRICS BAR ---
col1, col2, col3 = st.columns(3)
metrics = st.session_state.sim.get_portfolio_status({})
with col1: st.metric("EQUITY", f"${metrics['equity']:,.2f}", f"{metrics['unrealized_pnl']:+.2f}")
with col2: st.metric("MARGIN USED", f"${metrics['margin_used']:,.2f}")
with col3: st.metric("FREE MARGIN", f"${metrics['free_margin']:,.2f}")

st.markdown("---")

# --- AUTO PILOT LOGIC ---
scan_results = []
if auto_pilot:
    with st.spinner("Scanning..."):
        scan_results = st.session_state.scanner.scan_market()
        # Auto-Execute
        for op in scan_results:
            if op['score'] > 65:
                # Check duplicate
                if not any(p['Symbol'] == op['symbol'] for p in metrics['positions']):
                    st.toast(f"⚡ EXECUTING: {op['symbol']}", icon="🤖")
                    st.session_state.sim.execute_trade(op['symbol'], 'BUY' if op['signal']=='BUY' else 'SELL', op['entry'], 1000, op['tp'], op['sl'])

# --- MAIN SPLIT ---
m1, m2 = st.columns([2, 1])

with m1:
    st.subheader("📈 MARKET INTELLIGENCE")
    
    # Active Position Handling for Chart
    active_symbol = "BINANCE:BTCUSDT.P"
    if metrics['positions']:
        active_symbol = f"BINANCE:{metrics['positions'][0]['Symbol'].replace('/','')}P" # Approx
    
    render_tradingview_widget(active_symbol)
    
    st.divider()
    st.subheader("🛡️ SYSTEM INTEGRITY LOG (Last 50 Scans)")
    logs = st.session_state.scanner.get_recent_logs()
    
    # Stylize Log Table
    def color_status(val):
        color = '#00E676' if val == 'ACCEPTED' else '#ff4b4b'
        return f'color: {color}; font-weight: bold'
        
    if not logs.empty:
        st.dataframe(
            logs.style.applymap(color_status, subset=['Status']),
            use_container_width=True,
            height=300
        )
    else:
        st.info("No scan logs available yet. Activate System.")

with m2:
    st.subheader("⚔️ ACTIVE POSITIONS")
    
    # Calculate Live PnL
    # Simplified: We don't have streaming prices in this variable, but simulation_engine checks on update
    # Here we show the snapshot
    
    if not metrics['positions']:
        st.info("NO ACTIVE TRADES")
    else:
        for pos in metrics['positions']:
            with st.container():
                st.markdown(f"""
                <div style="background:#1e2329; padding:15px; border-radius:4px; margin-bottom:10px; border-left: 4px solid #F0B90B;">
                    <div style="display:flex; justify-content:space-between;">
                        <h4 style="margin:0">{pos['Symbol']}</h4>
                        <span style="color:{'#00E676' if pos['PnL ($)']>=0 else '#ff4b4b'}">{pos['PnL ($)']:+.2f}</span>
                    </div>
                    <div style="font-size:12px; color:#848e9c; margin-top:5px;">
                        Entry: {pos['Entry']} | Side: {pos['Side']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"CLOSE {pos['Symbol']}", key=f"cl_{pos['Symbol']}"):
                    st.session_state.sim.close_position(pos['Symbol'], pos['Mark'])
                    st.rerun()

    st.divider()
    st.subheader("🔮 ORACLE INSIGHT")
    if oracle:
        if st.button("Generate Analysis"):
             with st.spinner("Analyzing..."):
                ctx = f"Equity: {metrics['equity']}. Positions: {len(metrics['positions'])}. Last Log: {logs.iloc[0]['Symbol'] if not logs.empty else 'None'}"
                insight = oracle.analyze_market(ctx)
                st.info(insight)
    else:
        st.warning("Enter API Key to enable AI")
