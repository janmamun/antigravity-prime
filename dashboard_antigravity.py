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
# 1. SOVEREIGN ENGINE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="x.ANTIGRAVITY", layout="wide", page_icon="🪐")

st_autorefresh(interval=30000, key="datarefresh")

# UI INJECTION: THE MISSION CONSOLE ENGINE
# UI INJECTION: THE MISSION CONSOLE ENGINE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --obsidian-deep: #080a0c;
        --obsidian-card: rgba(17, 21, 28, 0.7);
        --obsidian-border: rgba(255, 255, 255, 0.08);
        --accent-cyan: #00e5ff;
        --accent-purple: #9d50bb;
        --accent-emerald: #00ffa3;
        --text-main: #f0f2f5;
        --text-dim: #94a3b8;
    }

    /* PREMIUM ENGINE OVERRIDE */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--obsidian-deep) !important;
        background-color: var(--obsidian-deep) !important;
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }
    
    .stDecoration { display: none !important; }
    [data-testid="stHeader"] { background: transparent !important; }

    /* BENTO MODULE SHELLS */
    .bento-card {
        background: var(--obsidian-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--obsidian-border);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .bento-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
        box-shadow: 0 8px 30px rgba(0, 229, 255, 0.1);
        transform: translateY(-2px);
    }

    /* MISSION STATUS HEADER */
    .mission-header {
        background: rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid var(--obsidian-border);
        padding: 24px 0;
        margin-bottom: 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* TYPOGRAPHY SYSTEM */
    .metric-label { 
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.15em;
        color: var(--text-dim);
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value-xl {
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .metric-value-lg {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .pill-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        border: 1px solid var(--obsidian-border);
        background: rgba(255, 255, 255, 0.03);
    }

    /* INTERACTIVE ELEMENTS */
    button[kind="secondary"], button[kind="primary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--obsidian-border) !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 600 !important;
        transition: 0.2s all !important;
    }
    button[kind="primary"]:hover {
        background: var(--accent-cyan) !important;
        color: black !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3) !important;
    }
    
    .status-active-pulse {
        width: 10px;
        height: 10px;
        background: var(--accent-emerald);
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 12px var(--accent-emerald);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }

    /* DATAFRAME CLEANING */
    [data-testid="stDataFrame"] {
        background: transparent !important;
        border: 1px solid var(--obsidian-border) !important;
    }

</style>

<canvas id="sovereign-viewport"></canvas>

<!-- TECHNICAL ENGINE LIBRARIES -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>

<script>
    // --- SOVEREIGN NEBULA ENGINE (OPTIMIZED) ---
    const canvas = document.getElementById('sovereign-viewport');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false }); // Antialias off for speed
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(1); // Force 1x for performance

    // Nebula Particle System (Reduced Count)
    const particleCount = 300; 
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 80;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 80;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 80;
        
        colors[i * 3] = 0.1;
        colors[i * 3 + 1] = 0.3;
        colors[i * 3 + 2] = 0.7;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        size: 0.15,
        vertexColors: true,
        transparent: true,
        opacity: 0.4,
        blending: THREE.AdditiveBlending
    }));
    scene.add(points);
    camera.position.z = 30;

    let targetX = 0, targetY = 0;
    window.addEventListener('mousemove', (e) => {
        targetX = (e.clientX / window.innerWidth - 0.5) * 3;
        targetY = (e.clientY / window.innerHeight - 0.5) * 3;
    });

    function animate() {
        if (document.hidden) {
            setTimeout(() => requestAnimationFrame(animate), 500);
            return;
        }
        requestAnimationFrame(animate);
        points.rotation.y += 0.0005;
        camera.position.x += (targetX - camera.position.x) * 0.02;
        camera.position.y += (-targetY - camera.position.y) * 0.02;
        camera.lookAt(scene.position);
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // --- SOVEREIGN ENTRANCE ---
    setTimeout(() => {
        gsap.to(".bento-card", {
            opacity: 1,
            y: 0,
            duration: 1.5,
            stagger: 0.2,
            ease: "expo.out"
        });
        
        gsap.from(".bento-card", {
            opacity: 0,
            y: 30,
            duration: 0.8,
            stagger: 0.1,
            ease: "expo.out"
        });

        gsap.from(".grok-value-mid, .grok-value-large", {
            opacity: 0,
            scale: 0.9,
            duration: 1.2,
            delay: 0.4,
            ease: "elastic.out(1, 0.8)"
        });
    }, 100);
</script>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. INTEL ENGINE initialization
# -----------------------------------------------------------------------------
if 'sim' not in st.session_state: st.session_state['sim'] = FuturesSimulator()
if 'scanner' not in st.session_state: st.session_state['scanner'] = MarketScanner()

# DATA FETCH WRAPPER (ULTRA-RESILIENT)
def get_mission_data():
    """Fetch dashboard data with full resilience - never blocks UI render"""
    default_metrics = {
        'score': 0, 'regime': 'SYNCING', 'strategy': 'Initializing...',
        'positions': [], 'sentiment': 'NEUTRAL', 'is_squeezing': False, 'spread_pct': 0.01,
        'unrealized_pnl': 0.0, 'equity': 100.0, 'macro_bias': 'NEUTRAL', 'narrative': 'Loading...',
        'tidal_shift': 0.5, 'vacuum_score': 10, 'turbo_active': False, 'cvd': 0.0, 'tidal_roc': 0.0
    }
    default_stats = {'win_rate': 0, 'total_pnl': 0, 'history': []}
    
    try:
        # Get positions from simulator (local, fast)
        current_positions = st.session_state.sim.get_positions()
        symbols = list(current_positions.keys())
        if "BTC/USDT" not in symbols: symbols.append("BTC/USDT")
        
        # Light price fetch with timeout protection
        prices = {}
        try:
            prices = get_realtime_prices_light(symbols)
        except:
            pass
        
        # Get portfolio status from simulator/live
        try:
            if st.session_state.scanner.bot.is_live:
                live_bal = st.session_state.scanner.bot.get_live_balance()
                metrics = default_metrics.copy()
                metrics['equity'] = live_bal
            else:
                metrics = st.session_state.sim.get_portfolio_status(prices)
        except:
            metrics = default_metrics.copy()

        # Ensure all required keys exist
        for key, val in default_metrics.items():
            if key not in metrics:
                metrics[key] = val

        # Live Position Sync (wrapped for safety)
        try:
            if st.session_state.scanner.bot.is_live:
                live_pos = st.session_state.scanner.bot.get_active_positions()
                if live_pos:
                    live_symbols = [p.get('symbol') for p in live_pos if p.get('symbol')]
                    open_orders = st.session_state.scanner.bot.get_open_orders_for_symbols(live_symbols)
                    
                    formatted_live = []
                    for lp in live_pos:
                        symbol = lp.get('symbol', 'N/A')
                        notional = abs(lp.get('size', 0) * lp.get('entry', 0))
                        lev = lp.get('leverage', 1) or 1
                        margin = notional / lev if lev > 0 else 1
                        roi = (lp.get('unrealized_pnl', 0) / margin) * 100 if margin > 0 else 0
                        
                        # Extract TP/SL from open orders
                        tp_val = 0
                        sl_val = 0
                        if symbol in open_orders:
                            for o in open_orders[symbol]:
                                o_type = o.get('type', '').upper()
                                if 'TAKE_PROFIT' in o_type:
                                    tp_val = o.get('stopPrice') or o.get('price') or 0
                                elif 'STOP' in o_type:
                                    sl_val = o.get('stopPrice') or o.get('price') or 0
                        
                        formatted_live.append({
                            'Symbol': symbol,
                            'Mode': '🔥 LIVE',
                            'Side': lp.get('side', 'N/A'),
                            'Size': abs(lp.get('size', 0)),
                            'Entry': lp.get('entry', 0),
                            'Mark': lp.get('mark_price', 0),
                            'Liq': lp.get('liquidation_price', 0),
                            'PnL ($)': lp.get('unrealized_pnl', 0),
                            'PnL (%)': roi,
                            'Leverage': lev,
                            'TP': tp_val,
                            'SL': sl_val
                        })
                    
                    metrics['positions'] = formatted_live + metrics.get('positions', [])
                    metrics['unrealized_pnl'] = sum([p['PnL ($)'] for p in formatted_live]) + metrics.get('unrealized_pnl', 0)
        except Exception as e:
            print(f"Shadow Sync Error: {e}")
        
        # Phase 48: Intelligence Upgrade Metrics
        try:
            metrics['session_peak'] = 0.0
            if os.path.exists("session_peak_equity.json"):
                with open("session_peak_equity.json", "r") as f:
                    peak_data = json.load(f)
                    if peak_data.get('date') == datetime.now().strftime('%Y-%m-%d'):
                        metrics['session_peak'] = float(peak_data.get('peak', 0))
            
            # Trailing Halt Calc
            if metrics['session_peak'] > 0:
                metrics['trailing_halt'] = metrics['session_peak'] * 0.95 # 5% drawdown
            else:
                metrics['trailing_halt'] = 0.0
        except: pass
        
        # Scanner logs (local file read, safe)
        try:
            latest_scan = st.session_state.scanner.get_recent_logs(limit=1)
            
            if not hasattr(st.session_state.scanner.bot, 'persistence_store'):
                st.session_state.scanner.bot.persistence_store = {}
                
            if not latest_scan.empty:
                metrics['score'] = int(latest_scan.iloc[0].get('Score', 0))
                metrics['regime'] = latest_scan.iloc[0].get('Regime', 'NEUTRAL')
                metrics['strategy'] = str(latest_scan.iloc[0].get('Reasons', 'WAITING'))[:50] + "..."
                
                reasons_str = str(latest_scan.iloc[0].get('Reasons', ''))
                metrics['sentiment'] = "HYPED" if "HYPED" in reasons_str else ("FUD" if "FUD" in reasons_str else "NEUTRAL")
                metrics['is_squeezing'] = "SQUEEZE" in reasons_str
                metrics['turbo_active'] = "TURBO" in reasons_str
                metrics['tidal_roc'] = latest_scan.iloc[0].get('Tidal_Roc', 0.0)
                metrics['tidal_shift'] = (metrics['tidal_roc'] + 100) / 200.0
                narr_val = latest_scan.iloc[0].get('Narrative', 'N/A')
                metrics['narrative'] = str(narr_val) if pd.notna(narr_val) else 'N/A'
        except Exception as e:
            print(f"Scanner log read error: {e}")

        # Trade metrics (skip if slow)
        try:
            target_sym = symbols[0] if symbols else "BTC/USDT"
            trade_metrics = st.session_state.scanner.bot._get_trade_based_metrics(target_sym)
            metrics['cvd'] = trade_metrics.get('cvd', 0.0)
        except:
            pass

        # Vacuum detection (skip if slow)
        try:
            is_vacuum = st.session_state.scanner.bot._detect_liquidity_vacuum(symbols[0] if symbols else "BTC/USDT")
            metrics['vacuum_score'] = 85 if is_vacuum else 20
        except:
            pass

        # Stats from simulator (local, safe)
        try:
            # Use a lookback of 25 trades to make the Win Rate update with each current trade
            stats = st.session_state.sim.calculate_stats(lookback=25)
        except:
            stats = default_stats.copy()
        
        # Config (local file read)
        try:
            metrics['macro_bias'] = st.session_state.scanner.bot.config.get('macro_bias', 'NEUTRAL')
        except:
            pass
        
        # Spread fetch (skip if slow, don't block UI)
        try:
            if 'exchange' in st.session_state:
                ob = st.session_state.exchange.fetch_order_book(symbols[0] if symbols else "BTC/USDT", limit=2)
                if ob and 'asks' in ob and 'bids' in ob and ob['asks'] and ob['bids']:
                    metrics['spread_pct'] = (ob['asks'][0][0] - ob['bids'][0][0]) / ob['bids'][0][0] * 100
        except:
            pass
        
        return metrics, stats, prices
    except Exception as e:
        print(f"Mission data fetch error: {e}")
        return default_metrics, default_stats, {}

# -----------------------------------------------------------------------------
# 2. STATE & DATA (SOVEREIGN PERSISTENCE)
# -----------------------------------------------------------------------------

if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'brain' not in st.session_state: st.session_state['brain'] = LocalBrain()
if 'booted' not in st.session_state: st.session_state['booted'] = False # Set to false initially
if 'active_tab' not in st.session_state: st.session_state['active_tab'] = 0

def update_tab(index):
    st.session_state['active_tab'] = index

def get_heartbeat():
    if os.path.exists("guardian_heartbeat_async.json"):
        with open("guardian_heartbeat_async.json", "r") as f:
            return json.load(f)
    return {"status": "OFFLINE", "last_heartbeat": 0}

def get_live_logs(filename="bot_output_async.log", lines=20):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.readlines()[-lines:]
    return ["Awaiting Sovereign Streamer initialization..."]


def get_realtime_prices_light(symbols):
    """Lighter version for dashboard to prevent hangs"""
    prices = {}
    if not symbols: return prices
    try:
        if 'exchange' not in st.session_state:
            st.session_state.exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 2000})
        # Try to use fetch_tickers but with a very short timeout
        tickers = st.session_state.exchange.fetch_tickers(symbols[:5]) # Limit to top 5 for speed
        for s in symbols:
            if s in tickers: prices[s] = tickers[s]['last']
    except:
        pass
    return prices

def render_tradingview(symbol="BTCUSDT"):
    tv_symbol = symbol.split(':')[0].replace("/", "")
    if not tv_symbol.endswith("USDT"): tv_symbol += "USDT"
    
    html_code = """
        <div class="tradingview-widget-container" style="height:100%;width:100%; border: 1px solid var(--grok-zinc); border-radius: 20px; overflow: hidden;">
          <div id="tradingview_54321" style="height:500px;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({
            "autosize": true,
            "symbol": "BINANCE:{{SYMBOL}}.P",
            "interval": "15",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_54321"
          });
          </script>
        </div>
    """.replace("{{SYMBOL}}", tv_symbol)
    
    components.html(html_code, height=520)

# -----------------------------------------------------------------------------
# 3. MISSION CONTROL DATA EXECUTION
# MISSION CONTROL DATA FETCH - Always render UI first
default_boot_metrics = {
    'score': 0, 'regime': 'INITIALIZING', 'strategy': 'Waking up Sovereign...',
    'positions': [], 'sentiment': 'NEUTRAL', 'is_squeezing': False, 'spread_pct': 0.01,
    'unrealized_pnl': 0.0, 'equity': 100.0, 'macro_bias': 'NEUTRAL', 'narrative': 'Waking up...',
    'tidal_shift': 0.5, 'vacuum_score': 10, 'turbo_active': False, 'cvd': 0.0, 'tidal_roc': 0.0
}
default_boot_stats = {'win_rate': 0, 'total_pnl': 0, 'history': []}

if not st.session_state.booted:
    # First run: Use defaults and mark as booted
    metrics = default_boot_metrics.copy()
    stats = default_boot_stats.copy()
    current_prices = {}
    st.session_state.booted = True
else:
    # Subsequent runs: Fetch data with full resilience
    try:
        metrics, stats, current_prices = get_mission_data()
    except Exception as e:
        print(f"Dashboard data fetch failed: {e}")
        metrics = default_boot_metrics.copy()
        stats = default_boot_stats.copy()
        current_prices = {}

# Ensure metrics is never None/empty - UI must always render
if not metrics:
    metrics = default_boot_metrics.copy()
if not stats:
    stats = default_boot_stats.copy()

# Prepare sidebar variables
import requests
hb = get_heartbeat()
hb_age = time.time() - hb.get('last_heartbeat', 0)
mission_status = "ACTIVE" if hb_age < 300 else "STALLED"
last_pulse = hb_age 

# -----------------------------------------------------------------------------
# 5. SIDEBAR: CONTROL CENTER (PREMIUM OBSIDIAN)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5. SIDEBAR: CONTROL CENTER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
        <div style="text-align:center; padding: 20px 0; margin-bottom: 30px;">
            <div style="font-size: 56px; filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4));">🦅</div>
            <div class="metric-label" style="font-size: 14px; margin-top: 20px; color: var(--text-main);">SOVEREIGN // V17.0</div>
            <div class='pill-tag' style='color:var(--accent-cyan); border-color:var(--accent-cyan); margin-top:12px; font-size: 11px;'>SYSTEM {mission_status}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='metric-label' style='margin-left:5px;'>Network Topology</div>", unsafe_allow_html=True)
    
    # Check Brain Connection
    try:
        brain_check = requests.get("http://127.0.0.1:11434/api/tags", timeout=1).status_code == 200
        brain_status = "var(--accent-emerald)" if (st.session_state.brain.gemini_model or brain_check) else "var(--obsidian-border)"
    except:
        brain_status = "var(--accent-emerald)" if st.session_state.brain.gemini_model else "var(--obsidian-border)"
        
    # Evolution Sentinel Check
    sentinel_active = False
    if os.path.exists("neural_bridge.log"):
        mtime = os.path.getmtime("neural_bridge.log")
        if time.time() - mtime < 3700: sentinel_active = True
        
    sentinel_status = "var(--accent-emerald)" if sentinel_active else "var(--obsidian-border)"
    exchange_status = "var(--accent-emerald)" if st.session_state.scanner.bot.is_live else "var(--accent-purple)"
    pulse_status = "var(--accent-emerald)" if last_pulse < 300 else "var(--obsidian-border)"

    # Watchdog Check
    watchdog_active = False
    if os.path.exists("watchdog.log"):
        mtime = os.path.getmtime("watchdog.log")
        if time.time() - mtime < 300: watchdog_active = True
    watchdog_status = "var(--accent-emerald)" if watchdog_active else "var(--obsidian-border)"

    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--obsidian-border); border-radius: 12px; padding: 18px; margin-bottom: 24px;">
            <div style="margin-bottom: 12px; display:flex; align-items:center; gap:10px;"><div style="width:8px; height:8px; border-radius:50%; background:{pulse_status}; box-shadow:0 0 8px {pulse_status};"></div> <span style="font-size:11px; color:var(--text-dim); font-family:'JetBrains Mono';">System Pulse</span></div>
            <div style="margin-bottom: 12px; display:flex; align-items:center; gap:10px;"><div style="width:8px; height:8px; border-radius:50%; background:{brain_status}; box-shadow:0 0 8px {brain_status};"></div> <span style="font-size:11px; color:var(--text-dim); font-family:'JetBrains Mono';">Neural Core (AI)</span></div>
            <div style="margin-bottom: 12px; display:flex; align-items:center; gap:10px;"><div style="width:8px; height:8px; border-radius:50%; background:{sentinel_status}; box-shadow:0 0 8px {sentinel_status};"></div> <span style="font-size:11px; color:var(--text-dim); font-family:'JetBrains Mono';">Evolution Sentinel</span></div>
            <div style="margin-bottom: 12px; display:flex; align-items:center; gap:10px;"><div style="width:8px; height:8px; border-radius:50%; background:{watchdog_status}; box-shadow:0 0 8px {watchdog_status};"></div> <span style="font-size:11px; color:var(--text-dim); font-family:'JetBrains Mono';">Perpetual Watchdog</span></div>
            <div style="margin-bottom: 12px; display:flex; align-items:center; gap:10px;"><div style="width:8px; height:8px; border-radius:50%; background:{exchange_status}; box-shadow:0 0 8px {exchange_status};"></div> <span style="font-size:11px; color:var(--text-dim); font-family:'JetBrains Mono';">Binance Sync</span></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='metric-label' style='margin-top:20px;'>Risk & Allocation</div>", unsafe_allow_html=True)
    risk = st.select_slider("ACCOUNT RISK (%)", options=[1, 2, 5, 10, 15, 20, 25], value=10, key="sidebar_risk_slider")
    lev = st.select_slider("LEVERAGE MULTIPLIER", options=[1, 3, 5, 10, 20], value=5, key="sidebar_lev_slider")
    
    if st.button("TRIGGER GLOBAL SCAN", width="stretch", key="sidebar_scan_btn"):
        with st.spinner("SCANNING..."):
            import asyncio
            asyncio.run(st.session_state.scanner.scan_market())
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚨 EMERGENCY PURGE", type="primary", use_container_width=True):
        st.warning("PERFORMING EMERGENCY EXIT...")
        try:
            import ccxt, os
            from dotenv import load_dotenv
            load_dotenv()
            ex = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET'),
                'options': {'defaultType': 'future'}
            })
            bal = ex.fetch_balance()
            for p in bal.get('info', {}).get('positions', []):
                if float(p.get('positionAmt', 0)) != 0:
                    sym = p['symbol']
                    side = 'sell' if float(p['positionAmt']) > 0 else 'buy'
                    ex.create_order(sym.replace('USDT', '/USDT:USDT'), 'MARKET', side, abs(float(p['positionAmt'])), params={'reduceOnly': True})
            st.success("ALL POSITIONS PURGED.")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"PURGE FAILED: {e}")

# PROFESSIONAL MISSION HEADER (DENSE)
total_net = stats.get('total_pnl', 0) + metrics.get('unrealized_pnl', 0)
is_live = st.session_state.scanner.bot.is_live
live_bal = metrics['equity']

try:
    if is_live: live_bal = st.session_state.scanner.bot.get_live_balance()
except:
    pass

target_goal = 500.00 
initial_capital = st.session_state.scanner.bot.session_start_equity if hasattr(st.session_state.scanner.bot, 'session_start_equity') else 288.46
session_pnl = live_bal - initial_capital
compounding_pct = min(100, max(0, ((live_bal - initial_capital) / (target_goal - initial_capital)) * 100))

st.markdown(f"""
<div class="mission-header" style="margin-top: -40px;">
    <div style="flex: 1.5;">
        <div class="metric-label">Operational Capital</div>
        <div style="display:flex; align-items:baseline; gap:16px;">
            <div class="metric-value-xl">${live_bal:,.2f}</div>
            <div class='pill-tag' style='color:#00e5ff; border-color:#00e5ff;'>{ 'LIVE' if is_live else 'SIM' } SOURCE</div>
        </div>
        <div style="display:flex; align-items:center; gap:12px; margin-top:12px;">
            <div style="background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px; flex: 1; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #9d50bb, #6e48aa); width: {compounding_pct}%; height: 100%;"></div>
            </div>
            <div class="metric-label" style="font-size: 9px; margin-bottom: 0;">STAGE 1: $500.00</div>
        </div>
    </div>
    <div style="flex: 1; text-align: center; border-left: 1px solid var(--obsidian-border);">
        <div class="metric-label">Session Peak</div>
        <div class="metric-value-lg" style="color:var(--accent-emerald);">${metrics.get('session_peak', 0):,.2f}</div>
    </div>
    <div style="flex: 1; text-align: center; border-left: 1px solid var(--obsidian-border);">
        <div class="metric-label">Real-Time PnL</div>
        <div class="metric-value-lg" style="color:{ 'var(--accent-emerald)' if session_pnl >= 0 else 'var(--accent-purple)' }; font-size:24px;">{session_pnl:+,.2f} USD</div>
    </div>
    <div style="flex: 1; text-align: center; border-left: 1px solid var(--obsidian-border);">
        <div class="metric-label">System Integrity</div>
        <div class="metric-value-lg" style="color:var(--accent-cyan); font-size: 24px;">{stats['win_rate']:.1f}%</div>
    </div>
    <div style="flex: 0.8; text-align: right; border-left: 1px solid var(--obsidian-border); padding-left: 20px;">
        <div class="metric-label">Market Bias</div>
        <div class="metric-value-lg" style="color:{ 'var(--accent-emerald)' if metrics.get('macro_bias') == 'RISK-ON' else 'var(--accent-purple)' if metrics.get('macro_bias') == 'RISK-OFF' else 'var(--accent-cyan)' };">{metrics.get('macro_bias', 'NEUTRAL')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# INTELLIGENCE HUB (DENSE)
ic1, ic2, ic3, ic4 = st.columns(4)
conf_score = metrics.get('score', 0)
conf_pct = min(max(abs(conf_score), 0), 100)

with ic1: 
    conviction_color = "var(--accent-cyan)"
    if conf_score > 120: conviction_color = "var(--accent-emerald)"
    elif conf_score < 60: conviction_color = "var(--accent-purple)"
    st.markdown(f"<div class='bento-card' style='height:140px; border-top: 2px solid {conviction_color};'><div class='metric-label'>Neural Conviction</div><div class='metric-value-lg' style='color:{conviction_color}; margin-top:8px;'>{conf_score:+d}</div><div style='background:rgba(255,255,255,0.05); height:4px; border-radius:2px; margin-top:12px;'><div style='width:{conf_pct}%; height:100%; background:{conviction_color};'></div></div></div>", unsafe_allow_html=True)
with ic2: st.markdown(f"<div class='bento-card' style='height:140px;'><div class='metric-label'>Structural Regime</div><div class='metric-value-lg' style='margin-top:8px; font-size:22px;'>{metrics.get('regime', 'NEUTRAL')}</div></div>", unsafe_allow_html=True)
with ic3: st.markdown(f"<div class='bento-card' style='height:140px;'><div class='metric-label'>Institutional Bias</div><div class='metric-value-lg' style='margin-top:8px; font-size:22px; color:{ 'var(--accent-emerald)' if metrics.get('sentiment') == 'HYPED' else ('var(--accent-purple)' if metrics.get('sentiment') == 'FUD' else 'white') }'>{metrics.get('sentiment', 'NEUTRAL')}</div></div>", unsafe_allow_html=True)
with ic4:
    squeeze_status = "ACTIVE" if metrics.get('is_squeezing') else "STANDBY"
    st.markdown(f"<div class='bento-card' style='height:140px;'><div class='metric-label'>Volatility Squeeze</div><div class='metric-value-lg' style='margin-top:8px; font-size:22px; color:{ 'var(--accent-cyan)' if metrics.get('is_squeezing') else 'var(--text-dim)' }'>{squeeze_status}</div></div>", unsafe_allow_html=True)

# Phase 15: TIDAL PULSE BAR (Institutional Flow)
st.markdown(f"""
<div style='margin-bottom: 32px; background: var(--obsidian-card); border: 1px solid var(--obsidian-border); border-radius: 12px; padding: 20px;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div class='metric-label' style='font-size: 11px; margin-bottom:0;'>INSTITUTIONAL TIDAL SHIFT (PHASE 51)</div>
        <div class='pill-tag' style='color: {"var(--accent-emerald)" if metrics.get("turbo_active") else "var(--accent-cyan)"}; border-color:transparent;'>{ "TURBO BOOSTED" if metrics.get("turbo_active") else "NOMINAL FLOW" }</div>
    </div>
    <div style='background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; margin-top: 16px; overflow: hidden;'>
        <div style='background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan), var(--accent-emerald)); width: {min(100, int(metrics.get("tidal_shift", 0) * 100))}%; height: 100%; transition: width 0.8s ease;'></div>
    </div>
</div>
""", unsafe_allow_html=True)

# BLOOMBERG WORKSPACE (STRICT 70/30 SPLIT)
col_l, col_r = st.columns([7, 3]) 
with col_l:
    st.markdown("<div class='metric-label'>Terminal Visualization</div>", unsafe_allow_html=True)
    active_sym = metrics['positions'][0]['Symbol'] if metrics['positions'] else "BTC/USDT"
    render_tradingview(active_sym)

with col_r:
    st.markdown("<div class='metric-label'>Neural Streamer</div>", unsafe_allow_html=True)
    all_logs = get_live_logs(lines=100)
    
    log_content = "".join(all_logs[-30:])
    st.markdown(f"""
        <div style='height:420px; background:var(--obsidian-card); border:1px solid var(--obsidian-border); border-radius:12px; padding:15px; font-family:"JetBrains Mono"; font-size:11px; overflow-y:auto; color:var(--text-dim); line-height:1.6;'>
            <pre style='white-space:pre-wrap; word-wrap:break-word; margin:0;'>{log_content}</pre>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='margin-top:24px;'>
        <div class='metric-label' style='font-size:10px; opacity:0.5; margin-bottom:8px;'>LIQUIDITY GAP (ENTROPY)</div>
        <div style='background: rgba(255,255,255,0.05); border-radius: 4px; height: 10px; overflow: hidden; border: 1px solid var(--obsidian-border);'>
            <div style='background: linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan), var(--accent-purple)); width: {min(100, int(metrics.get("spread_pct", 0) * 1000))}%; height: 100%;'></div>
        </div>
    </div>
    <div style='margin-top:20px;'>
        <div class='metric-label' style='font-size:10px; opacity:0.5; margin-bottom:8px;'>V15 VACUUM DETECTION</div>
        <div style='background: rgba(255,255,255,0.05); border-radius: 4px; height: 10px; overflow: hidden; border: 1px solid var(--obsidian-border);'>
            <div style='background: linear-gradient(90deg, var(--accent-cyan), var(--accent-emerald)); width: {metrics.get("vacuum_score", 10)}%; height: 100%;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# TABS: SOVEREIGN GRID (PERSISTENT)
st.markdown("<br>", unsafe_allow_html=True)
tab_titles = ["EXPOSURE", "AUDIT", "HISTORY", "SCANNER", "MEMORY", "NEURAL", "SYNDICATE"]
t1, t2, t3, t4, tm, t5, t6 = st.tabs(tab_titles)

# Streamlit tabs don't have a direct 'index' sync, but we can simulate it or just let the user navigate.
# To ensure the chat doesn't feel like it "disappeared", we'll check if a message was just sent.
if 'last_action' in st.session_state and st.session_state.last_action == 'chat':
    # This is a hack because st.tabs doesn't let us programmatically switch easily in this version
    # But we can at least show a notification.
    st.toast("Neural Transmission Received", icon="🦅")
    del st.session_state['last_action']

with t1:
    sub_live, sub_sim = st.tabs(["🔥 LIVE POSITION MANAGEMENT", "🧪 STRATEGY SIMULATION"])
    
    with sub_live:
        live_positions = [p for p in metrics.get('positions', []) if "LIVE" in p.get('Mode', '')]
        if live_positions:
            for pos in live_positions:
                side_color = "var(--accent-emerald)" if pos['Side'] == "BUY" else "var(--accent-purple)"
                pnl_color = "var(--accent-emerald)" if pos['PnL ($)'] >= 0 else "var(--accent-purple)"
                symbol = pos['Symbol']
                tv_symbol = f"BINANCE:{symbol.replace('/USDT:USDT', 'USDT').replace('/', '')}.P"
                
                card_html = f"""
<div class='bento-card' style='padding: 0; border-left: 2px solid {side_color}; overflow: hidden; margin-bottom: 16px;'>
    <div style='padding: 12px 16px; border-bottom: 1px solid var(--obsidian-border); display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <div class='metric-label' style='margin-bottom: 2px; font-size: 10px;'>{symbol} // {pos.get('Leverage', 1)}x</div>
            <div style='color:{side_color}; font-weight: 800; font-size: 16px;'>{pos['Side']} POSITION</div>
        </div>
        <div style='text-align: right;'>
            <div class='metric-label' style='font-size: 10px;'>Session PnL</div>
            <div style='color: {pnl_color}; font-weight: 800; font-size: 16px;'>{pos['PnL ($)']:+,.2f} USD ({pos['PnL (%)']:+,.2f}%)</div>
        </div>
    </div>
    <div style='height: 220px;'>
        <iframe 
            src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_762ed&symbol={tv_symbol}&interval=1&hidesidetoolbar=1&hidetoptoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC&studies_overrides=%7B%7D&overrides=%7B%7D&enabled_features=%5B%5D&disabled_features=%5B%5D&locale=en&utm_source=antigravity"
            width="100%" height="100%" frameborder="0" allowtransparency="true" scrolling="no" allowfullscreen>
        </iframe>
    </div>
    <div style='padding: 12px 16px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: rgba(255,255,255,0.01);'>
        <div><div class='metric-label' style='font-size: 9px;'>Size</div><div style='font-size: 12px; font-weight: 600;'>{pos['Size']:.4f}</div></div>
        <div><div class='metric-label' style='font-size: 9px;'>Entry</div><div style='font-size: 12px; color: var(--accent-cyan); font-weight: 600;'>{pos['Entry']:,.4f}</div></div>
        <div><div class='metric-label' style='font-size: 9px;'>Mark</div><div style='font-size: 12px; font-weight: 600;'>{pos['Mark']:,.4f}</div></div>
        <div><div class='metric-label' style='font-size: 9px;'>Liquidation</div><div style='font-size: 12px; color: #ffab00; font-weight: 600;'>{pos.get('Liq', 0):,.4f}</div></div>
    </div>
</div>
"""
                st.markdown(card_html, unsafe_allow_html=True)
                
                # LIVE CONTROLS ROW
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                with c1:
                    if st.button("🚨 TERMINATE", key=f"close_btn_{symbol}", use_container_width=True):
                        res = st.session_state.scanner.bot.close_live_position(symbol)
                        if res['status'] == 'SUCCESS': st.success(f"Terminated {symbol}")
                        else: st.error(res['msg'])
                        st.rerun()
                
                with c2:
                    current_tp = float(pos.get('TP', 0)) if pos.get('TP') else 0.0
                    new_tp = st.number_input("LIMIT TP", value=current_tp, key=f"tp_in_{symbol}", format="%.5f", label_visibility="visible")
                with c3:
                    current_sl = float(pos.get('SL', 0)) if pos.get('SL') else 0.0
                    new_sl = st.number_input("STOP SL", value=current_sl, key=f"sl_in_{symbol}", format="%.5f", label_visibility="visible")
                with c4:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    if st.button("🛰️ ATOMIC UPDATE", key=f"upd_btn_{symbol}", use_container_width=True):
                        res = st.session_state.scanner.bot.update_live_tp_sl(symbol, pos['Side'], new_tp, new_sl)
                        if res['status'] == 'SUCCESS': st.success(f"Atomic Update: {symbol}")
                        else: st.error(res['msg'])
                        st.rerun()
                st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='border:1px solid var(--obsidian-border); border-radius:12px; padding:80px; text-align:center; color:var(--text-dim); background:var(--obsidian-card);'>AWAITING ALPHA DEPLOYMENT // NO LIVE EXPOSURE</div>", unsafe_allow_html=True)

    with sub_sim:
        sim_positions = [p for p in metrics.get('positions', []) if "SIM" in p.get('Mode', '')]
        if sim_positions:
            for pos in sim_positions:
                side_color = "var(--accent-emerald)" if pos['Side'] == "BUY" else "var(--accent-purple)"
                pnl_color = "var(--accent-emerald)" if pos['PnL ($)'] >= 0 else "var(--accent-purple)"
                st.markdown(f"""
                    <div class='bento-card' style='padding: 12px 16px; border-left: 2px solid var(--obsidian-border); margin-bottom: 8px; opacity: 0.6;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span class='metric-label' style='margin-right:12px; font-size: 10px;'>{pos['Symbol']}</span>
                                <span class='pill-tag' style='padding: 2px 8px; font-size: 9px;'>SIMULATION</span>
                            </div>
                            <div style='color: {pnl_color}; font-weight: 700; font-size: 16px;'>{pos['PnL ($)']:+,.2f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; color:var(--text-dim); padding:40px;'>No simulation cycles active</div>", unsafe_allow_html=True)

with t2:
    st.markdown("<div class='metric-label'>Granular Execution Attribution</div>", unsafe_allow_html=True)
    
    col_audit1, col_audit2 = st.columns([7, 3])
    with col_audit1:
        if os.path.exists("sovereign_audit.json"):
            with open("sovereign_audit.json", "r") as f:
                audits = json.load(f)
            df_audit = pd.DataFrame(audits).sort_index(ascending=False)
            st.dataframe(df_audit, width="stretch")
        else:
            st.info("Registry clear. Monitoring standby.")

    with col_audit2:
        st.markdown("<div class='metric-label'>Asset Quarantine</div>", unsafe_allow_html=True)
        if os.path.exists("blacklist.json"):
            with open("blacklist.json", "r") as f:
                blacklist = json.load(f)
            if blacklist:
                for item in blacklist:
                    st.markdown(f"<div class='pill-tag' style='color:var(--accent-purple); border-color:var(--accent-purple); margin-bottom:8px; width:100%; text-align:center;'>💀 {item}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='opacity:0.3; font-size:11px;'>Quarantine Clear</div>", unsafe_allow_html=True)

with t3:
    hist_live, hist_sim = st.tabs(["🔥 LIVE AUDIT", "🧪 SIMULATION REPLAY"])
    with hist_live:
        st.markdown("<div class='metric-label'>Institutional Ledger</div>", unsafe_allow_html=True)
        if os.path.exists("sovereign_audit.json"):
            with open("sovereign_audit.json", "r") as f:
                audit_log = json.load(f)
            if audit_log:
                live_audit = [a for a in audit_log if "SIM" not in a.get('audit_tag', '')]
                if live_audit:
                    df_live = pd.DataFrame(live_audit).sort_values('timestamp', ascending=False)
                    st.dataframe(df_live, use_container_width=True)
                else:
                    st.info("No live cycles recorded.")
    with hist_sim:
        st.markdown("<div class='metric-label'>Simulation Replay Stream</div>", unsafe_allow_html=True)
        sim_history = stats.get('history', [])
        if sim_history:
            h_df = pd.DataFrame(sim_history).sort_values('Time', ascending=False)
            st.dataframe(h_df, use_container_width=True)

with tm:
    st.markdown("<div class='metric-label'>Neural Learning Density</div>", unsafe_allow_html=True)
    bot = st.session_state.scanner.bot
    memory_data = bot.memory.get("performance", {}) if hasattr(bot, 'memory') else {}
    if memory_data:
        m_rows = []
        for sym, data in memory_data.items():
            m_rows.append({"Asset": sym, "Win Rate": f"{(data.get('wins',0)/(data.get('wins',0)+data.get('losses',1))*100):.1f}%", "Streak": data.get("streak", 0)})
        st.dataframe(pd.DataFrame(m_rows), use_container_width=True)
    else:
        st.info("Neural Memory initializing...")

    st.markdown("<br><div class='grok-label'>Active Protections (Anti-Fragility)</div>", unsafe_allow_html=True)
    c_anti1, c_anti2 = st.columns(2)
    with c_anti1:
        st.markdown(f"""
        <div class='bento-card' style='border-left: 4px solid #00D1FF;'>
            <div class='grok-label'>Volatility Protector</div>
            <div style='font-size: 14px; opacity: 0.8;'>REJECTING SCAM WICKS ({st.session_state.scanner.bot.config.get('volatility_protector_multiplier', 4.0)}x Threshold)</div>
            <div style='color: #00FFAB; font-size: 12px; margin-top:5px;'>● ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)
    with c_anti2:
        st.markdown(f"""
        <div class='bento-card' style='border-left: 4px solid #FF00FF;'>
            <div class='grok-label'>Symbol Quarantining</div>
            <div style='font-size: 14px; opacity: 0.8;'>PENALIZING RECENT FAILURES (-20 pts)</div>
            <div style='color: #00FFAB; font-size: 12px; margin-top:5px;'>● ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)

with t4:
    st.markdown("<div class='grok-label'>Scanner Activity Stream</div>", unsafe_allow_html=True)
    scans = st.session_state.scanner.get_recent_logs(limit=50)
    if not scans.empty:
        st.dataframe(scans, width="stretch")
    else:
        st.info("Awaiting Market Scanner activity...")

with t5:
    st.markdown("<div class='grok-label'>Sovereign Neural Interface</div>", unsafe_allow_html=True)
    
    # Render chat history in an isolated, scrollable container via components.html for stability
    history_items = []
    if not st.session_state.chat_history:
        history_items.append('<div style="opacity: 0.3; font-size: 11px; color:white; font-family:sans-serif;">Awaiting mission transmissions...</div>')
    for chat in st.session_state.chat_history:
        role_label = "🦅 SOVEREIGN" if chat['role'] == "Sovereign" else "👤 COMMANDER"
        role_color = "#00D1FF" if chat['role'] == "Sovereign" else "#FFFFFF"
        item = f"""
            <div style="margin-bottom:15px; border-left: 2px solid {role_color}; padding-left: 15px; font-family: sans-serif;">
                <div style="color: {role_color}; opacity: 0.6; font-size: 10px; font-weight:700; text-transform:uppercase; letter-spacing:0.2em; margin-bottom:4px;">{role_label}</div>
                <div style="font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.9);">{chat['msg']}</div>
            </div>
        """
        history_items.append(item)
    
    full_history_inner = "".join(history_items)
    history_outer = f"""
    <div id="chat-scroller" style="height: 320px; overflow-y: auto; padding: 15px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px;">
        {full_history_inner}
        <div id="anchor"></div>
    </div>
    <script>
        var element = document.getElementById("chat-scroller");
        element.scrollTop = element.scrollHeight;
    </script>
    """
    components.html(history_outer, height=350)
    
    st.markdown("<div class='grok-label' style='margin-top:20px;'>Evolutionary Intelligence Feed</div>", unsafe_allow_html=True)
    if os.path.exists("neural_bridge.log"):
        with open("neural_bridge.log", "r") as f:
            bridge_logs = f.readlines()[-10:]
            st.markdown(f'<div class="streamer" style="height:120px; font-size:10px; opacity:0.6;"><pre>{"".join(bridge_logs)}</pre></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="streamer" style="height:120px; font-size:10px; opacity:0.6;"><pre>[STALED] Neural Bridge Offline...</pre></div>', unsafe_allow_html=True)

    with st.form("neural_bridge", clear_on_submit=True):
        u_in = st.text_input("COMMAND PROTOCOL:", placeholder="Initiate transmission...")
        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            submitted = st.form_submit_button("SEND")
        if submitted and u_in:
            st.session_state.chat_history.append({"role": "User", "msg": u_in})
            
            # Phase 7.0: Neural Directive Parsing (Hot-Patching)
            response_msg = ""
            directive_found = False
            
            cmd_lower = u_in.lower()
            if cmd_lower.startswith("eagle,"):
                directive = cmd_lower.replace("eagle,", "").strip()
                directive_found = True
                
                config_file = "bot_config.json"
                try:
                    with open(config_file, "r") as f: config = json.load(f)
                    
                    if "pause" in directive:
                        config["pause_trading"] = True
                        response_msg = "DIRECTIVE RECEIVED: HALTING ALL NEW ENTRIES. STANDBY."
                    elif "resume" in directive:
                        config["pause_trading"] = False
                        response_msg = "DIRECTIVE RECEIVED: RESUMING ALPHA-HUNT PROTOCOL."
                    elif "aggressive" in directive:
                        config["min_score"] = 55
                        config["risk_factor_pct"] = 0.15
                        response_msg = "DIRECTIVE RECEIVED: MAXIMIZING ALPHA AGGRESSION. RISK PARAMETERS ESCALATED."
                    elif "conservative" in directive:
                        config["min_score"] = 75
                        config["risk_factor_pct"] = 0.05
                        response_msg = "DIRECTIVE RECEIVED: DEFENSIVE PROTOCOL ENGAGED. TIGHTENING SHIP."
                    
                    with open(config_file, "w") as f: json.dump(config, f, indent=4)
                except Exception as e:
                    response_msg = f"DIRECTIVE FAILURE: {e}"

            if not directive_found:
                resp = st.session_state.brain.chat(u_in, context=f"Equity: {metrics['equity']} | PnL: {total_net}")
                response_msg = resp
                
            st.session_state.chat_history.append({"role": "Sovereign", "msg": response_msg})
            st.session_state.last_action = 'chat'
            st.rerun()

    # Phase 10: Neural Shadow Heatmap
    st.markdown("<div class='grok-label' style='margin-top:30px;'>Neural Shadow: Phase 10 Orderflow Heatmap</div>", unsafe_allow_html=True)
    
    # Generate mock heatmap values based on real scanner data if possible
    shadow_data = []
    import asyncio
    try:
        top_coins = asyncio.run(st.session_state.scanner.get_top_volume_coins(limit=10))
    except:
        top_coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for coin in top_coins:
        obi = st.session_state.scanner.bot.persistence_store.get(coin, {}).get('bids', [0])[-1] - st.session_state.scanner.bot.persistence_store.get(coin, {}).get('asks', [0])[-1]
        shadow_data.append({"Asset": coin, "Shadow_Density": obi})
    
    if shadow_data:
        sdf = pd.DataFrame(shadow_data)
        st.dataframe(sdf.style.background_gradient(cmap='RdYlGn', subset=['Shadow_Density']), width="stretch")
    else:
        st.info("Awaiting scan cycles to populate Neural Shadow Matrix...")

    st.markdown("<br><div class='grok-label'>Sovereign Advisor Feed (Gemini 2.0 + Alpha-Tuner)</div>", unsafe_allow_html=True)
    brain_logs = get_live_logs("brain_log.txt", lines=20)
    formatted_logs = []
    for line in brain_logs:
        if "--- [" in line:
            formatted_logs.append(f"<span style='color:var(--grok-accent); font-weight:700;'>{line}</span>")
        else:
            formatted_logs.append(line)
            
    st.markdown(f"<div class='streamer' style='height:300px; color:rgba(255,255,255,0.7); border: 1px solid rgba(255,255,255,0.1); background: rgba(0,209,255,0.02);'><pre>{''.join(formatted_logs)}</pre></div>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center; padding:100px 0 50px 0; color:rgba(255,255,255,0.1); font-family:JetBrains Mono; font-size:9px; letter-spacing:0.8em;'>x.ANTIGRAVITY // SOVEREIGN ENGINE // {datetime.now().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)

with t6:
    st.markdown("<div class='grok-label'>The Sentiment Syndicate (Gemini-Powered)</div>", unsafe_allow_html=True)
    
    # Extract Narrative from current metrics
    curr_metrics, _, _ = get_mission_data()
    narrative = curr_metrics.get('narrative', 'N/A')

    # 1. Primary Narrative Gauge
    if narrative != 'N/A':
        st.markdown(f"""
        <div class='bento-card' style='border-left: 4px solid #00D1FF; background: rgba(0, 209, 255, 0.05);'>
            <div class='grok-label'>Current Dominant Narrative</div>
            <div style='font-size: 18px; font-weight: 700; margin-top: 10px; color: #FFFFFF;'>{narrative}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Awaiting institutional pulse sync...")

    # 2. Narrative Matrix (Heatmap)
    st.markdown("<div class='grok-label'>Narrative Heatmap Matrix</div>", unsafe_allow_html=True)
    logs_sentiment = st.session_state.scanner.get_recent_logs(limit=20)
    if not logs_sentiment.empty and 'Symbol' in logs_sentiment.columns:
        cols = st.columns(2)
        for idx, row in logs_sentiment.iterrows():
            col_idx = idx % 2
            score = row.get('Score', 0) if pd.notna(row.get('Score')) else 0
            symbol = row.get('Symbol', 'UNKNOWN') if pd.notna(row.get('Symbol')) else 'UNKNOWN'
            sentiment_color = "#00FFAB" if score > 15 else ("#FF4B4B" if score < -15 else "rgba(255,255,255,0.4)")
            with cols[col_idx]:
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; margin-bottom: 10px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='font-family: JetBrains Mono; font-weight: 700; color: {sentiment_color};'>{symbol}</span>
                        <span style='font-family: JetBrains Mono; opacity: 0.5;'>SC: {score:+.0f}</span>
                    </div>
                    <div style='font-size: 12px; margin-top: 8px; line-height: 1.4; opacity: 0.8;'>{str(row.get('Narrative', 'N/A')) if pd.notna(row.get('Narrative')) else 'N/A'}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No narrative data in recent scans.")


    # 3. Live News Feed (Aggregator)
    st.markdown("<div class='grok-label' style='margin-top:20px;'>Live Intelligence Stream</div>", unsafe_allow_html=True)
    try:
        from news_syndicate import NewsSyndicate
        syndicate = NewsSyndicate()
        headlines = syndicate.fetch_latest_headlines()
        if headlines:
            for h in headlines[:8]:
                st.markdown(f"""
                <div style='font-size: 11px; padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.03); opacity: 0.6;'>
                    📡 {h['title']} <span style='float: right; opacity: 0.4;'>{h['date'][:16] if h.get('date') else 'N/A'}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Searching for headlines...")
    except:
        st.write("Stream interrupted.")
