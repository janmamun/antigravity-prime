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
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --grok-bg: #000000;
        --grok-primary: #FFFFFF;
        --grok-accent: #00D1FF;
        --grok-zinc: rgba(255, 255, 255, 0.05);
    }

    /* QUANTUM TRANSPARENCY PATCH */
    html, body, [data-testid="stAppViewContainer"], .stApp, .stMain, .stHeader, [data-testid="stHeader"], div.block-container {
        background: transparent !important;
        background-color: transparent !important;
        font-family: 'Inter', sans-serif;
        color: var(--grok-primary);
        overflow-x: hidden;
    }
    
    .stDecoration { display: none !important; }

    /* MISSION CANVAS & BACKGROUND INFUSION */
    canvas#sovereign-viewport {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -10;
        pointer-events: none;
        background: radial-gradient(circle at 50% 50%, #0a0c12 0%, #000000 100%);
    }

    /* BENTO ARCHITECTURE (NORMALIZED) */
    .bento-card {
        background: rgba(255, 255, 255, 0.012);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px; /* Professional tighter radius */
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        /* Fail-safe visibility */
        animation: fadeIn 1s forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .bento-card:hover {
        background: rgba(255, 255, 255, 0.02);
        border-color: rgba(0, 209, 255, 0.2);
    }

    /* MISSION HEADER (DENSE) */
    .mission-header {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px 32px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* TYPOGRAPHY (HI-RES) */
    .grok-label { 
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3em;
        color: rgba(255,255,255,0.3);
        margin-bottom: 4px;
    }
    .grok-value-large {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: -0.05em;
        color: var(--grok-primary);
    }
    .grok-value-mid {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--grok-primary);
    }
    .pill-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 4px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        display: inline-block;
    }
    
    .news-streamer {
        background: rgba(0, 10, 20, 0.4);
        border: 1px solid var(--grok-zinc);
        border-radius: 12px;
        padding: 15px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: var(--grok-accent);
        height: 100px;
        overflow-y: auto;
        margin-top: 20px;
    }
    .confidence-meter {
        height: 6px;
        background: var(--grok-zinc);
        border-radius: 3px;
        margin-top: 10px;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF00FF, #00D1FF);
        border-radius: 3px;
        transition: width 0.5s ease-out;
    }

    /* MISSION CORE GRID */
    [data-testid="column"] {
        padding: 0 10px !important;
    }
    
    .stAppViewMain {
        padding-top: 0 !important;
    }

    /* DATA GRIDS */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        overflow: hidden;
    }
    
    /* TABS & BUTTONS (PROFESSIONAL) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        color: rgba(255,255,255,0.4) !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.2em;
        padding: 12px 0px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--grok-accent) !important;
        border-bottom: 2px solid var(--grok-accent) !important;
    }

    /* LOG STREAMER (STABILIZED) */
    .streamer {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        line-height: 1.5;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 12px;
        color: rgba(255, 255, 255, 0.5);
        display: block;
        width: 100%;
        box-sizing: border-box;
    }
    .streamer pre {
        margin: 0;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    button[kind="secondary"], button[kind="primary"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover, button[kind="primary"]:hover {
        background: rgba(0, 209, 255, 0.1) !important;
        border-color: var(--grok-accent) !important;
        box-shadow: 0 0 15px rgba(0, 209, 255, 0.2);
    }
    
    .status-light {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 8px currentColor;
    }
    .status-active { color: #00FF94; background: #00FF94; }
    .status-warning { color: #FFB800; background: #FFB800; }
    .status-error { color: #FF4B4B; background: #FF4B4B; }

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
        gsap.from(".grok-value", {
            opacity: 0,
            y: 40,
            duration: 2,
            delay: 0.5,
            ease: "power4.out"
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
        
        # Get portfolio status from simulator
        try:
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
                    formatted_live = []
                    for lp in live_pos:
                        notional = abs(lp.get('size', 0) * lp.get('entry', 0))
                        lev = lp.get('leverage', 1) or 1
                        margin = notional / lev if lev > 0 else 1
                        roi = (lp.get('unrealized_pnl', 0) / margin) * 100 if margin > 0 else 0
                        
                        formatted_live.append({
                            'Symbol': lp.get('symbol', 'N/A'),
                            'Mode': '🔥 LIVE',
                            'Side': lp.get('side', 'N/A'),
                            'Size': abs(lp.get('size', 0)),
                            'Entry': lp.get('entry', 0),
                            'Mark': lp.get('mark_price', 0),
                            'PnL ($)': lp.get('unrealized_pnl', 0),
                            'PnL (%)': roi
                        })
                    
                    metrics['positions'] = formatted_live + metrics.get('positions', [])
                    metrics['unrealized_pnl'] = sum([p['PnL ($)'] for p in formatted_live]) + metrics.get('unrealized_pnl', 0)
        except Exception as e:
            print(f"Shadow Sync Error: {e}")
        
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
            stats = st.session_state.sim.calculate_stats()
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
    sim_stats = default_boot_stats.copy()
    current_prices = {}
    st.session_state.booted = True
else:
    # Subsequent runs: Fetch data with full resilience
    try:
        metrics, sim_stats, current_prices = get_mission_data()
    except Exception as e:
        print(f"Dashboard data fetch failed: {e}")
        metrics = default_boot_metrics.copy()
        sim_stats = default_boot_stats.copy()
        current_prices = {}

# Ensure metrics is never None/empty - UI must always render
if not metrics:
    metrics = default_boot_metrics.copy()
if not sim_stats:
    sim_stats = default_boot_stats.copy()

# Prepare sidebar variables
import requests
hb = get_heartbeat()
hb_age = time.time() - hb.get('last_heartbeat', 0)
mission_status = "ACTIVE" if hb_age < 300 else "STALLED"
last_pulse = hb_age 

# -----------------------------------------------------------------------------
# 5. SIDEBAR: CONTROL CENTER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
        <div style="text-align:center; padding: 20px 0; margin-bottom: 30px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div style="font-size: 48px; filter: drop-shadow(0 0 10px rgba(0, 209, 255, 0.3));">🦅</div>
            <div class="grok-label" style="opacity: 0.8; font-size: 14px; margin-top: 15px;">SOVEREIGN V6.0</div>
            <div class='pill-tag' style='color:var(--grok-accent); border-color:var(--grok-accent); margin-top:10px; background: rgba(0, 209, 255, 0.05);'>{mission_status}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='grok-label' style='margin-left:5px;'>Connectivity Matrix</div>", unsafe_allow_html=True)
    
    # Check Brain Connection
    try:
        brain_check = requests.get("http://127.0.0.1:11434/api/tags", timeout=1).status_code == 200
        brain_status = "status-active" if (st.session_state.brain.gemini_model or brain_check) else "status-error"
    except:
        brain_status = "status-active" if st.session_state.brain.gemini_model else "status-error"
        
    # Evolution Sentinel Check
    sentinel_active = False
    if os.path.exists("neural_bridge.log"):
        mtime = os.path.getmtime("neural_bridge.log")
        if time.time() - mtime < 3700: sentinel_active = True
        
    sentinel_status = "status-active" if sentinel_active else "status-error"
    exchange_status = "status-active" if st.session_state.scanner.bot.is_live else "status-warning"
    pulse_status = "status-active" if last_pulse < 300 else "status-error"

    # Watchdog Check (Check watchdog.log instead of neural_bridge)
    watchdog_active = False
    if os.path.exists("watchdog.log"):
        mtime = os.path.getmtime("watchdog.log")
        if time.time() - mtime < 300: watchdog_active = True
    watchdog_status = "status-active" if watchdog_active else "status-error"

    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; margin-bottom: 20px;">
            <div style="margin-bottom: 10px;"><span class="status-light {pulse_status}"></span> <span style="font-size:11px; opacity:0.7; font-family:'JetBrains Mono';">System Pulse</span></div>
            <div style="margin-bottom: 10px;"><span class="status-light {brain_status}"></span> <span style="font-size:11px; opacity:0.7; font-family:'JetBrains Mono';">Neural Core (AI)</span></div>
            <div style="margin-bottom: 10px;"><span class="status-light {sentinel_status}"></span> <span style="font-size:11px; opacity:0.7; font-family:'JetBrains Mono';">Evolution Sentinel</span></div>
            <div style="margin-bottom: 10px;"><span class="status-light {watchdog_status}"></span> <span style="font-size:11px; opacity:0.7; font-family:'JetBrains Mono';">Perpetual Watchdog</span></div>
            <div style="margin-bottom: 10px;"><span class="status-light {exchange_status}"></span> <span style="font-size:11px; opacity:0.7; font-family:'JetBrains Mono';">Binance Sync</span></div>
            <div style="margin-bottom: 10px;"><span class="status-light {'status-active' if st.session_state.scanner.bot.is_live else 'status-warning'}"></span> <span style="font-size:11px; opacity:0.7; font-family:'JetBrains Mono';">Liquidation Guard</span></div>
            <div style="margin-bottom: 10px;"><span class="status-light {'status-warning' if st.session_state.scanner.bot.hedge_active else 'status-active'}"></span> <span style="font-size:11px; opacity:0.7; font-family:'JetBrains Mono';">Crash Shield</span></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='grok-label' style='margin-top:20px;'>Risk & Allocation</div>", unsafe_allow_html=True)
    risk = st.select_slider("ACCOUNT RISK (%)", options=[1, 2, 5, 10, 15, 20, 25], value=10)
    lev = st.select_slider("LEVERAGE MULTIPLIER", options=[1, 3, 5, 10, 20], value=5)
    
    # Phase 31: Persist slider values to bot_config
    try:
        config_path = "bot_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                dash_config = json.load(f)
            if dash_config.get("risk_factor_pct") != risk/100 or dash_config.get("leverage") != lev:
                dash_config["risk_factor_pct"] = risk/100
                dash_config["leverage"] = lev
                with open(config_path, "w") as f:
                    json.dump(dash_config, f, indent=4)
                # No rerun here to avoid flicker, the next refresh will pick it up
    except:
        pass
    
    if st.button("TRIGGER GLOBAL SCAN", width="stretch"):
        with st.spinner("SCANNING..."):
            import asyncio
            asyncio.run(st.session_state.scanner.scan_market())
            st.rerun()

# PROFESSIONAL MISSION HEADER (DENSE)
total_net = sim_stats.get('total_pnl', 0) + metrics.get('unrealized_pnl', 0)
is_live = st.session_state.scanner.bot.is_live

# Phase 29: Timeout-Protective Balance Fetch
live_bal = metrics['equity'] # Default Fallback
if is_live:
    try:
        # Give it a very short window to avoid hanging the UI
        live_bal = st.session_state.scanner.bot.get_live_balance()
    except:
        pass

pnl_color = "var(--grok-accent)" if total_net >= 0 else "#FF00FF"

# Compounding Target Calculation
target_goal = 200.00 # Target to hit Phase 26 milestone
initial_capital = st.session_state.scanner.bot.session_start_equity if hasattr(st.session_state.scanner.bot, 'session_start_equity') else 111.09
compounding_pct = min(100, max(0, ((live_bal - 100) / (target_goal - 100)) * 100))
session_pnl = live_bal - initial_capital

st.markdown(f"""
<div class="mission-header" style="margin-top: -30px;">
    <div style="flex: 1.5;">
        <div class="grok-label">Capital Core</div>
        <div style="display:flex; align-items:baseline; gap:12px;">
            <div class="grok-value-mid" style="font-size:32px;">${live_bal:,.2f}</div>
            <div class='pill-tag' style='background:rgba(255,255,255,0.02); vertical-align:middle;'>{ 'LIVE' if is_live else 'SIM' }</div>
        </div>
        <div class="confidence-meter" style="width: 200px; height: 4px; margin-top: 5px;">
            <div class="confidence-fill" style="width: {compounding_pct}%; background: linear-gradient(90deg, #00D1FF, #00FF94);"></div>
        </div>
        <div class="grok-label" style="font-size: 8px; margin-top: 4px;">TARGET: $200.00 ({compounding_pct:.1f}%)</div>
    </div>
    <div style="flex: 1; text-align: center; border-left: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05);">
        <div class="grok-label">Session PnL</div>
        <div class="grok-value-mid" style="color:{ '#00FF94' if session_pnl >= 0 else '#FF00FF' }; font-size:24px;">{session_pnl:+,.2f} USD</div>
    </div>
    <div style="flex: 1; text-align: center; border-right: 1px solid rgba(255,255,255,0.05);">
        <div class="grok-label">Macro Sentience</div>
        <div class="grok-value-mid" style="color:{ '#00FF94' if metrics.get('macro_bias') == 'RISK-ON' else '#FF00FF' if metrics.get('macro_bias') == 'RISK-OFF' else 'var(--grok-accent)' }; font-size:24px;">{metrics.get('macro_bias', 'NEUTRAL')}</div>
    </div>
    <div style="flex: 0.7; text-align: right;">
        <div class="grok-label">Success Density</div>
        <div class="grok-value-mid" style="color:var(--grok-accent); font-size:24px;">{sim_stats['win_rate']:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# INTELLIGENCE HUB (DENSE)
ic1, ic2, ic3, ic4 = st.columns(4)
conf_score = metrics.get('score', 0)
conf_pct = min(max(abs(conf_score), 0), 100)

with ic1: st.markdown(f"<div class='bento-card' style='padding:20px; height:120px;'><div class='grok-label'>Conviction</div><div class='grok-value-mid' style='color:var(--grok-accent); font-size:20px;'>{conf_score:+d}</div><div class='confidence-meter' style='margin-top:8px;'><div class='confidence-fill' style='width: {conf_pct}%'></div></div></div>", unsafe_allow_html=True)
with ic2: st.markdown(f"<div class='bento-card' style='padding:20px; height:120px;'><div class='grok-label'>Regime</div><div class='grok-value-mid' style='font-size:18px;'>{metrics.get('regime', 'NEUTRAL')}</div></div>", unsafe_allow_html=True)
with ic3: st.markdown(f"<div class='bento-card' style='padding:20px; height:120px;'><div class='grok-label'>The Hype Meter</div><div class='grok-value-mid' style='font-size:18px; color:{'#00FF94' if metrics.get('sentiment') == 'HYPED' else ('#FF00FF' if metrics.get('sentiment') == 'FUD' else 'white')}'>{metrics.get('sentiment', 'NEUTRAL')}</div></div>", unsafe_allow_html=True)
with ic4:
    squeeze_status = "ACTIVE" if metrics.get('is_squeezing') else "STANDBY"
    st.markdown(f"<div class='bento-card' style='padding:20px; height:120px;'><div class='grok-label'>Squeeze Tracker</div><div class='grok-value-mid' style='font-size:18px; color:{'#00D1FF' if metrics.get('is_squeezing') else 'rgba(255,255,255,0.3)'}'>{squeeze_status}</div></div>", unsafe_allow_html=True)

# Phase 15: TIDAL PULSE BAR (Institutional Flow)
st.markdown(f"""
<div style='margin-bottom: 20px; background: rgba(0,0,0,0.2); border: 1px solid rgba(0, 209, 255, 0.1); border-radius: 12px; padding: 15px;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div class='grok-label' style='font-size: 12px;'>INSTITUTIONAL TIDAL SHIFT (BLOCK TRADES ROC)</div>
        <div class='pill-tag' style='color: {"#00FF94" if metrics.get("turbo_active") else "var(--grok-accent)"};'>{ "TURBO BOOSTED" if metrics.get("turbo_active") else "NOMINAL FLOW" }</div>
    </div>
    <div style='background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; margin-top: 10px; overflow: hidden;'>
        <div style='background: linear-gradient(90deg, #00D1FF, #00FF94); width: {min(100, int(metrics.get("tidal_shift", 0) * 100))}%; height: 100%; transition: width 0.8s ease;'></div>
    </div>
</div>
""", unsafe_allow_html=True)

# BLOOMBERG WORKSPACE (STRICT 70/30 SPLIT)
col_l, col_r = st.columns([7, 3]) 
with col_l:
    st.markdown("<div class='grok-label'>Intelligence Core Visualization</div>", unsafe_allow_html=True)
    active_sym = metrics['positions'][0]['Symbol'] if metrics['positions'] else "BTC/USDT"
    render_tradingview(active_sym)

with col_r:
    st.markdown("<div class='grok-label'>Sovereign Neural Stream</div>", unsafe_allow_html=True)
    all_logs = get_live_logs(lines=100)
    alpha_alerts = [line for line in all_logs if "[ALPHA_ALERT]" in line][-5:]
    
    if alpha_alerts:
        st.markdown("<div class='grok-label' style='color:#00FF94; font-size:10px;'>⚡ ALPHA ALERT TERMINAL</div>", unsafe_allow_html=True)
        alert_box = "".join([f"<div style='color:#00FF94; font-family:\"JetBrains Mono\"; font-size:10px; margin-bottom:5px; border-left: 2px solid #00FF94; padding-left:10px;'>{a}</div>" for a in alpha_alerts])
        st.markdown(f"<div style='background:rgba(0,255,148,0.05); padding:10px; border-radius:8px; margin-bottom:10px;'>{alert_box}</div>", unsafe_allow_html=True)

    log_content = "".join(all_logs[-25:])
    st.markdown(f"""
        <div class='streamer' style='height:360px;'>
            <pre>{log_content}</pre>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='grok-label' style='margin-top:20px;'>Mission Status Update</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='news-streamer' style='height:100px;'>
        [SYSTEM] Intelligence Hub stabilized. <br>
        [INTEL] Multi-matrix synchronization active. <br>
        [PULSE] Latency: {int(last_pulse)}ms
    </div>
    <div style='margin-top:20px;'>
        <div class='grok-label' style='font-size:10px; opacity:0.5; margin-bottom:5px;'>LIQUIDITY GAP (SPREAD)</div>
        <div style='background: rgba(255,255,255,0.05); border-radius: 4px; height: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);'>
            <div style='background: linear-gradient(90deg, #00FF94, #FFB800, #FF00FF); width: {min(100, int(metrics.get("spread_pct", 0) * 1000))}%; height: 100%; transition: width 0.5s ease;'></div>
        </div>
        <div style='display: flex; justify-content: space-between; font-family: "JetBrains Mono"; font-size: 9px; opacity: 0.4; margin-top: 4px;'>
            <span>SHARP</span>
            <span>GAP</span>
        </div>
    </div>
    <div style='margin-top:20px;'>
        <div class='grok-label' style='font-size:10px; opacity:0.5; margin-bottom:5px;'>V11 PROFIT LADDER (ACTIVE)</div>
        <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;'>
            <div style='font-size: 10px; opacity: 0.8; margin-bottom: 5px; color: #00FF94;'>[T1] 0.8% ATR Target (30% Scale-Out)</div>
            <div style='font-size: 10px; opacity: 0.8; margin-bottom: 5px; color: #00D1FF;'>[T2] 1.5% ATR Target (40% Scale-Out)</div>
            <div style='font-size: 10px; opacity: 0.8; color: #FF00FF;'>[T3] 3.0% ATR Target (30% Runner)</div>
        </div>
    </div>
    <div style='margin-top:20px;'>
        <div class='grok-label' style='font-size:10px; opacity:0.5; margin-bottom:5px;'>WALL PERSISTENCE (STABILITY)</div>
        <div style='background: rgba(255,255,255,0.05); border-radius: 4px; height: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);'>
            <div style='background: linear-gradient(90deg, #FF00FF, #00D1FF); width: {min(100, int(len(metrics.get("persistence", {}).get(active_sym, {}).get("times", [])) * 10)) if "active_sym" in locals() else 50}%; height: 100%; transition: width 0.5s ease;'></div>
        </div>
        <div style='display: flex; justify-content: space-between; font-family: "JetBrains Mono"; font-size: 9px; opacity: 0.4; margin-top: 4px;'>
            <span>FLICKER</span>
            <span>STABLE</span>
        </div>
    </div>
    <div style='margin-top:20px;'>
        <div class='grok-label' style='font-size:10px; opacity:0.5; margin-bottom:5px;'>V15 LIQUIDITY VACUUM MAP</div>
        <div style='background: rgba(255,255,255,0.05); border-radius: 4px; height: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);'>
            <div style='background: linear-gradient(90deg, #FFB800, #FF00FF); width: {metrics.get("vacuum_score", 10)}%; height: 100%; transition: width 0.5s ease;'></div>
        </div>
        <div style='display: flex; justify-content: space-between; font-family: "JetBrains Mono"; font-size: 9px; opacity: 0.4; margin-top: 4px;'>
            <span>DENSE</span>
            <span>VACUUM</span>
        </div>
    </div>
    <div style='margin-top:20px;'>
        <div class='grok-label' style='font-size:10px; opacity:0.5; margin-bottom:5px;'>INSTITUTIONAL ALPHA DEPTH</div>
        <div style='background: rgba(255,255,255,0.05); border-radius: 4px; height: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);'>
            <div style='background: linear-gradient(90deg, #00D1FF, #00FF94); width: {min(100, int(st.session_state.scanner.bot._get_liquidity_depth(active_sym)["depth"] * 33)) if "active_sym" in locals() else 50}%; height: 100%; transition: width 0.5s ease;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# TABS: SOVEREIGN GRID (PERSISTENT)
st.markdown("<br>", unsafe_allow_html=True)
tab_titles = ["EXPOSURE", "AUDIT", "HISTORY", "SCANNER", "NEURAL", "SYNDICATE"]
t1, t2, t3, t4, t5, t6 = st.tabs(tab_titles)

# Streamlit tabs don't have a direct 'index' sync, but we can simulate it or just let the user navigate.
# To ensure the chat doesn't feel like it "disappeared", we'll check if a message was just sent.
if 'last_action' in st.session_state and st.session_state.last_action == 'chat':
    # This is a hack because st.tabs doesn't let us programmatically switch easily in this version
    # But we can at least show a notification.
    st.toast("Neural Transmission Received", icon="🦅")
    del st.session_state['last_action']

with t1:
    if metrics['positions']:
        st.dataframe(pd.DataFrame(metrics['positions']), width="stretch")
    else:
        st.markdown("<div style='border:1px solid var(--grok-zinc); border-radius:20px; padding:80px; text-align:center; color:rgba(255,255,255,0.2)'>NO ACTIVE EXPOSURE</div>", unsafe_allow_html=True)

with t2:
    st.markdown("<div class='grok-label'>Granular Trade Attribution (God-Mode)</div>", unsafe_allow_html=True)
    
    col_audit1, col_audit2 = st.columns([7, 3])
    with col_audit1:
        if os.path.exists("sovereign_audit.json"):
            with open("sovereign_audit.json", "r") as f:
                audits = json.load(f)
            df_audit = pd.DataFrame(audits).sort_index(ascending=False)
            st.dataframe(df_audit.style.applymap(lambda x: 'color: #00D1FF' if x == 'BUY' else 'color: #FF00FF', subset=['side']), width="stretch")
            if not df_audit.empty:
                st.markdown(f"<div class='pill-tag'>Total Executions: {len(df_audit)}</div>", unsafe_allow_html=True)
        else:
            st.info("Awaiting first Sovereign Audit record...")

    with col_audit2:
        st.markdown("<div class='grok-label'>Toxic Assets (Blacklist)</div>", unsafe_allow_html=True)
        if os.path.exists("blacklist.json"):
            with open("blacklist.json", "r") as f:
                blacklist = json.load(f)
            if blacklist:
                for item in blacklist:
                    st.markdown(f"<div class='pill-tag' style='color:#FF4B4B; border-color:#FF4B4B; margin-bottom:5px; width:100%; text-align:center;'>💀 {item}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='opacity:0.3; font-size:10px;'>Registry Empty</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='opacity:0.3; font-size:10px;'>Blacklist Inactive</div>", unsafe_allow_html=True)

with t3:
    st.markdown("<div class='grok-label'>Historical Mission Playback (Smart Backtest)</div>", unsafe_allow_html=True)
    sim_history = sim_stats.get('history', [])
    if sim_history:
        h_df = pd.DataFrame(sim_history).sort_values('Time', ascending=False)
        st.markdown("<div style='margin-bottom:20px;'>", unsafe_allow_html=True)
        st.dataframe(h_df.style.applymap(lambda x: 'color: #00FF41' if x > 0 else 'color: #FF00FF', subset=['PnL']), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Hindsight Insights
        if os.path.exists("hindsight_data.json"):
            with open("hindsight_data.json", "r") as f:
                hindsight = json.load(f)
            st.markdown("<div class='grok-label'>Sovereign Predictions Registry</div>", unsafe_allow_html=True)
            
            # Display recent logs with expanders and block trade detection
            recent_logs = st.session_state.scanner.get_recent_logs(limit=30) # Assuming scanner logs are relevant here
            if not recent_logs.empty:
                for _, row in recent_logs.iterrows():
                    with st.expander(f"{row['Symbol']} - {row['Signal']} (Score: {row['Score']})"):
                        st.write(f"**Regime:** {row['Regime']}")
                        
                        # Highlight Block Trades
                        whale_bias = row.get('Whale_Bias', 'NEUTRAL')
                        if whale_bias != 'NEUTRAL':
                            st.warning(f"🚨 WHALE ACTIVITY: {whale_bias} BIAS DETECTED")
                        
                        st.write(f"**Reasons:** {row['Reasons']}")
                        st.write(f"**Funding:** {row['Funding']} | **OI Momentum:** {row['OI_Mom']}")
            else:
                st.info("No recent scanner logs for predictions registry.")
    else:
        st.info("Hindsight Matrix is empty.")

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
