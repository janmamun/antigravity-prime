
# GROK AI HANDOVER PROMPT

**Role**: You are taking over development of "Antigravity Prime" (aka God Mode Bot V12), a high-frequency AI crypto trading bot.

**Current State**:
- **Phase**: 12 (Super-Intelligence) Complete.
- **Status**: Live Verification.
- **Core Goal**: 100% Accuracy, "Unbeatable" performance using Hybrid Logic + ML + Sentiment.

**System Architecture**:

1.  **Core Logic (`trading_bot_v8.py`)**:
    - **Class**: `UltimateV8Bot`.
    - **Strategies**: 10 Weighted Signals including "Magic Numbers" (40/19 MA), Hurst Exponent (Fractal), RSI, MACD, and "Bear Trap" detection.
    - **Updates**: Recently integrated `NeuralOracle` and `SentimentScanner`.

2.  **Super-Intelligence Modules**:
    - **`neural_oracle.py`**: Uses `RandomForestRegressor` (sklearn) to predict next candle close. Confidence score (+10 to V8 score).
    - **`sentiment_scanner.py`**: Technical "Fear & Greed" index based on Volatility/Trend.

3.  **Self-Correction (`meta_brain.py`)**:
    - **Class**: `MetaBrain`.
    - **Function**: Analyzes `sim_state.json`, calculates Win Rate.
    - **Action**: Dynamically updates `bot_config.json` (e.g., tightens `min_score` if losing, adjusts Risk Factor).

4.  **Execution (`simulation_engine.py`)**:
    - **Class**: `FuturesSimulator` (renamed ExecutionEngine logic).
    - **Modes**: PAPER (Simulated), LIVE (Binance Futures via `ccxt`).
    - **State**: Maintains local state for speed, reconciles with API in LIVE mode.

5.  **Interface (`dashboard_antigravity.py`)**:
    - **Tech**: Streamlit.
    - **Style**: "Glassmorphism" / Apple-Dark Aesthetic.
    - **Features**: Real-time TradingView chart, "Market Matrix" signal overlay, Intelligence Log, manual overrides.

**Key Files**:
- `/Users/mamunjan/.gemini/antigravity/scratch/new_workspace/trading_bot_v8.py` (Brain)
- `/Users/mamunjan/.gemini/antigravity/scratch/new_workspace/dashboard_antigravity.py` (UI)
- `/Users/mamunjan/.gemini/antigravity/scratch/new_workspace/market_scanner.py` (Scanner loop)
- `/Users/mamunjan/.gemini/antigravity/scratch/new_workspace/bot_config.json` (Dynamic Config)

**Next Objectives (For Grok)**:
1.  **Paper Trading Verification**: We are currently running a $500 challenge. Monitor performance.
2.  **Optimization**: If trades are scarce, check `min_score` in `bot_config.json`. If losing, check `MetaBrain` logic.
3.  **Expansion**: Consider adding "Regime Detection" (Range vs Trend) as a primary switch for strategy selection.

**Context**: The user wants "Real Results" and "Profit". The system is designed to be autonomous ("The Singularity"). Keep the loop running and the logs clean.
