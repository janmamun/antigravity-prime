import os
import pandas as pd
import requests
import json
from datetime import datetime
import traceback
from dotenv import load_dotenv
load_dotenv()

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class MetaBrain:
    def __init__(self):
        self.config_file = "bot_config.json"
        self.state_file = "sim_state.json"
        self.default_config = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "magic_sensitivity": 1.0,
            "min_score": 60,
            "risk_factor": "NORMAL"
        }
        self.knowledge_file = "sovereign_knowledge.json"
        if not os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'w') as f:
                json.dump({"patterns": [], "insights": []}, f)

    def load_history(self):
        if not os.path.exists(self.state_file):
            return []
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                history = data.get("history", [])
                # Ensure each entry is a dict and has PnL
                return [h for h in history if isinstance(h, dict) and 'PnL' in h]
        except:
            return []

    def get_win_rate(self):
        history = self.load_history()
        if not history: return 0.0
        df = pd.DataFrame(history)
        wins = df[df['PnL'] > 0]
        return round(len(wins) / len(df) * 100, 2)

    def load_config(self):
        if not os.path.exists(self.config_file):
            return self.default_config
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def save_config(self, config):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def run_self_audit(self):
        """
        Analyze performance and adjust parameters.
        The Singularity Loop.
        """
        history = self.load_history()
        config = self.load_config()
        
        if len(history) < 5:
            return "Gathering Data..." # Need simulation data
            
        # Calculate Win Rate
        df = pd.DataFrame(history)
        if df.empty: return "No Data"
        
        wins = df[df['PnL'] > 0]
        win_rate = len(wins) / len(df) * 100
        
        # LOGIC: SELF CORRECTION
        adjustment = "STABLE"
        
        if win_rate < 50:
            # CRISIS: Tighten up
            config['min_score'] = min(85, config.get('min_score', 60) + 5)
            config['magic_sensitivity'] = 0.8 # Strict validation
            config['risk_factor'] = "DEFENSIVE"
            adjustment = "TIGHTENED (Low Win Rate)"
            
        elif win_rate > 80:
            # GOD MODE: Scale up / Loose
            config['min_score'] = max(50, config.get('min_score', 60) - 2)
            config['magic_sensitivity'] = 1.2 # Allow more wild trades
            config['risk_factor'] = "AGGRESSIVE"
            adjustment = "EXPANDED (High Win Rate)"
            
        else:
            # Normalization
            config['min_score'] = 60
            config['risk_factor'] = "NORMAL"
            
        self.save_config(config)
        self.update_knowledge(history)
        return f"Win Rate: {win_rate:.1f}% | Action: {adjustment}"

    def update_knowledge(self, history):
        """Phase 21: Extract successful patterns and store them as knowledge"""
        if len(history) < 10: return
        
        try:
            with open(self.knowledge_file, 'r') as f:
                knowledge = json.load(f)
            
            df = pd.DataFrame(history)
            successful_trades = df[df['PnL'] > 0]
            
            # Simple pattern extraction (Symbol performance)
            top_syms = successful_trades['Symbol'].value_counts().head(3).index.tolist()
            
            knowledge['insights'].append({
                "timestamp": datetime.now().isoformat(),
                "top_performing_assets": top_syms,
                "win_rate": len(successful_trades) / len(df) * 100
            })
            
            # Keep last 100 insights
            knowledge['insights'] = knowledge['insights'][-100:]
            
            with open(self.knowledge_file, 'w') as f:
                json.dump(knowledge, f, indent=4)
        except:
            pass

class LocalBrain:
    def __init__(self, model="qwen2.5-coder:7b", chat_model="llama3.2:1b"):
        self.url = "http://127.0.0.1:11434/api/generate"
        self.model = model
        self.chat_model = chat_model
        
        # Phase 24: Cloud Support (Gemini)
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if self.api_key and HAS_GEMINI:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel("gemini-flash-latest")
        else:
            self.gemini_model = None

    MISSION_MEMORY = """
Phase 15: Established Binance Futures connectivity, Paper Trading Mode, and ADX Regime Detection.
Phase 16: Integrated Institutional Intelligence (OI Momentum, Whale Flow) and real-time scanner metrics.
Phase 17: Sovereign Intelligence - Guardian Protocols (Self-Healing), Hindsight Learning, and Streamer UI.
Phase 18: Neural Bridge - Interactive AI communication and dual-matrix visual analysis.
"""

    def chat(self, user_msg, context=""):
        """Phase 18/25: High-Intelligence Neural Bridge with Data Context"""
        prompt = f"""
System: You are the Sovereign Intelligence of Antigravity Prime, an elite algorithmic trading system.
Your mission is to provide institutional-grade analysis and technical support.
Be concise, professional, and slightly futuristic. ALWAYS use the provided context to justify your answers.

MISSION MEMORY: {self.MISSION_MEMORY}
CURRENT PORTFOLIO MATRIX: {context}

Analytical Directive: 
- If user asks about performance, cite the Equity and Win Rate from the Matrix.
- If user asks for market views, synthesize current trends with the Portfolio context.
- If data is missing, recommend checking the 'Sovereign Audit' tab.

User Transmission: {user_msg}

Sovereign:"""
        
        # 1. Try Gemini (Cloud Native)
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                if response.text:
                    return response.text
            except Exception as e:
                print(f"Gemini Error: {e}")

        # 2. Try Ollama (Local)
        try:
            payload = {"model": self.chat_model, "prompt": prompt, "stream": False}
            response = requests.post(self.url, json=payload, timeout=5) # Tight timeout for web
            if response.status_code == 200:
                return response.json()["response"]
        except:
            pass
            
        # 3. Sovereign Proxy (Mock Response for offline/cloud without keys)
        return self._sovereign_proxy(user_msg, context)

    def _sovereign_proxy(self, msg, context):
        """Deterministically generate a professional response if AI is offline"""
        msg = msg.lower()
        if "status" in msg or "pnl" in msg:
            return f"Sovereign Intelligence is currently in Standard Mode. Context received: {context}. Performance appears stable. Please check the 'Sovereign Audit' tab for granular verification."
        if "btc" in msg or "bitcoin" in msg:
            return "Local AI Bridge is offline. Standard Intelligence indicates BTC is following institutional flow. Consult the TradingView Matrix for precise delta."
        return "Neural Bridge connection is restricted. I have received your transmission but am operating on backup protocol. Add a GEMINI_API_KEY to your env/secrets to restore full Cognitive Link."

    def evolve(self, performance_summary):
        prompt = f"""
You are the evolution engine for Antigravity Prime — an elite autonomous futures trading bot.

Current Performance:
{performance_summary}

Market: Extreme Fear, BTC ranging $86-88k.

Your mission: Make this bot compound to millions with zero human help.

Tasks:
1. Diagnose any issues (low win rate, drawdown, bad entries, bugs)
2. Suggest immediate fixes (raise/lower min_score, adjust risk, disable signal)
3. Propose next evolution (new filter, DCA rule, funding arb, etc.)
4. Indicator Sensitivity: Output a JSON block for "indicator_patches" (e.g., {{"rsi_oversold": 25, "ma_period": 21}}) based on current volatility.
5. If code change needed: Output clean, ready-to-apply Python patch.
6. Predict next 24h market behavior and optimal stance.

Respond clearly and actionably.
"""
        # 1. Try Gemini first if available (Phase 24)
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                if response.text:
                    advice = response.text
                    with open("brain_log.txt", "a") as f:
                        f.write(f"\n--- [GEMINI] {datetime.now()} ---\n{advice}\n")
                    return advice
            except Exception as e:
                print(f"Gemini Evolution Error: {e}")

        # 2. Try Ollama (Local)
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(self.url, json=payload, timeout=300)
            if response.status_code == 200:
                advice = response.json()["response"]
                with open("brain_log.txt", "a") as f:
                    f.write(f"\n--- [OLLAMA] {datetime.now()} ---\n{advice}\n")
                return advice
            else:
                return "Brain offline"
        except:
            return "Brain connection failed — retrying next cycle"
