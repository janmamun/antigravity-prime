
import google.generativeai as genai
import os

class LLMBrain:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if self.api_key:
            genai.configure(api_key=self.api_key)
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except:
                self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None

    def analyze_market(self, market_context):
        """
        Sends market data to Gemini and gets a 'Billionaire Trader' persona response.
        """
        if not self.model:
            return "⚠️ AI DISCONNECTED: Enter API Key in Sidebar"

        prompt = f"""
        You are a Billionaire Crypto Hedge Fund Manager (The "Magician").
        You are analyzing the current market setup.
        
        CONTEXT:
        {market_context}
        
        TASK:
        Provide a short, punchy, high-confidence commentary (max 3 sentences).
        Be aggressive but smart. Use terms like "Liquidity Injection", "Alpha", "Whale Movement".
        Decide if we should HOLD, ACCUMULATE, or DUMP.
        
        STYLE:
        "Wolf of Wall Street" meets "Quantum Physicist".
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ AI ERROR: {str(e)}"
