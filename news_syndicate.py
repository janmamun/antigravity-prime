
import os
import requests
import xml.etree.ElementTree as ET
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class NewsSyndicate:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            # Use gemini-flash-latest (matches list_models output)
            self.model = genai.GenerativeModel('gemini-flash-latest')
        else:
            self.model = None
            print("⚠️ [SYNDICATE] No Gemini API Key found in .env. News analysis will be mocked.")

        # News Sources (RSS)
        self.sources = [
            "https://cointelegraph.com/rss",
            "https://cryptoslate.com/feed/",
            "https://bitcoinmagazine.com/.rss/full/"
        ]
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_latest_headlines(self):
        """Fetch headlines from primary crypto RSS feeds"""
        headlines = []
        for url in self.sources:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    try:
                        root = ET.fromstring(response.text)
                        for item in root.findall('./channel/item'):
                            title = item.find('title').text
                            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                            headlines.append({
                                "title": title,
                                "date": pub_date
                            })
                    except:
                        # Fallback simple parsing if XML is malformed
                        import re
                        titles = re.findall(r"<title>(.*?)</title>", response.text)
                        for t in titles[1:10]: # Skip feed title
                            headlines.append({"title": t, "date": "Unknown"})
            except Exception as e:
                print(f"⚠️ [SYNDICATE] Failed to fetch from {url}: {e}")
        return headlines[:30]

    def _fallback_analysis(self, symbol, headlines):
        """Simple keyword-based analysis if Gemini fails"""
        bullish_keywords = ["bull", "surge", "approve", "etf", "growth", "buy", "gain", "higher", "record", "moon"]
        bearish_keywords = ["bear", "crash", "reject", "sec", "drop", "sell", "loss", "lower", "hack", "dump"]
        
        score = 0.0
        relevant_headlines = [h['title'].lower() for h in headlines if symbol.lower() in h['title'].lower()]
        if not relevant_headlines:
            relevant_headlines = [h['title'].lower() for h in headlines][:10] # Use general news if no asset-specific ones

        for h in relevant_headlines:
            for b in bullish_keywords:
                if b in h: score += 5.0
            for b in bearish_keywords:
                if b in h: score -= 7.0 # Bearish news often hits harder
        
        score = max(-75, min(75, score)) # Phase 21: Increased range for explosive events
        return score, f"Aggressive Keyword Analysis: {len(relevant_headlines)} units."

    def analyze_asset_sentiment(self, symbol, headlines):
        """Use Gemini to analyze sentiment, fallback to keyword method if needed"""
        if not self.model or not headlines:
            return 0.0, "N/A"

        news_blob = "\n".join([f"- {h['title']}" for h in headlines])
        prompt = f"""
        SOVEREIGN INTELLIGENCE PROTOCOL: DEEP SENTIMENT AGGRESSION
        Asset: '{symbol}'
        
        TASK: Perform high-intensity market impact analysis on the following headlines.
        Identify:
        - "Retail Euphoria/FOMO" (Score +30 to +75)
        - "Institutional Accumulation" (Score +10 to +30)
        - "Regulatory FUD/Fear" (Score -30 to -75)
        - "Liquidity Grab Indicators" (Score -10 to -30)
        
        Return ONLY a JSON object:
        1. 'sentiment_score': float (-100.0 to +100.0). 
           - Use > 60 for "EXPLOSIVE BULLISH"
           - Use < -60 for "CATASTROPHIC BEARISH"
        2. 'narrative': A sharp, 1-sentence tactical summary of the dominant force.
        
        Headlines:
        {news_blob}
        """

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
            import json
            data = json.loads(raw_text)
            return float(data.get('sentiment_score', 0)), data.get('narrative', 'No distinct narrative found.')
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ [SYNDICATE] Quota Hit. Using Keyword Fallback for {symbol}.")
            else:
                print(f"⚠️ [SYNDICATE] Analysis Failed for {symbol}: {e}. Using Keyword Fallback.")
            return self._fallback_analysis(symbol, headlines)

    def get_market_sentiment(self, symbol="BTC"):
        """Main entry point for the Syndicate"""
        headlines = self.fetch_latest_headlines()
        score, narrative = self.analyze_asset_sentiment(symbol, headlines)
        return {
            "symbol": symbol,
            "sentiment_score": score,
            "narrative": narrative,
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    syndicate = NewsSyndicate()
    result = syndicate.get_market_sentiment("BNB")
    print(f"📡 [SYNDICATE RESULT] {result}")
