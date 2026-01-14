
import os
import asyncio
import httpx
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TELEGRAM_BRIDGE")

class TelegramBridge:
    def __init__(self):
        load_dotenv()
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.is_active = bool(self.bot_token and self.chat_id)
        
        if self.is_active:
            logger.info("📡 [TELEGRAM] Bridge Active.")
        else:
            logger.warning("⚠️ [TELEGRAM] Bridge Inactive (Missing Token/ChatID).")

    async def send_message(self, message: str):
        """Send a message to the configured Telegram chat."""
        if not self.is_active:
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"❌ [TELEGRAM] Send Failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Exception: {e}")
            return False

    async def notify_trade(self, symbol, side, qty, price, tp, sl, pnl=None):
        """Format and send trade alerts."""
        emoji = "🚀" if side == "BUY" else "🔻"
        status = "OPENED" if pnl is None else "CLOSED"
        
        msg = f"{emoji} *TRADE {status}: {symbol}*\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"🔹 *Action:* {side}\n"
        msg += f"🔹 *Quantity:* {qty}\n"
        msg += f"🔹 *Price:* ${price:.2f}\n"
        
        if pnl is not None:
            p_emoji = "💰" if pnl > 0 else "🛑"
            msg += f"🔹 *PnL:* {p_emoji} ${pnl:.2f}\n"
        else:
            msg += f"🎯 *TP:* ${tp:.2f}\n"
            msg += f"🛡️ *SL:* ${sl:.2f}\n"
            
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"🕒 {os.popen('date').read().strip()}"
        
        return await self.send_message(msg)

    async def send_heartbeat(self, equity, open_pos_count):
        """Send periodic system health check."""
        msg = f"🔋 *SOVEREIGN HEARTBEAT*\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"💹 *Equity:* ${equity:.2f}\n"
        msg += f"📦 *Active Positions:* {open_pos_count}\n"
        msg += f"✅ *Status:* Engine Synchronized\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"🕒 {os.popen('date').read().strip()}"
        
        return await self.send_message(msg)

if __name__ == "__main__":
    # Test execution
    bridge = TelegramBridge()
    if bridge.is_active:
        asyncio.run(bridge.send_message("🛸 *Sovereign Test Signal Sent*"))
    else:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
