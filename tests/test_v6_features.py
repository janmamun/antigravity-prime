import pandas as pd
import numpy as np
from trading_bot_v17 import UltimateV17Bot
import unittest
from unittest.mock import MagicMock, patch
import os

class TestSovereignV6(unittest.TestCase):
    def setUp(self):
        # Prevent actual API calls during init
        with patch('trading_bot_v17.ccxt.binance'):
            self.bot = UltimateV17Bot()
        self.bot.is_live = True
        self.bot.api_key = "MOCK_KEY"
        self.bot.api_secret = "MOCK_SECRET"
        # Mock exchange and balance
        self.bot.exchange = MagicMock()
        self.bot.market_exch = MagicMock()
        self.bot.get_live_balance = MagicMock(return_value=1000.0) # Larger balance to avoid min-size limit
        self.bot.config['risk_factor_pct'] = 0.1 # 100 USDT per trade

    def test_portfolio_guardian_correlation(self):
        """Verify that size is halved when correlation is high"""
        symbol = "SOL/USDT"
        
        # Mock _calculate_correlation to return 0.95 (HIGH)
        with patch.object(UltimateV17Bot, '_calculate_correlation', return_value=0.95):
            with patch.object(UltimateV17Bot, 'check_safeguards', return_value=True):
                # We need to mock fetch_ohlcv because it's called inside execute_live_order
                self.bot.market_exch.fetch_ohlcv.return_value = [
                    [1625097600000, 100, 105, 95, 102, 1000] for _ in range(50)
                ]
                # Mock exchange methods for the order execution
                self.bot.exchange.fetch_balance.return_value = MagicMock()
                self.bot.exchange.fetch_balance.return_value.get.return_value = MagicMock()
                self.bot.exchange.fetch_balance.return_value.get.return_value.get.return_value = []
                
                self.bot.exchange.amount_to_precision.return_value = "1.0"
                self.bot.exchange.price_to_precision.return_value = "100.0"
                
                # Capture printed output to verify the halving message
                with patch('builtins.print') as mock_print:
                    self.bot.execute_live_order(symbol, "BUY", 100.0, 100.0, 110.0, 90.0)
                    
                    # Verify halving message was printed
                    messages = [str(call.args[0]) for call in mock_print.call_args_list]
                    print("Debug Messages:", messages)
                    self.assertTrue(any("Halving position size" in m for m in messages), "Halving message not found in prints")
                    self.assertTrue(any("Final Allocation: $50.00" in m for m in messages), "Final allocation amount incorrect")

    def test_whale_climax_block_trades(self):
        """Verify block trade detection (>$100k)"""
        symbol = "BTC/USDT"
        # Mock 200 trades, some of which are blocks
        mock_trades = [
            {'amount': 2.0, 'price': 60000, 'side': 'buy'}, # $120k (Block)
            {'amount': 0.1, 'price': 60000, 'side': 'sell'},# $6k (Normal)
            {'amount': 4.0, 'price': 60000, 'side': 'sell'} # $240k (Block) - Clearer bias
        ] * 66 # Total ~200 trades
        
        self.bot.market_exch.fetch_trades.return_value = mock_trades
        
        # Directly call and check for any internal exceptions
        metrics = self.bot._get_trade_based_metrics(symbol)
        print("Debug Metrics:", metrics)
            
        # Expected: 2 blocks * 66 = 132 block trades
        self.assertEqual(metrics['block_trades'], 132)
        # Verify whale bias logic (132k buy vs 264k sell) -> BEARISH (264 > 132 * 1.5 = 198)
        self.assertEqual(metrics['whale_bias'], "BEARISH")

if __name__ == '__main__':
    unittest.main()
