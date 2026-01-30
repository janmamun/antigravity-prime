import sys
import unittest
from unittest.mock import MagicMock, patch
import time
import json
from trading_bot_v17 import UltimateV17Bot

class TestZeroTrustExecution(unittest.TestCase):
    def setUp(self):
        self.bot = UltimateV17Bot()
        self.bot.exchange = MagicMock()
        self.bot.proxy_mgr = MagicMock()
        self.bot.proxy_mgr.get_proxy.return_value = None
        
        # Mock fetch_balance for position count
        self.bot.exchange.fetch_balance.return_value = {
            'info': {'positions': []},
            'free': {'USDT': 1000}
        }
        
    def test_successful_execution_and_verification(self):
        """Test that a trade is kept open if TP/SL are verified."""
        print("\n--- Testing Successful Execution & Verification ---")
        symbol = "BTC/USDT:USDT"
        
        # Mock main order success
        self.bot.exchange.create_order.side_effect = [
            {'id': 'main_order_123'}, # Main order
            {'id': 'sl_order_123'},   # SL order
            {'id': 'tp_order_123'}    # TP order
        ]
        
        # Mock verification success
        self.bot.exchange.fetch_open_orders.return_value = [
            {'type': 'STOP_MARKET', 'id': 'sl_order_123'},
            {'type': 'TAKE_PROFIT_MARKET', 'id': 'tp_order_123'}
        ]
        
        res = self.bot.execute_live_order(symbol, "BUY", 100, 50000, 55000, 48000)
        
        self.assertEqual(res['status'], 'SUCCESS')
        # Check that main order, SL, and TP were attempted
        self.assertEqual(self.bot.exchange.create_order.call_count, 3)
        # Check that fetch_open_orders was called for verification
        self.bot.exchange.fetch_open_orders.assert_called_with(symbol)
        print("✅ Success Test Passed: Position kept because TP/SL verified.")

    def test_emergency_close_on_verification_failure(self):
        """Test that the bot closes the position if TP/SL are missing."""
        print("\n--- Testing Emergency Close on Verification Failure ---")
        symbol = "ETH/USDT:USDT"
        
        # Mock main order success, but subsequent calls mock failures or missing verification
        self.bot.exchange.create_order.side_effect = [
            {'id': 'main_order_ETH'}, # Main order opens
            {'id': 'sl_order_ETH'},   # SL order
            {'id': 'tp_order_ETH'},   # TP order
            {'id': 'emergency_close_ETH'} # EMERGENCY CLOSE
        ]
        
        # Mock verification FAILURE (orders not appearing on exchange)
        self.bot.exchange.fetch_open_orders.return_value = [] # Nothing found!
        
        # We expect execute_live_order to return an error status after trying to close
        res = self.bot.execute_live_order(symbol, "BUY", 100, 2500, 2700, 2400)
        
        self.assertEqual(res['status'], 'ERROR')
        self.assertIn("Verification Failed", res['msg'])
        
        # Check that emergency close (side=sell) was called
        # Call 1: Main Buy
        # Call 2: SL (Sell)
        # Call 3: TP (Sell)
        # Call 4: Emergency Market Sell
        calls = self.bot.exchange.create_order.call_args_list
        self.assertEqual(len(calls), 4)
        
        emergency_call = calls[3]
        self.assertEqual(emergency_call[1]['type'], 'MARKET')
        self.assertEqual(emergency_call[1]['side'], 'sell')
        self.assertEqual(emergency_call[1]['params']['reduceOnly'], True)
        
        print("✅ Emergency Close Test Passed: Position CLOSED because TP/SL missing.")

if __name__ == "__main__":
    unittest.main()
