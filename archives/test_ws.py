import asyncio
import ccxt.pro

async def main():
    ex = ccxt.pro.binance({'options': {'defaultType': 'future'}})
    print("Connecting to WebSocket...")
    try:
        ticker = await ex.watch_ticker('BTC/USDT')
        print(f"✅ WS Success! BTC Price: {ticker['last']}")
    except Exception as e:
        print(f"❌ WS Failure: {e}")
    finally:
        await ex.close()

if __name__ == '__main__':
    asyncio.run(main())
