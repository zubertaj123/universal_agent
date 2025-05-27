#!/usr/bin/env python3
"""
Debug WebSocket connection issues
"""
import asyncio
import websockets
import ssl
import json

async def test_websocket_connection():
    """Test WebSocket connection to the server"""
    # Test both HTTP and HTTPS WebSocket connections
    
    urls_to_test = [
        "ws://localhost:8000/ws/call/test-session",
        "wss://localhost:8443/ws/call/test-session"
    ]
    
    for url in urls_to_test:
        print(f"\n🔄 Testing WebSocket connection to: {url}")
        
        try:
            # Create SSL context for HTTPS
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connect to WebSocket
            if url.startswith("wss://"):
                websocket = await websockets.connect(url, ssl=ssl_context)
            else:
                websocket = await websockets.connect(url)
            
            print("✅ WebSocket connected successfully!")
            
            # Send test message
            test_message = {
                "type": "test",
                "message": "Hello from debug script"
            }
            
            await websocket.send(json.dumps(test_message))
            print("✅ Test message sent")
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✅ Received response: {response}")
            except asyncio.TimeoutError:
                print("⚠️ No response received (timeout)")
            
            # Close connection
            await websocket.close()
            print("✅ WebSocket closed cleanly")
            
        except ConnectionRefusedError:
            print("❌ Connection refused - Server not running?")
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"❌ Invalid status code: {e}")
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")

async def test_http_endpoints():
    """Test HTTP endpoints"""
    import aiohttp
    
    urls_to_test = [
        "http://localhost:8000/health",
        "https://localhost:8443/health"
    ]
    
    for url in urls_to_test:
        print(f"\n🔄 Testing HTTP endpoint: {url}")
        
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ HTTP endpoint working: {data}")
                    else:
                        print(f"❌ HTTP endpoint returned status: {response.status}")
                        
        except Exception as e:
            print(f"❌ HTTP endpoint failed: {e}")

def main():
    """Run WebSocket debug tests"""
    print("🔍 WebSocket Connection Debug Tool")
    print("=" * 50)
    
    print("\n1. Testing HTTP endpoints...")
    asyncio.run(test_http_endpoints())
    
    print("\n2. Testing WebSocket connections...")
    asyncio.run(test_websocket_connection())
    
    print("\n" + "=" * 50)
    print("💡 If WebSocket connections fail:")
    print("1. Make sure the server is running")
    print("2. Check that both HTTP (8000) and HTTPS (8443) ports are working")
    print("3. Check browser console for specific error messages")
    print("4. Try opening https://localhost:8443 and accepting the certificate")

if __name__ == "__main__":
    main()