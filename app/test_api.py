"""Quick test script for the Legal Assistant API"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Health check passed!\n")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
        return False

def test_chat():
    """Test the chat endpoint"""
    print("🔍 Testing Chat Endpoint...")
    try:
        payload = {
            "message": "ما هو مصدر الإفتاء في المملكة العربية السعودية؟",
            "mode": "rag"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n📝 Question: {payload['message']}")
            print(f"\n💬 Answer: {data['answer']}")
            print(f"\n🆔 Session ID: {data['session_id']}")
            print("\n✅ Chat test passed!")
            return True
        else:
            print(f"❌ Chat test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def test_search():
    """Test the search endpoint"""
    print("\n🔍 Testing Search Endpoint...")
    try:
        payload = {
            "query": "أحكام العقود",
            "k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🔎 Query: {payload['query']}")
            print(f"📊 Results Found: {data['total']}")
            
            for i, result in enumerate(data['results'][:2], 1):  # Show first 2
                print(f"\n{i}. {result['article_title']}")
                print(f"   Law: {result['law_name']}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Text: {result['article_text'][:100]}...")
            
            print("\n✅ Search test passed!")
            return True
        else:
            print(f"❌ Search test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🏛️  Legal Assistant API - Test Suite")
    print("=" * 60)
    print()
    
    # Test health
    if not test_health():
        print("\n⚠️  Server might not be running!")
        print("Make sure to start it with:")
        print("  python -m uvicorn app.main:app --reload")
        return
    
    # Test chat
    test_chat()
    
    # Test search
    test_search()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
