import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.modules.ncloud_llm import NCloudLLM
from comparison.config import settings

def test_api():
    print("Testing NCloud HCX-007 API...")
    if not settings.NCLOUD_API_KEY:
        print("Error: NCLOUD_API_KEY not found in .env")
        return

    llm = NCloudLLM(
        api_key=settings.NCLOUD_API_KEY,
        api_url=settings.NCLOUD_API_URL
    )
    
    messages = [{"role": "user", "content": "Hello, are you working?"}]
    try:
        response = llm.generate(messages, thinking_effort="low")
        print(f"Success! Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
