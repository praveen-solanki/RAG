import os
import requests

API_KEY = os.environ.get("GEMINI_API_KEY")
ENDPOINT = "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview"

print("Testing GPT-4o-mini...")
print(f"API Key: {API_KEY[:20]}...")

headers = {
    "genaiplatform-farm-subscription-key": API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.1,
    "max_tokens": 50
}

try:
    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"? SUCCESS: {result['choices'][0]['message']['content']}")
    else:
        print(f"? FAILED: {response.text}")
except Exception as e:
    print(f"? ERROR: {e}")