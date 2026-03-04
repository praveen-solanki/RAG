import os
import requests

API_KEY = os.environ.get("GEMINI_API_KEY")
ENDPOINT = "https://aoai-farm.bosch-temp.com/api/openai/deployments/google-gemini-2-0-flash-lite/chat/completions"

print("Testing Gemini 2.0 Flash Lite (OpenAI-compatible format)...")
print(f"API Key: {API_KEY[:20]}...")

headers = {
    "genaiplatform-farm-subscription-key": API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "model": "gemini-2.0-flash-lite",
    "messages": [
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0.1,
    "max_tokens": 50
}

try:
    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        print(f"? SUCCESS: {answer}")
    else:
        print(f"? FAILED: {response.text}")
        
except Exception as e:
    print(f"? ERROR: {e}")