import os
import requests

API_KEY = os.environ.get("GEMINI_API_KEY")

MODELS = {
    "gpt-4o-mini": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview",
        "type": "openai",
        "model_name": "gpt-4o-mini"
    },
    "gemini-2.0-flash-lite": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/openai/deployments/google-gemini-2-0-flash-lite/chat/completions",
        "type": "openai",
        "model_name": "gemini-2.0-flash-lite"
    },
    "claude-haiku-4.5": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/google/v1/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
        "type": "anthropic",
        "model_name": "claude-haiku-4-5@20251001"
    },
    "claude-sonnet-4.5": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/google/v1/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
        "type": "anthropic",
        "model_name": "claude-sonnet-4-5@20250929"
    },
    "claude-3.5-haiku": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/google/v1/publishers/anthropic/models/claude-3-5-haiku@20241022:rawPredict",
        "type": "anthropic",
        "model_name": "claude-3-5-haiku@20241022"
    },
    "gpt-5": {
        "endpoint": "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-2025-08-07/chat/completions?api-version=2025-04-01-preview",
        "type": "openai",
        "model_name": "gpt-5-2025-08-07"
    }
}

print("Testing all models...\n")

for name, config in MODELS.items():
    print(f"Testing {name}...")
    
    try:
        if config["type"] == "openai":
            headers = {
                "genaiplatform-farm-subscription-key": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": config["model_name"],
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20
            }
        else:  # anthropic
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello"}]}],
                "max_tokens": 20
            }
        
        response = requests.post(config["endpoint"], headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            print(f"  ? {name} works!")
        else:
            print(f"  ? {name} failed: {response.status_code} - {response.text[:100]}")
    
    except Exception as e:
        print(f"  ? {name} error: {str(e)[:100]}")
    
    print()