import json
import requests
from typing import Dict, List, Optional

from haystack import component

# ==================================================
# OLLAMA CONFIG
# ==================================================

OLLAMA_BASE_URL = "http://localhost:11434"


# ==================================================
# 1. Model Discovery
# ==================================================

def get_available_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    response = requests.get(f"{base_url}/api/tags", timeout=None)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


# ==================================================
# 2. Pure Ollama Client (NO Haystack dependency)
# ==================================================

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url

    def generate(
        self,
        model: str,
        prompt: str,
        generation_kwargs: Optional[Dict] = None,
        stream: bool = True,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if generation_kwargs:
            payload.update(generation_kwargs)

        chunks = []

        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=stream,
            timeout=None,
        ) as response:
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    chunks.append(data.get("response", ""))
            else:
                data = response.json()
                chunks.append(data.get("response", ""))

        return "".join(chunks)


# ==================================================
# 3. Haystack-Compatible Generator
# ==================================================

@component
class OllamaGenerator:
    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_BASE_URL,
        generation_kwargs: Optional[Dict] = None,
    ):
        self.model = model
        self.client = OllamaClient(base_url)
        self.generation_kwargs = generation_kwargs or {}

    @component.output_types(replies=list)
    def run(self, prompt: str):
        text = self.client.generate(
            model=self.model,
            prompt=prompt,
            generation_kwargs=self.generation_kwargs,
            stream=True,
        )
        return {"replies": [text]}


# ==================================================
# 4. LOCAL TEST (runs ONLY when file is executed directly)
# ==================================================

if __name__ == "__main__":
    print("?? Testing ollama_models.py\n")

    # ---- Test 1: List models ----
    try:
        models = get_available_models()
        print(f"? Found {len(models)} models:")
        for m in models:
            print(" -", m)
    except Exception as e:
        print("? Failed to fetch models:", e)
        exit(1)

    if not models:
        print("\n?? No models found. Pull one using:")
        print("   docker exec -it <ollama_container> ollama pull mistral")
        exit(0)

    # ---- Test 2: Simple generation using first model ----
    print("\n?? Testing text generation...\n")

    client = OllamaClient()
    response = client.generate(
        model=models[0],
        prompt="Explain vector databases in one paragraph.",
        generation_kwargs={
            "temperature": 0.1,
            "num_predict": 100,
        },
    )

    print("?? Model response:\n")
    print(response)
