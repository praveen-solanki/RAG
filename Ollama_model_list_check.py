import json
import requests
from typing import Dict, List, Optional

from haystack import component


# ==================================================
# OLLAMA CONFIG
# ==================================================

OLLAMA_BASE_URL = "http://localhost:11434"


# ==================================================
# 1. Model Discovery (Pure Utility)
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
# 3. Haystack-Compatible Generator (Thin Wrapper)
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



if __name__ == "__main__":
    models = get_available_models()
    print("Available Ollama models:")
    for m in models:
        print("-", m)