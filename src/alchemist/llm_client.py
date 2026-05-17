from pathlib import Path
from typing import Dict, List

import openai


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str = "llama-3.1-8b",
        api_base: str = "https://yunwu.ai/v1",
        key_file: str = "src\\openai_key.txt",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.key_file = key_file
        self.api_key = self._load_api_key()

    def _load_api_key(self) -> str:
        key_path = Path(__file__).resolve().parents[2] / self.key_file
        if not key_path.exists():
            raise FileNotFoundError(f"API key file not found: {key_path}")
        lines = [x.strip() for x in key_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            raise ValueError("API key is empty in key file.")
        key = lines[0]
        if not key:
            raise ValueError("API key is empty in key file.")
        return key

    def chat(self, messages: List[Dict[str, str]]) -> str:
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]
