from pathlib import Path
from typing import Dict, List
import time

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str = "llama-3.1-8b",
        api_base: str = "https://yunwu.ai/v1",
        key_file: str = "src/openai_key.txt",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.key_file = key_file
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = self._load_api_key()
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=timeout)

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
        retryable = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = choice.finish_reason

                if content and content.strip():
                    return content

                print(f"[LLM] Empty response. finish_reason={finish_reason}, content={repr(content)}")
                if finish_reason == "content_filter":
                    print(f"[LLM] Content filter triggered. Retrying with simplified prompt.")
                if finish_reason == "length":
                    print(f"[LLM] Response truncated. max_tokens may be too low.")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"[LLM] All {self.max_retries} attempts returned empty. Returning empty.")
                    return ""
            except retryable as e:
                print(f"[LLM] API error (attempt {attempt+1}/{self.max_retries}): {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"[LLM] All {self.max_retries} attempts failed. Returning empty.")
                    return ""
        return ""
