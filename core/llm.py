import os
import time

from dotenv import load_dotenv

load_dotenv()


class BaseLLM:
    name = "base"

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    """Cloud Gemini model used by the research assistant."""

    def __init__(self):
        from google import genai
        from google.genai import errors

        self._errors = errors

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.client = genai.Client(api_key=api_key)

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

        self.name = "gemini"

    def generate(self, prompt: str, retries: int = 3, **kwargs) -> str:
        last_error = None

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                return response.text or ""

            except self._errors.ServerError as e:
                last_error = e

                delay = 2 ** attempt
                print(
                    f"Gemini {self.model} unavailable. "
                    f"Retrying in {delay}s..."
                )

                time.sleep(delay)

        raise RuntimeError(
            "Gemini is temporarily unavailable. "
            "Please try again in a minute."
        ) from last_error


def get_llm(backend: str = None, url: str = None) -> BaseLLM:
    """Return the Gemini LLM used by the application."""
    return GeminiLLM()