from google import genai
from google.genai import types

from app.services.llm.base import LLMProvider


class GeminiAPIError(RuntimeError):
    """A user-safe error raised when Gemini cannot generate a response."""


class GeminiTimeoutError(GeminiAPIError):
    """A Gemini request timeout."""


class GeminiLLMProvider(LLMProvider):
    """LLM provider backed by the Google Gemini API."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str,
        timeout: float,
    ) -> None:
        if api_key is None or not api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
            )

        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        self.api_key = api_key.strip()
        self.model_name = model_name.strip()
        self.timeout = timeout

        self.client = genai.Client(api_key=self.api_key)

    def _generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(
                        timeout=int(self.timeout * 1000),
                    ),
                ),
            )

        except TimeoutError:
            raise GeminiTimeoutError(
                "Gemini request timed out. Please try again."
            ) from None

        except Exception as exc:
            # Temporary debugging: print the original Gemini error.
            print(
                f"GEMINI ACTUAL ERROR: {repr(exc)}",
                flush=True,
            )

            message = str(exc)
            lowered = message.lower()

            if "timeout" in lowered or "timed out" in lowered:
                detail = "Gemini request timed out. Please try again."

            elif "401" in message or "403" in message:
                detail = (
                    "Gemini authentication or authorization failed."
                )

            elif "429" in message:
                detail = (
                    "Gemini rate limit exceeded. Please retry later."
                )

            elif "5" in message and "status" in lowered:
                detail = (
                    "Gemini is temporarily unavailable. Please retry later."
                )

            else:
                detail = "Gemini could not generate a response."

            raise GeminiAPIError(detail) from exc

        answer = getattr(response, "text", None)

        if not isinstance(answer, str) or not answer.strip():
            raise GeminiAPIError(
                "Gemini response did not contain generated text."
            )

        return answer.strip()