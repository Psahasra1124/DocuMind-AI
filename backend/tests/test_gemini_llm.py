import unittest
from unittest.mock import Mock, patch

from app.services.llm import (
    GeminiAPIError,
    GeminiLLMProvider,
    GeminiTimeoutError,
    LLMProvider,
)


class GeminiLLMProviderTests(unittest.TestCase):
    @patch("app.services.llm.gemini.genai.Client")
    def test_implements_llm_provider_and_returns_generated_text(self, client_class):
        client = client_class.return_value
        client.models.generate_content.return_value = Mock(text="generated answer")

        provider = GeminiLLMProvider(
            api_key="test-api-key",
            model_name="gemini-test",
            timeout=42.0,
        )

        result = provider.generate("hello")

        self.assertIsInstance(provider, LLMProvider)
        self.assertEqual(result, "generated answer")
        client_class.assert_called_once_with(api_key="test-api-key")
        call = client.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "gemini-test")
        self.assertEqual(call.kwargs["contents"], "hello")

    @patch("app.services.llm.gemini.genai.Client")
    def test_empty_prompt_is_rejected_before_api_call(self, client_class):
        provider = GeminiLLMProvider(
            api_key="test-api-key",
            model_name="gemini-test",
            timeout=42.0,
        )

        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            provider.generate("   ")

        client_class.return_value.models.generate_content.assert_not_called()

    @patch("app.services.llm.gemini.genai.Client")
    def test_missing_api_key_is_rejected(self, client_class):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY is required"):
            GeminiLLMProvider(
                api_key=None,
                model_name="gemini-test",
                timeout=42.0,
            )

        client_class.assert_not_called()

    @patch("app.services.llm.gemini.genai.Client")
    def test_api_failure_is_converted_to_safe_error(self, client_class):
        client_class.return_value.models.generate_content.side_effect = Exception(
            "status 429 resource exhausted"
        )
        provider = GeminiLLMProvider(
            api_key="test-api-key",
            model_name="gemini-test",
            timeout=42.0,
        )

        with self.assertRaisesRegex(GeminiAPIError, "rate limit exceeded"):
            provider.generate("hello")

    @patch("app.services.llm.gemini.genai.Client")
    def test_timeout_is_converted_to_safe_error(self, client_class):
        client_class.return_value.models.generate_content.side_effect = TimeoutError()
        provider = GeminiLLMProvider(
            api_key="test-api-key",
            model_name="gemini-test",
            timeout=42.0,
        )

        with self.assertRaisesRegex(GeminiTimeoutError, "timed out"):
            provider.generate("hello")

    @patch("app.services.llm.gemini.genai.Client")
    def test_malformed_response_is_rejected(self, client_class):
        client_class.return_value.models.generate_content.return_value = Mock(
            text=None
        )
        provider = GeminiLLMProvider(
            api_key="test-api-key",
            model_name="gemini-test",
            timeout=42.0,
        )

        with self.assertRaisesRegex(GeminiAPIError, "did not contain"):
            provider.generate("hello")


if __name__ == "__main__":
    unittest.main()
