import unittest
from unittest.mock import Mock, patch

import httpx

from app.services.llm import (
    CloudflareAPIError,
    CloudflareAuthenticationError,
    CloudflareModelUnavailableError,
    CloudflareRateLimitError,
    CloudflareTimeoutError,
    CloudflareWorkersAIProvider,
    LLMProvider,
)


class CloudflareWorkersAIProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = CloudflareWorkersAIProvider(
            account_id="test-account-id",
            api_token="test-api-token",
            model_name="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            timeout=42.0,
        )

    def test_implements_provider_and_sends_rag_context(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "success": True,
            "result": {"response": "grounded answer"},
        }
        prompt = (
            "Question:\nWhat happened?\n\n"
            "Context:\n[Source 1 — Page 4]\nThe event occurred in June."
        )

        with patch(
            "app.services.llm.cloudflare.httpx.post",
            return_value=response,
        ) as post:
            answer = self.provider.generate(prompt)

        self.assertIsInstance(self.provider, LLMProvider)
        self.assertEqual(answer, "grounded answer")
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(
            args[0],
            "https://api.cloudflare.com/client/v4/accounts/"
            "test-account-id/ai/run/"
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        )
        self.assertEqual(
            kwargs["headers"]["Authorization"],
            "Bearer test-api-token",
        )
        self.assertEqual(kwargs["timeout"], 42.0)
        self.assertEqual(
            kwargs["json"]["messages"][1],
            {"role": "user", "content": prompt},
        )
        self.assertIn(
            "only the document context",
            kwargs["json"]["messages"][0]["content"],
        )

    def test_missing_account_id_or_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CLOUDFLARE_ACCOUNT_ID"):
            CloudflareWorkersAIProvider(
                account_id=None,
                api_token="test-api-token",
                model_name="@cf/test/model",
                timeout=42.0,
            )
        with self.assertRaisesRegex(ValueError, "CLOUDFLARE_API_TOKEN"):
            CloudflareWorkersAIProvider(
                account_id="test-account-id",
                api_token=" ",
                model_name="@cf/test/model",
                timeout=42.0,
            )

    def test_invalid_credentials_map_to_authentication_error(self):
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                response = Mock(status_code=status_code)
                response.json.return_value = {
                    "success": False,
                    "errors": [{"message": "Unauthorized"}],
                }
                with patch(
                    "app.services.llm.cloudflare.httpx.post",
                    return_value=response,
                ):
                    with self.assertRaises(CloudflareAuthenticationError):
                        self.provider.generate("question with context")

    def test_rate_limit_maps_to_rate_limit_error(self):
        response = Mock(status_code=429)
        response.json.return_value = {
            "success": False,
            "errors": [{"message": "Rate limit exceeded"}],
        }
        with patch(
            "app.services.llm.cloudflare.httpx.post",
            return_value=response,
        ):
            with self.assertRaises(CloudflareRateLimitError):
                self.provider.generate("question with context")

    def test_missing_model_maps_to_model_unavailable_error(self):
        response = Mock(status_code=404)
        response.json.return_value = {
            "success": False,
            "errors": [{"message": "Model not found"}],
        }
        with patch(
            "app.services.llm.cloudflare.httpx.post",
            return_value=response,
        ):
            with self.assertRaises(CloudflareModelUnavailableError):
                self.provider.generate("question with context")

    def test_request_timeout_is_reported(self):
        with patch(
            "app.services.llm.cloudflare.httpx.post",
            side_effect=httpx.ReadTimeout("request timed out"),
        ):
            with self.assertRaises(CloudflareTimeoutError):
                self.provider.generate("question with context")

    def test_connection_error_is_reported(self):
        with patch(
            "app.services.llm.cloudflare.httpx.post",
            side_effect=httpx.ConnectError("connection failed"),
        ):
            with self.assertRaisesRegex(
                CloudflareAPIError,
                "Unable to connect",
            ):
                self.provider.generate("question with context")

    def test_empty_model_response_is_rejected(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "success": True,
            "result": {"response": "  "},
        }
        with patch(
            "app.services.llm.cloudflare.httpx.post",
            return_value=response,
        ):
            with self.assertRaisesRegex(CloudflareAPIError, "did not contain"):
                self.provider.generate("question with context")

    def test_invalid_json_response_is_rejected(self):
        response = Mock(status_code=200)
        response.json.side_effect = ValueError("invalid JSON")
        with patch(
            "app.services.llm.cloudflare.httpx.post",
            return_value=response,
        ):
            with self.assertRaisesRegex(CloudflareAPIError, "invalid response"):
                self.provider.generate("question with context")


if __name__ == "__main__":
    unittest.main()
