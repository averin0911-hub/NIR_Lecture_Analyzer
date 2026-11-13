"""Настоящий SDK с перехватом HTTP: тесты не обращаются к Google."""

import json
import unittest
from unittest.mock import patch

from lecture_analyzer.client import GeminiClient, GenerationError
from lecture_analyzer.settings import Settings

try:
    import httpx
    from google import genai
except ImportError:
    httpx = None


@unittest.skipIf(httpx is None, "Установите requirements.txt для проверки SDK")
class SDKAdapterTests(unittest.TestCase):
    def test_request_is_stateless_and_response_metadata_is_preserved(self):
        requests = []

        def send(instance, request, **kwargs):
            requests.append(json.loads(request.content))
            return httpx.Response(200, request=request, json={
                "candidates": [{"content": {"role": "model", "parts": [
                    {"text": "Служебный текст", "thought": True},
                    {"text": "Итоговый конспект"},
                ]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 123, "candidatesTokenCount": 10,
                                  "thoughtsTokenCount": 4, "totalTokenCount": 137},
                "modelVersion": "fixture-version",
            })

        with patch.object(httpx.Client, "send", autospec=True, side_effect=send):
            client = GeminiClient("offline-key", Settings())
            try:
                first = client.generate("Запрос M1")
                second = client.generate("Запрос M2")
            finally:
                client.close()
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(requests[1]["contents"]), 1)
        self.assertEqual(requests[1]["contents"][0]["parts"][0]["text"], "Запрос M2")
        self.assertNotIn("temperature", requests[0]["generationConfig"])
        self.assertEqual(first.text, "Итоговый конспект")
        self.assertEqual(second.usage["total_token_count"], 137)
        self.assertEqual(first.finish_reason, "STOP")
        self.assertEqual(first.model_version, "fixture-version")

    def test_sdk_does_not_retry_quota_error_and_redacts_key(self):
        calls = []

        def send(instance, request, **kwargs):
            calls.append(request)
            return httpx.Response(429, request=request, json={
                "error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                          "message": "Quota exceeded for offline-secret-key"},
            })

        with patch.object(httpx.Client, "send", autospec=True, side_effect=send):
            client = GeminiClient("offline-secret-key", Settings())
            try:
                with self.assertRaises(GenerationError) as caught:
                    client.generate("Лекция")
            finally:
                client.close()
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.code, 429)
        self.assertNotIn("offline-secret-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
