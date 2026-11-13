"""Адаптер официального google-genai; без истории диалога и инструментов."""

from dataclasses import dataclass
import re

from .settings import Settings


def redact(message: str, secret: str = "") -> str:
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return re.sub(r"AIza[\w-]+", "[REDACTED]", message)


class GenerationError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class Generation:
    text: str
    finish_reason: str | None
    model_version: str | None
    usage: dict
    raw: dict


class GeminiClient:
    def __init__(self, api_key: str, settings: Settings):
        from google import genai
        from google.genai import types

        self._secret = api_key
        # Один реальный запрос на попытку. Повторы и их учёт выполняет runner.
        self._client = genai.Client(
            api_key=api_key,
            vertexai=False,
            http_options=types.HttpOptions(
                timeout=int(settings.timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._model = settings.model
        self._config = types.GenerateContentConfig(
            **settings.generation_config(),
            tools=[],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(self, prompt: str) -> Generation:
        from google.genai import errors
        import httpx

        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt, config=self._config,
            )
        except errors.APIError as error:
            raise GenerationError(redact(str(error), self._secret), error.code) from None
        except (httpx.TransportError, ConnectionError, TimeoutError) as error:
            # Не повторяем таймауты автоматически: сервер мог уже обработать запрос.
            raise GenerationError(redact(str(error), self._secret)) from None

        candidates = response.candidates or []
        candidate = candidates[0] if candidates else None
        reason = getattr(candidate, "finish_reason", None)
        finish_reason = getattr(reason, "value", reason)
        # Читаем только финальный текст, без thought-частей.
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        text = "".join(part.text or "" for part in parts if not part.thought)
        usage = response.usage_metadata
        return Generation(
            text=text,
            finish_reason=finish_reason,
            model_version=response.model_version,
            usage=usage.model_dump(mode="json", exclude_none=True) if usage else {},
            raw=response.model_dump(mode="json", exclude_none=True,
                                    exclude={"sdk_http_response"}),
        )

    def close(self) -> None:
        self._client.close()
