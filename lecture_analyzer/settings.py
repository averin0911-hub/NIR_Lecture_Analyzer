from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Settings:
    model: str = "gemini-3.5-flash-lite"
    input: str = "data/lectures"
    output: str = "results"
    max_words: int = 500
    max_output_tokens: int = 8192
    temperature: float | None = None
    request_interval_seconds: float = 5.0
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_delay_seconds: float = 5.0

    @classmethod
    def load(cls, path: Path, overrides: dict) -> "Settings":
        with path.open("rb") as file:
            data = tomllib.load(file)
        allowed = {field.name for field in fields(cls)}
        unknown = data.keys() - allowed
        if unknown:
            raise ValueError(f"Неизвестные параметры: {', '.join(sorted(unknown))}")
        data.update({key: value for key, value in overrides.items() if value is not None})
        settings = cls(**data)
        settings.validate()
        return settings

    def validate(self) -> None:
        for key in ("model", "input", "output"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key}: требуется непустая строка")
        for key in ("max_words", "max_output_tokens", "max_retries"):
            value = getattr(self, key)
            minimum = 0 if key == "max_retries" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key}: требуется целое число >= {minimum}")
        for key in ("request_interval_seconds", "timeout_seconds", "retry_delay_seconds"):
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{key}: требуется конечное неотрицательное число")
        if self.timeout_seconds == 0:
            raise ValueError("timeout_seconds должен быть больше нуля")
        if self.temperature is not None:
            value = self.temperature
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 2):
                raise ValueError("temperature должна находиться в диапазоне [0, 2]")

    def generation_config(self) -> dict:
        config = {"max_output_tokens": self.max_output_tokens, "candidate_count": 1}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        return config

    def to_dict(self) -> dict:
        return asdict(self)
