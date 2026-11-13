"""Точки расширения: новые методы регистрируются в METHODS."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Method:
    name: str
    template: str

    def build_prompt(self, text: str, max_words: int) -> str:
        # replace, а не format: фигурные скобки в формулах лекции сохраняются.
        instruction = self.template.replace("{max_words}", str(max_words)).strip()
        return f"{instruction}\n\n<lecture>\n{text}\n</lecture>"


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
METHODS = {"M1": "m1.txt", "M2": "m2.txt"}


def load_methods(names: list[str]) -> list[Method]:
    return [
        Method(name, (PROMPT_DIR / METHODS[name]).read_text(encoding="utf-8"))
        for name in names
    ]
