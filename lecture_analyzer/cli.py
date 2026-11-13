import argparse
import os
from pathlib import Path
import sys

from .client import GeminiClient, redact
from .methods import METHODS, load_methods
from .runner import read_lectures, run_experiment
from .settings import Settings


ROOT = Path(__file__).resolve().parent.parent


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("требуется целое число больше нуля")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Сравнение M1 и M2 на текстовых лекциях через Gemini API.")
    result.add_argument("--config", type=Path, default=ROOT / "config.toml", help="файл настроек TOML")
    result.add_argument("--input", type=Path, help="один файл .txt или папка с лекциями (рекурсивно)")
    result.add_argument("--output", type=Path, help="папка для результатов")
    result.add_argument("--methods", nargs="+", choices=METHODS, default=["M1", "M2"])
    result.add_argument("--model", help="идентификатор модели из AI Studio")
    result.add_argument("--max-words", type=positive_int, help="общий предел слов в конспекте")
    result.add_argument("--repeat", type=positive_int, default=1, help="повторов каждого метода (по умолчанию 1)")
    result.add_argument("--dry-run", action="store_true", help="подготовить запросы без API и без ключа")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    client = None
    try:
        config_path = args.config.resolve()
        overrides = {
            "model": args.model, "max_words": args.max_words,
            "input": str(args.input.resolve()) if args.input else None,
            "output": str(args.output.resolve()) if args.output else None,
        }
        settings = Settings.load(config_path, overrides)
        source = (config_path.parent / settings.input).resolve()
        output = (config_path.parent / settings.output).resolve()
        lectures = read_lectures(source)
        methods = load_methods(list(dict.fromkeys(args.methods)))
        planned = len(lectures) * len(methods) * args.repeat
        print(f"Лекций: {len(lectures)}; методы: {', '.join(m.name for m in methods)}; "
              f"повторов: {args.repeat}; запросов без повторных попыток: {planned}.")
        print(f"Модель: {settings.model}; предел конспекта: {settings.max_words} слов.")
        if not args.dry_run:
            try:
                from dotenv import load_dotenv
                load_dotenv(config_path.parent / ".env", override=False)
            except ImportError:
                raise ValueError("Установите зависимости: python -m pip install -r requirements.txt") from None
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not key:
                raise ValueError("Нет GEMINI_API_KEY. Скопируйте .env.example в .env и заполните ключ.")
            client = GeminiClient(key, settings)
        _, code = run_experiment(
            lectures, methods, settings, output, client=client,
            repeats=args.repeat, dry_run=args.dry_run,
        )
        if args.dry_run:
            print("Пробный режим: запросы сохранены, API не вызывался, конспекты не создавались.")
        elif code:
            print("Есть незавершённые или неудачные ответы. Подробности в JSON и manifest.json.")
        else:
            print("Готово. Конспекты — в results/ внутри папки запуска; статистика — в metrics.csv.")
        return code
    except (OSError, ValueError, ImportError) as error:
        print(f"Ошибка: {redact(str(error), os.environ.get('GEMINI_API_KEY', ''))}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
