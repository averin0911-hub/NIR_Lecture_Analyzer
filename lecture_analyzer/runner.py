"""Последовательные запуски и единый формат артефактов M1/M2."""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import time
from uuid import uuid4

from .client import Generation, GenerationError
from .methods import Method
from .settings import Settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def word_count(text: str) -> int:
    return len(text.split())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Lecture:
    path: Path
    text: str


def read_lectures(source: Path) -> list[Lecture]:
    if not source.exists():
        raise ValueError(f"Путь не найден: {source}")
    paths = sorted(source.rglob("*.txt")) if source.is_dir() else [source]
    if not paths:
        raise ValueError(f"Нет файлов .txt: {source}")
    lectures = []
    for path in paths:
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Нужен текстовый файл .txt: {path}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            raise ValueError(f"Сохраните файл в UTF-8: {path}") from None
        if not text.strip():
            raise ValueError(f"Пустая лекция: {path}")
        lectures.append(Lecture(path.resolve(), text))
    return lectures


def generation_status(generation: Generation) -> str:
    if generation.finish_reason == "MAX_TOKENS":
        return "truncated"
    if generation.finish_reason not in (None, "STOP"):
        return "blocked"
    return "success" if generation.text.strip() else "empty"


CSV_FIELDS = [
    "result_id", "lecture_name", "method", "repeat", "status", "model",
    "model_version", "source_sha256", "prompt_sha256", "source_words",
    "summary_words", "summary_to_source_word_ratio", "within_word_limit",
    "elapsed_seconds", "api_seconds", "request_attempts", "prompt_token_count",
    "candidates_token_count", "thoughts_token_count", "total_token_count",
    "finish_reason", "error_code",
]


def csv_row(record: dict) -> dict:
    row = {key: record.get(key) for key in CSV_FIELDS}
    row.update({key: record.get("usage", {}).get(key) for key in (
        "prompt_token_count", "candidates_token_count", "thoughts_token_count",
        "total_token_count",
    )})
    row["request_attempts"] = len(record["attempts"])
    return row


def run_experiment(
    lectures: list[Lecture], methods: list[Method], settings: Settings,
    output: Path, *, client=None, repeats: int = 1, dry_run: bool = False,
) -> tuple[Path, int]:
    """client.generate(prompt) -> Generation; client.close() вызывает CLI.

    Код 0: все ответы завершены; 2: ошибки/обрезка/блокировка; 130: Ctrl+C.
    Неудачные попытки без usage_metadata не получают выдуманных токенов.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "lectures").mkdir()
    (run_dir / "prompts").mkdir()
    (run_dir / "results").mkdir()
    for method in methods:
        (run_dir / "prompts" / f"{method.name}.txt").write_text(
            method.template, encoding="utf-8",
        )
    try:
        sdk_version = version("google-genai")
    except PackageNotFoundError:
        sdk_version = None
    manifest = {
        "schema_version": 1, "run_id": run_id, "started_at": utc_now(),
        "status": "running", "dry_run": dry_run, "settings": settings.to_dict(),
        "generation_config": settings.generation_config(),
        "transport": {"tools": [], "sdk_retry_attempts": 1},
        "environment": {"python": platform.python_version(), "google_genai": sdk_version},
        "methods": [method.name for method in methods], "repeats": repeats,
        "planned_results": len(lectures) * len(methods) * repeats,
        "completed_results": 0, "api_attempts": 0, "lectures": [],
    }
    for index, lecture in enumerate(lectures, start=1):
        slug = re.sub(r"[^\w.-]+", "_", lecture.path.stem)[:60] or "lecture"
        lecture_id = f"{index:04d}_{slug}"
        snapshot = f"lectures/{lecture_id}.txt"
        (run_dir / snapshot).write_text(lecture.text, encoding="utf-8")
        manifest["lectures"].append({
            "id": lecture_id, "original_path": str(lecture.path), "snapshot": snapshot,
            "sha256": sha256(lecture.text),
        })
    write_json(run_dir / "manifest.json", manifest)
    next_request_at = 0.0
    exit_code = 0
    stop = False
    print(f"Результаты: {run_dir}")
    try:
        with (run_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            file.flush()
            for lecture, info in zip(lectures, manifest["lectures"]):
                for repeat in range(1, repeats + 1):
                    for method in methods:
                        result_id = f"{info['id']}_{method.name}_r{repeat:02d}"
                        prompt = method.build_prompt(lecture.text, settings.max_words)
                        record = {
                            "schema_version": 1, "result_id": result_id,
                            "lecture_name": lecture.path.name, "lecture_id": info["id"],
                            "lecture_snapshot": info["snapshot"], "source_sha256": info["sha256"],
                            "method": method.name, "repeat": repeat, "model": settings.model,
                            "model_version": None, "prompt": prompt, "prompt_sha256": sha256(prompt),
                            "generation_config": settings.generation_config(),
                            "started_at": utc_now(), "status": "dry_run" if dry_run else "running",
                            "source_words": word_count(lecture.text), "summary": None,
                            "summary_words": None, "summary_to_source_word_ratio": None,
                            "within_word_limit": None, "usage": {}, "attempts": [],
                            "api_seconds": 0.0, "error": None, "error_code": None,
                            "finish_reason": None,
                        }
                        result_path = run_dir / "results" / f"{result_id}.json"
                        write_json(result_path, record)
                        started = time.perf_counter()
                        try:
                            if not dry_run:
                                for attempt in range(settings.max_retries + 1):
                                    time.sleep(max(0.0, next_request_at - time.monotonic()))
                                    attempt_started = time.perf_counter()
                                    next_request_at = time.monotonic() + settings.request_interval_seconds
                                    manifest["api_attempts"] += 1
                                    attempt_info = {"number": attempt + 1, "started_at": utc_now()}
                                    record["attempts"].append(attempt_info)
                                    try:
                                        generation = client.generate(prompt)
                                    except GenerationError as error:
                                        attempt_info.update({"error": str(error), "code": error.code})
                                        record.update(status="error", error=str(error), error_code=error.code)
                                        if error.code in (500, 502, 503, 504) and attempt < settings.max_retries:
                                            delay = min(60.0, settings.retry_delay_seconds * 2 ** attempt)
                                            next_request_at = max(next_request_at, time.monotonic() + delay)
                                            print(f"  Ошибка {error.code}, повтор через {delay:g} с.")
                                            continue
                                        # Лимит/неверный ключ/модель/параметры: сохраняем и останавливаемся.
                                        stop = error.code in (400, 401, 403, 404, 429)
                                        break
                                    finally:
                                        duration = time.perf_counter() - attempt_started
                                        attempt_info["seconds"] = round(duration, 6)
                                        record["api_seconds"] += duration
                                    count = word_count(generation.text)
                                    record.update(
                                        status=generation_status(generation), summary=generation.text,
                                        model_version=generation.model_version,
                                        finish_reason=generation.finish_reason, usage=generation.usage,
                                        summary_words=count,
                                        summary_to_source_word_ratio=round(count / record["source_words"], 6),
                                        within_word_limit=count <= settings.max_words,
                                        error=None, error_code=None,
                                    )
                                    write_json(run_dir / "results" / f"{result_id}.response.json", generation.raw)
                                    (run_dir / "results" / f"{result_id}.md").write_text(
                                        generation.text, encoding="utf-8",
                                    )
                                    break
                        except KeyboardInterrupt:
                            record["status"] = "interrupted"
                            raise
                        finally:
                            record["finished_at"] = utc_now()
                            record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
                            record["api_seconds"] = round(record["api_seconds"], 6)
                            write_json(result_path, record)
                            writer.writerow(csv_row(record))
                            file.flush()
                            manifest["completed_results"] += 1
                            write_json(run_dir / "manifest.json", manifest)
                        print(f"  {lecture.path.name} / {method.name} / {repeat}: {record['status']}")
                        if record["status"] not in ("success", "dry_run"):
                            exit_code = 2
                        if stop:
                            manifest["stop_reason"] = record["error"]
                            break
                    if stop:
                        break
                if stop:
                    break
    except KeyboardInterrupt:
        exit_code = 130
        manifest["status"] = "interrupted"
    except Exception:
        manifest["status"] = "failed"
        raise
    else:
        manifest["status"] = "stopped" if stop else ("completed_with_errors" if exit_code else "completed")
    finally:
        manifest["finished_at"] = utc_now()
        write_json(run_dir / "manifest.json", manifest)
    return run_dir, exit_code
