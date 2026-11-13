import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lecture_analyzer.cli import main
from lecture_analyzer.client import Generation, GenerationError
from lecture_analyzer.methods import load_methods
from lecture_analyzer.runner import read_lectures, run_experiment
from lecture_analyzer.settings import Settings


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def response(reason="STOP", text="Краткий конспект лекции."):
    return Generation(text, reason, "test-model-version", {
        "prompt_token_count": 100, "candidates_token_count": 20,
        "thoughts_token_count": 5, "total_token_count": 125,
    }, {"offline_fixture": True})


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "lecture.txt"
        self.text = "Определение алгоритма. Условие: x_{i} >= 0. Вывод."
        self.source.write_text(self.text, encoding="utf-8-sig")
        self.lectures = read_lectures(self.source)
        self.settings = replace(Settings(), request_interval_seconds=0, retry_delay_seconds=0)

    def run_case(self, client, methods=None, **kwargs):
        with patch("sys.stdout", new_callable=io.StringIO):
            directory, code = run_experiment(
                self.lectures, load_methods(methods or ["M1", "M2"]),
                self.settings, self.root / "output", client=client, **kwargs,
            )
        records = [json.loads(path.read_text()) for path in
                   sorted((directory / "results").glob("*.json"))
                   if not path.name.endswith(".response.json")]
        manifest = json.loads((directory / "manifest.json").read_text())
        return directory, code, records, manifest

    def test_complete_batch_saves_sources_prompts_usage_and_csv(self):
        client = FakeClient([response()] * 4)
        directory, code, records, manifest = self.run_case(client, repeats=2)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["api_attempts"], 4)
        self.assertEqual(manifest["completed_results"], 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(client.prompts[0], client.prompts[2])
        self.assertEqual(client.prompts[1], client.prompts[3])
        self.assertNotEqual(client.prompts[0], client.prompts[1])
        snapshot = directory / manifest["lectures"][0]["snapshot"]
        self.assertEqual(snapshot.read_text(), self.text)
        self.assertIn("x_{i}", records[0]["prompt"])
        self.assertEqual(records[0]["model_version"], "test-model-version")
        self.assertEqual(records[0]["usage"]["thoughts_token_count"], 5)
        with (directory / "metrics.csv").open(encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["total_token_count"], "125")
        self.assertTrue((directory / "results" / f"{records[0]['result_id']}.md").exists())

    def test_quota_error_stops_batch_and_preserves_success(self):
        client = FakeClient([response(), GenerationError("Quota exceeded", 429)])
        directory, code, records, manifest = self.run_case(client, repeats=3)
        self.assertEqual(code, 2)
        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(manifest["status"], "stopped")
        self.assertEqual(manifest["planned_results"], 6)
        self.assertEqual([r["status"] for r in records], ["success", "error"])
        self.assertEqual(records[1]["usage"], {})
        self.assertIsNone(records[1]["summary_words"])

    def test_server_retry_is_counted_and_does_not_pollute_next_prompt(self):
        client = FakeClient([GenerationError("Unavailable", 503), response()])
        _, code, records, manifest = self.run_case(client, ["M1"])
        self.assertEqual(code, 0)
        self.assertEqual(manifest["api_attempts"], 2)
        self.assertEqual(len(records[0]["attempts"]), 2)
        self.assertEqual(client.prompts[0], client.prompts[1])
        self.assertEqual(records[0]["status"], "success")
        self.assertIsNone(records[0]["error"])

    def test_truncated_blocked_and_empty_are_not_success(self):
        for reason, text, expected in [
            ("MAX_TOKENS", "Часть ответа", "truncated"),
            ("SAFETY", "", "blocked"), ("STOP", "", "empty"),
        ]:
            with self.subTest(reason=reason):
                _, code, records, _ = self.run_case(FakeClient([response(reason, text)]), ["M1"])
                self.assertEqual(code, 2)
                self.assertEqual(records[0]["status"], expected)

    def test_dry_run_never_calls_client_and_does_not_fabricate_metrics(self):
        client = FakeClient([])
        _, code, records, manifest = self.run_case(client, dry_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(client.prompts, [])
        self.assertEqual(manifest["api_attempts"], 0)
        for record in records:
            self.assertEqual(record["status"], "dry_run")
            self.assertIsNone(record["summary_words"])
            self.assertEqual(record["usage"], {})

    def test_output_names_do_not_collide_for_same_named_lectures(self):
        other = self.root / "nested" / "lecture.txt"
        other.parent.mkdir()
        other.write_text("Другая лекция", encoding="utf-8")
        self.lectures = read_lectures(self.root)
        _, _, records, _ = self.run_case(FakeClient([response()] * 2), ["M1"])
        self.assertEqual(len({record["result_id"] for record in records}), 2)

    def test_config_relative_paths_and_cli_override_dry_run(self):
        config = self.root / "custom.toml"
        config.write_text('input = "lecture.txt"\noutput = "out"\n', encoding="utf-8")
        with patch("sys.stdout", new_callable=io.StringIO):
            code = main(["--config", str(config), "--dry-run", "--methods", "M2"])
        self.assertEqual(code, 0)
        self.assertEqual(len(list((self.root / "out").glob("*/manifest.json"))), 1)

    def test_empty_source_rejected_before_starting_run(self):
        self.source.write_text(" \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Пустая лекция"):
            read_lectures(self.source)

    def test_interruption_preserves_current_result_and_manifest(self):
        class InterruptedClient:
            def generate(self, prompt):
                raise KeyboardInterrupt

        _, code, records, manifest = self.run_case(InterruptedClient())
        self.assertEqual(code, 130)
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["completed_results"], 1)
        self.assertEqual(records[0]["status"], "interrupted")
        self.assertEqual(len(records[0]["attempts"]), 1)

    def test_exhausted_server_retries_remain_error_then_batch_continues(self):
        client = FakeClient([
            GenerationError("Unavailable", 503),
            GenerationError("Unavailable", 503),
            GenerationError("Unavailable", 503), response(),
        ])
        _, code, records, manifest = self.run_case(client)
        self.assertEqual(code, 2)
        self.assertEqual(manifest["status"], "completed_with_errors")
        self.assertEqual(manifest["api_attempts"], 4)
        self.assertEqual([record["status"] for record in records], ["error", "success"])
        self.assertEqual(len(records[0]["attempts"]), 3)
        self.assertEqual(records[0]["usage"], {})


if __name__ == "__main__":
    unittest.main()
