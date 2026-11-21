import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluators import LLM_as_evaluator, split_model_answer
from make_slice_json import make_slice_json


class EvaluatorTests(unittest.TestCase):
    def evaluate(self, content):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        constructor = Mock()
        constructor.return_value.__enter__ = Mock(return_value=client)
        constructor.return_value.__exit__ = Mock(return_value=False)
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=constructor)}):
            verdict = LLM_as_evaluator(
                "What is 2+3?", "<think>private reasoning</think>2+3=<<2+3=5>>5\n#### 5",
                "#### 5",
            )
        return verdict, client

    def test_boolean_verdicts_and_request(self):
        for expected in (True, False):
            with self.subTest(expected=expected):
                verdict, client = self.evaluate(json.dumps({"verdict": expected}))
                self.assertIs(verdict, expected)
                request = client.chat.completions.create.call_args.kwargs
                self.assertEqual(request["model"], "gpt-oss:20b")
                payload = json.loads(request["messages"][1]["content"])
                self.assertNotIn("private reasoning", payload["solution"])
                self.assertIn("<<2+3=5>>", payload["solution"])
                self.assertEqual(payload["math_task"], "What is 2+3?")
                self.assertEqual(payload["reference_solution"], "#### 5")

    def test_malformed_results_raise_instead_of_returning_false(self):
        for content in (None, "", "not json", '[]', '{}',
                        '{"verdict": "false"}', '{"verdict": 0}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.evaluate(content)

    def test_split_preserves_solution_and_calculation_annotations(self):
        self.assertEqual(
            split_model_answer("<think>first</think><think>second</think>\n<<2+3=5>>\n#### 5"),
            ("first\n\nsecond", "<<2+3=5>>\n#### 5"),
        )
        self.assertEqual(split_model_answer("#### 5"), ("", "#### 5"))
        with self.assertRaises(ValueError):
            split_model_answer("<think>unfinished")

    def test_thinking_without_solution_is_rejected_before_api_call(self):
        with self.assertRaises(ValueError):
            LLM_as_evaluator("task", "<think>thoughts</think>", "#### 5")


class SliceTests(unittest.TestCase):
    def test_slice_separates_fields_keeps_ids_and_creates_no_hints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tasks.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["задача", "ответ модели", "результат exact_match", "эталонный ответ"])
                for number in range(60):
                    writer.writerow([
                        f"task {number + 1}", "<think>thoughts</think>\n#### 5",
                        number % 2 == 0, "reference\n#### 5",
                    ])
            with redirect_stdout(io.StringIO()):
                path = make_slice_json(source, root / "audit")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 52)
            self.assertEqual([row["id"] for row in rows], [f"K09-{i:03d}" for i in range(1, 53)])
            self.assertEqual(sum(row["group"] == "core" for row in rows), 20)
            self.assertEqual(sum(row["group"] == "second" for row in rows), 32)
            for row in rows:
                self.assertEqual(row["model_thinking"], "thoughts")
                self.assertEqual(row["model_answer"], "#### 5")
                self.assertEqual(row["reference_answer"], "reference\n#### 5")
            self.assertEqual([file.name for file in path.parent.iterdir()], ["slice.jsonl"])


if __name__ == "__main__":
    unittest.main()
