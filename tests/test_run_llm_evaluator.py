import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_llm_evaluator import main, run_llm_evaluator


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "slice.jsonl"
        self.output = Path(self.temp.name) / "scorer.jsonl"
        self.objects = [
            {"id": f"K09-{i:03d}", "question": f"task {i}", "model_thinking": "private",
             "model_answer": "#### 5", "reference_answer": "#### 5"}
            for i in range(1, 4)
        ]
        self.source.write_text(
            "\n".join(json.dumps(obj) for obj in self.objects) + "\n", encoding="utf-8"
        )

    def run_with_mock(self, responses):
        with patch("run_llm_evaluator.LLM_as_evaluator", side_effect=responses) as evaluate:
            with redirect_stdout(io.StringIO()):
                run_llm_evaluator(self.source, self.output)
        return evaluate

    def read_results(self):
        return [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]

    def test_outputs_boolean_verdicts_without_sending_thinking(self):
        evaluate = self.run_with_mock([True, False, True])
        self.assertEqual(self.read_results(), [
            {"id": obj["id"], "verdict": verdict}
            for obj, verdict in zip(self.objects, [True, False, True])
        ])
        self.assertEqual(evaluate.call_count, 3)
        evaluate.assert_any_call(math_task="task 1", solution="#### 5", reference_solution="#### 5")

    def test_failure_preserves_rows_and_resume_skips_completed(self):
        with self.assertRaises(ConnectionError):
            self.run_with_mock([True, ConnectionError("offline")])
        self.assertEqual(self.read_results(), [{"id": "K09-001", "verdict": True}])
        evaluate = self.run_with_mock([False, True])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(len(self.read_results()), 3)
        evaluate = self.run_with_mock([])
        evaluate.assert_not_called()

    def test_resume_file_without_final_newline(self):
        self.output.write_text('{"id": "K09-001", "verdict": true}', encoding="utf-8")
        self.run_with_mock([False, True])
        self.assertEqual(len(self.read_results()), 3)

    def test_rejects_wrong_ids_and_nonboolean_verdict(self):
        self.output.write_text('{"id": "other", "verdict": true}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_with_mock([])
        self.output.write_text('{"id": "K09-001", "verdict": "false"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_with_mock([])

    def test_progress_resumes_from_saved_count(self):
        self.output.write_text('{"id": "K09-001", "verdict": true}\n', encoding="utf-8")
        with patch("run_llm_evaluator.tqdm") as progress:
            self.run_with_mock([False, True])
        self.assertEqual(progress.call_args.kwargs["initial"], 1)
        self.assertEqual(progress.call_args.kwargs["total"], 3)
        self.assertEqual(progress.return_value.__enter__.return_value.update.call_count, 2)

    def test_connection_error_has_actionable_message(self):
        stub = SimpleNamespace(APIConnectionError=ConnectionError, APITimeoutError=TimeoutError)
        with patch.dict(sys.modules, {"openai": stub}):
            with patch("run_llm_evaluator.run_llm_evaluator", side_effect=ConnectionError):
                with self.assertRaises(SystemExit) as error:
                    main()
        self.assertIn("ollama serve", str(error.exception))


if __name__ == "__main__":
    unittest.main()
