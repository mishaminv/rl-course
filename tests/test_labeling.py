import csv
import importlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.evaluators import exact_match


def import_with_mock_dependency(name, dependency, **attributes):
    stub = ModuleType(dependency)
    for key, value in attributes.items():
        setattr(stub, key, value)
    with patch.dict("sys.modules", {dependency: stub}):
        return importlib.import_module(name)


sampling = import_with_mock_dependency("src.sampling_for_labeling", "openai", OpenAI=Mock())
loading = import_with_mock_dependency("src.load_data", "datasets", load_dataset=Mock())


class ExactMatchTests(unittest.TestCase):
    def test_final_answer_only(self):
        self.assertTrue(exact_match("Wrong reasoning: 2+2=5\n#### 42", "Steps\n#### 42"))
        self.assertFalse(exact_match("42 appears in reasoning\n#### 5", "#### 42"))
        self.assertFalse(exact_match("The calculation gives 42", "#### 42"))
        self.assertFalse(exact_match("42", "#### 42"))
        self.assertFalse(exact_match("42", "42"))
        self.assertFalse(exact_match("#### 42", "42"))
        self.assertTrue(exact_match("2 + 3 = <<2+3=5>>5\n#### 5", "#### 5"))

    def test_numeric_normalization_and_invalid_answers(self):
        for predicted, reference in [
            ("#### 1,234.00", "#### 1234"),
            ("#### − 42.0", "#### -42"),
            ("#### 1\u00a0234", "#### 1234"),
            ("#### .50", "#### 0.5"),
        ]:
            with self.subTest(predicted=predicted):
                self.assertTrue(exact_match(predicted, reference))
        for answer in ["", "#### NaN", "####", "#### 1,23", "#### 42 or 43"]:
            with self.subTest(answer=answer):
                self.assertFalse(exact_match(answer, answer))
        self.assertFalse(exact_match("#### 42.0001", "#### 42"))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "test.csv"
        with self.dataset.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["question", "answer"])
            writer.writerows((f"question {i}", "reference steps\n#### 42") for i in range(5))

    def run_sampling(self, answers, quota=1):
        with patch.object(sampling, "get_model_answer", side_effect=answers) as get_answer:
            with redirect_stdout(io.StringIO()):
                path = sampling.sample_for_labeling(
                    self.dataset, self.root / "labeling", per_label=quota
                )
        with path.open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file)), get_answer

    def test_skip_full_group_stop_at_quotas_and_preserve_order(self):
        rows, get_answer = self.run_sampling([
            ("#### 42", "reasoning\n#### 42"),
            ("#### 42", "surplus answer"),
            ("#### 0", "wrong\n#### 0"),
        ])
        self.assertEqual(get_answer.call_count, 3)
        self.assertEqual([row["задача"] for row in rows], ["question 0", "question 2"])
        self.assertEqual([row["результат exact_match"] for row in rows], ["True", "False"])
        self.assertEqual(rows[0]["ответ модели"], "reasoning\n#### 42")
        self.assertEqual(rows[0]["эталонный ответ"], "reference steps\n#### 42")

    def test_exhaustion_with_missing_group(self):
        rows, get_answer = self.run_sampling([("#### 42", "#### 42")] * 5, quota=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(get_answer.call_count, 5)

    def test_api_failure_preserves_collected_rows(self):
        with self.assertRaises(ConnectionError):
            self.run_sampling([("#### 42", "#### 42"), ConnectionError("offline")])
        path, = (self.root / "labeling").glob("*.csv")
        with path.open(encoding="utf-8-sig", newline="") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 1)

    def test_thinking_is_saved_but_evaluated_separately(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="#### 42", reasoning="Consider 17")
            )]
        )
        content, full_answer = sampling.get_model_answer(client, "question", "gpt-oss:20b")
        self.assertEqual(content, "#### 42")
        self.assertIn("Consider 17", full_answer)
        self.assertNotIn("reference", str(client.chat.completions.create.call_args))

    def test_standard_completion_without_reasoning(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="#### 42"))]
        )
        self.assertEqual(
            sampling.get_model_answer(client, "question", "gpt-oss:20b"),
            ("#### 42", "#### 42"),
        )

    def test_local_openai_endpoint(self):
        with patch.object(sampling, "OpenAI") as client_class:
            self.run_sampling([("#### 42", "#### 42")] * 5)
        client_class.assert_called_once_with(
            base_url="http://localhost:11434/v1", api_key="ollama", timeout=600.0
        )

    def test_downloads_main_and_preserves_csv_multiline_answers(self):
        dataset = {split: [{"question": "a,b?", "answer": "step\n#### 42"}]
                   for split in ("train", "test")}
        with patch.object(loading, "load_dataset", return_value=dataset) as download:
            with redirect_stdout(io.StringIO()):
                paths = loading.download_dataset(self.root / "dataset")
        download.assert_called_once_with("openai/gsm8k", "main")
        self.assertEqual([path.name for path in paths], ["train.csv", "test.csv"])
        for path in paths:
            with path.open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(list(csv.DictReader(file)), dataset[path.stem])


if __name__ == "__main__":
    unittest.main()
