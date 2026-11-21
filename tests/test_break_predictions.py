import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import break_predictions as probes


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "selected_tasks.csv"
        self.slice = self.root / "slice.jsonl"
        self.bodies = ["target one <<1+1=2>>", "target two <<2+2=4>>",
                       "nearest donor <<3+3=6>>", "other donor <<4+4=8>>"]
        self.vectors = [[1, 0], [0.9, 0.1], [1, 0.05], [0, 1]]
        self.objects = [
            {"id": f"K09-{i + 1:03d}", "group": "core", "question": f"task {i}",
             "model_thinking": "private", "model_answer": f"{body}\n\n#### {i + 7}",
             "reference_answer": f"#### {i + 7}"}
            for i, body in enumerate(self.bodies)
        ]
        self.slice.write_text("\n".join(json.dumps(obj) for obj in self.objects), encoding="utf-8")
        pd.DataFrame([
            {"задача": obj["question"], "ответ модели": "<think>private</think>\n" + obj["model_answer"],
             "результат exact_match": "True", "эталонный ответ": obj["reference_answer"],
             "ручная проверка": "null"}
            for obj in self.objects
        ]).to_csv(self.source, index=False, encoding="utf-8-sig")
        self.client = MagicMock()
        self.client.embeddings.create.side_effect = self.embed
        self.constructor = MagicMock()
        self.constructor.return_value.__enter__.return_value = self.client

    def embed(self, **kwargs):
        self.assertEqual(kwargs["model"], "bge-m3:latest")
        self.assertEqual(kwargs["encoding_format"], "float")
        for text in kwargs["input"]:
            self.assertNotIn("####", text)
            self.assertNotIn("private", text)
        return SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=self.vectors[self.bodies.index(text)])
            for i, text in enumerate(kwargs["input"])
        ])

    def run_probes(self):
        with patch.object(probes, "OpenAI", self.constructor):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                paths = probes.break_predictions(self.slice, self.source, self.root)
        return [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                for path in paths]

    def test_both_probes_change_exact_quota_and_preserve_other_fields(self):
        first, second = self.run_probes()
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        for rows in (first, second):
            self.assertEqual(sum(a != b for a, b in zip(rows, self.objects)), 2)
            self.assertEqual(rows[2:], self.objects[2:])
            for changed, original in zip(rows, self.objects):
                self.assertEqual({k: v for k, v in changed.items() if k != "model_answer"},
                                 {k: v for k, v in original.items() if k != "model_answer"})
        self.assertEqual(first[0]["model_answer"], self.bodies[0] + "\n\n#### seven")
        self.assertEqual(first[1]["model_answer"], self.bodies[1] + "\n\n#### eight")
        # Один наиболее похожий донор разрешено использовать дважды.
        self.assertEqual(second[0]["model_answer"], self.bodies[2] + "\n\n#### 7")
        self.assertEqual(second[1]["model_answer"], self.bodies[2] + "\n\n#### 8")
        self.assertEqual(self.slice.read_text(encoding="utf-8"),
                         "\n".join(json.dumps(obj) for obj in self.objects))

    def test_cache_in_source_csv_avoids_repeated_requests(self):
        original = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        expected = self.run_probes()
        cached = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(original, cached[original.columns])
        self.assertTrue(cached["embedding"].str.len().gt(0).all())
        self.constructor.reset_mock()
        self.assertEqual(self.run_probes(), expected)
        self.constructor.assert_not_called()

    def test_same_question_with_different_solution_is_allowed(self):
        frame = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        frame.loc[2, "задача"] = self.objects[0]["question"]
        frame.to_csv(self.source, index=False, encoding="utf-8-sig")
        _, second = self.run_probes()
        self.assertEqual(second[0]["model_answer"], self.bodies[2] + "\n\n#### 7")

    def test_other_selected_solution_is_allowed(self):
        self.vectors = [[1, 0], [0.999, 0.001], [0, 1], [0.3, 0.7]]
        _, second = self.run_probes()
        self.assertEqual(second[0]["model_answer"], self.bodies[1] + "\n\n#### 7")
        self.assertEqual(second[1]["model_answer"], self.bodies[0] + "\n\n#### 8")

    def test_all_copies_of_current_solution_are_skipped(self):
        frame = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        frame.loc[2, "ответ модели"] = self.objects[0]["model_answer"]
        frame.to_csv(self.source, index=False, encoding="utf-8-sig")
        _, second = self.run_probes()
        self.assertEqual(second[0]["model_answer"], self.bodies[1] + "\n\n#### 7")

    def test_changed_reasoning_invalidates_only_its_cached_embedding(self):
        self.run_probes()
        frame = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        changed = "changed donor <<5+5=10>>"
        self.bodies.append(changed)
        self.vectors.append([1, 0.02])
        frame.loc[2, "ответ модели"] = changed + "\n\n#### 10"
        frame.to_csv(self.source, index=False, encoding="utf-8-sig")
        self.client.embeddings.create.reset_mock()
        self.run_probes()
        self.client.embeddings.create.assert_called_once()
        self.assertEqual(self.client.embeddings.create.call_args.kwargs["input"], [changed])

    def test_completed_batches_survive_embedding_failure(self):
        calls = 0

        def fail_second_batch(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ConnectionError("offline")
            return self.embed(**kwargs)

        self.client.embeddings.create.side_effect = fail_second_batch
        with patch.object(probes, "EMBEDDING_BATCH_SIZE", 2):
            with self.assertRaises(ConnectionError):
                self.run_probes()
        cached = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        self.assertEqual(cached["embedding"].ne("").sum(), 2)
        self.client.embeddings.create.side_effect = self.embed
        self.client.embeddings.create.reset_mock()
        self.run_probes()
        request = self.client.embeddings.create.call_args.kwargs
        self.assertEqual(request["input"], self.bodies[2:])

    def test_words_support_negative_and_decimal_answers(self):
        self.assertEqual(probes.number_to_words("steps\n#### -3"), "steps\n#### minus three")
        self.assertEqual(probes.number_to_words("steps\n#### 1.5"), "steps\n#### one point five")


if __name__ == "__main__":
    unittest.main()
