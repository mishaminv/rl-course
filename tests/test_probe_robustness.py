import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import check_exact_match_robustness as checks


class MetricTests(unittest.TestCase):
    def test_delta_and_rejection_use_different_denominators(self):
        rows = checks.calculate_metrics("probe", [True, True, False, False],
                                        [False, True, True, False])
        self.assertEqual(rows[0]["n"], 4)
        self.assertEqual(rows[0]["value"], 0)
        self.assertEqual(rows[1]["n"], 2)
        self.assertEqual(rows[1]["value"], 0.5)
        self.assertTrue(all(row["interpretation"] == "" for row in rows))

    def test_no_initial_positives_skips_undefined_metric(self):
        with redirect_stdout(io.StringIO()):
            rows = checks.calculate_metrics("probe", [False, False], [True, False])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], 0.5)


class ProbeCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audit = Path(self.temp.name)
        self.original = [
            {"id": f"K09-{i + 1:03d}", "question": f"task {i}", "group": "core",
             "model_thinking": "private", "reference_answer": "#### 5",
             "model_answer": f"steps {i}\n#### {5 if i % 2 == 0 else 9}"}
            for i in range(4)
        ]
        self.write_jsonl("slice.jsonl", self.original)
        for filename in ("slice_broken_1.jsonl", "slice_broken_2.jsonl"):
            rows = [obj.copy() for obj in self.original]
            for i in range(2):
                if "_1" in filename:
                    rows[i]["model_answer"] = f"steps {i}\n#### {'five' if i == 0 else 'nine'}"
                else:
                    rows[i]["model_answer"] = f"other steps {i}\n#### {5 if i == 0 else 9}"
            self.write_jsonl(filename, rows)
        pd.DataFrame({"id": [obj["id"] for obj in self.original],
                      "verdict": [True, False, True, False]}).to_csv(
            self.audit / "baseline_predictions.csv", index=False,
        )
        self.write_jsonl("scorer.jsonl", [
            {"id": obj["id"], "verdict": i < 2} for i, obj in enumerate(self.original)
        ])
        self.results = self.audit / "probe_results.csv"
        self.results.write_text(
            "probe,n,metric,value,interpretation\nprevious,1,old_metric,0.25,keep me\n",
            encoding="utf-8-sig",
        )

    def write_jsonl(self, filename, rows):
        (self.audit / filename).write_text(
            "\n".join(json.dumps(obj) for obj in rows) + "\n", encoding="utf-8"
        )

    def run_check(self, verdicts):
        with patch.object(checks, "LLM_as_evaluator", side_effect=verdicts) as llm:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                checks.check_exact_match_robustness(self.audit)
        return llm

    def test_checks_only_changed_rows_and_appends_eight_results(self):
        previous = self.results.read_bytes()
        llm = self.run_check([True, False, False, False])
        self.assertEqual(llm.call_count, 4)
        self.assertTrue(self.results.read_bytes().startswith(previous))
        rows = pd.read_csv(self.results, keep_default_na=False)
        self.assertEqual(len(rows), 9)
        self.assertTrue(rows.iloc[1:]["interpretation"].eq("").all())
        indexed = rows.set_index(["probe", "metric"])
        self.assertEqual(indexed.loc[("answer_in_words/exact_match", "acceptance_rate_delta"), "value"], -0.5)
        self.assertEqual(indexed.loc[("answer_in_words/exact_match", "true_to_false_rate"), "n"], 1)
        self.assertEqual(indexed.loc[("swapped_reasoning/exact_match", "acceptance_rate_delta"), "value"], 0)
        self.assertEqual(indexed.loc[("swapped_reasoning/LLM_as_evaluator", "true_to_false_rate"), "value"], 1)
        self.assertEqual(indexed.loc[("answer_in_words/LLM_as_evaluator", "true_to_false_rate"), "value"], 0.5)

    def test_repeat_uses_cache_but_still_appends_new_rows(self):
        self.run_check([True, False, False, False])
        llm = self.run_check([])
        llm.assert_not_called()
        self.assertEqual(len(pd.read_csv(self.results)), 17)

    def test_failure_preserves_existing_results_and_completed_calls(self):
        previous = self.results.read_bytes()
        with self.assertRaises(ConnectionError):
            self.run_check([True, ConnectionError("offline")])
        self.assertEqual(self.results.read_bytes(), previous)
        cache = checks.read_objects(self.audit / "scorer_broken_1.jsonl")
        self.assertEqual(list(cache), ["K09-001"])
        llm = self.run_check([False, False, False])
        self.assertEqual(llm.call_count, 3)
        self.assertEqual(len(pd.read_csv(self.results)), 9)

    def test_changed_input_invalidates_only_its_cached_verdict(self):
        self.run_check([True, False, False, False])
        rows = list(checks.read_objects(self.audit / "slice_broken_1.jsonl").values())
        rows[0]["model_answer"] = "different explanation\n#### five"
        self.write_jsonl("slice_broken_1.jsonl", rows)
        llm = self.run_check([False])
        self.assertEqual(llm.call_count, 1)

    def test_invalid_ids_are_rejected_before_calls_or_append(self):
        previous = self.results.read_bytes()
        rows = list(checks.read_objects(self.audit / "slice_broken_2.jsonl").values())
        rows[0]["id"] = "unknown"
        self.write_jsonl("slice_broken_2.jsonl", rows)
        with self.assertRaises(ValueError):
            self.run_check([])
        self.assertEqual(self.results.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
