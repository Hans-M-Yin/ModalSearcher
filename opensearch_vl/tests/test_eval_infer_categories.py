"""Full-benchmark accuracy and category reporting for fresh/resumed evaluations."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from opensearch_vl import eval_infer_with_llm as evaluator


class SearchWeaveCategoryTests(unittest.TestCase):
    def test_parquet_categories_enrich_resumed_results_without_judging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = root / "searchweave.parquet"
            pd.DataFrame([
                {"question_id": "q1", "answer": "a", "category": "image-start"},
                {"question_id": "q2", "answer": "a", "category": "image-start"},
                {"question_id": "q3", "answer": "a", "category": "image-end"},
                {"question_id": "q4", "answer": "a", "category": "image-interleave"},
                {"question_id": "q5", "answer": "a", "category": "image-end"},
            ]).to_parquet(answer_file)
            cached = []
            for i, acc in enumerate([1, 0, 1, 0], start=1):
                case_id = f"case{i}"
                (root / f"{case_id}_trajectory.json").write_text(json.dumps({
                    "case_id": case_id, "original_data": {"question_id": f"q{i}"},
                }))
                cached.append({"case_id": case_id, "acc": acc})
            details = root / "llm_eval_report_details.jsonl"
            details.write_text("".join(json.dumps(item) + "\n" for item in cached))
            with mock.patch.object(evaluator, "judge_answer") as judge:
                report = evaluator.run_eval(str(root), answer_file=str(answer_file))
            judge.assert_not_called()
            self.assertEqual(report["total"], 5)
            self.assertEqual(report["evaluated"], 4)
            self.assertEqual(report["missing"], 1)
            self.assertEqual(report["accuracy"], "40.00%")
            self.assertEqual(report["category_accuracy"], {
                "image-start": {"total": 2, "correct": 1, "accuracy": "50.00%"},
                "image-end": {"total": 2, "correct": 1, "accuracy": "50.00%"},
                "image-interleave": {"total": 1, "correct": 0, "accuracy": "0.00%"},
            })
            saved = [json.loads(line) for line in details.read_text().splitlines()]
            self.assertEqual([item["category"] for item in saved], [
                "image-start", "image-start", "image-end", "image-interleave",
            ])
            saved_report = json.loads((root / "llm_eval_report.json").read_text())
            self.assertEqual(saved_report["category_accuracy"], report["category_accuracy"])

    def test_trajectory_categories_include_judge_failures_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = root / "searchweave.parquet"
            pd.DataFrame([
                {"id": "a", "answer": "a", "category": "image-start"},
                {"id": "b", "answer": "a", "category": "image-end"},
                {"id": "c", "answer": "a", "category": ""},
            ]).to_parquet(answer_file)
            for case_id, category in [("a", "image-start"), ("b", "image-end"), ("c", "")]:
                (root / f"{case_id}_trajectory.json").write_text(json.dumps({
                    "case_id": case_id, "category": category,
                    "original_data": {"answer": "a"}, "final_response_text": "a",
                }))

            def judge(question, correct_answer, response, **kwargs):
                raise RuntimeError("judge unavailable")

            with mock.patch.object(evaluator, "judge_answer", side_effect=judge):
                report = evaluator.run_eval(str(root), answer_file=str(answer_file))
            self.assertEqual(report["error_count"], 3)
            self.assertEqual(report["accuracy"], "0.00%")
            self.assertEqual(set(report["category_accuracy"]), {"image-start", "image-end", "unknown"})
            self.assertTrue(all(item == {"total": 1, "correct": 0, "accuracy": "0.00%"}
                                for item in report["category_accuracy"].values()))

    def test_other_benchmark_does_not_get_searchweave_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = root / "other.parquet"
            pd.DataFrame([{"id": "q1", "answer": "a", "category": "science"}]).to_parquet(answer_file)
            (root / "q1_trajectory.json").write_text(json.dumps({
                "case_id": "q1", "category": "science", "original_data": {},
            }))
            with mock.patch.object(evaluator, "judge_answer", return_value={"acc": 1}):
                report = evaluator.run_eval(str(root), answer_file=str(answer_file))
            self.assertNotIn("category_accuracy", report)
            self.assertEqual(report["accuracy"], "100.00%")


class FullBenchmarkAccuracyTests(unittest.TestCase):
    def _dataset(self, root, count):
        path = root / "benchmark.parquet"
        pd.DataFrame([
            {
                "data_id": f"case_{i}", "question_id": f"q{i}",
                "question": f"case_{i}", "answer": "a",
            }
            for i in range(count)
        ], columns=["data_id", "question_id", "question", "answer"]).to_parquet(path)
        return path

    def _trajectory(self, root, case_id):
        (root / f"{case_id}_trajectory.json").write_text(json.dumps({
            "case_id": case_id,
            "prompt": [{"role": "user", "content": case_id}],
            "original_data": {"data_id": case_id, "answer": "a"},
            "final_response_text": "<answer>a</answer>",
        }))

    def test_missing_samples_count_as_incorrect_and_stale_cache_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = self._dataset(root, 4)
            for case_id in ("case_0", "case_1"):
                self._trajectory(root, case_id)
            (root / "failures").mkdir()
            (root / "failures/case_2_failure.json").write_text("{}")
            cached = [
                {"case_id": "case_0", "acc": 1},
                {"case_id": "case_1", "acc": 0},
                {"case_id": "unrelated_case", "acc": 1},
            ]
            details = root / "llm_eval_report_details.jsonl"
            details.write_text("".join(json.dumps(item) + "\n" for item in cached))
            with mock.patch.object(evaluator, "judge_answer") as judge:
                report = evaluator.run_eval(str(root), answer_file=str(answer_file))
            judge.assert_not_called()
            self.assertEqual(report["total"], 4)
            self.assertEqual(report["evaluated"], 2)
            self.assertEqual(report["missing"], 2)
            self.assertEqual(report["correct"], 1)
            self.assertEqual(report["accuracy"], "25.00%")
            self.assertEqual(report["error_count"], 0)
            self.assertEqual(report["label_distribution"], {0: 3, 1: 1})
            self.assertEqual(len(details.read_text().splitlines()), 2)
            self.assertEqual(json.loads((root / "llm_eval_report.json").read_text())["total"], 4)

    def test_no_completed_trajectories_reports_zero_accuracy(self):
        for count in (0, 3):
            with self.subTest(samples=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                answer_file = self._dataset(root, count)
                with mock.patch.object(evaluator, "judge_answer") as judge:
                    report = evaluator.run_eval(str(root), answer_file=str(answer_file))
                judge.assert_not_called()
                self.assertEqual(report["total"], count)
                self.assertEqual(report["evaluated"], 0)
                self.assertEqual(report["missing"], count)
                self.assertEqual(report["accuracy"], "0.00%")
                self.assertEqual(report["correct"], 0)
                self.assertEqual((root / "llm_eval_report_details.jsonl").read_text(), "")

    def test_dataset_path_can_be_recovered_from_run_config(self):
        for field in ("dataset", "arguments"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                answer_file = self._dataset(root, 3)
                self._trajectory(root, "case_0")
                snapshot = (
                    {"dataset": {"resolved_path": str(answer_file)}}
                    if field == "dataset"
                    else {"arguments": {"data_path": str(answer_file)}}
                )
                (root / "run_config.json").write_text(json.dumps(snapshot))
                with mock.patch.object(evaluator, "judge_answer", return_value={"acc": 1}):
                    report = evaluator.run_eval(str(root))
                self.assertEqual(report["total"], 3)
                self.assertEqual(report["correct"], 1)
                self.assertEqual(report["accuracy"], "33.33%")

    def test_full_dataset_is_required_before_judging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._trajectory(root, "case_0")
            with mock.patch.object(evaluator, "judge_answer") as judge:
                with self.assertRaisesRegex(ValueError, "--answer-file"):
                    evaluator.run_eval(str(root))
            judge.assert_not_called()

    def test_limit_does_not_reduce_denominator_or_reuse_other_cached_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = self._dataset(root, 4)
            cached = []
            for case_id in ("case_0", "case_1"):
                self._trajectory(root, case_id)
                cached.append({"case_id": case_id, "acc": 1})
            (root / "llm_eval_report_details.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in cached)
            )
            with mock.patch.object(evaluator, "judge_answer") as judge:
                report = evaluator.run_eval(str(root), answer_file=str(answer_file), limit=1)
            judge.assert_not_called()
            self.assertEqual(report["total"], 4)
            self.assertEqual(report["evaluated"], 1)
            self.assertEqual(report["missing"], 3)
            self.assertEqual(report["correct"], 1)
            self.assertEqual(report["accuracy"], "25.00%")

    def test_judge_errors_and_missing_trajectories_both_count_as_incorrect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = self._dataset(root, 4)
            for case_id in ("case_0", "case_1"):
                self._trajectory(root, case_id)

            def judge_answer(question, *args, **kwargs):
                if question == "case_1":
                    raise RuntimeError("judge unavailable")
                return {"acc": 1}

            with mock.patch.object(evaluator, "judge_answer", side_effect=judge_answer):
                report = evaluator.run_eval(str(root), answer_file=str(answer_file))
            self.assertEqual(report["total"], 4)
            self.assertEqual(report["correct"], 1)
            self.assertEqual(report["evaluated"], 2)
            self.assertEqual(report["missing"], 2)
            self.assertEqual(report["error_count"], 1)
            self.assertEqual(report["accuracy"], "25.00%")
            self.assertEqual(report["label_distribution"], {0: 3, 1: 1})

    def test_more_trajectories_than_samples_is_rejected_before_judging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = self._dataset(root, 1)
            for case_id in ("case_0", "case_1"):
                self._trajectory(root, case_id)
            with mock.patch.object(evaluator, "judge_answer") as judge:
                with self.assertRaisesRegex(ValueError, "more trajectory files"):
                    evaluator.run_eval(str(root), answer_file=str(answer_file))
            judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
