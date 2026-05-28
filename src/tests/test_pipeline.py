import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import requests
from pipeline import (
    OpenRouterError,
    OpenRouterClient,
    PipelineConfig,
    RecoverableOpenRouterError,
    accuracy,
    allowed_prompt_labels,
    blocked_prediction_keys_from_errors,
    blocked_stt_files_from_errors,
    ensure_resume_compatible,
    build_confusion_matrix,
    compute_subgroup_metrics,
    is_content_moderation_error,
    is_budget_related_error,
    load_existing_run_artifacts,
    load_dotenv,
    normalize_text,
    persist_checkpoint,
    select_few_shot_examples,
    should_route_to_human_review,
    summarize_method_rows,
    summarize_results,
    text_similarity,
    write_json,
    word_overlap,
)


def make_config() -> PipelineConfig:
    return PipelineConfig(
        csv_path=Path("data/overview-of-recordings.csv"),
        recordings_dir=Path("data/recordings"),
        split="test",
        output_dir=Path("outputs"),
        file_names=[],
        limit=2,
        language="en",
        stt_model="stt-model",
        extraction_model="chat-model",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        sleep_seconds=0,
        timeout_seconds=30,
        transcript_source="gold",
        methods=["zero_shot", "few_shot"],
        few_shot_k=2,
        few_shot_pool_split="train",
        review_threshold=0.6,
        high_risk_review_threshold=0.8,
        high_risk_labels=["Heart hurts", "Hard to breath"],
        bootstrap_samples=10,
        subgroup_fields=["audio_clipping", "background_noise_audible"],
        resume_run_dir=None,
    )


class TextHelpersTest(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("Heart hurts!!!"), "heart hurts")

    def test_similarity_scores(self) -> None:
        self.assertGreater(text_similarity("heart hurts", "heart hurts badly"), 0.7)
        self.assertGreater(word_overlap("pain in my arm", "arm pain"), 0.4)


class DotenvLoaderTest(unittest.TestCase):
    def test_load_dotenv_keeps_existing_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text("OPENROUTER_API_KEY=from_file\n", encoding="utf-8")
            original = {}
            try:
                import os

                original["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY")
                os.environ["OPENROUTER_API_KEY"] = "from_env"
                load_dotenv(dotenv_path)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "from_env")
            finally:
                import os

                if original["OPENROUTER_API_KEY"] is None:
                    os.environ.pop("OPENROUTER_API_KEY", None)
                else:
                    os.environ["OPENROUTER_API_KEY"] = original["OPENROUTER_API_KEY"]


class FewShotSelectionTest(unittest.TestCase):
    def test_allowed_prompt_labels(self) -> None:
        metadata = {
            "a.wav": {"prompt": "Heart hurts"},
            "b.wav": {"prompt": "Cough"},
            "c.wav": {"prompt": "Heart hurts"},
            "d.wav": {"prompt": " "},
        }
        self.assertEqual(allowed_prompt_labels(metadata), ["Cough", "Heart hurts"])

    def test_select_few_shot_examples_prefers_similar_examples(self) -> None:
        pool = [
            {"file_name": "a.wav", "phrase": "my chest hurts", "prompt": "Heart hurts"},
            {"file_name": "b.wav", "phrase": "i keep coughing", "prompt": "Cough"},
            {"file_name": "c.wav", "phrase": "my knee has pain", "prompt": "Knee pain"},
            {"file_name": "d.wav", "phrase": "my heart hurts badly", "prompt": "Heart hurts"},
        ]
        selected = select_few_shot_examples(
            transcript="my heart hurts",
            few_shot_pool=pool,
            k=2,
            exclude_file_name="d.wav",
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["prompt"], "Heart hurts")
        self.assertNotEqual(selected[0]["file_name"], "d.wav")


class ReviewPolicyTest(unittest.TestCase):
    def test_budget_error_detection(self) -> None:
        self.assertTrue(is_budget_related_error(Exception("insufficient credits")))
        self.assertFalse(is_budget_related_error(Exception("invalid schema")))
        moderation_exc = Exception(
            "OpenRouter request failed (403): flagged for self-harm/intent. No credits were charged."
        )
        self.assertTrue(is_content_moderation_error(moderation_exc))
        self.assertFalse(is_budget_related_error(moderation_exc))

    def test_low_confidence_routes_to_review(self) -> None:
        should_review, reason = should_route_to_human_review(
            predicted_label="Cough",
            confidence=0.4,
            review_threshold=0.6,
            high_risk_labels=["Heart hurts"],
            high_risk_review_threshold=0.8,
        )
        self.assertTrue(should_review)
        self.assertIn("low_confidence", reason)

    def test_high_risk_gets_stricter_threshold(self) -> None:
        should_review, reason = should_route_to_human_review(
            predicted_label="Heart hurts",
            confidence=0.7,
            review_threshold=0.6,
            high_risk_labels=["Heart hurts"],
            high_risk_review_threshold=0.8,
        )
        self.assertTrue(should_review)
        self.assertIn("high_risk_label_low_confidence", reason)


class OpenRouterClientTest(unittest.TestCase):
    def test_post_wraps_transport_errors_as_recoverable(self) -> None:
        client = OpenRouterClient(
            api_key="test-key",
            api_base="https://openrouter.ai/api/v1",
            timeout_seconds=30,
        )
        with mock.patch.object(
            client.session,
            "post",
            side_effect=requests.Timeout("timed out"),
        ):
            with self.assertRaises(RecoverableOpenRouterError):
                client._post("/chat/completions", {"model": "demo"})

    def test_classify_transcript_retries_choice_level_provider_error(self) -> None:
        client = OpenRouterClient(
            api_key="test-key",
            api_base="https://openrouter.ai/api/v1",
            timeout_seconds=30,
        )
        error_response = {
            "choices": [
                {
                    "error": {
                        "code": 502,
                        "message": "Upstream error from OpenInference: Unknown role: -assistant",
                        "metadata": {"error_type": "provider_unavailable"},
                    },
                    "message": {
                        "role": "assistant",
                        "content": None,
                    },
                }
            ]
        }
        success_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "clinical_label": "Shoulder pain",
                                "confidence": 0.86,
                                "top_3_labels": [
                                    "Shoulder pain",
                                    "Heart hurts",
                                    "Cough",
                                ],
                                "chief_complaint": "Patient reports shoulder pain.",
                                "symptoms": ["pain"],
                                "body_parts": ["shoulder"],
                                "acuity": "low",
                            }
                        ),
                    }
                }
            ],
            "usage": {"cost": 0},
        }
        with mock.patch.object(
            client,
            "_post",
            side_effect=[error_response, success_response],
        ) as post_mock, mock.patch("pipeline.time.sleep") as sleep_mock:
            result = client.classify_transcript(
                transcript="my shoulder hurts",
                model="demo-model",
                label_options=["Shoulder pain", "Heart hurts", "Cough"],
                method="zero_shot",
                few_shot_examples=[],
            )

        self.assertEqual(result["clinical_label"], "Shoulder pain")
        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(sleep_mock.call_count, 1)
        first_payload = post_mock.call_args_list[0].args[1]
        self.assertNotIn("plugins", first_payload)


class MetricsTest(unittest.TestCase):
    def test_build_confusion_matrix(self) -> None:
        rows = [
            {"ground_truth_prompt": "Cough", "predicted_clinical_label": "Cough"},
            {"ground_truth_prompt": "Cough", "predicted_clinical_label": "Heart hurts"},
        ]
        matrix = build_confusion_matrix(rows, ["Cough", "Heart hurts"])
        self.assertEqual(matrix["Cough"]["Cough"], 1)
        self.assertEqual(matrix["Cough"]["Heart hurts"], 1)

    def test_summarize_method_rows(self) -> None:
        rows = [
            {
                "clinical_label_exact_match": True,
                "actual_label_in_top_3": True,
                "confidence": 0.9,
                "human_review_recommended": False,
                "human_review_reason": "auto_accept",
                "chat_cost": 0.01,
                "ground_truth_prompt": "Cough",
                "predicted_clinical_label": "Cough",
            },
            {
                "clinical_label_exact_match": False,
                "actual_label_in_top_3": True,
                "confidence": 0.4,
                "human_review_recommended": True,
                "human_review_reason": "low_confidence",
                "chat_cost": 0.02,
                "ground_truth_prompt": "Heart hurts",
                "predicted_clinical_label": "Cough",
            },
        ]
        summary = summarize_method_rows(
            rows,
            label_options=["Cough", "Heart hurts"],
            bootstrap_samples=5,
            shared_stt_cost=0.005,
        )
        self.assertEqual(summary["prediction_rows"], 2)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["top_3_accuracy"], 1.0)
        self.assertAlmostEqual(summary["estimated_total_cost_if_run_alone"], 0.035)

    def test_compute_subgroup_metrics(self) -> None:
        rows = [
            {
                "ground_truth_prompt": "Cough",
                "predicted_clinical_label": "Cough",
                "clinical_label_exact_match": True,
                "actual_label_in_top_3": True,
                "human_review_recommended": False,
                "audio_clipping": "no_clipping",
                "background_noise_audible": "no_noise",
            },
            {
                "ground_truth_prompt": "Heart hurts",
                "predicted_clinical_label": "Cough",
                "clinical_label_exact_match": False,
                "actual_label_in_top_3": False,
                "human_review_recommended": True,
                "audio_clipping": "light_clipping",
                "background_noise_audible": "light_noise",
            },
        ]
        metrics = compute_subgroup_metrics(
            rows,
            subgroup_fields=["audio_clipping"],
            label_options=["Cough", "Heart hurts"],
        )
        self.assertIn("no_clipping", metrics["audio_clipping"])
        self.assertEqual(metrics["audio_clipping"]["no_clipping"]["count"], 1)

    def test_summarize_results(self) -> None:
        config = make_config()
        rows = [
            {
                "file_name": "a.wav",
                "method": "zero_shot",
                "ground_truth_prompt": "Cough",
                "predicted_clinical_label": "Cough",
                "clinical_label_exact_match": True,
                "actual_label_in_top_3": True,
                "confidence": 0.9,
                "human_review_recommended": False,
                "human_review_reason": "auto_accept",
                "chat_cost": 0.01,
                "audio_clipping": "no_clipping",
                "background_noise_audible": "no_noise",
            },
            {
                "file_name": "a.wav",
                "method": "few_shot",
                "ground_truth_prompt": "Cough",
                "predicted_clinical_label": "Heart hurts",
                "clinical_label_exact_match": False,
                "actual_label_in_top_3": True,
                "confidence": 0.55,
                "human_review_recommended": True,
                "human_review_reason": "low_confidence",
                "chat_cost": 0.02,
                "audio_clipping": "no_clipping",
                "background_noise_audible": "no_noise",
            },
        ]
        summary = summarize_results(
            rows=rows,
            missing_metadata=[],
            config=config,
            label_options=["Cough", "Heart hurts"],
            stt_records=[],
        )
        self.assertEqual(summary["evaluated_examples"], 1)
        self.assertEqual(summary["prediction_rows"], 2)
        self.assertIn("zero_shot", summary["method_metrics"])
        self.assertIn("few_shot", summary["method_metrics"])
        self.assertEqual(summary["stt_summary"], None)
        self.assertAlmostEqual(summary["estimated_total_chat_cost"], 0.03)


class CheckpointTest(unittest.TestCase):
    def test_persist_and_load_checkpoint(self) -> None:
        config = make_config()
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()
            rows = [{"file_name": "a.wav", "method": "zero_shot"}]
            stt_records = [{"file_name": "a.wav", "predicted_transcript": "hello"}]
            errors = [{"file_name": "b.wav", "stage": "classify"}]
            persist_checkpoint(
                run_dir=run_dir,
                config=config,
                rows=rows,
                stt_records=stt_records,
                errors=errors,
                missing_metadata_files=["c.wav"],
                status="running",
                stop_reason=None,
            )
            loaded_rows, loaded_stt, loaded_errors, loaded_state = load_existing_run_artifacts(
                run_dir
            )
            self.assertEqual(loaded_rows, rows)
            self.assertEqual(loaded_stt, stt_records)
            self.assertEqual(loaded_errors, errors)
            self.assertEqual(loaded_state["status"], "running")
            self.assertEqual(loaded_state["missing_metadata_files"], ["c.wav"])

    def test_resume_compatibility_rejects_review_policy_changes(self) -> None:
        config = make_config()
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()
            persist_checkpoint(
                run_dir=run_dir,
                config=config,
                rows=[],
                stt_records=[],
                errors=[],
                missing_metadata_files=[],
                status="running",
                stop_reason=None,
            )
            _, _, _, loaded_state = load_existing_run_artifacts(run_dir)
            updated_config = replace(
                config,
                high_risk_review_threshold=0.9,
            )
            with self.assertRaises(OpenRouterError):
                ensure_resume_compatible(loaded_state, updated_config)

    def test_write_json_keeps_previous_contents_if_replace_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "state.json"
            json_path.write_text(json.dumps({"status": "old"}), encoding="utf-8")
            with mock.patch("pipeline.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_json(json_path, {"status": "new"})
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"status": "old"},
            )

    def test_blocked_keys_are_derived_from_moderation_errors(self) -> None:
        errors = [
            {
                "file_name": "a.wav",
                "stage": "classify",
                "method": "zero_shot",
                "retryable": False,
                "error_type": "content_moderation",
            },
            {
                "file_name": "a.wav",
                "stage": "transcribe",
                "method": None,
                "retryable": False,
                "error_type": "content_moderation",
            },
            {
                "file_name": "b.wav",
                "stage": "classify",
                "method": "few_shot",
                "retryable": False,
                "error_type": "fatal",
            },
            {
                "file_name": "c.wav",
                "stage": "classify",
                "method": "few_shot",
                "retryable": True,
                "error_type": "recoverable",
            },
        ]
        self.assertEqual(
            blocked_prediction_keys_from_errors(errors),
            {("a.wav", "zero_shot")},
        )
        self.assertEqual(
            blocked_stt_files_from_errors(errors),
            {"a.wav"},
        )


if __name__ == "__main__":
    unittest.main()
