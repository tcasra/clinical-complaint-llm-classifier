import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pipeline
import workflow
from pipeline import PipelineConfig


class WorkflowTest(unittest.TestCase):
    def test_resolve_stage_names(self) -> None:
        self.assertEqual(
            workflow.resolve_stage_names("recommended"),
            ["development", "final_gold"],
        )
        self.assertEqual(
            workflow.resolve_stage_names("full"),
            ["development", "final_gold", "final_asr"],
        )

    def test_build_base_config(self) -> None:
        args = workflow.parse_args(
            [
                "--plan",
                "recommended",
                "--methods",
                "zero_shot,few_shot",
                "--few-shot-k",
                "3",
                "--bootstrap-samples",
                "25",
            ]
        )
        config = workflow.build_base_config(args)
        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config.methods, ["zero_shot", "few_shot"])
        self.assertEqual(config.few_shot_k, 3)
        self.assertEqual(config.bootstrap_samples, 25)

    def test_build_stage_config(self) -> None:
        args = workflow.parse_args(["--plan", "recommended"])
        base_config = workflow.build_base_config(args)
        stage_config = workflow.build_stage_config(
            base_config=base_config,
            stage_name="development",
            workflow_root=Path("/tmp/workflow"),
            limit=50,
        )
        self.assertEqual(stage_config.split, "validate")
        self.assertEqual(stage_config.transcript_source, "gold")
        self.assertEqual(stage_config.limit, None)
        self.assertEqual(len(stage_config.file_names), 50)
        self.assertEqual(stage_config.output_dir, Path("/tmp/workflow/development"))

    def test_build_stage_config_sampling_is_deterministic(self) -> None:
        args = workflow.parse_args(["--plan", "recommended"])
        base_config = workflow.build_base_config(args)
        first = workflow.build_stage_config(
            base_config=base_config,
            stage_name="final_gold",
            workflow_root=Path("/tmp/workflow"),
            limit=10,
        )
        second = workflow.build_stage_config(
            base_config=base_config,
            stage_name="final_gold",
            workflow_root=Path("/tmp/workflow"),
            limit=10,
        )
        self.assertEqual(first.file_names, second.file_names)
        self.assertEqual(len(first.file_names), 10)

    def test_find_latest_resumable_workflow(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = workflow.parse_args(
                [
                    "--plan",
                    "recommended",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            base_config = workflow.build_base_config(args)

            workflow_root = output_dir / "20260506T120000Z"
            stage_dir = workflow_root / "development"
            stage_dir.mkdir(parents=True)
            stage_config = workflow.build_stage_config(
                base_config=base_config,
                stage_name="development",
                workflow_root=workflow_root,
                limit=workflow.limit_for_stage(args, "development"),
            )
            pipeline.persist_checkpoint(
                run_dir=stage_dir,
                config=stage_config,
                rows=[],
                stt_records=[],
                errors=[],
                missing_metadata_files=[],
                status="partial_budget_stop",
                stop_reason="insufficient credits",
            )
            summary = {
                "generated_at_utc": "2026-05-06T12:00:00+00:00",
                "plan": "recommended",
                "workflow_root": str(workflow_root),
                "stages": [
                    {
                        "stage": "development",
                        "run_dir": str(stage_dir),
                        "status": "partial_budget_stop",
                        "stop_reason": "insufficient credits",
                        "summary_file": str(stage_dir / pipeline.SUMMARY_JSON_FILE),
                        "summary": {},
                    }
                ],
            }
            (workflow_root / "workflow_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

            latest = workflow.find_latest_resumable_workflow(
                output_dir=output_dir,
                plan="recommended",
                base_config=base_config,
                args=args,
            )
            self.assertEqual(latest, workflow_root)

    def test_resolve_workflow_root_respects_fresh(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = workflow.parse_args(
                [
                    "--plan",
                    "recommended",
                    "--output-dir",
                    str(output_dir),
                    "--fresh",
                ]
            )
            base_config = workflow.build_base_config(args)
            workflow_root = workflow.resolve_workflow_root(args, base_config)
            self.assertEqual(workflow_root.parent, output_dir)
            self.assertNotEqual(workflow_root, output_dir)

    def test_run_workflow_persists_stage_run_dir_before_stage_completes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = workflow.parse_args(
                [
                    "--plan",
                    "recommended",
                    "--output-dir",
                    str(output_dir),
                    "--fresh",
                ]
            )
            workflow_root = output_dir / "20260506T123000Z"
            with mock.patch.object(
                workflow,
                "workflow_timestamp",
                return_value="20260506T123000Z",
            ), mock.patch.object(
                workflow,
                "run_stage",
                side_effect=RuntimeError("interrupted"),
            ):
                with self.assertRaises(RuntimeError):
                    workflow.run_workflow(args)

            summary = json.loads(
                (workflow_root / "workflow_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["plan"], "recommended")
            self.assertEqual(len(summary["stages"]), 1)
            self.assertEqual(summary["stages"][0]["stage"], "development")
            self.assertEqual(
                Path(summary["stages"][0]["run_dir"]),
                workflow_root / "development",
            )
            self.assertEqual(summary["stages"][0]["status"], "running")


if __name__ == "__main__":
    unittest.main()
