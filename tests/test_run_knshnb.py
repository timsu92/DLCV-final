import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from scripts.run_knshnb import build_parser, build_train_command, prepare_debug_run


class RunKnshnbTests(unittest.TestCase):
    def test_build_debug_command(self):
        command = build_train_command(
            config_path="config/debug.yaml",
            exp_name="debug",
            out_base_dir="/project/results/run/predictions",
            in_base_dir="input",
            save_checkpoint=False,
        )
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "src.train",
                "--config_path",
                "config/debug.yaml",
                "--exp_name",
                "debug",
                "--out_base_dir",
                "/project/results/run/predictions",
                "--in_base_dir",
                "input",
            ],
        )

    def test_build_command_with_checkpoint(self):
        command = build_train_command(
            config_path="config/efficientnet_b6.yaml",
            exp_name="b6",
            out_base_dir="/project/results/run/predictions",
            in_base_dir="input",
            save_checkpoint=True,
        )
        self.assertIn("--save_checkpoint", command)

    def test_build_command_with_load_snapshot(self):
        command = build_train_command(
            config_path="config/efficientnet_b6.yaml",
            exp_name="b6",
            out_base_dir="/project/results/run/predictions",
            in_base_dir="input",
            save_checkpoint=True,
            load_snapshot=True,
        )
        self.assertIn("--load_snapshot", command)
        self.assertIn("--save_checkpoint", command)

    def test_build_command_without_load_snapshot_omits_flag(self):
        command = build_train_command(
            config_path="config/efficientnet_b6.yaml",
            exp_name="b6",
            out_base_dir="/project/results/run/predictions",
            in_base_dir="input",
            save_checkpoint=False,
        )
        self.assertNotIn("--load_snapshot", command)

    def test_prepare_debug_run_creates_run_dir_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, command = prepare_debug_run(
                base_dir=Path(tmp) / "results",
                timestamp="2026-05-17-010203",
            )
            self.assertEqual(run_dir.name, "2026-05-17-010203-knshnb-debug")
            self.assertTrue((run_dir / "predictions").is_dir())
            manifest = (run_dir / "run_manifest.yaml").read_text(encoding="utf-8")
            self.assertIn("run_name: knshnb-debug", manifest)
            self.assertIn(
                f"command: {sys.executable} -m src.train --config_path config/debug.yaml --exp_name debug "
                f"--out_base_dir {command[8]} --in_base_dir input",
                manifest,
            )
            self.assertIn("source_manifest: data/source_manifest.yaml", manifest)
            self.assertEqual(command[0], sys.executable)
            self.assertTrue(Path(command[8]).is_absolute())
            self.assertEqual(Path(command[8]), (run_dir / "predictions").resolve())
            self.assertEqual(
                command,
                [
                    sys.executable,
                    "-m",
                    "src.train",
                    "--config_path",
                    "config/debug.yaml",
                    "--exp_name",
                    "debug",
                    "--out_base_dir",
                    str((run_dir / "predictions").resolve()),
                    "--in_base_dir",
                    "input",
                ],
            )

    def test_prepare_debug_run_manifest_matches_checkpoint_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, command = prepare_debug_run(
                base_dir=Path(tmp) / "results",
                timestamp="2026-05-17-010203",
                in_base_dir="custom-input",
                save_checkpoint=True,
            )
            manifest = (run_dir / "run_manifest.yaml").read_text(encoding="utf-8")
            self.assertIn(f"command: {' '.join(command)}", manifest)

    def test_build_parser_accepts_debug_without_manual_command_args(self):
        parser = build_parser()
        args = parser.parse_args(["--debug", "--dry-run"])
        self.assertTrue(args.debug)
        self.assertTrue(args.dry_run)
        self.assertIsNone(args.config_path)
        self.assertIsNone(args.exp_name)
        self.assertIsNone(args.out_base_dir)

    def test_build_parser_load_snapshot_defaults_false_and_can_be_set(self):
        parser = build_parser()
        args = parser.parse_args(["--debug", "--dry-run"])
        self.assertFalse(args.load_snapshot)
        args = parser.parse_args(["--debug", "--dry-run", "--load-snapshot"])
        self.assertTrue(args.load_snapshot)

    def test_debug_dry_run_entrypoint_creates_run_dir_and_prints_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                "scripts/run_knshnb.py",
                "--debug",
                "--dry-run",
                "--base-dir",
                str(Path(tmp) / "results"),
                "--timestamp",
                "2026-05-17-010203",
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            lines = completed.stdout.strip().splitlines()
            self.assertEqual(lines[0], str(Path(tmp) / "results" / "2026-05-17-010203-knshnb-debug"))
            self.assertEqual(
                lines[1],
                f"{sys.executable} -m src.train --config_path config/debug.yaml --exp_name debug "
                f"--out_base_dir {(Path(tmp) / 'results' / '2026-05-17-010203-knshnb-debug' / 'predictions').resolve()} "
                "--in_base_dir input",
            )


if __name__ == "__main__":
    unittest.main()
