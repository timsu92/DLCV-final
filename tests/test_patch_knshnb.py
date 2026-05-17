import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.patch_knshnb import main, patch_default_yaml, patch_train_py


class PatchKnshnbTests(unittest.TestCase):
    def test_patch_default_yaml_adds_accumulation_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "default.yaml"
            path.write_text("batch_size: 8\nimage_size:\n- 768\n- 768\n", encoding="utf-8")
            patch_default_yaml(path)
            patch_default_yaml(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("accumulate_grad_batches: 1"), 1)

    def test_patch_default_yaml_ignores_commented_or_nested_occurrences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "default.yaml"
            path.write_text(
                "batch_size: 8\n"
                "# accumulate_grad_batches: 4\n"
                "trainer:\n"
                "  accumulate_grad_batches: 2\n",
                encoding="utf-8",
            )
            patch_default_yaml(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("accumulate_grad_batches: 1"), 1)
            self.assertIn("# accumulate_grad_batches: 4", text)
            self.assertIn("  accumulate_grad_batches: 2", text)

    def test_patch_train_py_patches_realistic_trainer_snippet_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            path.write_text(
                "    trainer = Trainer(\n"
                "        gpus=torch.cuda.device_count(),\n"
                '        max_epochs=cfg["max_epochs"],\n'
                "        logger=loggers,\n"
                "        callbacks=callbacks,\n"
                "        checkpoint_callback=args.save_checkpoint,\n"
                "    )\n",
                encoding="utf-8",
            )
            patch_train_py(path)
            patch_train_py(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(
                text.count('accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),'),
                1,
            )

    def test_patch_train_py_unrelated_occurrence_does_not_count_as_patched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            path.write_text(
                "# accumulate_grad_batches=cfg.get(\"accumulate_grad_batches\", 1),\n"
                "    trainer = Trainer(\n"
                "        gpus=torch.cuda.device_count(),\n"
                '        max_epochs=cfg["max_epochs"],\n'
                "        logger=loggers,\n"
                "        callbacks=callbacks,\n"
                "    )\n",
                encoding="utf-8",
            )
            patch_train_py(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(
                text.count('accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),'),
                2,
            )
            self.assertIn(
                "        logger=loggers,\n"
                '        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),\n'
                "        callbacks=callbacks,\n",
                text,
            )

    def test_patch_train_py_missing_anchor_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            path.write_text(
                "    trainer = Trainer(\n"
                "        gpus=torch.cuda.device_count(),\n"
                '        max_epochs=cfg["max_epochs"],\n'
                "    )\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "trainer = Trainer"):
                patch_train_py(path)

    def test_main_keeps_existing_patch_note_on_noop_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "repo"
            note = tmp_path / "patch-knshnb-gradient-accumulation.md"
            (repo / "config").mkdir(parents=True)
            (repo / "src").mkdir(parents=True)
            (repo / "config" / "default.yaml").write_text(
                "batch_size: 8\naccumulate_grad_batches: 1\n",
                encoding="utf-8",
            )
            (repo / "src" / "train.py").write_text(
                "    trainer = Trainer(\n"
                "        gpus=torch.cuda.device_count(),\n"
                '        max_epochs=cfg["max_epochs"],\n'
                "        logger=loggers,\n"
                '        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),\n'
                "        callbacks=callbacks,\n"
                "    )\n",
                encoding="utf-8",
            )
            original_note = "# patch: knshnb gradient accumulation\n\nRecorded at: 2026-05-17T13:27:00\n"
            note.write_text(original_note, encoding="utf-8")

            with mock.patch(
                "sys.argv",
                ["patch_knshnb.py", "--repo", str(repo), "--note", str(note)],
            ):
                main()

            self.assertEqual(note.read_text(encoding="utf-8"), original_note)


if __name__ == "__main__":
    unittest.main()
