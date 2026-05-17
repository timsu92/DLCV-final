import tempfile
import unittest
from pathlib import Path

from scripts.patch_knshnb import patch_default_yaml, patch_train_py


class PatchKnshnbTests(unittest.TestCase):
    def test_patch_default_yaml_adds_accumulation_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "default.yaml"
            path.write_text("batch_size: 8\nimage_size:\n- 768\n- 768\n", encoding="utf-8")
            patch_default_yaml(path)
            patch_default_yaml(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("accumulate_grad_batches: 1"), 1)

    def test_patch_train_py_adds_trainer_argument_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            path.write_text(
                "    trainer = Trainer(\n"
                "        gpus=torch.cuda.device_count(),\n"
                '        max_epochs=cfg["max_epochs"],\n'
                "        logger=loggers,\n"
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


if __name__ == "__main__":
    unittest.main()
