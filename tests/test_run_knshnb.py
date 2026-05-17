import unittest

from scripts.run_knshnb import build_train_command


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


if __name__ == "__main__":
    unittest.main()
