import unittest

from scripts.run_ensemble import build_ensemble_command


class RunEnsembleTests(unittest.TestCase):
    def test_build_ensemble_command(self):
        command = build_ensemble_command(
            model_dirs=["/project/results/b6/predictions/b6/-1", "/project/results/b7/predictions/b7/-1"],
            out_prefix="/project/results/ensemble/submissions/b6-b7",
        )
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "src.ensemble",
                "--model_dirs",
                "/project/results/b6/predictions/b6/-1",
                "/project/results/b7/predictions/b7/-1",
                "--out_prefix",
                "/project/results/ensemble/submissions/b6-b7",
            ],
        )


if __name__ == "__main__":
    unittest.main()
