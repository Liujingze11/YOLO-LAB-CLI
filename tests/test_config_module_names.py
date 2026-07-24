import importlib
import unittest
from pathlib import Path


class ConfigModuleNameTests(unittest.TestCase):
    def test_cli_path_defaults_live_in_cli_config(self):
        cli_config = importlib.import_module("cli_config")

        self.assertTrue(hasattr(cli_config, "DATA_YAML"))
        self.assertTrue(hasattr(cli_config, "MODEL_FILE"))
        self.assertTrue(hasattr(cli_config, "RESULTS_DIR"))
        self.assertTrue(hasattr(cli_config, "LOG_DIR"))
        self.assertTrue(hasattr(cli_config, "PREDICT_DIR"))
        self.assertTrue(hasattr(cli_config, "BEST_SEG_MODEL"))
        self.assertTrue(hasattr(cli_config, "TEST_IMAGES_DIR"))

    def test_train_config_lives_in_core_train_config(self):
        train_config = importlib.import_module("core.train_config")

        self.assertTrue(hasattr(train_config, "TrainConfig"))
        self.assertEqual(train_config.TrainConfig().lr_scheduler, "cosine")

    def test_old_ambiguous_config_modules_are_removed(self):
        project_root = Path(__file__).resolve().parents[1]

        self.assertFalse((project_root / "config.py").exists())
        self.assertFalse((project_root / "core" / "config.py").exists())

    def test_training_flows_module_has_clear_name(self):
        project_root = Path(__file__).resolve().parents[1]

        self.assertTrue((project_root / "training_flows.py").exists())
        self.assertFalse((project_root / "work_flows.py").exists())


if __name__ == "__main__":
    unittest.main()
