import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


if "ultralytics" not in sys.modules:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics

work_flows = importlib.import_module("work_flows")


class TrainFlowConfirmationTests(unittest.TestCase):
    def setUp(self):
        work_flows.set_locale(
            {
                "confirm.title": "About to execute: {mode}",
                "confirm.pt_file": "PT: {path}",
                "confirm.data_yaml": "Data: {path}",
                "confirm.exp_name": "Name: {name}",
                "confirm.epochs": "Epochs: {epochs}",
                "confirm.prompt": "confirm",
                "confirm.quit": "quit",
            }
        )

    def test_basic_confirmation_cancels_on_n(self):
        config = types.SimpleNamespace(
            data_yaml="data.yaml",
            experiment_name="exp",
            epochs=1,
        )

        with patch("builtins.input", return_value="n"):
            confirmed = work_flows.ask_confirm_train("new", "model.pt", config)

        self.assertFalse(confirmed)

    def test_basic_confirmation_accepts_y(self):
        config = types.SimpleNamespace(
            data_yaml="data.yaml",
            experiment_name="exp",
            epochs=1,
        )

        with patch("builtins.input", return_value="y"):
            confirmed = work_flows.ask_confirm_train("new", "model.pt", config)

        self.assertTrue(confirmed)

    def test_resume_training_sets_mode_label_before_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "results" / "exp" / "weights"
            exp_dir.mkdir(parents=True)
            (exp_dir / "last.pt").write_text("weights", encoding="utf-8")

            config = types.SimpleNamespace(
                results_dir=str(root / "results"),
                experiment_name="exp",
                last_pt=str(exp_dir / "last.pt"),
                best_pt=str(exp_dir / "best.pt"),
            )
            work_flows.set_locale(
                {
                    "train.resume_mode_label": "Resume training",
                    "log.resume_started": "start",
                    "log.resume_finished": "finish",
                    "log.resume_val": "val",
                }
            )

            with (
                patch.object(work_flows, "_run_confirmation_flow", return_value=(False, False, 0.0)) as confirm,
                patch.object(work_flows, "YOLO"),
            ):
                work_flows.resume_training(config)

        confirm.assert_called_once()
        self.assertEqual(confirm.call_args.args[2], "Resume training")


class LocaleCoverageTests(unittest.TestCase):
    REQUIRED_KEYS = {
        "confirm.quit",
        "mixup.title",
        "mixup.status_on",
        "mixup.status_off",
        "mixup.current",
        "mixup.prompt",
    }

    def test_all_locales_have_keys_used_by_confirmation_flow(self):
        locale_dir = Path(__file__).resolve().parents[1] / "locales"
        missing = {}
        for path in locale_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            missing_keys = self.REQUIRED_KEYS - data.keys()
            if missing_keys:
                missing[path.name] = sorted(missing_keys)

        self.assertEqual(missing, {})


if __name__ == "__main__":
    unittest.main()
