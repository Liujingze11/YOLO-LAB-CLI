import types
import unittest

from core.lr_schedulers import AdaptiveLRCallback, build_lr_callback


class AdaptiveLRCallbackTests(unittest.TestCase):
    def test_repeated_reductions_scale_from_original_schedule(self):
        callback = AdaptiveLRCallback(patience=1, factor=0.5, min_lr=0.0)
        trainer = types.SimpleNamespace(
            metrics={"metrics/mAP50-95(B)": 0.0},
            lf=lambda epoch: 1.0,
            scheduler=types.SimpleNamespace(lr_lambdas=[lambda epoch: 1.0]),
            optimizer=types.SimpleNamespace(
                param_groups=[{"lr": 1.0, "initial_lr": 1.0}]
            ),
        )

        callback.on_fit_epoch_end(trainer)
        self.assertEqual(trainer.lf(10), 0.5)

        callback.on_fit_epoch_end(trainer)
        self.assertEqual(trainer.lf(10), 0.25)

    def test_standard_cosine_uses_no_custom_callback(self):
        self.assertIsNone(build_lr_callback("cosine"))


if __name__ == "__main__":
    unittest.main()
