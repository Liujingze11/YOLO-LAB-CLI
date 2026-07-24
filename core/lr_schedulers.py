"""Custom LR scheduler callbacks for Ultralytics training.

Key insight: Ultralytics uses ``LambdaLR`` with ``scheduler.step()`` called
at the *start* of every epoch (before the training loop, line 420 in trainer.py).
Any manual ``optimizer.param_groups[i]['lr']`` modification in a callback at
epoch end will be **overwritten** by the scheduler step of the next epoch.

Therefore these callbacks replace ``trainer.lf`` (the lambda function used by
the scheduler) rather than touching the optimizer directly.
"""

import math


def _cosine_lf(steps: int, lrf: float = 0.01):
    """Return a cosine lambda: 1.0 → lrf over `steps` epochs.

    Matches Ultralytics ``one_cycle(1.0, lrf, steps)`` exactly.
    """
    def lf(x):
        return max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (lrf - 1.0) + 1.0
    return lf


class WarmRestartCallback:
    """Replace the scheduler's lambda so the cosine cycle restarts every `period` epochs.

    Must be registered via ``model.add_callback("on_fit_epoch_end", ...)``.
    The LR is affected from the *next* epoch onward because the scheduler steps
    at epoch start.
    """

    def __init__(self, period: int = 50, lrf: float = 0.01):
        self.period = period
        self.lrf = lrf
        self._applied = False  # only patch once

    def on_fit_epoch_end(self, trainer):
        """Replace trainer.lf with a periodic cosine lambda."""
        if self._applied:
            return
        self._applied = True

        base_lf = _cosine_lf(self.period, self.lrf)

        def periodic_lf(x, period=self.period, base=base_lf):
            return base(x % period)

        trainer.lf = periodic_lf
        # Also update the scheduler's internal lambda list
        n_groups = len(trainer.scheduler.lr_lambdas)
        trainer.scheduler.lr_lambdas = [periodic_lf] * n_groups

        print(
            f"\n[WarmRestart] Scheduler patched: cosine period={self.period}, "
            f"lrf={self.lrf}"
        )

    def __repr__(self):
        return f"WarmRestart(period={self.period}, lrf={self.lrf})"


class AdaptiveLRCallback:
    """Reduce LR by `factor` when mAP50-95(B) stalls for `patience` epochs.

    Since the scheduler overwrites LR every epoch, this callback works by
    scaling the trainer's optimizer param groups *and* adjusting the scheduler's
    base_lr so future steps respect the reduction.
    """

    def __init__(self, patience: int = 10, factor: float = 0.5, min_lr: float = 1e-7):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self._best_map = 0.0
        self._bad_epochs = 0
        self._scale = 1.0  # cumulative scale factor
        self._base_lf = None

    def on_fit_epoch_end(self, trainer):
        """Check mAP50-95(B) and scale down LR if stalled."""
        metrics = getattr(trainer, "metrics", {}) or {}
        current_map = metrics.get("metrics/mAP50-95(B)", 0.0)

        if current_map > self._best_map + 0.0005:
            self._best_map = current_map
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1

        if self._bad_epochs >= self.patience:
            self._scale *= self.factor
            old_lrs = []
            for pg in trainer.optimizer.param_groups:
                old_lrs.append(pg["lr"])
                pg["lr"] = max(pg["lr"] * self.factor, self.min_lr)
            # Also patch the scheduler's lambda to respect the new base
            n_groups = len(trainer.scheduler.lr_lambdas)
            if self._base_lf is None:
                self._base_lf = trainer.lf
            base_lf = self._base_lf
            scale = self._scale
            initial_lr = trainer.optimizer.param_groups[0].get("initial_lr") or old_lrs[0] or 1e-4
            min_factor = self.min_lr / initial_lr

            def scaled_lf(x, s=scale, b=base_lf, minimum=min_factor):
                return max(b(x) * s, minimum)

            trainer.lf = scaled_lf
            trainer.scheduler.lr_lambdas = [scaled_lf] * n_groups

            new_lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
            print(
                f"\n[AdaptiveLR] mAP stalled {self.patience} epochs "
                f"(best={self._best_map:.4f}), "
                f"LR: {old_lrs[0]:.2e} → {new_lrs[0]:.2e}"
            )
            self._bad_epochs = 0

    def __repr__(self):
        return f"AdaptiveLR(patience={self.patience}, factor={self.factor}, min_lr={self.min_lr})"


def build_lr_callback(scheduler_type: str, **kwargs) -> object | None:
    """Factory: return a callback instance or None (for standard cosine)."""
    if scheduler_type == "adaptive":
        return AdaptiveLRCallback(
            patience=kwargs.get("patience", 10),
            factor=kwargs.get("factor", 0.5),
            min_lr=kwargs.get("min_lr", 1e-7),
        )
    elif scheduler_type == "restart":
        return WarmRestartCallback(
            period=kwargs.get("period", 50),
            lrf=kwargs.get("lrf", 0.01),
        )
    else:  # 'cosine' (standard)
        return None
