"""Frozen 153D, float64 zero/ridge predictor for the opt-in first phase."""
import math
import numpy as np
import torch
from torch import nn
from .phase1 import PREDICTOR_REVISION


class StablePredictor(nn.Module):
    def __init__(self, input_dim, options):
        super().__init__()
        if input_dim != 153 or options["predictor_revision"] != PREDICTOR_REVISION:
            raise ValueError("stable predictor revision/dimension mismatch")
        self.input_dim = input_dim
        self.mode = options["predictor_mode"]
        self.alpha = float(options["ridge_alpha"])
        self.kappa = float(options["scale_kappa"])
        self.reference_scale = float(options["reference_scale"])
        self.shrink = float(options["lambda"])
        if (self.mode not in ("zero", "ridge") or not 0 <= self.shrink <= 1
                or any(not math.isfinite(x) or x <= 0 for x in (self.alpha, self.kappa, self.reference_scale))):
            raise ValueError("invalid stable predictor parameters")
        for name, value in (("center", torch.zeros(input_dim, dtype=torch.float64)),
                            ("scale", torch.full((input_dim,), self.reference_scale, dtype=torch.float64)),
                            ("coefficient", torch.zeros(input_dim, dtype=torch.float64)),
                            ("intercept", torch.tensor(0., dtype=torch.float64)),
                            ("fitted", torch.tensor(False))):
            self.register_buffer(name, value)

    @torch.no_grad()
    def reset_fit(self):
        self.center.zero_()
        self.scale.fill_(self.reference_scale)
        self.coefficient.zero_()
        self.intercept.zero_()
        self.fitted.fill_(False)

    def _features(self, features, n=None):
        x = torch.as_tensor(np.asarray(features), dtype=torch.float64)
        if n == 0 and x.numel() == 0:
            return torch.empty((0, self.input_dim), dtype=torch.float64)
        expected = (self.input_dim,) if n is None else (n, self.input_dim)
        if tuple(x.shape) != expected or not bool(torch.isfinite(x).all()):
            raise ValueError("invalid raw predictor features")
        return x

    @torch.no_grad()
    def fit(self, features, labels, *, dataset_ids=()):
        self.reset_fit()
        n = len(labels)
        x = self._features(features, n)
        y = torch.as_tensor(labels, dtype=torch.float64)
        if y.shape != (n,) or not bool(torch.isfinite(y).all()):
            raise ValueError("invalid raw predictor labels")
        reason = "requested_zero" if self.mode == "zero" else "insufficient_samples" if n < 2 else "all_zero_labels" if bool((y == 0).all()) else None
        mse = None
        if reason is None:
            try:
                self.center.copy_(x.mean(0))
                variance = x.var(0, unbiased=False)
                self.scale.copy_(((n * variance + self.kappa * self.reference_scale**2) /
                                  (n + self.kappa)).sqrt())
                z = (x - self.center) / self.scale
                self.intercept.copy_(y.mean())
                # Dual ridge solve, avoids a 153x153 inverse at tiny n.
                dual = torch.linalg.solve(z @ z.T + self.alpha * torch.eye(n, dtype=torch.float64),
                                          y - self.intercept)
                self.coefficient.copy_(z.T @ dual)
                if not all(bool(torch.isfinite(v).all()) for v in self.buffers()) or not bool((self.scale > 0).all()):
                    raise ArithmeticError("nonfinite ridge solution")
                self.fitted.fill_(True)
                mse = float(((z @ self.coefficient + self.intercept - y)**2).mean())
                if not math.isfinite(mse):
                    raise ArithmeticError("nonfinite fitted loss")
            except (RuntimeError, ArithmeticError, OverflowError):
                self.reset_fit()
                reason = "numerical_solve_failure"
                mse = None
        return self.report(n=n, dataset_ids=dataset_ids, reason=reason, mse=mse)

    def report(self, *, n=0, dataset_ids=(), reason=None, mse=None):
        return dict(requested_mode=self.mode, effective_mode="ridge" if bool(self.fitted) else "zero",
                    predictor_revision=PREDICTOR_REVISION, samples=n, dataset_ids=list(dataset_ids),
                    updates=int(bool(self.fitted)), lambda_value=self.shrink,
                    ridge_alpha=self.alpha, scale_kappa=self.kappa, reference_scale=self.reference_scale,
                    reference_scale_meaning="fixed_numerical_reference_not_physical_bound",
                    scale_min=float(self.scale.min()), scale_max=float(self.scale.max()),
                    fallback_reason=reason, mse=mse, mse_scope="pilot_training_only",
                    independent_error=None)

    @torch.no_grad()
    def forward(self, features):
        x = self._features(features)
        value = (((x - self.center) / self.scale) @ self.coefficient + self.intercept
                 if bool(self.fitted) else torch.tensor(0., dtype=torch.float64))
        value = self.shrink * value
        if not bool(torch.isfinite(value)):
            raise ArithmeticError("nonfinite prediction")
        return value

    @torch.no_grad()
    def freeze_predictions(self, trajectories):
        # Validate raw inputs before the numerical-fallback region.
        for t in trajectories:
            if t["tau"] is not None:
                self._features(t["features"])
        try:
            values = tuple(0. if t["tau"] is None else float(self(t["features"])) for t in trajectories)
            return values, None
        except (ArithmeticError, RuntimeError, OverflowError):
            self.reset_fit()
            return (0.,) * len(trajectories), "nonfinite_prediction_batch"
