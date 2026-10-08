"""Explicit learner placement while preserving the original CPU exploration RNG."""
from core.algorithms.noise import OUNoise
from chapter3_bser.experiments.baselines.common.model import build_model as original_build_model


class CpuExplorationNoise(OUNoise):
    """Keep the original OU recurrence and draws on CPU across policy moves.

    The core agent already copies sampled noise to its action device. Letting
    prep_rollouts move OU state to CUDA would change the random event stream.
    """
    def to(self, device=None, dtype=None):
        return super().to(device="cpu", dtype=dtype)


def build_model(config, env=None, device="cpu"):
    model = original_build_model(config, env=env, device="cpu")
    if device == "cuda":
        from .options import require_device
        require_device(dict(learner_device=device))
        for agent in model.agents:
            old = agent.noise
            noise = CpuExplorationNoise(old.action_dimension, mu=old.mu, theta=old.theta,
                sigma=old.sigma, scale=old.scale, device="cpu", dtype=old.dtype)
            noise.state = old.state.detach().cpu().clone()
            agent.noise = noise
        model.prep_training(device=device)
    elif device != "cpu":
        raise ValueError("performance model device must explicitly be cpu or cuda")
    return model
