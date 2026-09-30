"""Fixed development arms. No training or simulator configuration overrides."""
from dataclasses import asdict, dataclass


PARENTS = {"B0_search_prior": "V5", "B1_bser_prior": "V4"}
ARMS = {
    "R0": dict(latched_hold=False, braking=False, recovery=False),
    "R1": dict(latched_hold=True, braking=False, recovery=False),
    "R2": dict(latched_hold=True, braking=True, recovery=False),
    "R3": dict(latched_hold=True, braking=False, recovery=True),
    "R4": dict(latched_hold=True, braking=True, recovery=True),
}


@dataclass(frozen=True)
class Settings:
    # Navigation intent only: these values never alter physical limits/gains.
    tracking_threshold: float = 0.25
    hold_velocity_feedback: float = 0.8
    prediction_steps: int = 20
    disturbance_limit: float = 0.5
    disturbance_uncertainty: float = 0.08
    corner_speed: float = 0.65
    corner_deceleration: float = 0.4
    corner_angle_degrees: float = 35.0
    recovery_retry_steps: int = 10
    stall_window_steps: int = 20
    stall_distance: float = 0.15

    def record(self):
        return asdict(self)


SETTINGS = Settings()


def parse_arms(value):
    selected = value.split(",")
    if not selected or len(set(selected)) != len(selected) or any(x not in ARMS for x in selected):
        raise ValueError("arms must be unique comma-separated R0,R1,R2,R3,R4")
    if "R0" not in selected:
        raise ValueError("include R0 for an on-machine paired reference")
    return selected
