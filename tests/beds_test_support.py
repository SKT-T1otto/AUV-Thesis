"""Public navigation fixtures shared by BEDS tests; no TestCase classes."""
from chapter3_bser.online.safe_executor_standby import SafeStandbyNavigation, StandbyNavigationParameters

def parameters():
    # Fixture values match existing runtime semantics; production reads runtime.
    return StandbyNavigationParameters(10, .75, .75, 1.6, 2.4, 1.2, 1.0)


def navigation(state, **kwargs):
    return SafeStandbyNavigation(parameters(), state_factory=lambda: state,
        segment_clear=kwargs.pop('segment_clear', lambda a, b: True), **kwargs)

