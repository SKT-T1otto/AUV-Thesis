from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class Guidance:
    def __init__(self, *, reachable=True, hold=False, suffix="a"):
        self.assignments = tuple(
            SimpleNamespace(
                agent_id=index,
                assignment_kind="SEARCH_REGION",
                assignment_id=f"{index}-{suffix}",
                tracking_waypoint=(float(index), 0.0, 0.0),
                final_waypoint=(float(index) + 1.0, 0.0, 0.0),
                reachable=reachable,
                hold_state=hold,
            )
            for index in range(3)
        )

    def assignment_for(self, agent_id):
        return self.assignments[int(agent_id)]


def state(step=0, offset=0.0, known=(True, False), entropy=1.0, peak=0.5):
    return SimpleNamespace(
        step=step,
        occupancy=SimpleNamespace(known_mask=np.asarray(known, dtype=np.bool_)),
        target_belief=SimpleNamespace(entropy=entropy, peak_probability=peak),
        agents=tuple(
            SimpleNamespace(agent_id=index, position=(offset, float(index), 0.0))
            for index in range(4)
        ),
    )


def outputs(alignment=0.25):
    return tuple(
        SimpleNamespace(alignment_cosine=np.asarray([alignment], dtype=np.float32))
        for _ in range(4)
    )
