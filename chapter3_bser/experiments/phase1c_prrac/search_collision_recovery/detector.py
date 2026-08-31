"""Collision-edge detector with one collision-free SEARCH transition re-arm."""

from __future__ import annotations


class CollisionEdgeDetector:
    def __init__(self) -> None:
        self._armed = {0: True, 1: True, 2: True}
        self._streak = {0: 0, 1: 0, 2: 0}

    def observe(self, agent_id: int, collision: bool, *, search_active: bool) -> bool:
        agent_id = int(agent_id)
        if agent_id not in self._armed:
            raise ValueError("search collision recovery supports only agents 0, 1, and 2")
        if not search_active:
            self._armed[agent_id] = True
            self._streak[agent_id] = 0
            return False
        if collision:
            self._streak[agent_id] += 1
            edge = self._armed[agent_id]
            self._armed[agent_id] = False
            return edge
        self._streak[agent_id] = 0
        self._armed[agent_id] = True
        return False

    def streak(self, agent_id: int) -> int:
        return int(self._streak[int(agent_id)])


__all__ = ("CollisionEdgeDetector",)
