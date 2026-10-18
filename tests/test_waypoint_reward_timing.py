import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from chapter3_bser.experiments.d2_repair_v2.runtime import RepairedMissionRuntime
from tests.d2_v2_test_utils import config
from tests.d2_v2_scenarios import scenario

class Zero:
    def step(self,observations,explore=False): return [torch.zeros(1,3) for _ in range(4)]

def trace(kind,enabled,arrive):
    c=config(); c['max_steps']=4
    c['repairs']=dict(waypoint_arrival=enabled,inactive_actor=False,hold_residual=False)
    c.pop('waypoint_reward_timing')
    rt=RepairedMissionRuntime(c,scenario(kind,4),seed=31729)
    rows=[]
    try:
        if arrive: rt.env.unwrapped._search_waypoints[0].copy_(rt.env.unwrapped._agent_pos[0])
        while not rt.terminal:
            r=rt.advance(Zero(),explore=False); u=rt.env.unwrapped
            rows.append(dict(obs=r['next_observations'],reward=r['rewards'],count=int(u.waypoint_reached_counts[0]),
                found=u.task_found,components={k:v.clone() for k,v in u.last_reward_components.items()},
                map_revision=rt.env.get_mapping_state().map_revision,
                paths=copy.deepcopy(u._navigation_paths),bonus=float(u._waypoint_bonus[0])))
    finally:rt.close()
    return rows

class WaypointTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads,old); torch.set_num_threads(1)

    def test_real_arrival_once_without_other_reward_or_observation_changes(self):
        old,new=trace('normal',False,True),trace('normal',True,True)
        self.assertEqual(new[0]['count'],1); self.assertEqual(new[1]['count'],1)
        for t,(a,b) in enumerate(zip(old,new)):
            np.testing.assert_array_equal(a['obs'],b['obs'])
            self.assertEqual(a['map_revision'],b['map_revision'])
            for name,value in a['components'].items():
                expected=torch.tensor([b['bonus'],0,0,0]) if name=='reward_coverage' and t==0 else torch.zeros(4)
                torch.testing.assert_close(b['components'][name]-value,expected,rtol=0,atol=0)

    def test_no_arrival_and_map_replanning_preserve_all_components(self):
        old,new=trace('normal',False,False),trace('normal',True,False)
        self.assertGreater(len({r['map_revision'] for r in new}),1)
        for a,b in zip(old,new):
            np.testing.assert_array_equal(a['obs'],b['obs'])
            for name,value in a['components'].items(): torch.testing.assert_close(b['components'][name],value,rtol=0,atol=0)
            self.assertEqual(a['map_revision'],b['map_revision'])

    def test_discovery_precedes_waypoint_and_no_active_search_waypoint_after_found(self):
        old,new=trace('near',False,True),trace('near',True,True)
        self.assertTrue(new[0]['found'])
        for a,b in zip(old,new):
            self.assertEqual(b['count'],0)
            self.assertTrue(all(not path for path in b['paths'][:3]))
            for name,value in a['components'].items(): torch.testing.assert_close(b['components'][name],value,rtol=0,atol=0)

    def test_collision_wins_over_arrival_and_keeps_failure_reward(self):
        rows=trace('collision',True,True)
        self.assertEqual(len(rows),1)
        np.testing.assert_array_equal(rows[0]['reward'],[-2.]*4)
        self.assertEqual(rows[0]['count'],0)
