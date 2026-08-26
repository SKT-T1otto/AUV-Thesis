from dataclasses import replace
import numpy as np
import unittest
from chapter3_bser.objective import response_diagnostics
from tests.bser_test_utils import synthetic_instance

class ResponseUnreachableMassTest(unittest.TestCase):
    def test_unreachable_detected_mass_is_explicit(self):
        _,_,generated,context=synthetic_instance(); standby=generated.standby_candidates[0]; times=np.asarray(context.response_time_by_id[standby.candidate_id]).copy(); times[0]=np.inf; mapping=dict(context.response_time_by_id); mapping[standby.candidate_id]=times; changed=replace(context,response_time_by_id=mapping); diag=response_diagnostics(generated.search_candidates[:3],standby,changed); self.assertGreater(diag.unreachable_detected_mass,0.0)
