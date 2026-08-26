from dataclasses import replace
import numpy as np
import unittes
from chapter3_bser.objective import cell_detection_probability,response_diagnostics
from tests.bser_test_utils import synthetic_instance

class ResponseDenominatorTest(unittest.TestCase):
    def test_conditional_time_divides_by_reachable_mass(self):
        _,_,generated,context=synthetic_instance(); selected=generated.search_candidates[:3]; standby=generated.standby_candidates[0]; times=np.asarray(context.response_time_by_id[standby.candidate_id]).copy(); times[0]=np.inf; mapping=dict(context.response_time_by_id); mapping[standby.candidate_id]=times; changed=replace(context,response_time_by_id=mapping); mass=context.belief*cell_detection_probability(selected,context); finite=np.isfinite(times); expected=float(np.sum(mass[finite]*times[finite]))/(float(np.sum(mass[finite]))+context.epsilon); self.assertAlmostEqual(response_diagnostics(selected,standby,changed).conditional_reachable_response_time,expected,places=12)
