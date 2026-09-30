"""Synthetic, zero-simulator reproduction; not an experiment result."""
import json
from chapter3_bser.online.waypoint_manager import WaypointManager
from chapter3_bser.online.types import SearchAssignment, ExecutorAssignment, OnlineAllocation

def main():
    # Two model routes are assigned different objective values. The switch
    # distance is below the actual configured stabilization threshold (1.0).
    old_route=SearchAssignment(0,'old',(1.,0.,0.),((0.,0.,0.),(1.,0.,0.)),1.)
    new_route=SearchAssignment(0,'new',(1.5,0.,0.),((0.,0.,0.),(1.5,0.,0.)),1.5)
    standby=ExecutorAssignment(3,(0.,0.,0.),((0.,0.,0.),),0.,'standby',True)
    old=OnlineAllocation((old_route,),standby,.1,.2,2.,'old')
    proposed=OnlineAllocation((new_route,),standby,.7,.8,1.,'new')
    result=WaypointManager(tolerance=1.).stabilize(old,proposed,affected_agent_ids=(0,),step=20)
    assert result.search_assignments==old.search_assignments
    assert result.objective_value==proposed.objective_value
    assert result.detection_probability==proposed.detection_probability
    print(json.dumps(dict(synthetic_fixture=True,environment_steps=0,
        retained_route=result.search_assignments[0].candidate_id,
        stored_objective=result.objective_value,old_route_fixture_score=old.objective_value,
        proposal_not_installed=True,
        conclusion='Stabilization can retain an old route but preserve the new proposal score. Real-run incidence and performance impact require a shadow score observer.')))

if __name__=='__main__': main()
