import copy
import os

import numpy as np
import pytest
import torch

from drmc_rl.human.spatial_proposer import RECURRENT_PUBLIC, SpatialProposer
from drmc_rl.human.target_construction import TargetConstruction, TargetProposal
from drmc_rl.human.target_execution import ConstructionController
from tools.fit_expressive_proposer import session_validation
from tools.fit_target_construction import fit, requested_loss


def inputs():
    return (torch.zeros(1,8,16,8),torch.zeros(1,8),torch.tensor([[0,1]]),torch.tensor([[2,2]]))


def test_target_stays_original_while_route_memory_updates_and_budget_only_decreases():
    torch.manual_seed(29)
    model = TargetConstruction(8, 8).eval()
    plan = TargetProposal.start_requested(model, inputs(), frame=10, goal=0, anchor=120, budget=6)
    order = plan.rank(model, inputs(), [123,124,125])
    first = plan.memory.clone()
    assert plan.rank(model, inputs(), [123,124,125]) == order
    assert torch.equal(first, plan.memory)
    changed = (*inputs()[:3], torch.tensor([[0,0]]))
    plan.rank(model, changed, [123,124,125])
    assert plan.anchor == 120 and plan.remaining == 6
    assert not torch.equal(first, plan.memory)
    with torch.no_grad():
        model.target.bias.fill_(0)
        model.target.bias[383] = 1000
    plan.observe(frame=20, completed_placement=True, observed_goals=[0], cleared_cells=[(15,1,0)])
    plan.observe(frame=20, completed_placement=True)
    plan.rank(model, inputs(), [123,124,125])
    assert plan.anchor == 120 and plan.remaining == 5 and plan.elapsed == 1
    plan.observe(frame=30, completed_placement=True, observed_goals=[0], cleared_cells=[(15,0,0)])
    assert plan.reason == 'spatial_goal_observed'
    assert plan.rank(model, inputs(), [123]) == []
    interrupted = TargetProposal.start_requested(model, inputs(), frame=10, goal=0, anchor=120, budget=6)
    interrupted.observe(frame=11, incoming_garbage=True)
    assert interrupted.reason == 'board_changed' and interrupted.rank(model, inputs(), [123]) == []


def data_fixture():
    sessions = ([f's{i}' for i in range(100) if not session_validation(f's{i}',81029)][:2]+
                [f's{i}' for i in range(100) if session_validation(f's{i}',81029)][:2])
    rng = np.random.default_rng(13)
    targets = np.zeros((4,384),np.float32)
    targets[:,120:124] = .25
    return dict(planes=rng.normal(size=(12,8,16,8)).astype(np.float32),
        features=rng.normal(size=(12,8)).astype(np.float32),canonical_pill=np.zeros((12,2),np.int64),
        canonical_preview=np.ones((12,2),np.int64),feature_index=np.arange(12),
        action=np.array([123,124,125]*4),windows=np.array([[i*3,3,0,i] for i in range(4)]),
        spatial_targets=targets,sessions=np.array(sessions))


def test_hindsight_request_is_fixed_and_does_not_supply_true_future_duration():
    torch.set_num_threads(1)
    model = TargetConstruction(8,8)
    seen = []
    original = model.requested_actions
    def capture(memory,current,goal,elapsed,anchor,remaining):
        seen.append((anchor.tolist(),remaining.tolist()))
        return original(memory,current,goal,elapsed,anchor,remaining)
    model.requested_actions = capture
    requested_loss(model,data_fixture(),np.array([0,1]))[0].sum().backward()
    assert seen == [([120,120,120,121,121,121],[6,5,4,6,5,4])]
    assert model.recurrence.weight_hh.grad is not None


def test_requested_route_fitting_excludes_development_and_reloads_strictly(tmp_path):
    torch.manual_seed(12)
    torch.set_num_threads(1)
    original = TargetConstruction(8,8)
    data = data_fixture()
    saved = []
    for index in range(2):
        out = tmp_path/str(index)
        out.mkdir()
        if index:
            data['features'][6:] += 4
            data['spatial_targets'][2:] = np.roll(data['spatial_targets'][2:],12,axis=1)
            data['action'][6:] = 127
        model = copy.deepcopy(original)
        report = dict(source_sha256='fixture',competitive_sha256='fixture',root_proposer_sha256='fixture')
        fit(data,model,dict(epochs=2,batch_windows=2),out,report)
        checkpoint = torch.load(out/'target-final.pt',weights_only=False)
        loaded = TargetConstruction(checkpoint['feature_dim'],checkpoint['width'])
        loaded.load_state_dict(checkpoint['state_dict'],strict=True)
        assert report['action_presentations'] == 12 and not checkpoint['quality_admission']
        saved.append(checkpoint['state_dict'])
    assert all(torch.equal(v,saved[1][k]) for k,v in saved[0].items())


class ZeroEncoder:
    def __call__(self, boards,pills,previews):
        return boards.new_zeros((len(boards),8))


@pytest.mark.parametrize('control',[False,True])
def test_actual_controller_has_valid_native_tapes_and_shadow_preserves_baseline(control):
    from tests.test_event_rollout import FixedPolicy
    from tools.trainer_event_rollout import ParallelPlanning,run_event_batch
    torch.manual_seed(31)
    torch.set_num_threads(1)
    config = dict(native_library=os.environ.get('DRMC_FRAME_LIBRARY'),
        variants={'a':{'delay':4},'b':{'delay':4}},max_game_frames=3000,replay_games=0)
    match = dict(a='a',b='b',level=14,pace='normal')
    jobs = [(19071,0,0),(19071,1,1)]
    def controller():
        return ConstructionController(TargetConstruction(8,8).eval(),
            SpatialProposer(8,8,persistent=False,plan_update_schema=RECURRENT_PUBLIC).eval(),
            ZeroEncoder(),jobs,control=control)
    planner = ParallelPlanning(2)
    try:
        with pytest.raises(ValueError,match='explicit non-training'):
            run_event_batch(config,match,jobs,FixedPolicy(),planner,None,controller=controller())
        baseline,_ = run_event_batch(config,match,jobs,FixedPolicy(),planner,None)
        actor = controller()
        actual,_ = run_event_batch({**config,'allow_unadmitted_controller_experiment':True},
            match,jobs,FixedPolicy(),planner,None,controller=actor)
        actor.close()
    finally:
        planner.close()
    assert actor.selections and actor.transitions
    assert sum(s['counters']['verified_completions'] for s in actor.states.values()) > 0
    assert all(p['anchor_revisions'] == 0 for p in actor.plans)
    for _, moves, _ in actual:
        for row in moves:
            row.pop('unadmitted_construction',None)
    if control:
        assert any(r['selected_action'] != r['incumbent_action'] for r in actor.selections)
        assert actual != baseline
    else:
        assert actual == baseline
