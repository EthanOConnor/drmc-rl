import numpy as np
import os
import torch
import pytest

from drmc_rl.execution.pace import PACES, strategy_context
from drmc_rl.models.policy.pace_adapter import PaceAdapter
from tools.train_pace_strategy import restore_game_journal, terminal_samples, update_adapter
from tools.trainer_planning_arena import paired_jobs, run_batch


def features():
    torch.manual_seed(91700)
    mask = torch.tensor([[True,True,False],[True,True,True]])
    return (torch.randn(2,3,8),torch.randn(2,8),torch.tensor([[1.,1,1,1,1,1,1,1],[0.,0,0,1,1,1,0,0]]),
            torch.randn(2,3).masked_fill(~mask,-1e9),torch.tensor([.2,-.4]),mask)


def test_zero_adapter_and_disabled_fast_route_preserve_parent_exactly():
    model = PaceAdapter(width=8,hidden=8)
    x = features()
    logits, value = model(*x)
    assert torch.equal(logits,x[3]) and torch.equal(value,x[4])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    logits, value = model(*x)
    assert torch.equal(logits[1],x[3][1]) and torch.equal(value[1],x[4][1])
    assert not torch.equal(logits[0,:2],x[3][0,:2])
    assert logits[0,2] == -1e9


def test_motor_context_uses_only_public_limits_and_preserves_fast_gate():
    state = {"speed":2,"speed_ups":3}
    for pace in PACES:
        context = strategy_context(pace,state,max(4,pace.reaction_frames))
        assert np.isfinite(context).all()
        np.testing.assert_array_equal(context,strategy_context(pace,{**state,"rng":919,"pending_attack":25},max(4,pace.reaction_frames)))
        assert bool(context[-1]) == (pace.id not in ("super_human","frame_perfect"))


def test_terminal_credit_excludes_caps_and_weights_whole_games():
    def game(score,n,reason="clear"):
        return ({"score":score,"reason":reason},[{"learning":{"turn":i}} for i in range(n)],[])
    rows = terminal_samples([game(1,2),game(0,4),game(.5,3),game(.5,10,"timeout")])
    assert len(rows)==9
    assert sum(r["weight"] for r in rows if r["return"]==1)==1
    assert sum(r["weight"] for r in rows if r["return"]==-1)==1
    assert sum(r["weight"] for r in rows if r["return"]==0)==1


def test_outcome_update_is_finite_and_changes_only_adapter():
    from types import SimpleNamespace
    model = PaceAdapter(width=8,hidden=8)
    x = features()
    rows=[]
    for slot,outcome in [(0,1),(1,-1)]:
        rows.append({"candidate":x[0][0,:2].numpy(),"context":x[1][0].numpy(),"motor":x[2][0].numpy(),
            "base_logits":x[3][0,:2].numpy(),"base_value":.2,"slot":slot,
            "old_logprob":x[3][0,:2].log_softmax(-1)[slot].item(),"old_value":.2,"return":outcome,"weight":1.})
    original=model.actor[-1].weight.detach().clone()
    result=update_adapter(SimpleNamespace(adapter=model,device="cpu"),torch.optim.AdamW(model.parameters(),lr=.001),rows,{"minibatch":2,"epochs":2},5)
    assert all(np.isfinite(v) for v in result.values())
    assert not torch.equal(original,model.actor[-1].weight)


def test_explicit_evaluation_bank_stays_disjoint_and_side_balanced():
    match = {"games":4,"seeds":[917,3410]}
    assert paired_jobs({"seed_exclusions":[2,3]},match) == [(917,0,0),(917,1,1),(3410,0,2),(3410,1,3)]
    with pytest.raises(ValueError):
        paired_jobs({"seed_exclusions":[917]},match)
    with pytest.raises(ValueError):
        paired_jobs({}, {"games":4,"seeds":[917,917]})


def test_controller_rollout_distinguishes_physical_locks_from_planner_failures():
    class BrokenPlanner:
        def bfs_full(self, *_args, **_kwargs):
            raise RuntimeError("planner worker failed")
    config = {"native_library":os.environ.get("DRMC_FRAME_LIBRARY"),
        "variants":{"a":{"delay":4},"b":{"delay":4}},"max_game_frames":4,"replay_games":0}
    match = {"a":"a","b":"b","games":2,"level":20,"pace":"sloth"}
    batch,_ = run_batch(config,match,[(19071,0,0)],None,BrokenPlanner(),None)
    assert batch[0][0]["a_stats"]["no_reachable_after_delay"] == 1
    with pytest.raises(RuntimeError,match="planner worker failed"):
        run_batch(config,{**match,"level":14,"pace":"normal"},[(19071,0,0)],None,BrokenPlanner(),None)


def test_resume_keeps_only_games_in_the_optimizer_checkpoint(tmp_path):
    import json
    path = tmp_path/"training-games.jsonl"
    path.write_text('{"update":1,"score":1}\n{"update":2,"score":0}\n{"update":3')
    restore_game_journal(path,1)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"update":1,"score":1}]
    path.write_text('{"update":1}\ncorrupt completed row\n')
    with pytest.raises(json.JSONDecodeError):
        restore_game_journal(path,1)
