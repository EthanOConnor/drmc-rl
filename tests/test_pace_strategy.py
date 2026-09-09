import numpy as np
import os
import torch
import pytest

from drmc_rl.execution.pace import PACES, strategy_context
from drmc_rl.models.policy.pace_adapter import PaceAdapter, PacePolicy
from tools.train_pace_strategy import (
    add_game_totals, restore_game_journal, terminal_samples,
    training_target_met, update_adapter,
)
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


def test_mixed_scoring_preserves_parent_and_records_only_sampled_learner_rows():
    from types import SimpleNamespace
    torch.manual_seed(91702)
    actor = PacePolicy.__new__(PacePolicy)
    actor.adapter = PaceAdapter(width=8,hidden=8)
    with torch.no_grad():
        actor.adapter.actor[-1].weight.normal_(std=.1)
    actor.training, actor.rng = True, torch.Generator().manual_seed(51)
    actor.core = SimpleNamespace(motor=None,features=None)
    candidates, context = torch.randn(3,3,8), torch.randn(3,8)
    mask = torch.tensor([[True,True,False],[True,True,True],[True,True,True]])
    base = torch.randn(3,3).masked_fill(~mask,-1e9)
    values = torch.randn(3)

    class PackedPolicy:
        def score_and_value(self, observations, _infos):
            index = torch.as_tensor(observations[:,0],dtype=torch.long)
            actor.core.features = (candidates[index],context[index],actor.core.motor,
                                   base[index],values[index],mask[index])
            with torch.no_grad():
                logits, value = actor.adapter(*actor.core.features)
            return (np.broadcast_to(np.arange(3),logits.shape),mask[index].numpy(),
                    logits.numpy().copy(),value.numpy())

    actor.plain = PackedPolicy()
    observations = np.arange(3)[:,None]
    infos = [{"pace/context":np.ones(8,np.float32)} for _ in range(3)]
    _, _, logits = actor.score_mixed(observations,infos,[True,False,True])
    mixed = actor.learning_records
    np.testing.assert_array_equal(logits[1],base[1].numpy())
    assert mixed[1] is None and all(i["pace/context"][-1] == 1 for i in infos)
    saved = mixed[0]["candidate"].copy()
    for i in (0,2):
        record = mixed[i]
        n = len(record["base_logits"])
        inputs = tuple(torch.as_tensor(value)[None] for value in (
            record["candidate"],record["context"],record["motor"],record["base_logits"],
            record["base_value"],np.ones(n,bool)))
        with torch.no_grad():
            actual, value = actor.adapter(*inputs)
        assert actual.log_softmax(-1)[0,record["slot"]].item() == pytest.approx(record["old_logprob"],abs=1e-6)
        assert value.item() == pytest.approx(record["old_value"],abs=1e-6)
        assert logits[i].argmax() == record["action"]
    actor.rng.manual_seed(51)
    actor.score_mixed(observations[[0,2]],[infos[0],infos[2]],[True,True])
    assert [r["action"] for r in actor.learning_records] == [mixed[i]["action"] for i in (0,2)]
    np.testing.assert_array_equal(saved,mixed[0]["candidate"])


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


def test_frame_budget_also_requires_real_learning_coverage_at_every_pace():
    stats = {}
    add_game_totals(stats,{"frames":1000,"reason":"topout","score":0,
        "a_stats":{"decisions":10,"no_reachable_after_delay":7}})
    assert stats["learning_decisions"] == 3 and stats["frames"] == 1000
    config = {"target_frames":1000,"minimum_decisions_per_pace":4,"paces":["sloth","fast"]}
    progress = {"frames":1000,"paces":{"sloth":stats,"fast":{"learning_decisions":4000}}}
    assert not training_target_met(progress,config)
    add_game_totals(stats,{"frames":200,"reason":"clear","score":1,
        "a_stats":{"decisions":1}})
    assert training_target_met(progress,config)
    add_game_totals(stats,{"frames":9999,"reason":"timeout","score":.5,
        "a_stats":{"decisions":100}})
    assert stats["learning_decisions"] == 4
    config["target_decisions"] = 1_000_000
    assert not training_target_met(progress | {"decisions": 999_999}, config)
    assert training_target_met(progress | {"decisions": 1_000_000}, config)
