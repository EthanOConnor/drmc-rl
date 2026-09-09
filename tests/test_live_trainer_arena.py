from types import SimpleNamespace

import pytest

from tools.trainer_planning_arena import next_live_match, variant_policy


def test_live_round_robin_admits_new_checkpoint_and_deepens_all_ready_pairs(tmp_path):
    parent, milestone = tmp_path/"parent.pt", tmp_path/"25m.pt"
    parent.touch()
    config = {"checkpoint":str(parent),"variants":{
        "parent":{},"old":{},"25m":{"adapter_checkpoint":str(milestone)}},
        "schedule":[{"id":"new","a":"25m","b":"parent","games":8},
                    {"id":"old","a":"old","b":"parent","games":8}]}
    assert next_live_match(config,{})["id"] == "old"
    results = {"old":[{}]*4}
    milestone.touch()
    assert next_live_match(config,results)["id"] == "new"
    results["new"] = [{}]*8
    assert next_live_match(config,results)["id"] == "old"
    results["old"] = [{}]*8
    assert next_live_match(config,results) is None


def test_historical_checkpoint_loads_its_own_public_core(monkeypatch):
    calls = []
    historical = SimpleNamespace(aux_dim=72,aux_spec="zero_v1_vs")
    def load(path,device,**kwargs):
        calls.append((str(path),device,kwargs))
        return historical
    monkeypatch.setattr("tools.trainer_planning_arena.PlainPolicy",load)
    parent = object()
    config = {"checkpoint":"current.pt","device":"cpu"}
    assert variant_policy(config,{},parent) is parent
    assert variant_policy(config,{"checkpoint":"old.pt"},parent) is historical
    assert calls == [("old.pt","cpu",{"public_only":True})]
    historical.aux_spec = "v1"
    with pytest.raises(ValueError,match="public auxiliary-input contract"):
        variant_policy(config,{"checkpoint":"privileged.pt"},parent)
