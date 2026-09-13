from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from drmc_rl.training.controller_retention import (
    PaceRetention, RetentionRecorder, balance_pace_credit, save_anchor_bank, select_game_anchors,
)
from tests.test_controller_core_training import parent, controller_requests
from drmc_rl.models.policy.controller_core import ControllerCorePolicy
from drmc_rl.models.policy.pace_adapter import PacePolicy
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_planning_arena import run_batch
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.train_pace_strategy import prepare_training_records, update_adapter
from tools.vs_head_to_head import PlainPolicy


def test_retention_pressure_uses_pace_relative_budget_without_relaxing_guard():
    r=PaceRetention.__new__(PaceRetention)
    r.paces=('slow','fast'); r.baseline={'slow':.4,'fast':0.}
    r.max_kl_increase=.03; r.pressure_strength=31.
    r.set_pressure({'slow':.3,'fast':.03})
    assert r.pressure=={'slow':1.,'fast':32.}
    assert r.accepts({'slow':.4,'fast':.03})
    assert not r.accepts({'slow':.4,'fast':.030001})
    r.set_pressure({'slow':.415,'fast':0.})
    assert r.pressure['slow']==pytest.approx(8.75)
    assert r.pressure['fast']==1.


def test_mixed_and_cycling_arms_have_identical_per_pace_game_schedules():
    from tools.train_controller_retention import collection_schedule
    from drmc_rl.training.public_league import PublicOpponentPool
    opponents=PublicOpponentPool([dict(id='a',weight=.4),dict(id='b',weight=.6)],None,'unused','cpu')
    config=dict(arm='mixed_retention',paces=['sloth','normal','frame_perfect'],seed=700,
                games_per_pace={'sloth':8},games_per_update=4)
    for cycle in (1,2,100):
        mixed=collection_schedule(config,cycle,np.arange(1,100),opponents)
        cycling=[collection_schedule(dict(config,arm='cycling_control'),(cycle-1)*3+i+1,
                                     np.arange(1,100),opponents)[0] for i in range(3)]
        for (a,ja),(b,jb) in zip(mixed,cycling,strict=True):
            assert ja==jb
            assert {k:v for k,v in a.items() if k!='id'}=={k:v for k,v in b.items() if k!='id'}


def test_registered_anchor_bank_recovers_without_changing_frozen_teacher_games(parent,tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    encoder=ControllerCorePolicy(parent,training=False)
    initialized=tmp_path/'initialized.pt'; encoder.save(initialized,update=0)
    config=dict(checkpoint=str(parent),encoder_checkpoint=str(initialized),device='cpu',threads=1,
        opponent_pool=[dict(id='parent',weight=1.,checkpoint=str(parent))],
        references={'normal':{},'frame_perfect':{'checkpoint':str(initialized)}},
        paces=['normal','frame_perfect'],seeds={'normal':[17291,39577],'frame_perfect':[19071,20655]},
        holdout_seeds=[50000],output=str(tmp_path/'bank'),seed=841,pairs=2,
        minimum_seeds_per_pace=1,rows_per_game=3,planner_workers=1,max_game_frames=120000,
        native_library=os.environ.get('DRMC_FRAME_LIBRARY'))
    path=tmp_path/'bank-config.json'; path.write_text(json.dumps(config))
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
    command=[sys.executable,'-m','tools.program','launch','trainer-controller-retention-bank',
             '--set','controller_retention_bank_config='+str(path)]
    first=subprocess.run(command,text=True,capture_output=True,timeout=60,env=env)
    assert first.returncode==0,first.stdout+first.stderr
    before=json.loads((tmp_path/'bank/progress.json').read_text())
    assert before['status']=='Complete' and before['games']==8 and before['optimizer_updates']==0
    assert before['anchor_rows']>0
    bank=torch.load(tmp_path/'bank/normal.pt',map_location='cpu',weights_only=True)
    assert all(r['anchor_only'] and 'old_logprob' not in r and 'return' not in r for r in bank['records'])
    second=subprocess.run(command,text=True,capture_output=True,timeout=60,env=env)
    assert second.returncode==0,second.stdout+second.stderr
    after=json.loads((tmp_path/'bank/progress.json').read_text())
    assert all(after[k]==before[k] for k in ('frames','games','anchor_rows','conditions'))


def test_registered_retention_training_runs_a_natural_update_and_recovers(parent,tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    import hashlib
    actor=ControllerCorePolicy(parent,seed=500)
    initialized=tmp_path/'initialized.pt'; actor.save(initialized,update=0)
    obs,infos=controller_requests(actor); actor.score(obs,infos)
    rows=[actor.learning_records[i] for i in (1,3)]
    paces=['normal','frame_perfect']
    for i,r in enumerate(rows): r.update(pace=paces[i],game_seed=50000+i,learner_port=0,anchor_only=True)
    bank=tmp_path/'anchors.pt'; save_anchor_bank(bank,rows,{})
    cfg=dict(arm='mixed_retention',checkpoint=str(initialized),opponent_parent=str(parent),
        opponent_pool=[dict(id='parent',weight=1.,checkpoint=str(parent))],device='cpu',threads=1,
        seed=317,paces=paces,holdout_seeds=[40000],games_per_pace={},games_per_update=2,
        output=str(tmp_path/'training'),anchor_banks=[str(bank)],lr=3e-6,epochs=1,minibatch=32,
        retention_batch_size=4,max_update_kl=.03,max_anchor_kl_increase=.03,
        retention_pressure_strength=31.,reset_update_lr=True,minimum_learning_rate=1e-8,
        target_decisions=1,minimum_decisions_per_pace=0,updates=2,rollout_games=2,planner_workers=1,
        level20_fraction=0.,max_game_frames=120000,native_library=os.environ.get('DRMC_FRAME_LIBRARY'))
    config=tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    command=[sys.executable,'-m','tools.program','launch','trainer-controller-retention','--set',
             'controller_retention_config='+str(config)]
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
    completed=subprocess.run(command,text=True,capture_output=True,timeout=60,env=env)
    assert completed.returncode==0,completed.stdout+completed.stderr
    progress=json.loads((tmp_path/'training/training.json').read_text())
    assert progress['status']=='Training complete' and progress['decisions']>0
    assert set(progress['paces'])==set(paces) and progress['games']==4
    assert progress['losses']['optimizer_steps']>0
    saved=torch.load(tmp_path/'training/core-final.pt',map_location='cpu',weights_only=True)
    assert saved['progress']['identities']['initialization']==hashlib.sha256(initialized.read_bytes()).hexdigest()
    assert any(not torch.equal(v,saved['state_dict'][k]) for k,v in actor.net.state_dict().items())
    cfg['resume']=str(tmp_path/'training/core-final.pt'); config.write_text(json.dumps(cfg))
    recovered=subprocess.run(command,text=True,capture_output=True,timeout=60,env=env)
    assert recovered.returncode==0,recovered.stdout+recovered.stderr
    after=json.loads((tmp_path/'training/training.json').read_text())
    assert after['decisions']==progress['decisions'] and after['games']==progress['games']
    assert after['retention_baseline']==progress['retention_baseline']


def test_equal_pace_credit_keeps_episode_sums_and_counts_uncontrolled_games():
    rows=[dict(pace=p,weight=1/n,old_value=0.,**{'return':1.})
          for p,n in [('sloth',2),('fast',6)] for _ in range(n)]
    prepare_training_records(rows,dict(objective={'advantage_normalization':'none'}))
    balance_pace_credit(rows,{'sloth':4,'fast':2})
    np.testing.assert_allclose([r['actor_weight'] for r in rows],[.75]*2+[1.5]*6)
    # A long game retains six score terms. It is never reduced to one term.
    assert sum(r['actor_weight'] for r in rows if r['pace']=='fast')==9
    rows[0]['anchor_only']=True
    with pytest.raises(ValueError,match='not PPO'):
        balance_pace_credit(rows,{'sloth':4,'fast':2})


@pytest.mark.parametrize('adapted',[False,True])
def test_anchor_wrapper_preserves_reference_controller_actions(parent, adapted):
    encoder=ControllerCorePolicy(parent,training=False)
    reference=PacePolicy(parent) if adapted else PlainPolicy(parent,public_only=True)
    if adapted:
        with torch.no_grad():
            reference.adapter.actor[-1].weight.fill_(.05)
    config=dict(native_library=os.environ.get('DRMC_FRAME_LIBRARY'),
                variants={'a':{'delay':4},'b':{'delay':4}},max_game_frames=1500,replay_games=0)
    match=dict(a='a',b='b',games=2,level=14,pace='normal')
    jobs=[(17291,0,0),(17291,1,1)]
    planner=NativeReachabilityRunner()
    parallel=ParallelPlanning(1)
    try:
        expected,_=run_batch(config,match,jobs,None,planner,None,policies={'a':reference,'b':reference})
        actual,_=run_event_batch(config,match,jobs,None,parallel,None,policies={'a':reference,'b':reference},
            anchor_recorder=RetentionRecorder(encoder,parallel.planner,[0,3]))
        retained=0
        for (eg,em,_),(ag,am,_) in zip(expected,actual):
            assert eg==ag
            for e,a in zip(em,am,strict=True):
                clean={k:v for k,v in a.items() if k!='anchor'}
                assert clean==e
                if 'anchor' in a:
                    r=a['anchor']; retained+=1
                    assert r['anchor_only'] and r['action']==a['placement']['action']
                    assert len(set(r['actions']))==len(r['mask'])
                    assert r['mask'].all() and r['reference_candidates']<=len(r['actions'])
                    assert 'old_logprob' not in r and 'return' not in r
        assert retained>0
    finally:
        planner.close()
        parallel.close()


def test_retention_guard_checks_each_pace_and_rejects_eval_seeds(parent,tmp_path):
    actor=ControllerCorePolicy(parent)
    obs,infos=controller_requests(actor)
    actor.score(obs,infos)
    records=actor.learning_records
    names=['sloth','normal','top_humans','frame_perfect']
    for i,r in enumerate(records):
        r.update(pace=names[i],game_seed=100+i,learner_port=0,anchor_only=True)
    bank=tmp_path/'bank.pt'; save_anchor_bank(bank,records,{})
    retention=PaceRetention(actor,[bank],excluded_seeds=[900],paces=names,batch_size=4)
    assert retention.accepts(retention.baseline)
    bad=dict(retention.baseline); bad['sloth']+=.031
    assert not retention.accepts(bad)
    with pytest.raises(ValueError,match='evaluation data'):
        PaceRetention(actor,[bank],excluded_seeds=[100],paces=names)
    actor.net.zero_grad()
    loss=retention.loss(np.random.default_rng(1)); loss.backward()
    assert any(p.grad is not None for p in actor.net.parameters())


def test_retention_rejection_restores_parameters_and_optimizer(parent,tmp_path):
    actor=ControllerCorePolicy(parent,seed=34)
    obs,infos=controller_requests(actor); actor.score(obs,infos)
    records=actor.learning_records
    for i,r in enumerate(records):
        r.update(pace='normal',weight=1.,**{'return':float(i%2)*2-1})
    class Reject:
        def loss(self,rng):
            return sum(p.sum()*0 for p in actor.net.parameters())
        def measure(self): return {'normal':1.}
        def accepts(self,measured): return False
    optimizer=torch.optim.AdamW(actor.net.parameters(),lr=.001)
    before=deepcopy(actor.net.state_dict()); opt=deepcopy(optimizer.state_dict())
    result=update_adapter(actor,optimizer,records,dict(epochs=1,minibatch=2,kl_backtracks=0),
                          39,retention=Reject(),completed_games_by_pace={'normal':4})
    assert result['optimizer_steps']==0 and result['early_kl_stop']
    assert optimizer.state_dict()==opt
    for k,v in actor.net.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    optimizer.param_groups[0]['lr']=3e-13
    actor.net.eval(); actor.score(obs,infos)
    records=actor.learning_records
    for i,r in enumerate(records):
        r.update(pace='normal',weight=1.,**{'return':float(i%2)*2-1})
    reset=update_adapter(actor,optimizer,records,
        dict(epochs=1,minibatch=2,kl_backtracks=0,reset_update_lr=True,lr=3e-6),
        39,retention=Reject(),completed_games_by_pace={'normal':4})
    assert reset['optimizer_steps']==0
    assert reset['effective_learning_rate']==3e-6
    for k,v in actor.net.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
