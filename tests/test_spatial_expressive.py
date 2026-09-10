import copy
import json

import numpy as np
import pytest
import torch

from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.human.expressive_sequences import locked_field
from drmc_rl.human.spatial_proposer import (
    FIXED_ROOT, RECURRENT_PUBLIC, FrozenConstructionEncoder, SpatialProposal, SpatialProposer, spatial_clear_target,
)
from drmc_rl.models.policy.candidate_policy_g5 import G5CandidatePlacementPolicyNet
from tools.fit_expressive_proposer import session_validation
from tools.fit_spatial_expressive import fit_models, prepare_data
from tools.build_expressive_sequences import exclude_replays


def test_spatial_goal_identifies_actual_colored_clear_cells():
    board = np.full((16,8),0xFF,np.uint8)
    board[15,:3] = 0xD1
    result = resolve_cascade(locked_field(board,[1,1],123))
    target = spatial_clear_target(result,0)
    expected = np.zeros_like(target)
    expected[0,15,:5] = .2
    np.testing.assert_array_equal(target,expected)
    with pytest.raises(ValueError,match='selected spatial goal'):
        spatial_clear_target(result,3)


def test_fresh_replay_exclusions_cover_session_aliases_and_blob_reuse(tmp_path):
    from drmc_rl.human.expressive_sequences import SCHEMA
    source=tmp_path/'earlier.npz'
    np.savez(source,metadata=np.asarray(json.dumps(dict(schema=SCHEMA,sources=[
        dict(session='one',sha256='abc'),dict(session='two',sha256='def')]))))
    eligible,report=exclude_replays([('one','changed'),('alias','abc'),('new','fresh')],[source])
    assert eligible==[('new','fresh')]
    assert report['excluded_catalogue_rows']==2
    assert report['excluded_sessions']==report['excluded_blobs']==2


@pytest.mark.parametrize('aux_dim',[0,72])
def test_shared_features_match_real_competitive_own_bottle_path(aux_dim):
    torch.manual_seed(12)
    core = G5CandidatePlacementPolicyNet(in_channels=20,board_channels=16,encoder_blocks=1,
        d_model=16,pill_embed_dim=8,transformer_heads=4,cross_layers=0,interaction_layers=0,
        cand_hidden_dim=32,patch_kernel=3,aux_dim=aux_dim).eval()
    board = np.full((16,8),0xFF,np.uint8)
    board[15,4:6] = [0x60,0x70]
    own = torch.tensor(np.stack([board_bytes_to_semantic_planes(board)]*2))
    pill = torch.tensor([[0,1],[1,1]])
    preview = torch.tensor([[2,2],[0,2]])
    captured = []
    hook = core.bottle.register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
    historical = own.clone()
    historical[1,6:8] = 0
    with torch.inference_mode():
        core(torch.cat((historical,torch.zeros_like(own),torch.zeros(2,4,16,8)),1),pill,preview,
             torch.tensor([[123],[123]]),torch.ones(2,1),torch.ones(2,1,dtype=torch.bool),
             aux=torch.zeros(2,aux_dim) if aux_dim else None)
    hook.remove()
    expected = torch.cat((captured[0].mean((2,3)),captured[0].amax((2,3))),-1)
    encoder = FrozenConstructionEncoder(core,zero_auxiliary=aux_dim>0)
    with torch.inference_mode():
        actual = encoder(own,pill,preview)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert not any(p.requires_grad for p in encoder.parameters())
    core.aux_dim = 1
    with pytest.raises(ValueError,match='full public-context'):
        FrozenConstructionEncoder(core)


def test_spatial_persistence_requires_the_selected_location_and_handles_surprises():
    torch.manual_seed(1)
    model = SpatialProposer(8,16).eval()
    with torch.no_grad():
        model.target.weight.zero_()
        model.target.bias.fill_(-20)
        model.target.bias[120] = 20  # red, bottom-left cell
        model.horizon.weight.zero_()
        model.horizon.bias.fill_(-20)
        model.horizon.bias[2] = 20
    inputs = (torch.zeros(1,8,16,8),torch.zeros(1,8),torch.tensor([[0,1]]),torch.tensor([[2,2]]))
    plan = SpatialProposal.start(model,inputs,frame=10,goal=0)
    memory,spatial = plan.memory.clone(),plan.spatial.clone()
    assert plan.remaining == 3
    assert set(plan.rank(model,inputs,[123,124,125])) == {123,124,125}
    changed = (*inputs[:3],torch.tensor([[0,0]]))
    plan.rank(model,changed,[123,124,125])
    assert torch.equal(plan.memory,memory) and torch.equal(plan.spatial,spatial)
    plan.observe(frame=20,completed_placement=True,observed_goals=[0],cleared_cells=[(15,1,0)])
    plan.observe(frame=20,completed_placement=True)
    assert plan.remaining == 2 and plan.reason is None
    plan.observe(frame=30,completed_placement=True,observed_goals=[0],cleared_cells=[(15,0,0)])
    assert plan.reason == 'spatial_goal_observed'
    surprise = SpatialProposal.start(model,inputs,frame=10,goal=0)
    surprise.observe(frame=11,incoming_garbage=True)
    assert surprise.rank(model,inputs,[123]) == []
    with pytest.raises(ValueError,match='stateless'):
        SpatialProposal.start(SpatialProposer(8,16,persistent=False),inputs,frame=10)


def test_preparation_loads_checkpoint_and_deduplicates_only_actual_inputs(tmp_path,monkeypatch):
    import tools.eval_policy
    from drmc_rl.human.expressive_sequences import SCHEMA

    core = G5CandidatePlacementPolicyNet(in_channels=20,board_channels=16,encoder_blocks=1,
        d_model=16,pill_embed_dim=8,transformer_heads=4,cross_layers=0,interaction_layers=0,
        cand_hidden_dim=32,patch_kernel=3).eval()
    checkpoint = tmp_path/'core.pt'
    torch.save(dict(cfg=dict(smdp_ppo=dict(candidate_board_channels=16)),state_dict=core.state_dict()),checkpoint)
    monkeypatch.setattr(tools.eval_policy,'_build_net_from_cfg',lambda *a:(copy.deepcopy(core),0,512))
    board=np.full((2,16,8),0xFF,np.uint8)
    board[:,15,:3]=0xD1
    source=tmp_path/'source.npz'
    np.savez(source,board=board,pill=np.ones((2,2),np.uint8),preview=np.zeros((2,2),np.uint8),
        action=np.array([122,123]),windows=np.array([[0,2,0,0]]),sessions=np.array(['one']),
        metadata=np.asarray(json.dumps(dict(schema=SCHEMA))))
    output=tmp_path/'prepared'
    output.mkdir()
    report={}
    data=prepare_data(dict(source=str(source),checkpoint=str(checkpoint)),output,report)
    assert report['unique_public_inputs']==1 and data['feature_index'].tolist()==[0,0]
    assert data['features'].shape==(1,32)
    with np.load(output/'prepared.npz',allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['features'],data['features'])
        np.testing.assert_array_equal(archive['planes'],data['planes'])
    expected=np.zeros((3,16,8),np.float32)
    expected[0,15,:5]=.2
    np.testing.assert_array_equal(data['spatial_targets'][0],expected.ravel())


@pytest.mark.parametrize('plan_update_schema',[FIXED_ROOT,RECURRENT_PUBLIC])
def test_both_trained_controls_exclude_validation_and_do_not_decode_true_targets(tmp_path,plan_update_schema):
    torch.set_num_threads(1)
    sessions = ([f's{i}' for i in range(100) if not session_validation(f's{i}',81029)][:2]+
                [f's{i}' for i in range(100) if session_validation(f's{i}',81029)][:2])
    rng = np.random.default_rng(27)
    data = dict(planes=rng.normal(size=(12,8,16,8)).astype(np.float32),
        features=rng.normal(size=(12,8)).astype(np.float32),canonical_pill=np.zeros((12,2),np.int64),
        canonical_preview=np.ones((12,2),np.int64),feature_index=np.arange(12),
        action=np.asarray([123,124,125]*4),windows=np.asarray([[i*3,3,0,i] for i in range(4)]),
        spatial_targets=np.full((4,384),1/384,np.float32),sessions=np.asarray(sessions))
    config=dict(epochs=2,batch_windows=2,width=8,threads=1,plan_update_schema=plan_update_schema)
    results = []
    for attempt in range(2):
        output=tmp_path/str(attempt)
        output.mkdir()
        report=dict(source_sha256='fixture',competitive_sha256='fixture')
        if attempt:
            data['features'][6:] += 2
            data['spatial_targets'][2:] = 0
            data['spatial_targets'][2:,0] = 1
            data['action'][6:] = 127
        results.append(fit_models(data,config,output,report))
    for name in ('persistent','stateless'):
        a=torch.load(tmp_path/'0'/f'{name}-final.pt',weights_only=False)
        b=torch.load(tmp_path/'1'/f'{name}-final.pt',weights_only=False)
        assert all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
        loaded=SpatialProposer(a['feature_dim'],a['width'],persistent=a['persistent'],
                               plan_update_schema=a['plan_update_schema'])
        loaded.load_state_dict(a['state_dict'],strict=True)
        assert results[0]['arms'][name]['action_presentations']==12
        assert results[0]['arms'][name]['final']!=results[1]['arms'][name]['final']
    assert results[0]['paired_session_comparisons']['early_setup_nll']['sessions']==2
    # Labels have no argument in the action path; only predicted distributions
    # can reach the decoder. Neither trained arm modifies the competitive core.
    assert not a['quality_admission'] and a['diagnostic_only']
    json.dumps(results[0],allow_nan=False)


@pytest.mark.parametrize('plan_update_schema',[FIXED_ROOT,RECURRENT_PUBLIC])
def test_fresh_confirmation_has_no_optimizer_and_rejects_reused_content(tmp_path,monkeypatch,plan_update_schema):
    from tools import confirm_spatial_expressive as confirm
    from tools.fit_spatial_expressive import sha256
    from drmc_rl.human.spatial_proposer import SCHEMA

    torch.set_num_threads(1)
    earlier,fresh=tmp_path/'earlier.npz',tmp_path/'fresh.npz'
    def source(path,entries):
        np.savez(path,metadata=np.asarray(json.dumps(dict(sources=entries))))
    source(earlier,[dict(session='old',sha256='old-blob')])
    source(fresh,[dict(session='fresh-a',sha256='a'),dict(session='fresh-b',sha256='b')])
    core=tmp_path/'core.pt'
    core.write_bytes(b'frozen feature encoder supplied by the test')
    data=dict(planes=np.zeros((4,8,16,8),np.float32),features=np.zeros((4,8),np.float32),
        canonical_pill=np.zeros((4,2),np.int64),canonical_preview=np.ones((4,2),np.int64),
        feature_index=np.arange(4),action=np.array([123,124]*2),windows=np.array([[0,2,0,0],[2,2,0,1]]),
        spatial_targets=np.full((2,384),1/384,np.float32),sessions=np.array(['fresh-a','fresh-b']))
    monkeypatch.setattr(confirm,'prepare_data',lambda *a:data)
    def no_optimizer(*a,**k):
        raise AssertionError('confirmation cannot construct an optimizer')
    monkeypatch.setattr(torch.optim,'AdamW',no_optimizer)
    study=dict(schema=SCHEMA,status='Complete',source_sha256=sha256(earlier),competitive_sha256=sha256(core),
               config=dict(source=str(earlier),checkpoint=str(core),plan_update_schema=plan_update_schema),arms={})
    for name in ('persistent','stateless'):
        model=SpatialProposer(8,8,persistent=name=='persistent',plan_update_schema=plan_update_schema)
        checkpoint=dict(schema=SCHEMA,persistent=model.persistent,feature_dim=8,width=8,
            plan_update_schema=plan_update_schema,
            source_sha256=study['source_sha256'],competitive_sha256=study['competitive_sha256'],
            state_dict=model.state_dict(),training_priors=dict(spatial=torch.full((4,81,384),1/384),
                horizon=torch.full((4,81,6),1/6),intent=torch.full((81,4),1/4)))
        path=tmp_path/(name+'-final.pt')
        torch.save(checkpoint,path)
        study['arms'][name]=dict(checkpoint_sha256=sha256(path))
    study_path=tmp_path/'study.json'
    study_path.write_text(json.dumps(study))
    config=dict(study=str(study_path),source=str(fresh),output=str(tmp_path/'confirmation'))
    result=confirm.run(config)
    assert result['status']=='Complete' and result['optimizer_updates']==0
    assert result['arms']['persistent']['evaluated_actions']==4
    assert result['session_overlap']==result['blob_overlap']==0
    assert not result['quality_admission']
    source(fresh,[dict(session='new-alias',sha256='old-blob')])
    with pytest.raises(ValueError,match='reuses development'):
        confirm.run({**config,'output':str(tmp_path/'must-not-exist')})
    assert not (tmp_path/'must-not-exist').exists()
    previous=tmp_path/'earlier-confirmation.npz'
    source(previous,[dict(session='confirmed',sha256='confirmation-blob')])
    source(fresh,[dict(session='alias',sha256='confirmation-blob')])
    with pytest.raises(ValueError,match='earlier evaluation'):
        confirm.run({**config,'exclude_sources':[str(previous)],'output':str(tmp_path/'must-not-exist')})
    assert not (tmp_path/'must-not-exist').exists()


def test_recurrent_prefixes_are_causal_differentiable_and_match_event_runtime():
    torch.manual_seed(32)
    current=torch.randn(5,8,requires_grad=True)
    owner=torch.tensor([0,0,0,1,1])
    elapsed=torch.tensor([0,1,2,0,1])
    persistent=SpatialProposer(8,8,plan_update_schema=RECURRENT_PUBLIC)
    control=SpatialProposer(8,8,persistent=False,plan_update_schema=RECURRENT_PUBLIC)
    control.load_state_dict(persistent.state_dict(),strict=True)
    actual=persistent.sequence_memory(current,owner,elapsed,2)
    expected=[]
    for start,length in [(0,3),(3,2)]:
        previous=torch.zeros(1,8)
        for row in range(start,start+length):
            previous=persistent.update_memory(current[row:row+1],previous)
            expected.append(previous[0])
    torch.testing.assert_close(actual,torch.stack(expected))
    changed=current.detach().clone()
    changed[[2,4]] += 100
    revised=persistent.sequence_memory(changed,owner,elapsed,2)
    torch.testing.assert_close(actual[[0,1,3]],revised[[0,1,3]],rtol=0,atol=0)
    grad=torch.autograd.grad(actual[2].sum(),current,retain_graph=True)[0]
    assert grad[:2].abs().sum()>0 and grad[3:].count_nonzero()==0
    reset=control.sequence_memory(current,owner,elapsed,2)
    grad=torch.autograd.grad(reset[2].sum(),current)[0]
    assert grad[:2].count_nonzero()==0 and grad[2].abs().sum()>0
    torch.testing.assert_close(actual[[0,3]],reset[[0,3]])
    _,duration=persistent.plan_at(actual,torch.zeros(5,dtype=torch.long),elapsed)
    assert torch.isneginf(duration[elapsed==0,0]).all()
    assert torch.isfinite(duration[elapsed>0]).all()


def test_replanning_revises_current_slot_and_commits_only_unique_placement_events():
    torch.manual_seed(89)
    model=SpatialProposer(8,8,plan_update_schema=RECURRENT_PUBLIC).eval()
    with torch.no_grad():
        model.horizon.weight.zero_()
        model.horizon.bias.fill_(-20)
        model.horizon.bias[2]=20
    inputs=(torch.zeros(1,8,16,8),torch.zeros(1,8),torch.tensor([[0,1]]),torch.tensor([[2,2]]))
    proposal=SpatialProposal.start(model,inputs,frame=1,goal=0)
    assert proposal.remaining==3
    ranking=proposal.rank(model,inputs,list(range(512)))
    memory,spatial=proposal.memory.clone(),proposal.spatial.clone()
    assert proposal.rank(model,inputs,list(range(512)))==ranking
    assert torch.equal(proposal.memory,memory) and len(set(ranking))==512
    changed=(*inputs[:3],torch.tensor([[0,0]]))
    proposal.rank(model,changed,list(range(512)))
    assert not torch.equal(proposal.memory,memory) and not torch.equal(proposal.spatial,spatial)
    assert proposal.committed_memory.count_nonzero()==0 and proposal.remaining==3
    committed=proposal.memory.clone()
    proposal.observe(frame=2,completed_placement=True)
    proposal.observe(frame=2,completed_placement=True)
    torch.testing.assert_close(proposal.committed_memory,committed,rtol=0,atol=0)
    assert proposal.remaining==2 and proposal.elapsed==1
    with torch.inference_mode():
        expected=model.update_memory(model.encode(*inputs),committed)
    proposal.rank(model,inputs,[123,124])
    torch.testing.assert_close(proposal.memory,expected,rtol=0,atol=0)
    assert proposal.remaining==2  # Revised predictions cannot extend the root budget.
    proposal.observe(frame=3,completed_placement=True)
    proposal.observe(frame=4,completed_placement=True)
    assert proposal.reason=='placement_budget' and proposal.rank(model,inputs,[123])==[]
    for event,reason in [(dict(incoming_garbage=True),'board_changed'),
                         (dict(own_state_mismatch=True),'board_changed'),(dict(terminal=True),'terminal')]:
        fresh=SpatialProposal.start(model,inputs,frame=10)
        fresh.observe(frame=11,**event)
        assert fresh.reason==reason and fresh.rank(model,inputs,[123])==[]


def test_prepared_feature_reuse_requires_unchanged_source_encoder_and_content(tmp_path):
    from tools.fit_spatial_expressive import sha256
    from drmc_rl.human.spatial_proposer import SCHEMA

    source,core=tmp_path/'source.npz',tmp_path/'core.pt'
    source.write_bytes(b'verified replay source')
    core.write_bytes(b'frozen competitive model')
    np.savez_compressed(tmp_path/'prepared.npz',features=np.arange(8,dtype=np.float32))
    previous=dict(schema=SCHEMA,status='Complete',source_sha256=sha256(source),
        competitive_sha256=sha256(core),prepared_sha256=sha256(tmp_path/'prepared.npz'),
        competitive_aux_spec='none',unique_public_inputs=1,feature_rows=1,targets=1,unique_payoffs=1)
    study=tmp_path/'progress.json'
    study.write_text(json.dumps(previous))
    output=tmp_path/'reuse'
    output.mkdir()
    config=dict(source=str(source),checkpoint=str(core),prepared_from=str(study))
    report={}
    data=prepare_data(config,output,report)
    np.testing.assert_array_equal(data['features'],np.arange(8,dtype=np.float32))
    assert report['prepared_source']==str(tmp_path/'prepared.npz')
    assert not (output/'prepared.npz').exists()
    core.write_bytes(b'changed model')
    with pytest.raises(ValueError,match='identical completed source'):
        prepare_data(config,output,{})
    core.write_bytes(b'frozen competitive model')
    source.write_bytes(b'changed replay')
    with pytest.raises(ValueError,match='identical completed source'):
        prepare_data(config,output,{})
    source.write_bytes(b'verified replay source')
    (tmp_path/'prepared.npz').write_bytes(b'corrupt features')
    with pytest.raises(ValueError,match='feature content changed'):
        prepare_data(config,output,{})
