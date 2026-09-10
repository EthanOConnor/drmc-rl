import json

import numpy as np
import pytest
import torch

from drmc_rl.game.public_context import PUBLIC_CONTEXT_DIM, SIDE_FEATURE_DIM
from drmc_rl.training.public_alignment import alignment_kl, alignment_target, legacy_teacher_inputs


def test_teacher_view_and_targets_preserve_student_information_and_full_support():
    obs = torch.ones((2,20,16,8))
    original = obs.clone()
    pill = torch.tensor([[0,0],[0,1]])
    context = torch.zeros((2,PUBLIC_CONTEXT_DIM))
    context[:,SIDE_FEATURE_DIM:SIDE_FEATURE_DIM+6] = torch.tensor([[1,0,0,1,0,0],[1,0,0,0,1,0]])
    actions = torch.tensor([[1,129,258,-1],[1,129,258,-1]])
    mask = actions >= 0
    teacher_obs, teacher_mask = legacy_teacher_inputs(obs,pill,context,actions,mask)
    torch.testing.assert_close(obs,original,rtol=0,atol=0)
    assert not teacher_obs[0,6:8].any() and not teacher_obs[0,14:16].any()
    assert teacher_obs[1,6:8].all() and teacher_obs[1,14:16].all()
    assert teacher_mask.tolist() == [[True,True,False,False],[True,True,True,False]]
    logits = torch.tensor([[4.,1.,9.,0.],[4.,1.,9.,0.]])
    target, supported = alignment_target(logits,teacher_mask,mask)
    assert supported.all() and (target[mask]>0).all() and (target[~mask]==0).all()
    torch.testing.assert_close(target.sum(-1),torch.ones(2))
    assert target.argmax(-1).tolist() == [0,2]
    losses = alignment_kl(logits,target,mask)
    assert torch.isfinite(losses).all()
    _, supported = alignment_target(logits,torch.zeros_like(mask),mask)
    assert not supported.any()
    with pytest.raises(ValueError,match="support"):
        alignment_target(logits,teacher_mask,mask,support_epsilon=0)


def test_fixed_alignment_fit_is_seed_grouped_and_validation_never_changes_weights(tmp_path):
    from tools.eval_policy import _build_net_from_cfg
    from tools.fit_public_alignment import run
    from tools.vs_head_to_head import PlainPolicy

    torch.manual_seed(27)
    cfg = dict(candidate_architecture="g5", candidate_board_channels=16,
               candidate_d_model=16, encoder_blocks=1, pill_embed_dim=8,
               candidate_hidden_dim=24, candidate_cross_layers=1,
               candidate_interaction_layers=1, candidate_transformer_heads=2,
               candidate_patch_kernel=3, aux_spec="zero_v1_vs")
    parent,_,_ = _build_net_from_cfg(cfg,20,"cpu")
    parent_path = tmp_path/"parent.pt"
    torch.save({"cfg":cfg,"state_dict":parent.state_dict()},parent_path)
    n = 24
    obs = np.zeros((n,20,16,8),np.uint8)
    obs[:,0,15,0]=1
    obs[:,16:].reshape(n,512)[:,[113,241,370]]=1
    context = np.zeros((n,PUBLIC_CONTEXT_DIM),np.float32)
    context[:,[0,3,SIDE_FEATURE_DIM+1,SIDE_FEATURE_DIM+4]]=1
    arrays = dict(observation=obs,pill=np.zeros((n,2),np.int8),preview=np.ones((n,2),np.int8),
                  public_context=context,game_seed=np.repeat(np.arange(8),3),
                  actions=np.tile([113,241,370],n),costs=np.tile([40,50,60],n).astype(np.uint16),
                  offsets=np.arange(n+1)*3,
                  metadata=np.asarray(json.dumps({"schema":"drmc-public-controller-replay-v1",
                    "observation_schema":"public_pair_context_v3","pace":"normal","level":14})))
    source = tmp_path/"training.npz"
    np.savez_compressed(source,**arrays)
    config = dict(parent=str(parent_path),replays=[str(source)],output=str(tmp_path/"first"),
                  source_scope="training-only-controller-replay",device="cpu",seed=38,
                  epochs=2,batch_size=8,roots_per_seed_per_shard=3)
    report = run(config)
    assert report["status"]=="Complete" and report["unsupported_rows"]==0
    assert report["training_rows"]==18 and report["validation_rows"]==6
    assert report["root_presentations"]==36 and report["console_frames_trained"]==0
    loaded = PlainPolicy(tmp_path/"first/core-final.pt",public_only=True)
    assert loaded.requires_causal_observations
    assert loaded.aux_spec=="public_pair_context_v3"
    # Perturb held-out observations and targets only. Neither selection nor
    # optimizer updates may depend on validation metrics.
    validation = np.isin(arrays["game_seed"],report["validation_seeds"])
    arrays["observation"][validation,0,15,0]=0
    arrays["observation"][validation,1,14,3]=1
    np.savez_compressed(source,**arrays)
    second = run({**config,"output":str(tmp_path/"second")})
    assert second["validation_seeds"]==report["validation_seeds"]
    saved = torch.load(tmp_path/"second/core-final.pt",weights_only=False,map_location="cpu")
    for key,value in loaded.net.state_dict().items():
        torch.testing.assert_close(value,saved["state_dict"][key],rtol=0,atol=0)
    assert second["epochs"][-1]["validation"]!=report["epochs"][-1]["validation"]
