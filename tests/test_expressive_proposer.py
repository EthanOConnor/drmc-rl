import base64
import json

import numpy as np
import torch

from drmc_rl.human.expressive_proposer import ExpressiveProposer, PersistentProposal, public_planes
from drmc_rl.human.expressive_sequences import SCHEMA, construction_windows, replay_sequences
from tools.fit_expressive_proposer import run, session_validation


def replay_fixture():
    initial = np.full((16,8), 0xFF, np.uint8)
    initial[15,:3] = 0xD1
    initial[15,7] = 0xD2
    blue = initial.copy()
    blue[15,5:7] = (0x62,0x72)
    cleared = blue.copy()
    cleared[15,:3] = 0xFF
    def field(board):
        return base64.b64encode(board.tobytes()).decode()
    events = [dict(v=2,t='init',f=1,lvl=[14,14],field1=field(initial),field2=field(initial))]
    for i,(board,pill,preview,x) in enumerate(((initial,[2,2],[1,1],5),(blue,[1,1],[0,0],3),
                                             (cleared,[0,0],[2,1],0))):
        events += [dict(v=2,t='spawn',p=1,f=10+20*i,field=field(board),pill=pill,prev=preview,spd=2),
                   dict(v=2,t='lock',p=1,f=20+20*i,x=x,y=0,rot=0)]
    return events


def raw(events):
    return '\n'.join(json.dumps(e,separators=(',',':')) for e in events).encode()


def test_sequence_requires_exact_payoff_and_uninterrupted_setup():
    events = replay_fixture()
    sequences, counts = replay_sequences(raw(events))
    assert counts['verified'] == 2
    assert len(sequences) == 1
    assert list(construction_windows(sequences[0])) == [(0,2,0)]
    assert sequences[0][0]['action'] == 125
    assert sequences[0][1]['action'] == 123
    assert sequences[0][0]['goals'].sum() == 0
    assert sequences[0][1]['goals'].tolist() == [1,0,0,0]
    interrupted, counters = replay_sequences(raw(events+[dict(f=25,grb=1)]))
    assert counters['garbage'] == 1
    assert not any(list(construction_windows(s)) for s in interrupted)
    damaged = replay_fixture()
    damaged[5]['field'] = damaged[1]['field']  # wrong observed post-clear bottle
    broken, counters = replay_sequences(raw(damaged))
    assert counters['board_mismatch'] == 1
    assert not any(list(construction_windows(s)) for s in broken)


def test_persistent_intent_uses_live_inputs_and_stops_on_events():
    torch.manual_seed(12)
    model = ExpressiveProposer(width=8).eval()
    board = torch.tensor(public_planes(np.full((16,8),0xFF,np.uint8)))
    pill, preview = torch.tensor([[0,1]]), torch.tensor([[2,2]])
    plan = PersistentProposal.start(model,board,pill,preview,frame=10,goal=0,horizon=3)
    root = plan.root.clone()
    calls = []
    hook = model.state.register_forward_hook(lambda _,args,out:calls.append(out.detach().clone()))
    assert set(plan.rank(model,board,pill,preview,[2,3,7])) == {2,3,7}
    plan.rank(model,board,pill,torch.tensor([[0,0]]),[2,3,7])
    hook.remove()
    assert not torch.equal(calls[0],calls[1])
    assert torch.equal(root,plan.root)
    plan.observe(frame=20,completed_placement=True)
    plan.observe(frame=20,completed_placement=True)  # duplicate event cannot consume two turns
    assert plan.remaining == 2
    plan.observe(frame=21,incoming_garbage=True)
    assert plan.reason == 'board_changed'
    assert plan.rank(model,board,pill,preview,[2,3,7]) == []
    achieved = PersistentProposal.start(model,board,pill,preview,frame=10,goal=0,horizon=3)
    achieved.observe(frame=20,completed_placement=True,observed_goals=[0])
    assert achieved.reason == 'goal_observed'


def test_fit_keeps_whole_sessions_out_of_gradients(tmp_path):
    sequence = replay_sequences(raw(replay_fixture()))[0][0]
    training = [f's{i}' for i in range(100) if not session_validation(f's{i}',81029)][:2]
    validation = [f's{i}' for i in range(100) if session_validation(f's{i}',81029)][:2]
    sessions = training+validation
    rows = sequence*len(sessions)
    data = {k:np.asarray([r[k] for r in rows]) for k in ('board','pill','preview','action')}
    data.update(windows=np.asarray([(2*i,2,0,i) for i in range(4)],np.int64),
                sessions=np.asarray(sessions), metadata=np.asarray(json.dumps(dict(schema=SCHEMA))))
    source = tmp_path/'source.npz'
    np.savez_compressed(source,**data)
    config = dict(source=str(source),output=str(tmp_path/'fit'),epochs=2,batch_windows=2,width=8,threads=1)
    report = run(config)
    assert report['train_sessions'] == report['validation_sessions'] == 2
    assert report['window_presentations'] == 4
    assert report['action_presentations'] == 8
    assert report['console_frames_trained'] == 0
    state = torch.load(tmp_path/'fit/proposer-final.pt',weights_only=False)
    loaded = ExpressiveProposer(width=8)
    loaded.load_state_dict(state['state_dict'],strict=True)
    data['board'][4:,0,0] = 0xD0
    data['action'][4:] = 4
    other = tmp_path/'changed.npz'
    np.savez_compressed(other,**data)
    changed = run({**config,'source':str(other),'output':str(tmp_path/'changed-fit')})
    other_state = torch.load(tmp_path/'changed-fit/proposer-final.pt',weights_only=False)
    assert changed['epochs'][-1]['validation'] != report['epochs'][-1]['validation']
    assert all(torch.equal(v,other_state['state_dict'][k]) for k,v in state['state_dict'].items())
