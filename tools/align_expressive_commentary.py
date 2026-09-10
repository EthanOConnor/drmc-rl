"""Match supplied commentary times to video reconstructions for human review.

An internally consistent reconstructed prefix is not independent video
verification. These matches never become imitation/preference/VS-attack labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.human.expressive_sequences import locked_field, observed_goals


def commentary_windows(text):
    for section in re.split(r'^## ', text, flags=re.MULTILINE)[1:]:
        title = section.splitlines()[0]
        source = next((line for line in section.splitlines() if line.startswith('Source:')), '')
        times = re.findall(r'\d{2}:\d{2}:\d{2}', source)
        video = re.search(r'/yt_([A-Za-z0-9_-]{11})\.txt', source)
        if len(times) < 2:
            continue
        def seconds(value):
            hours, minutes, secs = map(int,value.split(':'))
            return 3600*hours+60*minutes+secs
        yield dict(title=title, source=source, video_id=video.group(1) if video else None,
                   start=seconds(times[0]), end=seconds(times[1]))


def display_board(board):
    rows = []
    for row in np.asarray(board).reshape(16,8):
        rows.append(''.join('.' if tile >= 0xF0 else 'YRB'[int(tile)&3].upper()
                            if (tile&0xF0) == 0xD0 else 'yrb'[int(tile)&3] for tile in row))
    return rows


def consistent_prefix(game):
    from tools.annotate_replay_events import POSE_TO_ACTION
    if len(game.get('b0',[])) != 16 or any(len(r) != 8 or any(c not in '.YRB' for c in r) for r in game['b0']):
        return [], 'initial_board_missing_or_bonds_unknown'
    board = np.asarray([[0xFF if c == '.' else 0xD0+'YRB'.index(c) for c in row] for row in game['b0']],np.uint8)
    verified = []
    for index,event in enumerate(game.get('events',[])):
        if event.get('i') != index:
            return verified, 'sequence_gap'
        if type(event.get('t')) not in (int,float) or not math.isfinite(event['t']):
            return verified, 'timestamp_missing'
        try:
            x,y,rotation = (int(event[k]) for k in ('x','y','o'))
            if not 0 <= x < 8 or not 0 <= y < 16 or not 0 <= rotation < 4:
                return verified,'invalid_pose'
            pill = ['YRB'.index(c) for c in event['p']]
            if len(pill) != 2:
                return verified,'invalid_pill'
            action = int(POSE_TO_ACTION[rotation*128+y*8+x])
            result = resolve_cascade(locked_field(board,pill,action))
            following = np.frombuffer(result.settled_field,np.uint8).reshape(16,8)
            if display_board(following) != event['b']:
                return verified,'board_mismatch'
        except (ValueError,KeyError,IndexError,TypeError):
            return verified,'invalid_event'
        verified.append(dict(index=index,time=float(event['t']),action=action,
                             observed_goals=observed_goals(result).tolist()))
        board = following.copy()
    return verified, 'end_of_recorded_sequence'


def run(evidence, reconstruction, output):
    output.mkdir(parents=True,exist_ok=False)
    text = evidence.read_text()
    raw = reconstruction.read_bytes()
    archive = json.loads(gzip.decompress(raw) if reconstruction.suffix == '.gz' else raw)
    windows = list(commentary_windows(text))
    videos = {w['video_id'] for w in windows if w['video_id']}
    games = {k:v for k,v in archive['games'].items() if v.get('video_id') in videos}
    prefixes = {k:consistent_prefix(v) for k,v in games.items()}
    records = []
    for window in windows:
        candidates = []
        for identity,game in games.items():
            if game.get('video_id') != window['video_id']:
                continue
            selected = [e for e in game.get('events',[]) if type(e.get('t')) in (int,float)
                        and type(e.get('i')) is int and window['start']-8 <= e['t'] <= window['end']+8]
            if not selected:
                continue
            first = max(0,int(selected[0]['i'])-6)
            last = int(selected[-1]['i'])
            verified, stopped = prefixes[identity]
            consistent = last < len(verified)
            before = next((e for e in game['events'] if e['i'] == first),selected[0])
            if type(before.get('t')) not in (int,float):
                before = selected[0]
            candidates.append(dict(game=identity,player=game.get('tag'),feed=game.get('feed'),level=game.get('level'),
                first_placement=first,last_placement=last,physics_consistent_prefix=consistent,
                prefix_placements=len(verified),prefix_stopped=stopped,
                watch_url=f"https://www.youtube.com/watch?v={window['video_id']}&t={max(0,int(before['t'])-3)}s",
                speaker_to_player_verified=False,video_verified=False,eligible_for_training=False))
        records.append({**window,'candidates':candidates})
    report = dict(schema='drmc-commentary-alignment-review-v1',status='Complete',
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        reconstruction_sha256=hashlib.sha256(raw).hexdigest(),windows=len(records),
        matched_windows=sum(bool(r['candidates']) for r in records),
        missing_video_identity=sum(not r['video_id'] for r in records),
        consistent_candidates=sum(c['physics_consistent_prefix'] for r in records for c in r['candidates']),
        prefix_stops=dict(Counter(reason for _,reason in prefixes.values())),records=records,
        scope='Review candidates only: verify video, commentator referent, timing and geometry before any preference/plan label.')
    (output/'alignment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--reconstruction',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.evidence,args.reconstruction,args.output)


if __name__ == '__main__':
    main()
