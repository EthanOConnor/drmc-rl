"""The fast public-event encoding must be bit-identical to the reference form."""
from dataclasses import replace

import numpy as np

from drmc_rl.game.pair_state import PairEvent, PairEventKind
from drmc_rl.models.policy.event_belief import EVENT_FEATURE_DIM, _EVENT_INDEX, _number, event_feature, pair_events_to_features


def _reference(event, current_frame):
    out = np.zeros(EVENT_FEATURE_DIM, dtype=np.float32)
    out[_EVENT_INDEX[event.kind]] = 1.0
    offset = len(PairEventKind)
    out[offset + (0 if event.side is None else 1 + int(event.side))] = 1.0
    c, p = offset + 3, event.public_payload
    out[c] = np.log1p(current_frame - event.frame_id) / np.log1p(600.0)
    out[c + 1] = np.clip(_number(p, ("garbage_size", "size", "volley_size")) / 4.0, 0.0, 1.0)
    out[c + 2] = np.clip(_number(p, ("tiles_cleared", "cleared_tiles")) / 32.0, 0.0, 1.0)
    out[c + 3] = np.clip(_number(p, ("viruses_cleared", "virus_delta")) / 16.0, 0.0, 1.0)
    out[c + 4] = np.clip(_number(p, ("row_top", "row"), -1.0) / 15.0, -1.0, 1.0)
    out[c + 5] = np.clip(_number(p, ("column", "col"), -1.0) / 7.0, -1.0, 1.0)
    out[c + 6] = np.clip(_number(p, ("rotation", "rot")) / 3.0, 0.0, 1.0)
    out[c + 7] = np.clip(_number(p, ("outcome", "terminal_outcome")), -1.0, 1.0)
    return out


def test_fast_event_features_match_the_reference_bit_for_bit():
    rng = np.random.default_rng(5)
    kinds = list(PairEventKind)
    events = []
    for frame in sorted(rng.integers(0, 5000, 400).tolist()):
        payload = {}
        for key, low, high in (("garbage_size", -2, 9), ("tiles_cleared", -1, 70), ("viruses_cleared", -3, 40),
                               ("row_top", -20, 30), ("column", -9, 12), ("rotation", -1, 7), ("outcome", -3, 3)):
            if rng.random() < 0.5:
                payload[key] = int(rng.integers(low, high))
        if rng.random() < 0.1:
            payload["size"] = float(rng.normal() * 4)
        events.append(PairEvent(kinds[rng.integers(len(kinds))], frame,
                                [None, 0, 1][rng.integers(3)], payload))
    now = events[-1].frame_id + 17
    for side in (0, 1):
        relative = [replace(e, side=None if e.side is None else int(e.side != side)) for e in events]
        for event, rel in zip(events, relative):
            fast = event_feature(event, current_frame=now, relative_to=side)
            assert fast.tobytes() == _reference(rel, now).tobytes()
        fast, mask = pair_events_to_features(events, current_frame=now, max_events=32, relative_to=side)
        slow = np.stack([_reference(e, now) for e in relative[-32:]])
        assert fast.tobytes() == slow.tobytes() and mask.all()
    for event in events:
        assert event_feature(event, current_frame=now).tobytes() == _reference(event, now).tobytes()
