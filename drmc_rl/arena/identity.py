"""Recognize arena comparisons whose two entrants make identical decisions.

Two stages, both conservative:

1. Static signature. Model bytes (checkpoint and adapter digests) and every
   variant parameter except its display name must agree once the requested
   ``delay`` is replaced by the delay the rollout actually charges,
   ``max(delay, pace.reaction_frames)``. That substitution is made only for
   reactive decisions (no anticipation, spawn decision point); any other
   decision mode keeps the raw delay in the signature. Parameters that only
   feed a network input (the raw ``delay`` and ``compute_input_frames``) are
   set aside rather than trusted to be ignored.

2. Probe. If the signatures agree but the network-input-only parameters
   differ, a short batch is played. Identical players make each seed's two
   side-swapped games the same physical game, so both complete move journals
   (frames, delays, placements, controller scripts, boards) must be
   byte-identical, as must lengths and terminal reasons, with opposite scores.
   Any difference means the entrants are not interchangeable and the
   comparison continues normally with the probe games counted.

Signatures that agree including those inputs describe the same player; the
comparison is recorded as exactly 0.5 without play.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from drmc_rl.execution.pace import resolve_pace

NETWORK_INPUT_ONLY = ("delay", "compute_input_frames")
NOT_PLAYING = ("name", "ready_when")


@lru_cache(maxsize=64)
def _digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def decision_signature(config, params, pace):
    """(signature, network-input-only settings) for one variant at one pace."""
    signature = {k: v for k, v in params.items() if k not in NOT_PLAYING}
    signature["checkpoint"] = _digest(str(Path(params.get("checkpoint", config["checkpoint"])).resolve()))
    if "adapter_checkpoint" in params:
        signature["adapter_checkpoint"] = _digest(str(Path(params["adapter_checkpoint"]).resolve()))
    inputs = {k: signature.pop(k) for k in NETWORK_INPUT_ONLY if k in signature}
    reactive = not signature.get("anticipation") and signature.get("decision_point", "spawn") == "spawn"
    if reactive:
        signature["charged_delay"] = max(int(inputs["delay"]), pace.reaction_frames)
    else:
        signature["delay"] = inputs.pop("delay")
    return signature, inputs


def static_identity(config, match):
    """"identical", "probe" or None (different players)."""
    pace = resolve_pace(match.get("pace", "frame_perfect"))
    a = decision_signature(config, config["variants"][match["a"]], pace)
    b = decision_signature(config, config["variants"][match["b"]], pace)
    if a[0] != b[0]:
        return None
    return "identical" if a[1] == b[1] else "probe"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def mirrored_games(rows, journals):
    """True if every complete seed's two games are the same physical game.

    ``journals`` maps a game index to its move list. Needs at least one seed.
    """
    seeds = {}
    for row in rows:
        seeds.setdefault(row["seed"], {})[row["side"]] = row
    complete = [games for games in seeds.values() if set(games) == {0, 1}]
    if not complete:
        return False
    for games in complete:
        first, second = games[0], games[1]
        if (first["score"] is None or second["score"] is None or first["score"] + second["score"] != 1
                or first["frames"] != second["frames"] or first["reason"] != second["reason"]):
            return False
        if _canonical(journals[first["index"]]) != _canonical(journals[second["index"]]):
            return False
    return True


def load_journals(output, match_id, rows):
    journals = {}
    for row in rows:
        path = Path(output) / "moves" / f"{match_id}-{row['index']:04d}.json.gz"
        with gzip.open(path, "rt") as stream:
            journals[row["index"]] = json.load(stream)["moves"]
    return journals
