"""Combo and showiness counters per entrant (visibility only; never part of a rating).

Workers resolve every placement of both sides exactly with the big-clear
scorer (``drmc_rl.eval.big_clear``, the same definition as the big-clear
setups experiment and ``tools.big_clear_pool_metrics``) and report compact
per-side counters with each game. The coordinator only sums counters.

Definitions per placement: a *clear* resolves at least one line; a *combo*
has 2+ lines in the resolution or 2+ cascade rounds; a *chain* has 2+ rounds;
*garbage* is the ROM's min(lines, 4) pieces for 2+ lines; T1/T2/T3 are
showiness scores >= 20 / 30 / 42 (a plain 4-line clear scores 0).
"""
from __future__ import annotations

KEYS = ("placements", "clears", "lines", "combos", "chains", "garbage", "attacks", "attacks34",
        "t1", "t2", "t3", "t1_score")

# Human reference rates from the full Fightcade corpus (drmc-rl-bigclear-data
# mining-summary.json: 57,341,660 placements, 20,180,178 clears), per 100 placements.
HUMAN = dict(combos=15.6, chains=12.8, clears=35.2, garbage=34.1, share34=0.157, t1=0.85, t2=0.15, t3=0.012)


def side_style(moves, physical):
    """Counters for the side playing ``physical`` in one game's move journal."""
    from drmc_rl.eval import big_clear as bc
    t1, t2, t3 = (bar for _, bar in bc.TIERS)
    out = dict.fromkeys(KEYS, 0)
    out["t1_score"] = 0.0
    best = None
    for move in moves:
        if int(move["side"]) != physical:
            continue
        out["placements"] += 1
        try:
            f = bc.placement_features(bytes(move["board"]), move["pill"], int(move["placement"]["action"]))
        except (ValueError, KeyError, TypeError):
            continue
        if not f.rounds:
            continue
        score = f.score()
        out["clears"] += 1
        out["lines"] += f.lines
        out["combos"] += int(f.lines >= 2 or f.rounds >= 2)
        out["chains"] += int(f.rounds >= 2)
        out["garbage"] += f.garbage
        out["attacks"] += int(f.garbage >= 2)
        out["attacks34"] += int(f.garbage >= 3)
        out["t1"] += int(score >= t1)
        out["t2"] += int(score >= t2)
        out["t3"] += int(score >= t3)
        if score >= t1:
            out["t1_score"] += score
        if best is None or score > best["score"]:
            best = dict(score=score, cells=f.cells, rounds=f.rounds, lines=f.lines)
    out["best"] = best
    return out


def game_style(row, moves):
    """[counters of entrant a, counters of entrant b] for one played game row, or None."""
    try:
        side = int(row["side"])
        return [side_style(moves, side), side_style(moves, 1 - side)]
    except Exception:  # style is visibility only; never fail a batch over it
        return None


def add(total, counters):
    for k in KEYS:
        total[k] = total.get(k, 0) + counters.get(k, 0)
    total["games"] = total.get("games", 0) + 1
    best = counters.get("best")
    if best and (total.get("best") is None or best["score"] > total["best"]["score"]):
        total["best"] = dict(best)
    return total


def metrics(total, *, min_games=64):
    p = max(total.get("placements", 0), 1)
    per = lambda k: round(100.0 * total.get(k, 0) / p, 3)  # noqa: E731
    return dict(games=total.get("games", 0), placements=total.get("placements", 0),
                few_games=total.get("games", 0) < min_games,
                combos=per("combos"), chains=per("chains"), clears=per("clears"),
                lines_per_clear=round(total["lines"] / total["clears"], 2) if total.get("clears") else None,
                garbage=per("garbage"),
                share34=round(total["attacks34"] / total["attacks"], 3) if total.get("attacks") else None,
                t1=per("t1"), t2=per("t2"), t3=per("t3"),
                t1_mean=round(total["t1_score"] / total["t1"], 1) if total.get("t1") else None,
                best=total.get("best"))
