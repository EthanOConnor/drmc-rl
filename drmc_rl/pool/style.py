"""Combo and showiness counters per entrant (visibility only; never part of a rating).

Workers resolve every placement of both sides exactly with the big-clear
scorer (``drmc_rl.eval.big_clear``, the same definition as the big-clear
setups experiment and ``tools.big_clear_pool_metrics``) and report compact
per-side counters with each game. The coordinator only sums counters.

Definitions per placement: a *clear* resolves at least one line; a *combo*
has 2+ lines in the resolution or 2+ cascade rounds; a *chain* has 2+ rounds;
*garbage* is the ROM's min(lines, 4) pieces for 2+ lines; ``t1p``/``t2``/``t3`` count
cumulative showiness >= 27 / 30 / 42 (``big_clear.TIERS``) and ``t1`` the old bar 20
(``big_clear.LEGACY_T1``); a plain 4-line clear scores 0.
"""
from __future__ import annotations

KEYS = ("placements", "clears", "lines", "combos", "chains", "garbage", "attacks", "attacks34",
        "t1", "t2", "t3", "t1_score")
# Counted only by workers since the T1 bar moved to 27 (big_clear.TIERS); rates use the
# placements and lines of the games that carry them, so older games never dilute them.
KEYS_V2 = ("placements2", "lines2", "t1p", "hlines")
# Counted since the horizontal-clear share was added: clears with at least one horizontal
# line of 4+ (the definition of big_clear_pool_metrics' horizontal_share), and those clears.
KEYS_V3 = ("clears3", "hclears")

# Human reference rows, per 100 placements (14-Hi-rated corpus analysis): old T1 (>=20),
# T1+ (>=27), T2+ (>=30), T3+ (>=42), share of clears containing a horizontal line (4+).
HUMAN_ROWS = (
    dict(label="Humans >2000", t1=1.25, t1p=0.442, t2=0.254, t3=0.022, horizontal=0.288),
    dict(label="Top 5 by 14-Hi", t1=1.49, t1p=0.557, t2=0.329, t3=0.032, horizontal=0.306),
    dict(label="All humans", t1=0.855, t1p=0.279, t2=0.156, t3=0.013, horizontal=0.241),
)

# Human reference rates from the full Fightcade corpus (drmc-rl-bigclear-data
# mining-summary.json: 57,341,660 placements, 20,180,178 clears), per 100 placements.
HUMAN = dict(combos=15.6, chains=12.8, clears=35.2, garbage=34.1, share34=0.157, t1=0.85, t1p=0.28, t2=0.15, t3=0.012)


def side_style(moves, physical):
    """Counters for the side playing ``physical`` in one game's move journal."""
    from drmc_rl.eval import big_clear as bc
    t1p, t2, t3 = (bar for _, bar in bc.TIERS)          # cumulative T1+ (27), T2+ (30), T3+ (42)
    t1 = bc.LEGACY_T1                                    # the original T1 bar (20), kept as "t1"
    out = dict.fromkeys(KEYS + KEYS_V2 + KEYS_V3, 0)
    out["t1_score"] = 0.0
    best = None
    for move in moves:
        if int(move["side"]) != physical:
            continue
        out["placements"] += 1
        out["placements2"] += 1
        try:
            f = bc.placement_features(bytes(move["board"]), move["pill"], int(move["placement"]["action"]))
        except (ValueError, KeyError, TypeError):
            continue
        if not f.rounds:
            continue
        score = f.score()
        out["clears"] += 1
        out["lines"] += f.lines
        out["lines2"] += f.lines
        out["hlines"] += getattr(f, "horizontal_lines", 0)
        out["clears3"] += 1
        out["hclears"] += int(getattr(f, "horizontal_lines", 0) > 0)
        out["t1p"] += int(score >= t1p)
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
    for k in KEYS + KEYS_V2 + KEYS_V3:
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
                t1p=round(100.0 * total["t1p"] / total["placements2"], 3) if total.get("placements2") else None,
                # Share of clears with a horizontal line (comparable with the human rows)...
                horizontal=round(total["hclears"] / total["clears3"], 3) if total.get("clears3") else None,
                # ...and the share of matched lines that are horizontal.
                horizontal_lines=round(total["hlines"] / total["lines2"], 3) if total.get("lines2") else None,
                t1_mean=round(total["t1_score"] / total["t1"], 1) if total.get("t1") else None,
                best=total.get("best"))
