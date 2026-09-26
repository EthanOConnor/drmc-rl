"""Per-placement skill statistics over ``tools.skill_gaps.extract`` rows (vectorized numpy).

    python -m tools.skill_gaps.features --rows 'data/human-*.npz' --out feats-human.npz [--trigger-fraction 0.2]

Every statistic is a pure function of the stored arrays (root bottle, settled afterstate,
opponent bottle, action, resolution) plus the next root of the same sequence, so the
human corpus and the pool traces are measured identically. Names and meanings:

danger       h34 (max stack height in spawn columns 3-4 after the placement), danger12
             (h34 >= 12), top3 (occupied cells in rows 0-2 of columns 3-4), near_topout
             (rows 0-1 of columns 3-4 occupied), recover6 (from danger12, h34 <= 9 within
             6 placements; NaN elsewhere)
efficiency   vcleared (viruses cleared by this placement), clear (any line), vir_root
             (viruses on the root bottle), nonvirus_clear (a clear without a virus)
setups       threats (near-complete lines: 3-of-4 supported horizontals + 3-deep surface
             runs), threat_colors (colors holding one), h2s, v3, trig_n / trig_attack /
             trig_quad (single-tile drops that clear / attack / quad, exact; subsample)
holes        holes, covered_viruses (virus with a different-colored pill tile directly
             above), buried (occupied cells above viruses, per virus), exposed_frac
             (viruses with nothing above), d_covered (covered viruses added by this placement)
stranding    isolated (pill tiles without a same-color orthogonal neighbour),
             isolated_covered (those with a different color directly above),
             edge_strand (stranded edge virus, ``stranded_edge`` definition: 1-3 viruses,
             edge column, 4+ empty cells below)
garbage      garb_in (tiles that arrived between this afterstate and the next root),
             garb_holes (holes those tiles added), counter4 (after garbage arrived: an
             attack within the next 4 placements; NaN elsewhere)
attack       attack (2+ lines), quad (4+), waste (lines beyond 4), garbage_sent,
             opp_h34 / opp_hmax (opponent bottle at the decision)
placement    horizontal, col, row (pill's top row), match_contact (pill halves touching a
             same-color tile on the root bottle), tempo (frames to the next decision)
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

EMPTY = 0xFF


def planes(fields):
    f = np.asarray(fields, np.uint8).reshape(-1, 16, 8)
    occ = f != EMPTY
    color = np.where(occ, f & 3, 3).astype(np.int8)
    virus = occ & ((f & 0xF0) == 0xD0)
    return occ, color, virus


def heights(occ):
    return 16 - np.where(occ.any(axis=1), occ.argmax(axis=1), 16)


def shift(a, dr, dc, fill):
    """out[r, c] = a[r + dr, c + dc] (fill outside)."""
    out = np.full_like(a, fill)
    n, h, w = a.shape
    rs, re = max(0, -dr), min(h, h - dr)
    cs, ce = max(0, -dc), min(w, w - dc)
    out[:, rs:re, cs:ce] = a[:, rs + dr:re + dr, cs + dc:ce + dc]
    return out


def board_stats(fields) -> dict[str, np.ndarray]:
    occ, color, virus = planes(fields)
    n = occ.shape[0]
    h = heights(occ)
    pill = occ & ~virus
    above_occ = shift(occ, -1, 0, False)
    above_color = shift(color, -1, 0, 3)
    above_pill = shift(pill, -1, 0, False)
    occ_above = np.cumsum(occ, axis=1) - occ
    nvir = virus.sum(axis=(1, 2))
    same = np.zeros_like(occ)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        same |= shift(occ, dr, dc, False) & (shift(color, dr, dc, 3) == color)
    iso = pill & ~same
    empty = ~occ
    above_clear = np.cumsum(occ, axis=1) == 0
    out = dict(
        h34=h[:, 3:5].max(axis=1).astype(np.float32),
        hmax=h.max(axis=1).astype(np.float32),
        top3=occ[:, :3, 3:5].sum(axis=(1, 2)).astype(np.float32),
        near_topout=occ[:, :2, 3:5].any(axis=(1, 2)).astype(np.float32),
        holes=(empty & ~above_clear).sum(axis=(1, 2)).astype(np.float32),
        bumpiness=np.abs(np.diff(h, axis=1)).sum(axis=1).astype(np.float32),
        covered_viruses=(virus & above_pill & (above_color != color)).sum(axis=(1, 2)).astype(np.float32),
        buried=((occ_above * virus).sum(axis=(1, 2)) / np.maximum(nvir, 1)).astype(np.float32),
        exposed_frac=np.where(nvir > 0, (virus & (occ_above == 0)).sum(axis=(1, 2)) / np.maximum(nvir, 1), np.nan
                              ).astype(np.float32),
        isolated=iso.sum(axis=(1, 2)).astype(np.float32),
        isolated_covered=(iso & above_occ & (above_color != color)).sum(axis=(1, 2)).astype(np.float32),
        viruses=nvir.astype(np.float32),
        pill_cells=pill.sum(axis=(1, 2)).astype(np.float32),
    )
    # stranded edge virus (drmc_rl.eval.stranded_edge defaults, without the other-cells cap)
    strand = np.zeros(n, np.float32)
    for i in np.flatnonzero((nvir >= 1) & (nvir <= 3)):
        for col in (0, 7):
            for r in np.flatnonzero(virus[i, :, col]):
                below = occ[i, r + 1:, col]
                gap = int(below.argmax()) if below.any() else 15 - r
                if gap >= 4:
                    strand[i] = 1
    out["edge_strand"] = strand
    return out


def setup_stats(fields) -> dict[str, np.ndarray]:
    from drmc_rl.style import showy_knob as sk
    x = sk.board_features(fields)
    idx = {k: i for i, k in enumerate(sk.FEATURE_NAMES)}
    g = lambda k: x[:, idx[k]]
    return dict(threats=g("threats"), h3s=g("h3s_sum"), h2s=g("h2s_sum"), v3=g("v3_sum"), surf_run3=g("surf_run3_sum"),
                threat_colors=np.maximum.reduce([g("h3s_colors"), g("surf_run3_colors"), g("v3_colors")]),
                vir_h3=g("vir_h3_sum") + g("vir_v3_sum"))


def placement_stats(d) -> dict[str, np.ndarray]:
    action = d["action"].astype(np.int64)
    o, cell = np.divmod(action, 128)
    row, col = np.divmod(cell, 8)
    occ, color, _ = planes(d["root"])
    second = np.array(((0, 1), (1, 0), (0, -1), (-1, 0)))
    touches = np.zeros(len(action), np.float32)
    n = len(action)
    ar = np.arange(n)
    for half, (r, c) in enumerate(((row, col), (row + second[o, 0], col + second[o, 1]))):
        pc = d["pill"][:, half].astype(np.int64)
        nes = np.array((1, 0, 2))[pc]
        hit = np.zeros(n, bool)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            ok = (rr >= 0) & (rr < 16) & (cc >= 0) & (cc < 8)
            rr, cc = np.clip(rr, 0, 15), np.clip(cc, 0, 7)
            hit |= ok & occ[ar, rr, cc] & (color[ar, rr, cc] == nes)
        touches += hit
    top = np.minimum(row, row + second[o, 0])
    lines = d["lines"].astype(np.float32)
    return dict(horizontal=((o % 2) == 0).astype(np.float32), col=col.astype(np.float32), row=top.astype(np.float32),
                match_contact=touches / 2, vcleared=d["vcleared"].astype(np.float32),
                clear=(d["rounds"] > 0).astype(np.float32),
                nonvirus_clear=((d["rounds"] > 0) & (d["vcleared"] == 0)).astype(np.float32),
                attack=(lines >= 2).astype(np.float32), quad=(lines >= 4).astype(np.float32),
                waste=np.maximum(lines - 4, 0), garbage_sent=np.where(lines >= 2, np.minimum(lines, 4), 0).astype(np.float32),
                t2=(d["score"] >= 30).astype(np.float32))


def sequence_stats(d, a, r, o) -> dict[str, np.ndarray]:
    """Statistics that look at the next placements of the same sequence."""
    seq = d["seq"]
    n = len(seq)
    nxt_same = np.r_[seq[1:] == seq[:-1], False]
    occ_a, _, _ = planes(d["after"])
    occ_n = np.zeros_like(occ_a)
    occ_n[:-1] = planes(d["root"][1:])[0]
    arrived = (~occ_a) & occ_n
    garb = np.where(nxt_same, arrived.sum(axis=(1, 2)), 0).astype(np.float32)
    holes_next = np.r_[r["holes"][1:], 0]
    garb_holes = np.where(garb > 0, holes_next - a["holes"], np.nan).astype(np.float32)
    frame = d["frame"].astype(np.float64)
    tempo = np.where(nxt_same, np.r_[frame[1:] - frame[:-1], 0], np.nan).astype(np.float32)
    # windows over the next k placements of the same sequence
    last = np.r_[np.flatnonzero(np.diff(seq)), n - 1]
    end = np.repeat(last, np.diff(np.r_[-1, last]))
    idx = np.arange(n)

    def within(event, k, start=1):
        cs = np.r_[0, np.cumsum(event.astype(np.int32))]
        lo, hi = np.minimum(idx + start, end + 1), np.minimum(idx + k, end) + 1
        return (cs[np.maximum(hi, lo)] - cs[lo]) > 0

    attack = d["lines"] >= 2
    counter4 = np.where(garb > 0, within(attack, 4, start=1), np.nan).astype(np.float32)
    danger = a["h34"] >= 12
    recover6 = np.where(danger, within(a["h34"] <= 9, 6), np.nan).astype(np.float32)
    return dict(garb_in=garb, garb_holes=garb_holes, counter4=counter4, recover6=recover6, tempo=tempo,
                d_covered=(a["covered_viruses"] - r["covered_viruses"]).astype(np.float32))


def compute(d, trigger_fraction=0.0, seed=0) -> dict[str, np.ndarray]:
    a = board_stats(d["after"])
    r = board_stats(d["root"])
    o = board_stats(d["opp"])
    out = {k: a[k] for k in ("h34", "hmax", "top3", "near_topout", "holes", "bumpiness", "covered_viruses", "buried",
                             "exposed_frac", "isolated", "isolated_covered", "edge_strand", "pill_cells")}
    out["vir_root"] = r["viruses"]
    out["danger12"] = (a["h34"] >= 12).astype(np.float32)
    out["opp_h34"] = o["h34"]
    out["opp_hmax"] = o["hmax"]
    out.update(setup_stats(d["after"]))
    out.update(placement_stats(d))
    out.update(sequence_stats(d, a, r, o))
    if trigger_fraction > 0:
        from drmc_rl.style import showy_knob as sk
        rng = np.random.default_rng(seed)
        pick = np.flatnonzero(rng.random(len(d["seq"])) < trigger_fraction)
        trig = np.full((len(d["seq"]), len(sk.TRIGGER_NAMES)), np.nan, np.float32)
        if len(pick):
            trig[pick] = sk.trigger_features(d["after"][pick])
        names = list(sk.TRIGGER_NAMES)
        for k in ("trig_n", "trig_attack", "trig_quad", "trig_multi"):
            out[k] = trig[:, names.index(k)]
    return out


META = ("seq", "game", "t", "won", "group", "rating", "pace", "lines", "rounds", "vcleared", "score", "hlines")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="glob of extract npz files (sequences renumbered across files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trigger-fraction", type=float, default=0.0)
    ap.add_argument("--keep-after", action="store_true", help="also store afterstates (knob fitting)")
    args = ap.parse_args(argv)
    parts, offset, goffset, meta = [], 0, 0, []
    for path in sorted(glob.glob(args.rows)):
        d = dict(np.load(path))
        meta.append(json.loads(str(d.pop("meta"))))
        feats = compute(d, args.trigger_fraction)
        feats.update({k: d[k] for k in META})
        feats["seq"] = d["seq"] + offset
        feats["game"] = d["game"] + goffset
        if args.keep_after:
            feats["after"] = d["after"]
        offset = int(feats["seq"].max()) + 1 if len(feats["seq"]) else offset
        goffset = int(feats["game"].max()) + 1 if len(feats["game"]) else goffset
        parts.append(feats)
        print(path, len(d["seq"]), flush=True)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    players = {}
    for m in meta:
        players.update(m.get("players", {}))
    np.savez_compressed(args.out, meta=json.dumps(dict(sources=[{k: v for k, v in m.items() if k != "players"} for m in meta],
                                                       players=players)), **out)


if __name__ == "__main__":
    main()
