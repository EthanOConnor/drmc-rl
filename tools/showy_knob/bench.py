"""Knob-feature speed and identity: native (C) path vs the numpy reference, per decision.

    python -m tools.showy_knob.bench --decisions decisions.npz        # recorded decisions
    python -m tools.showy_knob.bench --synthetic 2000                  # generated bottles
    python -m tools.showy_knob.bench record --out decisions.npz --pairs 12 \
        --knobs showy-t2@1:1,showy-hcombo@1:1 --tag "tools.trainer_planning_arena "

``bench`` times ``knobs.total_bias`` per decision for showy-t2@1, showy-quad@1,
showy-hcombo@1 and two 2-knob stacks with the native library and with
``DRMC_KNOB_NATIVE=0``, checks the biases are byte-identical, and prints a
stage breakdown of the numpy path. ``record`` plays h2h games
(``tools.showy_knob_h2h``, CPU) with the given knobs installed and saves every
knob decision's inputs (root bottle, pill, candidate actions, mask); the file is
a local dataset and must not be committed.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

SPECS = ("showy-t2@1:1", "showy-quad@1:1", "showy-hcombo@1:1", "showy-t2@1:1,showy-hcombo@1:1",
         "showy-t2@1:1.5,showy-quad@1:0.5")


def load_decisions(path, limit=None):
    d = np.load(path)
    widths, fields, pills, actions, mask = (d[k] for k in ("widths", "fields", "pills", "actions", "mask"))
    n = len(widths) if limit is None else min(limit, len(widths))
    return [(fields[i], pills[i], actions[i, :widths[i]], mask[i, :widths[i]]) for i in range(n)]


def save_decisions(path, rows):
    width = max(len(a) for _, _, a, _ in rows)
    actions = np.full((len(rows), width), -1, np.int64)
    mask = np.zeros((len(rows), width), bool)
    for i, (_, _, a, m) in enumerate(rows):
        actions[i, :len(a)], mask[i, :len(m)] = a, m
    np.savez_compressed(path, fields=np.stack([np.asarray(r[0], np.uint8).reshape(128) for r in rows]),
                        pills=np.stack([np.asarray(r[1], np.int64).reshape(2) for r in rows]),
                        actions=actions, mask=mask, widths=np.asarray([len(r[2]) for r in rows]))


def resting_actions(field):
    """Every placement into two empty cells with at least one half resting on something."""
    f = np.asarray(field, np.uint8).reshape(128)
    out = []
    for o, (dr, dc) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        for cell in range(128):
            r, c = divmod(cell, 8)
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < 16 and 0 <= c2 < 8) or f[cell] != 0xFF or f[r2 * 8 + c2] != 0xFF:
                continue
            if any(rr == 15 or f[(rr + 1) * 8 + cc] != 0xFF for rr, cc in ((r, c), (r2, c2))):
                out.append(o * 128 + cell)
    return np.asarray(out, np.int64)


def synthetic_decisions(count, seed=0):
    """Deterministic decision states from random play on random virus bottles (clears favoured)."""
    from drmc_rl.eval import big_clear as bc
    from drmc_rl.game.afterstate import resolve_placement
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        field = np.full(128, 0xFF, np.uint8)
        top = int(rng.integers(4, 12))
        for cell in range(top * 8, 128):
            if rng.random() < 0.55:
                field[cell] = 0xD0 | int(rng.integers(0, 3))
        for _ in range(60):
            actions = resting_actions(field)
            if len(actions) == 0 or (field[3] != 0xFF or field[4] != 0xFF):
                break
            pill = rng.integers(0, 3, size=2)
            mask = rng.random(len(actions)) < 0.9
            mask[0] = True
            rows.append((field.copy(), pill.copy(), actions, mask))
            if len(rows) >= count:
                break
            colors = (int(pill[0]), int(pill[1]))
            clears = [a for a in actions if bc.forms_line(bc.place(field.tobytes(), colors, int(a)),
                                                          _cells(int(a)))]
            pick = int(rng.choice(clears)) if clears and rng.random() < 0.6 else int(rng.choice(actions))
            field = np.frombuffer(resolve_placement(field, colors, pick)[0], np.uint8).copy()
    return rows


def _cells(action):
    o, cell = divmod(action, 128)
    r, c = divmod(cell, 8)
    dr, dc = ((0, 1), (1, 0), (0, -1), (-1, 0))[o]
    return cell, (r + dr) * 8 + c + dc


def _use_native(on):
    from drmc_rl.style import native
    os.environ["DRMC_KNOB_NATIVE"] = "1" if on else "0"
    native.reset()
    return native.available()


def bench(rows, specs=SPECS):
    from drmc_rl.style import knobs
    from drmc_rl.style import showy_knob as sk
    out = {}
    for spec in specs:
        entries = [knobs.parse(s) for s in spec.split(",")]
        models = [knobs.load_model(e) for e in entries]
        res, times = {}, {}
        for mode in ("numpy", "native"):
            if not _use_native(mode == "native") and mode == "native":
                continue
            t = time.perf_counter()
            res[mode] = [knobs.total_bias(entries, f, p, a, m, models=models) for f, p, a, m in rows]
            times[mode] = 1e3 * (time.perf_counter() - t) / len(rows)
        same = "native" in res and all(x.tobytes() == y.tobytes() for x, y in zip(res["numpy"], res["native"]))
        out[spec] = dict(numpy_ms=times["numpy"], native_ms=times.get("native"),
                         identical=same if "native" in res else None,
                         argmax_identical=None if "native" not in res else all(
                             int(np.argmax(x)) == int(np.argmax(y)) for x, y in zip(res["numpy"], res["native"])))
        print(f"{spec:34s} numpy {times['numpy']:7.3f} ms  native "
              + (f"{times['native']:6.3f} ms  x{times['numpy'] / times['native']:5.1f}  identical {same}"
                 if "native" in times else "absent"), flush=True)
    _use_native(True)
    # stage breakdown of the numpy path (one knob's work per decision)
    stages = dict.fromkeys(("afterstate", "board_features", "trigger_features", "immediate_clears", "logistic"), 0.0)
    model = knobs.load_model(knobs.parse("showy-quad@1:1"))
    for f, p, a, m in rows:
        legal, colors = np.flatnonzero(m), (int(p[0]), int(p[1]))
        t0 = time.perf_counter()
        after = sk.reference_afterstates(np.asarray(f, np.uint8).reshape(128), colors, a[legal])
        t1 = time.perf_counter()
        x = sk.board_features(after)
        t2 = time.perf_counter()
        x = np.concatenate([x, sk.trigger_features(after)], axis=1)
        t3 = time.perf_counter()
        sk.immediate_clears(f, colors, a[legal])
        t4 = time.perf_counter()
        model.logit_features(x)
        t5 = time.perf_counter()
        for k, v in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
            stages[k] += v
    stages = {k: 1e3 * v / len(rows) for k, v in stages.items()}
    print("numpy stages (ms/decision): " + ", ".join(f"{k} {v:.3f}" for k, v in stages.items()))
    return dict(specs=out, stages=stages, decisions=len(rows),
                mean_legal=float(np.mean([np.count_nonzero(m) for *_, m in rows])))


def record(args):
    """Play h2h games with knobs installed and save each knob decision's inputs."""
    import types
    from drmc_rl.style import knobs
    from tools import showy_knob_h2h as h
    rows = []
    inner = knobs.total_bias

    def recording(kn, root_field, pill, actions, mask, *, models=None):
        rows.append((np.asarray(root_field, np.uint8).reshape(128).copy(), np.asarray(pill, np.int64).reshape(2).copy(),
                     np.asarray(actions, np.int64).copy(), np.asarray(mask, bool).copy()))
        return inner(kn, root_field, pill, actions, mask, models=models)

    knobs.total_bias = recording
    ns = types.SimpleNamespace(anchor=h.ANCHOR, checkpoint=h.ANCHOR, base_checkpoint=None, device="cpu", threads=1,
                               planner_workers=0, native_library=f"{h.LIBS}/libdrmario_pool.dylib",
                               reach_library=f"{h.LIBS}/libdrm_reach_full.dylib")
    runner = h.Runner(ns)
    base = dict(name="base", delay=4, checkpoint=h.ANCHOR)
    variants = dict(knob=dict(base, name="knob", knobs=[knobs.parse(k) for k in args.knobs.split(",")]), base=base)
    seeds = h.pick_seeds(args.pairs, args.seed_start)
    for pace in args.paces:
        for s in range(0, len(seeds), 4):
            runner.play(variants, pace, seeds[s:s + 4])
            print(pace, s, len(rows), flush=True)
    runner.runtime.close()
    save_decisions(args.out, rows)
    print("saved", len(rows), args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="bench", choices=("bench", "record"))
    ap.add_argument("--decisions", help="recorded decisions (.npz)")
    ap.add_argument("--synthetic", type=int, default=0, help="generate this many decisions instead")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=4242)
    ap.add_argument("--paces", nargs="+", default=["normal", "top_humans"])
    ap.add_argument("--knobs", default="showy-t2@1:1,showy-hcombo@1:1")
    ap.add_argument("--tag", default="", help="ignored (process-visibility marker for pool workers)")
    args = ap.parse_args(argv)
    if args.action == "record":
        return record(args)
    rows = load_decisions(args.decisions, args.limit) if args.decisions else synthetic_decisions(args.synthetic or 500)
    report = bench(rows)
    print(f"{report['decisions']} decisions, {report['mean_legal']:.1f} legal candidates each")


if __name__ == "__main__":
    main()
