"""Write one-ply lookahead arena study configs (lookahead vs the plain champion).

usage: prepare_study.py NAME BACKEND GAMES SEED_OFFSET SPEC [SPEC ...]
  SPEC = pace:k:followups:mode:charge[:prior[:when]]   e.g. frame_perfect:4:1:value:11

Seeds: the afterstate-core-v1 panel seeds of every pace except top_humans
(reserved for the offline counterfactual study), then its quick seeds, in file
order; comparison i uses pool[SEED_OFFSET : SEED_OFFSET + GAMES/2] at every
pace. Arm A's tournament seeds are never used.
BACKEND events = spawn contract; frames = the shipped v17 contract
(lock_safe + early_preview repeat on both sides, lookahead when=early_only).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/Users/ethan/dev/drmario/drmc-rl-lookahead-data")
CHAMPION = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt"
NATIVE = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries/"
SEED_FILE = ROOT / "runs/review-20260909/afterstate-core-v1/seeds.json"

name, backend, games, offset, *specs = sys.argv[1:]
games, offset = int(games), int(offset)
seeds_json = json.loads(SEED_FILE.read_text())
pool = []
for pace, values in seeds_json["panel"].items():
    if pace != "top_humans":
        pool += values
for values in seeds_json["quick"].values():
    pool += values
if len(set(pool)) != len(pool):
    raise SystemExit("seed pool is not unique")
tournament = {s for values in seeds_json["tournament"].values() for s in values}
if tournament & set(pool):
    raise SystemExit("seed pool overlaps the tournament set")
seeds = pool[offset: offset + games // 2]
if len(seeds) != games // 2:
    raise SystemExit("not enough seeds")
v17 = backend == "frames"
contract = dict(decision_point="lock_safe", early_preview="repeat") if v17 else {}
variants = {"champion": dict(name="Mixed-v2 core" + (" · v17" if v17 else ""), delay=4, **contract)}
schedule = []
for spec in specs:
    pace, k, followups, mode, charge, *rest = spec.split(":")
    prior = float(rest[0]) if rest else 0.0
    when = rest[1] if len(rest) > 1 else ("early_only" if v17 else "always")
    vid = f"la-k{k}f{followups}-{mode}-c{charge}" + (f"-p{prior:g}" if prior else "") + ("-v17" if v17 else "")
    variants.setdefault(vid, dict(
        name=f"Lookahead K{k} F{followups} {mode} charge {charge}" + (f" prior {prior:g}" if prior else "") + (" · v17" if v17 else ""),
        delay=4, lookahead=dict(k=int(k), followups=int(followups), mode=mode, margin=0.5,
                                charge_frames=int(charge), prior=prior, when=when), **contract))
    schedule.append(dict(id=f"{name}-{pace}-{vid}", a=vid, b="champion", games=games, level=14, pace=pace,
                         phase="One-ply lookahead", rating_group="One-ply lookahead", seeds=seeds))
output = DATA / name
config = dict(
    checkpoint=CHAMPION, device="mps", native_library=NATIVE + "libdrmario_pool.dylib",
    reach_library=NATIVE + "libdrm_reach_full.dylib", native_commit="19f292c", threads=1,
    max_game_frames=120000, output=str(output), working_db=str(output / "working/arena.sqlite"),
    reactive_compute_frames=4, preparation_compute_frames=6, memoize=True,
    pairs=16 if v17 else 32, replay_games=0, watch=False, strict_fp32=True, rollout_backend=backend,
    planner_workers=3, async_planning=not v17,
    purpose="Stronger-4 one-ply value lookahead vs the plain champion; compute charged as decision delay "
            "(cost model: runs/review-20260909/one-ply-lookahead-v1/cost-model.json).",
    seed_file=str(SEED_FILE), seed_file_sha256=hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
    seed_pool_offset=offset, variants=variants, schedule=schedule)
path = DATA / f"{name}.json"
path.write_text(json.dumps(config, indent=1))
print(path)
