"""Gradient noise scale of the controller-retention PPO objective.

Measures B_simple = tr(Sigma) / |G|^2 (McCandlish et al. 2018) for each loss
term of `tools/train_controller_retention.py`, at saved checkpoints, on the
trainer's own persisted on-policy collections (`public-replay/update-*.npz`).

A checkpoint saved after update u is evaluated on the collections of updates
u+1 .. u+C. Update u+1 was collected by exactly these weights (verified); later
updates are a few thousandths of KL away and enter through the PPO ratio.

Estimators (all gradients are sums of per-decision loss gradients, exactly as
the trainer's minibatch losses are means of per-decision terms):

* |G|^2: mean inner product of full-collection gradients from DIFFERENT
  updates (independent rollouts at the same weights), so no noise bias.
* random-decision variance s2_dec: one pass over a random permutation in
  128-decision chunks; residuals against the full-collection gradient with the
  without-replacement correction. This is the trainer's minibatch sampling.
* game-grouped variance: one gradient per complete game; residuals against the
  collection mean. The per-decision equivalent s2_game / mean_length is the
  variance a random sample of whole games contributes per decision.
* update-level variance: half the squared difference between collections.
* retention: the trainer's own `PaceRetention.loss` draws versus the exact
  expected retention gradient over the anchor bank.

Per-game and per-chunk vectors are CountSketched (unbiased inner products);
full-collection and retention gradients are exact. `measure` runs on the GPU
host and writes a compact npz; `analyze` is numpy only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import time

import numpy as np

TERMS = ("policy_loss", "value_loss", "entropy", "parent_kl")
COMBOS = {"policy": (1, 0, 0, 0), "value": (0, 1, 0, 0), "entropy": (0, 0, 1, 0),
          "parent_kl": (0, 0, 0, 1), "ppo": (1, 1, 1, 1)}


# --------------------------------------------------------------------------- data

def natural_games(journal):
    counts = Counter()
    with open(journal) as handle:
        for line in handle:
            row = json.loads(line)
            if row["reason"] != "timeout":
                counts[row["update"], row["pace"]] += 1
    return counts


def load_collection(replay_dir, update, paces, natural):
    """Rebuild the trainer's PPO records for one update from its replay shards."""
    records = []
    for pace in paces:
        path = Path(replay_dir) / f"update-{update:05d}-{pace}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        meta = json.loads(str(z["metadata"]))
        if meta["update"] != update or meta["pace"] != pace:
            raise ValueError(f"replay metadata mismatch in {path}")
        offsets = z["offsets"]
        keys = list(zip(z["game_seed"].tolist(), z["learner_port"].tolist()))
        lengths = Counter(keys)
        arrays = {k: z[k] for k in ("observation", "pill", "preview", "public_context", "actions",
                                     "costs", "base_logits", "behavior_logp", "slot", "return",
                                     "old_logprob", "old_value")}
        for i, key in enumerate(keys):
            lo, hi = offsets[i], offsets[i + 1]
            records.append(dict(
                observation=arrays["observation"][i], pill=arrays["pill"][i],
                preview=arrays["preview"][i], public_context=arrays["public_context"][i],
                actions=arrays["actions"][lo:hi], costs=arrays["costs"][lo:hi],
                mask=np.ones(hi - lo, bool), base_logits=arrays["base_logits"][lo:hi],
                behavior_logp=arrays["behavior_logp"][lo:hi], slot=int(arrays["slot"][i]),
                old_logprob=float(arrays["old_logprob"][i]), old_value=float(arrays["old_value"][i]),
                **{"return": float(arrays["return"][i])},
                weight=1 / lengths[key], episode_length=lengths[key], pace=pace,
                game=(pace, *key)))
    present = {r["pace"] for r in records}
    if not records or present != set(paces):
        raise FileNotFoundError(f"update {update} replay incomplete: {sorted(present)}")
    completed = {p: natural[update, p] for p in paces}
    if not all(completed.values()):
        # An interrupted update has replay but no journal rows. Natural games with
        # zero learner decisions are then uncounted (they are rare; see report).
        completed = Counter(r["game"][0] for r in {r["game"]: r for r in records}.values())
        completed = {p: completed[p] for p in paces}
    return records, completed


# ------------------------------------------------------------------------ measure

def measure(args):
    import torch
    import torch.nn.functional as F
    from drmc_rl.models.policy.controller_core import ControllerCorePolicy
    from drmc_rl.training.controller_retention import PaceRetention, balance_pace_credit
    from drmc_rl.training.episodic_objective import categorical_kl, objective_contract
    from drmc_rl.training.utils.checkpoint_io import load_checkpoint
    from tools.train_pace_strategy import (prepare_training_records, training_loss_terms,
                                           weighted_training_terms)

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    config["objective"] = objective_contract(config)
    progress = json.loads((run / "training.json").read_text())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = args.device
    ckpt = Path(args.checkpoint)
    payload = load_checkpoint(ckpt, map_location="cpu")
    update = int(payload["update"])
    exp_avg_sq = None
    if args.metric == "adam":
        # AdamW was built over ALL net parameters (one group), but creates state
        # only for parameters that ever received a gradient. The WDL heads get
        # none from PPO or retention, so state has 187 of 191 entries; map by
        # the group's parameter index rather than assuming a dense list.
        group = payload["optimizer"]["param_groups"]
        if len(group) != 1:
            raise ValueError("expected the trainer's single AdamW parameter group")
        group, state = group[0], payload["optimizer"]["state"]
        beta2, eps = group["betas"][1], group["eps"]
        exp_avg_sq = {}
        for position, index in enumerate(group["params"]):
            if index in state:
                s_ = state[index]
                exp_avg_sq[position] = s_["exp_avg_sq"] / (1 - beta2 ** float(s_["step"]))
        group_size = len(group["params"])
    del payload
    actor = ControllerCorePolicy(config["checkpoint"], device, resume=ckpt, training=True,
                                 seed=int(config["seed"]))
    net = actor.net
    params = [p for p in net.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    if exp_avg_sq is not None:
        every = list(net.parameters())
        if group_size != len(every) or any(v.shape != every[i].shape for i, v in exp_avg_sq.items()):
            raise ValueError("optimizer state does not match the network's parameters")
        # Metric sqrt(D), D = 1/(sqrt(v_hat)+eps): tr(D Sigma)/(G' D G). Stateless
        # parameters never receive a gradient, so their metric value is immaterial.
        position = {id(p): i for i, p in enumerate(every)}
        metric = torch.cat([
            ((1 / (exp_avg_sq[position[id(p)]].sqrt() + eps)).sqrt() if position[id(p)] in exp_avg_sq
             else torch.ones_like(p, device="cpu")).reshape(-1) for p in params]).to(device)
        stateless = [i for i in range(len(every)) if i not in exp_avg_sq]
        print(f"adam metric: {len(exp_avg_sq)}/{len(every)} parameters with state; stateless {stateless}", flush=True)
        del exp_avg_sq
    else:
        metric = None

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    d = args.sketch_dim
    bucket = torch.randint(0, d, (P,), generator=gen).to(device)
    sign = (torch.randint(0, 2, (P,), generator=gen) * 2 - 1).to(device, torch.float32)

    def flat(grads):
        v = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                       for g, p in zip(grads, params)])
        return v * metric if metric is not None else v

    def sketch(v):
        return torch.zeros(d, device=device, dtype=torch.float64).index_add_(
            0, bucket, (v * sign).double())

    def row_terms(rows):
        """Per-decision coefficient-weighted terms; their means equal the trainer's."""
        features, data = actor.training_batch(rows)
        logits, values = actor.training_forward(features)
        log_probs = logits.log_softmax(-1)
        chosen = log_probs.gather(1, data["slot"][:, None]).squeeze(1)
        ratio = (chosen - data["old_logprob"]).exp()
        clip = config.get("clip", 0.15)
        adv = data["advantage"]
        policy = -(data["actor_weight"] * torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv))
        value = data["value_weight"] * F.smooth_l1_loss(values, data["return"], reduction="none")
        entropy = -(data["entropy_weight"] * (log_probs.exp() * log_probs).sum(-1))
        kl = data["parent_kl_weight"] * categorical_kl(data["parent_logp"], log_probs)
        coefficient = (1.0, config.get("value_coefficient", 0.5), -config.get("entropy", 0.003),
                       config.get("parent_kl", 0.02))
        terms = [c * t for c, t in zip(coefficient, (policy, value, entropy, kl))]
        return terms, features, data, chosen

    def term_sums(rows):
        terms, *_ = row_terms(rows)
        out = []
        for i, t in enumerate(terms):
            out.append(flat(torch.autograd.grad(t.sum(), params, allow_unused=True,
                                                retain_graph=i + 1 < len(terms))))
        return out

    # Retention: the trainer's pressure at the start of an update from this checkpoint.
    retention = PaceRetention(actor, config["anchor_banks"], excluded_seeds=config["holdout_seeds"],
        paces=config["paces"], max_kl_increase=config.get("max_anchor_kl_increase", .03),
        coefficient=config.get("retention_coefficient", .1),
        batch_size=config.get("retention_batch_size", 64),
        pressure_strength=config.get("retention_pressure_strength", 0.))
    retention.baseline = progress["retention_baseline"]
    measured = retention.measure()
    retention.set_pressure(measured)
    G_R = torch.zeros(P, device=device)
    for p_index, pace in enumerate(retention.paces):
        rows, weights = retention.by_pace[pace], retention.weights[pace]
        for start in range(0, len(rows), 128):
            kl = retention._kl(rows[start:start + 128])
            w = torch.as_tensor(weights[start:start + 128], device=device, dtype=torch.float32)
            scale = retention.coefficient * retention.pressure[pace] / len(retention.paces)
            G_R += flat(torch.autograd.grad(scale * (w * kl).sum(), params, allow_unused=True))
    rrng = np.random.default_rng(args.seed ^ 0x71A90)
    ret_draws = []
    for _ in range(args.retention_draws):
        g = flat(torch.autograd.grad(retention.loss(rrng), params, allow_unused=True))
        ret_draws.append(float((g - G_R).double().square().sum()))
    retention_rows = max(1, retention.batch_size // len(retention.paces)) * len(retention.paces)

    natural = natural_games(run / "training-games.jsonl")
    log = {}
    if args.log and Path(args.log).exists():
        for line in Path(args.log).read_text().splitlines():
            if line.startswith('{"updates"'):
                row = json.loads(line)
                log[row["updates"]] = row["losses"]

    exact = []           # full-collection mean gradients, per collection and term
    games_meta = []      # (collection, pace index, length)
    game_sketch = []     # per game: [4, d]
    chunk_out = []
    collections = []
    targets = args.updates or [update + 1 + c for c in range(args.collections)]
    for c, u in enumerate(targets):
        try:
            records, completed = load_collection(run / "public-replay", u, config["paces"], natural)
        except FileNotFoundError as error:
            print(f"stop at collection {c}: {error}", flush=True)
            break
        started = time.monotonic()
        inverse, center, scale = prepare_training_records(records, config)
        balance_pace_credit(records, completed)
        N = len(records)
        info = dict(update=u, decisions=N, completed_learning_games=int(round(inverse.sum())),
                    advantage_center=center, advantage_scale=scale, natural_games=completed)
        if u in log:
            info["log_advantage_center"] = log[u]["advantage_center"]
            info["log_advantage_scale"] = log[u]["advantage_scale"]
            info["log_initial_value_mse"] = log[u]["initial_value_mse"]
            if abs(log[u]["advantage_scale"] - scale) > 1e-6 or abs(log[u]["advantage_center"] - center) > 1e-6:
                raise RuntimeError(f"rebuilt update {u} records differ from the trainer's normalization")
        # Validate the trainer's loss and (for u+1) exact on-policy likelihoods.
        with torch.no_grad():
            probe = records[:256]
            terms, features, data, chosen = row_terms(probe)
            reference = weighted_training_terms(training_loss_terms(actor, features, data, config)[1], config)
            for t, name in zip(terms, TERMS):
                if abs(float(t.mean()) - float(reference[name])) > 1e-5 * max(1, abs(float(reference[name]))):
                    raise RuntimeError(f"per-decision {name} does not reproduce the trainer's loss")
            errors = []
            for start in range(0, N, 512):
                _, _, dd, ch = row_terms(records[start:start + 512])
                errors.append(float((ch - dd["old_logprob"]).abs().max()))
            info["max_logprob_shift"] = max(errors)
            if u == update + 1 and info["max_logprob_shift"] > 1e-3:
                raise RuntimeError("checkpoint is not the behavior policy of its next collection")

        # Game pass: exact collection gradient and per-game sketches.
        by_game = defaultdict(list)
        for r in records:
            by_game[r["game"]].append(r)
        total = [torch.zeros(P, device=device) for _ in TERMS]
        sketches = []
        for key, rows in by_game.items():
            vectors = term_sums(rows)
            for t in range(len(TERMS)):
                total[t] += vectors[t]
            sketches.append(torch.stack([sketch(v) for v in vectors]).float())
            games_meta.append((c, config["paces"].index(key[0]), len(rows)))
        game_sketch.append(torch.stack(sketches))
        mean = [v / N for v in total]
        exact.append(mean)
        del total

        # Random-decision pass(es): the trainer's minibatch sampling.
        perm_rng = np.random.default_rng(args.seed + 7919 * u)
        for r_index in range(args.permutations):
            order = perm_rng.permutation(N)
            sizes, residual, chunk_sk = [], [], []
            for start in range(0, N, args.chunk):
                rows = [records[i] for i in order[start:start + args.chunk]]
                vectors = term_sums(rows)
                n = len(rows)
                res = [v - n * g for v, g in zip(vectors, mean)]
                gram = torch.stack(res).double()
                residual.append((gram @ gram.T).cpu().numpy())
                chunk_sk.append(torch.stack([sketch(v) for v in vectors]).float())
                sizes.append(n)
            chunk_sk = torch.stack(chunk_sk)                       # [K, 4, d]
            chunk_out.append(dict(collection=c, sizes=np.asarray(sizes), exact_residual_gram=np.stack(residual),
                                  sketch_gram=torch.einsum("ktd,lsd->ktls", chunk_sk.double(), chunk_sk.double()).cpu().numpy()))
            del chunk_sk
        info["seconds"] = time.monotonic() - started
        collections.append(info)
        print(json.dumps(info), flush=True)
        del records, by_game

    C = len(exact)
    if C < 2:
        raise RuntimeError("at least two collections are needed for an unbiased |G|^2")
    vectors = torch.stack([v for mean in exact for v in mean] + [G_R]).double()
    exact_gram = (vectors @ vectors.T).cpu().numpy()
    del vectors
    sk = torch.cat(game_sketch)                                        # [J, 4, d] float32
    game_gram = []
    for q in COMBOS.values():
        v = torch.einsum("t,jtd->jd", torch.as_tensor(q, device=device, dtype=torch.float32), sk)
        game_gram.append((v @ v.T).double().cpu().numpy())
        del v
    game_gram = np.stack(game_gram)                                    # [combo, J, J]
    del sk, game_sketch
    np.savez_compressed(args.out,
        checkpoint=str(ckpt), update=update, frames=progress_frames(args.log, update),
        metric=args.metric, parameters=P, sketch_dim=d, terms=np.asarray(TERMS),
        paces=np.asarray(config["paces"]), collections=json.dumps(collections),
        exact_gram=exact_gram, game_meta=np.asarray(games_meta), game_gram=game_gram, combos=np.asarray(list(COMBOS)),
        chunk_collection=np.asarray([x["collection"] for x in chunk_out]),
        chunk_sizes=np.asarray([x["sizes"] for x in chunk_out], dtype=object),
        chunk_exact=np.asarray([x["exact_residual_gram"] for x in chunk_out], dtype=object),
        chunk_sketch=np.asarray([x["sketch_gram"] for x in chunk_out], dtype=object),
        retention_draws=np.asarray(ret_draws), retention_rows=retention_rows,
        retention_kl=json.dumps(measured), retention_pressure=json.dumps(retention.pressure),
        config=json.dumps({k: config[k] for k in ("lr", "epochs", "minibatch", "clip", "entropy", "parent_kl",
                                                  "value_coefficient", "retention_coefficient",
                                                  "retention_batch_size", "max_update_kl")}),
        created=datetime.now(UTC).isoformat())
    print(f"wrote {args.out}", flush=True)


def progress_frames(log, update):
    if log and Path(log).exists():
        for line in Path(log).read_text().splitlines():
            if line.startswith('{"updates"'):
                row = json.loads(line)
                if row["updates"] == update:
                    return row["frames"]
    return 0


# -------------------------------------------------------------------------- bench

def bench(args):
    """Wall-clock of one trainer optimizer step (batch build, loss, retention, Adam)."""
    import torch
    from drmc_rl.models.policy.controller_core import ControllerCorePolicy
    from drmc_rl.training.controller_retention import PaceRetention, balance_pace_credit
    from drmc_rl.training.episodic_objective import objective_contract
    from tools.train_pace_strategy import (_policy_snapshot, prepare_training_records,
                                           training_loss_terms, weighted_training_terms)

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    config["objective"] = objective_contract(config)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actor = ControllerCorePolicy(config["checkpoint"], args.device, resume=args.checkpoint, training=True,
                                 seed=int(config["seed"]))
    retention = PaceRetention(actor, config["anchor_banks"], excluded_seeds=config["holdout_seeds"],
        paces=config["paces"], coefficient=config.get("retention_coefficient", .1),
        batch_size=config.get("retention_batch_size", 64),
        pressure_strength=config.get("retention_pressure_strength", 0.))
    update = int(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["update"])
    records, completed = load_collection(run / "public-replay", update + 1, config["paces"],
                                         natural_games(run / "training-games.jsonl"))
    prepare_training_records(records, config)
    balance_pace_credit(records, completed)
    # Full-collection passes first, at the unchanged behavior weights (audit mode).
    snapshots = {}
    for size in args.sizes:
        try:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _policy_snapshot(actor, records, size)
            torch.cuda.synchronize(); snapshots[size] = time.perf_counter() - t0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    t0 = time.perf_counter(); retention.measure(); torch.cuda.synchronize()
    retention_measure = time.perf_counter() - t0
    optimizer = torch.optim.AdamW(actor.net.parameters(), lr=config["lr"], weight_decay=.001)
    rng, rrng = np.random.default_rng(0), np.random.default_rng(1)
    results = []
    for size in args.sizes:
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            if size not in snapshots:
                raise torch.cuda.OutOfMemoryError
            for step in range(args.steps + 1):
                rows = [records[i] for i in rng.choice(len(records), size, replace=False)]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                features, data = actor.training_batch(rows)
                _, terms = training_loss_terms(actor, features, data, config)
                loss = sum(weighted_training_terms(terms, config).values()) + retention.loss(rrng)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.net.parameters(), 0.7)
                optimizer.step()
                torch.cuda.synchronize()
                if step:
                    times.append(time.perf_counter() - t0)
            snapshot = snapshots[size] + retention_measure
        except torch.cuda.OutOfMemoryError:
            results.append(dict(minibatch=size, error="cuda out of memory"))
            torch.cuda.empty_cache()
            continue
        step_s = float(np.median(times))
        steps = -(-len(records) // size)
        epochs = config.get("epochs", 2)
        # Per epoch: the SGD steps plus one full-collection KL check and retention measure;
        # one more pair before the first epoch (collection audit, initial retention).
        results.append(dict(minibatch=size, step_seconds=step_s, check_seconds=snapshot,
            peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
            projected_optimizer_seconds=epochs * (steps * step_s + snapshot) + snapshot,
            decisions=len(records)))
        print(json.dumps(results[-1]), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


# ------------------------------------------------------------------------ analyze

def analyze_one(path, *, boots=400, seed=0):
    z = np.load(path, allow_pickle=True)
    collections = json.loads(str(z["collections"]))
    C, T = len(collections), len(TERMS)
    N = np.asarray([c["decisions"] for c in collections], float)
    E = z["exact_gram"]
    meta = z["game_meta"]
    coll, pace, L = meta[:, 0], meta[:, 1], meta[:, 2].astype(float)
    paces = [str(p) for p in z["paces"]]
    ret = z["retention_draws"]
    R = int(z["retention_rows"])
    rng = np.random.default_rng(seed)
    game_gram = z["game_gram"]          # load once; npz members decompress per access

    def exact_index(c, t):
        return c * T + t

    out = dict(checkpoint=str(z["checkpoint"]), update=int(z["update"]), frames=int(z["frames"]),
               metric=str(z["metric"]), parameters=int(z["parameters"]), collections=collections,
               mean_decisions=float(N.mean()), mean_game_length=float(L.mean()),
               combos={})
    GR2 = E[-1, -1]
    # Retention per-step noise at the trainer's retention batch; per-row units.
    ret_step = float(ret.mean())
    out["retention"] = dict(G2=float(GR2), step_noise=ret_step, rows=R, row_variance=ret_step * R,
                            B_simple_rows=ret_step * R / GR2,
                            step_noise_ci=[float(x) for x in np.percentile(
                                [rng.choice(ret, len(ret)).mean() for _ in range(boots)], [2.5, 97.5])],
                            kl=json.loads(str(z["retention_kl"])), pressure=json.loads(str(z["retention_pressure"])))
    for name, q in COMBOS.items():
        q = np.asarray(q, float)
        idx = [[exact_index(c, t) for t in range(T)] for c in range(C)]
        Gc = lambda a, b: q @ E[np.ix_(idx[a], idx[b])] @ q          # noqa: E731
        pairs = [(a, b) for a in range(C) for b in range(a + 1, C)]
        cross = np.asarray([Gc(a, b) for a, b in pairs])
        G2 = float(cross.mean())
        GN2 = float(np.mean([Gc(a, a) for a in range(C)]))
        V_upd = float(np.mean([0.5 * (Gc(a, a) + Gc(b, b) - 2 * Gc(a, b)) for a, b in pairs]))
        # Cross terms with the exact retention gradient.
        cr = np.asarray([q @ E[idx[c], -1] for c in range(C)])
        G2_total = float(G2 + 2 * cr.mean() + GR2)

        # Random-decision variance at the 128 level: exact residuals.
        s2_list, s2_per_c = [], defaultdict(list)
        level_ratio = defaultdict(list)
        for coll_i, sizes, ex, skg in zip(z["chunk_collection"], z["chunk_sizes"], z["chunk_exact"], z["chunk_sketch"]):
            n = np.asarray(sizes, float); Nc = n.sum()
            r2 = np.einsum("t,kts,s->k", q, ex, q)
            s2 = r2.sum() / np.sum(n * (1 - n / Nc))
            s2_list.append((r2, n, Nc)); s2_per_c[int(coll_i)].append(s2)
            # Aggregated levels from sketches: chunk Gram in combo space.
            K = np.einsum("t,ktls,s->kl", q, skg, q)
            tot = K.sum()
            for level in (1, 4, 16, 64):
                groups = [list(range(i, min(i + level, len(n)))) for i in range(0, len(n), level)]
                if len(groups) < 3:
                    continue
                res2, denom = 0.0, 0.0
                for g in groups:
                    m = n[g].sum()
                    Sg = K[np.ix_(g, g)].sum(); cross_g = K[g, :].sum()
                    res2 += Sg - 2 * m / Nc * cross_g + (m / Nc) ** 2 * tot
                    denom += m * (1 - m / Nc)
                level_ratio[int(level * n[0])].append(res2 / denom)
        s2_dec = float(np.mean([v for vs in s2_per_c.values() for v in vs]))

        # Game-clustered variance (sketch Gram, per collection; pooled and pace-stratified).
        K = game_gram[list(COMBOS).index(name)]
        s2_game, s2_strat, sig_eff = [], [], []
        for c in range(C):
            j = np.flatnonzero(coll == c)
            Kc = K[np.ix_(j, j)]; Lc = L[j]; Nc = Lc.sum(); M = len(j)
            e2 = np.diag(Kc) - 2 * Lc * Kc.sum(1) / Nc + Lc ** 2 * Kc.sum() / Nc ** 2
            s2_game.append(e2.sum() / (M - 1))
            sig_eff.append(e2.sum() / (M - 1) * M / Nc)
            var = 0.0
            for p in range(len(paces)):
                jp = np.flatnonzero(pace[j] == p)
                if len(jp) < 2:
                    continue
                Kp = Kc[np.ix_(jp, jp)]; Lp = Lc[jp]; Np = Lp.sum()
                e2p = np.diag(Kp) - 2 * Lp * Kp.sum(1) / Np + Lp ** 2 * Kp.sum() / Np ** 2
                var += e2p.sum() * len(jp) / (len(jp) - 1)
            s2_strat.append(var / Nc ** 2)            # predicted Var(G_N) from games, stratified
        sigma_eff = float(np.mean(sig_eff))
        pred_var_GN = float(np.mean(s2_strat))

        # Bootstrap: resample games within (collection, pace) strata and chunks within passes.
        boot = []
        strata = [np.flatnonzero((coll == c) & (pace == p)) for c in range(C) for p in range(len(paces))]
        strata = [s for s in strata if len(s)]
        for _ in range(boots):
            w = np.zeros(len(L))
            for s in strata:
                np.add.at(w, rng.choice(s, len(s)), 1)
            means = []
            for c in range(C):
                wc = np.where(coll == c, w, 0)
                means.append((wc, (wc * L).sum()))
            cr_b = [(means[a][0] @ K @ means[b][0]) / (means[a][1] * means[b][1]) for a, b in pairs]
            G2_b = float(np.mean(cr_b))
            # Game-level sigma_eff for this resample (duplicates act as distinct games).
            sig_b = []
            for c in range(C):
                wc = means[c][0]; Nc = means[c][1]; Mc = wc.sum()
                v = K @ wc                                     # sum_k w_k K_jk
                mean2 = wc @ K @ wc / Nc ** 2
                e2 = np.diag(K) - 2 * L * v / Nc + L ** 2 * mean2
                sig_b.append((wc * e2).sum() / (Mc - 1) * Mc / Nc)
            r2, n, Nc = s2_list[rng.integers(len(s2_list))]
            k = rng.integers(len(n), size=len(n))
            s2_b = r2[k].sum() / np.sum(n[k] * (1 - n[k] / Nc))
            boot.append((G2_b, s2_b / G2_b, float(np.mean(sig_b)) / G2_b))
        boot = np.asarray(boot)
        valid = boot[:, 0] > 0

        def ci(col):
            values = np.where(valid, boot[:, col], np.inf)
            return [float(x) for x in np.percentile(values, [2.5, 97.5], method="nearest")]

        res = dict(G2=G2, G2_pairs=cross.tolist(), G2_ci=[float(x) for x in np.percentile(boot[:, 0], [2.5, 97.5])],
                   G2_nonpositive_fraction=float(1 - valid.mean()),
                   GN2=GN2, update_variance=V_upd, predicted_update_variance_from_games=pred_var_GN,
                   update_variance_inflation=V_upd / pred_var_GN if pred_var_GN > 0 else None,
                   s2_dec=s2_dec, s2_dec_by_level={k: float(np.mean(v)) for k, v in sorted(level_ratio.items())},
                   sigma2_game_per_decision=sigma_eff, design_effect=sigma_eff / s2_dec,
                   ess_ratio=s2_dec / sigma_eff,
                   B_dec=s2_dec / G2 if G2 > 0 else float("inf"), B_dec_ci=ci(1),
                   B_game=sigma_eff / G2 if G2 > 0 else float("inf"), B_game_ci=ci(2),
                   B_update=float(N.mean()) * V_upd / G2 if G2 > 0 else float("inf"),
                   B_within_update=s2_dec / GN2,
                   signal_fraction_of_collection_gradient=G2 / GN2,
                   expected_grad_norm_at_128=float(np.sqrt(G2 + s2_dec / 128)))
        if name == "ppo":
            # Retention rows scale with the minibatch (R = B*63/128): per-decision noise adds
            # ret_row_var * (128/63) / 128 per decision-equivalent.
            ret_row_var = ret_step * R
            s2_total = s2_dec + ret_row_var * 128 / R
            res.update(G2_with_retention=G2_total,
                       B_dec_with_retention_scaled=s2_total / G2_total if G2_total > 0 else float("inf"),
                       retention_step_noise_equals_ppo_noise_at_B=s2_dec / ret_step,
                       cos_ppo_retention=float(cr.mean() / np.sqrt(max(G2, 1e-300) * GR2)))
        # Per pace (game-level only): cross-collection pace-block means.
        if name in ("ppo", "policy"):
            per_pace = {}
            for p, pname in enumerate(paces):
                blocks = []
                for c in range(C):
                    j = np.flatnonzero((coll == c) & (pace == p))
                    blocks.append((j, L[j].sum()))
                crossp = [K[np.ix_(blocks[a][0], blocks[b][0])].sum() / (blocks[a][1] * blocks[b][1]) for a, b in pairs]
                G2p = float(np.mean(crossp))
                sig = []
                for c in range(C):
                    j = blocks[c][0]; Kp = K[np.ix_(j, j)]; Lp = L[j]; Np = Lp.sum()
                    e2 = np.diag(Kp) - 2 * Lp * Kp.sum(1) / Np + Lp ** 2 * Kp.sum() / Np ** 2
                    sig.append(e2.sum() / (len(j) - 1) * len(j) / Np)
                per_pace[pname] = dict(G2=G2p, sigma2_game_per_decision=float(np.mean(sig)),
                                       B_game=float(np.mean(sig)) / G2p if G2p > 0 else float("inf"),
                                       decisions=float(np.mean([blocks[c][1] for c in range(C)])),
                                       games=float(np.mean([len(blocks[c][0]) for c in range(C)])))
            res["per_pace"] = per_pace
        out["combos"][name] = res
    return out


def tradeoff(B_crit, sizes=(128, 512, 1024, 2048, 4096, 8192)):
    return [dict(minibatch=b, steps_over_min=1 + B_crit / b, data_over_min=1 + b / B_crit) for b in sizes]


def analyze(args):
    reports = [analyze_one(p, boots=args.boots) for p in args.inputs]
    for r in reports:
        ppo = r["combos"]["ppo"]
        r["tradeoff_B_dec"] = tradeoff(ppo["B_dec"])
    Path(args.out).write_text(json.dumps(reports, indent=1, default=float))
    for r in reports:
        print(f"\n== update {r['update']} ({r['frames']/1e6:.0f}M frames) metric={r['metric']} "
              f"N={r['mean_decisions']:.0f} Lbar={r['mean_game_length']:.1f}")
        for name, c in r["combos"].items():
            print(f"  {name:9s} |G|={np.sqrt(max(c['G2'],0)):.4g} |G_N|={np.sqrt(c['GN2']):.4g} "
                  f"B_dec={c['B_dec']:.4g} {c['B_dec_ci']} B_game={c['B_game']:.4g} {c['B_game_ci']} "
                  f"B_upd={c['B_update']:.4g} deff={c['design_effect']:.3g} infl={c['update_variance_inflation']}")
        print("  retention", {k: r["retention"][k] for k in ("G2", "step_noise", "B_simple_rows")})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--run", required=True, help="trainer output directory")
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--log", help="trainer stdout log (per-update JSON) for record validation")
    m.add_argument("--collections", type=int, default=3)
    m.add_argument("--updates", type=int, nargs="+", help="explicit collection updates (default u+1..u+C)")
    m.add_argument("--permutations", type=int, default=2)
    m.add_argument("--chunk", type=int, default=128)
    m.add_argument("--sketch-dim", type=int, default=2 ** 15)
    m.add_argument("--retention-draws", type=int, default=64)
    m.add_argument("--metric", choices=("euclidean", "adam"), default="euclidean")
    m.add_argument("--device", default="cuda")
    m.add_argument("--seed", type=int, default=20260924)
    b = sub.add_parser("bench")
    b.add_argument("--run", required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--sizes", type=int, nargs="+", default=[128, 1024, 4096])
    b.add_argument("--steps", type=int, default=10)
    b.add_argument("--device", default="cuda")
    a = sub.add_parser("analyze")
    a.add_argument("inputs", nargs="+")
    a.add_argument("--out", required=True)
    a.add_argument("--boots", type=int, default=400)
    args = parser.parse_args()
    {"measure": measure, "bench": bench, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
