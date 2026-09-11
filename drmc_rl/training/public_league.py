"""Small, frozen public opponent pools for outcome learning."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drmc_rl.arena.meta_strategy import solve_entropy_regularized_zero_sum


def empirical_mixture(agents, comparisons, records, *, level, pace, speed=2, minimum_pairs=64):
    """Use measured paired edges only; missing/censored edges are not draws."""
    names = list(agents)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("league needs distinct public members")
    edges = {}
    for match in comparisons:
        if (
            match["level"] != level
            or match.get("pace", "frame_perfect") != pace
            or match.get("speed", 2) != speed
        ):
            continue
        a, b = match["a"], match["b"]
        if a not in names or b not in names or a == b:
            continue
        rows = [r for r in records if r["comparison"] == match["id"]]
        if any(r.get("reason") == "timeout" or r.get("score") is None for r in rows):
            raise ValueError("censored league matchup requires completed follow-up evidence")
        pairs = {}
        for row in rows:
            if row["side"] not in (0, 1) or row["score"] not in (0.0, 0.5, 1.0):
                raise ValueError("league records require physical side and natural WDL score")
            pair = pairs.setdefault(row["seed"], {})
            if row["side"] in pair:
                raise ValueError("duplicate side/seed evidence in league matchup")
            pair[row["side"]] = row["score"]
        key = tuple(sorted((a, b)))
        for seed, pair in pairs.items():
            if set(pair) != {0, 1}:
                continue
            score = float(np.mean(list(pair.values())))
            value = 2 * score - 1 if a == key[0] else 1 - 2 * score
            # Repeated deterministic games in different feeds are one unit.
            if seed in edges.setdefault(key, {}) and edges[key][seed] != value:
                raise ValueError("same frozen paired game has inconsistent outcomes")
            edges[key][seed] = value
    matrix = np.zeros((len(names), len(names)))
    counts = np.zeros_like(matrix, dtype=np.int64)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            key = tuple(sorted((a, b)))
            samples = edges.get(key, {})
            if len(samples) < minimum_pairs:
                raise ValueError(f"insufficient paired evidence for {a} vs {b}")
            value = sum(samples.values()) / (len(samples) + 1.0)  # one neutral prior pair
            matrix[i, j] = value if a == key[0] else -value
            matrix[j, i] = -matrix[i, j]
            counts[i, j] = counts[j, i] = len(samples)
    solved = solve_entropy_regularized_zero_sum(matrix, temperature=0.1, floor=0.02)
    return solved.to_dict(names) | dict(
        payoff=matrix.tolist(),
        paired_counts=counts.tolist(),
        level=level,
        pace=pace,
        speed=speed,
        minimum_pairs=minimum_pairs,
        diagnostic_only=True,
        note="Frozen training mixture; empirical payoff uncertainty is not a promotion certificate",
    )


class PublicOpponentPool:
    def __init__(self, specs, parent, parent_checkpoint, device):
        self.specs = list(
            specs or [dict(id="parent", weight=1.0, checkpoint=str(parent_checkpoint))]
        )
        self.names = [s["id"] for s in self.specs]
        if len(set(self.names)) != len(self.names) or "learner" in self.names:
            raise ValueError("public league member IDs must be distinct from the learner")
        weights = np.asarray([s["weight"] for s in self.specs], dtype=np.float64)
        if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("public opponent weights must be finite and positive")
        self.weights = weights / weights.sum()
        self.parent, self.parent_checkpoint, self.device = parent, Path(parent_checkpoint), device
        self.loaded = {}

    def identities(self):
        """Bind the mixture to checkpoint bytes before collection or resume."""
        from drmc_rl.teachers.counterfactual_release import sha256_file

        return {
            spec["id"]: dict(
                checkpoint_sha256=sha256_file(Path(spec["checkpoint"])),
                adapter_sha256=sha256_file(Path(spec["adapter_checkpoint"]))
                if spec.get("adapter_checkpoint")
                else None,
            )
            for spec in self.specs
        }

    def choose(self, rng):
        if len(self.names) == 1:
            return self.names[0]
        return str(rng.choice(self.names, p=self.weights))

    def load(self, name):
        if name not in self.loaded:
            from tools.vs_head_to_head import PlainPolicy
            from drmc_rl.models.policy.pace_adapter import PacePolicy

            spec = self.specs[self.names.index(name)]
            checkpoint = Path(spec["checkpoint"])
            if spec.get("adapter_checkpoint"):
                model = PacePolicy(checkpoint, self.device, adapter_path=spec["adapter_checkpoint"])
                base = model.plain
            else:
                model = (
                    self.parent
                    if checkpoint == self.parent_checkpoint
                    else PlainPolicy(checkpoint, self.device, public_only=True)
                )
                base = model
            from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
            if base.aux_spec not in ("zero_v1_vs", PUBLIC_CONTEXT_SCHEMA) or base.in_channels != 20:
                raise ValueError(
                    "paced league requires an audited public input contract"
                )
            self.loaded[name] = model
        return self.loaded[name]
