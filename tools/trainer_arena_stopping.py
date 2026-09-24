"""Opt-in early answers for trainer arena comparisons.

``sequential`` (config default, overridable or disabled per schedule row with
``null``) evaluates the pre-registered question after every batch and stops a
comparison once it is decided; ``games`` remains the maximum budget.
``skip_identical`` records a comparison between provably interchangeable
entrants as exactly 0.5, after a short mirrored-journal probe when only
network-input-only settings differ. Every verdict is recomputed from the
game journal and move traces, so a resumed worker reaches the same answer.
See ``drmc_rl.arena.sequential`` and ``drmc_rl.arena.identity``.
"""
from __future__ import annotations

from drmc_rl.arena.identity import load_journals, mirrored_games, static_identity
from drmc_rl.arena.sequential import SequentialRule, evaluate


class ComparisonStopping:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.static, self.probes, self.verdicts = {}, {}, {}
        self.probe_games = int(config.get("identity_probe_games", 16))
        if self.probe_games < 2 or self.probe_games % 2:
            raise ValueError("identity probes require complete side-swapped pairs")
        for match in config["schedule"]:
            self.rule(match)

    def _setting(self, match, key, default=None):
        return match[key] if key in match else self.config.get(key, default)

    def rule(self, match):
        return SequentialRule.from_config(self._setting(match, "sequential"))

    def _static(self, match):
        if not self._setting(match, "skip_identical", False):
            return None
        if match["id"] not in self.static:
            self.static[match["id"]] = static_identity(self.config, match)
        return self.static[match["id"]]

    def verdict(self, match, rows):
        """A final answer for this comparison, or None while it must continue."""
        id, budget = match["id"], match["games"]
        static, note = self._static(match), None
        if static == "identical":
            note = "same model bytes and decision settings at this pace; recorded 0.5 without play"
        elif static == "probe":
            if id not in self.probes and len(rows) >= self.probe_games:
                self.probes[id] = mirrored_games(rows, load_journals(self.output, id, rows))
            if self.probes.get(id):
                note = (f"same model bytes and charged delay; network-input-only settings differ; "
                        f"{len(rows)} probe games had byte-identical side-swapped move journals")
        if note:
            verdict = dict(decision="identical", score=0.5, stop=True, early=len(rows) < budget,
                           games=len(rows), budget_games=budget, note=note)
        else:
            rule = self.rule(match)
            verdict = evaluate(rule, rows, budget) if rule is not None and rows else None
            if verdict is not None and not verdict["stop"]:
                verdict = None
        if verdict is not None:
            self.verdicts[id] = verdict
        return verdict

    def batch_games(self, match, rows, default):
        """Games in the next batch: finish a pending probe, otherwise one look."""
        if self._static(match) == "probe" and match["id"] not in self.probes:
            return max(2, self.probe_games - len(rows))
        if self.rule(match) is not None:
            look = int(self._setting(match, "look_games", default))
            if look < 2 or look % 2:
                raise ValueError("sequential looks require complete side-swapped pairs")
            return min(default, look)
        return default

    def progress(self, match, rows):
        """Printable state of the current comparison."""
        verdict = self.verdicts.get(match["id"])
        if verdict is not None:
            return dict(stopping=verdict)
        rule = self.rule(match)
        if rule is None or not rows:
            return {}
        state = evaluate(rule, rows, match["games"])
        return dict(stopping={k: state[k] for k in ("games", "budget_games", "confidence_sequence")})
