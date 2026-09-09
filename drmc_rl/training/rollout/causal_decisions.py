"""Join strict pair events into complete, on-policy placement transitions."""

from __future__ import annotations

import numpy as np

from drmc_rl.training.rollout.decision_buffer import DecisionStep


class CausalDecisionCollector:
    """Keep one pending placement per learner until its next actual choice.

    Each learner has a fixed decision quota. Learners that finish their quota
    continue under the unchanged behavior policy while the remaining learners
    drain. Those extra actions are not training samples. No pending sample
    crosses a policy update, and bootstrap values are captured at each last
    stored decision's own successor boundary, not at a later global barrier.
    """

    def __init__(self, buffer):
        if buffer.gamma != 1.0:
            raise ValueError("causal event collection currently requires gamma one")
        if buffer.capacity < buffer.num_envs or buffer.size:
            raise ValueError("causal collection requires an empty buffer with one slot per learner")
        self.buffer = buffer
        self.quota = np.full(buffer.num_envs, buffer.capacity // buffer.num_envs, np.int64)
        self.quota[: buffer.capacity % buffer.num_envs] += 1
        self.completed = np.zeros(buffer.num_envs, np.int64)
        self.pending = [None] * buffer.num_envs
        self.started_at = np.zeros(buffer.num_envs, np.int64)
        self.bootstrap = np.zeros(buffer.num_envs, np.float32)

    @property
    def full(self):
        return bool(np.all(self.completed == self.quota))

    def _finish(self, side, observation, frame, *, done, value=0.0):
        step = self.pending[side]
        if step is None:
            return
        elapsed = int(frame) - int(self.started_at[side])
        if elapsed < 0:
            raise RuntimeError("a pending placement crossed a reset without its terminal outcome")
        step.tau = elapsed
        step.obs_next = observation.copy()
        step.done = bool(done)
        self.buffer.add(step)
        self.completed[side] += 1
        self.bootstrap[side] = 0.0 if done else float(value)
        self.pending[side] = None

    def arrive(self, observations, ready, frames, values):
        """Close placements only at an actionable successor, using its value."""
        for side in np.flatnonzero(ready):
            self._finish(side, observations[side], frames[side], done=False, value=values[side])

    def begin(self, observations, selected, ready, frames):
        actions, log_probs, values, masks, costs, pills, previews, aux = selected
        for side in np.flatnonzero(ready & (self.completed < self.quota)):
            if self.pending[side] is not None or not masks[side].any():
                raise RuntimeError("a placement must start at a new feasible decision boundary")
            self.started_at[side] = frames[side]
            self.pending[side] = DecisionStep(
                obs=observations[side].copy(),
                mask=masks[side].copy(),
                pill_colors=pills[side].copy(),
                preview_pill_colors=previews[side].copy(),
                action=int(actions[side]),
                log_prob=float(log_probs[side]),
                value=float(values[side]),
                tau=0,
                reward=0.0,
                obs_next=observations[side].copy(),
                done=False,
                cost_to_lock=None if costs is None else costs[side].copy(),
                aux=None if aux is None else aux[side].copy(),
                env_id=int(side),
            )

    def advance(self, rewards, dones, observations, frames):
        """Include rewards during waits and forced falls; preserve terminal credit."""
        for side, step in enumerate(self.pending):
            if step is None:
                continue
            step.reward += float(rewards[side])
            if dones[side]:
                self._finish(side, observations[side], frames[side], done=True)
