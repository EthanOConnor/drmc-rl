# Legacy own-board search distillation

The repository retains the original `SearchPolicy` and `search_distill` code so
historical checkpoints, measurements, and reproducibility tests continue to
load. It is **not** the governing search architecture and no new campaign
should be designed around it.

The retained implementation simulates the learner bottle with a one-player
pool and freezes or approximates the opponent context. That was useful for
proving that search could improve a fixed policy and for bootstrapping stronger
teachers, but it is not a causal model of the asynchronous pair game.

Current authority is:

- [`SEARCH_DESIGN.md`](SEARCH_DESIGN.md) for strict pair-event search;
- [`ROADMAP.md`](ROADMAP.md) for the teacher-first policy-iteration sequence;
- `drmc_rl/program/program.yaml` for launch status and gates;
- `drmc_rl/search/joint_event.py` for the backend-independent search contract.

Existing `search_distill` configs and tests are compatibility evidence only.
Search remains an offline teacher until the `joint-event-search` gate records a
same-weight paired improvement. Do not enable legacy `act_from_search` in a new
PPO campaign or describe it as on-policy PPO.
