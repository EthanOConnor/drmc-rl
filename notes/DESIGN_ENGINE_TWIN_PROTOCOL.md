# Engine / Digital Twin / Planner ABI Design

**Status:** Design Document  
**Author:** Ethan + Coding Agent (2025-12-21)  
**Motivation:** Push training throughput (100k+ FPS) and make the same engine interface usable for simulation, emulator, and real-hardware play while keeping game internals private behind an ABI.

---

## How We Got Here (Decision Trail)

- **Notebook prompt (2025-12-19):** For performance, Python should never receive full boards; it should submit desired placements/resets/specs and receive feasibility/costs/τ and the engine’s event detections + derived stats (reward/curriculum/monitoring).
- **Repo reality check (2025-12-21):**
  - There is a **frame/RAM interface** (`EmulatorBackend`) used to keep emulator integrations consistent.
  - There is already a **decision-boundary batched ABI** (`libdrmario_pool`) that returns feasibility masks, cost-to-lock, τ frames, and event counters with persistent NumPy buffers (training hot path).
- **Conclusion:** Standardize on the **decision-boundary ABI** as the canonical “game interface”, keep board internals behind it, and treat “simulation” vs “vision/hardware shadowing” as different backends of the same World Kernel.

---

## Goals

- **Keep game internals hidden:** board, RNG, and simulator state do not cross the ABI in normal operation.
- **One interface across backends:**
  - Pure simulation (engine is source of truth).
  - “Shadow mode” (engine is a digital twin tracking external observations).
  - Future “play humans” and streaming integrations (same protocol subsets).
- **Two consumer channels:**
  - **Decision channel (hot path):** fixed-shape, minimal arrays for inference + learning.
  - **Telemetry channel (cold path):** richer event stream/snapshots for logging/UI/debug.
- **Planner is co-located with the model:** same representation/timing; no board copies or Python-side feature extraction in the hot path.

## Non-Goals

- Designing the full vision model architecture, capture stack, or controller hardware.
- Exposing raw boards/tiles to Python by default (debug-only, opt-in).
- Perfect determinism in shadow mode (sensor noise/latency exists); the goal is robust tracking + graceful recovery.

---

## System Overview

```
                         (Decision channel)                      (Telemetry channel)
┌───────────────────────┐  reset/spec + actions      ┌──────────────────────────┐  events/snapshots/traces
│ Decision Maker /      │───────────────────────────▶│ World Kernel             │──────────────────────────▶ UI / Logs / Datasets
│ Learning Coordinator   │◀───────────────────────────│  + integrated Planner    │
│ (Python)               │   obs/mask/cost/τ/events   │  + event detectors       │
└───────────────────────┘                             └───────────┬─────────────┘
                                                                   │
                                                           (shadow mode only)
                                                                   │
                                                         ┌─────────▼─────────┐
                                                         │ Perception Adapter │  video in (HDMI/cam/VOD)
                                                         └─────────┬─────────┘
                                                                   │
                                                         ┌─────────▼─────────┐
                                                         │ Actuation Adapter  │  controller out (NES/emulator)
                                                         └────────────────────┘
```

**Key idea:** whether the World Kernel is “free running” (sim) or “shadowing” (vision/hardware), the coordinator sees the *same* decision-boundary contract. Only the internal correctness/clocking machinery changes.

---

## Components (Responsibilities and Boundaries)

### 1) World Kernel (C++/Rust)

**Owns the canonical game state** and the authoritative frame clock. Responsible for:
- Simulating Dr. Mario rules (placement → lock → clears → gravity → next spawn).
- Owning RNG and reset-time determinism knobs.
- Maintaining derived state needed for reward/curriculum/monitoring (events + stats).
- Providing a stable ABI for batched resets/steps.
- In shadow mode: maintaining a digital twin and reconciling/validating against external observations.

### 2) Integrated Planner (inside the World Kernel)

Computes at decision boundaries:
- `feasible_mask[action]` (0/1)
- `cost_to_lock[action]` (frames or sentinel for unreachable)
- Optional planner diagnostics (lock pose, reachability stats), kept off the hot path unless enabled.

### 3) Perception Adapter (device-specific)

Turns raw video into **evidence** and timing signals:
- Evidence examples: spawn-id changes, pill colors, detected board deltas, virus counts by color, “clearing active” flags.
- Timing examples: capture timestamps, inferred vblank/frame indices, latency estimates.

The perception adapter should be swappable (camera vs HDMI vs YouTube) without touching the ABI.

### 4) Actuation Adapter (device-specific)

Applies control outputs (buttons) to the external system and optionally reports:
- send timestamps, ack timestamps, estimated end-to-end latency.

### 5) Decision Maker / Learning Coordinator (Python)

Owns the experiment:
- inference, rollout collection, training updates
- curriculum scheduling, evaluation, checkpointing
- logging configuration and UI orchestration

It should never need to parse boards or re-implement engine-derived event detection in Python.

### 6) Telemetry + Visibility (Python + optional sidecars)

Consumes the telemetry stream and produces:
- debug UI overlays, episode videos, anomaly dumps
- offline datasets for imitation/pixel2state/vision fine-tuning

---

## Data Flow Specification

### Decision Channel (hot path, fixed-shape)

**Inputs (Python → World Kernel)**
- `reset(specs[N])` where a spec can include:
  - level/speed
  - RNG seed / pill sequence seed
  - (optional) synthetic curriculum knobs (e.g., virus target)
- `step(actions[N], reset_mask[N])` at decision boundaries
  - `actions` is the macro placement (e.g., 512 = rot×row×col)

**Outputs (World Kernel → Python)**
- Observation tensor(s) suitable for inference (already normalized/packed).
- Feasibility mask + cost-to-lock arrays (planner output).
- `tau_frames` for accepted actions (SMDP τ).
- Terminal flags + terminal reason codes.
- Compact event counters for reward/curriculum, e.g.:
  - tiles cleared (virus/nonvirus/total)
  - match events (clear count)
  - viruses remaining (+ optional by-color)
  - adjacency shaping flags
- Compact “context scalars” that are cheap for the kernel to compute:
  - speed, level, frame_count
  - max height, column heights
  - clearance-condition progress (“progress bar”)

**Rule:** anything that the coordinator might compute by scanning a board should be treated as a kernel responsibility.

### Telemetry Channel (cold path, variable-size)

**Purpose:** rich visibility without slowing training.
- Structured event stream (ring buffer): spawn, lock, clear, virus-clear, topout, desync, etc.
- Optional snapshots: packed board bytes, planner debug views, trace spans.
- Perf counters: planner time, kernel time, perception time (shadow mode).

**Rule:** telemetry must be **strictly optional** and never required for correctness of training.

---

## Control Flow Specification

### A) Pure simulation training (batched, N envs)

1. Coordinator sends `reset(specs)` (possibly with a per-env mask for autoresets).
2. Kernel returns decision outputs (obs + feasibility + costs + context stats).
3. Coordinator runs inference and chooses `actions[N]`.
4. Coordinator sends `step(actions, reset_mask)` once per macro decision batch.
5. Kernel executes each env to the next decision boundary (or terminal), returning τ + events/stats.
6. Coordinator computes rewards/GAE using kernel outputs (no board parsing), logs telemetry as configured, repeats.

### B) Shadow mode (vision/hardware, usually 1 env)

Core loop:
1. Perception adapter emits `(evidence, timestamps)` for the newest frame(s).
2. Kernel ingests evidence and updates the twin:
   - advance prediction forward in time
   - reconcile against evidence
   - update drift/latency estimates
3. At the next decision boundary, kernel exposes decision outputs to the coordinator.
4. Coordinator selects action and sends it to kernel (and/or directly to actuation adapter through kernel-owned scheduling).
5. Kernel schedules actuation (accounting for latency) and predicts the resulting future states.
6. Perception continues; kernel validates consistency and raises a `desync` event if invariants break, triggering a recovery policy.

Key point: the coordinator still “sees” the same decision-boundary contract; shadow-mode complexity stays inside the kernel + adapters.

---

## ABI Contract (Conceptual)

### Design principles

- **Batch-first**: one call per macro decision batch.
- **Zero-copy**: caller provides/owns output buffers; kernel writes into them.
- **Versioned**: explicit `protocol_version` and `struct_size` checks; optional capability bits.
- **Explicit timing**: τ and reason codes are first-class outputs.
- **Board is internal**: raw board bytes only exist as an optional debug field.

### Contract sketch (aligns with today’s `libdrmario_pool`)

- `create(cfg)` → handle
- `reset(handle, reset_mask, reset_specs, out)`:
  - returns post-reset **first actionable decision** state
- `step(handle, actions, reset_mask, reset_specs, out)`:
  - executes to next decision boundary (or terminal)
  - fills decision outputs + step-only outputs (τ/events/terminal)

### Suggested extensions for shadow mode (future)

- `ingest_observation(evidence_batch)`:
  - evidence is device-agnostic (decoded facts + timestamps + confidence)
- `quality_flags` output per env:
  - e.g., “twin locked”, “timebase stable”, “state partially observed”, “desync suspected”
- `telemetry_ring` shared memory:
  - variable-size messages without mallocs in the hot path

---

## Timing: Drift + Latency (Shadow Mode)

- The kernel maintains an **authoritative internal frame clock** and estimates a mapping from external timestamps to internal frames.
- Use “anchor events” (spawn-id increments, lock, clears) as discrete alignment points to correct drift.
- Maintain explicit telemetry:
  - estimated capture latency
  - controller-output latency
  - drift rate / PLL error

Design stance: drifting hardware clocks are expected; the system should adapt continuously rather than requiring perfect 60.000 Hz alignment.

---

## Desync Detection + Recovery

### Detection

The kernel maintains a consistency score comparing predicted vs observed evidence:
- illegal pill positions/orientations
- impossible board deltas (violating gravity/match rules)
- inconsistent virus counts or spawn-id sequence

### Recovery (policy choices)

- soft resync: re-lock the timebase, keep playing if evidence is mostly consistent
- hard resync: rebuild twin state from a high-confidence observation (or request a clean reset)
- abort: surface a terminal reason `DESYNC` (or truncate) and let the coordinator reset

---

## Where This Lives In The Repo Today (Grounding)

Existing pieces that already implement much of this design:
- Decision-boundary batched ABI: `game_engine/drmario_pool_capi.h` (+ `game_engine/drmario_pool_capi.cpp`)
- Python wrapper with persistent buffers: `envs/backends/drmario_pool.py`
- High-throughput vec env using the pool runner: `training/envs/drmario_pool_vec.py`
- Frame/RAM backend abstraction (useful for emulator parity + tooling): `envs/backends/base.py`
- C++ engine subprocess + SHM driver (separate from pool): `envs/backends/cpp_engine_backend.py`

---

## Open Questions (for follow-up docs)

- What is the minimal “evidence” set required for robust twin locking (spawn-id, virus count, board hash, full board decode, …)?
- How should the kernel expose “incompletely observed” states to the coordinator (mask/quality flags vs conservative fallback)?
- Where should inference live once env throughput is high (Python vs C++ service), and what batching shape does it prefer?

