# Multi-Env C++ Backend Design

**Status:** Design Document  
**Author:** Agent (2025-12-19)  
**Goal:** Scale C++ engine to N parallel envs with dynamic count, per-env views, and speed scaling metrics

---

## Current State Analysis

### What Works ✅
- **File-backed SHM per instance** (line 258-262 in `cpp_engine_backend.py`)
  - Uses `tempfile.NamedTemporaryFile` — unique per backend
  - No global `/drmario_shm` collision risk
- **Vector env factory** (`dr_mario_vec.py`)
  - Supports `SyncVectorEnv` and `AsyncVectorEnv`
  - `vectorization=auto` → async for num_envs > 1
- **Backend registry** — `cpp-engine` works with env factory

### Key Gaps 🔴

| Gap | Impact | Fix Effort |
|-----|--------|------------|
| Polling overhead (20µs spin + 10µs sleep) | ~50% of step time at N envs | Medium |
| Synthetic RAM layer | 0.07ms build_state per step | Medium |
| TUI shows only env[0] | No multi-env visibility | Medium |
| No dynamic env count | Must restart to change | High |
| No speed scaling metrics | Can't measure efficiency | Low |

---

## Architecture: Multi-Env with C++ Backend

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop (Python)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              SyncVectorEnv / AsyncVectorEnv                 │
│                  (Gymnasium standard)                       │
└─────────────────────────────────────────────────────────────┘
              │           │           │
              ▼           ▼           ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │ Env 0  │  │ Env 1  │  │ Env N  │   ← DrMarioPlacementEnv
         └────────┘  └────────┘  └────────┘
              │           │           │
              ▼           ▼           ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │Backend │  │Backend │  │Backend │   ← CppEngineBackend
         │ SHM 0  │  │ SHM 1  │  │ SHM N  │      (each has own temp file)
         └────────┘  └────────┘  └────────┘
              │           │           │
              ▼           ▼           ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │Engine 0│  │Engine 1│  │Engine N│   ← drmario_engine subprocess
         └────────┘  └────────┘  └────────┘
```

---

## Phase 1: Correctness (P0)

### 1.1 Verify Multi-Instance Isolation
- [ ] Write test: spawn 4 `CppEngineBackend` instances in same process
- [ ] Assert each has unique `_shm_file` path
- [ ] Assert independent stepping (different RNG seeds → different boards)

### 1.2 Test SyncVectorEnv + AsyncVectorEnv
- [ ] `num_envs=4, vectorization=sync` — all in main process
- [ ] `num_envs=4, vectorization=async` — worker processes
- [ ] Assert no SHM collisions, clean shutdown

### 1.3 Fix Potential Race: Process Exit Detection
- Current: checks `_proc.poll()` every 128 iterations
- Risk: subprocess crash between polls → stale data
- [ ] Add atomic heartbeat field in SHM struct (C++ increments, Python checks)

---

## Phase 2: Performance (P0)

### 2.1 Eliminate Polling Overhead
**Option A: Eventfd/Pipe notification (Recommended)**
```cpp
// Engine writes 1 byte to pipe when step complete
write(pipe_fd, "X", 1);
```
```python
# Python uses select() with instant wake
select.select([pipe_read_fd], [], [], timeout)
```
- Pro: Zero-latency wake, cross-platform (pipe)
- Con: Requires C++ change + pipe per instance

**Option B: Batched Stepping**
```python
def step_batch(self, buttons_seq: List[int], frames: int) -> None:
    # Request N frames at once, poll once at end
    self._state.frame_budget = frames
    self._state.control_flags |= 0x04
    # ... wait for frame_count >= expected
```
- Pro: Amortizes overhead for macro actions (~30 frames)
- Con: Still polling, just less often

### 2.2 Direct Struct → Observation (Skip RAM Layer)
```python
# Current (slow):
state → synthetic RAM (128 writes) → build_state() → bitplanes

# Proposed (fast):
state → direct bitplane construction from board[128]
```
- Only for C++ backend; libretro path unchanged
- Requires `BitplaneBuilder.from_engine_state(state)` method

### 2.3 Parallel Engine Stepping (AsyncVectorEnv)
- Worker processes already isolate engines
- Measure: `fps_total(num_envs=N) / fps_total(num_envs=1)` ≈ `N` ideal (linear scaling)

---

## Phase 3: Debug TUI Multi-Env Support (P0)

### 3.1 New Hotkeys
| Key | Action |
|-----|--------|
| `[` / `]` | Decrease / increase num_envs (controlled restart) |
| `Tab` | Cycle through per-env views (env 0, 1, ..., N, summary) |
| `1`-`9` | Jump to env N directly |
| `0` | Summary view (all boards grid + aggregate stats) |

### 3.2 Per-Env View
Same as current single-env, but header shows `Env 3/8`:
```
┌─ Board (Env 3/8) ─┐  ┌─ Perf ─┐  ┌─ Learning ─┐
│ [current board]   │  │ fps... │  │ ret...     │
└───────────────────┘  └────────┘  └────────────┘
```

### 3.3 Summary View (New)
```
┌─ All Boards (8 envs) ────────────────────────────┐
│ [0] [1] [2] [3]  │  Aggregate Stats             │
│ [4] [5] [6] [7]  │  ─────────────────           │
│                  │  fps(total): 2,400            │
│                  │  fps(per-env): 300            │
│                  │  speedup: 3.2x vs single      │
│                  │  efficiency: 80% of linear    │
│                  │  episodes: 12 (3 clear, 9 TO) │
└──────────────────────────────────────────────────┘
```

### 3.4 Speed Scaling Metrics
| Metric | Formula |
|--------|---------|
| `fps(total)` | Total frames / wall_time |
| `fps(per-env)` | fps(total) / num_envs |
| `speedup_vs_single` | fps(total) / fps_when_num_envs_was_1 |
| `efficiency` | speedup / num_envs × 100% |

Baseline policy:
- Store `fps_single` when `num_envs=1` as baseline for comparison (persisted only for the current UI session).
- If a baseline has not been measured yet, show `speedup_vs_single = N/A` and `efficiency = N/A`.

### 3.5 Dynamic Env Count
**Decision:** Treat env-count changes as **restart-only** (stop training thread, rebuild vec env, restart).

Why:
- Gymnasium VectorEnv sizes are not intended to change dynamically.
- PPO/SMDP rollout buffers, optimizer state, and UI env-index bookkeeping all assume a fixed `num_envs`.
- A controlled restart is simple, robust, and aligns with the goal (debug throughput + inspectability).

UI behavior:
- On `[`/`]`, show a short status line: `restarting with num_envs=N...` and restart immediately (or on next safe point if we need to avoid tearing down mid-step).

---

## Phase 4: Polish (P1)

### 4.1 Per-Env Coloring in Summary
- Green border: recently cleared
- Red border: recently topped out
- Yellow: active (currently stepping)

### 4.2 Env-Specific Stats Overlay
- Press `i` on summary view: show hover-style tooltip with env details

### 4.3 Async Health Monitoring
- Detect hung worker processes (no frame_count increment)
- Auto-restart with warning toast

---

## Implementation Order

```
P0 (Multi-Env MVP):
  1.1 Verify isolation          [1 day]
  1.2 Test vector envs          [0.5 day]
  3.1 Add TUI hotkeys           [0.5 day]
  3.3 Summary view              [1 day]
  3.4 Speed scaling metrics     [0.5 day]
  2.2 Direct struct→obs         [1 day]
  
P0 (Performance):
  2.1 Batched stepping          [1 day]
  2.3 Async measurement         [0.5 day]
  
P1 (Polish):
  1.3 Heartbeat detection       [0.5 day]
  3.5 Dynamic env count         [2 days - complex]
  4.* Visual polish             [1 day]
```

---

## Open Questions

1. **Dynamic env count:** Worth the complexity? Or just restart?
2. **Eventfd vs batched:** Eventfd is faster but requires C++ changes. Batched works today.
3. **Summary board size:** 8 mini-boards fit? Or max 4 with pagination?
