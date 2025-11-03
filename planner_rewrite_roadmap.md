# Counter-Aware Dr. Mario Placement Planner (Roadmap)

## Goal

Rebuild the placement planner so that it reproduces NES Dr. Mario’s per-frame behaviour exactly. Paths produced by the planner must match what the ROM will execute when replayed frame by frame. The planner should remain fast enough that the emulator stays the bottleneck (no perceptible slowdown per spawn).

## Required NES Facts (from disassembly)

### Gravity / Soft Drop (fallingPill_checkYMove in `drmario_prg_game_logic.asm`)
- `_frameCounter & fast_drop_speed` ($0043 & $01) gates soft drop polling (DOWN only checked every 2 frames).
- `currentP_speedCounter` ($0092) increments each frame; gravity triggers when it exceeds the threshold.
- Threshold derived from `speedCounterTable[ baseSpeedSettingValue[currentP_speedSetting] + currentP_speedUps ]` (tables in `drmario_data_game.asm`).
- When DOWN is the only d-pad direction held at polling time, the counter resets but the drop still happens one frame later (no teleport).
- On collision, `confirmPlacement` runs immediately and the falling action ends.

### Horizontal Movement (fallingPill_checkXMove)
- `currentP_horVelocity` ($0093) counts consecutive frames that a direction is held.
- First press moves immediately and resets velocity to 0.
- Holding continues to increment velocity; when it reaches `hor_accel_speed` (16) a move occurs, velocity reloads to `hor_accel_speed - hor_max_speed` (~6).
- Collisions reset velocity to `hor_accel_speed-1` so the next opportunity is fast.
- Pressing and releasing quickly is the only way to achieve faster-than-repeat double taps.

### Rotation (fallingPill_checkRotate)
- Checked after Y then X each frame.
- Tests rotation parity; tries in-place, “extra-left” for horizontal, then generic left kick.
- Uses button presses (currentP_btnsPressed) not holds.

### Zero-page RAM addresses
- `currentP_speedCounter` $0092
- `currentP_speedUps` $008A
- `currentP_speedSetting` $008B
- `currentP_horVelocity` $0093
- `currentP_btnsHeld` / `currentP_btnsPressed` per disassembly.
- `frameCounter` $0043
- `currentP_fallingPillY` $0306, `currentP_fallingPillX` $0305, `currentP_fallingPillRotation` $0325.

## Planner Design

### State per frame
```
State = (
    x, y, orient,
    speed_counter, speed_threshold,
    hor_velocity, hold_dir, held_down,
    frame_counter_mod2,
    lock_timer?,
)
```

- `speed_threshold` computed once using tables.
- `hold_dir ∈ {NONE, LEFT, RIGHT}`; `held_down ∈ {True, False}`.
- `frame_counter_mod2` to emulate fast_drop_speed.
- Optional: include lock buffer to permit slides before settling.

### Actions per frame
- `HoldLeft`, `HoldRight`, `Neutral` × `HoldDown/Neutral` × `RotateCW/RotateCCW/None`.
- Apply via NES logic in exact order: update button registers, run Y stage, X stage, Rot stage.
- Append WAIT frames automatically when counters forbid moves.

### Transitions
1. Update button states (`currentP_btnsHeld`/`Pressed`).
2. **Y stage**: if `frameCounter_mod2 & fast_drop_speed == 0` and DOWN-only held, force drop counter to 0. Otherwise increment `speed_counter` and compare with threshold. Only move down when allowed. After drop, recompute collision, reload counter, update `frameCounter_mod2`.
3. **X stage**: update `hor_velocity` per pressing vs holding. Only move left/right when velocity threshold reached or button pressed this frame. On block, reset velocity appropriately.
4. **Rotation**: evaluate parity, tests, kicks, and apply or reject.
5. If drop failed => landing: stop expansion, mark placement.
6. Record per-frame state & action.

### BFS Loop
- Frontier stores `(State, time, parent_id)`.
- Use deque (BFS) or min-heap (Dijkstra) depending on costs; one frame per step so BFS ok.
- Maintain best-known arrival frame for each `(x,y,orient)` to populate legality/cost arrays.
- Stop when queue empty, or after a max frame budget (safety).

### Reconstruction
- Walk parent pointers to generate full frame sequence.
- Emit `ControllerStep` per frame (matching emulator inputs). Include WAIT frames, hold flags, rotation taps.
- Generate `CapsuleState` per frame with counters for logging/comparison.

## Integration Tasks

1. **Snapshot** – extend `PillSnapshot` to read zero-page counters (speed counter, speed setting, speed ups, hor velocity, frame counter) and compute gravity threshold. Remove crude fallback guesses (e.g., gravity period = gravity counter).

2. **Planner API** – replace current BFS with counter-aware version (`build_reachability(board, snapshot, config)` returning masks, costs, and per-action frame plans). Keep orientation enum + index macro unchanged; only internal search changes.

3. **PlacementPlanner** – swap in new core, still returning `PlanResult` with controller sequence + state trace. Record BFS spawn as before but now frame logs include counters.

4. **PlacementTranslator / Executor**
   - Compare emulator state to planned state including counters where meaningful (row/col/orient primary; counters optional for diagnostics).
   - Log per-frame values (`row,col,orient,speed_counter,hor_velocity,hold flags`).
   - The action loop just replays the exact sequence; no re-issuing the same frame waiting for gravity.

5. **Tests & Diagnostics**
   - Unit tests: verify soft drop cadence, horizontal repeat behaviour, rotation with kicks, landing flow.
   - Integration check: capture emulator frame log and ensure it matches planned states for a few scripted inputs.

6. **Performance**
   - Keep branching minimal (prune no-op action combinations).
   - Bound search depth (gravity + DAS counters restrict frames naturally). If needed, pre-run WAIT frames while speed counter counts down.
   - Ensure planner remains microsecond-to-low-millisecond per spawn so emulator is still the long pole.

## Deliverables

- `envs/retro/fast_reach.py` (or replacement module): counter-aware frame simulator + BFS.
- Updated `PlacementPlanner`, `PlacementTranslator`, executor to consume new plans.
- New diagnostics: per-frame planner log vs emulator state, with counters.
- Updated tests ensuring frame-accurate playback.

## Notes

- Lock buffer (`currentP_fallingPillLockCounter` at $0307) can be integrated later for grounded slides; initial goal is to land accurately then stop.
- If the BFS state space becomes too large, consider caching speed thresholds and pruning repeated hold/neutral combinations.
- Keep “strict landing” behaviour configurable, but default to NES-accurate (stop once drop fails) for correctness.

