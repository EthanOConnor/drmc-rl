# Dr. Mario Dynamics Specification

> Extracted from NES disassembly as authoritative reference for C++ engine parity.

## 1. Gravity System (`fallingPill_checkYMove`)

### Constants
```
FAST_DROP_SPEED = 0x01     ; Check down button every 2 frames (frameCounter & 0x01)
BASE_SPEED_LOW  = 0x0F     ; Speed setting 0 (Low)
BASE_SPEED_MED  = 0x19     ; Speed setting 1 (Med) 
BASE_SPEED_HI   = 0x1F     ; Speed setting 2 (Hi)
```

### Pseudocode
```python
def check_y_move(state):
    # Fast drop: checked on ODD frames only (frame & 0x01 != 0)
    if (state.frame_counter & FAST_DROP_SPEED) != 0:
        # Down-only check: exactly btn_down, no other dpad
        if (state.buttons_held & BTNS_DPAD) == BTN_DOWN:
            attempt_lower_pill(state)
            return
    
    # Gravity
    state.speed_counter += 1
    
    # Calculate gravity threshold
    base = BASE_SPEED[state.speed_setting]  # 0x0F, 0x19, or 0x1F
    table_idx = base + state.speed_ups
    threshold = SPEED_COUNTER_TABLE[table_idx]
    
    # Drop if counter EXCEEDS threshold (bcs = branch if carry set = >=)
    if state.speed_counter > threshold:
        attempt_lower_pill(state)

def attempt_lower_pill(state):
    state.falling_pill_y -= 1
    state.speed_counter = 0
    
    if not pill_move_valid(state) or state.falling_pill_y == 0xFF:
        # Invalid: restore Y and lock
        state.falling_pill_y += 1
        confirm_placement(state)
```

### Critical Bug Found!
**C++ (line 305):** `if (state->speed_counter <= max_speed) return;`

**ASM (line 251-252):**
```asm
lda speedCounterTable,X
cmp currentP_speedCounter   ; Compare table value TO counter
bcs @exit                   ; Exit if table >= counter
```

The ASM uses `bcs` which means "branch if carry set" → exit if `threshold >= counter`.
So drop occurs when `counter > threshold`, i.e., **strictly greater than**.

The C++ says `<=` which means drop when `counter > threshold` — **this is correct!**

---

## 2. Horizontal Movement / DAS (`fallingPill_checkXMove`)

### Constants
```
HOR_ACCEL_SPEED = 0x10     ; 16 frames initial DAS delay
HOR_MAX_SPEED   = 0x06     ; 6 frames auto-repeat
```

### Pseudocode
```python
def check_x_move(state):
    pressed_lr = state.buttons_pressed & BTNS_LEFT_RIGHT
    held_lr = state.buttons_held & BTNS_LEFT_RIGHT
    
    if pressed_lr == 0:
        # No new press - check hold
        if held_lr == 0:
            return  # Nothing held either
        
        # Held: increment velocity
        state.hor_velocity += 1
        if state.hor_velocity < HOR_ACCEL_SPEED:
            return  # Still in initial delay
        
        # Reset to repeat period
        state.hor_velocity = HOR_ACCEL_SPEED - HOR_MAX_SPEED  # = 10
    else:
        # Fresh press: reset velocity, will move
        state.hor_velocity = 0
        play_sfx(SQ0_PILL_MOVE_X)  # SFX on PRESS only
    
    # Try right
    if state.buttons_held & BTN_RIGHT:
        # Boundary: col 6 if horizontal, col 7 if vertical
        boundary = LAST_COLUMN - 1 + (state.pill_orient & 0x01)
        if state.pill_col < boundary:
            state.pill_col += 1
            if not pill_move_valid(state):
                state.pill_col -= 1
                state.hor_velocity = HOR_ACCEL_SPEED - 1
            else:
                play_sfx(SQ0_PILL_MOVE_X)  # SFX on held-move too
    
    # Try left (independent of right, both can be pressed!)
    if state.buttons_held & BTN_LEFT:
        if state.pill_col > 0:
            state.pill_col -= 1
            if not pill_move_valid(state):
                state.pill_col += 1
                state.hor_velocity = HOR_ACCEL_SPEED - 1
```

### Notes
- Both L+R can be pressed simultaneously (each checked independently)
- On blocked movement: set velocity to 15 so next held frame = 16 = move

---

## 3. Rotation (`fallingPill_checkRotate` + `pillRotateValidation`)

### Pseudocode
```python
def check_rotate(state):
    saved_rot = state.pill_orient
    saved_col = state.pill_col
    
    if state.buttons_pressed & BTN_A:
        play_sfx(SQ0_PILL_ROTATE)
        # Clockwise = decrement (wrapping via AND 0x03)
        state.pill_orient = (state.pill_orient - 1) & 0x03
        pill_rotate_validation(state, saved_rot, saved_col)
    
    if state.buttons_pressed & BTN_B:
        play_sfx(SQ0_PILL_ROTATE)
        # Counter-clockwise = increment
        state.pill_orient = (state.pill_orient + 1) & 0x03
        pill_rotate_validation(state, saved_rot, saved_col)

def pill_rotate_validation(state, prev_rot, prev_col):
    is_horizontal = (state.pill_orient & 0x01) == 0
    
    if is_horizontal:  # Would be horizontal after rotation
        if pill_move_valid(state):
            # Valid! Check "double left" if left held
            if state.buttons_held & BTN_LEFT:
                state.pill_col -= 1
                if not pill_move_valid(state):
                    state.pill_col += 1
            return
        
        # Wall kick: try shifting left
        state.pill_col -= 1
        if pill_move_valid(state):
            return
    else:
        # Would be vertical
        if pill_move_valid(state):
            return
    
    # Invalid: restore
    state.pill_orient = prev_rot
    state.pill_col = prev_col
```

### Key Points
- A = clockwise (decrement rotation)
- B = counter-clockwise (increment rotation)
- Both A+B on same frame: both rotations attempted!
- Wall kick: only when rotating TO horizontal, shift left if blocked

---

## 4. Comparison with C++ Implementation

### Verified Correct ✅
- DAS timing (16 frame initial, 6 frame repeat)
- Wall kick direction (left only)
- Gravity threshold comparison (strictly greater than)
- Fast drop frame parity check
- Demo input replay (`demo_instructionSet` / `counterDemoInstruction`) and demo parity fixture

### Notes

1. **Fast drop frame check**
   - ASM: `beq @checkGravity` when `(frame & 0x01) == 0` → check down on ODD frames
   - C++: `if ((frame_count & FAST_DROP_SPEED) != 0)` → check down on ODD frames
   - ✅ Matches

2. **DAS velocity on blocked move**
   - ASM: `HOR_ACCEL_SPEED - 1` (= 15)
   - C++: `HOR_ACCEL_SPEED - 1` (= 15)
   - ✅ Matches

3. **Demo parity**
   - The C++ engine replays the demo input stream internally (`getInputs_checkMode` demo branch).
   - Verified by `tests/test_game_engine_demo.py::test_demo_trace_matches_nes_ground_truth` against `data/nes_demo.json`.
