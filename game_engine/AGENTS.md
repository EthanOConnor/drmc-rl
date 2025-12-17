# Dr. Mario Engine Reimplementation - Agent Guide

## Project Context & Mission
**Goal:** Develop a high-performance, "no holds barred" C++ implementation of the NES Dr. Mario game logic to serve as a headless environment for Reinforcement Learning (RL) speedrunning agents.

**Why:**
*   **Performance:** Remove emulator overhead to maximize training throughput.
*   **Precision:** Solve "placement translation" issues by having direct control over game logic and state.
*   **Control:** Create a deterministic environment where we control the "Game Loop" (Level start to Game End).

**Scope:**
*   **Gameplay Only:** No Title Screen, No Menu, No Music, No Graphical PPU emulation.
*   **Mechanics:** "Level start to game end". We recreate the intervals where the player has control.
*   **Architecture:**
    *   **Language:** C++.
    *   **Communication:** Shared Memory (Agent reads state, writes buttons; Engine reads buttons, writes state). Non-blocking.
    *   **Evolution:**
        1.  **Direct Port:** Strictly translate assembly logic 1:1 to C++ to ensure accuracy.
        2.  **Optimized:** Refactor for performance (language/processor specific optimizations) once correctness is verified.

---

## The Authoritative Source: `dr-mario-disassembly`
The `dr-mario-disassembly/` directory contains the reverse-engineered source code of the original NES game. This is our "Single Source of Truth".

### Quick Directory Guide
*   **`prg/drmario_prg_game_logic.asm`** (CRITICAL): Contains the core gameplay mechanics.
    *   **Key Routines:** `gameLoop`, `fallingPill_checkYMove` (Gravity), `fallingPill_checkXMove` (Input), `checkMatches` (Scoring/Clearing), `addVirus` (Level Gen).
    *   **Status:** This file is the primary reference for 90% of the reimplementation work.
*   **`data/drmario_data_game.asm`**: Contains critical lookup tables.
    *   `speedCounterTable` (Gravity speeds), `virusPlacement` tables, `colorCombination` tables.
*   **`defines/drmario_ram.asm`**: The Memory Map.
    *   Definitions for `p1_fallingPillX`, `p1_board` ($0400), `rng_state`, etc.
    *   Use this to understand what variables in `game_logic.asm` actually refer to.
*   **`defines/drmario_constants.asm`**: Enums and Magic Numbers.
    *   Tile IDs (`$D0` = Virus), Color Masks, State Flags.

---

## Current Implementation Status (Game Engine)

**Last Updated:** December 2025
**Location:** `game_engine/`
**Language:** C++

### Done ✅
*   **Shared Memory:** Basic structure defined (`GameState.h`) and visualized (`monitor.cpp`).
*   **RNG:** Accurate LFSR implementation (Bit 1 EOR + ROR) verified against assembly.
*   **Level Generation:** Accurate virus placement using original algorithms and tables (`GameLogic.cpp`).
*   **Gravity/Dropping:** Full implementation of `fallingPill_checkYMove` including fast-drop timing.
*   **Matching:** Horizontal/Vertical match detection and clearing.
*   **Post-Clear Gravity:** Floating blocks fall correctly with recursive match checking.
*   **DAS (Delayed Auto Shift):** Implemented in `fallingPill_checkXMove`:
    *   Initial delay: 16 frames (`HOR_ACCEL_SPEED = 0x10`)
    *   Auto-repeat: 6 frames (`HOR_MAX_SPEED = 0x06`)
    *   Button press resets velocity, button held increments
*   **Wall Kicks:** Implemented in `pillRotateValidation`:
    *   When rotating to horizontal and blocked, tries shifting left
    *   Matches original NES behavior at `$8E70`

### Remaining / To-Do
1.  **Python Interface:**
    *   **Task:** Build a Python driver to interface with the Shared Memory, allowing us to feed input sequences and assert board states for testing.
2.  **Parity Testing:**
    *   **Task:** Use the "Demo Mode" data (`dr-mario-disassembly/data/drmario_data_demo_*.asm`) to verify that our engine produces the *exact* same board and moves as the original game given the same seed/inputs.
3.  **Scoring:** 
    *   Not strictly needed for RL but useful for completeness. See `scoreIncrease` routine.

### Useful References (in `dr-mario-disassembly/`)
*   **Gravity:** `fallingPill_checkYMove` in `prg/drmario_prg_game_logic.asm` (lines 232-282)
*   **Input/DAS:** `fallingPill_checkXMove` in `prg/drmario_prg_game_logic.asm` (lines 286-352)
*   **Rotation:** `fallingPill_checkRotate` (lines 355-393) and `pillRotateValidation` (lines 396-433)
*   **Scoring:** `scoreIncrease` (Note: We might not strictly need scoring for RL immediately, but good for completeness).

---

## Development Protocol
1.  **Read Assembly:** Always check the `.asm` files first.
2.  **Implement:** Translate the logic to C++ in `GameLogic.cpp`.
3.  **Verify:** Use `monitor` to visually check or write a test case.
4.  **Optimize:** Only optimize AFTER the logic is proven correct against the NES implementation.