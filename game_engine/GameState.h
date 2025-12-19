#pragma once

#include <cstddef>
#include <cstdint>

// Fixed size types for exact NES memory mapping
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using s8 = int8_t;

// Constants
constexpr int BOARD_WIDTH = 8;
constexpr int BOARD_HEIGHT = 16;
constexpr int BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT;
constexpr u8 MODE_DEMO = 0x00;
constexpr u8 MODE_PLAYING = 0x04;

struct DrMarioState {
  // --- Input (Written by Agent, Read by Engine) ---
  u8 buttons;      // Standard NES controller encoding (A, B, Select, Start, Up,
                   // Down, Left, Right)
  u8 buttons_prev; // For edge detection (internal bookkeeping)
  u8 buttons_pressed; // Newly pressed this frame (edge triggered)
  u8 buttons_held;    // Currently held buttons (debounced)
  u8 control_flags; // Bit0: start gate; Bit1: manual stepping; Bit2: step once;
                    // Bit3: request exit; Bit4: request reset.
  u8 next_action; // currentP_nextAction ($0097), jumpTable_nextAction index.

  // --- Game State (Written by Engine, Read by Agent) ---

  // Board: 0x0400 - 0x047F
  // Each byte represents a tile.
  u8 board[BOARD_SIZE];

  // Falling Pill
  u8 falling_pill_row;     // 0x0306
  u8 falling_pill_col;     // 0x0305
  u8 falling_pill_orient;  // 0x0325 (0=Horiz, 1=Vert, etc)
  u8 falling_pill_color_l; // 0x0301
  u8 falling_pill_color_r; // 0x0302
  u8 falling_pill_size;    // 0x0326

  // Preview Pill
  u8 preview_pill_color_l; // 0x031A
  u8 preview_pill_color_r; // 0x031B
  u8 preview_pill_rotation; // 0x0322 (visual-only; useful for parity/debug)
  u8 preview_pill_size;    // 0x0323

  // Game Status
  u8 mode;          // 0x0046 (0x04 = Playing)
  u8 stage_clear;   // 0x0055 (0x01 = Cleared)
  u8 ending_active; // 0x0053 (0x0A = Not ending)
  u8 level_fail;    // top-out detection for loop/reset

  // Counters & Settings
  u8 pill_counter; // p1_pillsCounter ($0327), reserve index (wraps &0x7F)
  u16 pill_counter_total; // p1_pillsCounter_decimal/hundreds ($0310/$0311), packed BCD
  u8 level;             // p1_level ($0316)
  u8 speed_setting;     // p1_speedSetting ($030B), mirrored in ZP ($008B)
  u8 viruses_remaining; // p1_virusLeft ($0324), BCD
  u8 speed_ups;         // p1_speedUps ($030A), mirrored in ZP ($008A)

  // Timers
  // `waitFrames` ($0051): used during level intro delay (`waitFor_A_frames`).
  // We expose it to Python for parity + tooling; it is 0 during normal play.
  u8 wait_frames;
  u8 lock_counter;    // p1_pillPlacedStep ($0307)
  u8 speed_counter;   // p1_speedCounter ($0312), gravity timing accumulator
  u8 hor_velocity;    // p1_horVelocity ($0313), DAS accumulator

  // RNG State (for reproducibility)
  u8 rng_state[2]; // 0x0017, 0x0018
  u8 rng_override; // If non-zero, `reset()` preserves rng_state and clears this.

  // Frame Counter (Internal)
  u32 frame_count;
  u32 frame_budget; // Optional: break loop when reached (>0)
  u32 fail_count;
  u32 last_fail_frame;
  u8 last_fail_row;
  u8 last_fail_col;
  u8 spawn_delay; // Frames before new pill can be controlled (throw animation)
  u8 reset_wait_frames; // If non-zero, `reset()` uses this as the intro waitFrames seed.
  u16 reset_framecounter_lo_plus1; // Optional: (frameCounter_lo + 1) seed for parity resets.
};

static_assert(sizeof(DrMarioState) == 188,
              "DrMarioState layout changed unexpectedly");
static_assert(offsetof(DrMarioState, control_flags) == 4,
              "control_flags offset mismatch");
static_assert(offsetof(DrMarioState, frame_count) == 164,
              "frame_count offset mismatch");
