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
  u8 control_flags; // Bit0: start gate; Bit1: manual stepping; Bit2: step once
  u8 pad0;

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
  u8 preview_pill_size;    // 0x0323

  // Game Status
  u8 mode;          // 0x0046 (0x04 = Playing)
  u8 stage_clear;   // 0x0055 (0x01 = Cleared)
  u8 ending_active; // 0x0053 (0x0A = Not ending)
  u8 level_fail;    // top-out detection for loop/reset

  // Counters & Settings
  u8 pill_counter; // 0x0310
  u16 pill_counter_total;
  u8 level;             // 0x0316
  u8 speed_setting;     // 0x008B (Low/Med/Hi)
  u8 viruses_remaining; // 0x0324
  u8 speed_ups;         // Number of speed ups applied

  // Physics / Timers
  u8 gravity_counter; // 0x0312
  u8 lock_counter;    // 0x0307
  u8 speed_counter;   // Internal counter for gravity
  u8 hor_velocity;    // DAS accumulator

  // RNG State (for reproducibility)
  u8 rng_state[2]; // 0x0017, 0x0018

  // Frame Counter (Internal)
  u32 frame_count;
  u32 frame_budget; // Optional: break loop when reached (>0)
  u32 fail_count;
  u32 last_fail_frame;
  u8 last_fail_row;
  u8 last_fail_col;
  u8 spawn_delay; // Frames before new pill can be controlled (throw animation)
  u8 pad1[1];
};

static_assert(sizeof(DrMarioState) == 180,
              "DrMarioState layout changed unexpectedly");
static_assert(offsetof(DrMarioState, control_flags) == 4,
              "control_flags offset mismatch");
static_assert(offsetof(DrMarioState, frame_count) == 160,
              "frame_count offset mismatch");
