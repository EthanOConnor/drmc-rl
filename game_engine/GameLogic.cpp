#include "GameLogic.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

// === Overview ===
//
// This file is a *rules-exact* (frame-exact) reimplementation of the relevant
// parts of the Dr. Mario NES gameplay loop (demo + 1P play), based on:
//   - dr-mario-disassembly/prg/drmario_prg_game_logic.asm
//   - dr-mario-disassembly/prg/drmario_prg_visual_nametable.asm
//   - dr-mario-disassembly/defines/drmario_constants.asm
//
// Key design point:
//   The NES splits each frame into (1) NMI-time work and (2) main-thread work.
//   We emulate NMI-time side effects explicitly in `nmi_tick()` at the start of
//   each `step()`, then execute exactly one `nextAction` routine.

// === NES controller bits (same encoding used by the transcript tooling) ===
constexpr u8 BTN_RIGHT = 0x01;
constexpr u8 BTN_LEFT = 0x02;
constexpr u8 BTN_DOWN = 0x04;
constexpr u8 BTN_UP = 0x08;
[[maybe_unused]] constexpr u8 BTN_START = 0x10;
[[maybe_unused]] constexpr u8 BTN_SELECT = 0x20;
constexpr u8 BTN_B = 0x40;
constexpr u8 BTN_A = 0x80;

constexpr u8 BTNS_DPAD = BTN_RIGHT | BTN_LEFT | BTN_DOWN | BTN_UP;
constexpr u8 BTNS_LEFT_RIGHT = BTN_RIGHT | BTN_LEFT;

// === Timing constants (NTSC, match disassembly defaults) ===
constexpr u8 FAST_DROP_SPEED = 0x01; // fast_drop_speed
constexpr u8 HOR_ACCEL_SPEED = 0x10; // hor_accel_speed
constexpr u8 HOR_MAX_SPEED = 0x06;   // hor_max_speed
constexpr u8 SPEEDUPS_MAX = 0x31;    // speedUps_max

constexpr u8 PILL_START_ROW = 0x0F; // pillStartingY (0=bottom, 15=top)
constexpr u8 PILL_START_COL = 0x03; // pillStartingX

constexpr int LAST_COLUMN = 7;
constexpr int LAST_ROW = 15;

// Wait before the level starts (levelIntro_delay / waitFor_A_frames).
constexpr u16 LEVEL_INTRO_DELAY_FRAMES = 0x80;

// === Field object constants (drmario_constants.asm) ===
constexpr u8 MASK_FIELDOBJECT_TYPE = 0xF0;
constexpr u8 MASK_FIELDOBJECT_COLOR = 0x0F;
constexpr u8 MASK_COLOR = 0x03;

constexpr u8 TILE_EMPTY = 0xFF;              // fieldPosEmpty
constexpr u8 TILE_TOP = 0x40;                // topHalfPill
constexpr u8 TILE_BOTTOM = 0x50;             // bottomHalfPill
constexpr u8 TILE_LEFT = 0x60;               // leftHalfPill
constexpr u8 TILE_RIGHT = 0x70;              // rightHalfPill
constexpr u8 TILE_SINGLE = 0x80;             // singleHalfPill
constexpr u8 TILE_MIDDLE_VER = 0x90;         // middleVerHalfPill (unused in retail)
constexpr u8 TILE_MIDDLE_HOR = 0xA0;         // middleHorHalfPill (unused in retail)
constexpr u8 TILE_CLEARED = 0xB0;            // clearedPillOrVirus
constexpr u8 TILE_VIRUS = 0xD0;              // virus
constexpr u8 TILE_JUST_EMPTIED = 0xF0;       // fieldPosJustEmptied

// === Mario throw ("send pill") animation (drmario_prg_game_logic.asm $9FA1) ===
constexpr u8 SPR_MARIO_THROW_SPEED_LOW = 0x01;    // spr_mario_throw_speed_low
constexpr u8 PILL_THROWN_ROTATION_SPEED = 0x01;   // pillThrown_rotation_speed
constexpr u8 SPR_MARIO_THROW_FRAMES_MASK = 0x03;  // spr_mario_throw_frames_mask

// === BCD helpers (NES stores virus/pill counters as BCD) ===
constexpr u8 to_bcd(u8 value) {
  return static_cast<u8>(((value / 10) << 4) | (value % 10));
}

constexpr u8 bcd_inc_1byte(u8 value) {
  u8 next = static_cast<u8>(value + 1);
  if ((next & 0x0F) >= 0x0A)
    next = static_cast<u8>(next + 0x06);
  return next;
}

constexpr u8 bcd_dec_1byte(u8 value) {
  // Caller must ensure value != 0.
  u8 next = static_cast<u8>(value - 1);
  if ((next & 0x0F) == 0x0F)
    next = static_cast<u8>(next - 0x06);
  return next;
}

static void bcd_inc_2bytes(u8 &bcd_low, u8 &bcd_high) {
  // Mirrors `generateNextPill` ($8E9D) BCD logic:
  // - low: decimal (00..99)
  // - high: hundreds (00..99) (thousands support exists in disasm but is unused here)
  bcd_low = bcd_inc_1byte(bcd_low);
  if ((bcd_low & 0xF0) >= 0xA0) {
    // Carry 100 -> reset low and increment high.
    bcd_low = static_cast<u8>(bcd_low + 0x60); // A0 -> 00
    bcd_high = bcd_inc_1byte(bcd_high);
  }

  // Cap at 9999 (0x99 0x99). Retail games won't reach this in demo, but keep the behavior.
  if ((bcd_high & 0xF0) >= 0xA0) {
    bcd_low = 0x99;
    bcd_high = 0x99;
  }
}

// --- Lookup Tables (extracted from drmario_data_game.asm / constants) ---

static const u8 colorCombination_left[] = {0x00, 0x00, 0x00, 0x01, 0x01,
                                           0x01, 0x02, 0x02, 0x02};

static const u8 colorCombination_right[] = {0x00, 0x01, 0x02, 0x00, 0x01,
                                            0x02, 0x00, 0x01, 0x02};

static const u8 baseSpeedSettingValue[] = {0x0F, 0x19, 0x1F};

// Speed counters (NTSC) - frames to wait before gravity drop
static const u8 speedCounterTable[] = {
    // Source: dr-mario-disassembly/data/drmario_data_game.asm (NTSC / !ver_EU).
    0x45, 0x43, 0x41, 0x3F, 0x3D, 0x3B, 0x39, 0x37, 0x35, 0x33, 0x31, 0x2F,
    0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21, 0x1F, 0x1D, 0x1B, 0x19, 0x17,
    0x15, 0x13, 0x12, 0x11, 0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09,
    0x09, 0x08, 0x08, 0x07, 0x07, 0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00};

// `record_nes_demo.py` begins recording only after it detects the demo has
// started and the board is fully initialized, which in practice happens a few
// frames *after* the demo input replay has begun ticking in NMI.
//
// Empirically, for the captured ground-truth `data/nes_demo.json`, this offset
// is 7 frames: the first demo RIGHT occurs at input-stream frame 163, but the
// transcript records the resulting move at frame 156.
constexpr int DEMO_INPUT_PREROLL_FRAMES = 7;

static const u8 virusColor_random[] = {0x00, 0x01, 0x02, 0x02, 0x01, 0x00,
                                       0x00, 0x01, 0x02, 0x02, 0x01, 0x00,
                                       0x00, 0x01, 0x02, 0x01};

static const u8 addVirus_maxHeight_basedOnLvl[] = {
    0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
    0x09, 0x09, 0x09, 0x0A, 0x0A, 0x0B, 0x0B, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
    0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C};

// Demo pill reserve is 128 bytes in ROM. The disassembly labels >45 as "UNUSED",
// but the built-in demo can and does index beyond 45 (wrap is via &0x7F).
// Source: dr-mario-disassembly/data/drmario_data_demo_field_pills.asm
static const u8 demo_pills[] = {
    0x00, 0x00, 0x07, 0x02, 0x01, 0x05, 0x03, 0x05, 0x00, 0x06, 0x06, 0x03,
    0x05, 0x00, 0x05, 0x03, 0x05, 0x00, 0x06, 0x06, 0x04, 0x08, 0x07, 0x02,
    0x00, 0x02, 0x05, 0x00, 0x06, 0x07, 0x06, 0x04, 0x08, 0x06, 0x00, 0x06,
    0x06, 0x04, 0x00, 0x00, 0x07, 0x03, 0x04, 0x04, 0x03, 0x00, 0x03, 0x00,
    0x00, 0x07, 0x03, 0x03, 0x00, 0x02, 0x05, 0x00, 0x05, 0x04, 0x00, 0x01,
    0x01, 0x00, 0x06, 0x08, 0x02, 0x06, 0x02, 0x00, 0x02, 0x06, 0x02, 0x01,
    0x05, 0x04, 0x08, 0x06, 0x00, 0x05, 0x04, 0x08, 0x06, 0x08, 0x03, 0x00,
    0x01, 0x01, 0x01, 0x01, 0x00, 0x07, 0x02, 0x01, 0x05, 0x04, 0x08, 0x06,
    0x00, 0x06, 0x06, 0x04, 0x08, 0x07, 0x02, 0x01, 0x06, 0x06, 0x03, 0x05,
    0x08, 0x02, 0x06, 0x03, 0x04, 0x04, 0x03, 0x01, 0x05, 0x04, 0x00, 0x01,
    0x00, 0x06, 0x00, 0x05, 0x04, 0x00, 0x01, 0x01};

// Thrown-pill animation X positions. High nibble 0xF* are "markers" that update
// Mario's throw frame and do not consume a video frame.
// Source: `action_sendPill` ($9FA1).
static const u8 pillThrownAnim_XPos[] = {
    0xBE, 0xF1, 0xBE, 0xBC, 0xB8, 0xB4, 0xF2, 0xB0, 0xAC, 0xA8, 0xA4, 0xA0,
    0x9E, 0x98, 0x94, 0x90, 0x8C, 0x88, 0x84, 0x80, 0x7C, 0x7A, 0x78, 0xF0,
    0x78, 0xFF};

// Demo instruction set (512 bytes / 256 pairs).
// Source: dr-mario-disassembly/data/drmario_data_demo_inputs.asm (NTSC block).
static const u8 demo_instruction_set[] = {
    0x00, 0x00, 0x00, 0x08, 0x00, 0x97, 0x01, 0x08, 0x41, 0x06, 0x01, 0x17,
    0x00, 0x0E, 0x04, 0x08, 0x00, 0x24, 0x01, 0x06, 0x00, 0x03, 0x01, 0x06,
    0x00, 0x09, 0x04, 0x0E, 0x00, 0x31, 0x04, 0x0C, 0x00, 0x27, 0x01, 0x0C,
    0x41, 0x07, 0x01, 0x03, 0x41, 0x06, 0x40, 0x00, 0x00, 0x6C, 0x01, 0x09,
    0x00, 0x05, 0x01, 0x04, 0x00, 0x02, 0x40, 0x05, 0x00, 0x04, 0x40, 0x05,
    0x00, 0x11, 0x04, 0x03, 0x00, 0x6E, 0x02, 0x05, 0x00, 0x1B, 0x04, 0x07,
    0x00, 0x3A, 0x01, 0x04, 0x00, 0x07, 0x01, 0x06, 0x00, 0x09, 0x40, 0x04,
    0x00, 0x09, 0x01, 0x09, 0x00, 0x00, 0x40, 0x06, 0x00, 0x10, 0x04, 0x07,
    0x00, 0x27, 0x40, 0x04, 0x00, 0x05, 0x04, 0x07, 0x00, 0x69, 0x40, 0x05,
    0x00, 0x05, 0x40, 0x06, 0x00, 0x05, 0x04, 0x08, 0x00, 0x24, 0x40, 0x04,
    0x00, 0x05, 0x40, 0x04, 0x00, 0x11, 0x04, 0x06, 0x00, 0x7A, 0x02, 0x05,
    0x00, 0x0C, 0x04, 0x0C, 0x00, 0x27, 0x01, 0x1C, 0x41, 0x05, 0x01, 0x04,
    0x41, 0x04, 0x40, 0x01, 0x00, 0x11, 0x04, 0x0C, 0x00, 0x2D, 0x01, 0x05,
    0x41, 0x00, 0x40, 0x06, 0x00, 0x06, 0x04, 0x0E, 0x00, 0x24, 0x01, 0x11,
    0x41, 0x06, 0x01, 0x03, 0x41, 0x02, 0x40, 0x03, 0x00, 0x0F, 0x04, 0x07,
    0x00, 0x7B, 0x01, 0x04, 0x00, 0x00, 0x80, 0x08, 0x00, 0x09, 0x04, 0x09,
    0x00, 0x88, 0x01, 0x07, 0x00, 0x0E, 0x04, 0x0B, 0x00, 0x5F, 0x01, 0x15,
    0x40, 0x04, 0x00, 0x10, 0x04, 0x10, 0x00, 0x76, 0x01, 0x1A, 0x00, 0x04,
    0x40, 0x05, 0x00, 0x05, 0x40, 0x05, 0x00, 0x19, 0x04, 0x08, 0x00, 0x0C,
    0x01, 0x25, 0x41, 0x03, 0x01, 0x04, 0x00, 0x02, 0x40, 0x03, 0x00, 0x10,
    0x04, 0x0D, 0x00, 0x3C, 0x02, 0x07, 0x40, 0x05, 0x00, 0x09, 0x04, 0x0C,
    0x00, 0x3E, 0x02, 0x04, 0x00, 0x08, 0x40, 0x06, 0x00, 0x07, 0x04, 0x0B,
    0x00, 0x49, 0x01, 0x06, 0x00, 0x17, 0x80, 0x07, 0x00, 0x0A, 0x04, 0x0C,
    0x00, 0x6B, 0x01, 0x29, 0x00, 0x04, 0x04, 0x0C, 0x00, 0x51, 0x01, 0x06,
    0x00, 0x05, 0x01, 0x05, 0x00, 0x0B, 0x04, 0x09, 0x00, 0x67, 0x01, 0x04,
    0x00, 0x18, 0x80, 0x06, 0x00, 0x0C, 0x04, 0x0D, 0x00, 0x86, 0x02, 0x09,
    0x42, 0x07, 0x02, 0x03, 0x42, 0x06, 0x02, 0x0B, 0x00, 0x04, 0x04, 0x08,
    0x00, 0x2B, 0x01, 0x05, 0x00, 0x00, 0x40, 0x05, 0x00, 0x0B, 0x04, 0x14,
    0x00, 0x6F, 0x02, 0x0C, 0x42, 0x09, 0x02, 0x09, 0x00, 0x96, 0x02, 0x05,
    0x00, 0x08, 0x40, 0x05, 0x00, 0x05, 0x40, 0x05, 0x00, 0x14, 0x04, 0x0D,
    0x00, 0x7D, 0x04, 0x08, 0x00, 0x51, 0x02, 0x02, 0x42, 0x03, 0x40, 0x04,
    0x00, 0x0A, 0x04, 0x13, 0x00, 0x53, 0x01, 0x02, 0x00, 0x01, 0x40, 0x06,
    0x00, 0x02, 0x02, 0x05, 0x00, 0x0D, 0x04, 0x0E, 0x00, 0x95, 0x04, 0x0A,
    0x00, 0x34, 0x02, 0x11, 0x00, 0x22, 0x02, 0x1B, 0x42, 0x06, 0x02, 0x0C,
    0x00, 0x08, 0x04, 0x0D, 0x00, 0x3D, 0x02, 0x05, 0x00, 0x0C, 0x04, 0x09,
    0x00, 0x22, 0x02, 0x07, 0x00, 0x47, 0x04, 0x09, 0x00, 0x45, 0x02, 0x05,
    0x00, 0x0C, 0x02, 0x07, 0x00, 0x68, 0x02, 0x08, 0x00, 0x04, 0x02, 0x05,
    0x00, 0x03, 0x40, 0x04, 0x00, 0x06, 0x04, 0x09, 0x00, 0x40, 0x40, 0x05,
    0x00, 0x05, 0x04, 0x20, 0x00, 0x3E, 0x01, 0x05, 0x00, 0x0D, 0x40, 0x07,
    0x00, 0x16, 0x04, 0x17, 0x00, 0x39, 0x02, 0x1F, 0x00, 0x12, 0x04, 0x0F,
    0x00, 0x32, 0x01, 0x02, 0x00, 0x13, 0x40, 0x03, 0x00, 0x06, 0x40, 0x04,
    0x00, 0x08, 0x04, 0x10, 0x00, 0x15, 0x01, 0x1F, 0x00, 0x06, 0x01, 0x07,
    0x00, 0x14, 0x04, 0x13, 0x00, 0x24, 0x01, 0x09};

// Demo field (128 bytes) as stored in ROM.
// Source: dr-mario-disassembly/data/drmario_data_demo_field_pills.asm
static const u8 demo_field[] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xD2, 0xFF, 0xFF, 0xFF, 0xD2, 0xFF, 0xD1, 0xD0, 0xD2, 0xD1, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xD0, 0xFF, 0xD2, 0xFF, 0xD0, 0xD2, 0xD2, 0xFF,
    0xFF, 0xFF, 0xD2, 0xD0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xD2, 0xFF, 0xFF,
    0xD1, 0xD0, 0xFF, 0xD1, 0xFF, 0xFF, 0xD1, 0xFF, 0xD2, 0xD1, 0xD1, 0xD2,
    0xD2, 0xD1, 0xD0, 0xD2, 0xD2, 0xFF, 0xD1, 0xD0, 0xD1, 0xFF, 0xFF, 0xFF,
    0xFF, 0xD0, 0xD0, 0xD1, 0xFF, 0xD2, 0xD2, 0xD0, 0xD0, 0xD1, 0xFF, 0xD2,
    0xD2, 0xD1, 0xFF, 0xD0, 0xFF, 0xD1, 0xD1, 0xFF};

// Half-pill offsets and tile shapes per rotation (mirrors `confirmPlacement` tables).
static const int8_t half_row_offsets[4][2] = {
    {0, 0}, // rot 0: horizontal
    {0, 1}, // rot 1: vertical (2nd half is above)
    {0, 0}, // rot 2: horizontal (colors swapped)
    {1, 0}, // rot 3: vertical (colors swapped)
};

static const int8_t half_col_offsets[4][2] = {
    {0, 1}, // rot 0
    {0, 0}, // rot 1
    {1, 0}, // rot 2
    {0, 0}, // rot 3
};

static const u8 half_tile_shapes[4][2] = {
    {TILE_LEFT, TILE_RIGHT},
    {TILE_BOTTOM, TILE_TOP},
    {TILE_RIGHT, TILE_LEFT},
    {TILE_TOP, TILE_BOTTOM},
};

GameLogic::GameLogic(DrMarioState *sharedState) : state(sharedState) {}

void GameLogic::reset() {
  const bool demo = (state->mode == MODE_DEMO);
  state->mode = demo ? MODE_DEMO : MODE_PLAYING;

  state->stage_clear = 0;
  state->ending_active = 0x0A;
  state->level_fail = 0;
  state->frame_count = 0;

  state->buttons = 0;
  state->buttons_prev = 0;
  state->buttons_pressed = 0;
  state->buttons_held = 0;

  // Mirror initData_level ($8216): many RAM bytes are zeroed.
  state->speed_counter = 0;
  state->hor_velocity = 0;
  state->lock_counter = 0; // pillPlacedStep

  // `spawn_delay` is legacy; the NES uses nextAction/sendPill for this timing.
  state->spawn_delay = 0;

  // Speed-ups start at (level - 20), capped at SPEEDUPS_MAX.
  state->speed_ups =
      (state->level > 0x14) ? static_cast<u8>(state->level - 0x14) : 0;
  if (state->speed_ups > SPEEDUPS_MAX)
    state->speed_ups = SPEEDUPS_MAX;

  if (state->speed_setting > 2)
    state->speed_setting = 1;
  if (demo) {
    state->level = 0x0A;      // Demo level
    state->speed_setting = 1; // Medium
  }

  // RNG seed used by our engine for non-demo; demo mode uses fixed tables.
  state->rng_state[0] = 0x89;
  state->rng_state[1] = 0x88;

  // Reset internal (NES-private) state.
  next_action_ = NextAction::SendPill; // initData_level sets nextAction=sendPill
  pill_thrown_frame_ = 0;
  next_pill_rotation_ = 0;
  match_flag_ = 0;
  score_multiplier_ = 0;
  combo_counter_ = 0;
  chain_length_0based_ = 0;
  chain_length_ = 0;

  // Demo input replay state (getInputs_checkMode).
  demo_instr_ptr_ = 0;
  demo_inputs_held_ = 0;
  demo_counter_ = 0;

  // Align demo input replay with the NES ground-truth transcript start.
  if (demo) {
    for (int i = 0; i < DEMO_INPUT_PREROLL_FRAMES; ++i) {
      update_buttons_demo();
    }

    // The preroll represents skipped frames; clear edge-triggered presses.
    state->buttons_pressed = 0;
  }

  // The level intro sets status to 0x0F and then waits 0x80 frames (during which
  // NMI decrements status once per frame).
  status_row_ = 0x0F;
  intro_delay_frames_ = LEVEL_INTRO_DELAY_FRAMES;

  // Setup board/viruses.
  viruses_added = 0;
  viruses_to_place = 0;
  setupLevel();

  // Pill reserve + initial next pills (initData_level calls generateNextPill twice).
  generatePillsReserve();

  state->pill_counter = 0;       // pillsCounter (reserve index, wraps &0x7F)
  state->pill_counter_total = 0; // packed BCD (low=decimal, high=hundreds)

  state->preview_pill_color_l = 0;
  state->preview_pill_color_r = 0;
  state->preview_pill_size = 2;
  state->falling_pill_size = 2;

  generateNextPill();
  generateNextPill();

  // initData_level then sets pillsCounter_decimal to 1 (after the two calls).
  state->pill_counter_total = 0x0001;
}

void GameLogic::step() {
  nmi_tick();

  // Maintain legacy field for tooling; nextAction controls actual timing.
  state->spawn_delay = 0;

  if (!state->stage_clear && !state->level_fail) {
    if (intro_delay_frames_ > 0) {
      intro_delay_frames_--;
    } else {
      switch (next_action_) {
      case NextAction::PillFalling:
        action_pillFalling();
        break;
      case NextAction::PillPlaced:
        action_pillPlaced();
        break;
      case NextAction::CheckAttack:
        action_checkAttack();
        break;
      case NextAction::SendPillFinished:
        action_sendPillFinished();
        break;
      case NextAction::IncNextAction:
        action_incNextAction();
        break;
      case NextAction::SendPill:
        action_sendPill();
        break;
      default:
        break;
      }
    }
  }

  state->frame_count++;
}

// === NMI-time work (emulated) ===

void GameLogic::nmi_tick() {
  // In the NES, `render_fieldRow` decrements status once per frame (until 0xFF),
  // and checkDrop is gated on status == 0xFF.
  tick_status_row_render();

  // getInputs_checkMode ($9144) is also called during NMI.
  if (state->mode == MODE_DEMO) {
    update_buttons_demo();
  } else {
    update_buttons_from_raw(state->buttons);
  }
}

void GameLogic::tick_status_row_render() {
  if (status_row_ != 0xFF)
    status_row_--;
}

void GameLogic::update_buttons_from_raw(u8 raw_buttons) {
  state->buttons = raw_buttons;
  state->buttons_pressed = raw_buttons & static_cast<u8>(~state->buttons_prev);
  state->buttons_held = raw_buttons;
  state->buttons_prev = raw_buttons;
}

void GameLogic::update_buttons_demo() {
  // Implements getInputs_checkMode ($9144) demo branch (NTSC).
  //
  // `demo_counter_` is decremented every frame. When it reaches 0, we load the
  // next (buttons, duration) pair from ROM and apply it immediately.
  if (demo_counter_ != 0) {
    demo_counter_--;
    update_buttons_from_raw(demo_inputs_held_);
    return;
  }

  // End-of-table handling:
  // The retail ROM exits demo mode when the instruction pointer reaches the
  // end address, and it does so *immediately after* advancing the pointer for
  // the final (btn,duration) pair (i.e., the final pair is not applied).
  //
  // We approximate this by switching out of MODE_DEMO as soon as we would
  // consume the final pair.
  if (demo_instr_ptr_ + 1 >= sizeof(demo_instruction_set)) {
    state->mode = MODE_PLAYING;
    update_buttons_from_raw(demo_inputs_held_);
    return;
  }

  const u8 next_btns = demo_instruction_set[demo_instr_ptr_];
  const u8 next_dur = demo_instruction_set[demo_instr_ptr_ + 1];
  demo_instr_ptr_ += 2;

  if (demo_instr_ptr_ >= sizeof(demo_instruction_set)) {
    // Just consumed the last pair -> exit demo without applying it.
    state->mode = MODE_PLAYING;
    update_buttons_from_raw(demo_inputs_held_);
    return;
  }

  demo_inputs_held_ = next_btns;
  demo_counter_ = next_dur;
  update_buttons_from_raw(demo_inputs_held_);
}

// === Core gameplay actions (nextAction dispatcher, $9BD3) ===

void GameLogic::action_pillFalling() {
  fallingPill_checkYMove();
  fallingPill_checkXMove();
  fallingPill_checkRotate();
}

void GameLogic::action_pillPlaced() { toPillPlacedStep(); }

void GameLogic::action_checkAttack() {
  // 1P bypass: directly move to action_incNextAction ($9BE6).
  next_action_ = NextAction::IncNextAction;
}

void GameLogic::action_sendPillFinished() {
  next_action_ = NextAction::PillFalling;
}

void GameLogic::action_incNextAction() {
  next_action_ = static_cast<NextAction>(static_cast<u8>(next_action_) + 1);
}

void GameLogic::action_sendPill() {
  // `action_sendPill` ($9FA1): advances a throw animation and, when finished,
  // calls generateNextPill() to spawn the next falling pill.

  // Low speed updates only every other frame (uses the NES frameCounter LSB).
  if (state->speed_setting == 0) {
    if ((static_cast<u8>(state->frame_count) & SPR_MARIO_THROW_SPEED_LOW) == 0)
      return;
  }

  while (true) {
    pill_thrown_frame_++;
    const u8 anim_x = pillThrownAnim_XPos[pill_thrown_frame_];

    if (anim_x == 0xFF) {
      generateNextPill();
      next_pill_rotation_ = 0;
      pill_thrown_frame_ = 0;
      next_action_ = NextAction::SendPillFinished;

      if (!pillMoveValidation(state->falling_pill_row, state->falling_pill_col,
                              state->falling_pill_orient)) {
        confirmPlacement();
        state->level_fail = 1;
        state->fail_count++;
        state->last_fail_frame = state->frame_count;
        state->last_fail_row = state->falling_pill_row;
        state->last_fail_col = state->falling_pill_col;
      }
      return;
    }

    // Marker: update Mario throw frame and immediately advance to the next anim
    // entry *in the same frame* (does not consume a frame).
    if ((anim_x & 0xF0) == 0xF0) {
      const u8 mario_frame = anim_x & SPR_MARIO_THROW_FRAMES_MASK;
      (void)mario_frame; // visual-only in headless mode
      continue;
    }

    // Rotate the thrown pill every other thrown-frame (visual-only, but timing).
    if ((pill_thrown_frame_ & PILL_THROWN_ROTATION_SPEED) == 0) {
      next_pill_rotation_ = static_cast<u8>((next_pill_rotation_ - 1) & 0x03);
    }
    return;
  }
}

// === Falling pill logic ($8D66..$8E9C) ===

void GameLogic::fallingPill_checkYMove() {
  auto attempt_lower = [this]() {
    state->falling_pill_row = static_cast<u8>(state->falling_pill_row - 1);
    state->speed_counter = 0;

    if (!pillMoveValidation(state->falling_pill_row, state->falling_pill_col,
                            state->falling_pill_orient) ||
        state->falling_pill_row == 0xFF) {
      state->falling_pill_row++;
      confirmPlacement();
      next_action_ = NextAction::PillPlaced;
    }
  };

  // Fast drop: checked every 2 frames, only if down is the only dpad input.
  if ((static_cast<u8>(state->frame_count) & FAST_DROP_SPEED) != 0) {
    const u8 dpad = state->buttons_held & BTNS_DPAD;
    if (dpad == BTN_DOWN) {
      attempt_lower();
      return;
    }
  }

  state->speed_counter++;

  const int speed_setting_idx = std::clamp<int>(state->speed_setting, 0, 2);
  int table_idx =
      static_cast<int>(baseSpeedSettingValue[speed_setting_idx]) + state->speed_ups;
  const int table_size = static_cast<int>(sizeof(speedCounterTable));
  if (table_idx >= table_size)
    table_idx = table_size - 1;
  const u8 max_speed = speedCounterTable[table_idx];

  // Gravity triggers only when speedCounter > tableValue (cmp + bcs exit).
  if (state->speed_counter <= max_speed)
    return;

  attempt_lower();
}

void GameLogic::fallingPill_checkXMove() {
  const u8 pressed_lr = state->buttons_pressed & BTNS_LEFT_RIGHT;
  const u8 held_lr = state->buttons_held & BTNS_LEFT_RIGHT;

  if (pressed_lr == 0) {
    if (held_lr == 0)
      return;
    state->hor_velocity++;
    if (state->hor_velocity < HOR_ACCEL_SPEED)
      return;
    state->hor_velocity = static_cast<u8>(HOR_ACCEL_SPEED - HOR_MAX_SPEED);
  } else {
    state->hor_velocity = 0;
  }

  if (state->buttons_held & BTN_RIGHT) {
    const int right_boundary =
        (state->falling_pill_orient & 0x01) ? LAST_COLUMN : LAST_COLUMN - 1;
    if (state->falling_pill_col != right_boundary) {
      const u8 new_col = static_cast<u8>(state->falling_pill_col + 1);
      if (pillMoveValidation(state->falling_pill_row, new_col,
                             state->falling_pill_orient)) {
        state->falling_pill_col = new_col;
      } else {
        state->hor_velocity = static_cast<u8>(HOR_ACCEL_SPEED - 1);
      }
    }
  }

  if (state->buttons_held & BTN_LEFT) {
    if (state->falling_pill_col > 0) {
      const u8 new_col = static_cast<u8>(state->falling_pill_col - 1);
      if (pillMoveValidation(state->falling_pill_row, new_col,
                             state->falling_pill_orient)) {
        state->falling_pill_col = new_col;
      } else {
        state->hor_velocity = static_cast<u8>(HOR_ACCEL_SPEED - 1);
      }
    }
  }
}

void GameLogic::fallingPill_checkRotate() {
  const u8 prev_rot = state->falling_pill_orient;
  const u8 prev_col = state->falling_pill_col;

  if (state->buttons_pressed & BTN_A) { // clockwise (matches asm)
    state->falling_pill_orient =
        static_cast<u8>((state->falling_pill_orient + 3) & 0x03);
    pillRotateValidation(prev_rot, prev_col);
  }

  if (state->buttons_pressed & BTN_B) { // counter-clockwise
    state->falling_pill_orient =
        static_cast<u8>((state->falling_pill_orient + 1) & 0x03);
    pillRotateValidation(prev_rot, prev_col);
  }
}

void GameLogic::pillRotateValidation(u8 prev_rot, u8 prev_col) {
  const u8 rot_state = state->falling_pill_orient & 0x01;

  if (rot_state == 0) { // would be horizontal
    if (pillMoveValidation(state->falling_pill_row, state->falling_pill_col,
                           state->falling_pill_orient)) {
      if (state->buttons_held & BTN_LEFT) {
        if (state->falling_pill_col > 0) {
          const u8 new_col = static_cast<u8>(state->falling_pill_col - 1);
          if (pillMoveValidation(state->falling_pill_row, new_col,
                                 state->falling_pill_orient)) {
            state->falling_pill_col = new_col;
          }
        }
      }
      return;
    }

    // Wall kick: try offset to the left.
    if (state->falling_pill_col > 0) {
      const u8 kicked_col = static_cast<u8>(state->falling_pill_col - 1);
      if (pillMoveValidation(state->falling_pill_row, kicked_col,
                             state->falling_pill_orient)) {
        state->falling_pill_col = kicked_col;
        return;
      }
    }
  } else {
    if (pillMoveValidation(state->falling_pill_row, state->falling_pill_col,
                           state->falling_pill_orient)) {
      return;
    }
  }

  // Rotation invalid -> restore.
  state->falling_pill_orient = prev_rot;
  state->falling_pill_col = prev_col;
}

// === Pill placed state machine ($8C83 jump table) ===

void GameLogic::toPillPlacedStep() {
  switch (state->lock_counter) {
  case 0:
    pillPlaced_resetCombo();
    break;
  case 1:
  case 2:
    pillPlaced_incStep();
    break;
  case 3:
    pillPlaced_checkDrop();
    break;
  case 4:
    pillPlaced_resetMatchFlag();
    break;
  case 5:
    pillPlaced_checkHorMatch();
    break;
  case 6:
    pillPlaced_checkVerMatch();
    break;
  case 7:
    pillPlaced_updateField();
    break;
  case 8:
    pillPlaced_resetPillPlacedStep();
    break;
  default:
    // Safety: clamp to resetCombo.
    state->lock_counter = 0;
    pillPlaced_resetCombo();
    break;
  }
}

void GameLogic::pillPlaced_resetCombo() {
  score_multiplier_ = 0;
  state->lock_counter++;
}

void GameLogic::pillPlaced_incStep() { state->lock_counter++; }

void GameLogic::pillPlaced_checkDrop() {
  // checkDrop ($8CA4) is gated on status==0xFF.
  if (status_row_ != 0xFF)
    return;

  bool dropped_any = false;

  // Scan from bottom-right (127) backwards.
  for (int pos = BOARD_SIZE - 1; pos >= 0; --pos) {
    const u8 cur = state->board[pos];
    if (cur < TILE_JUST_EMPTIED)
      continue;

    // Empty / just-emptied positions are normalized to TILE_EMPTY (0xFF) here.
    state->board[pos] = TILE_EMPTY;

    const int above = pos - BOARD_WIDTH;
    if (above < 0)
      continue;

    const u8 top_tile = state->board[above];
    if (top_tile >= TILE_MIDDLE_HOR)
      continue;

    const u8 top_type = top_tile & MASK_FIELDOBJECT_TYPE;
    if (top_type == TILE_LEFT || top_type == TILE_MIDDLE_HOR)
      continue;

    if (top_type == TILE_RIGHT) {
      // Horizontal pill drop: we are under the right half. Require that the
      // corresponding below-left position(s) are empty.
      int top_pos = above;
      int bottom_pos = pos;
      int bottom_left = pos;
      int top_left = above;

      // Walk left until we find the left end of this horizontal pill (supports
      // longer pills in the disassembly).
      while (true) {
        top_left--;
        bottom_left--;
        if (bottom_left < 0 || top_left < 0)
          break;
        if (state->board[bottom_left] < TILE_JUST_EMPTIED)
          goto next_pos;

        const u8 tl_type = state->board[top_left] & MASK_FIELDOBJECT_TYPE;
        if (tl_type == TILE_LEFT)
          break;
        if (tl_type != TILE_MIDDLE_HOR)
          break;
      }

      // Drop from right to left until reaching the leftmost half.
      while (true) {
        state->board[bottom_pos] = state->board[top_pos];
        state->board[top_pos] = TILE_EMPTY;

        if (bottom_pos == bottom_left)
          break;

        bottom_pos--;
        top_pos--;
        dropped_any = true;
      }

      dropped_any = true;
      continue;
    }

    // Vertical or single: move the tile above down by one row.
    state->board[pos] = state->board[above];
    state->board[above] = TILE_EMPTY;
    dropped_any = true;

  next_pos:
    continue;
  }

  // After scanning, status is set to 0x0F (bottom row) and decremented by NMI.
  status_row_ = 0x0F;

  if (dropped_any) {
    // In the ROM, this is `dec pillPlacedStep` (loop drop logic again later).
    state->lock_counter--;
  } else {
    state->lock_counter++;
  }
}

void GameLogic::pillPlaced_resetMatchFlag() {
  match_flag_ = 0;
  state->lock_counter++;
}

static void maybe_virus_destroyed(DrMarioState *state, u8 &status_row) {
  if (state->viruses_remaining == 0x00)
    return;
  state->viruses_remaining = bcd_dec_1byte(state->viruses_remaining);
  status_row = 0x0F;
  if (state->viruses_remaining == 0x00)
    state->stage_clear = 1;
}

void GameLogic::pillPlaced_checkHorMatch() {
  // checkHorMatch ($920B): scans top-left to bottom-right row-by-row.
  for (int row = 0; row < BOARD_HEIGHT; ++row) {
    for (int col = 0; col <= LAST_COLUMN - 3; ++col) { // lastColumn_forMatch == 4
      const int start = row * BOARD_WIDTH + col;
      const u8 tile = state->board[start];
      if (tile >= TILE_JUST_EMPTIED)
        continue;

      const u8 color = tile & MASK_FIELDOBJECT_COLOR;
      int chain = 1;
      while (col + chain < BOARD_WIDTH) {
        const u8 nxt = state->board[row * BOARD_WIDTH + (col + chain)];
        if ((nxt & MASK_FIELDOBJECT_COLOR) != color)
          break;
        chain++;
      }

      if (chain < 4)
        continue;

      // Mark chain as cleared.
      combo_counter_++;
      chain_length_0based_ = static_cast<u8>(chain - 1);
      chain_length_ = static_cast<u8>(chain);

      for (int k = 0; k < chain; ++k) {
        const int idx = row * BOARD_WIDTH + (col + k);
        u8 &t = state->board[idx];
        if ((t & MASK_FIELDOBJECT_TYPE) == TILE_VIRUS) {
          maybe_virus_destroyed(state, status_row_);
        }
        t = static_cast<u8>(TILE_CLEARED | (t & MASK_FIELDOBJECT_COLOR));
      }

      match_flag_++;
      col += chain - 1; // Skip to end of chain (matches assembly behavior).
    }
  }

  state->lock_counter++;
}

void GameLogic::pillPlaced_checkVerMatch() {
  // checkVerMatch ($9479): scans column-by-column, top-to-bottom within column.
  for (int col = 0; col < BOARD_WIDTH; ++col) {
    for (int row = 0; row <= LAST_ROW - 3; ++row) { // lastRow_forMatch is 3 up from bottom
      const int start = row * BOARD_WIDTH + col;
      const u8 tile = state->board[start];
      if (tile >= TILE_JUST_EMPTIED)
        continue;

      const u8 color = tile & MASK_FIELDOBJECT_COLOR;
      int chain = 1;
      while (row + chain < BOARD_HEIGHT) {
        const u8 nxt = state->board[(row + chain) * BOARD_WIDTH + col];
        if ((nxt & MASK_FIELDOBJECT_COLOR) != color)
          break;
        chain++;
      }

      if (chain < 4)
        continue;

      combo_counter_++;
      chain_length_0based_ = static_cast<u8>(chain - 1);
      chain_length_ = static_cast<u8>(chain);

      for (int k = 0; k < chain; ++k) {
        const int idx = (row + k) * BOARD_WIDTH + col;
        u8 &t = state->board[idx];
        if ((t & MASK_FIELDOBJECT_TYPE) == TILE_VIRUS) {
          maybe_virus_destroyed(state, status_row_);
        }
        t = static_cast<u8>(TILE_CLEARED | (t & MASK_FIELDOBJECT_COLOR));
      }

      match_flag_++;
      row += chain - 1;
    }
  }

  state->lock_counter++;
}

void GameLogic::pillPlaced_updateField() {
  // updateField ($92EB): converts cleared tiles -> just emptied, and converts
  // orphan pill halves into singles (or adjusts middle halves for unsupported sizes).
  for (int pos = BOARD_SIZE - 1; pos >= 0; --pos) {
    u8 tile = state->board[pos];
    const u8 type = tile & MASK_FIELDOBJECT_TYPE;

    if (type == TILE_CLEARED) {
      state->board[pos] = static_cast<u8>(tile | TILE_JUST_EMPTIED);
      continue;
    }

    if (type == TILE_TOP) {
      const int below = pos + BOARD_WIDTH;
      const u8 below_type =
          (below < BOARD_SIZE) ? (state->board[below] & MASK_FIELDOBJECT_TYPE) : TILE_EMPTY;
      if (below_type != TILE_BOTTOM && below_type != TILE_MIDDLE_VER) {
        state->board[pos] = static_cast<u8>(TILE_SINGLE | (tile & MASK_FIELDOBJECT_COLOR));
      }
      continue;
    }

    if (type == TILE_BOTTOM) {
      const int above = pos - BOARD_WIDTH;
      const u8 above_type =
          (above >= 0) ? (state->board[above] & MASK_FIELDOBJECT_TYPE) : TILE_EMPTY;
      if (above_type != TILE_TOP && above_type != TILE_MIDDLE_VER) {
        state->board[pos] = static_cast<u8>(TILE_SINGLE | (tile & MASK_FIELDOBJECT_COLOR));
      }
      continue;
    }

    if (type == TILE_LEFT) {
      const int right = pos + 1;
      const u8 right_type =
          (right < BOARD_SIZE && (right % BOARD_WIDTH) != 0)
              ? (state->board[right] & MASK_FIELDOBJECT_TYPE)
              : TILE_EMPTY;
      if (right_type != TILE_RIGHT && right_type != TILE_MIDDLE_HOR) {
        state->board[pos] = static_cast<u8>(TILE_SINGLE | (tile & MASK_FIELDOBJECT_COLOR));
      }
      continue;
    }

    if (type == TILE_RIGHT) {
      const int left = pos - 1;
      const u8 left_type =
          (left >= 0 && (left % BOARD_WIDTH) != (BOARD_WIDTH - 1))
              ? (state->board[left] & MASK_FIELDOBJECT_TYPE)
              : TILE_EMPTY;
      if (left_type != TILE_LEFT && left_type != TILE_MIDDLE_HOR) {
        state->board[pos] = static_cast<u8>(TILE_SINGLE | (tile & MASK_FIELDOBJECT_COLOR));
      }
      continue;
    }

    if (type == TILE_MIDDLE_VER) {
      const int above = pos - BOARD_WIDTH;
      const int below = pos + BOARD_WIDTH;
      const u8 above_type =
          (above >= 0) ? (state->board[above] & MASK_FIELDOBJECT_TYPE) : TILE_EMPTY;
      const u8 below_type =
          (below < BOARD_SIZE) ? (state->board[below] & MASK_FIELDOBJECT_TYPE) : TILE_EMPTY;

      if (above_type != TILE_MIDDLE_VER && above_type != TILE_TOP) {
        state->board[pos] = static_cast<u8>(TILE_TOP | (tile & MASK_FIELDOBJECT_COLOR));
        tile = state->board[pos];
      }

      if (below_type != TILE_MIDDLE_VER && below_type != TILE_BOTTOM) {
        state->board[pos] = static_cast<u8>(TILE_BOTTOM | (tile & MASK_FIELDOBJECT_COLOR));
      }
      continue;
    }

    if (type == TILE_MIDDLE_HOR) {
      const int left = pos - 1;
      const int right = pos + 1;
      const u8 left_type =
          (left >= 0 && (left % BOARD_WIDTH) != (BOARD_WIDTH - 1))
              ? (state->board[left] & MASK_FIELDOBJECT_TYPE)
              : TILE_EMPTY;
      const u8 right_type =
          (right < BOARD_SIZE && (right % BOARD_WIDTH) != 0)
              ? (state->board[right] & MASK_FIELDOBJECT_TYPE)
              : TILE_EMPTY;

      if (left_type != TILE_MIDDLE_HOR && left_type != TILE_LEFT) {
        state->board[pos] = static_cast<u8>(TILE_LEFT | (tile & MASK_FIELDOBJECT_COLOR));
        tile = state->board[pos];
      }

      if (right_type != TILE_MIDDLE_HOR && right_type != TILE_RIGHT) {
        state->board[pos] = static_cast<u8>(TILE_RIGHT | (tile & MASK_FIELDOBJECT_COLOR));
      }
      continue;
    }
  }

  state->lock_counter++;
  status_row_ = 0x0F;

  if (match_flag_ != 0) {
    // Loop back to pillPlaced_nextStep1 (1) to re-run drops/matches after NMI gating.
    state->lock_counter = 1;
  }
}

void GameLogic::pillPlaced_resetPillPlacedStep() {
  state->lock_counter = 0;
  next_action_ = static_cast<NextAction>(static_cast<u8>(next_action_) + 1);
}

// === Pill generation + placement ($8E9D / $8F62) ===

void GameLogic::generateNextPill() {
  // Copy preview -> falling.
  state->falling_pill_color_l = state->preview_pill_color_l;
  state->falling_pill_color_r = state->preview_pill_color_r;
  state->falling_pill_orient = 0;
  state->falling_pill_size = state->preview_pill_size;

  // Select new preview from reserve or demo pills.
  u8 pill_id = 0;
  if (state->mode == MODE_DEMO) {
    pill_id = demo_pills[state->pill_counter & 0x7F];
  } else {
    pill_id = pillsReserve[state->pill_counter & 0x7F];
  }

  state->preview_pill_color_l = colorCombination_left[pill_id];
  state->preview_pill_color_r = colorCombination_right[pill_id];
  state->preview_pill_size = 2;

  // Increment reserve index (wrap at 128).
  state->pill_counter = static_cast<u8>((state->pill_counter + 1) & 0x7F);

  // Reset throw animation state (mirrors asm, though visual-only).
  pill_thrown_frame_ = 0;
  next_pill_rotation_ = 0;

  // Spawn the falling pill at its starting location.
  state->falling_pill_col = PILL_START_COL;
  state->falling_pill_row = PILL_START_ROW;

  // BCD pill counter + speedups (every 10 pills).
  u8 bcd_dec = static_cast<u8>(state->pill_counter_total & 0xFF);
  u8 bcd_hund = static_cast<u8>((state->pill_counter_total >> 8) & 0xFF);
  bcd_inc_2bytes(bcd_dec, bcd_hund);
  state->pill_counter_total = static_cast<u16>(bcd_dec | (static_cast<u16>(bcd_hund) << 8));

  if ((bcd_dec & 0x0F) == 0x00) {
    if (state->speed_ups < SPEEDUPS_MAX) {
      state->speed_ups++;
    }
  }

  // Reset combo counter (asm does this in generateNextPill).
  combo_counter_ = 0;
}

bool GameLogic::pillMoveValidation(int row, int col, int rot) {
  if (row < 0)
    return false;
  if (col < 0 || col >= BOARD_WIDTH)
    return false;

  rot &= 0x03;
  const int8_t *row_offsets = half_row_offsets[rot];
  const int8_t *col_offsets = half_col_offsets[rot];

  for (int i = 0; i < 2; i++) {
    const int target_row = row + row_offsets[i];
    const int target_col = col + col_offsets[i];

    if (target_row < 0)
      return false;
    if (target_col < 0 || target_col >= BOARD_WIDTH)
      return false;

    // Above top is allowed (half pills disappear when above the bottle).
    if (target_row >= BOARD_HEIGHT)
      continue;

    const int idx = (LAST_ROW - target_row) * BOARD_WIDTH + target_col;
    if (state->board[idx] != TILE_EMPTY)
      return false;
  }

  return true;
}

void GameLogic::confirmPlacement() {
  // confirmPlacement ($8F62): writes falling pill halves into the field and sets status.
  const int row = state->falling_pill_row;
  const int col = state->falling_pill_col;
  const int rot = state->falling_pill_orient & 0x03;

  const int8_t *row_offsets = half_row_offsets[rot];
  const int8_t *col_offsets = half_col_offsets[rot];
  const u8 *tiles = half_tile_shapes[rot];

  const u8 colors[2] = {static_cast<u8>(state->falling_pill_color_l & MASK_COLOR),
                        static_cast<u8>(state->falling_pill_color_r & MASK_COLOR)};

  for (int i = 0; i < 2; i++) {
    const int r = row + row_offsets[i];
    const int c = col + col_offsets[i];
    if (r < 0 || c < 0 || c >= BOARD_WIDTH)
      continue;
    if (r >= BOARD_HEIGHT)
      continue; // Above top disappears

    const int idx = (LAST_ROW - r) * BOARD_WIDTH + c;
    state->board[idx] = static_cast<u8>(tiles[i] | (colors[i] & MASK_FIELDOBJECT_COLOR));
  }

  status_row_ = static_cast<u8>(state->falling_pill_row ^ 0x0F);
}

// === Level setup / RNG ===

void GameLogic::setupLevel() {
  std::memset(state->board, TILE_EMPTY, BOARD_SIZE);

  if (state->mode == MODE_DEMO) {
    std::memcpy(state->board, demo_field, BOARD_SIZE);
    int virus_count = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
      if ((state->board[i] & MASK_FIELDOBJECT_TYPE) == TILE_VIRUS)
        virus_count++;
    }
    state->viruses_remaining = to_bcd(static_cast<u8>(virus_count));
    return;
  }

  // Normal level setup: place viruses using the NES rules.
  int level = state->level;
  if (level > 20)
    level = 20;
  viruses_to_place = (level + 1) * 4;
  viruses_added = 0;

  state->viruses_remaining = 0x00; // BCD, incremented per placed virus.

  int attempts = 0;
  const int max_attempts = std::max(viruses_to_place * 64, 256);
  while (viruses_added < viruses_to_place && attempts < max_attempts) {
    addVirus();
    attempts++;
  }
}

bool GameLogic::addVirus() {
  int level = state->level;
  if (level > 20)
    level = 20;

  rng_step();
  const u8 r1 = rng_get();
  const int virusHeight = r1 & 0x0F; // 0 (bottom) .. 15 (top)
  if (virusHeight > addVirus_maxHeight_basedOnLvl[level])
    return false;

  rng_step();
  const u8 r2 = rng_get();
  const int col = r2 & 0x07; // 0..7

  if (viruses_added >= viruses_to_place)
    return false;

  const int pos = (LAST_ROW - virusHeight) * BOARD_WIDTH + col;
  if (state->board[pos] != TILE_EMPTY)
    return false;

  const int color_cycle = viruses_added & 0x03;
  u8 virusColor = 0;
  if (color_cycle == 3) {
    rng_step();
    virusColor = virusColor_random[rng_get() & 0x0F];
  } else {
    virusColor = static_cast<u8>(color_cycle);
  }

  auto has_same_color = [&](int row, int column) -> bool {
    const u8 tile = get_board_tile(row, column);
    if (tile == TILE_EMPTY)
      return false;
    return (tile & MASK_COLOR) == virusColor;
  };

  for (int delta = 1; delta <= 2; delta++) {
    if (has_same_color(virusHeight + delta, col))
      return false;
    if (has_same_color(virusHeight - delta, col))
      return false;
    if (has_same_color(virusHeight, col + delta))
      return false;
    if (has_same_color(virusHeight, col - delta))
      return false;
  }

  state->board[pos] = static_cast<u8>(TILE_VIRUS | (virusColor & MASK_COLOR));
  viruses_added++;

  // BCD increment virusLeft.
  state->viruses_remaining = bcd_inc_1byte(state->viruses_remaining);
  return true;
}

void GameLogic::generatePillsReserve() {
  int pillId = 0;
  for (int i = 0; i < 128; i++) {
    rng_step();
    const u8 r = rng_get();
    pillId = (pillId + (r & 0x0F));
    while (pillId >= 9)
      pillId -= 9;
    pillsReserve[i] = static_cast<u8>(pillId);
  }
}

u8 GameLogic::rng_get() { return state->rng_state[0]; }

void GameLogic::rng_step() {
  // EOR bit1 of rng0/rng1 -> carry, then ROR across both bytes.
  u8 carry = ((state->rng_state[0] ^ state->rng_state[1]) & 0x02) ? 1 : 0;

  for (int i = 0; i < 2; i++) {
    const u8 val = state->rng_state[i];
    const u8 new_val = static_cast<u8>((val >> 1) | (carry << 7));
    carry = val & 0x01;
    state->rng_state[i] = new_val;
  }
}

u8 GameLogic::get_board_tile(int row, int col) {
  if (row < 0 || row > 15 || col < 0 || col > 7)
    return TILE_EMPTY;
  return state->board[(15 - row) * 8 + col];
}

void GameLogic::set_board_tile(int row, int col, u8 val) {
  if (row >= 0 && row <= 15 && col >= 0 && col <= 7) {
    state->board[(15 - row) * 8 + col] = val;
  }
}

bool GameLogic::is_free(int row, int col) {
  return get_board_tile(row, col) == TILE_EMPTY;
}
