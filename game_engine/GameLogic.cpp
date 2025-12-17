#include "GameLogic.h"
#include <algorithm>
#include <cstring>
#include <iostream>

// Controller bits (NES order)
constexpr u8 BTN_RIGHT = 0x01;
constexpr u8 BTN_LEFT = 0x02;
constexpr u8 BTN_DOWN = 0x04;
constexpr u8 BTN_UP = 0x08;
constexpr u8 BTN_START = 0x10;
constexpr u8 BTN_SELECT = 0x20;
constexpr u8 BTN_B = 0x40;
constexpr u8 BTN_A = 0x80;

constexpr u8 BTNS_DPAD = BTN_RIGHT | BTN_LEFT | BTN_DOWN | BTN_UP;
constexpr u8 BTNS_LEFT_RIGHT = BTN_RIGHT | BTN_LEFT;

// Modes
// Timing constants (match asm defaults, NTSC)
constexpr u8 FAST_DROP_SPEED = 0x01;
constexpr u8 HOR_ACCEL_SPEED = 0x10;
constexpr u8 HOR_MAX_SPEED = 0x06;
constexpr u8 SPEEDUPS_MAX = 0x31;

constexpr u8 PILL_START_ROW = 0x0F;
constexpr u8 PILL_START_COL = 0x03;
constexpr int LAST_COLUMN = 7;
constexpr int LAST_ROW = 15;

// Tile Constants
constexpr u8 TILE_EMPTY = 0xFF;
constexpr u8 TILE_VIRUS_START = 0xD0;
constexpr u8 TILE_LEFT = 0x60;
constexpr u8 TILE_RIGHT = 0x70;
constexpr u8 TILE_TOP = 0x40;
constexpr u8 TILE_BOTTOM = 0x50;
constexpr u8 TILE_SINGLE = 0x80;
constexpr u8 MASK_COLOR = 0x03;
constexpr u8 MASK_TYPE = 0xF0;

// --- Lookup Tables (extracted from drmario_data_game.asm) ---

static const u8 colorCombination_left[] = {0x00, 0x00, 0x00, 0x01, 0x01,
                                           0x01, 0x02, 0x02, 0x02};

static const u8 colorCombination_right[] = {0x00, 0x01, 0x02, 0x00, 0x01,
                                            0x02, 0x00, 0x01, 0x02};

// Speed counters (NTSC) - frames to wait before gravity drop
static const u8 speedCounterTable[] = {
    0x35, 0x33, 0x31, 0x2F, 0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21,
    0x1F, 0x1D, 0x1B, 0x19, 0x17, 0x15, 0x13, 0x12, 0x11, 0x10, 0x0F,
    0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x09, 0x08, 0x08, 0x07, 0x07,
    0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x04, 0x04, 0x04, 0x04, 0x04, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00};

static const u8 baseSpeedSettingValue[] = {0x0F, 0x19, 0x1F};

static const u8 virusColor_random[] = {0x00, 0x01, 0x02, 0x02, 0x01, 0x00,
                                       0x00, 0x01, 0x02, 0x02, 0x01, 0x00,
                                       0x00, 0x01, 0x02, 0x01};

static const u8 addVirus_maxHeight_basedOnLvl[] = {
    0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
    0x09, 0x09, 0x09, 0x0A, 0x0A, 0x0B, 0x0B, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
    0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C};

static const u8 demo_pills[] = {
    0x00, 0x00, 0x07, 0x02, 0x01, 0x05, 0x03, 0x05, 0x00, 0x06, 0x06, 0x03,
    0x05, 0x00, 0x05, 0x03, 0x05, 0x00, 0x06, 0x06, 0x04, 0x08, 0x07, 0x02,
    0x00, 0x02, 0x05, 0x00, 0x06, 0x07, 0x06, 0x04, 0x08, 0x06, 0x00, 0x06,
    0x06, 0x04, 0x00, 0x00, 0x07, 0x03, 0x04, 0x04, 0x03};

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

// Maps fallingPillY (0=bottom, 15=top) to board offset (0=top, 120=bottom)
static const u8 pill_fieldPos_relativeToPillY[] = {
    0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
    0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00};

// Half-pill offsets and tile shapes per rotation (mirror asm tables)
static const int8_t half_row_offsets[4][2] = {
    {0, 0}, // rot 0
    {0, 1}, // rot 1 (second half is one row above)
    {0, 0}, // rot 2 (first half is to the right)
    {1, 0}  // rot 3 (first half is one row above base)
};

static const int8_t half_col_offsets[4][2] = {
    {0, 1}, // rot 0
    {0, 0}, // rot 1
    {1, 0}, // rot 2
    {0, 0}  // rot 3
};

static const u8 half_tile_shapes[4][2] = {
    {0x60, 0x70}, // Left, Right
    {0x50, 0x40}, // Bottom, Top
    {0x70, 0x60}, // Right, Left
    {0x40, 0x50}  // Top, Bottom
};

GameLogic::GameLogic(DrMarioState *sharedState) : state(sharedState) {}

void GameLogic::reset() {
  u8 start_mode = (state->mode == MODE_DEMO) ? MODE_DEMO : MODE_PLAYING;
  state->mode = start_mode;
  state->stage_clear = 0;
  state->ending_active = 0x0A;
  state->level_fail = 0;
  state->frame_count = 0;

  state->buttons = 0;
  state->buttons_prev = 0;
  state->buttons_pressed = 0;
  state->buttons_held = 0;

  state->falling_pill_size = 2;
  state->preview_pill_size = 2;

  state->pill_counter = 0;
  state->pill_counter_total = 0;
  state->viruses_remaining = 0;
  state->speed_counter = 0;
  state->gravity_counter = 0;
  state->lock_counter = 0;
  state->speed_ups =
      (state->level > 0x14) ? static_cast<u8>(state->level - 0x14) : 0;
  if (state->speed_ups > SPEEDUPS_MAX)
    state->speed_ups = SPEEDUPS_MAX;
  state->hor_velocity = 0;
  state->spawn_delay = 35; // First pill also has throw animation delay

  state->rng_state[0] = 0x89;
  state->rng_state[1] = 0x88;

  if (state->speed_setting > 2)
    state->speed_setting = 1; // Default to medium
  if (start_mode == MODE_DEMO) {
    state->level = 0x0A;      // Demo level
    state->speed_setting = 1; // Medium
  }

  viruses_added = 0;
  viruses_to_place = 0;

  setupLevel();
  state->stage_clear = (state->viruses_remaining == 0) ? 1 : 0;
  generatePillsReserve();
  generateNextPill(); // First call - consumes demo_pills[0], sets preview
  generateNextPill(); // Second call - NES does this too (see level_init.asm
                      // line 125-126)
  spawnNewFallingPill();
}

void GameLogic::step() {
  update_buttons();

  if (state->stage_clear || state->level_fail) {
    return;
  }

  if (state->mode == MODE_PLAYING || state->mode == MODE_DEMO) {
    if (!state->stage_clear) {
      // During spawn delay (throw animation), inputs are consumed but pill
      // doesn't respond
      if (state->spawn_delay > 0) {
        state->spawn_delay--;
      } else {
        fallingPill_checkYMove();
        fallingPill_checkXMove();
        fallingPill_checkRotate();
      }
    }
  }

  state->frame_count++;
}

void GameLogic::update_buttons() {
  u8 current = state->buttons;
  state->buttons_pressed = current & static_cast<u8>(~state->buttons_prev);
  state->buttons_held = current;
  state->buttons_prev = current;
}

void GameLogic::generateNextPill() {
  u8 pillId = 0;
  if (state->mode == MODE_DEMO) {
    pillId = demo_pills[state->pill_counter % 45];
  } else {
    pillId = pillsReserve[state->pill_counter & 0x7F];
  }

  state->preview_pill_color_l = colorCombination_left[pillId];
  state->preview_pill_color_r = colorCombination_right[pillId];
  state->preview_pill_size = 2;

  state->pill_counter = (state->pill_counter + 1) & 0x7F;
  state->pill_counter_total++;

  if (state->speed_ups < SPEEDUPS_MAX &&
      (state->pill_counter_total % 10 == 0)) {
    state->speed_ups++;
  }
}

bool GameLogic::pillMoveValidation(int row, int col, int rot) {
  if (row < 0)
    return false; // Below the bottle
  if (col < 0 || col >= BOARD_WIDTH)
    return false;

  rot &= 0x03;
  const int8_t *row_offsets = half_row_offsets[rot];
  const int8_t *col_offsets = half_col_offsets[rot];

  for (int i = 0; i < 2; i++) {
    int target_row = row + row_offsets[i];
    int target_col = col + col_offsets[i];

    if (target_row < 0)
      return false;
    if (target_col < 0 || target_col >= BOARD_WIDTH)
      return false;

    // Above top of bottle is allowed (no collision)
    if (target_row >= BOARD_HEIGHT)
      continue;

    int idx = (LAST_ROW - target_row) * BOARD_WIDTH + target_col;
    if (state->board[idx] != TILE_EMPTY)
      return false;
  }

  return true;
}

void GameLogic::spawnNewFallingPill() {
  state->falling_pill_row = PILL_START_ROW;
  state->falling_pill_col = PILL_START_COL;
  state->falling_pill_orient = 0;
  state->falling_pill_color_l = state->preview_pill_color_l;
  state->falling_pill_color_r = state->preview_pill_color_r;
  state->falling_pill_size = 2;
  state->speed_counter = 0;
  state->hor_velocity = 0;
  state->spawn_delay = 35; // NES throw animation is ~35 frames

  // Top-out check: if initial placement invalid, mark fail.
  if (!pillMoveValidation(state->falling_pill_row, state->falling_pill_col,
                          state->falling_pill_orient)) {
    confirmPlacement();
    state->level_fail = 1;
    state->fail_count++;
    state->last_fail_frame = state->frame_count;
    state->last_fail_row = state->falling_pill_row;
    state->last_fail_col = state->falling_pill_col;
    return;
  }

  generateNextPill();
}

void GameLogic::fallingPill_checkYMove() {
  auto attempt_lower = [this]() {
    state->falling_pill_row = static_cast<u8>(state->falling_pill_row - 1);
    state->speed_counter = 0;

    if (!pillMoveValidation(state->falling_pill_row, state->falling_pill_col,
                            state->falling_pill_orient) ||
        state->falling_pill_row == 0xFF) {
      state->falling_pill_row++;
      confirmPlacement();
      if (!state->stage_clear) {
        spawnNewFallingPill();
      }
    }
  };

  // Fast drop: checked every 2 frames, only if down is the only dpad input.
  if ((state->frame_count & FAST_DROP_SPEED) != 0) {
    u8 dpad = state->buttons_held & BTNS_DPAD;
    if (dpad == BTN_DOWN) {
      attempt_lower();
      return;
    }
  }

  state->speed_counter++;
  int speed_setting_idx = std::clamp<int>(state->speed_setting, 0, 2);
  int table_idx = baseSpeedSettingValue[speed_setting_idx] +
                  static_cast<int>(state->speed_ups);
  int table_size = static_cast<int>(sizeof(speedCounterTable));
  if (table_idx >= table_size)
    table_idx = table_size - 1;
  u8 max_speed = speedCounterTable[table_idx];

  if (state->speed_counter <= max_speed)
    return;

  attempt_lower();
}

void GameLogic::fallingPill_checkXMove() {
  u8 pressed_lr = state->buttons_pressed & BTNS_LEFT_RIGHT;
  u8 held_lr = state->buttons_held & BTNS_LEFT_RIGHT;

  if (pressed_lr == 0) {
    if (held_lr == 0)
      return;
    state->hor_velocity++;
    if (state->hor_velocity < HOR_ACCEL_SPEED)
      return;
    state->hor_velocity = HOR_ACCEL_SPEED - HOR_MAX_SPEED;
  } else {
    state->hor_velocity = 0;
  }

  if (state->buttons_held & BTN_RIGHT) {
    int right_boundary =
        (state->falling_pill_orient & 0x01) ? LAST_COLUMN : LAST_COLUMN - 1;
    if (state->falling_pill_col != right_boundary) {
      u8 new_col = static_cast<u8>(state->falling_pill_col + 1);
      if (pillMoveValidation(state->falling_pill_row, new_col,
                             state->falling_pill_orient)) {
        state->falling_pill_col = new_col;
      } else {
        state->hor_velocity = HOR_ACCEL_SPEED - 1;
      }
    }
  }

  if (state->buttons_held & BTN_LEFT) {
    if (state->falling_pill_col > 0) {
      u8 new_col = static_cast<u8>(state->falling_pill_col - 1);
      if (pillMoveValidation(state->falling_pill_row, new_col,
                             state->falling_pill_orient)) {
        state->falling_pill_col = new_col;
      } else {
        state->hor_velocity = HOR_ACCEL_SPEED - 1;
      }
    }
  }
}

void GameLogic::fallingPill_checkRotate() {
  u8 rot_copy = state->falling_pill_orient;
  u8 col_copy = state->falling_pill_col;

  if (state->buttons_pressed & BTN_A) { // Clockwise (matches asm)
    state->falling_pill_orient =
        static_cast<u8>((state->falling_pill_orient + 3) & 0x03);
    pillRotateValidation(rot_copy, col_copy);
  }

  if (state->buttons_pressed & BTN_B) { // Counter-clockwise
    state->falling_pill_orient =
        static_cast<u8>((state->falling_pill_orient + 1) & 0x03);
    pillRotateValidation(rot_copy, col_copy);
  }
}

void GameLogic::pillRotateValidation(u8 prev_rot, u8 prev_col) {
  u8 rot_state = state->falling_pill_orient & 0x01;

  if (rot_state == 0) { // Rotation would be horizontal
    if (pillMoveValidation(state->falling_pill_row, state->falling_pill_col,
                           state->falling_pill_orient)) {
      if (state->buttons_held & BTN_LEFT) {
        if (state->falling_pill_col > 0) {
          u8 new_col = static_cast<u8>(state->falling_pill_col - 1);
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
      u8 kicked_col = static_cast<u8>(state->falling_pill_col - 1);
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

void GameLogic::action_pillFalling() {
  fallingPill_checkYMove();
  fallingPill_checkXMove();
  fallingPill_checkRotate();
}

void GameLogic::confirmPlacement() {
  int row = state->falling_pill_row;
  int col = state->falling_pill_col;
  int rot = state->falling_pill_orient & 0x03;

  const int8_t *row_offsets = half_row_offsets[rot];
  const int8_t *col_offsets = half_col_offsets[rot];
  const u8 *tiles = half_tile_shapes[rot];

  u8 colors[2] = {static_cast<u8>(state->falling_pill_color_l & MASK_COLOR),
                  static_cast<u8>(state->falling_pill_color_r & MASK_COLOR)};

  for (int i = 0; i < 2; i++) {
    int r = row + row_offsets[i];
    int c = col + col_offsets[i];

    if (r < 0 || c < 0 || c >= BOARD_WIDTH)
      continue;
    if (r >= BOARD_HEIGHT)
      continue; // Above the top disappears (matches asm)

    int idx = (LAST_ROW - r) * BOARD_WIDTH + c;
    state->board[idx] = tiles[i] | colors[i];
  }

  checkMatches();
}

void GameLogic::checkMatches() {
  bool cleared = false;
  bool to_clear[BOARD_SIZE] = {false};

  // Horizontal
  for (int r = 0; r < 16; r++) {
    for (int c = 0; c < 8; c++) {
      int idx = (15 - r) * 8 + c;
      u8 tile = state->board[idx];
      if (tile == TILE_EMPTY)
        continue;

      int match_len = 1;
      u8 color = tile & MASK_COLOR;

      for (int k = c + 1; k < 8; k++) {
        int k_idx = (15 - r) * 8 + k;
        u8 k_tile = state->board[k_idx];
        if (k_tile == TILE_EMPTY || (k_tile & MASK_COLOR) != color)
          break;
        match_len++;
      }

      if (match_len >= 4) {
        for (int k = 0; k < match_len; k++) {
          to_clear[(15 - r) * 8 + (c + k)] = true;
        }
      }
    }
  }

  // Vertical
  for (int c = 0; c < 8; c++) {
    for (int r = 0; r < 16; r++) {
      int idx = (15 - r) * 8 + c;
      u8 tile = state->board[idx];
      if (tile == TILE_EMPTY)
        continue;

      int match_len = 1;
      u8 color = tile & MASK_COLOR;

      for (int k = r + 1; k < 16; k++) { // Going UP
        int k_idx = (15 - k) * 8 + c;
        u8 k_tile = state->board[k_idx];
        if (k_tile == TILE_EMPTY || (k_tile & MASK_COLOR) != color)
          break;
        match_len++;
      }

      if (match_len >= 4) {
        for (int k = 0; k < match_len; k++) {
          to_clear[(15 - (r + k)) * 8 + c] = true;
        }
      }
    }
  }

  // Apply Clear
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (to_clear[i]) {
      u8 tile = state->board[i];
      if ((tile & MASK_TYPE) == TILE_VIRUS_START &&
          state->viruses_remaining > 0)
        state->viruses_remaining--;
      state->board[i] = TILE_EMPTY;
      cleared = true;
      // Update halves to singles if needed?
      // If we clear one half, the other becomes single.
      // This requires checking neighbors of cleared tiles BEFORE clearing?
      // Or after?
      // If I clear a Left, the Right neighbor (if exists) becomes Single.
    }
  }

  if (cleared) {
    // Fix broken halves
    for (int i = 0; i < BOARD_SIZE; i++) {
      if (state->board[i] == TILE_EMPTY)
        continue;
      u8 type = state->board[i] & MASK_TYPE;
      int r = 15 - (i / 8);
      int c = i % 8;

      if (type == TILE_LEFT) {
        if (c + 1 >= 8 || state->board[i + 1] == TILE_EMPTY)
          state->board[i] = TILE_SINGLE | (state->board[i] & MASK_COLOR);
      } else if (type == TILE_RIGHT) {
        if (c - 1 < 0 || state->board[i - 1] == TILE_EMPTY)
          state->board[i] = TILE_SINGLE | (state->board[i] & MASK_COLOR);
      } else if (type == TILE_TOP) {
        // Top is supported by Bottom (below, r-1).
        // Wait, Top is usually ABOVE Bottom.
        // In my coords (0=Bottom), Top is at r, Bottom is at r-1.
        // So check r-1.
        int below = (15 - (r - 1)) * 8 + c;
        if (r - 1 < 0 || state->board[below] == TILE_EMPTY)
          state->board[i] = TILE_SINGLE | (state->board[i] & MASK_COLOR);
      } else if (type == TILE_BOTTOM) {
        // Bottom supports Top (above, r+1).
        int above = (15 - (r + 1)) * 8 + c;
        if (r + 1 > 15 || state->board[above] == TILE_EMPTY)
          state->board[i] = TILE_SINGLE | (state->board[i] & MASK_COLOR);
      }
    }

    checkDrop();
  }

  state->stage_clear = (state->viruses_remaining == 0) ? 1 : 0;
}

void GameLogic::checkDrop() {
  bool moved_any = true;
  bool moved_once = false;
  while (moved_any) {
    moved_any = false;

    for (int r = 1; r < BOARD_HEIGHT; r++) { // bottom row cannot fall
      for (int c = 0; c < BOARD_WIDTH; c++) {
        int idx = (LAST_ROW - r) * BOARD_WIDTH + c;
        u8 tile = state->board[idx];
        if (tile == TILE_EMPTY)
          continue;

        u8 type = tile & MASK_TYPE;
        if (type == TILE_VIRUS_START)
          continue;

        if (type == TILE_RIGHT || type == TILE_TOP) {
          continue; // handled by their counterparts
        }

        if (type == TILE_LEFT) {
          int right_idx = idx + 1;
          int below_left = idx + BOARD_WIDTH;
          int below_right = right_idx + BOARD_WIDTH;
          if (below_left >= BOARD_SIZE || below_right >= BOARD_SIZE)
            continue;
          if (state->board[below_left] != TILE_EMPTY ||
              state->board[below_right] != TILE_EMPTY)
            continue;

          state->board[below_left] = state->board[idx];
          state->board[below_right] = state->board[right_idx];
          state->board[idx] = TILE_EMPTY;
          state->board[right_idx] = TILE_EMPTY;
          moved_any = true;
          moved_once = true;
        } else if (type == TILE_BOTTOM) {
          int above_idx = idx - BOARD_WIDTH;
          int below_idx = idx + BOARD_WIDTH;
          if (below_idx >= BOARD_SIZE)
            continue;
          if (state->board[below_idx] != TILE_EMPTY)
            continue;

          state->board[below_idx] = state->board[idx];
          state->board[idx] = state->board[above_idx];
          state->board[above_idx] = TILE_EMPTY;
          moved_any = true;
          moved_once = true;
        } else { // TILE_SINGLE
          int below_idx = idx + BOARD_WIDTH;
          if (below_idx >= BOARD_SIZE)
            continue;
          if (state->board[below_idx] != TILE_EMPTY)
            continue;
          state->board[below_idx] = tile;
          state->board[idx] = TILE_EMPTY;
          moved_any = true;
          moved_once = true;
        }
      }
    }
  }

  if (moved_once) {
    // After all drops, re-check matches
    checkMatches();
  }
}

void GameLogic::setupLevel() {
  // Clear board
  for (int i = 0; i < 128; i++) {
    state->board[i] = TILE_EMPTY;
  }

  if (state->mode == MODE_DEMO) {
    // Copy demo field
    for (int i = 0; i < 128; i++) {
      state->board[i] = demo_field[i];
    }
    state->viruses_remaining = 0;
    for (int i = 0; i < 128; i++) {
      if ((state->board[i] & MASK_TYPE) == TILE_VIRUS_START)
        state->viruses_remaining++;
    }
    return;
  }

  // Normal Level Setup
  // Calculate virus count: (level + 1) * 4
  int level = state->level;
  if (level > 20)
    level = 20; // Cap at 20 for calculation
  viruses_to_place = (level + 1) * 4;
  state->viruses_remaining = viruses_to_place;
  viruses_added = 0;

  int attempts = 0;
  int max_attempts = std::max(viruses_to_place * 64, 256);
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
  u8 r1 = rng_get();
  int virusHeight = r1 & 0xF; // 0 (bottom) .. 15 (top)
  if (virusHeight > addVirus_maxHeight_basedOnLvl[level])
    return false;

  rng_step();
  u8 r2 = rng_get();
  int col = r2 & 0x7; // 0..7

  if (viruses_added >= viruses_to_place)
    return false;

  int base = pill_fieldPos_relativeToPillY[virusHeight];
  int virusPos = base + col;
  if (virusPos < 0 || virusPos >= BOARD_SIZE)
    return false;
  if (state->board[virusPos] != TILE_EMPTY)
    return false;

  int color_cycle = viruses_added & 0x3;
  u8 virusColor = 0;
  if (color_cycle == 3) {
    rng_step();
    virusColor = virusColor_random[rng_get() & 0x0F];
  } else {
    virusColor = static_cast<u8>(color_cycle);
  }

  auto has_same_color = [&](int row, int column) -> bool {
    u8 tile = get_board_tile(row, column);
    if (tile == TILE_EMPTY)
      return false;
    return (tile & MASK_COLOR) == virusColor;
  };

  // Reject placement if same color within 2 cells horizontally/vertically.
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

  state->board[virusPos] = TILE_VIRUS_START | (virusColor & MASK_COLOR);
  viruses_added++;
  return true;
}

void GameLogic::generatePillsReserve() {
  int pillsToGenerate = 128;
  int pillId = 0;
  for (int i = 0; i < pillsToGenerate; i++) {
    rng_step();
    u8 r = rng_get();
    pillId = (pillId + (r & 0xF));
    while (pillId >= 9)
      pillId -= 9; // 9 color combos
    pillsReserve[i] = pillId;
  }
}

u8 GameLogic::rng_get() { return state->rng_state[0]; }

void GameLogic::rng_step() {
  // EOR bit1 of rng0/rng1 → carry, then ROR across both bytes.
  u8 carry = ((state->rng_state[0] ^ state->rng_state[1]) & 0x02) ? 1 : 0;

  for (int i = 0; i < 2; i++) {
    u8 val = state->rng_state[i];
    u8 new_val = (val >> 1) | (carry << 7);
    carry = val & 0x01; // Carry out for next byte
    state->rng_state[i] = new_val;
  }
}

u8 GameLogic::get_board_tile(int row, int col) {
  if (row < 0 || row > 15 || col < 0 || col > 7)
    return TILE_EMPTY; // Boundary treated as empty
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
