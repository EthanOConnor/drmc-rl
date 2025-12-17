#pragma once
#include "GameState.h"

class GameLogic {
public:
  GameLogic(DrMarioState *state);
  void reset();
  void step();

private:
  DrMarioState *state;

  // Internal helpers matching assembly routines
  void update_buttons();
  void generateNextPill();
  bool pillMoveValidation(int row, int col, int rot);
  void confirmPlacement();
  u8 rng_get();
  void rng_step();

  // Helper for board access
  u8 get_board_tile(int row, int col);
  void set_board_tile(int row, int col, u8 val);

  // Internal state for RNG if not in shared mem (though it is in shared mem)
  // We use the shared mem one.

  // Assembly routine equivalents (to be implemented/integrated)
  void action_pillFalling();
  void fallingPill_checkYMove();
  void fallingPill_checkXMove();
  void fallingPill_checkRotate();
  void pillRotateValidation(u8 prev_rot, u8 prev_col);
  void checkDrop();
  void checkMatches(); // Horizontal and Vertical
  void spawnNewFallingPill();

  // Level Setup
  void setupLevel();
  void generatePillsReserve();
  bool addVirus();

  // Utility
  bool is_free(int row, int col);

  // Internal Data
  u8 pillsReserve[128];
  int viruses_added;
  int viruses_to_place;
};
