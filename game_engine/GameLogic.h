#pragma once
#include "GameState.h"
#include <cstddef>

class GameLogic {
public:
  GameLogic(DrMarioState *state);
  void reset();
  void step();

private:
  DrMarioState *state;

  // === High-level emulation model ===
  //
  // The NES game is split into two halves:
  //   1) NMI-time work (frameCounter++, input sampling / demo input replay,
  //      rendering one bottle row, etc.)
  //   2) Main-thread work (the per-frame gameplay state machine: nextAction,
  //      pillPlacedStep, sendPill throw animation, etc.)
  //
  // For parity we simulate the NMI effects explicitly at the start of each
  // step(), then run one main-loop action.

  enum class NextAction : u8 {
    PillFalling = 0x00,
    PillPlaced = 0x01,
    CheckAttack = 0x02,
    SendPillFinished = 0x03,
    DoNothingUnused = 0x04,
    IncNextAction = 0x05,
    SendPill = 0x06,
  };

  // Internal helpers matching assembly routines
  void nmi_tick();
  void update_buttons_from_raw(u8 raw_buttons);
  void update_buttons_demo();
  void tick_status_row_render();

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

  void action_pillPlaced();
  void toPillPlacedStep();
  void pillPlaced_resetCombo();
  void pillPlaced_incStep();
  void pillPlaced_checkDrop();
  void pillPlaced_resetMatchFlag();
  void pillPlaced_checkHorMatch();
  void pillPlaced_checkVerMatch();
  void pillPlaced_updateField();
  void pillPlaced_resetPillPlacedStep();

  void action_checkAttack();
  void action_sendPillFinished();
  void action_incNextAction();
  void action_sendPill();

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

  // === NES-private state (not exposed via shared memory) ===

  // Per-frame gameplay dispatcher (p1_nextAction in NES RAM, 0..6).
  NextAction next_action_ = NextAction::PillFalling;

  // sendPill animation state (pillThrownFrame + nextPillRotation in NES).
  u8 pill_thrown_frame_ = 0;
  u8 next_pill_rotation_ = 0;

  // pillPlaced micro-step state and match bookkeeping.
  // Note: we expose pillPlacedStep via state->lock_counter for parity with the
  // NES RAM map ($0307), but keep additional bookkeeping here.
  u8 match_flag_ = 0;
  u8 score_multiplier_ = 0;
  u8 combo_counter_ = 0;
  u8 chain_length_0based_ = 0;
  u8 chain_length_ = 0;

  // Field row render countdown (currentP_status / p1_status in NES).
  // Decremented by NMI-time bottle row rendering; checkDrop is gated on 0xFF.
  u8 status_row_ = 0xFF;

  // Level intro delay (waitFor_A_frames(levelIntro_delay)).
  u16 intro_delay_frames_ = 0;

  // Demo input replay state (getInputs_checkMode).
  size_t demo_instr_ptr_ = 0;     // byte offset into demo instruction stream
  u8 demo_inputs_held_ = 0;       // "demo_inputs" in disassembly
  u8 demo_counter_ = 0;           // counterDemoInstruction
};
