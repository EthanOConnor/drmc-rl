#pragma once

#include "GameLogic.h"
#include "drmario_pool_capi.h"

#include <array>
#include <cstdint>
#include <vector>

class DrMarioPool {
public:
  explicit DrMarioPool(const DrmPoolConfig& cfg);
  ~DrMarioPool() = default;

  DrMarioPool(const DrMarioPool&) = delete;
  DrMarioPool& operator=(const DrMarioPool&) = delete;

  int reset(const uint8_t* reset_mask, const DrmResetSpec* reset_specs, DrmPoolOutputs* out);
  int step(const int32_t* actions, const uint8_t* reset_mask, const DrmResetSpec* reset_specs,
           DrmPoolOutputs* out);

  uint32_t num_envs() const { return num_envs_; }
  uint32_t obs_channels() const { return obs_channels_; }

private:
  struct PlannerCache {
    bool valid = false;
    uint8_t spawn_id = 0xFF;
    uint32_t board_hash = 0;
    // Native BFS outputs indexed by base pose (x,y,rot) pose_index = x + 8*y + 128*rot.
    std::array<uint16_t, 512> costs_u16{};
    std::array<uint16_t, 512> offsets_u16{};
    std::array<uint16_t, 512> lengths_u16{};
    std::vector<uint8_t> script_buf;
    int script_used = 0;
    // Macro-action indexed outputs (flattened 512).
    std::array<uint8_t, 512> feasible{};
    std::array<uint16_t, 512> cost_to_lock{};
    uint32_t feasible_count = 0;
  };

  struct ClearEdgeState {
    bool prev_clearing_active = false;
  };

  uint32_t num_envs_ = 0;
  uint32_t obs_spec_ = DRM_POOL_OBS_NONE;
  uint32_t obs_channels_ = 0;
  uint32_t max_lock_frames_ = 2048;
  uint32_t max_wait_frames_ = 6000;

  std::vector<DrMarioState> states_;
  std::vector<GameLogic> games_;
  std::vector<PlannerCache> planner_;
  std::vector<ClearEdgeState> clear_edge_;

  // --- helpers
  static uint8_t canonical_color_index(uint8_t raw_color_bits);
  static uint32_t hash_board_u32(const uint8_t* board);
  static uint16_t count_viruses_u16(const uint8_t* board);
  static uint8_t to_bcd_u8(uint16_t value);

  static int macro_action_from_base(int x, int y, int rot);
  static bool base_pose_for_macro_action(int action, int& out_x, int& out_y, int& out_rot);

  static uint8_t buttons_mask_from_reach_action(uint8_t reach_action_index);
  static int compute_speed_threshold(int speed_setting, int speed_ups);
  static void build_cols_u16(const uint8_t* board, uint16_t out_cols[8]);

  void invalidate_planner(uint32_t env_i);
  int ensure_planner(uint32_t env_i);

  void update_decision_outputs(uint32_t env_i, DrmPoolOutputs* out);

  void apply_reset_spec(uint32_t env_i, const DrmResetSpec& spec);
  void apply_synthetic_virus_target(uint32_t env_i, const DrmResetSpec& spec);

  // Runs with NOOP buttons until the next actionable decision (feasible_count>0),
  // terminal, or timeout. Returns 0 on success, and writes flags.
  int run_until_actionable_decision(uint32_t env_i, uint8_t last_spawn_id, uint32_t& tau_frames,
                                    uint16_t& match_events, bool& terminated, bool& truncated,
                                    uint8_t& terminal_reason);

  void note_clear_edge(uint32_t env_i, uint16_t& match_events);

  void compute_adjacency_flags(const uint8_t* board_prev, const uint8_t* board_lock,
                               uint8_t out_adj_pair[3], uint8_t out_adj_triplet[3],
                               uint8_t out_v_pair[3], uint8_t out_v_triplet[3]);

  void build_obs(uint32_t env_i, float* out_obs_ptr);
};
