#include "DrMarioPool.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {
constexpr uint8_t TILE_EMPTY = 0xFF;
constexpr uint8_t TILE_CLEARED = 0xB0;
constexpr uint8_t TILE_JUST_EMPTIED = 0xF0;
constexpr uint8_t TILE_VIRUS = 0xD0;

constexpr uint8_t MASK_TYPE = 0xF0;
constexpr uint8_t MASK_COLOR = 0x03;

constexpr uint8_t BTN_RIGHT = 0x01;
constexpr uint8_t BTN_LEFT = 0x02;
constexpr uint8_t BTN_DOWN = 0x04;
constexpr uint8_t BTN_B = 0x40;
constexpr uint8_t BTN_A = 0x80;

constexpr int GRID_W = 8;
constexpr int GRID_H = 16;

extern "C" void drm_reach_free_thread_ctx(void);


uint32_t parse_worker_override() {
  const char* env = std::getenv("DRMARIO_POOL_WORKERS");
  if (!env || env[0] == '\0')
    return 0u;
  char* end = nullptr;
  const long raw = std::strtol(env, &end, 10);
  if (end == env)
    return 0u;
  if (raw <= 0)
    return 1u;
  return static_cast<uint32_t>(raw);
}

template <typename Fn>
void parallel_for_envs(uint32_t n, uint32_t workers, Fn&& fn) {
  if (n == 0)
    return;
  if (workers <= 1 || n <= 1) {
    for (uint32_t i = 0; i < n; ++i) {
      fn(i);
    }
    return;
  }
  if (workers > n)
    workers = n;

  std::atomic<uint32_t> next{0u};
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers - 1));

  auto worker_loop = [&]() {
    while (true) {
      const uint32_t i = next.fetch_add(1u, std::memory_order_relaxed);
      if (i >= n)
        break;
      fn(i);
    }
  };
  auto worker_thread = [&]() {
    worker_loop();
    drm_reach_free_thread_ctx();
  };

  for (uint32_t t = 1; t < workers; ++t) {
    threads.emplace_back(worker_thread);
  }
  worker_loop();
  for (auto& th : threads) {
    th.join();
  }
}

// Native BFS helper (reach_native/drm_reach_full.c).
extern "C" int drm_reach_bfs_full(
    const uint16_t cols[8],
    int sx,
    int sy,
    int srot,
    int sc,
    int hv,
    int hold_dir,
    int parity,
    int rot_hold,
    int speed_threshold,
    int max_frames,
    uint16_t out_costs[512],
    uint16_t out_offsets[512],
    uint16_t out_lengths[512],
    uint8_t* out_script_buf,
    int script_buf_cap,
    int* out_script_used);

inline bool is_terminal(const DrMarioState& s) { return (s.stage_clear != 0) || (s.level_fail != 0); }

inline bool is_decision_point(const DrMarioState& s, uint8_t last_spawn_id) {
  if (s.mode != MODE_PLAYING)
    return false;
  if (s.next_action != 0) // NextAction::PillFalling
    return false;
  if (last_spawn_id == 0xFF)
    return true;
  return s.pill_counter != last_spawn_id;
}

inline int pose_index(int x, int y, int rot) { return (rot & 3) * (GRID_H * GRID_W) + (y * GRID_W) + x; }

// Planner geometry offsets: same as placement_planner.ROT_OFFSETS.
constexpr int8_t ROT_OFFSETS[4][2][2] = {
    {{0, 0}, {0, 1}},   // rot 0: first at base, second right
    {{0, 0}, {-1, 0}},  // rot 1: first at base, second above
    {{0, 1}, {0, 0}},   // rot 2: first right, second at base
    {{-1, 0}, {0, 0}},  // rot 3: first above, second at base
};

// ORIENT_OFFSETS: o=0 (0,1), o=1 (1,0), o=2 (0,-1), o=3 (-1,0)
inline int orient_index(int dr, int dc) {
  if (dr == 0 && dc == 1)
    return 0;
  if (dr == 1 && dc == 0)
    return 1;
  if (dr == 0 && dc == -1)
    return 2;
  if (dr == -1 && dc == 0)
    return 3;
  return -1;
}

// Small deterministic PRNG for synthetic virus selection (xorshift32).
inline uint32_t xorshift32(uint32_t& s) {
  uint32_t x = s ? s : 0xA5A5A5A5u;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  s = x;
  return x;
}

inline int rand_bounded(uint32_t& seed, int bound) {
  if (bound <= 0)
    return 0;
  // Rejection sampling to avoid modulo bias (bound is tiny here).
  const uint32_t lim = UINT32_MAX - (UINT32_MAX % static_cast<uint32_t>(bound));
  uint32_t r = 0;
  do {
    r = xorshift32(seed);
  } while (r >= lim);
  return static_cast<int>(r % static_cast<uint32_t>(bound));
}

} // namespace

DrMarioPool::DrMarioPool(const DrmPoolConfig& cfg) {
  num_envs_ = std::max<uint32_t>(1u, cfg.num_envs);
  obs_spec_ = cfg.obs_spec;
  max_lock_frames_ = (cfg.max_lock_frames > 0) ? cfg.max_lock_frames : 2048;
  max_wait_frames_ = (cfg.max_wait_frames > 0) ? cfg.max_wait_frames : 6000;
  uint32_t workers = parse_worker_override();
  if (workers == 0u) {
    workers = std::thread::hardware_concurrency();
    if (workers == 0u)
      workers = 1u;
  }
  if (workers > num_envs_)
    workers = num_envs_;
  worker_count_ = std::max<uint32_t>(1u, workers);

  if (obs_spec_ == DRM_POOL_OBS_BITPLANE_BOTTLE) {
    obs_channels_ = 4;
  } else if (obs_spec_ == DRM_POOL_OBS_BITPLANE_BOTTLE_MASK) {
    obs_channels_ = 8;
  } else {
    obs_channels_ = 0;
  }

  states_.resize(num_envs_);
  games_.reserve(num_envs_);
  for (uint32_t i = 0; i < num_envs_; ++i) {
    std::memset(&states_[i], 0, sizeof(DrMarioState));
    // Default to playing mode; reset() will run intro delay and set mode=0x04.
    states_[i].mode = MODE_PLAYING;
    games_.emplace_back(&states_[i]);
  }

  planner_.resize(num_envs_);
  clear_edge_.resize(num_envs_);

  // Allocate per-env planner script buffers once.
  const int script_cap = static_cast<int>(512u * max_lock_frames_);
  for (uint32_t i = 0; i < num_envs_; ++i) {
    planner_[i].script_buf.resize(script_cap);
    planner_[i].script_used = 0;
    planner_[i].valid = false;
    planner_[i].spawn_id = 0xFF;
    planner_[i].board_hash = 0;
  }
}

// --------------------------------------------------------------------------- helpers

uint8_t DrMarioPool::canonical_color_index(uint8_t raw_color_bits) {
  // envs/specs/ram_to_state.py: COLOR_VALUE_TO_INDEX = {1:0, 0:1, 2:2}
  const uint8_t v = raw_color_bits & 0x03u;
  if (v == 1u)
    return 0u; // red
  if (v == 0u)
    return 1u; // yellow
  if (v == 2u)
    return 2u; // blue
  return 0u;
}

uint32_t DrMarioPool::hash_board_u32(const uint8_t* board) {
  // FNV-1a 32-bit over 128 bytes.
  uint32_t h = 2166136261u;
  for (int i = 0; i < 128; ++i) {
    h ^= static_cast<uint32_t>(board[i]);
    h *= 16777619u;
  }
  return h;
}

uint16_t DrMarioPool::count_viruses_u16(const uint8_t* board) {
  uint16_t count = 0;
  for (int i = 0; i < 128; ++i) {
    if ((board[i] & MASK_TYPE) == TILE_VIRUS)
      count++;
  }
  return count;
}

uint8_t DrMarioPool::to_bcd_u8(uint16_t value) {
  const uint16_t v = value;
  const uint8_t tens = static_cast<uint8_t>((v / 10u) & 0x0Fu);
  const uint8_t ones = static_cast<uint8_t>((v % 10u) & 0x0Fu);
  return static_cast<uint8_t>((tens << 4) | ones);
}

int DrMarioPool::macro_action_from_base(int x, int y, int rot) {
  const int rot_i = rot & 3;
  const int r1 = y + ROT_OFFSETS[rot_i][0][0];
  const int c1 = x + ROT_OFFSETS[rot_i][0][1];
  const int r2 = y + ROT_OFFSETS[rot_i][1][0];
  const int c2 = x + ROT_OFFSETS[rot_i][1][1];
  if (r1 < 0 || r1 >= GRID_H || c1 < 0 || c1 >= GRID_W)
    return -1;
  if (r2 < 0 || r2 >= GRID_H || c2 < 0 || c2 >= GRID_W)
    return -1;
  const int dr = r2 - r1;
  const int dc = c2 - c1;
  const int o = orient_index(dr, dc);
  if (o < 0)
    return -1;
  return o * (GRID_H * GRID_W) + r1 * GRID_W + c1;
}

bool DrMarioPool::base_pose_for_macro_action(int action, int& out_x, int& out_y, int& out_rot) {
  const int a = action;
  if (a < 0 || a >= 512)
    return false;
  const int o = a / (GRID_H * GRID_W);
  const int rem = a % (GRID_H * GRID_W);
  const int row = rem / GRID_W;
  const int col = rem % GRID_W;
  if (o == 0) {
    // H+: partner at (row, col+1)
    if (col + 1 >= GRID_W)
      return false;
    out_x = col;
    out_y = row;
    out_rot = 0;
    return true;
  }
  if (o == 2) {
    // H-: partner at (row, col-1) -> base at left cell
    if (col - 1 < 0)
      return false;
    out_x = col - 1;
    out_y = row;
    out_rot = 2;
    return true;
  }
  if (o == 3) {
    // V-: partner at (row-1, col) -> base is bottom cell at (row, col)
    if (row - 1 < 0)
      return false;
    out_x = col;
    out_y = row;
    out_rot = 1;
    return true;
  }
  if (o == 1) {
    // V+: partner at (row+1, col) -> base is bottom cell at (row+1, col)
    if (row + 1 >= GRID_H)
      return false;
    out_x = col;
    out_y = row + 1;
    out_rot = 3;
    return true;
  }
  return false;
}

uint8_t DrMarioPool::buttons_mask_from_reach_action(uint8_t reach_action_index) {
  // Action encoding matches reach_native: see ACT_* tables in drm_reach_full.c.
  // index in 0..17:
  //   hold_dir: 0..2 (neutral,left,right) in blocks of 6
  //   hold_down: 0/1 (in blocks of 3)
  //   rot: 0 none, 1 cw(A), 2 ccw(B) (stride 1)
  const int idx = static_cast<int>(reach_action_index);
  const int clamped = (idx < 0) ? 0 : (idx > 17 ? 17 : idx);
  const int hold_dir = clamped / 6;      // 0..2
  const int sub = clamped % 6;           // 0..5
  const int hold_down = (sub >= 3) ? 1 : 0;
  const int rot = sub % 3; // 0..2

  uint8_t mask = 0;
  if (hold_dir == 1)
    mask |= BTN_LEFT;
  else if (hold_dir == 2)
    mask |= BTN_RIGHT;
  if (hold_down != 0)
    mask |= BTN_DOWN;
  if (rot == 1)
    mask |= BTN_A;
  else if (rot == 2)
    mask |= BTN_B;
  return mask;
}

int DrMarioPool::compute_speed_threshold(int speed_setting, int speed_ups) {
  // Matches envs/retro/fast_reach.py + GameLogic.cpp `fallingPill_checkYMove`.
  static constexpr uint8_t baseSpeedSettingValue[3] = {0x0F, 0x19, 0x1F};
  static constexpr uint8_t speedCounterTable[] = {
      // Source: dr-mario-disassembly/data/drmario_data_game.asm (NTSC / !ver_EU).
      0x45, 0x43, 0x41, 0x3F, 0x3D, 0x3B, 0x39, 0x37, 0x35, 0x33, 0x31, 0x2F,
      0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21, 0x1F, 0x1D, 0x1B, 0x19, 0x17,
      0x15, 0x13, 0x12, 0x11, 0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09,
      0x09, 0x08, 0x08, 0x07, 0x07, 0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05,
      0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x04, 0x04, 0x04, 0x04, 0x04,
      0x03, 0x03, 0x03, 0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x01, 0x01,
      0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
  };
  const int setting_idx = std::clamp(speed_setting, 0, 2);
  const int base_index = static_cast<int>(baseSpeedSettingValue[setting_idx]);
  const int raw_index = base_index + std::max(0, speed_ups);
  const int table_max = static_cast<int>(sizeof(speedCounterTable)) - 1;
  const int idx = std::clamp(raw_index, 0, table_max);
  return static_cast<int>(speedCounterTable[idx]);
}

void DrMarioPool::build_cols_u16(const uint8_t* board, uint16_t out_cols[8]) {
  for (int c = 0; c < GRID_W; ++c) {
    uint16_t bits = 0;
    for (int r = 0; r < GRID_H; ++r) {
      const uint8_t tile = board[r * GRID_W + c];
      // Treat 0x00 and >=0xF0 (empty/just-emptied) as empty for collision masks.
      if (tile != 0x00u && tile < TILE_JUST_EMPTIED) {
        bits |= static_cast<uint16_t>(1u << static_cast<unsigned>(r));
      }
    }
    out_cols[c] = bits;
  }
}

void DrMarioPool::invalidate_planner(uint32_t env_i) {
  if (env_i >= num_envs_)
    return;
  planner_[env_i].valid = false;
  planner_[env_i].spawn_id = 0xFF;
  planner_[env_i].board_hash = 0;
  planner_[env_i].feasible_count = 0;
}

int DrMarioPool::ensure_planner(uint32_t env_i) {
  if (env_i >= num_envs_)
    return -1;
  DrMarioState& s = states_[env_i];
  PlannerCache& cache = planner_[env_i];

  if (s.mode != MODE_PLAYING || s.next_action != 0) {
    cache.valid = false;
    return -2;
  }

  const uint8_t spawn_id = s.pill_counter;
  const uint32_t board_hash = hash_board_u32(s.board);

  if (cache.valid && cache.spawn_id == spawn_id && cache.board_hash == board_hash) {
    return 0;
  }

  uint16_t cols[8];
  build_cols_u16(s.board, cols);

  const int base_col = static_cast<int>(s.falling_pill_col);
  const int row_from_bottom = static_cast<int>(s.falling_pill_row);
  const int base_row_top = (GRID_H - 1) - row_from_bottom;
  const int rot = static_cast<int>(s.falling_pill_orient) & 3;
  const int speed_counter = static_cast<int>(s.speed_counter);
  const int hv = static_cast<int>(s.hor_velocity) & 0x0F;

  // Hold dir / rotation hold from held buttons.
  const uint8_t held = s.buttons_held;
  int hold_dir = 0;
  if ((held & BTN_LEFT) && !(held & BTN_RIGHT))
    hold_dir = 1;
  else if ((held & BTN_RIGHT) && !(held & BTN_LEFT))
    hold_dir = 2;

  int rot_hold = 0;
  const bool hold_a = (held & BTN_A) != 0;
  const bool hold_b = (held & BTN_B) != 0;
  if (hold_a && !hold_b)
    rot_hold = 1;
  else if (hold_b && !hold_a)
    rot_hold = 2;

  const int parity = static_cast<int>(s.frame_count & 1u);
  const int speed_threshold =
      compute_speed_threshold(static_cast<int>(s.speed_setting), static_cast<int>(s.speed_ups));

  int used = 0;
  const int rc = drm_reach_bfs_full(cols, base_col, base_row_top, rot, speed_counter, hv, hold_dir,
                                   parity, rot_hold, speed_threshold,
                                   static_cast<int>(max_lock_frames_), cache.costs_u16.data(),
                                   cache.offsets_u16.data(), cache.lengths_u16.data(),
                                   cache.script_buf.data(), static_cast<int>(cache.script_buf.size()), &used);
  if (rc != 0) {
    cache.valid = false;
    cache.feasible_count = 0;
    return rc;
  }
  cache.script_used = std::max(0, std::min(used, static_cast<int>(cache.script_buf.size())));

  cache.feasible.fill(0u);
  cache.cost_to_lock.fill(0xFFFFu);
  cache.feasible_count = 0;

  for (int pose = 0; pose < 512; ++pose) {
    const uint16_t cost = cache.costs_u16[pose];
    if (cost == 0xFFFFu)
      continue;
    const int x = pose & 7;
    const int y = (pose >> 3) & 15;
    const int r = (pose >> 7) & 3;
    const int action = macro_action_from_base(x, y, r);
    if (action < 0 || action >= 512)
      continue;
    if (cache.feasible[static_cast<size_t>(action)] == 0u) {
      cache.feasible_count += 1;
    }
    cache.feasible[static_cast<size_t>(action)] = 1u;
    cache.cost_to_lock[static_cast<size_t>(action)] = cost;
  }

  cache.valid = true;
  cache.spawn_id = spawn_id;
  cache.board_hash = board_hash;
  return 0;
}

void DrMarioPool::note_clear_edge(uint32_t env_i, uint16_t& match_events) {
  DrMarioState& s = states_[env_i];
  ClearEdgeState& st = clear_edge_[env_i];

  if (s.mode != MODE_PLAYING)
    return;

  int tiles_clearing = 0;
  for (int i = 0; i < 128; ++i) {
    const uint8_t tile = s.board[i];
    const uint8_t hi = tile & MASK_TYPE;
    if (hi == TILE_CLEARED) {
      tiles_clearing += 1;
      continue;
    }
    if (hi == TILE_JUST_EMPTIED && tile != TILE_EMPTY) {
      tiles_clearing += 1;
      continue;
    }
  }

  const bool clearing_active = tiles_clearing >= 4;
  if (clearing_active && !st.prev_clearing_active) {
    match_events = static_cast<uint16_t>(match_events + 1);
  }
  st.prev_clearing_active = clearing_active;
}

void DrMarioPool::compute_adjacency_flags(const uint8_t* board_prev, const uint8_t* board_lock,
                                         uint8_t out_adj_pair[3], uint8_t out_adj_triplet[3],
                                         uint8_t out_v_pair[3], uint8_t out_v_triplet[3]) {
  std::memset(out_adj_pair, 0, 3);
  std::memset(out_adj_triplet, 0, 3);
  std::memset(out_v_pair, 0, 3);
  std::memset(out_v_triplet, 0, 3);

  // Build masks for prev/next:
  //  - static pills by color (exclude viruses)
  //  - virus masks by color
  bool static_prev[3][GRID_H][GRID_W]{};
  bool static_next[3][GRID_H][GRID_W]{};
  bool virus_prev[3][GRID_H][GRID_W]{};
  bool virus_next[3][GRID_H][GRID_W]{};
  bool new_cells[3][GRID_H][GRID_W]{};

  for (int r = 0; r < GRID_H; ++r) {
    for (int c = 0; c < GRID_W; ++c) {
      const int idx = r * GRID_W + c;
      const uint8_t a = board_prev[idx];
      const uint8_t b = board_lock[idx];

      const uint8_t a_type = a & MASK_TYPE;
      const uint8_t b_type = b & MASK_TYPE;
      const uint8_t a_col = a & MASK_COLOR;
      const uint8_t b_col = b & MASK_COLOR;

      if (a_type == TILE_VIRUS) {
        const uint8_t ci = canonical_color_index(a_col);
        virus_prev[ci][r][c] = true;
      } else if (a != 0x00u && a < TILE_JUST_EMPTIED && a_type != TILE_CLEARED) {
        // Any non-empty, non-clearing tile that is not a virus counts as static pill.
        const uint8_t ci = canonical_color_index(a_col);
        static_prev[ci][r][c] = true;
      }

      if (b_type == TILE_VIRUS) {
        const uint8_t ci = canonical_color_index(b_col);
        virus_next[ci][r][c] = true;
      } else if (b != 0x00u && b < TILE_JUST_EMPTIED && b_type != TILE_CLEARED) {
        const uint8_t ci = canonical_color_index(b_col);
        static_next[ci][r][c] = true;
      }
    }
  }

  for (int ci = 0; ci < 3; ++ci) {
    for (int r = 0; r < GRID_H; ++r) {
      for (int c = 0; c < GRID_W; ++c) {
        new_cells[ci][r][c] = static_next[ci][r][c] && !static_prev[ci][r][c];
      }
    }
  }

  auto run_len_dir = [](const bool mask[GRID_H][GRID_W], int r, int c, int dr, int dc) -> int {
    int n = 0;
    int rr = r + dr;
    int cc = c + dc;
    while (rr >= 0 && rr < GRID_H && cc >= 0 && cc < GRID_W && mask[rr][cc]) {
      n += 1;
      rr += dr;
      cc += dc;
    }
    return n;
  };

  for (int ci = 0; ci < 3; ++ci) {
    bool pair_awarded = false;
    bool triplet_awarded = false;
    bool v_pair_awarded = false;
    bool v_triplet_awarded = false;

    // Combined masks for virus-adjacency.
    bool comb_prev[GRID_H][GRID_W]{};
    bool comb_next[GRID_H][GRID_W]{};
    for (int r = 0; r < GRID_H; ++r) {
      for (int c = 0; c < GRID_W; ++c) {
        comb_prev[r][c] = static_prev[ci][r][c] || virus_prev[ci][r][c];
        comb_next[r][c] = static_next[ci][r][c] || virus_next[ci][r][c];
      }
    }

    for (int r = 0; r < GRID_H; ++r) {
      for (int c = 0; c < GRID_W; ++c) {
        if (!new_cells[ci][r][c])
          continue;

        // ---------------- adjacency (static-only)
        const int left_prev = run_len_dir(static_prev[ci], r, c, 0, -1);
        const int right_prev = run_len_dir(static_prev[ci], r, c, 0, 1);
        const int left_new = run_len_dir(static_next[ci], r, c, 0, -1);
        const int right_new = run_len_dir(static_next[ci], r, c, 0, 1);
        const int run_prev_best_h = std::max(left_prev, right_prev);
        const int run_new_total_h = left_new + 1 + right_new;

        if (run_prev_best_h < 3 && run_new_total_h >= 3) {
          triplet_awarded = true;
        } else if (run_prev_best_h < 2 && run_new_total_h >= 2) {
          pair_awarded = true;
        }

        const int up_prev = run_len_dir(static_prev[ci], r, c, -1, 0);
        const int down_prev = run_len_dir(static_prev[ci], r, c, 1, 0);
        const int up_new = run_len_dir(static_next[ci], r, c, -1, 0);
        const int down_new = run_len_dir(static_next[ci], r, c, 1, 0);
        const int run_prev_best_v = std::max(up_prev, down_prev);
        const int run_new_total_v = up_new + 1 + down_new;

        if (run_prev_best_v < 3 && run_new_total_v >= 3) {
          triplet_awarded = true;
        } else if (run_prev_best_v < 2 && run_new_total_v >= 2) {
          pair_awarded = true;
        }

        // ---------------- virus adjacency (static+viruses, requires virus in run)
        const int left_prev_v = run_len_dir(comb_prev, r, c, 0, -1);
        const int right_prev_v = run_len_dir(comb_prev, r, c, 0, 1);
        const int left_new_v = run_len_dir(comb_next, r, c, 0, -1);
        const int right_new_v = run_len_dir(comb_next, r, c, 0, 1);
        const int run_prev_best_hv = std::max(left_prev_v, right_prev_v);
        const int run_new_total_hv = left_new_v + 1 + right_new_v;

        bool virus_in_run_h = false;
        if (run_new_total_hv >= 2) {
          const int c0 = std::max(0, c - left_new_v);
          const int c1 = std::min(GRID_W - 1, c + right_new_v);
          for (int cc = c0; cc <= c1; ++cc) {
            if (virus_next[ci][r][cc]) {
              virus_in_run_h = true;
              break;
            }
          }
        }

        const int up_prev_vv = run_len_dir(comb_prev, r, c, -1, 0);
        const int down_prev_vv = run_len_dir(comb_prev, r, c, 1, 0);
        const int up_new_vv = run_len_dir(comb_next, r, c, -1, 0);
        const int down_new_vv = run_len_dir(comb_next, r, c, 1, 0);
        const int run_prev_best_vv = std::max(up_prev_vv, down_prev_vv);
        const int run_new_total_vv = up_new_vv + 1 + down_new_vv;

        bool virus_in_run_v = false;
        if (run_new_total_vv >= 2) {
          const int r0 = std::max(0, r - up_new_vv);
          const int r1 = std::min(GRID_H - 1, r + down_new_vv);
          for (int rr = r0; rr <= r1; ++rr) {
            if (virus_next[ci][rr][c]) {
              virus_in_run_v = true;
              break;
            }
          }
        }

        const bool virus_in_run = virus_in_run_h || virus_in_run_v;
        if (virus_in_run && ((run_prev_best_hv < 3 && run_new_total_hv >= 3) ||
                             (run_prev_best_vv < 3 && run_new_total_vv >= 3))) {
          v_triplet_awarded = true;
        } else if (virus_in_run && ((run_prev_best_hv < 2 && run_new_total_hv >= 2) ||
                                    (run_prev_best_vv < 2 && run_new_total_vv >= 2))) {
          v_pair_awarded = true;
        }
      }
    }

    out_adj_triplet[ci] = triplet_awarded ? 1u : 0u;
    out_adj_pair[ci] = (!triplet_awarded && pair_awarded) ? 1u : 0u;
    out_v_triplet[ci] = v_triplet_awarded ? 1u : 0u;
    out_v_pair[ci] = (!v_triplet_awarded && v_pair_awarded) ? 1u : 0u;
  }
}

void DrMarioPool::build_obs(uint32_t env_i, float* out_obs_ptr) {
  if (obs_channels_ == 0 || out_obs_ptr == nullptr)
    return;
  const DrMarioState& s = states_[env_i];
  std::memset(out_obs_ptr, 0, sizeof(float) * obs_channels_ * GRID_H * GRID_W);

  // Channels:
  //  0..2: red,yellow,blue (type-blind; bottle tiles only, excludes clearing markers)
  //     3: virus_mask
  //  4..7: feasible mask planes (optional)
  for (int r = 0; r < GRID_H; ++r) {
    for (int c = 0; c < GRID_W; ++c) {
      const uint8_t tile = s.board[r * GRID_W + c];
      const uint8_t type_hi = tile & MASK_TYPE;
      const uint8_t color_lo = tile & MASK_COLOR;

      const bool is_empty = tile == TILE_EMPTY;
      const bool is_zero = tile == 0x00u;
      const bool is_just_emptied = (type_hi == TILE_JUST_EMPTIED) && !is_empty;
      const bool is_clearing = (type_hi == TILE_CLEARED) || is_just_emptied;
      const bool color_valid = !(is_empty || is_zero || is_clearing);

      if (color_valid) {
        const uint8_t ci = canonical_color_index(color_lo);
        const size_t off = static_cast<size_t>(ci) * GRID_H * GRID_W + r * GRID_W + c;
        out_obs_ptr[off] = 1.0f;
      }
      if (type_hi == TILE_VIRUS) {
        const size_t off = static_cast<size_t>(3) * GRID_H * GRID_W + r * GRID_W + c;
        out_obs_ptr[off] = 1.0f;
      }
    }
  }

  if (obs_spec_ != DRM_POOL_OBS_BITPLANE_BOTTLE_MASK)
    return;

  // Feasible mask planes (channels 4..7) from cached reachability.
  const PlannerCache& cache = planner_[env_i];
  if (!cache.valid)
    return;
  for (int a = 0; a < 512; ++a) {
    if (cache.feasible[static_cast<size_t>(a)] == 0u)
      continue;
    const int o = a / (GRID_H * GRID_W);
    const int rem = a % (GRID_H * GRID_W);
    const int row = rem / GRID_W;
    const int col = rem % GRID_W;
    if (o < 0 || o >= 4)
      continue;
    const size_t ch = static_cast<size_t>(4 + o);
    out_obs_ptr[ch * GRID_H * GRID_W + row * GRID_W + col] = 1.0f;
  }
}

void DrMarioPool::update_decision_outputs(uint32_t env_i, DrmPoolOutputs* out) {
  if (out == nullptr)
    return;
  DrMarioState& s = states_[env_i];

  // Ensure planner cache is up to date if we are at a decision point.
  if (s.mode == MODE_PLAYING && s.next_action == 0) {
    (void)ensure_planner(env_i);
  } else {
    invalidate_planner(env_i);
  }

  const size_t mask_off = static_cast<size_t>(env_i) * 512u;
  const size_t u16_off = static_cast<size_t>(env_i) * 512u;
  const size_t colors_off = static_cast<size_t>(env_i) * 2u;

  if (out->spawn_id) {
    out->spawn_id[env_i] = static_cast<uint8_t>(s.pill_counter);
  }

  if (out->pill_colors) {
    out->pill_colors[colors_off + 0] = canonical_color_index(s.falling_pill_color_l);
    out->pill_colors[colors_off + 1] = canonical_color_index(s.falling_pill_color_r);
  }
  if (out->preview_colors) {
    out->preview_colors[colors_off + 0] = canonical_color_index(s.preview_pill_color_l);
    out->preview_colors[colors_off + 1] = canonical_color_index(s.preview_pill_color_r);
  }

  const uint16_t v = count_viruses_u16(s.board);
  if (out->viruses_rem) {
    out->viruses_rem[env_i] = v;
  }
  if (out->board_bytes) {
    const size_t board_off = static_cast<size_t>(env_i) * 128u;
    std::memcpy(out->board_bytes + board_off, s.board, 128u * sizeof(uint8_t));
  }

  const PlannerCache& cache = planner_[env_i];
  if (out->feasible_mask) {
    if (cache.valid) {
      std::memcpy(out->feasible_mask + mask_off, cache.feasible.data(), 512u * sizeof(uint8_t));
    } else {
      std::memset(out->feasible_mask + mask_off, 0, 512u * sizeof(uint8_t));
    }
  }
  if (out->cost_to_lock) {
    if (cache.valid) {
      std::memcpy(out->cost_to_lock + u16_off, cache.cost_to_lock.data(), 512u * sizeof(uint16_t));
    } else {
      for (size_t k = 0; k < 512u; ++k) {
        out->cost_to_lock[u16_off + k] = 0xFFFFu;
      }
    }
  }

  if (out->obs) {
    const size_t obs_off = static_cast<size_t>(env_i) * static_cast<size_t>(obs_channels_) * GRID_H * GRID_W;
    build_obs(env_i, out->obs + obs_off);
  }
}

void DrMarioPool::apply_synthetic_virus_target(uint32_t env_i, const DrmResetSpec& spec) {
  const int target_raw = spec.synthetic_virus_target;
  if (target_raw < 0)
    return;
  const int target = std::max(0, std::min(255, target_raw));

  DrMarioState& s = states_[env_i];

  // Collect virus positions (indices into board[128]).
  int virus_pos[128];
  int virus_n = 0;
  for (int i = 0; i < 128; ++i) {
    if ((s.board[i] & MASK_TYPE) == TILE_VIRUS) {
      virus_pos[virus_n++] = i;
    }
  }
  if (target >= virus_n)
    return;

  if (target <= 0) {
    for (int i = 0; i < virus_n; ++i) {
      s.board[virus_pos[i]] = TILE_EMPTY;
    }
  } else {
    // Choose a target-sized subset to keep, via an in-place shuffle.
    uint32_t seed = spec.synthetic_seed ^ (static_cast<uint32_t>(env_i) * 0x9E3779B9u);
    for (int i = virus_n - 1; i > 0; --i) {
      const int j = rand_bounded(seed, i + 1);
      std::swap(virus_pos[i], virus_pos[j]);
    }
    // Keep first `target`, remove the rest.
    for (int i = target; i < virus_n; ++i) {
      s.board[virus_pos[i]] = TILE_EMPTY;
    }
  }

  if (spec.synthetic_patch_counter) {
    s.viruses_remaining = to_bcd_u8(static_cast<uint16_t>(target));
  }
  // Clearing tasks must not be short-circuited by stale flags.
  s.stage_clear = 0;
}

void DrMarioPool::apply_reset_spec(uint32_t env_i, const DrmResetSpec& spec) {
  DrMarioState& s = states_[env_i];

  // Configure base params before reset (GameLogic.reset reads these).
  int level = std::max(0, std::min(25, static_cast<int>(spec.level)));
  int speed_setting = std::max(0, std::min(2, static_cast<int>(spec.speed_setting)));
  s.level = static_cast<uint8_t>(level);
  s.speed_setting = static_cast<uint8_t>(speed_setting);

  const int speed_ups = std::max(0, std::min(255, static_cast<int>(spec.speed_ups)));
  s.speed_ups = static_cast<uint8_t>(speed_ups);

  if (spec.rng_override) {
    s.rng_state[0] = spec.rng_state[0];
    s.rng_state[1] = spec.rng_state[1];
    s.rng_override = 1;
  } else {
    s.rng_override = 0;
  }

  s.reset_wait_frames = spec.intro_wait_frames;
  s.reset_framecounter_lo_plus1 = spec.intro_frame_counter_lo_plus1;

  // Reset engine.
  games_[env_i].reset();

  // Apply synthetic virus target if requested (after level generation).
  apply_synthetic_virus_target(env_i, spec);

  clear_edge_[env_i].prev_clearing_active = false;
  invalidate_planner(env_i);
}

int DrMarioPool::run_until_actionable_decision(uint32_t env_i, uint8_t last_spawn_id, uint32_t& tau_frames,
                                               uint16_t& match_events, bool& terminated, bool& truncated,
                                               uint8_t& terminal_reason) {
  DrMarioState& s = states_[env_i];

  terminated = false;
  truncated = false;
  terminal_reason = DRM_POOL_TERMINAL_NONE;

  uint32_t remaining = max_wait_frames_;
  bool saw_no_feasible = false;
  while (remaining > 0) {
    if (is_terminal(s)) {
      terminated = true;
      terminal_reason = (s.stage_clear != 0) ? DRM_POOL_TERMINAL_CLEAR : DRM_POOL_TERMINAL_TOPOUT;
      return 0;
    }

    if (is_decision_point(s, last_spawn_id)) {
      const int rc = ensure_planner(env_i);
      if (rc != 0) {
        // Planner failure is treated as a hard timeout (truncate) for safety.
        truncated = true;
        terminal_reason = DRM_POOL_TERMINAL_TIMEOUT;
        return rc;
      }
      if (planner_[env_i].feasible_count > 0) {
        (void)saw_no_feasible;
        return 0;
      }
      // No feasible in-bounds actions; mark this spawn as consumed and keep stepping.
      saw_no_feasible = true;
      last_spawn_id = s.pill_counter;
      invalidate_planner(env_i);
    }

    // Advance one frame with NOOP.
    s.buttons = 0;
    games_[env_i].step();
    tau_frames += 1u;
    remaining -= 1u;
    note_clear_edge(env_i, match_events);
  }

  truncated = true;
  terminal_reason = DRM_POOL_TERMINAL_TIMEOUT;
  return 0;
}

// --------------------------------------------------------------------------- public API

int DrMarioPool::reset(const uint8_t* reset_mask, const DrmResetSpec* reset_specs, DrmPoolOutputs* out) {
  if (out == nullptr || out->struct_size != sizeof(DrmPoolOutputs))
    return -1;

  bool any_reset = false;
  for (uint32_t i = 0; i < num_envs_; ++i) {
    const bool do_reset = (reset_mask == nullptr) ? true : (reset_mask[i] != 0);
    if (!do_reset)
      continue;
    any_reset = true;
    if (reset_specs == nullptr || reset_specs[i].struct_size != sizeof(DrmResetSpec))
      return -2;
  }

  if (!any_reset)
    return 0;

  parallel_for_envs(num_envs_, worker_count_, [&](uint32_t i) {
    const bool do_reset = (reset_mask == nullptr) ? true : (reset_mask[i] != 0);
    if (!do_reset)
      return;

    apply_reset_spec(i, reset_specs[i]);

    // Run to the first actionable decision; do not count these frames as tau.
    uint32_t tau = 0;
    uint16_t match_events = 0;
    bool terminated = false;
    bool truncated = false;
    uint8_t reason = DRM_POOL_TERMINAL_NONE;
    (void)run_until_actionable_decision(i, 0xFF, tau, match_events, terminated, truncated, reason);

    // Reset-time: we expect not to be terminal. If we are, just surface outputs.
    update_decision_outputs(i, out);
  });
  return 0;
}

int DrMarioPool::step(const int32_t* actions, const uint8_t* reset_mask, const DrmResetSpec* reset_specs,
                      DrmPoolOutputs* out) {
  if (actions == nullptr || out == nullptr || out->struct_size != sizeof(DrmPoolOutputs))
    return -1;

  if (reset_mask != nullptr) {
    for (uint32_t i = 0; i < num_envs_; ++i) {
      if (reset_mask[i] == 0)
        continue;
      if (reset_specs == nullptr || reset_specs[i].struct_size != sizeof(DrmResetSpec))
        return -2;
    }
  }

  parallel_for_envs(num_envs_, worker_count_, [&](uint32_t i) {
    const bool do_reset = (reset_mask != nullptr && reset_mask[i] != 0);
    if (do_reset) {
      apply_reset_spec(i, reset_specs[i]);
      // Run to first decision (no tau reported).
      uint32_t tau = 0;
      uint16_t match_events = 0;
      bool terminated = false;
      bool truncated = false;
      uint8_t reason = DRM_POOL_TERMINAL_NONE;
      (void)run_until_actionable_decision(i, 0xFF, tau, match_events, terminated, truncated, reason);

      // Step-only outputs for reset envs.
      if (out->tau_frames)
        out->tau_frames[i] = 0;
      if (out->terminated)
        out->terminated[i] = 0;
      if (out->truncated)
        out->truncated[i] = 0;
      if (out->terminal_reason)
        out->terminal_reason[i] = DRM_POOL_TERMINAL_NONE;
      if (out->invalid_action)
        out->invalid_action[i] = -1;
      if (out->tiles_cleared_total)
        out->tiles_cleared_total[i] = 0;
      if (out->tiles_cleared_virus)
        out->tiles_cleared_virus[i] = 0;
      if (out->tiles_cleared_nonvirus)
        out->tiles_cleared_nonvirus[i] = 0;
      if (out->match_events)
        out->match_events[i] = 0;
      if (out->adj_pair)
        std::memset(out->adj_pair + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->adj_triplet)
        std::memset(out->adj_triplet + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->virus_adj_pair)
        std::memset(out->virus_adj_pair + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->virus_adj_triplet)
        std::memset(out->virus_adj_triplet + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->lock_x)
        out->lock_x[i] = -1;
      if (out->lock_y)
        out->lock_y[i] = -1;
      if (out->lock_rot)
        out->lock_rot[i] = -1;

      update_decision_outputs(i, out);
      return;
    }

    DrMarioState& s = states_[i];
    // Ensure planner for current decision (defensive).
    (void)ensure_planner(i);

    PlannerCache& cache = planner_[i];
    const int action = static_cast<int>(actions[i]);
    bool accepted = false;
    if (action >= 0 && action < 512 && cache.valid && cache.feasible[static_cast<size_t>(action)] != 0u) {
      accepted = true;
    }

    if (!accepted) {
      if (out->invalid_action)
        out->invalid_action[i] = action;
      if (out->tau_frames)
        out->tau_frames[i] = 0;
      if (out->terminated)
        out->terminated[i] = 0;
      if (out->truncated)
        out->truncated[i] = 0;
      if (out->terminal_reason)
        out->terminal_reason[i] = DRM_POOL_TERMINAL_NONE;
      if (out->tiles_cleared_total)
        out->tiles_cleared_total[i] = 0;
      if (out->tiles_cleared_virus)
        out->tiles_cleared_virus[i] = 0;
      if (out->tiles_cleared_nonvirus)
        out->tiles_cleared_nonvirus[i] = 0;
      if (out->match_events)
        out->match_events[i] = 0;
      if (out->adj_pair)
        std::memset(out->adj_pair + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->adj_triplet)
        std::memset(out->adj_triplet + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->virus_adj_pair)
        std::memset(out->virus_adj_pair + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->virus_adj_triplet)
        std::memset(out->virus_adj_triplet + static_cast<size_t>(i) * 3u, 0, 3);
      if (out->lock_x)
        out->lock_x[i] = -1;
      if (out->lock_y)
        out->lock_y[i] = -1;
      if (out->lock_rot)
        out->lock_rot[i] = -1;

      // Re-emit the same decision context.
      update_decision_outputs(i, out);
      return;
    }

    if (out->invalid_action)
      out->invalid_action[i] = -1;

    // Enable tile-clear counters via the existing fields (used by cpp-engine batched runs).
    s.run_tiles_cleared_total = 0;
    s.run_tiles_cleared_virus = 0;
    s.run_tiles_cleared_nonvirus = 0;
    s.run_request_id = 1;
    s.run_ack_id = 0;

    // Capture board at decision (static) for adjacency checks at lock.
    uint8_t board_prev[128];
    std::memcpy(board_prev, s.board, 128);

    // Execute controller script for chosen action.
    uint32_t tau = 0;
    uint16_t match_events = 0;
    bool terminated = false;
    bool truncated = false;
    uint8_t terminal_reason = DRM_POOL_TERMINAL_NONE;

    int bx = 0, by = 0, brot = 0;
    if (!base_pose_for_macro_action(action, bx, by, brot)) {
      // Should not happen for feasible actions.
      truncated = true;
      terminal_reason = DRM_POOL_TERMINAL_TIMEOUT;
    } else {
      const int pose = pose_index(bx, by, brot);
      const uint16_t len_u16 = cache.lengths_u16[static_cast<size_t>(pose)];
      const uint16_t off_u16 = cache.offsets_u16[static_cast<size_t>(pose)];
      const int len = static_cast<int>(len_u16);
      const int off = static_cast<int>(off_u16);
      const int end = off + len;
      const bool script_ok = (len > 0) && (off >= 0) && (end <= cache.script_used);

      if (!script_ok) {
        truncated = true;
        terminal_reason = DRM_POOL_TERMINAL_TIMEOUT;
      } else {
        for (int t = 0; t < len; ++t) {
          const uint8_t a_idx = cache.script_buf[static_cast<size_t>(off + t)];
          s.buttons = buttons_mask_from_reach_action(a_idx);
          games_[i].step();
          tau += 1u;
          note_clear_edge(i, match_events);
          if (is_terminal(s)) {
            terminated = true;
            terminal_reason = (s.stage_clear != 0) ? DRM_POOL_TERMINAL_CLEAR : DRM_POOL_TERMINAL_TOPOUT;
            break;
          }
        }
      }
    }

    // Adjacency flags at lock boundary (only if we actually locked).
    uint8_t adj_pair[3]{0, 0, 0};
    uint8_t adj_triplet[3]{0, 0, 0};
    uint8_t v_pair[3]{0, 0, 0};
    uint8_t v_triplet[3]{0, 0, 0};

    if (!terminated && !truncated) {
      // Capture lock pose and board after lock (if lock occurred).
      const bool locked = (s.next_action != 0);
      if (locked) {
        compute_adjacency_flags(board_prev, s.board, adj_pair, adj_triplet, v_pair, v_triplet);

        if (out->lock_x)
          out->lock_x[i] = static_cast<int16_t>(static_cast<int>(s.falling_pill_col));
        if (out->lock_y) {
          const int row_from_bottom = static_cast<int>(s.falling_pill_row);
          const int y_top = (GRID_H - 1) - row_from_bottom;
          out->lock_y[i] = static_cast<int16_t>(y_top);
        }
        if (out->lock_rot)
          out->lock_rot[i] = static_cast<int16_t>(static_cast<int>(s.falling_pill_orient) & 3);
      } else {
        // If we did not lock, treat as timeout to avoid infinite loops.
        truncated = true;
        terminal_reason = DRM_POOL_TERMINAL_TIMEOUT;
      }
    } else {
      if (out->lock_x)
        out->lock_x[i] = -1;
      if (out->lock_y)
        out->lock_y[i] = -1;
      if (out->lock_rot)
        out->lock_rot[i] = -1;
    }

    if (out->adj_pair)
      std::memcpy(out->adj_pair + static_cast<size_t>(i) * 3u, adj_pair, 3);
    if (out->adj_triplet)
      std::memcpy(out->adj_triplet + static_cast<size_t>(i) * 3u, adj_triplet, 3);
    if (out->virus_adj_pair)
      std::memcpy(out->virus_adj_pair + static_cast<size_t>(i) * 3u, v_pair, 3);
    if (out->virus_adj_triplet)
      std::memcpy(out->virus_adj_triplet + static_cast<size_t>(i) * 3u, v_triplet, 3);

    // Phase 2: run until the next actionable decision (skip empty-feasible spawns).
    if (!terminated && !truncated) {
      s.buttons = 0;
      const uint8_t last_spawn = cache.spawn_id;
      invalidate_planner(i); // decision changed
      bool term2 = false;
      bool trunc2 = false;
      uint8_t reason2 = DRM_POOL_TERMINAL_NONE;
      (void)run_until_actionable_decision(i, last_spawn, tau, match_events, term2, trunc2, reason2);
      if (term2) {
        terminated = true;
        terminal_reason = reason2;
      } else if (trunc2) {
        truncated = true;
        terminal_reason = reason2;
      }
    }

    // Disable run counters.
    s.run_ack_id = s.run_request_id;

    // Step outputs.
    if (out->tau_frames)
      out->tau_frames[i] = tau;
    if (out->terminated)
      out->terminated[i] = terminated ? 1u : 0u;
    if (out->truncated)
      out->truncated[i] = truncated ? 1u : 0u;
    if (out->terminal_reason)
      out->terminal_reason[i] = terminal_reason;

    if (out->tiles_cleared_total)
      out->tiles_cleared_total[i] =
          static_cast<uint16_t>(std::min<uint32_t>(s.run_tiles_cleared_total, 0xFFFFu));
    if (out->tiles_cleared_virus)
      out->tiles_cleared_virus[i] =
          static_cast<uint16_t>(std::min<uint32_t>(s.run_tiles_cleared_virus, 0xFFFFu));
    if (out->tiles_cleared_nonvirus)
      out->tiles_cleared_nonvirus[i] =
          static_cast<uint16_t>(std::min<uint32_t>(s.run_tiles_cleared_nonvirus, 0xFFFFu));
    if (out->match_events)
      out->match_events[i] = match_events;

    // Decision outputs: if terminal/truncated, still emit observation; masks are zeroed.
    update_decision_outputs(i, out);
  });

  return 0;
}
