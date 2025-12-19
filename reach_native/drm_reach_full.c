// Dr. Mario Falling-Pill Reachability (NES-accurate, native helper)
//
// This module implements a native BFS over the *full* per-frame fall state used by
// `envs/retro/fast_reach.py`:
//   (x, y, rot, speed_counter, hor_velocity, hold_dir, parity, rot_hold)
//
// It is intended as a performance accelerator for the placement macro-action
// environment. The Python reference implementation is correct but can be slow
// when enumerating reachability for every spawn.
//
// Key invariants mirrored from the Python stepper:
// - Frame order is Y (gravity/soft-drop) -> X (DAS) -> Rotate.
// - "Down-only" soft drop triggers on parity frames (frameCounter & 1 == 0) and
//   resets speed_counter.
// - Gravity triggers when (speed_counter + 1) > speed_threshold, then resets.
// - Horizontal movement:
//     - Edge press moves immediately and resets hor_velocity.
//     - When holding L/R, hor_velocity increments; on >= 16 it triggers a move,
//       then reloads to 10 (repeat every 6 frames).
//     - Blocked movement sets hor_velocity = 15.
// - Rotation quirks:
//     - Rotation uses btnsPressed edge semantics: holding A/B across consecutive
//       frames triggers only on the first frame.
//     - Rotation-to-horizontal accepts an additional "held-left double-left"
//       move if it fits.
//     - If blocked, a kick-left attempt is made.
//
// Public API for Python (ctypes):
//   int drm_reach_bfs_full(..., out_costs[512], out_offsets[512], out_lengths[512],
//                          out_script_buf[cap], *out_used)
//
// Outputs are indexed by base pose index:
//   pose_idx = x + 8*y + 8*16*(rot & 3)   (512 total)
// where (x,y,rot) is the *locked* base-cell pose.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { GRID_W = 8, GRID_H = 16 };

enum { HOLD_NEUTRAL = 0, HOLD_LEFT = 1, HOLD_RIGHT = 2 };

// NES constants (must match envs/retro/fast_reach.py)
enum { FAST_DROP_MASK = 0x01 };
enum { HOR_ACCEL_SPEED = 0x10, HOR_RELOAD = 0x0A, HOR_BLOCKED = 0x0F };

// Action encoding: same stable order as Python `_ACTION_SPACE`:
//   for hold_dir in (NEUTRAL, LEFT, RIGHT)
//     for hold_down in (False, True)
//       for rotation in (NONE, CW, CCW)
//
// Performance note: this table-driven decode avoids div/mod in the inner BFS
// loop while preserving the exact action ordering of the Python planner.
static const uint8_t ACT_HOLD_DIR[18] = {
    0, 0, 0, 0, 0, 0,  // NEUTRAL × (down,no-down) × (none,cw,ccw)
    1, 1, 1, 1, 1, 1,  // LEFT
    2, 2, 2, 2, 2, 2,  // RIGHT
};
static const uint8_t ACT_HOLD_DOWN[18] = {
    0, 0, 0, 1, 1, 1,  // NEUTRAL
    0, 0, 0, 1, 1, 1,  // LEFT (down is redundant but kept for script identity)
    0, 0, 0, 1, 1, 1,  // RIGHT (down is redundant but kept for script identity)
};
static const uint8_t ACT_ROT[18] = {
    0, 1, 2, 0, 1, 2,  // NEUTRAL
    0, 1, 2, 0, 1, 2,  // LEFT
    0, 1, 2, 0, 1, 2,  // RIGHT
};

static inline int pose_index(int x, int y, int rot) {
    return (int)((rot & 3) * (GRID_H * GRID_W) + (y * GRID_W) + x);
}

// ---------------------------------------------------------------------------
// Optional instrumentation (enabled via DRMARIO_REACH_STATS=1)
// ---------------------------------------------------------------------------

typedef struct {
    uint32_t visited_states;   // number of unique full states enqueued (qt)
    uint32_t expanded_states;  // number of states popped/expanded
    uint32_t transitions;      // action applications attempted
    uint32_t locks_found;      // unique locked poses discovered (out_costs set)
    uint32_t queue_nodes_enqueued;   // number of queued (key,xmask) nodes enqueued
    uint32_t queue_nodes_expanded;   // number of queued (key,xmask) nodes popped
    uint16_t max_depth;        // per-call depth cap
    uint16_t depth_processed;  // last fully processed depth (level-order)
    uint16_t wanted_count;     // terminal poses targeted for early stop
    uint16_t found_wanted;     // how many of those were found
} DrmReachStats;

static DrmReachStats g_last_stats;
static int g_stats_enabled = -1;

static inline int stats_enabled(void) {
    if (g_stats_enabled >= 0) return g_stats_enabled;
    const char* env = getenv("DRMARIO_REACH_STATS");
    g_stats_enabled = (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    return g_stats_enabled;
}

int drm_reach_get_last_stats(DrmReachStats* out, int out_size) {
    if (!out) return -1;
    if (out_size < (int)sizeof(DrmReachStats)) return -1;
    memcpy(out, &g_last_stats, sizeof(DrmReachStats));
    return 0;
}

// ---------------------------------------------------------------------------
// Collision masks (precomputed per-board)
// ---------------------------------------------------------------------------

// Convert the 8×uint16 column bitboards into fast row masks:
//   occ[y] bit x = 1  iff  cell (x,y) is occupied
//
// Then build, for each row y:
//   fit_mask[0][y] (horizontal) bit x = 1 iff base at x fits (cells x and x+1 empty)
//   fit_mask[1][y] (vertical)   bit x = 1 iff base at x fits (cell x,y empty and x,y-1 empty unless y==0)
static inline void build_fit_masks(const uint16_t cols[GRID_W], uint8_t fit_mask[2][GRID_H]) {
    uint8_t occ[GRID_H];
    uint8_t empty[GRID_H];
    for (int y = 0; y < GRID_H; ++y) occ[y] = 0u;

    for (int x = 0; x < GRID_W; ++x) {
        const uint16_t col = cols[x];
        for (int y = 0; y < GRID_H; ++y) {
            if (col & (uint16_t)(1u << (unsigned)y)) {
                occ[y] |= (uint8_t)(1u << (unsigned)x);
            }
        }
    }
    for (int y = 0; y < GRID_H; ++y) {
        empty[y] = (uint8_t)(~occ[y]) & 0xFFu;
    }
    for (int y = 0; y < GRID_H; ++y) {
        // Horizontal: base at x uses cells (x,y) and (x+1,y).
        // `empty >> 1` aligns the partner cell to the base.
        fit_mask[0][y] = (uint8_t)(empty[y] & (uint8_t)(empty[y] >> 1));
    }
    // Vertical: base at x uses cells (x,y) and (x,y-1). Allow y==0 (partner offscreen).
    fit_mask[1][0] = empty[0];
    for (int y = 1; y < GRID_H; ++y) {
        fit_mask[1][y] = (uint8_t)(empty[y] & empty[y - 1]);
    }
}

static inline int fits_masked(const uint8_t fit_mask[2][GRID_H], int x, int y, int rot) {
    if ((unsigned)x >= (unsigned)GRID_W || (unsigned)y >= (unsigned)GRID_H) return 0;
    const uint8_t mask = fit_mask[(rot & 1) ? 1 : 0][y];
    return (int)((mask >> (unsigned)x) & 1u);
}

static inline void apply_rotation_masked(
    const uint8_t fit_mask[2][GRID_H], int* x, int y, int* rot, int rotation, int hold_left
);

static int build_wanted_terminal_poses_reachable(
    const uint8_t fit_mask[2][GRID_H], int sx, int sy, int srot, uint8_t wanted[512]
) {
    // Conservative "wanted" set: terminal poses that are reachable from the
    // spawn in a *timer-free* flood fill over (x, y, rot).
    //
    // This intentionally ignores the falling-pill counters (speed_counter,
    // parity, DAS) and allows arbitrary sequences of left/right/rotate/down
    // moves. That makes it a *superset* of the real per-frame reachable set:
    // if a pose is unreachable even with these relaxed rules, it is definitely
    // unreachable in the real game.
    //
    // We use this to prune the early-termination target set: many boards have
    // macro-legal lock poses that are geometrically unreachable due to sealed
    // cavities. Without pruning, the frame-accurate BFS must explore to
    // `max_lock_frames` just to prove those are unreachable, which is wasted
    // work for macro-action planning.
    //
    // Importantly: this pruning never removes a pose that could be reachable
    // under the real rules; it can only keep extra poses (false positives).
    // If the relaxed flood fill over-approximates some pose as reachable but
    // the real BFS cannot reach it, early termination simply won't trigger.
    memset(wanted, 0, 512u);

    if ((unsigned)sx >= (unsigned)GRID_W || (unsigned)sy >= (unsigned)GRID_H) return 0;
    if (!fits_masked(fit_mask, sx, sy, srot & 3)) return 0;

    uint8_t visited[512];
    memset(visited, 0, sizeof(visited));

    uint16_t queue[512];
    uint16_t qh = 0;
    uint16_t qt = 0;

    const uint16_t start = (uint16_t)pose_index(sx, sy, srot & 3);
    visited[start] = 1u;
    queue[qt++] = start;

    int wanted_count = 0;
    while (qh < qt) {
        const uint16_t pose = queue[qh++];
        const int x = (int)(pose & 7u);
        const int y = (int)((pose >> 3) & 15u);
        const int rot = (int)((pose >> 7) & 3u);

        // Terminal (lock) pose for macro actions: cannot move one row lower and
        // both halves are on-screen.
        const int can_fall = (y + 1 < GRID_H) && fits_masked(fit_mask, x, y + 1, rot);
        if (!can_fall) {
            if ((rot & 1) == 0) {
                // Horizontal: requires x+1 in bounds (fit_mask already encodes this).
                if (x + 1 < GRID_W) {
                    if (!wanted[pose]) {
                        wanted[pose] = 1u;
                        wanted_count += 1;
                    }
                }
            } else {
                // Vertical: requires the upper half (y-1) to be on-screen.
                if (y >= 1) {
                    if (!wanted[pose]) {
                        wanted[pose] = 1u;
                        wanted_count += 1;
                    }
                }
            }
        }

        // Left / right moves.
        if (fits_masked(fit_mask, x - 1, y, rot)) {
            const uint16_t np = (uint16_t)pose_index(x - 1, y, rot);
            if (!visited[np]) {
                visited[np] = 1u;
                queue[qt++] = np;
            }
        }
        if (fits_masked(fit_mask, x + 1, y, rot)) {
            const uint16_t np = (uint16_t)pose_index(x + 1, y, rot);
            if (!visited[np]) {
                visited[np] = 1u;
                queue[qt++] = np;
            }
        }

        // Down move (one row).
        if (y + 1 < GRID_H && fits_masked(fit_mask, x, y + 1, rot)) {
            const uint16_t np = (uint16_t)pose_index(x, y + 1, rot);
            if (!visited[np]) {
                visited[np] = 1u;
                queue[qt++] = np;
            }
        }

        // Rotation moves. Consider both hold_left states to include the double-left
        // quirk when rotating to horizontal.
        for (int rotation = 1; rotation <= 2; ++rotation) {
            for (int hold_left = 0; hold_left <= 1; ++hold_left) {
                int rx = x;
                int rrot = rot;
                apply_rotation_masked(fit_mask, &rx, y, &rrot, rotation, hold_left);
                if (rx == x && (rrot & 3) == (rot & 3)) continue;
                const uint16_t np = (uint16_t)pose_index(rx, y, rrot);
                if (!visited[np]) {
                    visited[np] = 1u;
                    queue[qt++] = np;
                }
            }
        }
    }
    return wanted_count;
}

static inline void apply_rotation_masked(
    const uint8_t fit_mask[2][GRID_H], int* x, int y, int* rot, int rotation, int hold_left
) {
    if (rotation == 0) return;
    const int x0 = *x;
    const int rot0 = (*rot) & 3;
    int rot1 = rot0;
    if (rotation == 1) rot1 = (rot0 - 1) & 3;     // CW: decrement (NES A)
    else rot1 = (rot0 + 1) & 3;                   // CCW: increment (NES B)

    if ((rot1 & 1) == 0) {
        // Target is horizontal.
        if (fits_masked(fit_mask, x0, y, rot1)) {
            // Rotation accepted in-place.
            if (hold_left && fits_masked(fit_mask, x0 - 1, y, rot1)) {
                *x = x0 - 1;
                *rot = rot1;
                return;
            }
            *x = x0;
            *rot = rot1;
            return;
        }
        // Kick-left attempt.
        if (fits_masked(fit_mask, x0 - 1, y, rot1)) {
            *x = x0 - 1;
            *rot = rot1;
            return;
        }
        // Reject.
        return;
    }

    // Target is vertical: only in-place validation.
    if (fits_masked(fit_mask, x0, y, rot1)) {
        *rot = rot1;
    }
}

// ---------------------------------------------------------------------------
// BFS workspace (persistent; no per-call malloc/free)
// ---------------------------------------------------------------------------

// Queue node for a *set* of x positions that share identical counter state.
//
// Key insight: many states differ only in x. We can apply one action to all x
// positions at once using 8-bit masks, greatly reducing work without changing
// semantics.
typedef struct {
    uint32_t key;     // mixed-radix index for (y,rot,sc,hv,hd,p,rh) (no x)
    uint8_t xmask;    // bit x=1 => state with that x is present at this depth
    uint8_t y, rot;   // 0..15, 0..3
    uint8_t sc;       // 0..speed_threshold
    uint8_t hv;       // 0..15
    uint8_t hd;       // 0..2
    uint8_t p;        // 0..1
    uint8_t rh;       // 0..2 (A/B held in previous frame)
} NodeMask;

typedef struct {
    uint32_t cap_keys;
    uint32_t cap_states;  // cap_keys * GRID_W
    uint8_t* visited_xmask;  // cap_keys bytes; per-key visited x positions
    uint8_t* next_xmask;     // cap_keys bytes; next-frontier xmask accumulator
    NodeMask* frontier_a;    // cap_keys nodes
    NodeMask* frontier_b;    // cap_keys nodes
    uint32_t* parent;
    uint8_t* parent_action;
} ReachCtx;

static ReachCtx g_ctx = {0};

static int ensure_ctx(uint32_t nkeys) {
    if (nkeys == 0) return -2;
    const uint32_t nstates = nkeys * (uint32_t)GRID_W;
    if (
        g_ctx.cap_keys >= nkeys
        && g_ctx.visited_xmask != NULL
        && g_ctx.next_xmask != NULL
        && g_ctx.frontier_a != NULL
        && g_ctx.frontier_b != NULL
        && g_ctx.parent != NULL
        && g_ctx.parent_action != NULL
    ) {
        return 0;
    }

    free(g_ctx.visited_xmask);
    free(g_ctx.next_xmask);
    free(g_ctx.frontier_a);
    free(g_ctx.frontier_b);
    free(g_ctx.parent);
    free(g_ctx.parent_action);
    g_ctx.visited_xmask = NULL;
    g_ctx.next_xmask = NULL;
    g_ctx.frontier_a = NULL;
    g_ctx.frontier_b = NULL;
    g_ctx.parent = NULL;
    g_ctx.parent_action = NULL;
    g_ctx.cap_keys = 0;
    g_ctx.cap_states = 0;

    g_ctx.visited_xmask = (uint8_t*)malloc((size_t)nkeys * sizeof(uint8_t));
    g_ctx.next_xmask = (uint8_t*)malloc((size_t)nkeys * sizeof(uint8_t));
    g_ctx.frontier_a = (NodeMask*)malloc((size_t)nkeys * sizeof(NodeMask));
    g_ctx.frontier_b = (NodeMask*)malloc((size_t)nkeys * sizeof(NodeMask));
    g_ctx.parent = (uint32_t*)malloc((size_t)nstates * sizeof(uint32_t));
    g_ctx.parent_action = (uint8_t*)malloc((size_t)nstates * sizeof(uint8_t));

    if (
        !g_ctx.visited_xmask || !g_ctx.next_xmask || !g_ctx.frontier_a || !g_ctx.frontier_b || !g_ctx.parent
        || !g_ctx.parent_action
    ) {
        free(g_ctx.visited_xmask);
        free(g_ctx.next_xmask);
        free(g_ctx.frontier_a);
        free(g_ctx.frontier_b);
        free(g_ctx.parent);
        free(g_ctx.parent_action);
        g_ctx.visited_xmask = NULL;
        g_ctx.next_xmask = NULL;
        g_ctx.frontier_a = NULL;
        g_ctx.frontier_b = NULL;
        g_ctx.parent = NULL;
        g_ctx.parent_action = NULL;
        return -2;
    }

    g_ctx.cap_keys = nkeys;
    g_ctx.cap_states = nstates;
    return 0;
}

typedef struct {
    uint32_t stride_rot;
    uint32_t stride_sc;
    uint32_t stride_hv;
    uint32_t stride_hd;
    uint32_t stride_p;
    uint32_t stride_rh;
} KeyStrides;

static inline uint32_t key_from_fields(
    const KeyStrides* s, uint32_t y, uint32_t rot, uint32_t sc, uint32_t hv, uint32_t hd, uint32_t p, uint32_t rh
) {
    return y + rot * s->stride_rot + sc * s->stride_sc + hv * s->stride_hv + hd * s->stride_hd + p * s->stride_p
           + rh * s->stride_rh;
}

static inline uint32_t full_idx(uint32_t key, uint32_t x) {
    // GRID_W is 8, so multiply is a shift.
    return (key << 3) + x;
}

static inline int compute_max_lock_frames(int y0, int sc0, int speed_threshold) {
    // Exact upper bound on time-to-lock (in frames) under the slowest possible descent:
    // never "down-only" soft-drop (which can only speed things up).
    //
    // Let T=speed_threshold and sc0 be the initial speed counter (clamped 0..T).
    // The first gravity drop triggers after (T - sc0) + 1 frames; each subsequent
    // drop triggers every (T+1) frames. Lock occurs on the first drop attempt that
    // fails, so the maximum number of drop attempts is (GRID_H - y0) (the number
    // of rows remaining including the final failed attempt at y==15).
    const int T = speed_threshold;
    int sc = sc0;
    if (sc < 0) sc = 0;
    if (sc > T) sc = T;
    int m_max = GRID_H - y0;
    if (m_max < 1) m_max = 1;
    const int first = (T - sc) + 1;
    const int per = T + 1;
    int total = first;
    if (m_max > 1) total += (m_max - 1) * per;
    if (total < 1) total = 1;
    return total;
}

int drm_reach_bfs_full(
    const uint16_t cols[GRID_W],
    int sx, int sy, int srot,
    int speed_counter, int hor_velocity,
    int hold_dir, int parity, int rot_hold,
    int speed_threshold,
    int max_frames,
    uint16_t out_costs[512],
    uint16_t out_offsets[512],
    uint16_t out_lengths[512],
    uint8_t* out_script_buf,
    int script_buf_cap,
    int* out_script_used
) {
    if (!cols || !out_costs || !out_offsets || !out_lengths || !out_script_buf || !out_script_used) return -1;
    if (script_buf_cap <= 0 || max_frames <= 0) return -1;

    // Initialise outputs.
    for (int i = 0; i < 512; ++i) {
        out_costs[i] = 0xFFFFu;
        out_offsets[i] = 0u;
        out_lengths[i] = 0u;
    }
    *out_script_used = 0;

    if ((unsigned)sx >= (unsigned)GRID_W || (unsigned)sy >= (unsigned)GRID_H) return 0;

    if (speed_threshold < 0) speed_threshold = 0;
    if (speed_threshold > 0x7F) speed_threshold = 0x7F;
    const int sc_range = speed_threshold + 1;
    if (sc_range <= 0) return 0;

    if (speed_counter < 0) speed_counter = 0;
    if (speed_counter > speed_threshold) speed_counter = speed_threshold;
    hor_velocity &= 0xFF;
    if (hor_velocity < 0) hor_velocity = 0;
    if (hor_velocity > 15) hor_velocity &= 0x0F;  // state space stores 4 bits in the Python packer.
    if (hold_dir < 0 || hold_dir > 2) hold_dir = 0;
    parity &= FAST_DROP_MASK;
    if (rot_hold < 0 || rot_hold > 2) rot_hold = 0;

    // Exact, per-call upper bound for time-to-lock; clamps work without changing semantics.
    const int max_lock_frames = compute_max_lock_frames(sy, speed_counter, speed_threshold);
    if (max_frames > max_lock_frames) max_frames = max_lock_frames;

    // Precompute per-row collision masks (cheap) so the inner loop avoids repeated
    // column bit tests.
    uint8_t fit_mask[2][GRID_H];
    build_fit_masks(cols, fit_mask);
    if (!fits_masked(fit_mask, sx, sy, srot & 3)) return 0;

    // State space (non-x key):
    //   (y:16) × (rot:4) × (sc:sc_range) × (hv:16) × (hd:3) × (p:2) × (rh:3)
    //
    // Full state count is nkeys * 8 (one x bit per key).
    KeyStrides strides;
    strides.stride_rot = GRID_H;
    strides.stride_sc = strides.stride_rot * 4u;
    strides.stride_hv = strides.stride_sc * (uint32_t)sc_range;
    strides.stride_hd = strides.stride_hv * 16u;
    strides.stride_p = strides.stride_hd * 3u;
    strides.stride_rh = strides.stride_p * 2u;
    const uint32_t nkeys = strides.stride_rh * 3u;
    const uint32_t nstates = nkeys * (uint32_t)GRID_W;

    // Precompute mixed-radix terms to avoid multiplications in the hot loop.
    uint32_t rot_term[4];
    rot_term[0] = 0u;
    rot_term[1] = strides.stride_rot;
    rot_term[2] = strides.stride_rot * 2u;
    rot_term[3] = strides.stride_rot * 3u;

    uint32_t hv_term[16];
    for (uint32_t i = 0; i < 16u; ++i) hv_term[i] = i * strides.stride_hv;

    uint32_t hd_term[3];
    hd_term[0] = 0u;
    hd_term[1] = strides.stride_hd;
    hd_term[2] = strides.stride_hd * 2u;

    uint32_t p_term[2];
    p_term[0] = 0u;
    p_term[1] = strides.stride_p;

    uint32_t rh_term[3];
    rh_term[0] = 0u;
    rh_term[1] = strides.stride_rh;
    rh_term[2] = strides.stride_rh * 2u;

    uint32_t sc_term[128];
    for (uint32_t i = 0; i < (uint32_t)sc_range; ++i) sc_term[i] = i * strides.stride_sc;

    const int ctx_rc = ensure_ctx(nkeys);
    if (ctx_rc != 0) return ctx_rc;

    // Clear only the per-key visited masks; parent arrays are written only for
    // visited full states.
    memset(g_ctx.visited_xmask, 0, (size_t)nkeys * sizeof(uint8_t));

    uint32_t term_parent_state[512];
    uint8_t term_parent_action[512];
    for (int i = 0; i < 512; ++i) {
        term_parent_state[i] = UINT32_MAX;
        term_parent_action[i] = 0xFFu;
    }

    const uint32_t start_key = (uint32_t)(sy & 15) + rot_term[(unsigned)(srot & 3)]
                               + sc_term[(unsigned)speed_counter] + hv_term[(unsigned)(hor_velocity & 15)]
                               + hd_term[(unsigned)hold_dir] + p_term[(unsigned)(parity & 1)]
                               + rh_term[(unsigned)rot_hold];
    const uint8_t start_xmask = (uint8_t)((uint8_t)1u << (unsigned)(sx & 7));
    g_ctx.visited_xmask[start_key] = start_xmask;

    const uint32_t start_full = full_idx(start_key, (uint32_t)(sx & 7));
    g_ctx.parent[start_full] = UINT32_MAX;
    g_ctx.parent_action[start_full] = 0xFFu;

    // Two-frontier BFS with per-depth key aggregation.
    //
    // The core speed win: for each depth, we aggregate all x-bits that share the
    // same (y,rot,sc,hv,hd,p,rh) key into a single node. This enables bitmask
    // propagation over x without the "degenerate" behaviour where each queue
    // entry carries only one x bit.
    memset(g_ctx.next_xmask, 0, (size_t)nkeys * sizeof(uint8_t));
    NodeMask* cur_frontier = g_ctx.frontier_a;
    NodeMask* next_frontier = g_ctx.frontier_b;
    uint32_t cur_n = 1;
    uint32_t next_n = 0;
    cur_frontier[0].key = start_key;
    cur_frontier[0].xmask = start_xmask;
    cur_frontier[0].y = (uint8_t)(sy & 15);
    cur_frontier[0].rot = (uint8_t)(srot & 3);
    cur_frontier[0].sc = (uint8_t)speed_counter;
    cur_frontier[0].hv = (uint8_t)(hor_velocity & 15);
    cur_frontier[0].hd = (uint8_t)hold_dir;
    cur_frontier[0].p = (uint8_t)(parity & 1);
    cur_frontier[0].rh = (uint8_t)rot_hold;

    const uint16_t max_depth = (uint16_t)max_frames;

    uint32_t visited_states = 1;
    uint32_t expanded_states = 0;
    uint32_t transitions = 0;
    uint32_t locks_found = 0;
    uint32_t queue_nodes_enqueued = 1;
    uint32_t queue_nodes_expanded = 0;
    uint16_t depth_processed = 0;

    // Pre-compute how many *macro-action terminal poses* exist on this static
    // board. If we find all of them, deeper exploration cannot improve any
    // placement and we can terminate early (large perf win on open boards).
    uint8_t wanted_pose[512];
    const int wanted_count = build_wanted_terminal_poses_reachable(
        fit_mask, (int)(sx & 7), (int)(sy & 15), (int)(srot & 3), wanted_pose
    );
    int found_wanted = 0;

    // Action set: skip redundant (hold_down=True) variants for LEFT/RIGHT.
    // They are byte-for-byte identical transitions because "down-only" requires
    // HOLD_NEUTRAL; keeping only the earlier variants preserves script identity.
    static const uint8_t ACTIONS_EVEN[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};
    // On odd-parity frames, down-only soft drop is not checked, so the
    // (NEUTRAL, hold_down=True, rot=*) variants are redundant with the earlier
    // hold_down=False ones and can be skipped safely.
    static const uint8_t ACTIONS_ODD[9] = {0, 1, 2, 6, 7, 8, 12, 13, 14};

    // Level-order BFS (exact): one frontier per depth.
    for (uint16_t cur_depth = 0; cur_depth < max_depth && cur_n > 0; ++cur_depth) {
        depth_processed = cur_depth;
        queue_nodes_expanded += cur_n;
        next_n = 0;

        const uint16_t next_depth = (uint16_t)(cur_depth + 1);
        for (uint32_t ni = 0; ni < cur_n; ++ni) {
            const NodeMask cur = cur_frontier[ni];
            const uint8_t cur_xmask = cur.xmask;
            const int cur_bits = __builtin_popcount((unsigned)cur_xmask);
            expanded_states += (uint32_t)cur_bits;
            const uint32_t cur_key = cur.key;
            const uint32_t cur_full_base = cur_key << 3;

            const int parity_cur = (int)cur.p & 1;
            const uint8_t* actions = parity_cur ? ACTIONS_ODD : ACTIONS_EVEN;
            const int action_count = parity_cur ? 9 : 12;

            for (int ai = 0; ai < action_count; ++ai) {
                const int act = (int)actions[ai];
                transitions += (uint32_t)cur_bits;

                uint8_t xmask = cur_xmask;
                int y = (int)cur.y;
                const int rot0 = (int)cur.rot & 3;
                int rot = rot0;
                int sc = (int)cur.sc;
                const int hv0 = (int)cur.hv & 0x0F;
                const int hd_prev = (int)cur.hd;
                const int parity0 = parity_cur;
                const int rh_prev = (int)cur.rh;

                const int hold_dir_now = (int)ACT_HOLD_DIR[act];
                const int hold_down = (int)ACT_HOLD_DOWN[act];
                const int rotation = (int)ACT_ROT[act];  // 0 none, 1 cw, 2 ccw

                const int prev_left = (hd_prev == HOLD_LEFT);
                const int prev_right = (hd_prev == HOLD_RIGHT);
                const int hold_left = (hold_dir_now == HOLD_LEFT);
                const int hold_right = (hold_dir_now == HOLD_RIGHT);

                const int press_left = hold_left && !prev_left;
                const int press_right = hold_right && !prev_right;
                const int press_lr = press_left || press_right;

                // ---------------- Y stage (gravity / down-only soft drop) ----------------
                const int down_only = (hold_down != 0) && (hold_dir_now == HOLD_NEUTRAL);
                int drop_triggered = 0;

                if ((parity0 & FAST_DROP_MASK) == 0 && down_only) {
                    drop_triggered = 1;
                    sc = 0;
                } else {
                    sc = sc + 1;
                    if (sc > speed_threshold) {
                        drop_triggered = 1;
                        sc = 0;
                    }
                }

                if (drop_triggered) {
                    const int ny = y + 1;
                    uint8_t drop_ok = 0u;
                    if ((unsigned)ny < (unsigned)GRID_H) {
                        drop_ok = fit_mask[(rot & 1) ? 1 : 0][ny];
                    }
                    uint8_t xm_drop = (uint8_t)(xmask & drop_ok);
                    uint8_t xm_lock = (uint8_t)(xmask & (uint8_t)(~drop_ok));
                    while (xm_lock) {
                        const int lx = __builtin_ctz((unsigned)xm_lock);
                        xm_lock &= (uint8_t)(xm_lock - 1u);
                        const int pose = pose_index(lx, y, rot0);
                        if ((unsigned)pose >= 512u) continue;
                        if (out_costs[pose] != 0xFFFFu) continue;
                        out_costs[pose] = next_depth;
                        term_parent_state[pose] = cur_full_base + (uint32_t)lx;
                        term_parent_action[pose] = (uint8_t)act;
                        locks_found += 1;
                        if (wanted_pose[pose]) {
                            found_wanted += 1;
                            if (found_wanted >= wanted_count) goto bfs_done;
                        }
                    }
                    xmask = xm_drop;
                    if (!xmask) continue;  // all x positions locked for this action
                    y = ny;
                }

                // ---------------- X stage (DAS movement) ----------------
                int allow_move = 0;
                int hv = hv0;
                if (press_lr) {
                    hv = 0;
                    allow_move = 1;
                } else if (hold_dir_now != HOLD_NEUTRAL) {
                    hv = hv + 1;
                    if (hv >= HOR_ACCEL_SPEED) {
                        hv = HOR_RELOAD;
                        allow_move = 1;
                    }
                }

                typedef struct {
                    uint8_t xmask;
                    uint8_t hv;
                    int8_t dx;
                } Tmp;
                Tmp tmp[2];
                int ntmp = 0;

                if (!allow_move || hold_dir_now == HOLD_NEUTRAL) {
                    tmp[0].xmask = xmask;
                    tmp[0].hv = (uint8_t)(hv & 0x0F);
                    tmp[0].dx = 0;
                    ntmp = 1;
                } else if (hold_right) {
                    const uint8_t fits_row = fit_mask[(rot & 1) ? 1 : 0][y];
                    const uint8_t ok = (uint8_t)(fits_row >> 1);
                    const uint8_t movable = (uint8_t)(xmask & ok);
                    const uint8_t blocked = (uint8_t)(xmask & (uint8_t)(~ok));
                    if (movable) {
                        tmp[ntmp].xmask = (uint8_t)(movable << 1);
                        tmp[ntmp].hv = (uint8_t)(hv & 0x0F);
                        tmp[ntmp].dx = 1;
                        ntmp += 1;
                    }
                    if (blocked) {
                        tmp[ntmp].xmask = blocked;
                        tmp[ntmp].hv = (uint8_t)HOR_BLOCKED;
                        tmp[ntmp].dx = 0;
                        ntmp += 1;
                    }
                } else {
                    const uint8_t fits_row = fit_mask[(rot & 1) ? 1 : 0][y];
                    const uint8_t ok = (uint8_t)((fits_row << 1) & 0xFFu);
                    const uint8_t movable = (uint8_t)(xmask & ok);
                    const uint8_t blocked = (uint8_t)(xmask & (uint8_t)(~ok));
                    if (movable) {
                        tmp[ntmp].xmask = (uint8_t)(movable >> 1);
                        tmp[ntmp].hv = (uint8_t)(hv & 0x0F);
                        tmp[ntmp].dx = -1;
                        ntmp += 1;
                    }
                    if (blocked) {
                        tmp[ntmp].xmask = blocked;
                        tmp[ntmp].hv = (uint8_t)HOR_BLOCKED;
                        tmp[ntmp].dx = 0;
                        ntmp += 1;
                    }
                }

                // ---------------- Rotate stage ----------------
                const int rotation_pressed = (rotation != 0) && (rotation != rh_prev);
                const uint8_t p_next = (uint8_t)((parity0 ^ 1) & FAST_DROP_MASK);
                const uint8_t hd_next = (uint8_t)hold_dir_now;
                const uint8_t rh_next = (uint8_t)rotation;

                for (int ti = 0; ti < ntmp; ++ti) {
                    const uint8_t xm_in = tmp[ti].xmask;
                    const uint8_t hv_in = tmp[ti].hv;
                    const int8_t dx_in = tmp[ti].dx;

                    // Small list of output groups for this tmp bucket.
                    struct Out {
                        uint8_t xmask;
                        uint8_t rot;
                        uint8_t hv;
                        int8_t dx;
                    } outg[3];
                    int nout = 0;

                    if (!rotation_pressed) {
                        outg[0].xmask = xm_in;
                        outg[0].rot = (uint8_t)rot;
                        outg[0].hv = hv_in;
                        outg[0].dx = dx_in;
                        nout = 1;
                    } else {
                        int rot1 = rot;
                        if (rotation == 1) rot1 = (rot - 1) & 3;
                        else rot1 = (rot + 1) & 3;

                        if ((rot1 & 1) != 0) {
                            const uint8_t fit_v = fit_mask[1][y];
                            const uint8_t acc = (uint8_t)(xm_in & fit_v);
                            const uint8_t rej = (uint8_t)(xm_in & (uint8_t)(~fit_v));
                            if (acc) {
                                outg[nout].xmask = acc;
                                outg[nout].rot = (uint8_t)rot1;
                                outg[nout].hv = hv_in;
                                outg[nout].dx = dx_in;
                                nout += 1;
                            }
                            if (rej) {
                                outg[nout].xmask = rej;
                                outg[nout].rot = (uint8_t)rot;
                                outg[nout].hv = hv_in;
                                outg[nout].dx = dx_in;
                                nout += 1;
                            }
                        } else {
                            const uint8_t fit_h = fit_mask[0][y];
                            const uint8_t acc_inplace = (uint8_t)(xm_in & fit_h);
                            const uint8_t rej_inplace = (uint8_t)(xm_in & (uint8_t)(~fit_h));
                            const uint8_t ok_left = (uint8_t)((fit_h << 1) & 0xFFu);
                            const uint8_t dbl = hold_left ? (uint8_t)(acc_inplace & ok_left) : 0u;
                            const uint8_t acc_noshift = (uint8_t)(acc_inplace & (uint8_t)(~dbl));
                            const uint8_t kick = (uint8_t)(rej_inplace & ok_left);
                            const uint8_t rej = (uint8_t)(rej_inplace & (uint8_t)(~kick));
                            const uint8_t shifted_src = (uint8_t)(dbl | kick);
                            if (shifted_src) {
                                outg[nout].xmask = (uint8_t)(shifted_src >> 1);
                                outg[nout].rot = (uint8_t)rot1;
                                outg[nout].hv = hv_in;
                                outg[nout].dx = (int8_t)(dx_in - 1);
                                nout += 1;
                            }
                            if (acc_noshift) {
                                outg[nout].xmask = acc_noshift;
                                outg[nout].rot = (uint8_t)rot1;
                                outg[nout].hv = hv_in;
                                outg[nout].dx = dx_in;
                                nout += 1;
                            }
                            if (rej) {
                                outg[nout].xmask = rej;
                                outg[nout].rot = (uint8_t)rot;
                                outg[nout].hv = hv_in;
                                outg[nout].dx = dx_in;
                                nout += 1;
                            }
                        }
                    }

                    for (int oi = 0; oi < nout; ++oi) {
                        const uint8_t xm_out = outg[oi].xmask;
                        if (!xm_out) continue;
                        const uint8_t rot_out = outg[oi].rot;
                        const uint8_t hv_out = outg[oi].hv;
                        const int8_t dx_out = outg[oi].dx;

                        const uint32_t next_key = (uint32_t)(y & 15) + rot_term[(unsigned)(rot_out & 3)]
                                                  + sc_term[(unsigned)sc] + hv_term[(unsigned)(hv_out & 15u)]
                                                  + hd_term[(unsigned)hd_next] + p_term[(unsigned)p_next]
                                                  + rh_term[(unsigned)rh_next];
                        uint8_t seen = g_ctx.visited_xmask[next_key];
                        uint8_t new_bits = (uint8_t)(xm_out & (uint8_t)(~seen));
                        if (!new_bits) continue;
                        g_ctx.visited_xmask[next_key] = (uint8_t)(seen | new_bits);
                        visited_states += (uint32_t)__builtin_popcount((unsigned)new_bits);

                        const uint32_t next_full_base = next_key << 3;
                        uint8_t bits_mask = new_bits;
                        while (bits_mask) {
                            const int xo = __builtin_ctz((unsigned)bits_mask);
                            bits_mask &= (uint8_t)(bits_mask - 1u);
                            const int xp = xo - (int)dx_out;
                            if ((unsigned)xp >= (unsigned)GRID_W) continue;  // should not happen
                            const uint32_t child_full = next_full_base + (uint32_t)xo;
                            const uint32_t parent_full = cur_full_base + (uint32_t)xp;
                            g_ctx.parent[child_full] = parent_full;
                            g_ctx.parent_action[child_full] = (uint8_t)act;
                        }

                        const uint8_t prev_accum = g_ctx.next_xmask[next_key];
                        g_ctx.next_xmask[next_key] = (uint8_t)(prev_accum | new_bits);
                        if (prev_accum == 0) {
                            if (next_n >= nkeys) return -2;
                            next_frontier[next_n].key = next_key;
                            next_frontier[next_n].xmask = 0u;  // filled after aggregation
                            next_frontier[next_n].y = (uint8_t)(y & 15);
                            next_frontier[next_n].rot = (uint8_t)(rot_out & 3);
                            next_frontier[next_n].sc = (uint8_t)sc;
                            next_frontier[next_n].hv = (uint8_t)(hv_out & 15u);
                            next_frontier[next_n].hd = hd_next;
                            next_frontier[next_n].p = p_next;
                            next_frontier[next_n].rh = rh_next;
                            next_n += 1;
                            queue_nodes_enqueued += 1;
                        }
                    }
                }
            }
        }

        // Finalize next frontier: fill xmask from accumulator and clear.
        for (uint32_t ni = 0; ni < next_n; ++ni) {
            const uint32_t key = next_frontier[ni].key;
            next_frontier[ni].xmask = g_ctx.next_xmask[key];
            g_ctx.next_xmask[key] = 0u;
        }

        // Swap frontiers.
        NodeMask* tmp_ptr = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = tmp_ptr;
        cur_n = next_n;
    }

bfs_done:
    ;
    if (stats_enabled()) {
        g_last_stats.visited_states = visited_states;
        g_last_stats.expanded_states = expanded_states;
        g_last_stats.transitions = transitions;
        g_last_stats.locks_found = locks_found;
        g_last_stats.queue_nodes_enqueued = queue_nodes_enqueued;
        g_last_stats.queue_nodes_expanded = queue_nodes_expanded;
        g_last_stats.max_depth = (uint16_t)max_depth;
        g_last_stats.depth_processed = (uint16_t)depth_processed;
        g_last_stats.wanted_count = (uint16_t)(wanted_count < 0 ? 0 : wanted_count);
        g_last_stats.found_wanted = (uint16_t)(found_wanted < 0 ? 0 : found_wanted);
    }

    // Reconstruct scripts into caller-provided buffer (one script per reachable locked pose).
    int used = 0;
    for (int pose = 0; pose < 512; ++pose) {
        const uint16_t cost = out_costs[pose];
        if (cost == 0xFFFFu) continue;
        const int len = (int)cost;
        if (len <= 0) {
            out_costs[pose] = 0xFFFFu;
            continue;
        }
        if (used + len > script_buf_cap) {
            return -3;
        }
        out_offsets[pose] = (uint16_t)used;
        out_lengths[pose] = (uint16_t)len;

        int pos = used + len;
        out_script_buf[pos - 1] = term_parent_action[pose];
        pos -= 1;
        uint32_t cur_s = term_parent_state[pose];
        while (cur_s != UINT32_MAX) {
            const uint32_t pcur = g_ctx.parent[cur_s];
            if (pcur == UINT32_MAX) break;
            out_script_buf[pos - 1] = g_ctx.parent_action[cur_s];
            pos -= 1;
            cur_s = pcur;
        }
        if (pos != used) {
            // Inconsistency between stored cost and parent chain (should not happen).
            out_costs[pose] = 0xFFFFu;
            out_offsets[pose] = 0u;
            out_lengths[pose] = 0u;
            continue;
        }
        used += len;
    }
    *out_script_used = used;
    return 0;
}
