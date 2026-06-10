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

static _Thread_local DrmReachStats g_last_stats;
static _Thread_local int g_stats_enabled = -1;

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

static _Thread_local ReachCtx g_ctx = {0};

void drm_reach_free_thread_ctx(void) {
    if (
        g_ctx.visited_xmask == NULL
        && g_ctx.next_xmask == NULL
        && g_ctx.frontier_a == NULL
        && g_ctx.frontier_b == NULL
        && g_ctx.parent == NULL
        && g_ctx.parent_action == NULL
    ) {
        g_ctx.cap_keys = 0;
        g_ctx.cap_states = 0;
        return;
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
}

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

// ===========================================================================
// v2 BFS: exact state-space reduction + optional costs-only mode
// ===========================================================================
//
// Semantics are identical to drm_reach_bfs_full. The state space is reduced by
// one *exact* observation:
//
//   hor_velocity (hv) is semantically dead whenever hold_dir == NEUTRAL.
//   It is only ever read on a same-direction hold continuation (no edge), and
//   any edge press resets it to 0 before it is read. From a neutral-hold state
//   every lateral input is an edge press, so all hv values are equivalent.
//
// The v1 key space stores hv for every state (x16). v2 stores hv only for
// states with hold_dir != NEUTRAL:
//
//   micro m in [0, 33):  m == 0            -> hd = NEUTRAL (hv dropped)
//                        m in [1, 17)      -> hd = LEFT,  hv = m - 1
//                        m in [17, 33)     -> hd = RIGHT, hv = m - 17
//
//   key = y + 16*rot + 64*p + 128*rh + 384*m + 12672*sc
//   nkeys = 12672 * sc_range          (v1: 18432 * sc_range)
//
// Parity stays in the key: visited-dedup spans depths of both parities, and
// identical counter states at different parities behave differently (down-only
// soft-drop gating). It never widens a single depth's frontier, only memory.
//
// Costs-only mode: pass out_script_buf == NULL (and/or out_offsets == NULL) to
// skip all parent bookkeeping. out_costs is always filled.

typedef struct {
    uint32_t cap_keys;
    uint32_t cap_states;
    uint8_t* visited_xmask;
    uint8_t* next_xmask;
    NodeMask* frontier_a;   // reuse NodeMask; hv field stores micro m
    NodeMask* frontier_b;
    uint32_t* parent;       // allocated lazily, only for script mode
    uint8_t* parent_action;
    uint32_t cap_states_parent;
} ReachCtxV2;

static _Thread_local ReachCtxV2 g_ctx2 = {0};

void drm_reach_free_thread_ctx_v2(void) {
    free(g_ctx2.visited_xmask);
    free(g_ctx2.next_xmask);
    free(g_ctx2.frontier_a);
    free(g_ctx2.frontier_b);
    free(g_ctx2.parent);
    free(g_ctx2.parent_action);
    memset(&g_ctx2, 0, sizeof(g_ctx2));
}

static int ensure_ctx2(uint32_t nkeys, int want_parents) {
    if (nkeys == 0) return -2;
    const uint32_t nstates = nkeys * (uint32_t)GRID_W;
    if (g_ctx2.cap_keys < nkeys || !g_ctx2.visited_xmask) {
        free(g_ctx2.visited_xmask);
        free(g_ctx2.next_xmask);
        free(g_ctx2.frontier_a);
        free(g_ctx2.frontier_b);
        g_ctx2.visited_xmask = (uint8_t*)malloc((size_t)nkeys);
        g_ctx2.next_xmask = (uint8_t*)malloc((size_t)nkeys);
        g_ctx2.frontier_a = (NodeMask*)malloc((size_t)nkeys * sizeof(NodeMask));
        g_ctx2.frontier_b = (NodeMask*)malloc((size_t)nkeys * sizeof(NodeMask));
        if (!g_ctx2.visited_xmask || !g_ctx2.next_xmask || !g_ctx2.frontier_a || !g_ctx2.frontier_b) {
            drm_reach_free_thread_ctx_v2();
            return -2;
        }
        g_ctx2.cap_keys = nkeys;
        g_ctx2.cap_states = nstates;
    }
    if (want_parents && (g_ctx2.cap_states_parent < nstates || !g_ctx2.parent)) {
        free(g_ctx2.parent);
        free(g_ctx2.parent_action);
        g_ctx2.parent = (uint32_t*)malloc((size_t)nstates * sizeof(uint32_t));
        g_ctx2.parent_action = (uint8_t*)malloc((size_t)nstates);
        if (!g_ctx2.parent || !g_ctx2.parent_action) {
            drm_reach_free_thread_ctx_v2();
            return -2;
        }
        g_ctx2.cap_states_parent = nstates;
    }
    return 0;
}

// micro encoding helpers
enum { V2_MICRO_N = 33 };
static inline uint32_t v2_micro(uint32_t hd, uint32_t hv) {
    // hd==0 -> 0; hd==1 -> 1+hv; hd==2 -> 17+hv
    return hd == 0u ? 0u : (hd == 1u ? 1u + hv : 17u + hv);
}

int drm_reach_bfs_v2(
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
    const int want_scripts = (out_script_buf != NULL && out_offsets != NULL && out_lengths != NULL
                              && out_script_used != NULL && script_buf_cap > 0);
    if (!cols || !out_costs) return -1;
    if (max_frames <= 0) return -1;

    for (int i = 0; i < 512; ++i) out_costs[i] = 0xFFFFu;
    if (want_scripts) {
        for (int i = 0; i < 512; ++i) { out_offsets[i] = 0u; out_lengths[i] = 0u; }
        *out_script_used = 0;
    }

    if ((unsigned)sx >= (unsigned)GRID_W || (unsigned)sy >= (unsigned)GRID_H) return 0;

    if (speed_threshold < 0) speed_threshold = 0;
    if (speed_threshold > 0x7F) speed_threshold = 0x7F;
    const int sc_range = speed_threshold + 1;

    if (speed_counter < 0) speed_counter = 0;
    if (speed_counter > speed_threshold) speed_counter = speed_threshold;
    hor_velocity &= 0x0F;
    if (hold_dir < 0 || hold_dir > 2) hold_dir = 0;
    parity &= FAST_DROP_MASK;
    if (rot_hold < 0 || rot_hold > 2) rot_hold = 0;

    const int max_lock_frames = compute_max_lock_frames(sy, speed_counter, speed_threshold);
    if (max_frames > max_lock_frames) max_frames = max_lock_frames;

    uint8_t fit_mask[2][GRID_H];
    build_fit_masks(cols, fit_mask);
    if (!fits_masked(fit_mask, sx, sy, srot & 3)) return 0;

    // key = y + 16*rot + 64*p + 128*rh + 384*m + 12672*sc
    enum {
        K_ROT = GRID_H,            // 16
        K_P = K_ROT * 4,           // 64
        K_RH = K_P * 2,            // 128
        K_M = K_RH * 3,            // 384
        K_SC = K_M * V2_MICRO_N,   // 12672
    };
    const uint32_t nkeys = (uint32_t)K_SC * (uint32_t)sc_range;

    const int ctx_rc = ensure_ctx2(nkeys, want_scripts);
    if (ctx_rc != 0) return ctx_rc;

    memset(g_ctx2.visited_xmask, 0, (size_t)nkeys);
    memset(g_ctx2.next_xmask, 0, (size_t)nkeys);

    uint32_t term_parent_state[512];
    uint8_t term_parent_action[512];
    if (want_scripts) {
        for (int i = 0; i < 512; ++i) { term_parent_state[i] = UINT32_MAX; term_parent_action[i] = 0xFFu; }
    }

    const uint32_t start_m = v2_micro((uint32_t)hold_dir, (uint32_t)hor_velocity);
    const uint32_t start_key = (uint32_t)(sy & 15) + (uint32_t)K_ROT * (uint32_t)(srot & 3)
                               + (uint32_t)K_P * (uint32_t)parity + (uint32_t)K_RH * (uint32_t)rot_hold
                               + (uint32_t)K_M * start_m + (uint32_t)K_SC * (uint32_t)speed_counter;
    const uint8_t start_xmask = (uint8_t)(1u << (unsigned)(sx & 7));
    g_ctx2.visited_xmask[start_key] = start_xmask;
    if (want_scripts) {
        const uint32_t start_full = (start_key << 3) + (uint32_t)(sx & 7);
        g_ctx2.parent[start_full] = UINT32_MAX;
        g_ctx2.parent_action[start_full] = 0xFFu;
    }

    NodeMask* cur_frontier = g_ctx2.frontier_a;
    NodeMask* next_frontier = g_ctx2.frontier_b;
    uint32_t cur_n = 1, next_n = 0;
    cur_frontier[0].key = start_key;
    cur_frontier[0].xmask = start_xmask;
    cur_frontier[0].y = (uint8_t)(sy & 15);
    cur_frontier[0].rot = (uint8_t)(srot & 3);
    cur_frontier[0].sc = (uint8_t)speed_counter;
    cur_frontier[0].hv = (uint8_t)hor_velocity;  // valid only when hd != N
    cur_frontier[0].hd = (uint8_t)hold_dir;
    cur_frontier[0].p = (uint8_t)parity;
    cur_frontier[0].rh = (uint8_t)rot_hold;

    const uint16_t max_depth = (uint16_t)max_frames;

    uint32_t visited_states = 1, expanded_states = 0, transitions = 0, locks_found = 0;
    uint32_t queue_nodes_enqueued = 1, queue_nodes_expanded = 0;
    uint16_t depth_processed = 0;

    uint8_t wanted_pose[512];
    const int wanted_count = build_wanted_terminal_poses_reachable(
        fit_mask, (int)(sx & 7), (int)(sy & 15), (int)(srot & 3), wanted_pose
    );
    int found_wanted = 0;

    static const uint8_t ACTIONS_EVEN[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};
    static const uint8_t ACTIONS_ODD[9] = {0, 1, 2, 6, 7, 8, 12, 13, 14};

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
            const uint32_t cur_full_base = cur.key << 3;

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
                const int rh_prev = (int)cur.rh;

                const int hold_dir_now = (int)ACT_HOLD_DIR[act];
                const int hold_down = (int)ACT_HOLD_DOWN[act];
                const int rotation = (int)ACT_ROT[act];

                const int prev_left = (hd_prev == HOLD_LEFT);
                const int prev_right = (hd_prev == HOLD_RIGHT);
                const int hold_left = (hold_dir_now == HOLD_LEFT);
                const int hold_right = (hold_dir_now == HOLD_RIGHT);

                const int press_left = hold_left && !prev_left;
                const int press_right = hold_right && !prev_right;
                const int press_lr = press_left || press_right;

                // ---------------- Y stage ----------------
                const int down_only = (hold_down != 0) && (hold_dir_now == HOLD_NEUTRAL);
                int drop_triggered = 0;
                if ((parity_cur & FAST_DROP_MASK) == 0 && down_only) {
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
                        if (want_scripts) {
                            term_parent_state[pose] = cur_full_base + (uint32_t)lx;
                            term_parent_action[pose] = (uint8_t)act;
                        }
                        locks_found += 1;
                        if (wanted_pose[pose]) {
                            found_wanted += 1;
                            if (found_wanted >= wanted_count) goto bfs2_done;
                        }
                    }
                    xmask = xm_drop;
                    if (!xmask) continue;
                    y = ny;
                }

                // ---------------- X stage ----------------
                int allow_move = 0;
                int hv = hv0;
                if (press_lr) {
                    hv = 0;
                    allow_move = 1;
                } else if (hold_dir_now != HOLD_NEUTRAL) {
                    // Same-direction hold continuation: hv is live (hd_prev == hold_dir_now here,
                    // because a different prev dir would have been an edge press).
                    hv = hv + 1;
                    if (hv >= HOR_ACCEL_SPEED) {
                        hv = HOR_RELOAD;
                        allow_move = 1;
                    }
                }

                typedef struct { uint8_t xmask; uint8_t hv; int8_t dx; } Tmp;
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
                    if (movable) { tmp[ntmp].xmask = (uint8_t)(movable << 1); tmp[ntmp].hv = (uint8_t)(hv & 0x0F); tmp[ntmp].dx = 1; ntmp += 1; }
                    if (blocked) { tmp[ntmp].xmask = blocked; tmp[ntmp].hv = (uint8_t)HOR_BLOCKED; tmp[ntmp].dx = 0; ntmp += 1; }
                } else {
                    const uint8_t fits_row = fit_mask[(rot & 1) ? 1 : 0][y];
                    const uint8_t ok = (uint8_t)((fits_row << 1) & 0xFFu);
                    const uint8_t movable = (uint8_t)(xmask & ok);
                    const uint8_t blocked = (uint8_t)(xmask & (uint8_t)(~ok));
                    if (movable) { tmp[ntmp].xmask = (uint8_t)(movable >> 1); tmp[ntmp].hv = (uint8_t)(hv & 0x0F); tmp[ntmp].dx = -1; ntmp += 1; }
                    if (blocked) { tmp[ntmp].xmask = blocked; tmp[ntmp].hv = (uint8_t)HOR_BLOCKED; tmp[ntmp].dx = 0; ntmp += 1; }
                }

                // ---------------- Rotate stage ----------------
                const int rotation_pressed = (rotation != 0) && (rotation != rh_prev);
                const uint8_t p_next = (uint8_t)((parity_cur ^ 1) & FAST_DROP_MASK);
                const uint8_t hd_next = (uint8_t)hold_dir_now;
                const uint8_t rh_next = (uint8_t)rotation;

                for (int ti = 0; ti < ntmp; ++ti) {
                    const uint8_t xm_in = tmp[ti].xmask;
                    const uint8_t hv_in = tmp[ti].hv;
                    const int8_t dx_in = tmp[ti].dx;

                    struct Out { uint8_t xmask; uint8_t rot; uint8_t hv; int8_t dx; } outg[3];
                    int nout = 0;

                    if (!rotation_pressed) {
                        outg[0].xmask = xm_in; outg[0].rot = (uint8_t)rot; outg[0].hv = hv_in; outg[0].dx = dx_in;
                        nout = 1;
                    } else {
                        int rot1 = rot;
                        if (rotation == 1) rot1 = (rot - 1) & 3;
                        else rot1 = (rot + 1) & 3;

                        if ((rot1 & 1) != 0) {
                            const uint8_t fit_v = fit_mask[1][y];
                            const uint8_t acc = (uint8_t)(xm_in & fit_v);
                            const uint8_t rej = (uint8_t)(xm_in & (uint8_t)(~fit_v));
                            if (acc) { outg[nout].xmask = acc; outg[nout].rot = (uint8_t)rot1; outg[nout].hv = hv_in; outg[nout].dx = dx_in; nout += 1; }
                            if (rej) { outg[nout].xmask = rej; outg[nout].rot = (uint8_t)rot; outg[nout].hv = hv_in; outg[nout].dx = dx_in; nout += 1; }
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
                            if (shifted_src) { outg[nout].xmask = (uint8_t)(shifted_src >> 1); outg[nout].rot = (uint8_t)rot1; outg[nout].hv = hv_in; outg[nout].dx = (int8_t)(dx_in - 1); nout += 1; }
                            if (acc_noshift) { outg[nout].xmask = acc_noshift; outg[nout].rot = (uint8_t)rot1; outg[nout].hv = hv_in; outg[nout].dx = dx_in; nout += 1; }
                            if (rej) { outg[nout].xmask = rej; outg[nout].rot = (uint8_t)rot; outg[nout].hv = hv_in; outg[nout].dx = dx_in; nout += 1; }
                        }
                    }

                    for (int oi = 0; oi < nout; ++oi) {
                        const uint8_t xm_out = outg[oi].xmask;
                        if (!xm_out) continue;
                        const uint8_t rot_out = outg[oi].rot;
                        const uint8_t hv_out = outg[oi].hv;
                        const int8_t dx_out = outg[oi].dx;

                        const uint32_t m_out = v2_micro((uint32_t)hd_next, (uint32_t)(hv_out & 15u));
                        const uint32_t next_key = (uint32_t)(y & 15) + (uint32_t)K_ROT * (uint32_t)(rot_out & 3)
                                                  + (uint32_t)K_P * (uint32_t)p_next + (uint32_t)K_RH * (uint32_t)rh_next
                                                  + (uint32_t)K_M * m_out + (uint32_t)K_SC * (uint32_t)sc;
                        const uint8_t seen = g_ctx2.visited_xmask[next_key];
                        const uint8_t new_bits = (uint8_t)(xm_out & (uint8_t)(~seen));
                        if (!new_bits) continue;
                        g_ctx2.visited_xmask[next_key] = (uint8_t)(seen | new_bits);
                        visited_states += (uint32_t)__builtin_popcount((unsigned)new_bits);

                        if (want_scripts) {
                            const uint32_t next_full_base = next_key << 3;
                            uint8_t bits_mask = new_bits;
                            while (bits_mask) {
                                const int xo = __builtin_ctz((unsigned)bits_mask);
                                bits_mask &= (uint8_t)(bits_mask - 1u);
                                const int xp = xo - (int)dx_out;
                                if ((unsigned)xp >= (unsigned)GRID_W) continue;
                                g_ctx2.parent[next_full_base + (uint32_t)xo] = cur_full_base + (uint32_t)xp;
                                g_ctx2.parent_action[next_full_base + (uint32_t)xo] = (uint8_t)act;
                            }
                        }

                        const uint8_t prev_accum = g_ctx2.next_xmask[next_key];
                        g_ctx2.next_xmask[next_key] = (uint8_t)(prev_accum | new_bits);
                        if (prev_accum == 0) {
                            if (next_n >= nkeys) return -2;
                            next_frontier[next_n].key = next_key;
                            next_frontier[next_n].xmask = 0u;
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

        for (uint32_t ni = 0; ni < next_n; ++ni) {
            const uint32_t key = next_frontier[ni].key;
            next_frontier[ni].xmask = g_ctx2.next_xmask[key];
            g_ctx2.next_xmask[key] = 0u;
        }
        NodeMask* tmp_ptr = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = tmp_ptr;
        cur_n = next_n;
    }

bfs2_done:
    ;
    if (stats_enabled()) {
        g_last_stats.visited_states = visited_states;
        g_last_stats.expanded_states = expanded_states;
        g_last_stats.transitions = transitions;
        g_last_stats.locks_found = locks_found;
        g_last_stats.queue_nodes_enqueued = queue_nodes_enqueued;
        g_last_stats.queue_nodes_expanded = queue_nodes_expanded;
        g_last_stats.max_depth = max_depth;
        g_last_stats.depth_processed = depth_processed;
        g_last_stats.wanted_count = (uint16_t)(wanted_count < 0 ? 0 : wanted_count);
        g_last_stats.found_wanted = (uint16_t)(found_wanted < 0 ? 0 : found_wanted);
    }

    if (!want_scripts) return 0;

    int used = 0;
    for (int pose = 0; pose < 512; ++pose) {
        const uint16_t cost = out_costs[pose];
        if (cost == 0xFFFFu) continue;
        const int len = (int)cost;
        if (len <= 0) { out_costs[pose] = 0xFFFFu; continue; }
        if (used + len > script_buf_cap) return -3;
        out_offsets[pose] = (uint16_t)used;
        out_lengths[pose] = (uint16_t)len;

        int pos = used + len;
        out_script_buf[pos - 1] = term_parent_action[pose];
        pos -= 1;
        uint32_t cur_s = term_parent_state[pose];
        while (cur_s != UINT32_MAX) {
            const uint32_t pcur = g_ctx2.parent[cur_s];
            if (pcur == UINT32_MAX) break;
            out_script_buf[pos - 1] = g_ctx2.parent_action[cur_s];
            pos -= 1;
            cur_s = pcur;
        }
        if (pos != used) {
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

// ===========================================================================
// v3 BFS: bit-sliced over (x × speed_counter), costs-only
// ===========================================================================
//
// Exact same per-frame semantics as v1/v2. Observation: both `x` and
// `speed_counter` factor out of the transition function:
//   - gravity depends only on sc (and parity/down), uniformly across x
//   - lateral movement / rotation / collision depend only on x, uniformly
//     across sc
// So a node carries an 8-lane (x) by 128-bit (sc) occupancy block and one
// frame expansion processes all (x, sc) combinations of a
// (y, rot, hd, hv, rh) key at once:
//   - gravity: per-lane 128-bit shift left by one; the bit at `thr` drops
//     out and re-enters at bit 0 one row lower
//   - down-only soft drop: whole lane collapses to bit 0 one row lower
//   - lateral move: lane permute with collision-partitioned x masks
//   - rotation: x-mask partitions, sc lanes carried through
//
// Key space: (y:16, rot:4, micro:33, rh:3, p:2) = 12,672 keys, each a 128-byte
// block. micro encodes hd/hv exactly as v2 (hv live only when hd != NEUTRAL).
// Visited blocks are cleared via a touched-key list (no full memset).
//
// Costs-only: no parent pointers, no scripts. Training consumes feasibility +
// cost; controller scripts come from v1/v2 when needed (parity tools).

typedef struct {
    uint64_t lo[8];  // lane x -> sc bits 0..63
    uint64_t hi[8];  // lane x -> sc bits 64..127
} V3Block;

typedef struct {
    uint32_t key;    // y + 16*rot + 64*micro + 2112*rh (parity handled separately)
    uint8_t y, rot, hv, hd, rh;
} V3NodeMeta;

enum {
    V3_K_ROT = GRID_H,                 // 16
    V3_K_M = V3_K_ROT * 4,             // 64
    V3_K_RH = V3_K_M * V2_MICRO_N,     // 2112
    V3_NKEYS = V3_K_RH * 3,            // 6336 (per parity)
};

typedef struct {
    V3Block* visited;      // [2][V3_NKEYS] blocks (parity-major)
    V3Block* accum;        // [V3_NKEYS] next-frontier accumulator
    uint32_t* touched_v;   // visited keys touched (idx incl. parity bit)
    uint32_t n_touched_v;
    uint32_t* in_accum;    // accum keys in next frontier (dedup flags by stamp)
    V3NodeMeta* frontier_a;
    V3NodeMeta* frontier_b;
    V3Block* fblk_a;
    V3Block* fblk_b;
    uint8_t* accum_flag;
    int initialized;
} V3Ctx;

static _Thread_local V3Ctx g_ctx3 = {0};

void drm_reach_free_thread_ctx_v3(void) {
    free(g_ctx3.visited);
    free(g_ctx3.accum);
    free(g_ctx3.touched_v);
    free(g_ctx3.in_accum);
    free(g_ctx3.frontier_a);
    free(g_ctx3.frontier_b);
    free(g_ctx3.fblk_a);
    free(g_ctx3.fblk_b);
    free(g_ctx3.accum_flag);
    memset(&g_ctx3, 0, sizeof(g_ctx3));
}

static int ensure_ctx3(void) {
    if (g_ctx3.initialized) return 0;
    g_ctx3.visited = (V3Block*)calloc((size_t)V3_NKEYS * 2u, sizeof(V3Block));
    g_ctx3.accum = (V3Block*)calloc((size_t)V3_NKEYS, sizeof(V3Block));
    g_ctx3.touched_v = (uint32_t*)malloc((size_t)V3_NKEYS * 2u * sizeof(uint32_t));
    g_ctx3.in_accum = (uint32_t*)malloc((size_t)V3_NKEYS * sizeof(uint32_t));
    g_ctx3.frontier_a = (V3NodeMeta*)malloc((size_t)V3_NKEYS * sizeof(V3NodeMeta));
    g_ctx3.frontier_b = (V3NodeMeta*)malloc((size_t)V3_NKEYS * sizeof(V3NodeMeta));
    g_ctx3.fblk_a = (V3Block*)malloc((size_t)V3_NKEYS * sizeof(V3Block));
    g_ctx3.fblk_b = (V3Block*)malloc((size_t)V3_NKEYS * sizeof(V3Block));
    g_ctx3.accum_flag = (uint8_t*)calloc((size_t)V3_NKEYS, 1u);
    if (!g_ctx3.visited || !g_ctx3.accum || !g_ctx3.touched_v || !g_ctx3.in_accum
        || !g_ctx3.frontier_a || !g_ctx3.frontier_b || !g_ctx3.fblk_a || !g_ctx3.fblk_b
        || !g_ctx3.accum_flag) {
        drm_reach_free_thread_ctx_v3();
        return -2;
    }
    g_ctx3.n_touched_v = 0;
    g_ctx3.initialized = 1;
    return 0;
}

static inline int v3_blk_empty(const V3Block* b) {
    uint64_t acc = 0;
    for (int x = 0; x < 8; ++x) acc |= b->lo[x] | b->hi[x];
    return acc == 0;
}

static inline uint8_t v3_blk_xmask(const V3Block* b) {
    uint8_t m = 0;
    for (int x = 0; x < 8; ++x) m |= (uint8_t)(((b->lo[x] | b->hi[x]) != 0) << x);
    return m;
}

int drm_reach_bfs_v3(
    const uint16_t cols[GRID_W],
    int sx, int sy, int srot,
    int speed_counter, int hor_velocity,
    int hold_dir, int parity, int rot_hold,
    int speed_threshold,
    int max_frames,
    uint16_t out_costs[512]
) {
    if (!cols || !out_costs) return -1;
    if (max_frames <= 0) return -1;

    for (int i = 0; i < 512; ++i) out_costs[i] = 0xFFFFu;

    if ((unsigned)sx >= (unsigned)GRID_W || (unsigned)sy >= (unsigned)GRID_H) return 0;

    if (speed_threshold < 0) speed_threshold = 0;
    if (speed_threshold > 0x7F) speed_threshold = 0x7F;

    if (speed_counter < 0) speed_counter = 0;
    if (speed_counter > speed_threshold) speed_counter = speed_threshold;
    hor_velocity &= 0x0F;
    if (hold_dir < 0 || hold_dir > 2) hold_dir = 0;
    parity &= FAST_DROP_MASK;
    if (rot_hold < 0 || rot_hold > 2) rot_hold = 0;

    const int max_lock_frames = compute_max_lock_frames(sy, speed_counter, speed_threshold);
    if (max_frames > max_lock_frames) max_frames = max_lock_frames;

    uint8_t fit_mask[2][GRID_H];
    build_fit_masks(cols, fit_mask);
    if (!fits_masked(fit_mask, sx, sy, srot & 3)) return 0;

    const int rc_ctx = ensure_ctx3();
    if (rc_ctx != 0) return rc_ctx;

    // sc-shift bookkeeping: bit `thr` shifting out means a gravity drop.
    const int thr = speed_threshold;
    // Mask of valid sc bits 0..thr.
    uint64_t thr_keep_lo, thr_keep_hi;
    if (thr < 63) { thr_keep_lo = (1ull << (thr + 1)) - 1ull; thr_keep_hi = 0; }
    else if (thr == 63) { thr_keep_lo = ~0ull; thr_keep_hi = 0; }
    else if (thr < 127) { thr_keep_lo = ~0ull; thr_keep_hi = (1ull << (thr - 63)) - 1ull; }
    else { thr_keep_lo = ~0ull; thr_keep_hi = ~0ull; }

    uint8_t wanted_pose[512];
    const int wanted_count = build_wanted_terminal_poses_reachable(
        fit_mask, sx, sy, srot & 3, wanted_pose);
    int found_wanted = 0;

    // Seed.
    V3NodeMeta* cur_meta = g_ctx3.frontier_a;
    V3NodeMeta* next_meta = g_ctx3.frontier_b;
    V3Block* cur_blk = g_ctx3.fblk_a;
    V3Block* next_blk = g_ctx3.fblk_b;
    uint32_t cur_n = 0, next_n = 0;

    const uint32_t start_m = v2_micro((uint32_t)hold_dir, (uint32_t)hor_velocity);
    const uint32_t start_key = (uint32_t)(sy & 15) + (uint32_t)V3_K_ROT * (uint32_t)(srot & 3)
                               + (uint32_t)V3_K_M * start_m + (uint32_t)V3_K_RH * (uint32_t)rot_hold;

    cur_meta[0].key = start_key;
    cur_meta[0].y = (uint8_t)(sy & 15);
    cur_meta[0].rot = (uint8_t)(srot & 3);
    cur_meta[0].hv = (uint8_t)hor_velocity;
    cur_meta[0].hd = (uint8_t)hold_dir;
    cur_meta[0].rh = (uint8_t)rot_hold;
    memset(&cur_blk[0], 0, sizeof(V3Block));
    if (speed_counter < 64) cur_blk[0].lo[sx & 7] = 1ull << speed_counter;
    else cur_blk[0].hi[sx & 7] = 1ull << (speed_counter - 64);
    cur_n = 1;

    {
        const uint32_t vidx = ((uint32_t)parity * (uint32_t)V3_NKEYS) + start_key;
        g_ctx3.visited[vidx] = cur_blk[0];
        g_ctx3.touched_v[g_ctx3.n_touched_v++] = vidx;
    }

    uint32_t stat_nodes = 0, stat_groups = 0;
    uint16_t depth_processed = 0;

    int p_cur = parity;
    const uint16_t max_depth = (uint16_t)max_frames;

    for (uint16_t cur_depth = 0; cur_depth < max_depth && cur_n > 0; ++cur_depth) {
        depth_processed = cur_depth;
        stat_nodes += cur_n;
        next_n = 0;
        const uint16_t next_depth = (uint16_t)(cur_depth + 1);
        const int p_next = p_cur ^ 1;
        const int even_parity = ((p_cur & FAST_DROP_MASK) == 0);

        for (uint32_t ni = 0; ni < cur_n; ++ni) {
            const V3NodeMeta nm = cur_meta[ni];
            const V3Block* blk = &cur_blk[ni];
            const int y = (int)nm.y;
            const int rot = (int)nm.rot & 3;
            const int hv0 = (int)nm.hv & 0x0F;
            const int hd_prev = (int)nm.hd;
            const int rh_prev = (int)nm.rh;
            const int vparity = rot & 1;

            // ---- Y stage variants (shared across actions) ----
            // Variant A (no down-only): sc increments; bit thr drops out.
            // Variant B (down-only, even parity only): every state drops, sc -> 0.
            //
            // For each variant: stay-block (same y) and drop fate at y+1
            // partitioned by collision into fallen lanes (sc=0) and locks.
            const uint8_t drop_ok = (y + 1 < GRID_H) ? fit_mask[vparity][y + 1] : 0u;

            // Variant A:
            V3Block stayA;
            uint8_t dropA_x = 0;   // lanes that had bit thr set (drop attempt)
            for (int x = 0; x < 8; ++x) {
                uint64_t lo = blk->lo[x], hi = blk->hi[x];
                int dropped;
                if (thr < 64) dropped = (int)((lo >> thr) & 1ull);
                else dropped = (int)((hi >> (thr - 64)) & 1ull);
                dropA_x |= (uint8_t)(dropped << x);
                // shift left by one across 128 bits, then mask to 0..thr
                uint64_t nlo = (lo << 1);
                uint64_t nhi = (hi << 1) | (lo >> 63);
                stayA.lo[x] = nlo & thr_keep_lo;
                stayA.hi[x] = nhi & thr_keep_hi;
            }
            const uint8_t lockA_x = (uint8_t)(dropA_x & (uint8_t)(~drop_ok));
            const uint8_t fallA_x = (uint8_t)(dropA_x & drop_ok);
            const uint8_t stayA_x = v3_blk_xmask(&stayA);

            // Variant B (only if even parity): all lanes drop.
            const uint8_t allx = v3_blk_xmask(blk);
            const uint8_t lockB_x = (uint8_t)(allx & (uint8_t)(~drop_ok));
            const uint8_t fallB_x = (uint8_t)(allx & drop_ok);

            // Process the 9 (dir, rot) combos plus down variants, mirroring the
            // v1/v2 action tables exactly:
            //   even parity: actions {0..8, 12..14}  (NEUTRAL down variants live)
            //   odd parity:  actions {0..2, 6..8, 12..14}
            const int n_act = even_parity ? 12 : 9;
            static const uint8_t ACTS_E[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};
            static const uint8_t ACTS_O[9] = {0, 1, 2, 6, 7, 8, 12, 13, 14};
            const uint8_t* acts = even_parity ? ACTS_E : ACTS_O;

            for (int ai = 0; ai < n_act; ++ai) {
                const int act = (int)acts[ai];
                const int dir = (int)ACT_HOLD_DIR[act];
                const int dn = (int)ACT_HOLD_DOWN[act];
                const int rotation = (int)ACT_ROT[act];

                const int down_only = (dn != 0) && (dir == HOLD_NEUTRAL);
                const int use_B = even_parity && down_only;

                // Y-stage result for this action:
                //   groups[g]: y_out, blockptr (stay or fall), lock xmask handled once
                // Locks: record now (first-found wins).
                uint8_t lock_x = use_B ? lockB_x : lockA_x;
                while (lock_x) {
                    const int lx = __builtin_ctz((unsigned)lock_x);
                    lock_x &= (uint8_t)(lock_x - 1u);
                    const int pose = pose_index(lx, y, rot);
                    if (out_costs[pose] != 0xFFFFu) continue;
                    out_costs[pose] = next_depth;
                    if (wanted_pose[pose]) {
                        found_wanted += 1;
                        if (found_wanted >= wanted_count) goto v3_done;
                    }
                }

                // Build the two Y-groups for this action.
                // G0: stay at y (variant A only), block = stayA restricted later by x ops
                // G1: fall to y+1, lanes fallX with sc=0
                struct YG { int y; uint8_t xm; int sc_zero; } ygs[2];
                int n_yg = 0;
                if (!use_B) {
                    if (stayA_x) { ygs[n_yg].y = y; ygs[n_yg].xm = stayA_x; ygs[n_yg].sc_zero = 0; n_yg++; }
                    if (fallA_x) { ygs[n_yg].y = y + 1; ygs[n_yg].xm = fallA_x; ygs[n_yg].sc_zero = 1; n_yg++; }
                } else {
                    if (fallB_x) { ygs[n_yg].y = y + 1; ygs[n_yg].xm = fallB_x; ygs[n_yg].sc_zero = 1; n_yg++; }
                }

                const int prev_left = (hd_prev == HOLD_LEFT);
                const int prev_right = (hd_prev == HOLD_RIGHT);
                const int hold_left = (dir == HOLD_LEFT);
                const int hold_right = (dir == HOLD_RIGHT);
                const int press_lr = (hold_left && !prev_left) || (hold_right && !prev_right);

                int allow_move = 0;
                int hv = hv0;
                if (press_lr) { hv = 0; allow_move = 1; }
                else if (dir != HOLD_NEUTRAL) {
                    hv = hv + 1;
                    if (hv >= HOR_ACCEL_SPEED) { hv = HOR_RELOAD; allow_move = 1; }
                }

                const int rotation_pressed = (rotation != 0) && (rotation != rh_prev);

                for (int gi = 0; gi < n_yg; ++gi) {
                    const int gy = ygs[gi].y;
                    const uint8_t g_xm = ygs[gi].xm;
                    const int g_sc0 = ygs[gi].sc_zero;
                    const int gvp = vparity;  // rot unchanged until rotate stage

                    // X stage on xmask level: ≤2 sub-groups (moved / blocked)
                    struct XG { uint8_t xm; int8_t dx; uint8_t hv; } xgs[2];
                    int n_xg = 0;
                    if (!allow_move || dir == HOLD_NEUTRAL) {
                        xgs[0].xm = g_xm; xgs[0].dx = 0; xgs[0].hv = (uint8_t)(hv & 0x0F);
                        n_xg = 1;
                    } else if (hold_right) {
                        const uint8_t fits_row = fit_mask[gvp][gy];
                        const uint8_t ok = (uint8_t)(fits_row >> 1);
                        const uint8_t mv = (uint8_t)(g_xm & ok);
                        const uint8_t bl = (uint8_t)(g_xm & (uint8_t)(~ok));
                        if (mv) { xgs[n_xg].xm = mv; xgs[n_xg].dx = 1; xgs[n_xg].hv = (uint8_t)(hv & 0x0F); n_xg++; }
                        if (bl) { xgs[n_xg].xm = bl; xgs[n_xg].dx = 0; xgs[n_xg].hv = (uint8_t)HOR_BLOCKED; n_xg++; }
                    } else {
                        const uint8_t fits_row = fit_mask[gvp][gy];
                        const uint8_t ok = (uint8_t)((fits_row << 1) & 0xFFu);
                        const uint8_t mv = (uint8_t)(g_xm & ok);
                        const uint8_t bl = (uint8_t)(g_xm & (uint8_t)(~ok));
                        if (mv) { xgs[n_xg].xm = mv; xgs[n_xg].dx = -1; xgs[n_xg].hv = (uint8_t)(hv & 0x0F); n_xg++; }
                        if (bl) { xgs[n_xg].xm = bl; xgs[n_xg].dx = 0; xgs[n_xg].hv = (uint8_t)HOR_BLOCKED; n_xg++; }
                    }

                    for (int xi = 0; xi < n_xg; ++xi) {
                        const uint8_t xm_pre = xgs[xi].xm;       // x positions BEFORE dx
                        const int8_t dx = xgs[xi].dx;
                        const uint8_t hv_out = xgs[xi].hv;
                        const uint8_t xm_moved = (dx == 1) ? (uint8_t)(xm_pre << 1)
                                              : (dx == -1) ? (uint8_t)(xm_pre >> 1)
                                              : xm_pre;

                        // Rotation stage: ≤3 sub-groups on the post-move xmask.
                        struct RG { uint8_t xm; uint8_t rot; int8_t dx2; } rgs[3];
                        int n_rg = 0;
                        if (!rotation_pressed) {
                            rgs[0].xm = xm_moved; rgs[0].rot = (uint8_t)rot; rgs[0].dx2 = 0;
                            n_rg = 1;
                        } else {
                            int rot1 = rot;
                            if (rotation == 1) rot1 = (rot - 1) & 3;
                            else rot1 = (rot + 1) & 3;
                            if ((rot1 & 1) != 0) {
                                const uint8_t fit_v = fit_mask[1][gy];
                                const uint8_t acc = (uint8_t)(xm_moved & fit_v);
                                const uint8_t rej = (uint8_t)(xm_moved & (uint8_t)(~fit_v));
                                if (acc) { rgs[n_rg].xm = acc; rgs[n_rg].rot = (uint8_t)rot1; rgs[n_rg].dx2 = 0; n_rg++; }
                                if (rej) { rgs[n_rg].xm = rej; rgs[n_rg].rot = (uint8_t)rot; rgs[n_rg].dx2 = 0; n_rg++; }
                            } else {
                                const uint8_t fit_h = fit_mask[0][gy];
                                const uint8_t acc_inplace = (uint8_t)(xm_moved & fit_h);
                                const uint8_t rej_inplace = (uint8_t)(xm_moved & (uint8_t)(~fit_h));
                                const uint8_t ok_left = (uint8_t)((fit_h << 1) & 0xFFu);
                                const uint8_t dbl = hold_left ? (uint8_t)(acc_inplace & ok_left) : 0u;
                                const uint8_t acc_noshift = (uint8_t)(acc_inplace & (uint8_t)(~dbl));
                                const uint8_t kick = (uint8_t)(rej_inplace & ok_left);
                                const uint8_t rej = (uint8_t)(rej_inplace & (uint8_t)(~kick));
                                const uint8_t shifted_src = (uint8_t)(dbl | kick);
                                if (shifted_src) { rgs[n_rg].xm = (uint8_t)(shifted_src >> 1); rgs[n_rg].rot = (uint8_t)rot1; rgs[n_rg].dx2 = -1; n_rg++; }
                                if (acc_noshift) { rgs[n_rg].xm = acc_noshift; rgs[n_rg].rot = (uint8_t)rot1; rgs[n_rg].dx2 = 0; n_rg++; }
                                if (rej) { rgs[n_rg].xm = rej; rgs[n_rg].rot = (uint8_t)rot; rgs[n_rg].dx2 = 0; n_rg++; }
                            }
                        }

                        for (int ri = 0; ri < n_rg; ++ri) {
                            const uint8_t xm_fin = rgs[ri].xm;
                            if (!xm_fin) continue;
                            const uint8_t rot_fin = rgs[ri].rot;
                            const int total_dx = (int)dx + (int)rgs[ri].dx2;

                            const uint32_t m_out = v2_micro((uint32_t)dir, (uint32_t)(hv_out & 15u));
                            const uint32_t key2 = (uint32_t)(gy & 15)
                                                  + (uint32_t)V3_K_ROT * (uint32_t)(rot_fin & 3)
                                                  + (uint32_t)V3_K_M * m_out
                                                  + (uint32_t)V3_K_RH * (uint32_t)rotation;
                            const uint32_t vidx = ((uint32_t)p_next * (uint32_t)V3_NKEYS) + key2;
                            V3Block* vis = &g_ctx3.visited[vidx];
                            V3Block* acc = &g_ctx3.accum[key2];
                            stat_groups += 1;

                            // Source block for this group: stayA or the
                            // sc-zero drop pattern, with lanes selected by the
                            // PRE-move x positions, then shifted by total_dx.
                            uint64_t any_new = 0;
                            const int sc_zero = ygs[gi].sc_zero;
                            uint8_t xm_iter = xm_fin;
                            const int first_touch_v = v3_blk_empty(vis);
                            int wrote = 0;
                            while (xm_iter) {
                                const int xo = __builtin_ctz((unsigned)xm_iter);
                                xm_iter &= (uint8_t)(xm_iter - 1u);
                                const int xs = xo - total_dx;
                                if ((unsigned)xs >= 8u) continue;
                                uint64_t slo, shi;
                                if (sc_zero) { slo = 1ull; shi = 0ull; }
                                else { slo = stayA.lo[xs]; shi = stayA.hi[xs]; }
                                const uint64_t nlo = slo & ~vis->lo[xo];
                                const uint64_t nhi = shi & ~vis->hi[xo];
                                if (!(nlo | nhi)) continue;
                                vis->lo[xo] |= nlo;
                                vis->hi[xo] |= nhi;
                                acc->lo[xo] |= nlo;
                                acc->hi[xo] |= nhi;
                                any_new |= nlo | nhi;
                                wrote = 1;
                            }
                            if (!any_new) continue;
                            if (first_touch_v) g_ctx3.touched_v[g_ctx3.n_touched_v++] = vidx;
                            (void)wrote;
                            if (!g_ctx3.accum_flag[key2]) {
                                g_ctx3.accum_flag[key2] = 1u;
                                next_meta[next_n].key = key2;
                                next_meta[next_n].y = (uint8_t)(gy & 15);
                                next_meta[next_n].rot = (uint8_t)(rot_fin & 3);
                                next_meta[next_n].hv = (uint8_t)(hv_out & 15u);
                                next_meta[next_n].hd = (uint8_t)dir;
                                next_meta[next_n].rh = (uint8_t)rotation;
                                next_n += 1;
                            }
                        }
                    }
                }
            }
        }

        // Materialize next frontier blocks and clear accumulators.
        for (uint32_t ni = 0; ni < next_n; ++ni) {
            const uint32_t key = next_meta[ni].key;
            next_blk[ni] = g_ctx3.accum[key];
            memset(&g_ctx3.accum[key], 0, sizeof(V3Block));
            g_ctx3.accum_flag[key] = 0u;
        }

        V3NodeMeta* tm = cur_meta; cur_meta = next_meta; next_meta = tm;
        V3Block* tb = cur_blk; cur_blk = next_blk; next_blk = tb;
        cur_n = next_n;
        p_cur = p_next;
    }

v3_done:
    ;
    // An early exit can leave pending accumulator blocks for the (never
    // materialized) next frontier; clear them or subsequent calls inherit
    // garbage state.
    for (uint32_t ni = 0; ni < next_n; ++ni) {
        const uint32_t key = next_meta[ni].key;
        memset(&g_ctx3.accum[key], 0, sizeof(V3Block));
        g_ctx3.accum_flag[key] = 0u;
    }
    if (stats_enabled()) {
        g_last_stats.visited_states = 0;
        g_last_stats.expanded_states = 0;
        g_last_stats.transitions = stat_groups;
        g_last_stats.locks_found = 0;
        g_last_stats.queue_nodes_enqueued = stat_nodes;
        g_last_stats.queue_nodes_expanded = stat_nodes;
        g_last_stats.max_depth = (uint16_t)max_frames;
        g_last_stats.depth_processed = depth_processed;
        g_last_stats.wanted_count = (uint16_t)(wanted_count < 0 ? 0 : wanted_count);
        g_last_stats.found_wanted = (uint16_t)(found_wanted < 0 ? 0 : found_wanted);
    }

    // Clear touched visited blocks for the next call.
    for (uint32_t i = 0; i < g_ctx3.n_touched_v; ++i) {
        memset(&g_ctx3.visited[g_ctx3.touched_v[i]], 0, sizeof(V3Block));
    }
    g_ctx3.n_touched_v = 0;

    return 0;
}

// ===========================================================================
// v4 BFS: greedy-witness upper bounds + admissible lower-bound pruning
// ===========================================================================
//
// Exact costs, same semantics as v1/v2. Strategy:
//   1. For each geometrically-wanted pose, synthesize a few greedy controller
//      scripts (rotate + lateral taps, then soft drop) and simulate them with
//      an exact single-state stepper. A script that locks at the target pose
//      yields an achievable upper bound UB(pose).
//   2. Run the v2-style BFS, but prune any state that provably cannot improve
//      on ANY unresolved pose's UB. The lower bound used is vertical-only and
//      admissible: reaching a pose at row py from row y needs at least
//      2*(py-y)-1 frames when thr>0 (soft drop fires every other frame), and
//      (py-y) when thr==0. The per-state gate reduces to
//          depth < A + 2*y    (A maintained incrementally)
//      which is O(1) per node.
//   3. When all wanted poses are found, or the gated frontier dies out, any
//      pose still unresolved takes cost = UB(pose): any strictly better path
//      would have survived the gate, so UB is minimal.
//
// Costs-only (scripts come from v1 when needed).

typedef struct {
    int x, y, rot;
    int sc, hv, hd, p, rh;
    int locked;       // set when a drop fails; (x,y,rot) is the lock pose
} V4State;

// Exact single-state per-frame step, mirroring the v1 transition order
// (Y -> X -> rotate). `act` is the 18-action index.
static void v4_step(
    const uint8_t fit_mask[2][GRID_H], int thr, V4State* s, int act
) {
    const int dir = (int)ACT_HOLD_DIR[act];
    const int dn = (int)ACT_HOLD_DOWN[act];
    const int rotation = (int)ACT_ROT[act];

    const int prev_left = (s->hd == HOLD_LEFT);
    const int prev_right = (s->hd == HOLD_RIGHT);
    const int hold_left = (dir == HOLD_LEFT);
    const int hold_right = (dir == HOLD_RIGHT);
    const int press_lr = (hold_left && !prev_left) || (hold_right && !prev_right);

    // Y stage
    const int down_only = dn && (dir == HOLD_NEUTRAL);
    int drop = 0;
    if ((s->p & FAST_DROP_MASK) == 0 && down_only) {
        drop = 1;
        s->sc = 0;
    } else {
        s->sc += 1;
        if (s->sc > thr) { drop = 1; s->sc = 0; }
    }
    if (drop) {
        const int ny = s->y + 1;
        if (ny >= GRID_H || !fits_masked(fit_mask, s->x, ny, s->rot)) {
            s->locked = 1;
            // lock pose is the pre-drop pose; finalize hd/rh/p for completeness
            s->hd = dir;
            s->rh = rotation;
            s->p ^= 1;
            return;
        }
        s->y = ny;
    }

    // X stage
    int allow_move = 0;
    if (press_lr) {
        s->hv = 0;
        allow_move = 1;
    } else if (dir != HOLD_NEUTRAL) {
        s->hv += 1;
        if (s->hv >= HOR_ACCEL_SPEED) { s->hv = HOR_RELOAD; allow_move = 1; }
    }
    if (allow_move && dir != HOLD_NEUTRAL) {
        const int nx = s->x + (hold_right ? 1 : -1);
        if (fits_masked(fit_mask, nx, s->y, s->rot)) s->x = nx;
        else s->hv = HOR_BLOCKED;
    }

    // Rotate stage
    const int rotation_pressed = (rotation != 0) && (rotation != s->rh);
    if (rotation_pressed) {
        int rx = s->x;
        int rrot = s->rot;
        apply_rotation_masked(fit_mask, &rx, s->y, &rrot, rotation, hold_left);
        s->x = rx;
        s->rot = rrot;
    }

    s->hd = dir;
    s->rh = rotation;
    s->p ^= 1;
}

// Action helpers (18-action indices): dir*6 + down*3 + rot
static inline int v4_act(int dir, int dn, int rot) { return dir * 6 + dn * 3 + rot; }

// Simulate a greedy plan toward (px, py, prot): perform rotations and lateral
// taps (interleaved, edge-respecting), then soft-drop. Returns frames to lock
// if the script locks exactly at the target pose, else -1.
static int v4_greedy_try(
    const uint8_t fit_mask[2][GRID_H], int thr,
    const V4State* spawn, int px, int py, int prot,
    int order  // 0: rotate-first, 1: move-first, 2: interleave
) {
    V4State s = *spawn;
    int frames = 0;
    const int max_frames_guard = 4 * (GRID_H + 1) * (thr + 2) + 64;

    // Orders 4/5 mirror 2/3 with one leading soft-drop frame: shifts the
    // tap/down parity phase, which often saves a frame on the final approach.
    if (order >= 4) {
        v4_step(fit_mask, thr, &s, v4_act(HOLD_NEUTRAL, 1, 0));
        frames += 1;
        if (s.locked) {
            return (s.x == px && s.y == py && ((s.rot & 3) == (prot & 3))) ? frames : -1;
        }
        if (s.y > py) return -1;
        order -= 2;
    }

    // NES rotation: A (=1) decrements rot, B (=2) increments. CW presses
    // needed to reach prot from srot: (srot - prot) & 3; 3 CW = 1 CCW.
    int rots_needed = (s.rot - prot) & 3;
    int rot_btn = 1;  // A / CW (decrement)
    int nrots = rots_needed;
    if (rots_needed == 3) { rot_btn = 2; nrots = 1; }  // one B / CCW

    while (!s.locked && frames < max_frames_guard) {
        const int dx = px - s.x;
        const int want_rot = (nrots > 0);
        const int want_move = (dx != 0);

        int dir = HOLD_NEUTRAL;
        int rot = 0;
        int dn = 0;

        if (want_rot || want_move) {
            int do_rot = 0, do_move = 0;
            if (order == 0 || order == 3) { do_rot = want_rot; do_move = want_move && !want_rot; }
            else if (order == 1) { do_move = want_move; do_rot = want_rot && !want_move; }
            else { do_rot = want_rot; do_move = want_move; }

            if (do_move) {
                const int d = (dx > 0) ? HOLD_RIGHT : HOLD_LEFT;
                // Edge press requires hd != d on the previous frame.
                if (s.hd != d) dir = d;
            }
            if (do_rot && s.rh != rot_btn) rot = rot_btn;
            // orders 2/3: soft-drop on frames where the dpad is free (down-only
            // requires a neutral lateral hold; harmless otherwise).
            if (order >= 2 && dir == HOLD_NEUTRAL) dn = 1;

            const int rot_before = s.rot;
            v4_step(fit_mask, thr, &s, v4_act(dir, dn, rot));
            frames += 1;
            if (rot != 0 && s.rot != rot_before) nrots -= 1;
            // Descending while unaligned can pass the target row; the plan is
            // then dead for this pose.
            if (!s.locked && s.y > py) return -1;
            continue;
        }

        // Aligned: soft drop (down-only). Down is parity-gated inside the step.
        v4_step(fit_mask, thr, &s, v4_act(HOLD_NEUTRAL, 1, 0));
        frames += 1;
        // Abort if drifted off target column/rotation (e.g. gravity slid us past
        // a ledge): the plan is then invalid for this pose.
        if (!s.locked && (s.x != px || (s.rot & 3) != (prot & 3))) return -1;
    }

    if (s.locked && s.x == px && s.y == py && ((s.rot & 3) == (prot & 3))) return frames;
    return -1;
}

// Tuck plan: descend the shaft at column cx in a chosen descent rotation
// (possibly vertical, to fit single-column shafts), then at the target row
// rotate to the final orientation and slide under the overhang, then
// soft-drop to lock. Tries descent rotations {prot, 1, 3} x finish orders
// {rotate-then-slide, slide-then-rotate}; exact simulation arbitrates.
static int v4_greedy_tuck(
    const uint8_t fit_mask[2][GRID_H], int thr,
    const V4State* spawn, int px, int py, int prot, int cx
) {
    int best = -1;
    static const int drots[3] = {-1, 1, 3};  // -1 => use prot
    for (int di = 0; di < 3; ++di) {
        const int drot = (drots[di] < 0) ? (prot & 3) : drots[di];
        for (int finish_rot_first = 0; finish_rot_first <= 1; ++finish_rot_first) {
            V4State s = *spawn;
            int frames = 0;
            const int guard = 4 * (GRID_H + 1) * (thr + 2) + 64;
            int ok = 1;

            while (!s.locked && frames < guard) {
                const int at_row = (s.y == py);
                const int target_rot = at_row ? (prot & 3) : drot;
                const int target_x = at_row ? px : cx;

                const int rots_needed = (s.rot - target_rot) & 3;
                const int rot_btn = (rots_needed == 3) ? 2 : 1;
                const int want_rot = (rots_needed != 0);
                const int dx_t = target_x - s.x;
                int want_move = (dx_t != 0);
                int do_rot = want_rot;
                if (at_row && finish_rot_first && want_rot) want_move = 0;
                if (at_row && !finish_rot_first && want_move) do_rot = 0;

                int dir = HOLD_NEUTRAL, rot = 0, dn = 0;
                if (want_move) {
                    const int d = (dx_t > 0) ? HOLD_RIGHT : HOLD_LEFT;
                    if (s.hd != d) dir = d;
                }
                if (do_rot && s.rh != rot_btn) rot = rot_btn;
                const int tucked = at_row && s.x == px && ((s.rot & 3) == (prot & 3));
                if (dir == HOLD_NEUTRAL && (s.y < py || tucked)) dn = 1;

                v4_step(fit_mask, thr, &s, v4_act(dir, dn, rot));
                frames += 1;
                if (!s.locked && s.y > py) { ok = 0; break; }
            }
            if (ok && s.locked && s.x == px && s.y == py && ((s.rot & 3) == (prot & 3))) {
                if (best < 0 || frames < best) best = frames;
            }
        }
    }
    return best;
}

// Composite single-frame geometric moves (timer-free relaxation): from pose
// (x,y,rot), one engine frame can optionally drop one row (Y stage), then
// optionally move one column (X stage), then optionally rotate with kick
// quirks (rotate stage). Every real frame is one composite edge, so BFS
// distance over this graph is an admissible frame lower bound.
static int v4_composite_succ(
    const uint8_t fit_mask[2][GRID_H], int x, int y, int rot, uint16_t succ[16]
) {
    int n = 0;
    for (int dy = 0; dy <= 1; ++dy) {
        const int y2 = y + dy;
        if (dy) {
            if (y2 >= GRID_H || !fits_masked(fit_mask, x, y2, rot)) continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int x2 = x + dx;
            if (dx && !fits_masked(fit_mask, x2, y2, rot)) continue;
            // no rotation
            succ[n++] = (uint16_t)pose_index(x2, y2, rot);
            // rotations (both buttons, both hold_left variants)
            for (int rotation = 1; rotation <= 2; ++rotation) {
                for (int hl = 0; hl <= 1; ++hl) {
                    int rx = x2, rrot = rot;
                    apply_rotation_masked(fit_mask, &rx, y2, &rrot, rotation, hl);
                    if (rx == x2 && ((rrot & 3) == (rot & 3))) continue;
                    succ[n++] = (uint16_t)pose_index(rx, y2, rrot);
                }
            }
        }
    }
    return n;
}

// Gradient-follow greedy: one-step lookahead over the action set, descending
// the composite-geometric distance field gd[] toward the target pose. Exact
// simulation; returns frames-to-lock at the target, else -1. Handles winding
// paths (tucks through gaps) that the pattern greedies cannot.
static int v4_greedy_follow(
    const uint8_t fit_mask[2][GRID_H], int thr,
    const V4State* spawn, int target_pose, const uint8_t* gd
) {
    V4State s = *spawn;
    int frames = 0;
    const int guard = 4 * (GRID_H + 1) * (thr + 2) + 64;
    int stall = 0;
    const int stall_limit = 2 * (thr + 2) + 8;
    uint8_t cur_gd = gd[pose_index(s.x, s.y, s.rot & 3)];
    if (cur_gd == 0xFFu) return -1;

    static const uint8_t ACTS[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};

    while (!s.locked && frames < guard) {
        int best_act = -1;
        int best_score = 0x7FFFFFFF;
        V4State best_state;
        const int n_act = ((s.p & 1) == 0) ? 12 : 9;
        for (int ai = 0; ai < n_act; ++ai) {
            // odd parity: skip the NEUTRAL down variants (indices 3..5)
            int act = (int)ACTS[ai];
            if ((s.p & 1) != 0 && ai >= 3 && ai < 6) continue;
            V4State t = s;
            v4_step(fit_mask, thr, &t, act);
            int score;
            if (t.locked) {
                score = (pose_index(t.x, t.y, t.rot & 3) == target_pose) ? -1 : 0x7FFFFFFF;
            } else {
                const uint8_t g2 = gd[pose_index(t.x, t.y, t.rot & 3)];
                score = (g2 == 0xFFu) ? 0x7FFFFFFE : (int)g2;
            }
            if (score < best_score) { best_score = score; best_act = act; best_state = t; }
        }
        if (best_act < 0 || best_score >= 0x7FFFFFFE) return -1;
        s = best_state;
        frames += 1;
        if (s.locked) break;
        const uint8_t g_now = gd[pose_index(s.x, s.y, s.rot & 3)];
        if (g_now < cur_gd) { cur_gd = g_now; stall = 0; }
        else if (++stall > stall_limit) return -1;
    }
    if (s.locked && pose_index(s.x, s.y, s.rot & 3) == target_pose) return frames;
    return -1;
}

// Per-call scratch for the geometric LB machinery.
static _Thread_local uint8_t g_v4_gd[512][512];   // [wanted pose][pose'] frames LB
static _Thread_local int32_t g_v4_G[512];          // allowance per pose'

int drm_reach_bfs_v4(
    const uint16_t cols[GRID_W],
    int sx, int sy, int srot,
    int speed_counter, int hor_velocity,
    int hold_dir, int parity, int rot_hold,
    int speed_threshold,
    int max_frames,
    uint16_t out_costs[512]
) {
    if (!cols || !out_costs) return -1;
    if (max_frames <= 0) return -1;

    for (int i = 0; i < 512; ++i) out_costs[i] = 0xFFFFu;

    if ((unsigned)sx >= (unsigned)GRID_W || (unsigned)sy >= (unsigned)GRID_H) return 0;

    if (speed_threshold < 0) speed_threshold = 0;
    if (speed_threshold > 0x7F) speed_threshold = 0x7F;
    const int thr = speed_threshold;
    const int sc_range = thr + 1;

    if (speed_counter < 0) speed_counter = 0;
    if (speed_counter > thr) speed_counter = thr;
    hor_velocity &= 0x0F;
    if (hold_dir < 0 || hold_dir > 2) hold_dir = 0;
    parity &= FAST_DROP_MASK;
    if (rot_hold < 0 || rot_hold > 2) rot_hold = 0;

    const int max_lock_frames = compute_max_lock_frames(sy, speed_counter, thr);
    if (max_frames > max_lock_frames) max_frames = max_lock_frames;

    uint8_t fit_mask[2][GRID_H];
    build_fit_masks(cols, fit_mask);
    if (!fits_masked(fit_mask, sx, sy, srot & 3)) return 0;

    uint8_t wanted_pose[512];
    const int wanted_count = build_wanted_terminal_poses_reachable(
        fit_mask, sx, sy, srot & 3, wanted_pose);
    if (wanted_count == 0) {
        return drm_reach_bfs_v2(cols, sx, sy, srot, speed_counter, hor_velocity,
                                hold_dir, parity, rot_hold, speed_threshold,
                                max_frames, out_costs, NULL, NULL, NULL, 0, NULL);
    }

    // ---- Phase 1a: composite geometric lower bounds (backward BFS per pose) ----
    static _Thread_local uint16_t radj[512][20];
    static _Thread_local uint8_t radj_n[512];
    memset(radj_n, 0, sizeof(radj_n));
    for (int pose = 0; pose < 512; ++pose) {
        const int x = pose & 7;
        const int y = (pose >> 3) & 15;
        const int rot = (pose >> 7) & 3;
        if (!fits_masked(fit_mask, x, y, rot)) continue;
        uint16_t succ[16];
        const int ns = v4_composite_succ(fit_mask, x, y, rot, succ);
        for (int i = 0; i < ns; ++i) {
            const uint16_t s2 = succ[i];
            if (s2 == (uint16_t)pose) continue;
            uint8_t* rn = &radj_n[s2];
            if (*rn < 20) { radj[s2][*rn] = (uint16_t)pose; *rn += 1; }
        }
    }
    int wanted_ids[512];
    int wanted_wi[512];
    int n_wanted_ids = 0;
    for (int pose = 0; pose < 512; ++pose) wanted_wi[pose] = -1;
    for (int pose = 0; pose < 512; ++pose) {
        if (!wanted_pose[pose]) continue;
        const int wi = n_wanted_ids++;
        wanted_ids[wi] = pose;
        wanted_wi[pose] = wi;
        uint8_t* gd = g_v4_gd[wi];
        memset(gd, 0xFF, 512);
        uint16_t q[512];
        int qh = 0, qt = 0;
        gd[pose] = 0;
        q[qt++] = (uint16_t)pose;
        while (qh < qt) {
            const uint16_t cu = q[qh++];
            const uint8_t dc = gd[cu];
            const int rn = (int)radj_n[cu];
            for (int i = 0; i < rn; ++i) {
                const uint16_t pv = radj[cu][i];
                if (gd[pv] != 0xFFu) continue;
                gd[pv] = (uint8_t)(dc + 1);
                q[qt++] = pv;
            }
        }
    }

    // ---- Phase 1b: greedy upper bounds ----
    V4State spawn;
    spawn.x = sx; spawn.y = sy; spawn.rot = srot & 3;
    spawn.sc = speed_counter; spawn.hv = hor_velocity; spawn.hd = hold_dir;
    spawn.p = parity; spawn.rh = rot_hold; spawn.locked = 0;

    uint16_t ub[512];
    for (int i = 0; i < 512; ++i) ub[i] = 0xFFFFu;
    for (int pose = 0; pose < 512; ++pose) {
        if (!wanted_pose[pose]) continue;
        const int px = pose & 7;
        const int py = (pose >> 3) & 15;
        const int prot = (pose >> 7) & 3;
        for (int order = 0; order < 6; ++order) {
            const int f = v4_greedy_try(fit_mask, thr, &spawn, px, py, prot, order);
            if (f > 0 && (uint16_t)f < ub[pose]) ub[pose] = (uint16_t)f;
        }
        if (ub[pose] != 0xFFFFu) continue;
        for (int d = -2; d <= 2; ++d) {
            // d == 0 matters: descend vertically in the pose's own column,
            // then rotate to the final orientation at the bottom.
            const int cx = px + d;
            if ((unsigned)cx >= (unsigned)GRID_W) continue;
            const int f = v4_greedy_tuck(fit_mask, thr, &spawn, px, py, prot, cx);
            if (f > 0 && (uint16_t)f < ub[pose]) ub[pose] = (uint16_t)f;
        }
        if (ub[pose] != 0xFFFFu) continue;
        // Last resort: gradient-follow on the geometric distance field.
        const int f = v4_greedy_follow(fit_mask, thr, &spawn, pose, g_v4_gd[wanted_wi[pose]]);
        if (f > 0 && (uint16_t)f < ub[pose]) ub[pose] = (uint16_t)f;
    }

    // Effective bounds and allowance refresh.
    uint8_t resolved[512];
    memset(resolved, 0, sizeof(resolved));
    const int32_t NEG = INT32_MIN / 4;
    const int32_t POS = INT32_MAX / 4;

    // G[pose'] = max over unresolved wanted p of
    //            (B(p) - 1 - max(2*max(0, prow - y'), gd_p[pose']))
    // Recomputed lazily; staleness only raises G (safe).
#define V4_REFRESH_G() do { \
        for (int i = 0; i < 512; ++i) g_v4_G[i] = NEG; \
        for (int wi = 0; wi < n_wanted_ids; ++wi) { \
            const int p = wanted_ids[wi]; \
            if (resolved[p]) continue; \
            const int prow = (p >> 3) & 15; \
            const int32_t B = (ub[p] == 0xFFFFu) ? POS : (int32_t)ub[p]; \
            const uint8_t* gd = g_v4_gd[wi]; \
            for (int q2 = 0; q2 < 512; ++q2) { \
                const uint8_t gdq = gd[q2]; \
                if (gdq == 0xFFu) continue; \
                const int yq = (q2 >> 3) & 15; \
                int32_t lb = 2 * (prow - yq); \
                if (lb < (int32_t)gdq) lb = (int32_t)gdq; \
                const int32_t allow = B - 1 - lb; \
                if (allow > g_v4_G[q2]) g_v4_G[q2] = allow; \
            } \
        } \
    } while (0)

    V4_REFRESH_G();
    int pending_resolves = 0;

    // ---- Phase 2: gated exact BFS over the v2 state space ----
    enum {
        G_K_ROT = GRID_H,
        G_K_P = G_K_ROT * 4,
        G_K_RH = G_K_P * 2,
        G_K_M = G_K_RH * 3,
        G_K_SC = G_K_M * V2_MICRO_N,
    };
    const uint32_t nkeys = (uint32_t)G_K_SC * (uint32_t)sc_range;
    const int ctx_rc = ensure_ctx2(nkeys, 0);
    if (ctx_rc != 0) return ctx_rc;

    memset(g_ctx2.visited_xmask, 0, (size_t)nkeys);
    memset(g_ctx2.next_xmask, 0, (size_t)nkeys);

    const uint32_t start_m = v2_micro((uint32_t)hold_dir, (uint32_t)hor_velocity);
    const uint32_t start_key = (uint32_t)(sy & 15) + (uint32_t)G_K_ROT * (uint32_t)(srot & 3)
                               + (uint32_t)G_K_P * (uint32_t)parity + (uint32_t)G_K_RH * (uint32_t)rot_hold
                               + (uint32_t)G_K_M * start_m + (uint32_t)G_K_SC * (uint32_t)speed_counter;
    const uint8_t start_xmask = (uint8_t)(1u << (unsigned)(sx & 7));
    g_ctx2.visited_xmask[start_key] = start_xmask;

    NodeMask* cur_frontier = g_ctx2.frontier_a;
    NodeMask* next_frontier = g_ctx2.frontier_b;
    uint32_t cur_n = 1, next_n = 0;
    cur_frontier[0].key = start_key;
    cur_frontier[0].xmask = start_xmask;
    cur_frontier[0].y = (uint8_t)(sy & 15);
    cur_frontier[0].rot = (uint8_t)(srot & 3);
    cur_frontier[0].sc = (uint8_t)speed_counter;
    cur_frontier[0].hv = (uint8_t)hor_velocity;
    cur_frontier[0].hd = (uint8_t)hold_dir;
    cur_frontier[0].p = (uint8_t)parity;
    cur_frontier[0].rh = (uint8_t)rot_hold;

    const uint16_t max_depth = (uint16_t)max_frames;
    int found_wanted = 0;
    uint32_t stat_nodes = 0, stat_trans = 0;
    uint16_t depth_processed = 0;

    static const uint8_t ACTIONS_EVEN[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};
    static const uint8_t ACTIONS_ODD[9] = {0, 1, 2, 6, 7, 8, 12, 13, 14};

    uint8_t amask[GRID_H][4];

    for (uint16_t cur_depth = 0; cur_depth < max_depth && cur_n > 0; ++cur_depth) {
        depth_processed = cur_depth;
        stat_nodes += cur_n;
        next_n = 0;
        const uint16_t next_depth = (uint16_t)(cur_depth + 1);

        if (pending_resolves >= 4) {
            V4_REFRESH_G();
            pending_resolves = 0;
        }
        // Allowance bitmask for children enqueued at next_depth.
        for (int y = 0; y < GRID_H; ++y) {
            for (int r = 0; r < 4; ++r) {
                uint8_t m = 0;
                const int base = r * 128 + y * 8;
                for (int x = 0; x < 8; ++x) {
                    if ((int32_t)next_depth <= g_v4_G[base + x]) m |= (uint8_t)(1u << x);
                }
                amask[y][r] = m;
            }
        }

        for (uint32_t ni = 0; ni < cur_n; ++ni) {
            const NodeMask cur = cur_frontier[ni];
            const uint8_t cur_xmask = cur.xmask;
            const int parity_cur = (int)cur.p & 1;
            const uint8_t* actions = parity_cur ? ACTIONS_ODD : ACTIONS_EVEN;
            const int action_count = parity_cur ? 9 : 12;
            const int cur_bits = __builtin_popcount((unsigned)cur_xmask);

            for (int ai = 0; ai < action_count; ++ai) {
                const int act = (int)actions[ai];
                stat_trans += (uint32_t)cur_bits;

                uint8_t xmask = cur_xmask;
                int y = (int)cur.y;
                const int rot0 = (int)cur.rot & 3;
                int rot = rot0;
                int sc = (int)cur.sc;
                const int hv0 = (int)cur.hv & 0x0F;
                const int hd_prev = (int)cur.hd;
                const int rh_prev = (int)cur.rh;

                const int hold_dir_now = (int)ACT_HOLD_DIR[act];
                const int hold_down = (int)ACT_HOLD_DOWN[act];
                const int rotation = (int)ACT_ROT[act];

                const int prev_left = (hd_prev == HOLD_LEFT);
                const int prev_right = (hd_prev == HOLD_RIGHT);
                const int hold_left = (hold_dir_now == HOLD_LEFT);
                const int hold_right = (hold_dir_now == HOLD_RIGHT);
                const int press_lr = (hold_left && !prev_left) || (hold_right && !prev_right);

                const int down_only = (hold_down != 0) && (hold_dir_now == HOLD_NEUTRAL);
                int drop_triggered = 0;
                if ((parity_cur & FAST_DROP_MASK) == 0 && down_only) {
                    drop_triggered = 1;
                    sc = 0;
                } else {
                    sc = sc + 1;
                    if (sc > speed_threshold) { drop_triggered = 1; sc = 0; }
                }

                if (drop_triggered) {
                    const int ny = y + 1;
                    uint8_t drop_ok = 0u;
                    if ((unsigned)ny < (unsigned)GRID_H) drop_ok = fit_mask[(rot & 1) ? 1 : 0][ny];
                    uint8_t xm_drop = (uint8_t)(xmask & drop_ok);
                    uint8_t xm_lock = (uint8_t)(xmask & (uint8_t)(~drop_ok));
                    while (xm_lock) {
                        const int lx = __builtin_ctz((unsigned)xm_lock);
                        xm_lock &= (uint8_t)(xm_lock - 1u);
                        const int pose = pose_index(lx, y, rot0);
                        if ((unsigned)pose >= 512u) continue;
                        if (out_costs[pose] != 0xFFFFu) continue;
                        out_costs[pose] = next_depth;
                        if (wanted_pose[pose] && !resolved[pose]) {
                            resolved[pose] = 1u;
                            found_wanted += 1;
                            if (found_wanted >= wanted_count) goto v4_done;
                            pending_resolves += 1;
                        }
                    }
                    xmask = xm_drop;
                    if (!xmask) continue;
                    y = ny;
                }

                int allow_move = 0;
                int hv = hv0;
                if (press_lr) { hv = 0; allow_move = 1; }
                else if (hold_dir_now != HOLD_NEUTRAL) {
                    hv = hv + 1;
                    if (hv >= HOR_ACCEL_SPEED) { hv = HOR_RELOAD; allow_move = 1; }
                }

                typedef struct { uint8_t xmask; uint8_t hv; } Tmp;
                Tmp tmp[2];
                int ntmp = 0;
                if (!allow_move || hold_dir_now == HOLD_NEUTRAL) {
                    tmp[0].xmask = xmask; tmp[0].hv = (uint8_t)(hv & 0x0F); ntmp = 1;
                } else if (hold_right) {
                    const uint8_t fits_row = fit_mask[(rot & 1) ? 1 : 0][y];
                    const uint8_t ok = (uint8_t)(fits_row >> 1);
                    const uint8_t mv = (uint8_t)(xmask & ok);
                    const uint8_t bl = (uint8_t)(xmask & (uint8_t)(~ok));
                    if (mv) { tmp[ntmp].xmask = (uint8_t)(mv << 1); tmp[ntmp].hv = (uint8_t)(hv & 0x0F); ntmp++; }
                    if (bl) { tmp[ntmp].xmask = bl; tmp[ntmp].hv = (uint8_t)HOR_BLOCKED; ntmp++; }
                } else {
                    const uint8_t fits_row = fit_mask[(rot & 1) ? 1 : 0][y];
                    const uint8_t ok = (uint8_t)((fits_row << 1) & 0xFFu);
                    const uint8_t mv = (uint8_t)(xmask & ok);
                    const uint8_t bl = (uint8_t)(xmask & (uint8_t)(~ok));
                    if (mv) { tmp[ntmp].xmask = (uint8_t)(mv >> 1); tmp[ntmp].hv = (uint8_t)(hv & 0x0F); ntmp++; }
                    if (bl) { tmp[ntmp].xmask = bl; tmp[ntmp].hv = (uint8_t)HOR_BLOCKED; ntmp++; }
                }

                const int rotation_pressed = (rotation != 0) && (rotation != rh_prev);
                const uint8_t p_next = (uint8_t)((parity_cur ^ 1) & FAST_DROP_MASK);
                const uint8_t hd_next = (uint8_t)hold_dir_now;
                const uint8_t rh_next = (uint8_t)rotation;

                for (int ti = 0; ti < ntmp; ++ti) {
                    const uint8_t xm_in = tmp[ti].xmask;
                    const uint8_t hv_in = tmp[ti].hv;

                    struct Out { uint8_t xmask; uint8_t rot; uint8_t hv; } outg[3];
                    int nout = 0;
                    if (!rotation_pressed) {
                        outg[0].xmask = xm_in; outg[0].rot = (uint8_t)rot; outg[0].hv = hv_in; nout = 1;
                    } else {
                        int rot1 = rot;
                        if (rotation == 1) rot1 = (rot - 1) & 3;
                        else rot1 = (rot + 1) & 3;
                        if ((rot1 & 1) != 0) {
                            const uint8_t fit_v = fit_mask[1][y];
                            const uint8_t acc = (uint8_t)(xm_in & fit_v);
                            const uint8_t rej = (uint8_t)(xm_in & (uint8_t)(~fit_v));
                            if (acc) { outg[nout].xmask = acc; outg[nout].rot = (uint8_t)rot1; outg[nout].hv = hv_in; nout++; }
                            if (rej) { outg[nout].xmask = rej; outg[nout].rot = (uint8_t)rot; outg[nout].hv = hv_in; nout++; }
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
                            if (shifted_src) { outg[nout].xmask = (uint8_t)(shifted_src >> 1); outg[nout].rot = (uint8_t)rot1; outg[nout].hv = hv_in; nout++; }
                            if (acc_noshift) { outg[nout].xmask = acc_noshift; outg[nout].rot = (uint8_t)rot1; outg[nout].hv = hv_in; nout++; }
                            if (rej) { outg[nout].xmask = rej; outg[nout].rot = (uint8_t)rot; outg[nout].hv = hv_in; nout++; }
                        }
                    }

                    for (int oi = 0; oi < nout; ++oi) {
                        uint8_t xm_out = outg[oi].xmask;
                        if (!xm_out) continue;
                        const uint8_t rot_out = outg[oi].rot;
                        // Gate: drop children that provably cannot improve any
                        // unresolved pose.
                        xm_out &= amask[y][rot_out & 3];
                        if (!xm_out) continue;
                        const uint8_t hv_out = outg[oi].hv;
                        const uint32_t m_out = v2_micro((uint32_t)hd_next, (uint32_t)(hv_out & 15u));
                        const uint32_t next_key = (uint32_t)(y & 15) + (uint32_t)G_K_ROT * (uint32_t)(rot_out & 3)
                                                  + (uint32_t)G_K_P * (uint32_t)p_next + (uint32_t)G_K_RH * (uint32_t)rh_next
                                                  + (uint32_t)G_K_M * m_out + (uint32_t)G_K_SC * (uint32_t)sc;
                        const uint8_t seen = g_ctx2.visited_xmask[next_key];
                        const uint8_t new_bits = (uint8_t)(xm_out & (uint8_t)(~seen));
                        if (!new_bits) continue;
                        g_ctx2.visited_xmask[next_key] = (uint8_t)(seen | new_bits);

                        const uint8_t prev_accum = g_ctx2.next_xmask[next_key];
                        g_ctx2.next_xmask[next_key] = (uint8_t)(prev_accum | new_bits);
                        if (prev_accum == 0) {
                            if (next_n >= nkeys) return -2;
                            next_frontier[next_n].key = next_key;
                            next_frontier[next_n].xmask = 0u;
                            next_frontier[next_n].y = (uint8_t)(y & 15);
                            next_frontier[next_n].rot = (uint8_t)(rot_out & 3);
                            next_frontier[next_n].sc = (uint8_t)sc;
                            next_frontier[next_n].hv = (uint8_t)(hv_out & 15u);
                            next_frontier[next_n].hd = hd_next;
                            next_frontier[next_n].p = p_next;
                            next_frontier[next_n].rh = rh_next;
                            next_n += 1;
                        }
                    }
                }
            }
        }

        for (uint32_t ni = 0; ni < next_n; ++ni) {
            const uint32_t key = next_frontier[ni].key;
            next_frontier[ni].xmask = g_ctx2.next_xmask[key];
            g_ctx2.next_xmask[key] = 0u;
        }
        NodeMask* tp = cur_frontier; cur_frontier = next_frontier; next_frontier = tp;
        cur_n = next_n;
    }

v4_done:
    ;
    if (stats_enabled()) {
        g_last_stats.visited_states = 0;
        g_last_stats.expanded_states = 0;
        g_last_stats.transitions = stat_trans;
        g_last_stats.locks_found = 0;
        g_last_stats.queue_nodes_enqueued = stat_nodes;
        g_last_stats.queue_nodes_expanded = stat_nodes;
        g_last_stats.max_depth = max_depth;
        g_last_stats.depth_processed = depth_processed;
        g_last_stats.wanted_count = (uint16_t)wanted_count;
        g_last_stats.found_wanted = (uint16_t)(found_wanted < 0 ? 0 : found_wanted);
    }

    // Finalize: unresolved poses take their greedy UB (the gate guarantees no
    // strictly better path was pruned), and BFS finds never beat an
    // achievable UB by less than the gate allows.
    for (int pose = 0; pose < 512; ++pose) {
        if (ub[pose] == 0xFFFFu) continue;
        if (out_costs[pose] == 0xFFFFu || ub[pose] < out_costs[pose]) out_costs[pose] = ub[pose];
    }
    return 0;
#undef V4_REFRESH_G
}
