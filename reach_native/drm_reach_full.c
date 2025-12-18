// Dr. Mario Falling-Pill Reachability (NES-accurate, native helper)
//
// This module implements a native BFS over the *full* per-frame fall state used by
// `envs/retro/fast_reach.py`:
//   (x, y, rot, speed_counter, hor_velocity, hold_dir, parity)
//
// It is intended as a performance accelerator for the placement macro-action
// environment. The Python reference implementation is correct but can be slow
// when enumerating reachability for every spawn.
//
// Key invariants mirrored from the Python stepper:
// - Frame order is Y (gravity/soft-drop) -> X (DAS) -> Rotate.
// - "Down-only" soft drop triggers on parity frames (frameCounter & 1 == 1) and
//   resets speed_counter.
// - Gravity triggers when (speed_counter + 1) > speed_threshold, then resets.
// - Horizontal movement:
//     - Edge press moves immediately and resets hor_velocity.
//     - When holding L/R, hor_velocity increments; on >= 16 it triggers a move,
//       then reloads to 10 (repeat every 6 frames).
//     - Blocked movement sets hor_velocity = 15.
// - Rotation quirks:
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
static inline void decode_action(int action_idx, int* hold_dir, int* hold_down, int* rotation) {
    int hd = action_idx / 6;
    int rem = action_idx % 6;
    int down = rem / 3;
    int rot = rem % 3;
    *hold_dir = hd;
    *hold_down = down;
    *rotation = rot;  // 0 none, 1 cw, 2 ccw
}

static inline int pose_index(int x, int y, int rot) {
    return (int)((rot & 3) * (GRID_H * GRID_W) + (y * GRID_W) + x);
}

static inline int fits(const uint16_t cols[GRID_W], int x, int y, int rot) {
    const int x_i = x;
    const int y_i = y;
    const int rot_i = rot & 3;
    if ((unsigned)x_i >= (unsigned)GRID_W || (unsigned)y_i >= (unsigned)GRID_H) return 0;
    if (cols[x_i] & (uint16_t)(1u << (unsigned)y_i)) return 0;
    if ((rot_i & 1) == 0) {
        if (x_i + 1 >= GRID_W) return 0;
        if (cols[x_i + 1] & (uint16_t)(1u << (unsigned)y_i)) return 0;
    } else {
        if (y_i - 1 >= 0 && (cols[x_i] & (uint16_t)(1u << (unsigned)(y_i - 1)))) return 0;
    }
    return 1;
}

static inline void apply_rotation(
    const uint16_t cols[GRID_W], int* x, int y, int* rot, int rotation, int hold_left
) {
    if (rotation == 0) return;
    int x0 = *x;
    int rot0 = (*rot) & 3;
    int rot1 = rot0;
    if (rotation == 1) rot1 = (rot0 - 1) & 3;     // CW: decrement
    else rot1 = (rot0 + 1) & 3;                   // CCW: increment

    if ((rot1 & 1) == 0) {
        // Target is horizontal.
        if (fits(cols, x0, y, rot1)) {
            if (hold_left && fits(cols, x0 - 1, y, rot1)) {
                *x = x0 - 1;
                *rot = rot1;
                return;
            }
            *x = x0;
            *rot = rot1;
            return;
        }
        // Kick-left attempt.
        if (fits(cols, x0 - 1, y, rot1)) {
            *x = x0 - 1;
            *rot = rot1;
            return;
        }
        // Reject.
        return;
    }

    // Target is vertical: only in-place validation.
    if (fits(cols, x0, y, rot1)) {
        *rot = rot1;
    }
}

static inline void step_state(
    const uint16_t cols[GRID_W],
    int speed_threshold,
    int x, int y, int rot,
    int speed_counter, int hor_velocity,
    int hold_dir_prev, int parity,
    int action_idx,
    int* out_x, int* out_y, int* out_rot,
    int* out_speed_counter, int* out_hor_velocity,
    int* out_hold_dir, int* out_parity,
    int* out_locked
) {
    int hold_dir = HOLD_NEUTRAL;
    int hold_down = 0;
    int rotation = 0;
    decode_action(action_idx, &hold_dir, &hold_down, &rotation);

    const int prev_left = (hold_dir_prev == HOLD_LEFT);
    const int prev_right = (hold_dir_prev == HOLD_RIGHT);
    const int hold_left = (hold_dir == HOLD_LEFT);
    const int hold_right = (hold_dir == HOLD_RIGHT);

    const int press_left = hold_left && !prev_left;
    const int press_right = hold_right && !prev_right;
    const int press_lr = press_left || press_right;

    const int down_only = (hold_down != 0) && (hold_dir == HOLD_NEUTRAL);
    int drop_triggered = 0;

    // ---------------- Y stage (gravity / down-only soft drop) ----------------
    if ((parity & FAST_DROP_MASK) != 0 && down_only) {
        drop_triggered = 1;
        speed_counter = 0;
    } else {
        speed_counter = speed_counter + 1;
        if (speed_counter > speed_threshold) {
            drop_triggered = 1;
            speed_counter = 0;
        }
    }

    if (drop_triggered) {
        if (fits(cols, x, y + 1, rot)) {
            y = y + 1;
        } else {
            // Immediate lock: confirmPlacement happens in checkYMove.
            *out_x = x;
            *out_y = y;
            *out_rot = rot & 3;
            *out_speed_counter = 0;
            *out_hor_velocity = hor_velocity & 0xFF;
            *out_hold_dir = hold_dir;
            *out_parity = (parity ^ 1) & FAST_DROP_MASK;
            *out_locked = 1;
            return;
        }
    }

    // ---------------- X stage (DAS movement) ----------------
    int allow_move = 0;
    if (press_lr) {
        hor_velocity = 0;
        allow_move = 1;
    } else {
        if (hold_dir != HOLD_NEUTRAL) {
            hor_velocity = hor_velocity + 1;
            if (hor_velocity >= HOR_ACCEL_SPEED) {
                hor_velocity = HOR_RELOAD;
                allow_move = 1;
            }
        }
    }

    if (allow_move) {
        // ROM order: right check then left check.
        if (hold_right) {
            if (fits(cols, x + 1, y, rot)) x = x + 1;
            else hor_velocity = HOR_BLOCKED;
        }
        if (hold_left) {
            if (fits(cols, x - 1, y, rot)) x = x - 1;
            else hor_velocity = HOR_BLOCKED;
        }
    }

    // ---------------- Rotate stage ----------------
    apply_rotation(cols, &x, y, &rot, rotation, hold_left);

    *out_x = x;
    *out_y = y;
    *out_rot = rot & 3;
    *out_speed_counter = speed_counter;
    *out_hor_velocity = hor_velocity;
    *out_hold_dir = hold_dir;
    *out_parity = (parity ^ 1) & FAST_DROP_MASK;
    *out_locked = 0;
}

static inline uint32_t state_index(
    int sc_range,
    int x, int y, int rot,
    int speed_counter,
    int hor_velocity,
    int hold_dir,
    int parity
) {
    // Layout:
    //   x(8) -> y(16) -> rot(4) -> sc(sc_range) -> hv(16) -> hold_dir(3) -> parity(2)
    // idx = x + 8*y + 8*16*rot + 8*16*4*sc + 8*16*4*sc_range*hv + ... + parity*...
    const uint32_t x_u = (uint32_t)(x & 7);
    const uint32_t y_u = (uint32_t)(y & 15);
    const uint32_t rot_u = (uint32_t)(rot & 3);
    const uint32_t sc_u = (uint32_t)(speed_counter);
    const uint32_t hv_u = (uint32_t)(hor_velocity & 15);
    const uint32_t hd_u = (uint32_t)(hold_dir & 3);
    const uint32_t p_u = (uint32_t)(parity & 1);

    const uint32_t stride_x = 1;
    const uint32_t stride_y = GRID_W;
    const uint32_t stride_rot = GRID_W * GRID_H;
    const uint32_t stride_sc = GRID_W * GRID_H * 4u;
    const uint32_t stride_hv = stride_sc * (uint32_t)sc_range;
    const uint32_t stride_hd = stride_hv * 16u;
    const uint32_t stride_p = stride_hd * 3u;

    return x_u * stride_x + y_u * stride_y + rot_u * stride_rot + sc_u * stride_sc + hv_u * stride_hv +
           hd_u * stride_hd + p_u * stride_p;
}

int drm_reach_bfs_full(
    const uint16_t cols[GRID_W],
    int sx, int sy, int srot,
    int speed_counter, int hor_velocity,
    int hold_dir, int parity,
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

    if ((unsigned)sx >= 8u || (unsigned)sy >= 16u) return 0;
    if (!fits(cols, sx, sy, srot)) return 0;

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

    // Allocate BFS workspace.
    const uint32_t stride_sc = (uint32_t)(GRID_W * GRID_H * 4u);
    const uint32_t stride_hv = stride_sc * (uint32_t)sc_range;
    const uint32_t stride_hd = stride_hv * 16u;
    const uint32_t stride_p = stride_hd * 3u;
    const uint32_t nstates = stride_p * 2u;

    uint16_t* depth = (uint16_t*)malloc((size_t)nstates * sizeof(uint16_t));
    uint32_t* parent = (uint32_t*)malloc((size_t)nstates * sizeof(uint32_t));
    uint8_t* parent_action = (uint8_t*)malloc((size_t)nstates * sizeof(uint8_t));
    uint32_t* queue = (uint32_t*)malloc((size_t)nstates * sizeof(uint32_t));
    if (!depth || !parent || !parent_action || !queue) {
        free(depth);
        free(parent);
        free(parent_action);
        free(queue);
        return -2;
    }

    memset(depth, 0xFF, (size_t)nstates * sizeof(uint16_t));
    // parent/parent_action only need to be valid on visited states, but memset is cheap and simple.
    for (uint32_t i = 0; i < nstates; ++i) {
        parent[i] = UINT32_MAX;
        parent_action[i] = 0xFFu;
    }

    uint32_t term_parent_state[512];
    uint8_t term_parent_action[512];
    for (int i = 0; i < 512; ++i) {
        term_parent_state[i] = UINT32_MAX;
        term_parent_action[i] = 0xFFu;
    }

    const uint32_t start = state_index(
        sc_range, sx, sy, srot & 3, speed_counter, hor_velocity, hold_dir, parity
    );
    depth[start] = 0;
    parent[start] = UINT32_MAX;
    parent_action[start] = 0xFFu;

    uint32_t qh = 0;
    uint32_t qt = 0;
    queue[qt++] = start;

    // BFS.
    while (qh < qt) {
        const uint32_t cur = queue[qh++];
        const uint16_t cur_depth = depth[cur];
        if ((int)cur_depth >= max_frames) continue;

        // Decode state index -> fields (inverse of state_index).
        uint32_t tmp = cur;
        const uint32_t x = tmp % 8u; tmp /= 8u;
        const uint32_t y = tmp % 16u; tmp /= 16u;
        const uint32_t rot = tmp % 4u; tmp /= 4u;
        const uint32_t sc = tmp % (uint32_t)sc_range; tmp /= (uint32_t)sc_range;
        const uint32_t hv = tmp % 16u; tmp /= 16u;
        const uint32_t hd = tmp % 3u; tmp /= 3u;
        const uint32_t p = tmp % 2u;

        for (int act = 0; act < 18; ++act) {
            int nx, ny, nrot, nsc, nhv, nhd, np, locked;
            step_state(
                cols,
                speed_threshold,
                (int)x, (int)y, (int)rot,
                (int)sc, (int)hv,
                (int)hd, (int)p,
                act,
                &nx, &ny, &nrot,
                &nsc, &nhv,
                &nhd, &np,
                &locked
            );

            const uint16_t next_depth = (uint16_t)(cur_depth + 1);

            if (locked) {
                const int pose = pose_index(nx, ny, nrot);
                if (pose < 0 || pose >= 512) continue;
                if (out_costs[pose] != 0xFFFFu) continue;  // already have a minimal path
                out_costs[pose] = next_depth;
                term_parent_state[pose] = cur;
                term_parent_action[pose] = (uint8_t)act;
                continue;
            }

            // Keep within state space bounds.
            if ((unsigned)nx >= 8u || (unsigned)ny >= 16u) continue;
            if ((unsigned)nrot >= 4u) continue;
            if (nsc < 0) nsc = 0;
            if (nsc > speed_threshold) nsc = speed_threshold;
            if (nhv < 0) nhv = 0;
            nhv &= 15;
            if (nhd < 0 || nhd > 2) nhd = 0;
            np &= FAST_DROP_MASK;

            const uint32_t next = state_index(sc_range, nx, ny, nrot, nsc, nhv, nhd, np);
            if (depth[next] != 0xFFFFu) continue;
            depth[next] = next_depth;
            parent[next] = cur;
            parent_action[next] = (uint8_t)act;
            queue[qt++] = next;
        }
    }

    // Reconstruct scripts into caller-provided buffer.
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
            free(depth);
            free(parent);
            free(parent_action);
            free(queue);
            return -3;
        }
        out_offsets[pose] = (uint16_t)used;
        out_lengths[pose] = (uint16_t)len;

        int pos = used + len;
        // Last action (parent -> locked pose)
        out_script_buf[pos - 1] = term_parent_action[pose];
        pos -= 1;
        uint32_t cur = term_parent_state[pose];
        while (cur != UINT32_MAX) {
            // Root has parent_action == 0xFF; do not write it.
            const uint8_t a = parent_action[cur];
            const uint32_t pcur = parent[cur];
            if (pcur == UINT32_MAX) break;
            out_script_buf[pos - 1] = a;
            pos -= 1;
            cur = pcur;
        }
        // Sanity: ensure we filled exactly `len` actions.
        if (pos != used) {
            // If this triggers, we have an inconsistency between cost and parent chain.
            // Mark as infeasible.
            out_costs[pose] = 0xFFFFu;
            out_offsets[pose] = 0u;
            out_lengths[pose] = 0u;
            continue;
        }
        used += len;
    }
    *out_script_used = used;

    free(depth);
    free(parent);
    free(parent_action);
    free(queue);
    return 0;
}

