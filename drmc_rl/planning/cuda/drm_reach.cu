// drm_reach.cu — CUDA port of the Dr. Mario reachability planner (v4 semantics).
//
// Compiled at runtime with NVRTC (see reach_cuda/host.py); deliberately
// self-contained: no includes, fixed-width types defined locally.
//
// Semantic ground truth: game_engine/third_party/reach_native/drm_reach_full.c
// (the copy linked into libdrmario_pool.so, i.e. the production annotation
// path). out_costs must be bit-exact against drm_reach_bfs_v4 == v1/v2/v3.
// Device functions below mirror that file line-for-line; comments cite it.
//
// Execution model: persistent kernel, one instance per thread block. Blocks
// pop instances off a global cursor until the batch is drained. All phases of
// one solve run device-side; __syncthreads() separates parallel stages.
//
// Stages: [x] skeleton  [x] phase 1 geometry  [x] phase 2 greedy UB
//         [x] phase 3 gated BFS  [ ] scripts
//
// Phase-3 note: the CPU v4 runs a v2-keyed scalar BFS under the UB/LB gate;
// we run a v3-style bit-sliced BFS (8 x-lanes x sc-bits per key) under the
// same gate. Both are exact, and exact costs are unique, so output equality
// with CPU v4 is the correctness bar (enforced by tools/test_reach_cuda_parity).
// The gate is a function of (x, y, rot) only, so it applies uniformly across
// a key's sc bits — it becomes a per-lane AND mask, which is what makes the
// bit-sliced form legal under gating.

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long long u64;
typedef signed char    i8;

#define GRID_W 8
#define GRID_H 16
#define N_POSES 512
#define COST_INF 0xFFFFu

#define HOLD_NEUTRAL 0
#define HOLD_LEFT 1
#define HOLD_RIGHT 2
#define FAST_DROP_MASK 0x01
#define HOR_ACCEL_SPEED 0x10
#define HOR_RELOAD 0x0A
#define HOR_BLOCKED 0x0F

// Action tables (drm_reach_full.c:55-69). 18-action encoding dir*6+dn*3+rot.
__device__ __constant__ u8 ACT_HOLD_DIR[18]  = {0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2};
__device__ __constant__ u8 ACT_HOLD_DOWN[18] = {0,0,0,1,1,1, 0,0,0,1,1,1, 0,0,0,1,1,1};
__device__ __constant__ u8 ACT_ROT[18]       = {0,1,2,0,1,2, 0,1,2,0,1,2, 0,1,2,0,1,2};

__device__ inline int pose_index(int x, int y, int rot) {
    return ((rot & 3) * (GRID_H * GRID_W)) + (y * GRID_W) + x;
}

// ---------------------------------------------------------------------------
// Instance ABI (32 bytes, frozen — host.py packs this exact layout)
// ---------------------------------------------------------------------------
struct Instance {
    u16 cols[GRID_W];   // column bitboards, bit y (top-origin) = occupied
    u8  sx, sy, srot, sc, hv, hd, p, rh;
    u8  thr;            // speed threshold 0..127
    u8  flags;          // reserved
    u16 max_frames;     // BFS depth cap
    u32 _pad;
};

// ---------------------------------------------------------------------------
// Per-slot workspace in a global-memory arena (host sizes: one per resident
// block slot, not per instance). Only phase-1/2 fields are used so far; BFS
// fields are declared to freeze the layout.
// ---------------------------------------------------------------------------
// v3 key space (drm_reach_full.c:1515-1520): key = y + 16*rot + 64*micro
// + 2112*rh; 6336 keys per parity. Parity lives in a separate index (it is a
// pure function of BFS depth, uniform across each frontier).
#define WS_GD_MAX 512              // gd rows (CPU supports all 512 wanted)
#define SCRIPT_REC_CAP 384         // max recordable script length per pose
#define SCRIPT_BUF_CAP 24576       // per-instance packed script buffer
#define WS_MAX_THREADS 1024        // upper bound on blockDim.x
// Parent arena slot: one u32 per full state (2 parities x 6336 keys x 8 lanes
// x 64 sc). Used only by the scripts re-BFS (which requires thr < 64).
#define PARENT_SLOT_U32 (2 * 6336 * 8 * 64)
#define WS_NKEYS_P 6336
#define V3_K_ROT 16
#define V3_K_M 64
#define V3_K_RH 2112
#define V2_MICRO_N 33
#define LANE_WORDS 2               // sc bits per lane: 2x u64 (thr <= 127)

struct Workspace {
    // phase 1
    u16 radj[N_POSES][20];
    u8  radj_n[N_POSES];
    u8  wanted[N_POSES];
    u16 wanted_ids[N_POSES];
    u8  gd[WS_GD_MAX][N_POSES];    // composite LB fields, wanted-major
    // phase 2
    u16 ub[N_POSES];
    // scripts stage
    u8  script_scratch[N_POSES][SCRIPT_REC_CAP];
    u8  tuck_tmp[WS_MAX_THREADS][SCRIPT_REC_CAP];
    u16 script_len[N_POSES];       // 0xFFFF = finite cost but no greedy match
    u16 pose_wi[N_POSES];          // pose -> wanted index (gd row), 0xFFFF none
    u32 term_parent[N_POSES];      // re-BFS: lock's parent state id | act<<25
    // phase 3
    int G[N_POSES];                // gate allowance per pose
    u64 visited[2][WS_NKEYS_P][GRID_W][LANE_WORDS];
    u64 accum[2][WS_NKEYS_P][GRID_W][LANE_WORDS];   // ping-pong frontier blocks
    u32 visited_flag[2][WS_NKEYS_P];                // u32: atomicExch needs it
    u32 accum_flag[2][WS_NKEYS_P];
    u32 touched_v[2 * WS_NKEYS_P];                  // dirtied visited blocks
    u32 frontier[2][WS_NKEYS_P];                    // key lists, ping-pong
};

// Per-parity action tables (drm_reach_full.c:635-639).
__device__ __constant__ u8 ACTS_EVEN[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};
__device__ __constant__ u8 ACTS_ODD[9]   = {0, 1, 2, 6, 7, 8, 12, 13, 14};

__device__ inline u32 v2_micro(u32 hd, u32 hv) {
    return hd == 0u ? 0u : (hd == 1u ? 1u + hv : 17u + hv);
}

// drm_reach_full.c:452-473
__device__ inline int compute_max_lock_frames(int y0, int sc0, int thr) {
    int sc = sc0;
    if (sc < 0) sc = 0;
    if (sc > thr) sc = thr;
    int m_max = GRID_H - y0;
    if (m_max < 1) m_max = 1;
    const int first = (thr - sc) + 1;
    const int per = thr + 1;
    int total = first;
    if (m_max > 1) total += (m_max - 1) * per;
    if (total < 1) total = 1;
    return total;
}

// ---------------------------------------------------------------------------
// Collision masks (drm_reach_full.c:119-151)
// ---------------------------------------------------------------------------
struct FitMask {
    u8 m[2][GRID_H];   // [0]=horizontal fit, [1]=vertical fit
};

__device__ inline void build_fit_masks(const u16* cols, FitMask* fm) {
    u8 occ[GRID_H];
    u8 empty[GRID_H];
    for (int y = 0; y < GRID_H; ++y) occ[y] = 0u;
    for (int x = 0; x < GRID_W; ++x) {
        const u16 col = cols[x];
        for (int y = 0; y < GRID_H; ++y) {
            if (col & (u16)(1u << (unsigned)y)) occ[y] |= (u8)(1u << (unsigned)x);
        }
    }
    for (int y = 0; y < GRID_H; ++y) empty[y] = (u8)(~occ[y]) & 0xFFu;
    for (int y = 0; y < GRID_H; ++y) fm->m[0][y] = (u8)(empty[y] & (u8)(empty[y] >> 1));
    fm->m[1][0] = empty[0];
    for (int y = 1; y < GRID_H; ++y) fm->m[1][y] = (u8)(empty[y] & empty[y - 1]);
}

__device__ inline int fits_masked(const FitMask* fm, int x, int y, int rot) {
    if ((unsigned)x >= (unsigned)GRID_W || (unsigned)y >= (unsigned)GRID_H) return 0;
    const u8 mask = fm->m[(rot & 1) ? 1 : 0][y];
    return (int)((mask >> (unsigned)x) & 1u);
}

// drm_reach_full.c:269-306
__device__ void apply_rotation_masked(
    const FitMask* fm, int* x, int y, int* rot, int rotation, int hold_left
) {
    if (rotation == 0) return;
    const int x0 = *x;
    const int rot0 = (*rot) & 3;
    int rot1 = rot0;
    if (rotation == 1) rot1 = (rot0 - 1) & 3;
    else rot1 = (rot0 + 1) & 3;

    if ((rot1 & 1) == 0) {
        if (fits_masked(fm, x0, y, rot1)) {
            if (hold_left && fits_masked(fm, x0 - 1, y, rot1)) {
                *x = x0 - 1; *rot = rot1; return;
            }
            *x = x0; *rot = rot1; return;
        }
        if (fits_masked(fm, x0 - 1, y, rot1)) { *x = x0 - 1; *rot = rot1; return; }
        return;
    }
    if (fits_masked(fm, x0, y, rot1)) *rot = rot1;
}

// ---------------------------------------------------------------------------
// Timer-free wanted-pose flood fill (drm_reach_full.c:157-267).
// Serial on one thread (<=512 nodes, trivial); queue/visited in shared.
// ---------------------------------------------------------------------------
__device__ int build_wanted_terminal_poses_reachable_serial(
    const FitMask* fm, int sx, int sy, int srot,
    u8* wanted /*512*/, u8* visited /*512 shared scratch*/, u16* queue /*512 shared*/
) {
    for (int i = 0; i < N_POSES; ++i) { wanted[i] = 0; visited[i] = 0; }
    if ((unsigned)sx >= (unsigned)GRID_W || (unsigned)sy >= (unsigned)GRID_H) return 0;
    if (!fits_masked(fm, sx, sy, srot & 3)) return 0;

    int qh = 0, qt = 0;
    const u16 start = (u16)pose_index(sx, sy, srot & 3);
    visited[start] = 1u;
    queue[qt++] = start;

    int wanted_count = 0;
    while (qh < qt) {
        const u16 pose = queue[qh++];
        const int x = (int)(pose & 7u);
        const int y = (int)((pose >> 3) & 15u);
        const int rot = (int)((pose >> 7) & 3u);

        const int can_fall = (y + 1 < GRID_H) && fits_masked(fm, x, y + 1, rot);
        if (!can_fall) {
            if ((rot & 1) == 0) {
                if (x + 1 < GRID_W) {
                    if (!wanted[pose]) { wanted[pose] = 1u; wanted_count += 1; }
                }
            } else {
                if (y >= 1) {
                    if (!wanted[pose]) { wanted[pose] = 1u; wanted_count += 1; }
                }
            }
        }

        if (fits_masked(fm, x - 1, y, rot)) {
            const u16 np = (u16)pose_index(x - 1, y, rot);
            if (!visited[np]) { visited[np] = 1u; queue[qt++] = np; }
        }
        if (fits_masked(fm, x + 1, y, rot)) {
            const u16 np = (u16)pose_index(x + 1, y, rot);
            if (!visited[np]) { visited[np] = 1u; queue[qt++] = np; }
        }
        if (y + 1 < GRID_H && fits_masked(fm, x, y + 1, rot)) {
            const u16 np = (u16)pose_index(x, y + 1, rot);
            if (!visited[np]) { visited[np] = 1u; queue[qt++] = np; }
        }
        for (int rotation = 1; rotation <= 2; ++rotation) {
            for (int hold_left = 0; hold_left <= 1; ++hold_left) {
                int rx = x;
                int rrot = rot;
                apply_rotation_masked(fm, &rx, y, &rrot, rotation, hold_left);
                if (rx == x && (rrot & 3) == (rot & 3)) continue;
                const u16 np = (u16)pose_index(rx, y, rrot);
                if (!visited[np]) { visited[np] = 1u; queue[qt++] = np; }
            }
        }
    }
    return wanted_count;
}

// ---------------------------------------------------------------------------
// v4 exact single-state stepper + greedy planners (drm_reach_full.c:1978-2263)
// ---------------------------------------------------------------------------
struct V4State {
    int x, y, rot;
    int sc, hv, hd, p, rh;
    int locked;
};

__device__ void v4_step(const FitMask* fm, int thr, V4State* s, int act) {
    const int dir = (int)ACT_HOLD_DIR[act];
    const int dn = (int)ACT_HOLD_DOWN[act];
    const int rotation = (int)ACT_ROT[act];

    const int prev_left = (s->hd == HOLD_LEFT);
    const int prev_right = (s->hd == HOLD_RIGHT);
    const int hold_left = (dir == HOLD_LEFT);
    const int hold_right = (dir == HOLD_RIGHT);
    const int press_lr = (hold_left && !prev_left) || (hold_right && !prev_right);

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
        if (ny >= GRID_H || !fits_masked(fm, s->x, ny, s->rot)) {
            s->locked = 1;
            s->hd = dir;
            s->rh = rotation;
            s->p ^= 1;
            return;
        }
        s->y = ny;
    }

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
        if (fits_masked(fm, nx, s->y, s->rot)) s->x = nx;
        else s->hv = HOR_BLOCKED;
    }

    const int rotation_pressed = (rotation != 0) && (rotation != s->rh);
    if (rotation_pressed) {
        int rx = s->x;
        int rrot = s->rot;
        apply_rotation_masked(fm, &rx, s->y, &rrot, rotation, hold_left);
        s->x = rx;
        s->rot = rrot;
    }

    s->hd = dir;
    s->rh = rotation;
    s->p ^= 1;
}

__device__ inline int v4_act(int dir, int dn, int rot) { return dir * 6 + dn * 3 + rot; }

// drm_reach_full.c:2058-2131. `rec` (optional): record the action byte of
// every step; used by the scripts stage. A rollout that would exceed
// SCRIPT_REC_CAP steps is aborted when recording (it cannot match a
// cap-checked cost anyway).
__device__ int v4_greedy_try(
    const FitMask* fm, int thr, const V4State* spawn, int px, int py, int prot, int order,
    u8* rec = nullptr
) {
    V4State s = *spawn;
    int frames = 0;
    const int max_frames_guard = 4 * (GRID_H + 1) * (thr + 2) + 64;

    if (order >= 4) {
        if (rec) rec[frames] = (u8)v4_act(HOLD_NEUTRAL, 1, 0);
        v4_step(fm, thr, &s, v4_act(HOLD_NEUTRAL, 1, 0));
        frames += 1;
        if (s.locked) {
            return (s.x == px && s.y == py && ((s.rot & 3) == (prot & 3))) ? frames : -1;
        }
        if (s.y > py) return -1;
        order -= 2;
    }

    int rots_needed = (s.rot - prot) & 3;
    int rot_btn = 1;
    int nrots = rots_needed;
    if (rots_needed == 3) { rot_btn = 2; nrots = 1; }

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
                if (s.hd != d) dir = d;
            }
            if (do_rot && s.rh != rot_btn) rot = rot_btn;
            if (order >= 2 && dir == HOLD_NEUTRAL) dn = 1;

            const int rot_before = s.rot;
            if (rec) {
                if (frames >= SCRIPT_REC_CAP) return -1;
                rec[frames] = (u8)v4_act(dir, dn, rot);
            }
            v4_step(fm, thr, &s, v4_act(dir, dn, rot));
            frames += 1;
            if (rot != 0 && s.rot != rot_before) nrots -= 1;
            if (!s.locked && s.y > py) return -1;
            continue;
        }

        if (rec) {
            if (frames >= SCRIPT_REC_CAP) return -1;
            rec[frames] = (u8)v4_act(HOLD_NEUTRAL, 1, 0);
        }
        v4_step(fm, thr, &s, v4_act(HOLD_NEUTRAL, 1, 0));
        frames += 1;
        if (!s.locked && (s.x != px || (s.rot & 3) != (prot & 3))) return -1;
    }

    if (s.locked && s.x == px && s.y == py && ((s.rot & 3) == (prot & 3))) return frames;
    return -1;
}

// drm_reach_full.c:2137-2180 (game_engine copy: rot_early/rot_late variants)
// `rec`/`rec_len`: when recording, the best matching variant's actions are
// left in rec with *rec_len = its frame count.
__device__ int v4_greedy_tuck(
    const FitMask* fm, int thr, const V4State* spawn, int px, int py, int prot, int cx,
    u8* rec = nullptr, u8* tmp = nullptr
) {
    int best = -1;
    for (int rot_early = 0; rot_early <= 1; ++rot_early) {
        V4State s = *spawn;
        int frames = 0;
        const int guard = 4 * (GRID_H + 1) * (thr + 2) + 64;

        int rots_needed = (s.rot - prot) & 3;
        int rot_btn = 1;
        int nrots = rots_needed;
        if (rots_needed == 3) { rot_btn = 2; nrots = 1; }

        int ok = 1;
        while (!s.locked && frames < guard) {
            const int at_row = (s.y == py);
            const int dx_t = (at_row ? px : cx) - s.x;
            const int want_rot = (nrots > 0) && (rot_early || at_row);
            const int want_move = (dx_t != 0);

            int dir = HOLD_NEUTRAL, rot = 0, dn = 0;
            if (want_move) {
                const int d = (dx_t > 0) ? HOLD_RIGHT : HOLD_LEFT;
                if (s.hd != d) dir = d;
            }
            if (want_rot && s.rh != rot_btn) rot = rot_btn;
            const int tucked = at_row && s.x == px && ((s.rot & 3) == (prot & 3));
            if (dir == HOLD_NEUTRAL && (s.y < py || tucked)) dn = 1;

            const int rot_before = s.rot;
            if (tmp) {
                if (frames >= SCRIPT_REC_CAP) { ok = 0; break; }
                tmp[frames] = (u8)v4_act(dir, dn, rot);
            }
            v4_step(fm, thr, &s, v4_act(dir, dn, rot));
            frames += 1;
            if (rot != 0 && s.rot != rot_before) nrots -= 1;
            if (!s.locked && s.y > py) { ok = 0; break; }
        }
        if (ok && s.locked && s.x == px && s.y == py && ((s.rot & 3) == (prot & 3))) {
            if (best < 0 || frames < best) {
                best = frames;
                if (rec && tmp) {
                    for (int i = 0; i < frames; ++i) rec[i] = tmp[i];
                }
            }
        }
    }
    return best;
}

// Newer tuck family (reach_native/drm_reach_full.c:2133-2185, the drmc-rl
// copy): descend the shaft at cx in a chosen descent rotation (possibly
// vertical, fitting single-column shafts), then at the target row rotate to
// the final orientation and slide under the overhang. Descent rotations
// {prot, 1, 3} x finish orders {rotate-first, slide-first}. NOT used for the
// bit-exact phase-2 UB table (which mirrors the game_engine copy); used only
// by the scripts stage, where any cost-matching rollout is valid.
__device__ int v4_greedy_tuck2(
    const FitMask* fm, int thr, const V4State* spawn, int px, int py, int prot, int cx,
    u8* rec, u8* tmp
) {
    int best = -1;
    const int drots[3] = {-1, 1, 3};  // -1 => use prot
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

                if (tmp) {
                    if (frames >= SCRIPT_REC_CAP) { ok = 0; break; }
                    tmp[frames] = (u8)v4_act(dir, dn, rot);
                }
                v4_step(fm, thr, &s, v4_act(dir, dn, rot));
                frames += 1;
                if (!s.locked && s.y > py) { ok = 0; break; }
            }
            if (ok && s.locked && s.x == px && s.y == py && ((s.rot & 3) == (prot & 3))) {
                if (best < 0 || frames < best) {
                    best = frames;
                    if (rec && tmp) {
                        for (int i = 0; i < frames; ++i) rec[i] = tmp[i];
                    }
                }
            }
        }
    }
    return best;
}

// drm_reach_full.c:2187-2213
__device__ int v4_composite_succ(
    const FitMask* fm, int x, int y, int rot, u16 succ[16]
) {
    int n = 0;
    for (int dy = 0; dy <= 1; ++dy) {
        const int y2 = y + dy;
        if (dy) {
            if (y2 >= GRID_H || !fits_masked(fm, x, y2, rot)) continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int x2 = x + dx;
            if (dx && !fits_masked(fm, x2, y2, rot)) continue;
            succ[n++] = (u16)pose_index(x2, y2, rot);
            for (int rotation = 1; rotation <= 2; ++rotation) {
                for (int hl = 0; hl <= 1; ++hl) {
                    int rx = x2, rrot = rot;
                    apply_rotation_masked(fm, &rx, y2, &rrot, rotation, hl);
                    if (rx == x2 && ((rrot & 3) == (rot & 3))) continue;
                    succ[n++] = (u16)pose_index(rx, y2, rrot);
                }
            }
        }
    }
    return n;
}

// drm_reach_full.c:2219-2263
__device__ int v4_greedy_follow(
    const FitMask* fm, int thr, const V4State* spawn, int target_pose, const u8* gd,
    u8* rec = nullptr
) {
    V4State s = *spawn;
    int frames = 0;
    const int guard = 4 * (GRID_H + 1) * (thr + 2) + 64;
    int stall = 0;
    const int stall_limit = 2 * (thr + 2) + 8;
    u8 cur_gd = gd[pose_index(s.x, s.y, s.rot & 3)];
    if (cur_gd == 0xFFu) return -1;

    const u8 ACTS[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};

    while (!s.locked && frames < guard) {
        int best_act = -1;
        int best_score = 0x7FFFFFFF;
        V4State best_state;
        const int n_act = ((s.p & 1) == 0) ? 12 : 9;
        for (int ai = 0; ai < n_act; ++ai) {
            int act = (int)ACTS[ai];
            if ((s.p & 1) != 0 && ai >= 3 && ai < 6) continue;
            V4State t = s;
            v4_step(fm, thr, &t, act);
            int score;
            if (t.locked) {
                score = (pose_index(t.x, t.y, t.rot & 3) == target_pose) ? -1 : 0x7FFFFFFF;
            } else {
                const u8 g2 = gd[pose_index(t.x, t.y, t.rot & 3)];
                score = (g2 == 0xFFu) ? 0x7FFFFFFE : (int)g2;
            }
            if (score < best_score) { best_score = score; best_act = act; best_state = t; }
        }
        if (best_act < 0 || best_score >= 0x7FFFFFFE) return -1;
        if (rec) {
            if (frames >= SCRIPT_REC_CAP) return -1;
            rec[frames] = (u8)best_act;
        }
        s = best_state;
        frames += 1;
        if (s.locked) break;
        const u8 g_now = gd[pose_index(s.x, s.y, s.rot & 3)];
        if (g_now < cur_gd) { cur_gd = g_now; stall = 0; }
        else if (++stall > stall_limit) return -1;
    }
    if (s.locked && pose_index(s.x, s.y, s.rot & 3) == target_pose) return frames;
    return -1;
}

// ---------------------------------------------------------------------------
// Phase 1+2 for one instance (block-cooperative).
// Mirrors drm_reach_bfs_v4 phases 1a/1b (drm_reach_full.c:2304-2388).
// On return: ws->wanted, ws->wanted_ids[0..n), ws->gd, ws->ub are final.
// Returns n_wanted via shared broadcast.
// ---------------------------------------------------------------------------
struct Phase12Shared {
    FitMask fm;
    u8  scratch_visited[N_POSES];
    u16 scratch_queue[N_POSES];
    int n_wanted;
    int spawn_ok;
    V4State spawn;
    int thr;
};

__device__ int run_phase12(const Instance* inst, Workspace* ws, Phase12Shared* sh) {
    const int tid = threadIdx.x;

    // --- setup (thread 0): sanitize inputs exactly as v4 (:2283-2302) ---
    if (tid == 0) {
        int sx = inst->sx, sy = inst->sy, srot = inst->srot;
        int thr = inst->thr;
        if (thr < 0) thr = 0;
        if (thr > 0x7F) thr = 0x7F;
        int sc = inst->sc;
        if (sc < 0) sc = 0;
        if (sc > thr) sc = thr;
        int hv = inst->hv & 0x0F;
        int hd = inst->hd;
        if (hd < 0 || hd > 2) hd = 0;
        int p = inst->p & FAST_DROP_MASK;
        int rh = inst->rh;
        if (rh < 0 || rh > 2) rh = 0;
        sh->thr = thr;

        build_fit_masks(inst->cols, &sh->fm);
        sh->spawn_ok = ((unsigned)sx < (unsigned)GRID_W) && ((unsigned)sy < (unsigned)GRID_H)
                       && fits_masked(&sh->fm, sx, sy, srot & 3);

        sh->spawn.x = sx; sh->spawn.y = sy; sh->spawn.rot = srot & 3;
        sh->spawn.sc = sc; sh->spawn.hv = hv; sh->spawn.hd = hd;
        sh->spawn.p = p; sh->spawn.rh = rh; sh->spawn.locked = 0;

        sh->n_wanted = 0;
        if (sh->spawn_ok) {
            build_wanted_terminal_poses_reachable_serial(
                &sh->fm, sx, sy, srot & 3, ws->wanted, sh->scratch_visited, sh->scratch_queue);
            // wanted_ids in ascending pose order (v4 :2334-2339)
            int nw = 0;
            for (int pose = 0; pose < N_POSES; ++pose) {
                ws->pose_wi[pose] = 0xFFFFu;
                if (ws->wanted[pose]) {
                    ws->pose_wi[pose] = (u16)nw;
                    ws->wanted_ids[nw++] = (u16)pose;
                }
            }
            sh->n_wanted = nw;
        } else {
            for (int i = 0; i < N_POSES; ++i) ws->wanted[i] = 0;
        }
    }
    __syncthreads();
    if (!sh->spawn_ok || sh->n_wanted == 0) {
        for (int i = tid; i < N_POSES; i += blockDim.x) ws->ub[i] = COST_INF;
        __syncthreads();
        return sh->n_wanted;
    }

    // --- reverse adjacency (thread 0, serial: order determines the 20-cap
    //     truncation, which must match the CPU exactly; :2314-2330) ---
    if (tid == 0) {
        for (int i = 0; i < N_POSES; ++i) ws->radj_n[i] = 0;
        for (int pose = 0; pose < N_POSES; ++pose) {
            const int x = pose & 7;
            const int y = (pose >> 3) & 15;
            const int rot = (pose >> 7) & 3;
            if (!fits_masked(&sh->fm, x, y, rot)) continue;
            u16 succ[16];
            const int ns = v4_composite_succ(&sh->fm, x, y, rot, succ);
            for (int i = 0; i < ns; ++i) {
                const u16 s2 = succ[i];
                if (s2 == (u16)pose) continue;
                u8 rn = ws->radj_n[s2];
                if (rn < 20) { ws->radj[s2][rn] = (u16)pose; ws->radj_n[s2] = rn + 1; }
            }
        }
    }
    for (int i = tid; i < N_POSES; i += blockDim.x) ws->ub[i] = COST_INF;
    __syncthreads();

    // --- gd fields + greedy UBs: one thread per wanted pose ---
    const int n_wanted = sh->n_wanted;
    const int thr = sh->thr;
    for (int wi = tid; wi < n_wanted; wi += blockDim.x) {
        const int pose = (int)ws->wanted_ids[wi];
        u8* gd = ws->gd[wi];
        // backward BFS over composite pose graph (:2340-2356)
        for (int i = 0; i < N_POSES; ++i) gd[i] = 0xFFu;
        u16 q[N_POSES];
        int qh = 0, qt = 0;
        gd[pose] = 0;
        q[qt++] = (u16)pose;
        while (qh < qt) {
            const u16 cu = q[qh++];
            const u8 dc = gd[cu];
            const int rn = (int)ws->radj_n[cu];
            for (int i = 0; i < rn; ++i) {
                const u16 pv = ws->radj[cu][i];
                if (gd[pv] != 0xFFu) continue;
                gd[pv] = (u8)(dc + 1);
                q[qt++] = pv;
            }
        }
        // greedy UB chain with v4's conditional fallbacks (:2367-2388)
        const int px = pose & 7;
        const int py = (pose >> 3) & 15;
        const int prot = (pose >> 7) & 3;
        u16 ub = COST_INF;
        for (int order = 0; order < 6; ++order) {
            const int f = v4_greedy_try(&sh->fm, thr, &sh->spawn, px, py, prot, order);
            if (f > 0 && (u16)f < ub) ub = (u16)f;
        }
        if (ub == COST_INF) {
            for (int d = -2; d <= 2; ++d) {
                if (d == 0) continue;
                const int cx = px + d;
                if ((unsigned)cx >= (unsigned)GRID_W) continue;
                const int f = v4_greedy_tuck(&sh->fm, thr, &sh->spawn, px, py, prot, cx);
                if (f > 0 && (u16)f < ub) ub = (u16)f;
            }
        }
        if (ub == COST_INF) {
            const int f = v4_greedy_follow(&sh->fm, thr, &sh->spawn, pose, gd);
            if (f > 0 && (u16)f < ub) ub = (u16)f;
        }
        ws->ub[pose] = ub;
    }
    __syncthreads();
    return n_wanted;
}

// ---------------------------------------------------------------------------
// Debug kernel: run phases 1+2 and dump wanted/ub/gd tables per instance.
// gd output: [instance][gd_cap][512] wanted-major, rows past n_wanted zeroed.
// ---------------------------------------------------------------------------
extern "C" __global__ void drm_reach_debug_phase12_kernel(
    const Instance* __restrict__ insts,
    int n,
    unsigned long long* cursor,
    Workspace* ws_arena,
    u8*  __restrict__ out_wanted,   // n x 512
    u16* __restrict__ out_ub,       // n x 512
    u8*  __restrict__ out_gd,       // n x gd_cap x 512
    int gd_cap,
    int* __restrict__ out_n_wanted  // n
) {
    __shared__ Phase12Shared sh;
    __shared__ int s_idx;
    Workspace* ws = ws_arena + blockIdx.x;

    for (;;) {
        if (threadIdx.x == 0) s_idx = (int)atomicAdd(cursor, 1ull);
        __syncthreads();
        const int idx = s_idx;
        __syncthreads();
        if (idx >= n) return;

        const Instance inst = insts[idx];
        const int nw = run_phase12(&inst, ws, &sh);

        if (threadIdx.x == 0) out_n_wanted[idx] = nw;
        for (int i = threadIdx.x; i < N_POSES; i += blockDim.x) {
            out_wanted[(size_t)idx * N_POSES + i] = ws->wanted[i];
            out_ub[(size_t)idx * N_POSES + i] = ws->ub[i];
        }
        for (int r = 0; r < gd_cap; ++r) {
            u8* dst = out_gd + ((size_t)idx * gd_cap + r) * N_POSES;
            if (r < nw) {
                for (int i = threadIdx.x; i < N_POSES; i += blockDim.x) dst[i] = ws->gd[r][i];
            } else {
                for (int i = threadIdx.x; i < N_POSES; i += blockDim.x) dst[i] = 0;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Phase 3: gated exact BFS, bit-sliced (v3 layout + v4 gate + early exit).
// ---------------------------------------------------------------------------
struct Phase3Shared {
    int cur_n, next_n;
    int found_wanted, pending_resolves, do_refresh;
    int stop;                      // all wanted resolved
    u32 resolved_mask[16];         // bit per pose, 512 bits
    u8  amask[GRID_H][4];          // gate mask for children at next_depth
    u32 n_touched_v;
};

// Full-state id for parent chains (re-BFS only; requires thr < 64 so sc fits
// 6 bits): 23-bit id = ((p*6336 + key)*8 + x)*64 + sc. Parent entries pack
// id | act << 25 into a u32.
__device__ inline u32 state_id(int p, u32 key, int x, int sc) {
    return ((((u32)p * WS_NKEYS_P + key) * 8u + (u32)x) << 6) | (u32)sc;
}

// One frontier-key expansion task: loads the key's block once, computes the
// gravity shift once, then applies every legal action. Mirrors the v3
// expansion body (drm_reach_full.c:1677-1907) with the v4 allowance gate at
// enqueue (drm_reach_full.c:2623-2655).
//
// `two_words`: sc bits fit one u64 when thr < 64 (all real gameplay); the
// second word's loads/atomics are skipped entirely then.
//
// REC=true (scripts re-BFS): record a parent entry for every newly visited
// state (the atomicOr winner is the unique writer), and record term parents
// for locks at poses still needing scripts (script_len == 0xFFFF). In REC
// mode the caller guarantees thr < 64.
template <bool REC>
__device__ void expand_key(
    const FitMask* fm, Workspace* ws, Phase3Shared* s3,
    u32 key, const u8* acts, int n_act, int p_cur, int cur_buf, u16 next_depth,
    u64 thr_keep_lo, u64 thr_keep_hi, int thr, int two_words,
    u16* out_costs, u32* parents
) {
    // decode key -> (y, rot, micro, rh); hv/hd from micro (hv dead when hd==N)
    const int y = (int)(key % 16u);
    const int rot = (int)((key / 16u) % 4u);
    const u32 m = (key / 64u) % 33u;
    const int rh_prev = (int)(key / 2112u);
    const int hd_prev = (m == 0u) ? 0 : ((m <= 16u) ? 1 : 2);
    const int hv0 = (m == 0u) ? 0 : ((m <= 16u) ? (int)(m - 1u) : (int)(m - 17u));

    const u64 (*blk)[LANE_WORDS] = ws->accum[cur_buf][key];
    const int even_parity = ((p_cur & FAST_DROP_MASK) == 0);
    const int p_next = p_cur ^ 1;

    // ---- Y stage, variant A (shared across actions; v3 :1687-1717) ----
    const int vparity = rot & 1;
    const u8 drop_ok = (y + 1 < GRID_H) ? fm->m[vparity][y + 1] : 0u;

    u64 stay_lo[GRID_W], stay_hi[GRID_W];
    u8 dropA_x = 0, stayA_x = 0, allx = 0;
    for (int x = 0; x < GRID_W; ++x) {
        const u64 lo = blk[x][0];
        const u64 hi = two_words ? blk[x][1] : 0ull;
        if (lo | hi) allx |= (u8)(1u << x);
        int dropped;
        if (thr < 64) dropped = (int)((lo >> thr) & 1ull);
        else dropped = (int)((hi >> (thr - 64)) & 1ull);
        dropA_x |= (u8)(dropped << x);
        stay_lo[x] = (lo << 1) & thr_keep_lo;
        stay_hi[x] = two_words ? (((hi << 1) | (lo >> 63)) & thr_keep_hi) : 0ull;
        if (stay_lo[x] | stay_hi[x]) stayA_x |= (u8)(1u << x);
    }
    const u8 lockA_x = (u8)(dropA_x & (u8)(~drop_ok));
    const u8 fallA_x = (u8)(dropA_x & drop_ok);
    const u8 lockB_x = (u8)(allx & (u8)(~drop_ok));
    const u8 fallB_x = (u8)(allx & drop_ok);

    for (int ai = 0; ai < n_act; ++ai) {
        const int act = (int)acts[ai];
        const int dir = (int)ACT_HOLD_DIR[act];
        const int dn = (int)ACT_HOLD_DOWN[act];
        const int rotation = (int)ACT_ROT[act];

        const int down_only = (dn != 0) && (dir == HOLD_NEUTRAL);
        const int use_B = even_parity && down_only;

        // Locks (first-found wins; level-synchronous so same-depth races write
        // the same value; resolved bookkeeping is atomic so counting is exact).
        u8 lock_x = use_B ? lockB_x : lockA_x;
        while (lock_x) {
            const int lx = __ffs((unsigned)lock_x) - 1;
            lock_x &= (u8)(lock_x - 1u);
            const int pose = pose_index(lx, y, rot);
            if (REC) {
                // Re-BFS: record the lock's parent only for poses that still
                // need a script. Exactness guarantees the first lock at such a
                // pose happens at depth == its cost.
                if (ws->script_len[pose] != 0xFFFFu) continue;
                const u32 old = atomicOr(&s3->resolved_mask[pose >> 5], 1u << (pose & 31));
                if (old & (1u << (pose & 31))) continue;
                int src_sc;
                if (use_B) {
                    const u64 lane = ws->accum[cur_buf][key][lx][0];
                    src_sc = (int)(__ffsll((long long)lane) - 1);
                } else {
                    src_sc = thr;
                }
                ws->term_parent[pose] = state_id(p_cur, key, lx, src_sc) | ((u32)act << 25);
                atomicAdd(&s3->found_wanted, 1);
                atomicAdd(&s3->pending_resolves, 1);
            } else {
                if (out_costs[pose] != (u16)COST_INF) continue;
                out_costs[pose] = next_depth;
                if (ws->wanted[pose]) {
                    const u32 old = atomicOr(&s3->resolved_mask[pose >> 5], 1u << (pose & 31));
                    if (!(old & (1u << (pose & 31)))) {
                        atomicAdd(&s3->found_wanted, 1);
                        atomicAdd(&s3->pending_resolves, 1);
                    }
                }
            }
        }

        // Y groups: {stay at y (variant A), fall to y+1 (sc -> 0)}
        struct YG { int y; u8 xm; int sc_zero; } ygs[2];
        int n_yg = 0;
        if (!use_B) {
            if (stayA_x) { ygs[n_yg].y = y; ygs[n_yg].xm = stayA_x; ygs[n_yg].sc_zero = 0; n_yg++; }
            if (fallA_x) { ygs[n_yg].y = y + 1; ygs[n_yg].xm = fallA_x; ygs[n_yg].sc_zero = 1; n_yg++; }
        } else {
            if (fallB_x) { ygs[n_yg].y = y + 1; ygs[n_yg].xm = fallB_x; ygs[n_yg].sc_zero = 1; n_yg++; }
        }
        if (!n_yg) continue;

        // ---- X stage decision (v3 :1765-1777) ----
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
        const u8 g_xm = ygs[gi].xm;
        const int g_sc0 = ygs[gi].sc_zero;

        // X sub-groups (moved / blocked), v3 :1788-1807
        struct XG { u8 xm; i8 dx; u8 hv; } xgs[2];
        int n_xg = 0;
        if (!allow_move || dir == HOLD_NEUTRAL) {
            xgs[0].xm = g_xm; xgs[0].dx = 0; xgs[0].hv = (u8)(hv & 0x0F);
            n_xg = 1;
        } else if (hold_right) {
            const u8 fits_row = fm->m[vparity][gy];
            const u8 ok = (u8)(fits_row >> 1);
            const u8 mv = (u8)(g_xm & ok);
            const u8 bl = (u8)(g_xm & (u8)(~ok));
            if (mv) { xgs[n_xg].xm = mv; xgs[n_xg].dx = 1; xgs[n_xg].hv = (u8)(hv & 0x0F); n_xg++; }
            if (bl) { xgs[n_xg].xm = bl; xgs[n_xg].dx = 0; xgs[n_xg].hv = (u8)HOR_BLOCKED; n_xg++; }
        } else {
            const u8 fits_row = fm->m[vparity][gy];
            const u8 ok = (u8)((fits_row << 1) & 0xFFu);
            const u8 mv = (u8)(g_xm & ok);
            const u8 bl = (u8)(g_xm & (u8)(~ok));
            if (mv) { xgs[n_xg].xm = mv; xgs[n_xg].dx = -1; xgs[n_xg].hv = (u8)(hv & 0x0F); n_xg++; }
            if (bl) { xgs[n_xg].xm = bl; xgs[n_xg].dx = 0; xgs[n_xg].hv = (u8)HOR_BLOCKED; n_xg++; }
        }

        for (int xi = 0; xi < n_xg; ++xi) {
            const u8 xm_pre = xgs[xi].xm;
            const i8 dx = xgs[xi].dx;
            const u8 hv_out = xgs[xi].hv;
            const u8 xm_moved = (dx == 1) ? (u8)(xm_pre << 1)
                              : (dx == -1) ? (u8)(xm_pre >> 1)
                              : xm_pre;

            // Rotate sub-groups (v3 :1817-1847)
            struct RG { u8 xm; u8 rot; i8 dx2; } rgs[3];
            int n_rg = 0;
            if (!rotation_pressed) {
                rgs[0].xm = xm_moved; rgs[0].rot = (u8)rot; rgs[0].dx2 = 0;
                n_rg = 1;
            } else {
                int rot1 = rot;
                if (rotation == 1) rot1 = (rot - 1) & 3;
                else rot1 = (rot + 1) & 3;
                if ((rot1 & 1) != 0) {
                    const u8 fit_v = fm->m[1][gy];
                    const u8 acc = (u8)(xm_moved & fit_v);
                    const u8 rej = (u8)(xm_moved & (u8)(~fit_v));
                    if (acc) { rgs[n_rg].xm = acc; rgs[n_rg].rot = (u8)rot1; rgs[n_rg].dx2 = 0; n_rg++; }
                    if (rej) { rgs[n_rg].xm = rej; rgs[n_rg].rot = (u8)rot; rgs[n_rg].dx2 = 0; n_rg++; }
                } else {
                    const u8 fit_h = fm->m[0][gy];
                    const u8 acc_inplace = (u8)(xm_moved & fit_h);
                    const u8 rej_inplace = (u8)(xm_moved & (u8)(~fit_h));
                    const u8 ok_left = (u8)((fit_h << 1) & 0xFFu);
                    const u8 dbl = hold_left ? (u8)(acc_inplace & ok_left) : 0u;
                    const u8 acc_noshift = (u8)(acc_inplace & (u8)(~dbl));
                    const u8 kick = (u8)(rej_inplace & ok_left);
                    const u8 rej = (u8)(rej_inplace & (u8)(~kick));
                    const u8 shifted_src = (u8)(dbl | kick);
                    if (shifted_src) { rgs[n_rg].xm = (u8)(shifted_src >> 1); rgs[n_rg].rot = (u8)rot1; rgs[n_rg].dx2 = -1; n_rg++; }
                    if (acc_noshift) { rgs[n_rg].xm = acc_noshift; rgs[n_rg].rot = (u8)rot1; rgs[n_rg].dx2 = 0; n_rg++; }
                    if (rej) { rgs[n_rg].xm = rej; rgs[n_rg].rot = (u8)rot; rgs[n_rg].dx2 = 0; n_rg++; }
                }
            }

            for (int ri = 0; ri < n_rg; ++ri) {
                u8 xm_fin = rgs[ri].xm;
                if (!xm_fin) continue;
                const u8 rot_fin = rgs[ri].rot;
                // v4 gate: children that provably cannot improve any
                // unresolved pose are dropped (uniform across sc).
                xm_fin &= s3->amask[gy][rot_fin & 3];
                if (!xm_fin) continue;
                const int total_dx = (int)dx + (int)rgs[ri].dx2;

                const u32 m_out = v2_micro((u32)dir, (u32)(hv_out & 15u));
                const u32 key2 = (u32)gy + (u32)V3_K_ROT * (u32)(rot_fin & 3)
                                 + (u32)V3_K_M * m_out + (u32)V3_K_RH * (u32)rotation;

                u64 (*vis)[LANE_WORDS] = ws->visited[p_next][key2];
                u64 (*acc)[LANE_WORDS] = ws->accum[cur_buf ^ 1][key2];

                u64 any_new = 0;
                u8 xm_iter = xm_fin;
                while (xm_iter) {
                    const int xo = __ffs((unsigned)xm_iter) - 1;
                    xm_iter &= (u8)(xm_iter - 1u);
                    const int xs = xo - total_dx;
                    if ((unsigned)xs >= 8u) continue;
                    u64 slo, shi;
                    if (g_sc0) { slo = 1ull; shi = 0ull; }
                    else { slo = stay_lo[xs]; shi = stay_hi[xs]; }
                    if (!(slo | shi)) continue;
                    const u64 old_lo = slo ? atomicOr(&vis[xo][0], slo) : vis[xo][0];
                    const u64 old_hi = shi ? atomicOr(&vis[xo][1], shi) : 0ull;
                    const u64 nlo = slo & ~old_lo;
                    const u64 nhi = shi & ~old_hi;
                    if (!(nlo | nhi)) continue;
                    if (nlo) atomicOr(&acc[xo][0], nlo);
                    if (nhi) atomicOr(&acc[xo][1], nhi);
                    if (REC) {
                        // Unique writer for each new bit: record its parent.
                        // Dest sc bit b came from src bit b-1 (gravity shift),
                        // or from bit thr / lowest set bit on a fall (sc -> 0).
                        u64 nb = nlo;   // REC mode: thr < 64, no hi word
                        while (nb) {
                            const int b = (int)(__ffsll((long long)nb) - 1);
                            nb &= nb - 1;
                            int src_sc;
                            if (!g_sc0) {
                                src_sc = b - 1;
                            } else if (!use_B) {
                                src_sc = thr;
                            } else {
                                const u64 lane = ws->accum[cur_buf][key][xs][0];
                                src_sc = (int)(__ffsll((long long)lane) - 1);
                            }
                            parents[state_id(p_next, key2, xo, b)] =
                                state_id(p_cur, key, xs, src_sc) | ((u32)act << 25);
                        }
                    }
                    any_new |= nlo | nhi;
                }
                if (!any_new) continue;

                // Track dirtied visited blocks (cleared at instance end).
                if (ws->visited_flag[p_next][key2] == 0) {
                    // benign race: multiple threads may pass the check; the
                    // atomicExch below admits exactly one.
                    if (atomicExch(&ws->visited_flag[p_next][key2], 1u) == 0) {
                        const u32 t = atomicAdd(&s3->n_touched_v, 1u);
                        ws->touched_v[t] = (u32)p_next * WS_NKEYS_P + key2;
                    }
                }
                // Enqueue key into the next frontier exactly once.
                if (ws->accum_flag[cur_buf ^ 1][key2] == 0) {
                    if (atomicExch(&ws->accum_flag[cur_buf ^ 1][key2], 1u) == 0) {
                        const int slot = atomicAdd(&s3->next_n, 1);
                        ws->frontier[cur_buf ^ 1][slot] = key2;
                    }
                }
            }
        }
        }
    }
}

// Refresh gate allowances (v4 V4_REFRESH_G, drm_reach_full.c:2399-2417).
// Parallel over poses; each thread scans the unresolved wanted set.
__device__ void refresh_G(Workspace* ws, Phase3Shared* s3, int n_wanted) {
    const int NEG = -(0x7FFFFFFF / 4);
    const int POS = 0x7FFFFFFF / 4;
    for (int q2 = threadIdx.x; q2 < N_POSES; q2 += blockDim.x) {
        int g = NEG;
        const int yq = (q2 >> 3) & 15;
        for (int wi = 0; wi < n_wanted; ++wi) {
            const int p = (int)ws->wanted_ids[wi];
            if (s3->resolved_mask[p >> 5] & (1u << (p & 31))) continue;
            const u8 gdq = ws->gd[wi][q2];
            if (gdq == 0xFFu) continue;
            const int prow = (p >> 3) & 15;
            const u16 ubp = ws->ub[p];
            const int B = (ubp == (u16)COST_INF) ? POS : (int)ubp;
            int lb = 2 * (prow - yq);
            if (lb < (int)gdq) lb = (int)gdq;
            const int allow = B - 1 - lb;
            if (allow > g) g = allow;
        }
        ws->G[q2] = g;
    }
}

// Full solve for one instance: phases 1+2 must have run (ws->wanted/ub/gd).
__device__ void run_phase3(
    const Instance* inst, Workspace* ws, Phase12Shared* sh, Phase3Shared* s3,
    int n_wanted, u16* out_costs, bool certify_witnesses
) {
    const int tid = threadIdx.x;

    for (int i = tid; i < N_POSES; i += blockDim.x) out_costs[i] = (u16)COST_INF;
    __syncthreads();
    if (!sh->spawn_ok) return;
    // n_wanted == 0 with a valid spawn happens when every reachable terminal
    // pose is macro-illegal (vertical at y==0 / horizontal at x==7). CPU v4
    // falls back to plain v2 (drm_reach_full.c:2307); the equivalent here is
    // the same BFS with the gate wide open and no early exit.
    const int gated = (n_wanted > 0);

    const int thr = sh->thr;
    const int sc0 = sh->spawn.sc;
    const int sx = sh->spawn.x, sy = sh->spawn.y, srot = sh->spawn.rot;
    const int p0 = sh->spawn.p;

    int max_frames = (int)inst->max_frames;
    if (max_frames <= 0) return;
    const int mlf = compute_max_lock_frames(sy, sc0, thr);
    if (max_frames > mlf) max_frames = mlf;

    // sc keep masks (v3 :1622-1628)
    u64 thr_keep_lo, thr_keep_hi;
    if (thr < 63) { thr_keep_lo = (1ull << (thr + 1)) - 1ull; thr_keep_hi = 0; }
    else if (thr == 63) { thr_keep_lo = ~0ull; thr_keep_hi = 0; }
    else if (thr < 127) { thr_keep_lo = ~0ull; thr_keep_hi = (1ull << (thr - 63)) - 1ull; }
    else { thr_keep_lo = ~0ull; thr_keep_hi = ~0ull; }

    // init shared control + seed
    if (tid == 0) {
        s3->found_wanted = 0;
        s3->pending_resolves = 0;
        s3->do_refresh = 0;
        s3->stop = 0;
        s3->n_touched_v = 0;
        for (int i = 0; i < 16; ++i) s3->resolved_mask[i] = 0;

        const u32 start_m = v2_micro((u32)sh->spawn.hd, (u32)sh->spawn.hv);
        const u32 start_key = (u32)sy + (u32)V3_K_ROT * (u32)(srot & 3)
                              + (u32)V3_K_M * start_m + (u32)V3_K_RH * (u32)sh->spawn.rh;
        u64* vis = &ws->visited[p0][start_key][sx][0];
        u64* acc = &ws->accum[0][start_key][sx][0];
        if (sc0 < 64) { vis[0] = 1ull << sc0; acc[0] = 1ull << sc0; }
        else { vis[1] = 1ull << (sc0 - 64); acc[1] = 1ull << (sc0 - 64); }
        ws->visited_flag[p0][start_key] = 1;
        ws->touched_v[s3->n_touched_v++] = (u32)p0 * WS_NKEYS_P + start_key;
        ws->accum_flag[0][start_key] = 1;
        ws->frontier[0][0] = start_key;
        s3->cur_n = 1;
        s3->next_n = 0;
    }
    __syncthreads();

    // A feasible witness is exact when it meets the fastest possible vertical
    // schedule. In the collision-free relaxation, choose down-only whenever
    // fast-drop parity permits it and otherwise let the natural speed counter
    // advance. Reaching row py and locking needs (py-sy) successful drops plus
    // one failed drop. Obstacles, lateral movement, and rotation can only make
    // the real path slower, so equality with a feasible UB certifies optimality.
    // Pre-resolving these witnesses shrinks the exact BFS target count. They
    // deliberately remain in the allowance gates: the truncated reverse
    // adjacency used by gd is safe for the original union gate, but removing
    // one target's permissive region can over-prune paths to another target.
    if (gated && certify_witnesses) {
        for (int wi = tid; wi < n_wanted; wi += blockDim.x) {
            const int pose = (int)ws->wanted_ids[wi];
            const u16 ub = ws->ub[pose];
            if (ub == (u16)COST_INF) continue;
            int drops_left = ((pose >> 3) & 15) - sy + 1;
            if (drops_left < 1) drops_left = 1;
            int p = p0;
            int sc = sc0;
            int lb = 0;
            while (drops_left > 0) {
                bool drop = false;
                if ((p & FAST_DROP_MASK) == 0) {
                    drop = true;
                    sc = 0;
                } else {
                    sc += 1;
                    if (sc > thr) {
                        drop = true;
                        sc = 0;
                    }
                }
                p ^= 1;
                lb += 1;
                if (drop) drops_left -= 1;
            }
            if ((int)ub != lb) continue;
            out_costs[pose] = ub;
            atomicAdd(&s3->found_wanted, 1);
        }
        __syncthreads();
        if (tid == 0 && s3->found_wanted >= n_wanted) s3->stop = 1;
        __syncthreads();
    }

    if (gated) {
        refresh_G(ws, s3, n_wanted);
    }
    __syncthreads();

    int p_cur = p0;
    int cur_buf = 0;

    for (int depth = 0; depth < max_frames; ++depth) {
        if (s3->cur_n == 0 || s3->stop) break;
        const u16 next_depth = (u16)(depth + 1);

        // gate refresh cadence (v4 :2473-2476)
        if (tid == 0) {
            s3->do_refresh = 0;
            if (gated && s3->pending_resolves >= 4) {
                s3->do_refresh = 1;
                s3->pending_resolves = 0;
            }
        }
        __syncthreads();
        if (s3->do_refresh) {
            refresh_G(ws, s3, n_wanted);
            __syncthreads();
        }
        // allowance bitmask for children at next_depth (v4 :2478-2487)
        for (int i = tid; i < GRID_H * 4; i += blockDim.x) {
            const int yy = i & 15;
            const int rr = i >> 4;
            u8 mask = 0xFFu;
            if (gated) {
                mask = 0;
                const int base = rr * 128 + yy * 8;
                for (int x = 0; x < 8; ++x) {
                    if ((int)next_depth <= ws->G[base + x]) mask |= (u8)(1u << x);
                }
            }
            s3->amask[yy][rr] = mask;
        }
        __syncthreads();

        // expansion: one task per frontier key (block loaded once, all actions)
        const int even_parity = ((p_cur & FAST_DROP_MASK) == 0);
        const int n_act = even_parity ? 12 : 9;
        const u8* acts = even_parity ? ACTS_EVEN : ACTS_ODD;
        const int cur_n = s3->cur_n;
        const int two_words = (thr >= 64);
        for (int t = tid; t < cur_n; t += blockDim.x) {
            expand_key<false>(&sh->fm, ws, s3, ws->frontier[cur_buf][t], acts, n_act,
                              p_cur, cur_buf, next_depth,
                              thr_keep_lo, thr_keep_hi, thr, two_words,
                              out_costs, nullptr);
        }
        __syncthreads();

        if (tid == 0 && gated && s3->found_wanted >= n_wanted) s3->stop = 1;

        // clear consumed frontier blocks + flags; swap buffers
        for (int i = tid; i < cur_n; i += blockDim.x) {
            const u32 key = ws->frontier[cur_buf][i];
            for (int x = 0; x < GRID_W; ++x) {
                ws->accum[cur_buf][key][x][0] = 0;
                ws->accum[cur_buf][key][x][1] = 0;
            }
            ws->accum_flag[cur_buf][key] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            s3->cur_n = s3->next_n;
            s3->next_n = 0;
        }
        cur_buf ^= 1;
        p_cur ^= 1;
        __syncthreads();
    }

    // Epilogue: clear whatever remains dirty for the next instance.
    // (a) unconsumed current frontier blocks, (b) partially built next
    // frontier blocks, (c) all dirtied visited blocks.
    for (int b = 0; b < 2; ++b) {
        const int nn = (b == 0) ? s3->cur_n : s3->next_n;
        for (int i = tid; i < nn; i += blockDim.x) {
            const u32 key = ws->frontier[(b == 0) ? cur_buf : (cur_buf ^ 1)][i];
            const int buf = (b == 0) ? cur_buf : (cur_buf ^ 1);
            for (int x = 0; x < GRID_W; ++x) {
                ws->accum[buf][key][x][0] = 0;
                ws->accum[buf][key][x][1] = 0;
            }
            ws->accum_flag[buf][key] = 0;
        }
    }
    __syncthreads();
    const u32 ntv = s3->n_touched_v;
    for (u32 i = tid; i < ntv; i += blockDim.x) {
        const u32 pk = ws->touched_v[i];
        const u32 pp = pk / WS_NKEYS_P;
        const u32 kk = pk % WS_NKEYS_P;
        for (int x = 0; x < GRID_W; ++x) {
            ws->visited[pp][kk][x][0] = 0;
            ws->visited[pp][kk][x][1] = 0;
        }
        ws->visited_flag[pp][kk] = 0;
    }
    __syncthreads();

    // Finalize from UBs (v4 :2688-2691)
    for (int pose = tid; pose < N_POSES; pose += blockDim.x) {
        const u16 ubp = ws->ub[pose];
        if (ubp == (u16)COST_INF) continue;
        if (out_costs[pose] == (u16)COST_INF || ubp < out_costs[pose]) out_costs[pose] = ubp;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Full costs kernel: phases 1+2+3 per instance.
// The workspace arena must be zero-initialized once at allocation; solves
// leave visited/accum/flags clean behind them (touched-list clearing).
// ---------------------------------------------------------------------------
extern "C" __global__ void drm_reach_costs_kernel(
    const Instance* __restrict__ insts,
    int n,
    unsigned long long* cursor,
    Workspace* ws_arena,
    u16* __restrict__ out_costs   // n x 512
) {
    __shared__ Phase12Shared sh;
    __shared__ Phase3Shared s3;
    __shared__ int s_idx;
    Workspace* ws = ws_arena + blockIdx.x;

    for (;;) {
        if (threadIdx.x == 0) s_idx = (int)atomicAdd(cursor, 1ull);
        __syncthreads();
        const int idx = s_idx;
        __syncthreads();
        if (idx >= n) return;

        const Instance inst = insts[idx];
        const int nw = run_phase12(&inst, ws, &sh);
        run_phase3(
            &inst, ws, &sh, &s3, nw,
            out_costs + (size_t)idx * N_POSES, true
        );
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Scripts stage: an optimal input script for every macro-legal reachable pose.
//
// Every finite-cost macro-legal pose is in the wanted set (the wanted flood
// fill is a superset of real reachability and applies the same macro-legality
// rules), so gd fields exist for all of them. Any greedy rollout whose frame
// count equals the exact cost IS an optimal script — scripts are verified by
// replay semantics, not byte-identity with CPU v1 (optimal scripts are not
// unique; CPU's tie-break is an enumeration artifact).
//
// Poses where no greedy variant matches get script_len = 0xFFFF and the
// instance is flagged; the host falls back to CPU v2 (parent-tracked) for
// those instances. Status bits: 1 = unmatched pose(s), 2 = script buffer
// overflow, 4 = greedy beat the "exact" cost (impossible; parity alarm).
// ---------------------------------------------------------------------------
__device__ void run_rebfs_scripts(
    const Instance* inst, Workspace* ws, Phase12Shared* sh, Phase3Shared* s3,
    int n_wanted, int n_pending, const u16* costs, u32* parents);

__device__ void run_scripts_stage(
    const Instance* inst, Workspace* ws, Phase12Shared* sh, Phase3Shared* s3,
    int n_wanted,
    const u16* costs,          // this instance's out_costs (global)
    u32* parents,              // parent arena slot (nullptr = no re-BFS)
    u16* out_offsets, u16* out_lengths, u8* out_scripts, int* out_status
) {
    const int tid = threadIdx.x;
    __shared__ int s_status;
    __shared__ int s_used;
    if (tid == 0) { s_status = 0; s_used = 0; }
    for (int i = tid; i < N_POSES; i += blockDim.x) ws->script_len[i] = 0;
    __syncthreads();
    if (!sh->spawn_ok || n_wanted == 0) {
        for (int i = tid; i < N_POSES; i += blockDim.x) {
            out_offsets[i] = 0; out_lengths[i] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            // If anything is reachable despite no wanted pose (boxed-in spawn,
            // macro-illegal locks only), there is nothing consumers can play.
            *out_status = 0;
        }
        return;
    }
    const int thr = sh->thr;

    for (int wi = tid; wi < n_wanted; wi += blockDim.x) {
        const int pose = (int)ws->wanted_ids[wi];
        const u16 c = costs[pose];
        if (c == (u16)COST_INF) continue;
        if ((int)c > SCRIPT_REC_CAP) {
            ws->script_len[pose] = 0xFFFFu;
            atomicOr(&s_status, 1);
            continue;
        }
        u8* rec = ws->script_scratch[pose];
        u8* tmp = ws->tuck_tmp[tid];
        int matched = 0;
        for (int order = 0; order < 6 && !matched; ++order) {
            const int f = v4_greedy_try(&sh->fm, thr, &sh->spawn,
                                        pose & 7, (pose >> 3) & 15, (pose >> 7) & 3,
                                        order, rec);
            if (f > 0 && (u16)f == c) matched = 1;
            else if (f > 0 && (u16)f < c) atomicOr(&s_status, 4);
        }
        for (int d = -2; d <= 2 && !matched; ++d) {
            if (d == 0) continue;
            const int cx = (pose & 7) + d;
            if ((unsigned)cx >= (unsigned)GRID_W) continue;
            const int f = v4_greedy_tuck(&sh->fm, thr, &sh->spawn,
                                         pose & 7, (pose >> 3) & 15, (pose >> 7) & 3,
                                         cx, rec, tmp);
            if (f > 0 && (u16)f == c) matched = 1;
            else if (f > 0 && (u16)f < c) atomicOr(&s_status, 4);
        }
        // Newer tuck family: descent rotations x finish orders, incl. the
        // pose's own column (single-column shafts).
        for (int d = -2; d <= 2 && !matched; ++d) {
            const int cx = (pose & 7) + d;
            if ((unsigned)cx >= (unsigned)GRID_W) continue;
            const int f = v4_greedy_tuck2(&sh->fm, thr, &sh->spawn,
                                          pose & 7, (pose >> 3) & 15, (pose >> 7) & 3,
                                          cx, rec, tmp);
            if (f > 0 && (u16)f == c) matched = 1;
            else if (f > 0 && (u16)f < c) atomicOr(&s_status, 4);
        }
        if (!matched) {
            const int f = v4_greedy_follow(&sh->fm, thr, &sh->spawn, pose,
                                           ws->gd[wi], rec);
            if (f > 0 && (u16)f == c) matched = 1;
            else if (f > 0 && (u16)f < c) atomicOr(&s_status, 4);
        }
        if (matched) {
            ws->script_len[pose] = c;
        } else {
            ws->script_len[pose] = 0xFFFFu;
            atomicOr(&s_status, 1);
        }
    }
    __syncthreads();

    // Exact fallback: parent-tracked re-BFS for the unmatched poses. Only
    // poses with cost <= SCRIPT_REC_CAP participate (others keep status 1).
    if ((s_status & 1) && parents != nullptr && thr < 64) {
        __shared__ int s_pending;
        if (tid == 0) s_pending = 0;
        __syncthreads();
        for (int wi = tid; wi < n_wanted; wi += blockDim.x) {
            const int pose = (int)ws->wanted_ids[wi];
            if (ws->script_len[pose] != 0xFFFFu) continue;
            if ((int)costs[pose] > SCRIPT_REC_CAP) continue;   // stays unmatched
            atomicAdd(&s_pending, 1);
        }
        __syncthreads();
        if (s_pending > 0) {
            run_rebfs_scripts(inst, ws, sh, s3, n_wanted, s_pending, costs, parents);
            // Recompute status bit 1: cleared iff nothing is left unmatched.
            if (tid == 0) {
                int still = 0;
                for (int wi = 0; wi < n_wanted; ++wi) {
                    if (ws->script_len[(int)ws->wanted_ids[wi]] == 0xFFFFu) still = 1;
                }
                if (!still) atomicAnd(&s_status, ~1);
            }
            __syncthreads();
        }
    }

    // Compact into the per-instance output buffer (pose-ascending offsets,
    // mirroring the CPU v1/v2 layout). Serial prefix on thread 0 (512 adds).
    if (tid == 0) {
        int used = 0;
        for (int pose = 0; pose < N_POSES; ++pose) {
            const u16 len = ws->script_len[pose];
            if (len == 0 || len == 0xFFFFu) {
                out_offsets[pose] = 0;
                out_lengths[pose] = 0;
                continue;
            }
            if (used + (int)len > SCRIPT_BUF_CAP) {
                s_status |= 2;
                out_offsets[pose] = 0;
                out_lengths[pose] = 0;
                ws->script_len[pose] = 0;   // skip in the copy pass
                continue;
            }
            out_offsets[pose] = (u16)used;
            out_lengths[pose] = (u16)len;
            used += (int)len;
        }
        s_used = used;
        *out_status = s_status;
    }
    __syncthreads();

    for (int pose = 0; pose < N_POSES; ++pose) {
        const u16 len = ws->script_len[pose];
        if (len == 0 || len == 0xFFFFu) continue;
        const u16 off = out_offsets[pose];
        for (int i = tid; i < (int)len; i += blockDim.x) {
            out_scripts[off + i] = ws->script_scratch[pose][i];
        }
    }
    __syncthreads();
}

// Gate refresh for the scripts re-BFS: same shape as refresh_G, but the
// "unresolved wanted" set is {poses still needing scripts} and B is the
// already-known exact cost — the tightest admissible gate possible.
__device__ void refresh_G_scripts(
    Workspace* ws, Phase3Shared* s3, int n_wanted, const u16* costs
) {
    const int NEG = -(0x7FFFFFFF / 4);
    for (int q2 = threadIdx.x; q2 < N_POSES; q2 += blockDim.x) {
        int g = NEG;
        const int yq = (q2 >> 3) & 15;
        for (int wi = 0; wi < n_wanted; ++wi) {
            const int p = (int)ws->wanted_ids[wi];
            if (ws->script_len[p] != 0xFFFFu) continue;
            if (s3->resolved_mask[p >> 5] & (1u << (p & 31))) continue;
            const u8 gdq = ws->gd[wi][q2];
            if (gdq == 0xFFu) continue;
            const int prow = (p >> 3) & 15;
            const int B = (int)costs[p];
            int lb = 2 * (prow - yq);
            if (lb < (int)gdq) lb = (int)gdq;
            const int allow = B - 1 - lb;
            if (allow > g) g = allow;
        }
        ws->G[q2] = g;
    }
}

// Exact parent-tracked re-BFS for poses no greedy rollout could match.
// Requires thr < 64 and a parent arena. On success every such pose's optimal
// script is reconstructed into script_scratch and script_len is set.
// Returns (via shared state) whether all pending poses were completed.
__device__ void run_rebfs_scripts(
    const Instance* inst, Workspace* ws, Phase12Shared* sh, Phase3Shared* s3,
    int n_wanted, int n_pending, const u16* costs, u32* parents
) {
    const int tid = threadIdx.x;
    const int thr = sh->thr;
    const int sc0 = sh->spawn.sc;
    const int sx = sh->spawn.x, sy = sh->spawn.y, srot = sh->spawn.rot;
    const int p0 = sh->spawn.p;

    int max_frames = (int)inst->max_frames;
    const int mlf = compute_max_lock_frames(sy, sc0, thr);
    if (max_frames > mlf) max_frames = mlf;

    const u64 thr_keep_lo = (thr < 63) ? ((1ull << (thr + 1)) - 1ull) : ~0ull;
    const u64 thr_keep_hi = 0ull;   // thr < 64 guaranteed here

    if (tid == 0) {
        s3->found_wanted = 0;
        s3->pending_resolves = 0;
        s3->do_refresh = 0;
        s3->stop = 0;
        s3->n_touched_v = 0;
        for (int i = 0; i < 16; ++i) s3->resolved_mask[i] = 0;

        const u32 start_m = v2_micro((u32)sh->spawn.hd, (u32)sh->spawn.hv);
        const u32 start_key = (u32)sy + (u32)V3_K_ROT * (u32)(srot & 3)
                              + (u32)V3_K_M * start_m + (u32)V3_K_RH * (u32)sh->spawn.rh;
        ws->visited[p0][start_key][sx][0] = 1ull << sc0;
        ws->accum[0][start_key][sx][0] = 1ull << sc0;
        ws->visited_flag[p0][start_key] = 1;
        ws->touched_v[s3->n_touched_v++] = (u32)p0 * WS_NKEYS_P + start_key;
        ws->accum_flag[0][start_key] = 1;
        ws->frontier[0][0] = start_key;
        s3->cur_n = 1;
        s3->next_n = 0;
    }
    __syncthreads();

    refresh_G_scripts(ws, s3, n_wanted, costs);
    __syncthreads();

    int p_cur = p0;
    int cur_buf = 0;

    for (int depth = 0; depth < max_frames; ++depth) {
        if (s3->cur_n == 0 || s3->stop) break;
        const u16 next_depth = (u16)(depth + 1);

        if (tid == 0) {
            s3->do_refresh = 0;
            if (s3->pending_resolves >= 4) { s3->do_refresh = 1; s3->pending_resolves = 0; }
        }
        __syncthreads();
        if (s3->do_refresh) {
            refresh_G_scripts(ws, s3, n_wanted, costs);
            __syncthreads();
        }
        for (int i = tid; i < GRID_H * 4; i += blockDim.x) {
            const int yy = i & 15;
            const int rr = i >> 4;
            u8 mask = 0;
            const int base = rr * 128 + yy * 8;
            for (int x = 0; x < 8; ++x) {
                if ((int)next_depth <= ws->G[base + x]) mask |= (u8)(1u << x);
            }
            s3->amask[yy][rr] = mask;
        }
        __syncthreads();

        const int even_parity = ((p_cur & FAST_DROP_MASK) == 0);
        const int n_act = even_parity ? 12 : 9;
        const u8* acts = even_parity ? ACTS_EVEN : ACTS_ODD;
        const int cur_n = s3->cur_n;
        for (int t = tid; t < cur_n; t += blockDim.x) {
            expand_key<true>(&sh->fm, ws, s3, ws->frontier[cur_buf][t], acts, n_act,
                             p_cur, cur_buf, next_depth,
                             thr_keep_lo, thr_keep_hi, thr, 0,
                             nullptr, parents);
        }
        __syncthreads();

        if (tid == 0 && s3->found_wanted >= n_pending) s3->stop = 1;

        for (int i = tid; i < cur_n; i += blockDim.x) {
            const u32 key = ws->frontier[cur_buf][i];
            for (int x = 0; x < GRID_W; ++x) {
                ws->accum[cur_buf][key][x][0] = 0;
                ws->accum[cur_buf][key][x][1] = 0;
            }
            ws->accum_flag[cur_buf][key] = 0;
        }
        __syncthreads();
        if (tid == 0) { s3->cur_n = s3->next_n; s3->next_n = 0; }
        cur_buf ^= 1;
        p_cur ^= 1;
        __syncthreads();
    }

    // cleanup dirtied blocks (same as phase 3 epilogue)
    for (int b = 0; b < 2; ++b) {
        const int nn = (b == 0) ? s3->cur_n : s3->next_n;
        const int buf = (b == 0) ? cur_buf : (cur_buf ^ 1);
        for (int i = tid; i < nn; i += blockDim.x) {
            const u32 key = ws->frontier[buf][i];
            for (int x = 0; x < GRID_W; ++x) {
                ws->accum[buf][key][x][0] = 0;
                ws->accum[buf][key][x][1] = 0;
            }
            ws->accum_flag[buf][key] = 0;
        }
    }
    __syncthreads();
    const u32 ntv = s3->n_touched_v;
    for (u32 i = tid; i < ntv; i += blockDim.x) {
        const u32 pk = ws->touched_v[i];
        const u32 pp = pk / WS_NKEYS_P;
        const u32 kk = pk % WS_NKEYS_P;
        for (int x = 0; x < GRID_W; ++x) {
            ws->visited[pp][kk][x][0] = 0;
            ws->visited[pp][kk][x][1] = 0;
        }
        ws->visited_flag[pp][kk] = 0;
    }
    __syncthreads();

    // Reconstruct scripts by walking parent chains backward (count-bounded;
    // every state on the chain was written this instance).
    for (int wi = tid; wi < n_wanted; wi += blockDim.x) {
        const int pose = (int)ws->wanted_ids[wi];
        if (ws->script_len[pose] != 0xFFFFu) continue;
        if (!(s3->resolved_mask[pose >> 5] & (1u << (pose & 31)))) continue;  // not found (shouldn't happen)
        const int c = (int)costs[pose];
        u8* rec = ws->script_scratch[pose];
        u32 entry = ws->term_parent[pose];
        rec[c - 1] = (u8)(entry >> 25);
        u32 id = entry & 0x1FFFFFFu;
        int ok = 1;
        for (int i = c - 2; i >= 0; --i) {
            entry = parents[id];
            rec[i] = (u8)(entry >> 25);
            id = entry & 0x1FFFFFFu;
        }
        if (ok) ws->script_len[pose] = (u16)c;
    }
    __syncthreads();
}

extern "C" __global__ void drm_reach_scripts_kernel(
    const Instance* __restrict__ insts,
    int n,
    unsigned long long* cursor,
    Workspace* ws_arena,
    u32* parents,                    // grid_blocks x PARENT_SLOT_U32, or null
    u16* __restrict__ out_costs,     // n x 512
    u16* __restrict__ out_offsets,   // n x 512
    u16* __restrict__ out_lengths,   // n x 512
    u8*  __restrict__ out_scripts,   // n x SCRIPT_BUF_CAP
    int* __restrict__ out_status     // n
) {
    __shared__ Phase12Shared sh;
    __shared__ Phase3Shared s3;
    __shared__ int s_idx;
    Workspace* ws = ws_arena + blockIdx.x;

    for (;;) {
        if (threadIdx.x == 0) s_idx = (int)atomicAdd(cursor, 1ull);
        __syncthreads();
        const int idx = s_idx;
        __syncthreads();
        if (idx >= n) return;

        const Instance inst = insts[idx];
        const int nw = run_phase12(&inst, ws, &sh);
        u16* costs = out_costs + (size_t)idx * N_POSES;
        // Script reconstruction needs the complete exact-search parent
        // surface; cost-only witness certification may stop before that
        // surface has been traversed.
        run_phase3(&inst, ws, &sh, &s3, nw, costs, false);
        u32* pslot = parents ? parents + (size_t)blockIdx.x * PARENT_SLOT_U32 : nullptr;
        run_scripts_stage(&inst, ws, &sh, &s3, nw, costs, pslot,
                          out_offsets + (size_t)idx * N_POSES,
                          out_lengths + (size_t)idx * N_POSES,
                          out_scripts + (size_t)idx * SCRIPT_BUF_CAP,
                          out_status + idx);
        __syncthreads();
    }
}

// Host queries sizeof(Workspace) at init so the Python arena allocation can
// never drift from the device struct layout.
extern "C" __global__ void drm_reach_ws_size_probe(u64* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = (u64)sizeof(Workspace);
}

// ---------------------------------------------------------------------------
// Stage 0 identity kernel (data-flow test; kept for the harness smoke test)
// ---------------------------------------------------------------------------
extern "C" __global__ void drm_reach_identity_kernel(
    const Instance* __restrict__ insts,
    int n,
    unsigned long long* cursor,
    u16* __restrict__ out_costs
) {
    __shared__ int s_idx;
    for (;;) {
        if (threadIdx.x == 0) s_idx = (int)atomicAdd(cursor, 1ull);
        __syncthreads();
        const int idx = s_idx;
        __syncthreads();
        if (idx >= n) return;

        const Instance inst = insts[idx];
        u16* out = out_costs + (size_t)idx * N_POSES;
        for (int i = threadIdx.x; i < N_POSES; i += blockDim.x) {
            out[i] = (u16)(inst.cols[i & 7] ^ ((u16)inst.p << 8) ^ (u16)inst.thr ^ (u16)(i << 4));
        }
    }
}
