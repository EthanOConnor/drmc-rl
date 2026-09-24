// drm_reach_full.cu — byte-exact CUDA port of drm_reach_bfs_full.
//
// Compiled at runtime with NVRTC (see full.py); self-contained.
//
// Ground truth: reach_native/drm_reach_full.c::drm_reach_bfs_full (the CPU
// arena planner for unconstrained paces). Every output array — costs,
// offsets, lengths and script bytes, including off-screen poses and the
// early-termination cut — must equal the CPU call byte for byte.
//
// The CPU BFS is sequential: one frontier per depth, nodes aggregated by the
// non-x key (x lives in an 8-bit mask), and ties broken by *first discovery*
// in frontier order x action order x DAS group x rotation group. Exact costs
// are unique; scripts are not, so we reproduce the sequential order:
//
//   * Every event gets an order value (global frontier index G of its parent
//     node, action slot, DAS group type, rotation group type). Group types are
//     canonical (wall < movable < blocked, shifted < in-place < rejected),
//     which matches the CPU's list order whether or not a group is empty.
//   * A child state's parent is the minimum-order event offering it while it
//     was unvisited before this depth (atomicMin). Sequentially, the first
//     such event marks it seen, so later events cannot claim it.
//   * A next-depth node's frontier position is the minimum order of any event
//     giving it a new bit; orders are unique per key, so a rank is a stable
//     bucket count: per parent node, a 108-bit mask of winning event slots,
//     then an exclusive scan over parents.
//   * Lock events carry (G, action slot, x); the first depth that reaches all
//     "wanted" poses cuts non-wanted locks ordered after the last wanted one,
//     exactly where the CPU `goto bfs_done` stops.
//
// Scripts are rebuilt by walking stored min-orders back to the spawn state.

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long long u64;

#define GRID_W 8
#define GRID_H 16
#define N_POSES 512
#define HOLD_NEUTRAL 0
#define HOLD_LEFT 1
#define HOLD_RIGHT 2
#define FAST_DROP_MASK 0x01
#define HOR_ACCEL_SPEED 0x10
#define HOR_RELOAD 0x0A
#define HOR_BLOCKED 0x0F

#define ORD_INF 0xFFFFFFFFu
#define ORD_SUB 108u        // 12 action slots x 3 DAS groups x 3 rotation groups
#define ORD_TERM 96u        // 12 action slots x 8 lock columns
#define MAX_DEPTH_CAP 2048
#define SCAN_MAX 1024

// Status bits (host falls back to the CPU planner when non-zero).
#define ST_SCRIPT_CHAIN 1   // a parent walk did not land on the spawn at its cost
#define ST_SCRIPT_CAP 2     // packed scripts exceed the per-instance buffer
#define ST_FRONTIER_CAP 4   // frontier log exceeded its capacity
#define ST_THRESHOLD_CAP 8  // speed threshold exceeds the workspace key space
#define ST_BAD_ARGS 16      // CPU would return an error code

__device__ __constant__ u8 ACT_HOLD_DIR[18]  = {0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2};
__device__ __constant__ u8 ACT_HOLD_DOWN[18] = {0,0,0,1,1,1, 0,0,0,1,1,1, 0,0,0,1,1,1};
__device__ __constant__ u8 ACT_ROT[18]       = {0,1,2,0,1,2, 0,1,2,0,1,2, 0,1,2,0,1,2};
__device__ __constant__ u8 ACTS_EVEN[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14};
__device__ __constant__ u8 ACTS_ODD[9]   = {0, 1, 2, 6, 7, 8, 12, 13, 14};

struct Instance {           // 32 bytes; host.INSTANCE_DTYPE
    u16 cols[GRID_W];
    u8  sx, sy, srot, sc, hv, hd, p, rh;
    u8  thr;
    u8  flags;
    u16 max_frames;
    u32 _pad;
};

struct FitMask { u8 m[2][GRID_H]; };

__device__ inline int pose_index(int x, int y, int rot) {
    return ((rot & 3) * (GRID_H * GRID_W)) + (y * GRID_W) + x;
}

__device__ inline void build_fit_masks(const u16* cols, FitMask* fm) {
    u8 occ[GRID_H];
    for (int y = 0; y < GRID_H; ++y) occ[y] = 0u;
    for (int x = 0; x < GRID_W; ++x)
        for (int y = 0; y < GRID_H; ++y)
            if (cols[x] & (u16)(1u << (unsigned)y)) occ[y] |= (u8)(1u << (unsigned)x);
    u8 empty[GRID_H];
    for (int y = 0; y < GRID_H; ++y) empty[y] = (u8)(~occ[y]) & 0xFFu;
    for (int y = 0; y < GRID_H; ++y) fm->m[0][y] = (u8)(empty[y] & (u8)(empty[y] >> 1));
    fm->m[1][0] = empty[0];
    for (int y = 1; y < GRID_H; ++y) fm->m[1][y] = (u8)(empty[y] & empty[y - 1]);
}

__device__ inline int fits_masked(const FitMask* fm, int x, int y, int rot) {
    if ((unsigned)x >= (unsigned)GRID_W || (unsigned)y >= (unsigned)GRID_H) return 0;
    return (int)((fm->m[(rot & 1) ? 1 : 0][y] >> (unsigned)x) & 1u);
}

__device__ void apply_rotation_masked(const FitMask* fm, int* x, int y, int* rot,
                                      int rotation, int hold_left) {
    if (rotation == 0) return;
    const int x0 = *x;
    const int rot0 = (*rot) & 3;
    const int rot1 = rotation == 1 ? (rot0 - 1) & 3 : (rot0 + 1) & 3;
    if ((rot1 & 1) == 0) {
        if (fits_masked(fm, x0, y, rot1)) {
            if (hold_left && fits_masked(fm, x0 - 1, y, rot1)) { *x = x0 - 1; *rot = rot1; return; }
            *x = x0; *rot = rot1; return;
        }
        if (fits_masked(fm, x0 - 1, y, rot1)) { *x = x0 - 1; *rot = rot1; return; }
        return;
    }
    if (fits_masked(fm, x0, y, rot1)) *rot = rot1;
}

// drm_reach_full.c::build_wanted_terminal_poses_reachable (set + count only).
__device__ int build_wanted(const FitMask* fm, int sx, int sy, int srot,
                            u8* wanted, u8* visited, u16* queue) {
    for (int i = 0; i < N_POSES; ++i) { wanted[i] = 0; visited[i] = 0; }
    int qh = 0, qt = 0, count = 0;
    const u16 start = (u16)pose_index(sx, sy, srot & 3);
    visited[start] = 1u;
    queue[qt++] = start;
    while (qh < qt) {
        const u16 pose = queue[qh++];
        const int x = (int)(pose & 7u), y = (int)((pose >> 3) & 15u), rot = (int)((pose >> 7) & 3u);
        const int can_fall = (y + 1 < GRID_H) && fits_masked(fm, x, y + 1, rot);
        if (!can_fall && (((rot & 1) == 0) ? (x + 1 < GRID_W) : (y >= 1)) && !wanted[pose]) {
            wanted[pose] = 1u;
            count += 1;
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
                int rx = x, rrot = rot;
                apply_rotation_masked(fm, &rx, y, &rrot, rotation, hold_left);
                if (rx == x && (rrot & 3) == (rot & 3)) continue;
                const u16 np = (u16)pose_index(rx, y, rrot);
                if (!visited[np]) { visited[np] = 1u; queue[qt++] = np; }
            }
        }
    }
    return count;
}

__device__ inline int compute_max_lock_frames(int y0, int sc0, int thr) {
    int sc = sc0;
    if (sc < 0) sc = 0;
    if (sc > thr) sc = thr;
    int m_max = GRID_H - y0;
    if (m_max < 1) m_max = 1;
    int total = (thr - sc) + 1;
    if (m_max > 1) total += (m_max - 1) * (thr + 1);
    if (total < 1) total = 1;
    return total;
}

// Key: y | rot<<4 | ((((rh*2 + p)*3 + hd)*16 + hv)*scr + sc) << 6.
// nkeys = 64 * scr * 16 * 3 * 2 * 3 = 18432 * scr (same count as the CPU).
struct KeyFields { int y, rot, sc, hv, hd, p, rh; };

__device__ inline u32 make_key(int scr, int y, int rot, int sc, int hv, int hd, int p, int rh) {
    return (u32)y | ((u32)rot << 4) |
           ((u32)((((rh * 2 + p) * 3 + hd) * 16 + hv) * scr + sc) << 6);
}

__device__ inline KeyFields split_key(int scr, u32 key) {
    KeyFields f;
    f.y = (int)(key & 15u);
    f.rot = (int)((key >> 4) & 3u);
    u32 rest = key >> 6;
    f.sc = (int)(rest % (u32)scr); rest /= (u32)scr;
    f.hv = (int)(rest & 15u); rest >>= 4;
    f.hd = (int)(rest % 3u); rest /= 3u;
    f.p = (int)(rest & 1u);
    f.rh = (int)(rest >> 1);
    return f;
}

struct Slot {
    u32* sord;      // [nkeys*8]  min event order per full state (INF when unvisited)
    u32* acc;       // [nkeys]    next-depth new x bits
    u32* kord;      // [nkeys]    next-depth node order (scratch, set in finalize)
    u32* nxt;       // [nkeys]    next-depth keys, unordered
    u32* rbits;     // [nkeys*4]  per parent node: winning event slots
    u32* prefix;    // [nkeys]    exclusive scan over parents
    u32* fr_key;    // [fr_cap]   frontier log, all depths, CPU order
    u8*  vis;       // [nkeys]    visited x bits before the current depth
    u8*  fr_x;      // [fr_cap]
};

__device__ inline Slot slot_at(u8* base, u32 nkeys, u32 fr_cap) {
    Slot s;
    u8* p = base;
    s.sord = (u32*)p;   p += (u64)nkeys * 8u * 4u;
    s.acc = (u32*)p;    p += (u64)nkeys * 4u;
    s.kord = (u32*)p;   p += (u64)nkeys * 4u;
    s.nxt = (u32*)p;    p += (u64)nkeys * 4u;
    s.rbits = (u32*)p;  p += (u64)nkeys * 16u;
    s.prefix = (u32*)p; p += (u64)nkeys * 4u;
    s.fr_key = (u32*)p; p += (u64)fr_cap * 4u;
    s.vis = p;          p += (u64)nkeys;
    s.fr_x = p;
    return s;
}

// full.py::_slot_layout mirrors this layout.

__device__ inline void offer(const Slot& s, u32* s_next, u32 key, u8 xm, u32 order) {
    const u8 fresh = (u8)(xm & (u8)(~s.vis[key]));
    if (!fresh) return;
    const u32 old = atomicOr(&s.acc[key], (u32)fresh);
    if (old == 0u) s.nxt[atomicAdd(s_next, 1u)] = key;
    u8 bits = fresh;
    const u32 base = key << 3;
    while (bits) {
        const int xo = __ffs((int)bits) - 1;
        bits &= (u8)(bits - 1u);
        atomicMin(&s.sord[base + (u32)xo], order);
    }
}

// One (frontier node, action slot) expansion; mirrors drm_reach_bfs_full's
// inner loop line for line, but emits ordered events instead of mutating.
__device__ void expand(const Slot& s, const FitMask* fm, int scr, int thr, u32 G, int ai,
                       u32* term, u32* s_next) {
    const u32 key = s.fr_key[G];
    u8 xmask = s.fr_x[G];
    const KeyFields f = split_key(scr, key);
    const int parity0 = f.p & 1;
    const int act = parity0 ? (int)ACTS_ODD[ai] : (int)ACTS_EVEN[ai];

    int y = f.y;
    const int rot0 = f.rot & 3;
    const int rot = rot0;
    int sc = f.sc;
    const int hv0 = f.hv & 0x0F;
    const int hd_prev = f.hd;
    const int rh_prev = f.rh;

    const int hold_dir_now = (int)ACT_HOLD_DIR[act];
    const int hold_down = (int)ACT_HOLD_DOWN[act];
    const int rotation = (int)ACT_ROT[act];
    const int hold_left = hold_dir_now == HOLD_LEFT;
    const int hold_right = hold_dir_now == HOLD_RIGHT;
    const int press_lr = (hold_left && hd_prev != HOLD_LEFT) || (hold_right && hd_prev != HOLD_RIGHT);

    // Y stage.
    const int down_only = hold_down && hold_dir_now == HOLD_NEUTRAL;
    int drop = 0;
    if ((parity0 & FAST_DROP_MASK) == 0 && down_only) {
        drop = 1;
        sc = 0;
    } else {
        sc = sc + 1;
        if (sc > thr) { drop = 1; sc = 0; }
    }
    if (drop) {
        const int ny = y + 1;
        u8 drop_ok = 0u;
        if ((unsigned)ny < (unsigned)GRID_H) drop_ok = fm->m[(rot & 1) ? 1 : 0][ny];
        u8 xm_lock = (u8)(xmask & (u8)(~drop_ok));
        const u32 tbase = G * ORD_TERM + (u32)ai * 8u;
        while (xm_lock) {
            const int lx = __ffs((int)xm_lock) - 1;
            xm_lock &= (u8)(xm_lock - 1u);
            atomicMin(&term[pose_index(lx, y, rot0)], tbase + (u32)lx);
        }
        xmask = (u8)(xmask & drop_ok);
        if (!xmask) return;
        y = ny;
    }

    // X stage. Group type index t: 0 pass/wall, 1 movable, 2 blocked.
    int allow_move = 0;
    int hv = hv0;
    if (press_lr) {
        hv = 0;
        allow_move = 1;
    } else if (hold_dir_now != HOLD_NEUTRAL) {
        hv = hv + 1;
        if (hv >= HOR_ACCEL_SPEED) { hv = HOR_RELOAD; allow_move = 1; }
    }
    u8 txm[3] = {0u, 0u, 0u};
    u8 thv[3];
    int tdx[3] = {0, 0, 0};
    thv[0] = thv[1] = (u8)(hv & 0x0F);
    thv[2] = (u8)HOR_BLOCKED;
    if (!allow_move || hold_dir_now == HOLD_NEUTRAL) {
        txm[0] = xmask;
    } else if (hold_right) {
        const u8 ok = (u8)(fm->m[(rot & 1) ? 1 : 0][y] >> 1);
        const u8 wall = (u8)(xmask & (1u << (6 + (rot & 1))));
        txm[0] = wall;
        txm[1] = (u8)((u8)(xmask & ok) << 1);
        tdx[1] = 1;
        txm[2] = (u8)(xmask & (u8)(~(ok | wall)));
    } else {
        const u8 ok = (u8)((fm->m[(rot & 1) ? 1 : 0][y] << 1) & 0xFFu);
        const u8 wall = (u8)(xmask & 1u);
        txm[0] = wall;
        txm[1] = (u8)((u8)(xmask & ok) >> 1);
        tdx[1] = -1;
        txm[2] = (u8)(xmask & (u8)(~(ok | wall)));
    }

    // Rotate stage. Group type index o: horizontal target 0 shifted,
    // 1 in place, 2 rejected; vertical target 0 accepted, 1 rejected.
    const int rotation_pressed = rotation != 0 && rotation != rh_prev;
    const int p_next = (parity0 ^ 1) & FAST_DROP_MASK;
    const u32 order_ai = G * ORD_SUB + (u32)ai * 9u;
    for (int ti = 0; ti < 3; ++ti) {
        const u8 xm_in = txm[ti];
        if (!xm_in) continue;
        const int hv_in = thv[ti];
        const u32 order_t = order_ai + (u32)ti * 3u;
        if (!rotation_pressed) {
            offer(s, s_next, make_key(scr, y, rot, sc, hv_in, hold_dir_now, p_next, rotation),
                  xm_in, order_t);
            continue;
        }
        const int rot1 = rotation == 1 ? (rot - 1) & 3 : (rot + 1) & 3;
        if ((rot1 & 1) != 0) {
            const u8 fit_v = fm->m[1][y];
            const u8 accepted = (u8)(xm_in & fit_v);
            const u8 rejected = (u8)(xm_in & (u8)(~fit_v));
            if (accepted)
                offer(s, s_next, make_key(scr, y, rot1, sc, hv_in, hold_dir_now, p_next, rotation),
                      accepted, order_t);
            if (rejected)
                offer(s, s_next, make_key(scr, y, rot, sc, hv_in, hold_dir_now, p_next, rotation),
                      rejected, order_t + 1u);
        } else {
            const u8 fit_h = fm->m[0][y];
            const u8 acc_inplace = (u8)(xm_in & fit_h);
            const u8 rej_inplace = (u8)(xm_in & (u8)(~fit_h));
            const u8 ok_left = (u8)((fit_h << 1) & 0xFFu);
            const u8 dbl = hold_left ? (u8)(acc_inplace & ok_left) : (u8)0u;
            const u8 acc_noshift = (u8)(acc_inplace & (u8)(~dbl));
            const u8 kick = (u8)(rej_inplace & ok_left);
            const u8 rej = (u8)(rej_inplace & (u8)(~kick));
            const u8 shifted = (u8)(dbl | kick);
            if (shifted)
                offer(s, s_next, make_key(scr, y, rot1, sc, hv_in, hold_dir_now, p_next, rotation),
                      (u8)(shifted >> 1), order_t);
            if (acc_noshift)
                offer(s, s_next, make_key(scr, y, rot1, sc, hv_in, hold_dir_now, p_next, rotation),
                      acc_noshift, order_t + 1u);
            if (rej)
                offer(s, s_next, make_key(scr, y, rot, sc, hv_in, hold_dir_now, p_next, rotation),
                      rej, order_t + 2u);
        }
    }
}

// x displacement of the event that produced a child (for the parent walk).
__device__ inline int event_dx(const KeyFields& parent, int act, int ti, int oi) {
    const int dir = (int)ACT_HOLD_DIR[act];
    int dx = ti == 1 ? (dir == HOLD_RIGHT ? 1 : -1) : 0;
    const int rotation = (int)ACT_ROT[act];
    if (rotation != 0 && rotation != parent.rh && oi == 0) {
        const int rot1 = rotation == 1 ? (parent.rot - 1) & 3 : (parent.rot + 1) & 3;
        if ((rot1 & 1) == 0) dx -= 1;
    }
    return dx;
}

__device__ inline int slot_action(int parity, int ai) {
    return parity ? (int)ACTS_ODD[ai] : (int)ACTS_EVEN[ai];
}

struct FullShared {
    FitMask fm;
    u8  wanted[N_POSES];
    u8  fl_vis[N_POSES];
    u16 fl_q[N_POSES];
    u32 term[N_POSES];
    u32 offs[N_POSES];
    u32 depth_base[MAX_DEPTH_CAP + 2];
    u32 scan[SCAN_MAX];
    u32 next_n;
    u32 found;
    u32 stop;
    int idx, status, wanted_count, max_depth, depths, finalized, thr, sx, p0, empty;
    u32 start_key, total;
};

extern "C" __global__ void __launch_bounds__(SCAN_MAX)
drm_reach_full_kernel(
    const Instance* __restrict__ insts,
    int n,
    unsigned long long* cursor,
    u8* arena,
    unsigned int slot_units,        // slot size / 256 bytes
    int scr,                        // speed-counter range of the key space
    unsigned int fr_cap,
    int scap,                       // script bytes per instance
    u16* __restrict__ out_costs,    // n x 512
    u16* __restrict__ out_offsets,  // n x 512
    u16* __restrict__ out_lengths,  // n x 512
    u8*  __restrict__ out_scripts,  // n x scap
    int* __restrict__ out_used,     // n
    int* __restrict__ out_status,   // n
    unsigned int* __restrict__ out_nodes   // n: frontier entries (diagnostic)
) {
    __shared__ FullShared sh;
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const u32 nkeys = 18432u * (u32)scr;
    const Slot s = slot_at(arena + (u64)blockIdx.x * (u64)slot_units * 256u, nkeys, fr_cap);

    for (;;) {
        if (tid == 0) sh.idx = (int)atomicAdd(cursor, 1ull);
        __syncthreads();
        const int idx = sh.idx;
        if (idx >= n) return;
        const Instance in = insts[idx];

        if (tid == 0) {
            sh.status = 0;
            sh.empty = 0;
            sh.total = 0;
            sh.depths = 0;
            sh.finalized = 1;
            int thr = (int)in.thr;
            if (thr > 0x7F) thr = 0x7F;
            int sc = (int)in.sc;
            if (sc > thr) sc = thr;
            int hv = (int)in.hv;
            if (hv > 15) hv &= 0x0F;
            int hd = (int)in.hd;
            if (hd > 2) hd = 0;
            const int p = (int)in.p & FAST_DROP_MASK;
            int rh = (int)in.rh;
            if (rh > 2) rh = 0;
            int max_frames = (int)in.max_frames;
            const int sx = (int)in.sx, sy = (int)in.sy, srot = (int)in.srot & 3;
            sh.thr = thr;
            sh.sx = sx;
            sh.p0 = p;
            if (max_frames <= 0) {
                sh.status = ST_BAD_ARGS;
            } else if ((unsigned)sx >= (unsigned)GRID_W || (unsigned)sy >= (unsigned)GRID_H) {
                sh.empty = 1;
            } else if (thr + 1 > scr) {
                sh.status = ST_THRESHOLD_CAP;
            } else {
                const int max_lock = compute_max_lock_frames(sy, sc, thr);
                if (max_frames > max_lock) max_frames = max_lock;
                if (max_frames > MAX_DEPTH_CAP) max_frames = MAX_DEPTH_CAP;
                sh.max_depth = max_frames;
                build_fit_masks(in.cols, &sh.fm);
                if (!fits_masked(&sh.fm, sx, sy, srot)) {
                    sh.empty = 1;
                } else {
                    sh.wanted_count = build_wanted(&sh.fm, sx, sy, srot, sh.wanted, sh.fl_vis, sh.fl_q);
                    const u32 start_key = make_key(scr, sy, srot, sc, hv, hd, p, rh);
                    sh.start_key = start_key;
                    s.fr_key[0] = start_key;
                    s.fr_x[0] = (u8)(1u << (unsigned)sx);
                    s.vis[start_key] = (u8)(1u << (unsigned)sx);
                    sh.depth_base[0] = 0u;
                    sh.depth_base[1] = 1u;
                    sh.total = 1u;
                }
            }
        }
        for (int i = tid; i < N_POSES; i += T) sh.term[i] = ORD_INF;
        __syncthreads();
        const int solving = sh.status == 0 && !sh.empty;

        if (solving) {
            u32 base = 0u, n_cur = 1u;
            const int thr = sh.thr;
            for (int d = 0; d < sh.max_depth && n_cur > 0u; ++d) {
                const int parity = (sh.p0 ^ (d & 1)) & 1;
                const u32 A = parity ? 9u : 12u;
                if (tid == 0) { sh.next_n = 0u; sh.found = 0u; sh.stop = 0u; }
                __syncthreads();
                const u32 items = n_cur * A;
                for (u32 i = (u32)tid; i < items; i += (u32)T)
                    expand(s, &sh.fm, scr, thr, base + i / A, (int)(i % A), sh.term, &sh.next_n);
                __syncthreads();

                // Early termination: all wanted poses have a lock order.
                u32 found = 0u;
                for (int pose = tid; pose < N_POSES; pose += T)
                    found += (sh.wanted[pose] && sh.term[pose] != ORD_INF) ? 1u : 0u;
                if (found) atomicAdd(&sh.found, found);
                __syncthreads();
                const int done = sh.wanted_count > 0 && (int)sh.found >= sh.wanted_count;
                if (done) {
                    for (int pose = tid; pose < N_POSES; pose += T)
                        if (sh.wanted[pose]) atomicMax(&sh.stop, sh.term[pose]);
                    __syncthreads();
                    for (int pose = tid; pose < N_POSES; pose += T)
                        if (!sh.wanted[pose] && sh.term[pose] != ORD_INF && sh.term[pose] > sh.stop)
                            sh.term[pose] = ORD_INF;
                }

                const u32 n_next = sh.next_n;
                const u32 next_base = base + n_cur;
                // Lock orders of this depth lie in [base[d], base[d+1]).
                const int last = done || d + 1 >= sh.max_depth || n_next == 0u;
                const int overflow = !last && (u64)next_base + n_next > (u64)fr_cap;
                __syncthreads();
                if (tid == 0) {
                    sh.depths = d + 1;
                    if (overflow) sh.status |= ST_FRONTIER_CAP;
                    if (last || overflow) sh.finalized = 0;
                }
                __syncthreads();
                if (last || overflow) break;
                // Rank next-depth nodes by their minimum event order: the
                // minimum over its new bits of their winning event orders.
                for (u32 j = (u32)tid; j < n_next; j += (u32)T) {
                    const u32 key = s.nxt[j];
                    u8 bits = (u8)s.acc[key];
                    u32 o = ORD_INF;
                    while (bits) {
                        const int x = __ffs((int)bits) - 1;
                        bits &= (u8)(bits - 1u);
                        o = min(o, s.sord[(key << 3) + (u32)x]);
                    }
                    s.kord[key] = o;
                    const u32 pl = o / ORD_SUB - base, sub = o % ORD_SUB;
                    atomicOr(&s.rbits[pl * 4u + (sub >> 5)], 1u << (sub & 31u));
                }
                __syncthreads();
                const u32 chunk = (n_cur + (u32)T - 1u) / (u32)T;
                const u32 lo = min((u32)tid * chunk, n_cur), hi = min(lo + chunk, n_cur);
                u32 sum = 0u;
                for (u32 pl = lo; pl < hi; ++pl)
                    sum += __popc(s.rbits[pl * 4u]) + __popc(s.rbits[pl * 4u + 1u]) +
                           __popc(s.rbits[pl * 4u + 2u]) + __popc(s.rbits[pl * 4u + 3u]);
                // Block exclusive scan: warp shuffles, then one warp over warp totals.
                const int lane = tid & 31, warp = tid >> 5;
                u32 inclusive = sum;
                for (int off = 1; off < 32; off <<= 1) {
                    const u32 up = __shfl_up_sync(0xFFFFFFFFu, inclusive, off);
                    if (lane >= off) inclusive += up;
                }
                if (lane == 31) sh.scan[warp] = inclusive;
                __syncthreads();
                if (warp == 0) {
                    const int warps = (T + 31) >> 5;
                    u32 total = lane < warps ? sh.scan[lane] : 0u;
                    for (int off = 1; off < 32; off <<= 1) {
                        const u32 up = __shfl_up_sync(0xFFFFFFFFu, total, off);
                        if (lane >= off) total += up;
                    }
                    if (lane < warps) sh.scan[lane] = total;
                }
                __syncthreads();
                u32 running = (warp ? sh.scan[warp - 1] : 0u) + inclusive - sum;
                for (u32 pl = lo; pl < hi; ++pl) {
                    s.prefix[pl] = running;
                    running += __popc(s.rbits[pl * 4u]) + __popc(s.rbits[pl * 4u + 1u]) +
                               __popc(s.rbits[pl * 4u + 2u]) + __popc(s.rbits[pl * 4u + 3u]);
                }
                __syncthreads();
                for (u32 j = (u32)tid; j < n_next; j += (u32)T) {
                    const u32 key = s.nxt[j];
                    const u32 o = s.kord[key];
                    const u32 pl = o / ORD_SUB - base, sub = o % ORD_SUB;
                    const u32 w = sub >> 5;
                    u32 rank = s.prefix[pl] + __popc(s.rbits[pl * 4u + w] & ((1u << (sub & 31u)) - 1u));
                    for (u32 ww = 0; ww < w; ++ww) rank += __popc(s.rbits[pl * 4u + ww]);
                    const u32 bits = s.acc[key];
                    s.fr_key[next_base + rank] = key;
                    s.fr_x[next_base + rank] = (u8)bits;
                    s.vis[key] = (u8)(s.vis[key] | bits);
                    s.acc[key] = 0u;
                }
                __syncthreads();
                for (u32 pl = (u32)tid; pl < n_cur * 4u; pl += (u32)T) s.rbits[pl] = 0u;
                if (tid == 0) {
                    sh.depth_base[d + 2] = next_base + n_next;
                    sh.total = next_base + n_next;
                }
                __syncthreads();
                base = next_base;
                n_cur = n_next;
            }
        }

        // Costs, packed offsets (CPU casts the running total to u16), lengths.
        u16* costs = out_costs + (u64)idx * N_POSES;
        u16* offsets = out_offsets + (u64)idx * N_POSES;
        u16* lengths = out_lengths + (u64)idx * N_POSES;
        for (int pose = tid; pose < N_POSES; pose += T) {
            const u32 t = sh.term[pose];
            u32 cost = 0u;
            if (t != ORD_INF) {
                const u32 G = t / ORD_TERM;
                int lo = 0, hi = sh.depths;      // depth d holds [base[d], base[d+1])
                while (lo < hi) {
                    const int mid = (lo + hi + 1) >> 1;
                    if (sh.depth_base[mid] <= G) lo = mid; else hi = mid - 1;
                }
                cost = (u32)lo + 1u;
            }
            sh.offs[pose] = cost;
        }
        __syncthreads();
        if (tid == 0) {
            u32 used = 0u;
            for (int pose = 0; pose < N_POSES; ++pose) {
                const u32 len = sh.offs[pose];
                sh.offs[pose] = used;
                used += len;
            }
            if (solving && (int)used > scap) sh.status |= ST_SCRIPT_CAP;
            out_used[idx] = (int)used;
            out_nodes[idx] = sh.total;
        }
        __syncthreads();
        for (int pose = tid; pose < N_POSES; pose += T) {
            const u32 t = sh.term[pose];
            if (t == ORD_INF) {
                costs[pose] = 0xFFFFu; offsets[pose] = 0u; lengths[pose] = 0u;
                continue;
            }
            const u32 off = sh.offs[pose];
            const u32 next_off = pose + 1 < N_POSES ? sh.offs[pose + 1] : (u32)out_used[idx];
            const u32 len = next_off - off;
            costs[pose] = (u16)len;
            offsets[pose] = (u16)off;
            lengths[pose] = (u16)len;
            if (sh.status & ST_SCRIPT_CAP) continue;
            u8* script = out_scripts + (u64)idx * (u64)scap;
            // Terminal event, then min-order parents back to the spawn.
            u32 pos = off + len;
            u32 G = t / ORD_TERM;
            u32 key = s.fr_key[G];
            KeyFields pf = split_key(scr, key);
            script[--pos] = (u8)slot_action(pf.p, (int)((t % ORD_TERM) >> 3));
            int x = (int)(t & 7u);
            int ok = 1;
            while (!(key == sh.start_key && x == sh.sx)) {
                const u32 o = s.sord[(key << 3) + (u32)x];
                if (o == ORD_INF || pos <= off) { ok = 0; break; }
                const u32 sub = o % ORD_SUB;
                const u32 pkey = s.fr_key[o / ORD_SUB];
                pf = split_key(scr, pkey);
                const int act = slot_action(pf.p, (int)(sub / 9u));
                script[--pos] = (u8)act;
                x -= event_dx(pf, act, (int)((sub / 3u) % 3u), (int)(sub % 3u));
                key = pkey;
                if ((unsigned)x >= (unsigned)GRID_W) { ok = 0; break; }
            }
            if (!ok || pos != off) atomicOr(&sh.status, ST_SCRIPT_CHAIN);
        }
        __syncthreads();

        // Leave the workspace clean for the next instance.
        if (!sh.finalized) {
            const u32 n_next = sh.next_n;
            for (u32 j = (u32)tid; j < n_next; j += (u32)T) {
                const u32 key = s.nxt[j];
                u8 bits = (u8)s.acc[key];
                while (bits) {
                    const int x = __ffs((int)bits) - 1;
                    bits &= (u8)(bits - 1u);
                    s.sord[(key << 3) + (u32)x] = ORD_INF;
                }
                s.acc[key] = 0u;
                s.kord[key] = ORD_INF;
            }
            __syncthreads();
        }
        const u32 total = sh.total;
        for (u32 G = (u32)tid; G < total; G += (u32)T) {
            const u32 key = s.fr_key[G];
            s.vis[key] = 0u;
            u8 bits = s.fr_x[G];
            while (bits) {
                const int x = __ffs((int)bits) - 1;
                bits &= (u8)(bits - 1u);
                s.sord[(key << 3) + (u32)x] = ORD_INF;
            }
        }
        if (tid == 0) out_status[idx] = sh.status;
        __syncthreads();
    }
}

