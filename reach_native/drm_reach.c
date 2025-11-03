// Dr. Mario Falling-Pill Reachability (C skeleton)
//
// Implements the 512-state BFS core with two modes:
//  - Planner mode: 1 primitive + gravity
//  - NES mode: Y -> X -> Rotate, with parity-based collision and left kicks
//
// Memory footprint: 1KB core (arrival + parent) + small queue.

#include <stdint.h>
#include <string.h>

enum { GRID_W = 8, GRID_H = 16 };

enum { O_HPOS = 0, O_VPOS = 1, O_HNEG = 2, O_VNEG = 3 };

static inline int drm_idx(int x, int y, int o) { return (o << 7) | (y << 3) | x; }

enum ParentCode {
    PC_NONE = 0,
    PC_WAIT = 1,
    PC_DROP = 2,
    PC_FROM_LEFT = 3,
    PC_FROM_RIGHT = 4,
    PC_ROT_CW = 5,
    PC_ROT_CCW = 6,
    PC_ROT_CW_KICK = 7,
    PC_ROT_CCW_KICK = 8,
};

typedef struct {
    uint8_t arrival[512];
    uint8_t parent[512];
} bfs_result_t;

// Convert 8x uint16 column occupancy to [4][16][8] boolean fit planes.
static void build_fit_planes(const uint16_t cols[8], uint8_t fit[4][16][8], int nes_parity) {
    uint16_t occ_hpos[8], occ_hneg[8], occ_vpos[8], occ_vneg[8];
    for (int x = 0; x < 8; ++x) {
        uint16_t occ = cols[x];
        uint16_t r = (x + 1 < 8) ? cols[x + 1] : 0xFFFF;
        uint16_t l = (x - 1 >= 0) ? cols[x - 1] : 0xFFFF;
        occ_hpos[x] = (uint16_t)(occ | r);
        occ_hneg[x] = (uint16_t)(occ | l);
        occ_vpos[x] = (uint16_t)(occ | (occ >> 1) | 0x8000u);
        occ_vneg[x] = (uint16_t)(occ | (occ << 1) | 0x0001u);
    }
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint16_t b0 = (occ_hpos[x] >> y) & 1u;
            uint16_t b1 = (occ_vpos[x] >> y) & 1u;
            uint16_t b2 = (occ_hneg[x] >> y) & 1u;
            uint16_t b3 = (occ_vneg[x] >> y) & 1u;
            fit[0][y][x] = (uint8_t)(!b0);
            fit[1][y][x] = (uint8_t)(!b1);
            fit[2][y][x] = (uint8_t)(!b2);
            fit[3][y][x] = (uint8_t)(!b3);
        }
    }
    if (nes_parity) {
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 8; ++x) {
                fit[2][y][x] = fit[0][y][x];
                fit[3][y][x] = fit[1][y][x];
            }
        }
    }
}

static inline int fits(const uint8_t fit[4][16][8], int x, int y, int o) {
    if ((unsigned)x >= 8u || (unsigned)y >= 16u) return 0;
    return fit[o & 3][y][x] != 0;
}

static inline void maybe_push(uint8_t arrival[512], uint8_t parent[512], uint16_t q[512], int *qh, int *qt,
                              int x, int y, int o, int t_next, enum ParentCode pc) {
    const int idx = drm_idx(x, y, o);
    if (arrival[idx] == 0xFF || arrival[idx] > (uint8_t)t_next) {
        arrival[idx] = (uint8_t)t_next;
        parent[idx] = (uint8_t)pc;
        q[*qt = ((*qt) + 1) & 511] = (uint16_t)idx;
    }
}

static void expand_planner(const uint8_t fit[4][16][8], int x, int y, int o, int t,
                           uint8_t arrival[512], uint8_t parent[512], uint16_t q[512], int *qh, int *qt) {
    // wait
    if (y + 1 < 16 && fits(fit, x, y + 1, o)) maybe_push(arrival, parent, q, qh, qt, x, y + 1, o, t + 1, PC_DROP);
    else                                      maybe_push(arrival, parent, q, qh, qt, x, y, o, t + 1, PC_WAIT);
    // left
    if (x - 1 >= 0 && fits(fit, x - 1, y, o)) {
        if (y + 1 < 16 && fits(fit, x - 1, y + 1, o)) maybe_push(arrival, parent, q, qh, qt, x - 1, y + 1, o, t + 1, PC_DROP);
        else                                           maybe_push(arrival, parent, q, qh, qt, x - 1, y, o, t + 1, PC_FROM_RIGHT);
    }
    // right
    if (x + 1 < 8 && fits(fit, x + 1, y, o)) {
        if (y + 1 < 16 && fits(fit, x + 1, y + 1, o)) maybe_push(arrival, parent, q, qh, qt, x + 1, y + 1, o, t + 1, PC_DROP);
        else                                          maybe_push(arrival, parent, q, qh, qt, x + 1, y, o, t + 1, PC_FROM_LEFT);
    }
    // rot cw
    {
        int co = (o + 1) & 3;
        if (fits(fit, x, y, co)) {
            if (y + 1 < 16 && fits(fit, x, y + 1, co)) maybe_push(arrival, parent, q, qh, qt, x, y + 1, co, t + 1, PC_DROP);
            else                                       maybe_push(arrival, parent, q, qh, qt, x, y, co, t + 1, PC_ROT_CW);
        }
    }
    // rot ccw
    {
        int co = (o + 3) & 3;
        if (fits(fit, x, y, co)) {
            if (y + 1 < 16 && fits(fit, x, y + 1, co)) maybe_push(arrival, parent, q, qh, qt, x, y + 1, co, t + 1, PC_DROP);
            else                                       maybe_push(arrival, parent, q, qh, qt, x, y, co, t + 1, PC_ROT_CCW);
        }
    }
}

static void expand_nes(const uint8_t fit[4][16][8], int x, int y, int o, int t,
                       uint8_t arrival[512], uint8_t parent[512], uint16_t q[512], int *qh, int *qt) {
    // Y stage
    int px = x, py = y, po = o; enum ParentCode ppc = PC_WAIT;
    if (y + 1 < 16 && fits(fit, x, y + 1, o)) { py = y + 1; ppc = PC_DROP; }

    // X stage – try both sides and the no-move path
    int cand[3][3]; int cnum = 0;
    cand[cnum][0] = px; cand[cnum][1] = py; cand[cnum][2] = po; cnum++;
    if (px - 1 >= 0 && fits(fit, px - 1, py, po)) { cand[cnum][0] = px - 1; cand[cnum][1] = py; cand[cnum][2] = po; cnum++; }
    if (px + 1 < 8 && fits(fit, px + 1, py, po)) { cand[cnum][0] = px + 1; cand[cnum][1] = py; cand[cnum][2] = po; cnum++; }

    for (int i = 0; i < cnum; ++i) {
        int cx = cand[i][0], cy = cand[i][1], co = cand[i][2];
        // CW
        int ro = (co + 1) & 3;
        if (fits(fit, cx, cy, ro)) maybe_push(arrival, parent, q, qh, qt, cx, cy, ro, t + 1, PC_ROT_CW);
        // extra-left if horizontal parity
        if ((ro == O_HPOS || ro == O_HNEG) && cx - 1 >= 0 && fits(fit, cx - 1, cy, ro))
            maybe_push(arrival, parent, q, qh, qt, cx - 1, cy, ro, t + 1, PC_ROT_CW_KICK);
        // wall-kick left
        if (cx - 1 >= 0 && fits(fit, cx - 1, cy, ro))
            maybe_push(arrival, parent, q, qh, qt, cx - 1, cy, ro, t + 1, PC_ROT_CW_KICK);

        // CCW
        ro = (co + 3) & 3;
        if (fits(fit, cx, cy, ro)) maybe_push(arrival, parent, q, qh, qt, cx, cy, ro, t + 1, PC_ROT_CCW);
        if ((ro == O_HPOS || ro == O_HNEG) && cx - 1 >= 0 && fits(fit, cx - 1, cy, ro))
            maybe_push(arrival, parent, q, qh, qt, cx - 1, cy, ro, t + 1, PC_ROT_CCW_KICK);
        if (cx - 1 >= 0 && fits(fit, cx - 1, cy, ro))
            maybe_push(arrival, parent, q, qh, qt, cx - 1, cy, ro, t + 1, PC_ROT_CCW_KICK);

        // Write the non-rotated advancement too
        maybe_push(arrival, parent, q, qh, qt, cx, cy, co, t + 1, ppc);
    }
}

void drm_reach_bfs(const uint16_t cols[8], int sx, int sy, int so, int nes_mode, bfs_result_t *out) {
    uint8_t fit[4][16][8];
    build_fit_planes(cols, fit, nes_mode ? 1 : 0);
    for (int i = 0; i < 512; ++i) { out->arrival[i] = 0xFF; out->parent[i] = 0; }
    if (!(so >= 0 && so < 4 && sx >= 0 && sx < 8 && sy >= 0 && sy < 16)) return;
    if (!fits(fit, sx, sy, so)) return;

    uint16_t q[512]; int qh = 0, qt = 0;
    const int sidx = drm_idx(sx, sy, so);
    out->arrival[sidx] = 0; out->parent[sidx] = PC_NONE; q[qt] = (uint16_t)sidx;
    while (qh != qt) {
        int idx = q[qh = (qh + 1) & 511];
        int o = (idx >> 7) & 3; int y = (idx >> 3) & 0xF; int x = idx & 7;
        int t = out->arrival[idx];
        if (nes_mode) expand_nes(fit, x, y, o, t, out->arrival, out->parent, q, &qh, &qt);
        else          expand_planner(fit, x, y, o, t, out->arrival, out->parent, q, &qh, &qt);
    }
}

