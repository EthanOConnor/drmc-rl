/* Native twin of the decision-time knob features (drmc_rl/style/showy_knob.py).
 *
 * One call per decision: every candidate's settled afterstate
 * (drmc_rl.game.afterstate.resolve_placement), its engineered board features
 * (showy_knob.board_features, all but log_occ), the exact trigger features
 * (showy_knob.trigger_features) and the placement's own clear
 * (showy_knob.immediate_clears). Everything is integer arithmetic (the only
 * non-integer values are multiples of 1/8 and 1/2, exact in float32), so the
 * results equal the numpy reference bit for bit; tests/test_knob_native.py
 * checks that on recorded decisions.
 *
 * The cascade is the ROM's (drmc_rl.game.cascade / drmc_rl.eval.big_clear,
 * ported from drmario-native GameLogic.cpp). Build: see native.py.
 */
#include <stdint.h>
#include <string.h>

#define W 8
#define H 16
#define N 128
#define T_EMPTY 0xFF
#define T_TOP 0x40
#define T_BOTTOM 0x50
#define T_LEFT 0x60
#define T_RIGHT 0x70
#define T_SINGLE 0x80
#define T_MID_VER 0x90
#define T_MID_HOR 0xA0
#define T_CLEARED 0xB0
#define T_JUST_EMPTIED 0xF0

#define NBOARD 57   /* board features, log_occ excluded (computed in numpy) */
#define NTRIG 12

int knob_features_abi(void) { return 1; }

static int drop_pass(uint8_t *b) {
    int any = 0;
    for (int pos = N - 1; pos >= 0; pos--) {
        uint8_t cur = b[pos];
        if (cur < T_JUST_EMPTIED) continue;
        b[pos] = T_EMPTY;
        int above = pos - W;
        if (above < 0) continue;
        uint8_t top = b[above];
        if (top >= T_MID_HOR) continue;
        uint8_t tt = top & 0xF0;
        if (tt == T_LEFT || tt == T_MID_HOR) continue;
        if (tt == T_RIGHT) {
            int top_pos = above, bottom_pos = pos, top_left = above, bottom_left = pos, blocked = 0;
            for (;;) {
                top_left--; bottom_left--;
                if (bottom_left < 0 || top_left < 0) break;
                if (b[bottom_left] < T_JUST_EMPTIED) { blocked = 1; break; }
                uint8_t tl = b[top_left] & 0xF0;
                if (tl == T_LEFT || tl != T_MID_HOR) break;
            }
            if (blocked) continue;
            for (;;) {
                b[bottom_pos] = b[top_pos];
                b[top_pos] = T_EMPTY;
                if (bottom_pos == bottom_left) break;
                bottom_pos--; top_pos--;
            }
            any = 1;
            continue;
        }
        b[pos] = b[above];
        b[above] = T_EMPTY;
        any = 1;
    }
    return any;
}

static void update_field(uint8_t *b) {
    for (int pos = N - 1; pos >= 0; pos--) {
        uint8_t t = b[pos], ty = t & 0xF0;
        if (ty == T_CLEARED) { b[pos] = t | T_JUST_EMPTIED; continue; }
        if (ty == T_TOP) {
            int below = pos + W;
            uint8_t bt = below < N ? (b[below] & 0xF0) : 0xF0;
            if (bt != T_BOTTOM && bt != T_MID_VER) b[pos] = T_SINGLE | (t & 0x0F);
        } else if (ty == T_BOTTOM) {
            int above = pos - W;
            uint8_t at = above >= 0 ? (b[above] & 0xF0) : 0xF0;
            if (at != T_TOP && at != T_MID_VER) b[pos] = T_SINGLE | (t & 0x0F);
        } else if (ty == T_LEFT) {
            int right = pos + 1;
            uint8_t rt = (right < N && right % W != 0) ? (b[right] & 0xF0) : 0xF0;
            if (rt != T_RIGHT && rt != T_MID_HOR) b[pos] = T_SINGLE | (t & 0x0F);
        } else if (ty == T_RIGHT) {
            int left = pos - 1;
            uint8_t lt = (left >= 0 && left % W != W - 1) ? (b[left] & 0xF0) : 0xF0;
            if (lt != T_LEFT && lt != T_MID_HOR) b[pos] = T_SINGLE | (t & 0x0F);
        }
    }
}

typedef struct {
    int cells, viruses, rounds, lines, max_round_lines, long_tiles, max_line, span_rows, cross, colors, garbage, horizontal;
} clear_t;

/* One ROM scan (big_clear._mark_lines): marks lines, flags cells per orientation. Returns lines found. */
static int mark_lines(uint8_t *b, uint8_t *hcell, uint8_t *vcell, int *long_tiles, int *max_line, int *colors, int *hcount) {
    int found = 0;
    for (int row = 0; row < H; row++) {
        int col = 0;
        while (col <= W - 4) {
            uint8_t tile = b[row * W + col];
            if (tile >= T_JUST_EMPTIED) { col++; continue; }
            uint8_t color = tile & 0x0F;
            int chain = 1;
            while (col + chain < W && (b[row * W + col + chain] & 0x0F) == color) chain++;
            if (chain >= 4) {
                found++; (*hcount)++;
                *long_tiles += chain - 4;
                if (chain > *max_line) *max_line = chain;
                *colors |= 1 << (color & 3);
                for (int k = 0; k < chain; k++) {
                    int i = row * W + col + k;
                    hcell[i] = 1;
                    b[i] = T_CLEARED | (b[i] & 0x0F);
                }
                col += chain;
            } else col++;
        }
    }
    for (int col = 0; col < W; col++) {
        int row = 0;
        while (row <= H - 4) {
            uint8_t tile = b[row * W + col];
            if (tile >= T_JUST_EMPTIED) { row++; continue; }
            uint8_t color = tile & 0x0F;
            int chain = 1;
            while (row + chain < H && (b[(row + chain) * W + col] & 0x0F) == color) chain++;
            if (chain >= 4) {
                found++;
                *long_tiles += chain - 4;
                if (chain > *max_line) *max_line = chain;
                *colors |= 1 << (color & 3);
                for (int k = 0; k < chain; k++) {
                    int i = (row + k) * W + col;
                    vcell[i] = 1;
                    b[i] = T_CLEARED | (b[i] & 0x0F);
                }
                row += chain;
            } else row++;
        }
    }
    return found;
}

/* big_clear.resolve: settle in place, fill the clear features. */
static void resolve(uint8_t *b, clear_t *f) {
    memset(f, 0, sizeof *f);
    int colors = 0, rmin = 99, rmax = -1;
    uint8_t pre[N], hcell[N], vcell[N];
    for (;;) {
        while (drop_pass(b)) {}
        memcpy(pre, b, N);
        memset(hcell, 0, N); memset(vcell, 0, N);
        int found = mark_lines(b, hcell, vcell, &f->long_tiles, &f->max_line, &colors, &f->horizontal);
        if (!found) break;
        f->rounds++;
        int cross = 0;
        for (int i = 0; i < N; i++) {
            if (hcell[i] && vcell[i]) cross = 1;
            if ((b[i] & 0xF0) == T_CLEARED && (pre[i] & 0xF0) != T_CLEARED) {
                f->cells++;
                if ((pre[i] & 0xF0) == 0xD0) f->viruses++;
                int r = i / W;
                if (r < rmin) rmin = r;
                if (r > rmax) rmax = r;
            }
        }
        f->cross += cross;
        f->lines += found;
        if (found > f->max_round_lines) f->max_round_lines = found;
        update_field(b);
    }
    for (int i = 0; i < N; i++) if (b[i] >= T_JUST_EMPTIED) b[i] = T_EMPTY;
    f->colors = (colors & 1) + ((colors >> 1) & 1) + ((colors >> 2) & 1) + ((colors >> 3) & 1);
    f->span_rows = rmax >= 0 ? rmax - rmin + 1 : 0;
    f->garbage = f->lines >= 2 ? (f->lines < 4 ? f->lines : 4) : 0;
}

static double score(const clear_t *f) {   /* ClearFeatures.score (big_clear.WEIGHTS) */
    if (!f->rounds) return 0.0;
    double s = 0.0;
    s += 1.0 * (f->cells > 4 ? f->cells - 4 : 0);
    s += 3.0 * (f->rounds > 1 ? f->rounds - 1 : 0);
    s += 2.0 * (f->max_round_lines > 1 ? f->max_round_lines - 1 : 0);
    s += 1.5 * f->long_tiles;
    s += 1.0 * (f->viruses > 2 ? f->viruses - 2 : 0);
    s += 0.5 * (f->span_rows > 6 ? f->span_rows - 6 : 0);
    s += 2.0 * (f->cross > 0);
    s += 2.0 * (f->colors >= 3);
    s += f->garbage >= 4 ? 2.0 : f->garbage == 3 ? 1.0 : 0.0;
    return s;
}

static int forms_line(const uint8_t *p, int index) {   /* big_clear.forms_line for one cell */
    uint8_t color = p[index] & 0x0F;
    int row = index / W, col = index % W, run = 1;
    for (int c = col - 1; c >= 0 && p[row * W + c] != T_EMPTY && (p[row * W + c] & 0x0F) == color; c--) run++;
    for (int c = col + 1; c < W && p[row * W + c] != T_EMPTY && (p[row * W + c] & 0x0F) == color; c++) run++;
    if (run >= 4) return 1;
    run = 1;
    for (int r = row - 1; r >= 0 && p[r * W + col] != T_EMPTY && (p[r * W + col] & 0x0F) == color; r--) run++;
    for (int r = row + 1; r < H && p[r * W + col] != T_EMPTY && (p[r * W + col] & 0x0F) == color; r++) run++;
    return run >= 4;
}

static const int SECOND[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
static const uint8_t TILES[4][2] = {{0x60, 0x70}, {0x40, 0x50}, {0x70, 0x60}, {0x50, 0x40}};
static const uint8_t NES[3] = {1, 0, 2};

static void trigger_row(const uint8_t *f, float *row) {   /* showy_knob.trigger_features, one bottle */
    for (int k = 0; k < NTRIG; k++) row[k] = 0.0f;
    for (int col = 0; col < W; col++) {
        int depth = H;
        for (int r = 0; r < H; r++) if (f[r * W + col] != T_EMPTY) { depth = r; break; }
        int r = depth - 1;
        if (r < 0) continue;
        int index = r * W + col;
        for (int c = 0; c < 3; c++) {
            uint8_t b[N];
            memcpy(b, f, N);
            b[index] = 0x80 | NES[c];
            if (!forms_line(b, index)) continue;
            clear_t ft;
            resolve(b, &ft);
            if (!ft.rounds) continue;
            float s = (float)score(&ft);
            row[0] += 1.0f;
            row[1] += (float)(ft.lines >= 2 || ft.rounds >= 2);
            if (s > row[2]) row[2] = s;
            row[3] += s;
            if ((float)ft.rounds > row[4]) row[4] = (float)ft.rounds;
            if ((float)ft.lines > row[5]) row[5] = (float)ft.lines;
            row[6] += (float)(ft.horizontal > 0);
            row[7] += (float)(s >= 27.0f);
            row[8] += (float)(ft.lines >= 2);
            row[9] += (float)(ft.lines >= 4);
            float g = ft.lines >= 2 ? (float)(ft.lines < 4 ? ft.lines : 4) : 0.0f;
            if (g > row[10]) row[10] = g;
            if ((float)(ft.lines - 4) > row[11]) row[11] = (float)(ft.lines - 4);
        }
    }
}

enum { P_H3, P_H3S, P_H2, P_H2S, P_H5_4, P_H6_4, P_H6_5, P_V3, P_V2, P_HV3, P_VIR_H3, P_VIR_V3, P_SURF3, P_SURF2, P_STACK, NPER };

static int hwin(const uint8_t *a, int row, int s, int w) {
    int n = 0;
    for (int k = 0; k < w; k++) n += a[row * W + s + k];
    return n;
}

static int vwin(const uint8_t *a, int s, int col, int w) {
    int n = 0;
    for (int k = 0; k < w; k++) n += a[(s + k) * W + col];
    return n;
}

static void board_row(const uint8_t *f, float *out) {   /* showy_knob.board_features minus log_occ */
    uint8_t occ[N], empty[N], color[N], virus[N], supported[N], above_clear[N], surface[N];
    int depth[W], heights[W];
    for (int i = 0; i < N; i++) {
        occ[i] = f[i] != T_EMPTY;
        empty[i] = !occ[i];
        color[i] = occ[i] ? (f[i] & 3) : 3;
        virus[i] = occ[i] && (f[i] & 0xF0) == 0xD0;
    }
    int occupied = 0, viruses = 0, holes = 0, surface_cells = 0;
    for (int c = 0; c < W; c++) {
        depth[c] = H;
        int seen = 0;
        for (int r = 0; r < H; r++) {
            int i = r * W + c;
            if (occ[i] && !seen) { depth[c] = r; }
            seen |= occ[i];
            above_clear[i] = !seen;
            int below_solid = r < H - 1 ? occ[i + W] : 1;
            supported[i] = empty[i] && below_solid;
            surface[i] = supported[i] && above_clear[i];
            holes += empty[i] && !above_clear[i];
            surface_cells += surface[i];
            occupied += occ[i];
            viruses += virus[i];
        }
        heights[c] = H - depth[c];
    }
    int hmax = 0, hsum = 0, bump = 0, tall = 0, danger = 0;
    for (int c = 0; c < W; c++) {
        if (heights[c] > hmax) hmax = heights[c];
        hsum += heights[c];
        if (c) bump += heights[c] > heights[c - 1] ? heights[c] - heights[c - 1] : heights[c - 1] - heights[c];
        tall += heights[c] >= 12;
    }
    for (int r = 0; r < 3; r++) danger += occ[r * W + 3] + occ[r * W + 4];
    int per[NPER][3];
    memset(per, 0, sizeof per);
    for (int cl = 0; cl < 3; cl++) {
        uint8_t cc[N], cv[N];
        for (int i = 0; i < N; i++) { cc[i] = color[i] == cl; cv[i] = cc[i] && virus[i]; }
        int h3rows = 0, v3any = 0;
        for (int row = 0; row < H; row++) {
            int rowh3 = 0;
            for (int s = 0; s <= W - 4; s++) {
                int n4 = hwin(cc, row, s, 4), e4 = hwin(empty, row, s, 4);
                int h3 = n4 == 3 && e4 == 1, h2 = n4 == 2 && e4 == 2;
                int sup = hwin(supported, row, s, 4) >= 1;
                per[P_H3][cl] += h3;
                per[P_H3S][cl] += h3 && sup;
                per[P_H2][cl] += h2;
                per[P_H2S][cl] += h2 && sup;
                per[P_VIR_H3][cl] += h3 && hwin(cv, row, s, 4) > 0;
                rowh3 |= h3;
            }
            for (int s = 0; s <= W - 5; s++)
                per[P_H5_4][cl] += hwin(cc, row, s, 5) == 4 && hwin(empty, row, s, 5) == 1;
            for (int s = 0; s <= W - 6; s++) {
                int n6 = hwin(cc, row, s, 6), e6 = hwin(empty, row, s, 6);
                per[P_H6_4][cl] += n6 == 4 && e6 == 2;
                per[P_H6_5][cl] += n6 == 5 && e6 == 1;
            }
            h3rows += rowh3;
        }
        for (int s = 0; s <= H - 4; s++) {
            for (int col = 0; col < W; col++) {
                int v4 = vwin(cc, s, col, 4), e4 = vwin(empty, s, col, 4);
                int top = empty[s * W + col];
                int v3 = v4 == 3 && e4 == 1 && top;
                int v2 = v4 == 2 && e4 == 2 && top && empty[(s + 1) * W + col];
                per[P_V3][cl] += v3;
                per[P_V2][cl] += v2;
                per[P_VIR_V3][cl] += v3 && vwin(cv, s, col, 4) > 0;
                v3any |= v3;
            }
        }
        per[P_HV3][cl] = h3rows > 0 && v3any;
        for (int col = 0; col < W; col++) {
            int run = 0, alive = 1;
            for (int k = 0; k < 4; k++) {
                int r = depth[col] + k;
                int same = r < H && cc[r * W + col] && alive;
                run += same;
                alive = alive && same;
            }
            per[P_SURF3][cl] += run >= 3;
            per[P_SURF2][cl] += run == 2;
        }
        per[P_STACK][cl] = h3rows;
    }
    int k = 0;
    out[k++] = (float)occupied;
    out[k++] = (float)viruses;
    out[k++] = (float)hmax;
    out[k++] = (float)((double)hsum / 8.0);
    out[k++] = (float)bump;
    out[k++] = (float)holes;
    out[k++] = (float)danger;
    out[k++] = (float)tall;
    float h3s_sum = 0, surf3_sum = 0, stack_sum = 0;
    for (int p = 0; p < NPER; p++) {
        int sum = per[p][0] + per[p][1] + per[p][2];
        int mx = per[p][0];
        if (per[p][1] > mx) mx = per[p][1];
        if (per[p][2] > mx) mx = per[p][2];
        int colors = (per[p][0] > 0) + (per[p][1] > 0) + (per[p][2] > 0);
        out[k++] = (float)sum;
        out[k++] = (float)mx;
        out[k++] = (float)colors;
        if (p == P_H3S) h3s_sum = (float)sum;
        if (p == P_SURF3) surf3_sum = (float)sum;
        if (p == P_STACK) stack_sum = (float)sum;
    }
    float threats = h3s_sum + surf3_sum;
    out[k++] = threats;
    out[k++] = threats * threats;
    out[k++] = stack_sum;
    out[k++] = (float)surface_cells;
}

/* One decision. For each of n actions (orientation*128 + anchor cell) from `root` with canonical
 * pill colors (c0, c1): after[n*128] settled afterstate, board[n*57], trig[n*12] (when want_trig),
 * imm_score[n] (-1: no clear) and imm_lines[n]. Returns 0, or -(1+j) when action j is not a
 * placement into empty bottle cells (the caller raises like the reference). */
int knob_decision(const uint8_t *root, int c0, int c1, const int64_t *actions, int n, int want_trig,
                  uint8_t *after, float *board, float *trig, float *imm_score, int32_t *imm_lines) {
    if (c0 < 0 || c0 > 2 || c1 < 0 || c1 > 2) return -1000000;
    int pill[2] = {c0, c1};
    for (int j = 0; j < n; j++) {
        int64_t action = actions[j];
        if (action < 0 || action >= 512) return -(1 + j);
        int o = (int)(action / 128), cell = (int)(action % 128);
        int row = cell / W, col = cell % W;
        int rr[2] = {row, row + SECOND[o][0]}, cc[2] = {col, col + SECOND[o][1]};
        uint8_t *b = after + (size_t)j * N;
        memcpy(b, root, N);
        for (int h = 0; h < 2; h++) {
            if (rr[h] < 0 || rr[h] >= H || cc[h] < 0 || cc[h] >= W || b[rr[h] * W + cc[h]] != T_EMPTY) return -(1 + j);
            b[rr[h] * W + cc[h]] = TILES[o][h] | NES[pill[h]];
        }
        int line = forms_line(b, rr[0] * W + cc[0]) || forms_line(b, rr[1] * W + cc[1]);
        clear_t f;
        resolve(b, &f);
        imm_score[j] = -1.0f;
        imm_lines[j] = 0;
        if (line && f.rounds) {
            imm_score[j] = (float)score(&f);
            imm_lines[j] = f.lines;
        }
        board_row(b, board + (size_t)j * NBOARD);
        if (want_trig) trigger_row(b, trig + (size_t)j * NTRIG);
    }
    return 0;
}
