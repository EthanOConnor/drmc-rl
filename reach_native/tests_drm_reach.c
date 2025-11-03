#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

typedef struct { uint8_t arrival[512]; uint8_t parent[512]; } bfs_result_t;
void drm_reach_bfs(const uint16_t cols[8], int sx, int sy, int so, int nes_mode, bfs_result_t *out);

static void test_empty_drop() {
    uint16_t cols[8] = {0};
    bfs_result_t res; memset(&res, 0, sizeof(res));
    drm_reach_bfs(cols, 3, 0, 0, 0, &res);
    // Expect arrival at (3,y,0) equals y
    for (int y = 0; y < 16; ++y) {
        int idx = (0 << 7) | (y << 3) | 3;
        assert(res.arrival[idx] == (uint8_t)y);
    }
}

int main(void) {
    test_empty_drop();
    printf("OK\n");
    return 0;
}

