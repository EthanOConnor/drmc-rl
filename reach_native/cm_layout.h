// Column-major fit-plane helpers for Dr. Mario (NEON + SSE2)
//
// Layout:
// - 8 lanes of 16 bits (one lane per column)
// - bit 0 = top row, bit 15 = bottom row
// - bit 1 = occupied, bit 0 = empty
//
// Orientation names (frozen):
//   O_HPOS = 0  // horizontal, partner at x+1
//   O_VPOS = 1  // vertical,   partner at y+1
//   O_HNEG = 2  // horizontal, partner at x-1
//   O_VNEG = 3  // vertical,   partner at y-1

#pragma once

#include <stdint.h>

enum {
    O_HPOS = 0,
    O_VPOS = 1,
    O_HNEG = 2,
    O_VNEG = 3
};

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

static inline uint16x8_t cm_lane_shift_right1_u16(uint16x8_t cols) { // c+1 -> c
    uint8x16_t b = vreinterpretq_u8_u16(cols);
    uint8x16_t s = vextq_u8(b, vdupq_n_u8(0), 2);
    return vreinterpretq_u16_u8(s);
}
static inline uint16x8_t cm_lane_shift_left1_u16(uint16x8_t cols) {  // c-1 -> c
    uint8x16_t b = vreinterpretq_u8_u16(cols);
    uint8x16_t s = vextq_u8(vdupq_n_u8(0), b, 14);
    return vreinterpretq_u16_u8(s);
}

static inline uint16x8_t cm_mask_edge_left_lane(void)  { return vsetq_lane_u16(0xFFFF, vdupq_n_u16(0), 0); }
static inline uint16x8_t cm_mask_edge_right_lane(void) { return vsetq_lane_u16(0xFFFF, vdupq_n_u16(0), 7); }

static inline uint16x8_t cm_ref_occ_hpos(uint16x8_t occ) {
    uint16x8_t sh = cm_lane_shift_right1_u16(occ);
    return vorrq_u16(vorrq_u16(occ, sh), cm_mask_edge_right_lane());
}
static inline uint16x8_t cm_ref_occ_hneg(uint16x8_t occ) {
    uint16x8_t sh = cm_lane_shift_left1_u16(occ);
    return vorrq_u16(vorrq_u16(occ, sh), cm_mask_edge_left_lane());
}
static inline uint16x8_t cm_ref_occ_vpos(uint16x8_t occ) {
    uint16x8_t sh = vsriq_n_u16(vdupq_n_u16(0xFFFF), occ, 1);  // pull 1 down, force bottom=1
    return vorrq_u16(occ, sh);
}
static inline uint16x8_t cm_ref_occ_vneg(uint16x8_t occ) {
    uint16x8_t sh = vsliq_n_u16(vdupq_n_u16(0xFFFF), occ, 1);  // push 1 up, force top=1
    return vorrq_u16(occ, sh);
}

static inline void cm_build_fit_planes_neon(
    const uint16x8_t occ,
    uint16x8_t *fit0,
    uint16x8_t *fit1,
    uint16x8_t *fit2,
    uint16x8_t *fit3,
    int nes_parity
) {
    const uint16x8_t ALL1 = vdupq_n_u16(0xFFFF);
    uint16x8_t occ_hpos = cm_ref_occ_hpos(occ);
    uint16x8_t occ_vpos = cm_ref_occ_vpos(occ);
    uint16x8_t occ_hneg = cm_ref_occ_hneg(occ);
    uint16x8_t occ_vneg = cm_ref_occ_vneg(occ);
    uint16x8_t f0 = veorq_u16(occ_hpos, ALL1);
    uint16x8_t f1 = veorq_u16(occ_vpos, ALL1);
    uint16x8_t f2 = veorq_u16(occ_hneg, ALL1);
    uint16x8_t f3 = veorq_u16(occ_vneg, ALL1);
    if (nes_parity) { f2 = f0; f3 = f1; }
    *fit0 = f0; *fit1 = f1; *fit2 = f2; *fit3 = f3;
}

#endif

#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>

static inline __m128i cm_lane_shift_right1_u16(__m128i cols) { return _mm_srli_si128(cols, 2); }
static inline __m128i cm_lane_shift_left1_u16 (__m128i cols) { return _mm_slli_si128(cols, 2); }

static inline __m128i cm_mask_edge_left_lane (void) { return _mm_set_epi16(0,0,0,0,0,0,0,(short)0xFFFF); }
static inline __m128i cm_mask_edge_right_lane(void) { return _mm_set_epi16((short)0xFFFF,0,0,0,0,0,0,0); }

static inline __m128i cm_ref_occ_hpos(__m128i occ) {
    __m128i sh = cm_lane_shift_right1_u16(occ);
    return _mm_or_si128(_mm_or_si128(occ, sh), cm_mask_edge_right_lane());
}
static inline __m128i cm_ref_occ_hneg(__m128i occ) {
    __m128i sh = cm_lane_shift_left1_u16(occ);
    return _mm_or_si128(_mm_or_si128(occ, sh), cm_mask_edge_left_lane());
}
static inline __m128i cm_ref_occ_vpos(__m128i occ) {
    __m128i sh = _mm_or_si128(_mm_srli_epi16(occ, 1), _mm_set1_epi16((short)0x8000));
    return _mm_or_si128(occ, sh);
}
static inline __m128i cm_ref_occ_vneg(__m128i occ) {
    __m128i sh = _mm_or_si128(_mm_slli_epi16(occ, 1), _mm_set1_epi16(0x0001));
    return _mm_or_si128(occ, sh);
}

static inline void cm_build_fit_planes_sse2(
    const __m128i occ,
    __m128i *fit0,
    __m128i *fit1,
    __m128i *fit2,
    __m128i *fit3,
    int nes_parity
) {
    const __m128i ALL1 = _mm_set1_epi16((short)0xFFFF);
    __m128i occ_hpos = cm_ref_occ_hpos(occ);
    __m128i occ_vpos = cm_ref_occ_vpos(occ);
    __m128i occ_hneg = cm_ref_occ_hneg(occ);
    __m128i occ_vneg = cm_ref_occ_vneg(occ);
    __m128i f0 = _mm_xor_si128(occ_hpos, ALL1);
    __m128i f1 = _mm_xor_si128(occ_vpos, ALL1);
    __m128i f2 = _mm_xor_si128(occ_hneg, ALL1);
    __m128i f3 = _mm_xor_si128(occ_vneg, ALL1);
    if (nes_parity) { f2 = f0; f3 = f1; }
    *fit0 = f0; *fit1 = f1; *fit2 = f2; *fit3 = f3;
}

#endif

