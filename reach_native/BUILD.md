Build and Inspect (Native Planner)

Requirements
- Clang or GCC with SSE2/NEON support

Compile (Apple Silicon, NEON)
- clang -O3 -mcpu=apple-m1 -c drm_reach.c -o drm_reach.o
- clang -O3 -mcpu=apple-m1 -S reach_native/cm_layout.h -o /dev/null  # header-only SIMD helpers
- clang -O3 -mcpu=apple-m1 tests_drm_reach.c drm_reach.c -o tests_drm_reach

Compile (x86-64, SSE2)
- clang -O3 -msse2 tests_drm_reach.c drm_reach.c -o tests_drm_reach

Emit Assembly for SIMD helpers
- Create small TU wrappers that call cm_ref_occ_* and compile with -S:
  - clang -O3 -mcpu=apple-m1 -S cm_layout_neon.c -o cm_layout_neon.s
  - clang -O3 -msse2 -S cm_layout_sse.c -o cm_layout_sse.s

What to verify (per spec)
- NEON: 2–3 vector ops in cm_ref_occ_*; inverts are eor against 0xFFFF
- SSE2: shifts are psrldq/pslldq + por chains; pxor with all-ones for invert

Disassembly Cross-Checks
- In dr-mario-disassembly files (drmario_prg_game_logic.asm):
  - fallingPill_checkYMove, fallingPill_checkXMove, fallingPill_checkRotate order
  - pillRotateValidation masks rotation with and #$01 (parity)
  - double-left attempt and wall-kick left in rotation routine
  - pillMoveValidation checks both halves for occupancy

