#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI / protocol versioning for the in-process pool.
//
// The intent is to keep this a stable C ABI that can be called via ctypes,
// similar to `reach_native`.
#define DRMARIO_POOL_PROTOCOL_VERSION 1u

typedef enum DrmPoolObsSpec {
  DRM_POOL_OBS_NONE = 0,
  // 4 channels: bitplane_bottle (color_{R,Y,B}, virus_mask)
  DRM_POOL_OBS_BITPLANE_BOTTLE = 1,
  // 8 channels: bitplane_bottle_mask (bitplane_bottle + feasible mask channels 4..7)
  DRM_POOL_OBS_BITPLANE_BOTTLE_MASK = 2,
} DrmPoolObsSpec;

typedef enum DrmPoolTerminalReason {
  DRM_POOL_TERMINAL_NONE = 0,
  DRM_POOL_TERMINAL_CLEAR = 1,
  DRM_POOL_TERMINAL_TOPOUT = 2,
  DRM_POOL_TERMINAL_TIMEOUT = 3,
} DrmPoolTerminalReason;

typedef struct DrmPoolConfig {
  uint32_t protocol_version;  // must be DRMARIO_POOL_PROTOCOL_VERSION
  uint32_t struct_size;       // must be sizeof(DrmPoolConfig)

  uint32_t num_envs;
  uint32_t obs_spec;          // DrmPoolObsSpec
  uint32_t max_lock_frames;   // planner depth cap (BFS)
  uint32_t max_wait_frames;   // decision-point wait cap (safety)
} DrmPoolConfig;

typedef struct DrmResetSpec {
  uint32_t struct_size;  // must be sizeof(DrmResetSpec)

  // Core game config (clamped to ROM-valid ranges).
  int32_t level;          // 0..25 (negative values clamp to 0)
  int32_t speed_setting;  // 0..2
  int32_t speed_ups;      // 0..255 (optional override)

  // Determinism / parity knobs.
  uint8_t rng_state[2];   // engine LFSR seed bytes
  uint8_t rng_override;   // if non-zero, use rng_state on reset
  uint8_t intro_wait_frames;  // if non-zero, seed waitFrames ($0051) before gameplay
  uint8_t _reserved0;
  uint16_t intro_frame_counter_lo_plus1;  // if non-zero, seeds frameCounter low byte (+1 encoding)

  // Synthetic curriculum helper: remove viruses down to this target after level gen.
  // Use -1 to disable (default).
  int32_t synthetic_virus_target;    // -1 (disabled) or 0..255
  uint8_t synthetic_patch_counter;   // if non-zero, patch the BCD virus counter to match target
  uint8_t _reserved1[3];
  uint32_t synthetic_seed;           // used only when synthetic_virus_target >= 0
} DrmResetSpec;

typedef struct DrmPoolOutputs {
  uint32_t struct_size;  // must be sizeof(DrmPoolOutputs)

  // Decision-time outputs (valid after reset and after each accepted step).
  // Shapes:
  //   obs:            [N, C, 16, 8] float32
  //   feasible_mask:  [N, 512] uint8 (0/1)
  //   cost_to_lock:   [N, 512] uint16 (0xFFFF == unreachable)
  //   pill_colors:    [N, 2] uint8 (canonical indices 0=R,1=Y,2=B)
  //   preview_colors: [N, 2] uint8 (canonical indices 0=R,1=Y,2=B)
  //   spawn_id:       [N] uint8
  //   viruses_rem:    [N] uint16 (decoded count from board)
  //   board_bytes:    [N, 128] uint8 (NES tile encoding; optional debug output)
  float* obs;
  uint8_t* feasible_mask;
  uint16_t* cost_to_lock;
  uint8_t* pill_colors;
  uint8_t* preview_colors;
  uint8_t* spawn_id;
  uint16_t* viruses_rem;
  uint8_t* board_bytes;

  // Step-only outputs (valid after drm_pool_step).
  uint32_t* tau_frames;          // [N] frames advanced for accepted actions (0 for invalid/reset envs)
  uint8_t* terminated;           // [N] 0/1
  uint8_t* truncated;            // [N] 0/1 (engine-side timeout only)
  uint8_t* terminal_reason;      // [N] DrmPoolTerminalReason
  int32_t* invalid_action;       // [N] -1 if accepted, else the invalid action index

  uint16_t* tiles_cleared_total;     // [N]
  uint16_t* tiles_cleared_virus;     // [N]
  uint16_t* tiles_cleared_nonvirus;  // [N]
  uint16_t* match_events;            // [N] count of clear events (rising edge) during this macro step

  // Adjacency shaping flags, per color (R,Y,B). Values are 0/1.
  uint8_t* adj_pair;           // [N,3]
  uint8_t* adj_triplet;        // [N,3]
  uint8_t* virus_adj_pair;     // [N,3]
  uint8_t* virus_adj_triplet;  // [N,3]

  // Debug: observed lock pose for the executed action, in planner coordinates.
  // Base cell uses (x,y,rot) where y is top-origin (0=top,15=bottom).
  int16_t* lock_x;    // [N]
  int16_t* lock_y;    // [N]
  int16_t* lock_rot;  // [N]
} DrmPoolOutputs;

void* drm_pool_create(const DrmPoolConfig* cfg);
void drm_pool_destroy(void* handle);

// Reset selected envs (mask bytes are 0/1). If reset_mask is NULL, resets all envs.
// Writes decision outputs for the post-reset first actionable decision.
int drm_pool_reset(void* handle, const uint8_t* reset_mask, const DrmResetSpec* reset_specs,
                   DrmPoolOutputs* out);

// Step selected envs at decision boundaries. If reset_mask[i] is 1, env i is reset
// and the corresponding action is ignored (NEXT_STEP autoreset semantics).
//
// Writes:
//  - decision outputs for the post-step state (or post-reset state)
//  - step-only outputs for envs whose action was accepted
int drm_pool_step(void* handle, const int32_t* actions, const uint8_t* reset_mask,
                  const DrmResetSpec* reset_specs, DrmPoolOutputs* out);

#ifdef __cplusplus
} // extern "C"
#endif
