#include "DrMarioPool.h"

#include <new>

namespace {
int validate_cfg(const DrmPoolConfig* cfg) {
  if (cfg == nullptr)
    return -1;
  if (cfg->protocol_version != DRMARIO_POOL_PROTOCOL_VERSION)
    return -2;
  if (cfg->struct_size != sizeof(DrmPoolConfig))
    return -3;
  return 0;
}
} // namespace

extern "C" {

void* drm_pool_create(const DrmPoolConfig* cfg) {
  if (validate_cfg(cfg) != 0)
    return nullptr;
  try {
    return new DrMarioPool(*cfg);
  } catch (const std::bad_alloc&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

void drm_pool_destroy(void* handle) {
  if (handle == nullptr)
    return;
  try {
    delete static_cast<DrMarioPool*>(handle);
  } catch (...) {
    // Never throw across C ABI.
  }
}

int drm_pool_reset(void* handle, const uint8_t* reset_mask, const DrmResetSpec* reset_specs,
                   DrmPoolOutputs* out) {
  if (handle == nullptr)
    return -1;
  auto* pool = static_cast<DrMarioPool*>(handle);
  try {
    return pool->reset(reset_mask, reset_specs, out);
  } catch (...) {
    return -99;
  }
}

int drm_pool_step(void* handle, const int32_t* actions, const uint8_t* reset_mask,
                  const DrmResetSpec* reset_specs, DrmPoolOutputs* out) {
  if (handle == nullptr)
    return -1;
  auto* pool = static_cast<DrMarioPool*>(handle);
  try {
    return pool->step(actions, reset_mask, reset_specs, out);
  } catch (...) {
    return -99;
  }
}

} // extern "C"

