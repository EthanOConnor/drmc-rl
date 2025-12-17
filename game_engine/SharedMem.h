#pragma once

#include "GameState.h"
#include <string>

namespace SharedMem {
const std::string SHM_NAME = "/drmario_shm";
const size_t SHM_SIZE = sizeof(DrMarioState);
} // namespace SharedMem
