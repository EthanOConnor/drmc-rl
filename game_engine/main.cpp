#include "GameLogic.h"
#include "GameState.h"
#include "SharedMem.h"
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <string>
#include <cstdlib>

int main(int argc, char **argv) {
  bool demo_mode = false;
  bool no_sleep = false;
  bool wait_start = false;
  bool manual_step = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--demo")
      demo_mode = true;
    else if (arg == "--no-sleep")
      no_sleep = true;
    else if (arg == "--wait-start")
      wait_start = true;
    else if (arg == "--manual-step")
      manual_step = true;
  }

  // Shared Memory / file-backed mmap Setup
  const char *shm_file_env = std::getenv("DRMARIO_SHM_FILE");
  int shm_fd = -1;
  if (shm_file_env && shm_file_env[0] != '\0') {
    shm_fd = open(shm_file_env, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
      std::cerr << "Failed to open shm file: " << strerror(errno) << std::endl;
      return 1;
    }
  } else {
    shm_unlink(SharedMem::SHM_NAME.c_str()); // Unlink if exists to start fresh
    shm_fd = shm_open(SharedMem::SHM_NAME.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
      std::cerr << "Failed to open shared memory: " << strerror(errno)
                << std::endl;
      return 1;
    }
  }

  if (ftruncate(shm_fd, SharedMem::SHM_SIZE) == -1) {
    std::cerr << "Failed to truncate shared memory: " << strerror(errno)
              << std::endl;
    return 1;
  }

  void *ptr = mmap(0, SharedMem::SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                   shm_fd, 0);
  if (ptr == MAP_FAILED) {
    std::cerr << "Failed to map shared memory" << std::endl;
    return 1;
  }

  DrMarioState *state = static_cast<DrMarioState *>(ptr);
  std::memset(state, 0, sizeof(DrMarioState));

  const char *mode_env = std::getenv("DRMARIO_MODE");
  if (mode_env) {
    std::string env_mode = mode_env;
    if (env_mode == "demo" || env_mode == "DEMO") {
      demo_mode = true;
    } else if (env_mode == "play" || env_mode == "PLAY" ||
               env_mode == "game") {
      demo_mode = false;
    }
  }
  state->mode = demo_mode ? MODE_DEMO : MODE_PLAYING;

  const char *level_env = std::getenv("DRMARIO_LEVEL");
  int level = 0;
  if (level_env) {
    level = std::atoi(level_env);
  }
  if (level < 0)
    level = 0;
  if (level > 25)
    level = 25;
  state->level = static_cast<u8>(level);

  const char *speed_env = std::getenv("DRMARIO_SPEED");
  int speed_setting = 1;
  if (speed_env) {
    speed_setting = std::atoi(speed_env);
  }
  if (speed_setting < 0)
    speed_setting = 0;
  if (speed_setting > 2)
    speed_setting = 2;
  state->speed_setting = static_cast<u8>(speed_setting);

  // Informational bits in the high nibble (do not affect protocol):
  // 0x20 = wait_start requested, 0x40 = manual-step arg, 0x80 = no-sleep arg
  if (wait_start)
    state->control_flags |= 0x20;
  if (manual_step)
    state->control_flags |= 0x40;
  if (no_sleep)
    state->control_flags |= 0x80;
  if (manual_step)
    state->control_flags |= 0x02; // ensure manual-mode bit set for drivers

  // Initialize Game Logic
  GameLogic game(state);

  std::cout << "Dr. Mario Engine Running..."
            << (demo_mode ? " (demo mode)" : "") << std::endl;

  if (wait_start) {
    while ((state->control_flags & 0x01) == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Perform the initial reset only after the optional start gate is released,
  // so external drivers can pre-seed RNG/level settings before level setup.
  game.reset();

  // Main Loop
  // For now, run at ~60Hz to simulate NES speed
  auto next_frame = std::chrono::steady_clock::now();

  while (true) {
    // Out-of-band commands from the driver (do not require a step request).
    if (state->control_flags & 0x10) { // reset requested
      game.reset();
      state->control_flags &= static_cast<u8>(~0x10);
      continue;
    }

    const bool manual_mode = manual_step || ((state->control_flags & 0x02) != 0);
    if (manual_mode) { // manual stepping mode: only step when bit2 set
      if (state->control_flags & 0x04) {
        game.step();
        state->control_flags &= static_cast<u8>(~0x04);
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    } else {
      game.step();
      if (!no_sleep) {
        next_frame += std::chrono::microseconds(16667); // ~60 FPS
        std::this_thread::sleep_until(next_frame);
      }
    }

    if (state->frame_budget > 0 && state->frame_count >= state->frame_budget)
      break;
    if (state->control_flags & 0x08)
      break;
  }

  return 0;
}
