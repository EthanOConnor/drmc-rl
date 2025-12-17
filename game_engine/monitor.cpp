#include "GameState.h"
#include "SharedMem.h"
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

constexpr u8 TILE_EMPTY = 0xFF;
constexpr u8 TILE_VIRUS_START = 0xD0;
constexpr u8 TILE_LEFT = 0x60;
constexpr u8 TILE_RIGHT = 0x70;
constexpr u8 TILE_TOP = 0x40;
constexpr u8 TILE_BOTTOM = 0x50;
constexpr u8 TILE_SINGLE = 0x80;
constexpr u8 MASK_COLOR = 0x03;
constexpr u8 MASK_TYPE = 0xF0;

int main() {
  int shm_fd = shm_open(SharedMem::SHM_NAME.c_str(), O_RDONLY, 0666);
  if (shm_fd == -1) {
    std::cerr << "Shared memory open failed" << std::endl;
    return 1;
  }

  void *ptr = mmap(0, SharedMem::SHM_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    std::cerr << "Map failed" << std::endl;
    return 1;
  }

  DrMarioState *state = static_cast<DrMarioState *>(ptr);

  std::cout << "Monitoring Dr. Mario Engine..." << std::endl;

  while (true) {
    // Clear screen (ANSI)
    std::cout << "\033[2J\033[1;1H";

    std::cout << "Dr. Mario Engine Monitor" << std::endl;
    std::cout << "Frame: " << state->frame_count << std::endl;
    std::cout << "Mode: " << (int)state->mode << std::endl;
    std::cout << "State: " << (int)state->stage_clear << std::endl;
    std::cout << "Level: " << (int)state->level << std::endl;
    std::cout << "Speed: " << (int)state->speed_setting << std::endl;
    std::cout << "SpeedUps: " << (int)state->speed_ups << std::endl;
    std::cout << "DAS: " << (int)state->hor_velocity << std::endl;
    std::cout << "Viruses: " << (int)state->viruses_remaining << std::endl;
    std::cout << "Pill Counter: " << (int)state->pill_counter << std::endl;
    std::cout << "Pill Counter Total: " << (int)state->pill_counter_total
              << std::endl;
    std::cout << "Falling Pill: (" << (int)state->falling_pill_row << ", "
              << (int)state->falling_pill_col
              << ") Orient: " << (int)state->falling_pill_orient << std::endl;

    std::cout << "Board:" << std::endl;
    for (int r = 0; r < 16; r++) {
      for (int c = 0; c < 8; c++) {
        int idx = r * 8 + c; // Row 0 is top
        u8 tile = state->board[idx];
        if (tile == TILE_EMPTY) {
          std::cout << ".. ";
          continue;
        }

        u8 type = tile & MASK_TYPE;
        u8 color = tile & MASK_COLOR;
        char colorChar = '?';
        if (color == 0)
          colorChar = 'Y'; // Yellow
        else if (color == 1)
          colorChar = 'R'; // Red
        else if (color == 2)
          colorChar = 'B'; // Blue

        char symbol = '?';
        if (type == TILE_VIRUS_START)
          symbol = 'V';
        else if (type == TILE_LEFT)
          symbol = 'L';
        else if (type == TILE_RIGHT)
          symbol = 'R';
        else if (type == TILE_TOP)
          symbol = 'T';
        else if (type == TILE_BOTTOM)
          symbol = 'B';
        else if (type == TILE_SINGLE)
          symbol = 'S';

        std::cout << colorChar << symbol << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  return 0;
}
