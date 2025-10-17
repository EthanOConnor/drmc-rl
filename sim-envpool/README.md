High-FPS rules-exact Dr. Mario simulator (EnvPool host) — placeholder.

Plan:
- Implement grid update, gravity, lock delay, clears, settle, RNG.
- Maintain golden parity: replay action/seed traces from Stable-Retro and assert identical board hashes and clear times.
- Expect 10–100× FPS over emulator once vectorized.
