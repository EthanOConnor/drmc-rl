Seed registry for deterministic evaluation.

Format (registry.json):

{
  "level_00_seed_0001": {
    "savestate": "level_select/seed_0001.state",
    "frame_offset": 123  
  },
  "...": { }
}

- `savestate` is created at Level Select with a fixed power-on frame counter.
- `frame_offset` is the number of frames from savestate load to selection.
- Per-level subdirs recommended; v1 catalog = 120 seeds per level (2s at 60 Hz).
- For each seed, dump first 128 pills and store a virus grid hash.

Use this registry to reproduce virus placement and pill sequences.
