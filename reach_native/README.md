Dr. Mario Falling-Pill Reachability

Summary
- 512-node BFS over (x=0..7, y=0..15, o=0..3)
- 1 KB core arrays (arrival + parent), cache-resident
- Fit tested via 4 planes derived from column-major occupancy in ~10 ops
- Two stepping modes: Planner (1 prim + gravity) and NES (Y -> X -> Rot)

Planner vs NES
- NES applies Y, then X, then Rotate each frame (matches disasm)
- NES rotation collision is parity-based (H vs V): fit[0]==fit[2], fit[1]==fit[3]
- NES includes extra-left on horizontal rotations and a generic left wall-kick
- Planner models four distinct planes and applies a single primitive per frame

Y Coordinate Note
- NES decrements Y to move down; planner coordinates increment Y downward.

Intended Usage
- Build fit planes once per frame/spawn
- Run BFS to fill arrival/parent arrays
- Project to 464-way placement action mask for policy masking and costs
- Reconstruct chosen placement by walking parent codes backward

