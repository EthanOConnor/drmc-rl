* EOC Notes during DRMC_rl development *

19 Oct 2025

Two thoughts for drmc_rl - add another layer of intent, implementing "specify row, column, and orientation" for pill placement; translator maintains internal game state and model of board evolution and will emit properly timed controls to the emulator to achieve the desired placement with closed loop observation of state and response; this should dramatically tighten the learning curve for RL agents, as they can focus on high-level strategy rather than low-level control timing. The action space is an outcome space rather than a path space and is orders of magnitude smaller. We could even have the translator emit a legal placements (or even a possible placements based on translator internal game model) mask to further constrain the action space.

THEN, we suddenly have a way to dramatically reduce inference compute demand. We no longer have to update the policy input/output for every emulator step. The translator simply notifies the system when a new decision is needed (i.e. at pill spawn, or if the original game model was wrong). This could reduce the number of inferences by a factor of 10-20x, depending on how fast the pills are falling. This opens the door to running on much more constrained hardware, or running multiple agents in parallel on the same hardware.

In fact, the change to asynchronous (or temporally sparser) decision making can be applied independently of the placement intent layer. Even if our intents are directions instead of placements, there's now a window where the translator can handle the low-level control until the next decision point. This could still yield significant compute savings, though not as dramatic as with placement intents. Depending on the game dynamics (gravity, etc), this could still be a 2-5x reduction in inference frequency.

*** Roadmap from here ***
Even more than a fast / strong Dr Mario speedrunner / speed bracketer / eventual vs bracketer, I'm interested in doing 
