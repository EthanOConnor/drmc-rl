To the software engineer or coding AI agent taking on the Dr. Mario RL project: Below is the guiding specification for this project, which aims to build a reinforcement learning environment and training setup for the NES game Dr. Mario, with a focus on speedrunning and advanced evaluation techniques.
-----
'''spec.md - Dr. Mario RL Environment & Training Design
Below is a current (Oct 16, 2025) survey of the ecosystem + an exact environment design and step-by-step stand-up plan (macOS first, Linux/Windows included), plus the full project spec (algorithms, evaluator heads, datasets, RNG control, multi-agent vs. mode, real-console I/O, and a reverse-engineering plan for Dr. Mario’s ROM).

⸻

1) What to build on in Oct 2025 (survey)

Emulator + RL env wrapper
	•	Stable-Retro (Farama) — actively maintained fork of Gym Retro; Gymnasium API; Libretro cores (Mesen/Nestopia, etc.); works on macOS/Linux/Windows. Mac needs Lua 5.1 instead of LuaJIT. Docs include build prereqs and import flow.  ￼
	•	Libretro cores: Mesen or Nestopia. Both are accurate and widely maintained. Good state support, rewind/trace in standalones; libretro variants are documented.  ￼

High-throughput RL training
	•	Sample Factory (SF) 2.x (PyTorch) — production-grade PPO (and derivatives), designed for extreme actor throughput on CPU-bound envs; easy multi-machine scale.  ￼
	•	EnvPool — C++ batched env engine; gym/dm_env compatible. Ideal for a bespoke Dr. Mario rules-exact simulator once you outgrow emulation FPS; published results show 1M+ FPS on Atari.  ￼

Multi-agent later (2-player battle)
	•	PettingZoo (Parallel API) — standard multi-agent interface that drops into TorchRL or RLlib later.  ￼

Alg/Tooling
	•	PyTorch + TorchRL — modern collectors, loss modules, distributional heads (QR-DQN/IQN style).  ￼
	•	Debug/instrumentation (to bootstrap state labels & RNG studies):
	•	FCEUX Code/Data Logger & Trace Logger — proven for 6502/NES RE and RAM mapping.  ￼
	•	Mesen (standalone) — excellent debugger/trace; multi-platform in 2025.  ￼

Dr. Mario-specific community facts (for seeding/RNG tests)
	•	ROM/mapper: Dr. Mario NES uses MMC1 (SEROM) boards. Useful when you map PRG/CHR for RE.  ￼
	•	Level layout & pill sequence are fixed once chosen — determined by the frame count since power-on when level is selected (TAS community observation; great for deterministic seeding experiments).  ￼
	•	Virus placement rules (no adjacent same-color second neighbors, etc.) — community docs exist; you’ll verify against disassembly.  ￼

⸻

2) Exact environment spec (single-agent speedrun mode)

2.1 Observation spaces

Two parallel obs modes (you can swap at will):

A) Pixel-only (for parity with real-console video)
	•	Input: 128×128 RGB resized from NES 256×240, frame-stack 4 (≈ 4/60 s temporal window).
	•	Normalization: uint8→float32 in [0,1].
	•	Optional: downscale to 96×128 or 120×160 if you prefer aspect-aware resizing.

B) “State tensor” (fast training & dense labels)
	•	16×8 board grid (top-visible rows; clamp to bottle height per level).
	•	Channels (one-hot):
	•	3 virus colors (R/Y/B),
	•	3 pill colors for fixed blocks,
	•	3 pill colors for falling pill halves,
	•	1 orientation plane for falling pill (0=vertical,1=horizontal),
	•	1 gravity/lock counter scalar broadcast,
	•	1 level/time scalar broadcast.
	•	Total channels: 12–16 (depending on design). Values in {0,1} except scalars normalized to [0,1].
	•	Keep a 4-frame stack here as well.

You’ll train on B for speed and use a supervised pixel→state net to bridge into A when moving to camera frames later.

2.2 Action space (discrete)

NES buttons mapped to 60 Hz input clock:

0 noop
1 left      (1 frame press)
2 right     (1)
3 down      (soft drop; held if repeated)
4 rotate_A  (clockwise; 1 frame)
5 rotate_B  (counterclockwise; 1 frame)
6 left_hold (held until release macro)
7 right_hold
8 down_hold
9 both_rot  (A+B on same frame; optional)

Default frame-skip = 1 (act every 1/60 s). For throughput you can try frame-skip=2, but keep 60 Hz available so “tight fits” remain learnable.

2.3 Episode/termination
	•	Episode = one level (your choice to treat levels independently).
	•	Terminate on: full clear; or timeout at T_max frames (pick 2–4× median human/agent clear time to avoid degenerate loops).
	•	Info: record viruses cleared, total frames, #chains, #drops, etc.

2.4 Rewards (speedrun baseline + dense shaping)

Let t be frames elapsed; V remaining viruses; ΔV viruses cleared this step.

r_t = -1                           # minimize time-to-clear
    + α * ΔV                       # shaping for progress (α≈5–10)
    + β * chain_bonus(ΔV_chain)    # tiny bonus for multi-clears to reflect skill
    - γ * settle_penalty           # small penalty for long settle cascades, optional

At terminal clear add +C (e.g., +500). For “risk-aware” variants, you’ll optimize quantiles or CVaR of time-to-clear (see §4).

2.5 Seeding / determinism
	•	Use savestates at “Level Select” with a fixed power-on frame counter to pin down virus layout + pill sequence. Keep a registry: seed_id → (state_file, frame_offset). (Community finding: layout & sequence are determined at selection time by frames since start. Validate via trace.)  ￼
	•	Expose RNG controls in env metadata so evaluation runs can sweep seed_ids and pill windows.

⸻

3) Stand-up: macOS (Apple Silicon), Linux, Windows

You’ll be training on GPU (Linux/Windows) and developing on macOS. Stable-Retro supports all; macOS needs Lua 5.1.

3.1 Common
	1.	Python 3.10/3.11, Poetry or uv, PyTorch 2.x (CUDA 12 on Linux/Win; MPS on macOS).
	2.	Stable-Retro + NES core:
	•	Install prereqs; on macOS: brew install cmake pkg-config lua@5.1 (LuaJIT is not used on macOS per docs).  ￼
	•	pip install stable-retro (or build from source per docs).  ￼
	•	Get a libretro NES core on your system (Mesen or Nestopia). You can install RetroArch to fetch cores, or build the core from source; cores live as *_libretro.{dylib,so,dll} and Stable-Retro can point to them. Docs for Mesen/Nestopia cores:  ￼
	3.	Import the game (you must use a legally-obtained ROM you own). Stable-Retro provides an import CLI akin to Gym Retro’s:
	•	python -m retro.import <path-to-roms> then create a basic game integration folder with .json and savestates. (Follow Stable-Retro dev/import pages.)  ￼
	4.	Quick smoke test
	•	Launch a simple Gymnasium loop with retro.make (or the Stable-Retro wrapper) and step frames; verify observations & actions.

3.2 Linux (training)
	•	Install CUDA toolkit 12.x, PyTorch w/ CUDA wheels.
	•	pip install sample-factory and run the provided PPO examples to confirm throughput.  ￼

3.3 Windows
	•	Works, but for largest runs you’ll likely prefer Linux for CUDA and clusterizing; Windows is fine for dev.

⸻

4) Algorithms & network heads (risk-aware, evaluation-driven)

4.1 RL backbone
	•	PPO (Sample Factory) with IMPALA-style CNN (3 conv blocks) + optional small LSTM (128–256) because action-effect timing matters. SF’s actor-learner architecture is ideal for thousands of CPU envs feeding a single GPU learner.  ￼

Baseline SF config sketch (key deltas):
	•	num_envs_per_worker=32–64, num_workers=8–16 (scale with cores)
	•	rollout=128, batch_size=rollout*num_envs*workers
	•	lr=3e-4, gamma=0.997, gae_lambda=0.95, entropy_coef=0.003
	•	frame_skip=1, grayscale=True (for pixel mode)
	•	Enable policy conditioning on a scalar risk_tau (see below).

4.2 Distributional time-to-clear evaluator (position model)

Goal: Given a position, predict a distribution over T_clear (frames). You’ll use it for:
	•	dense bootstrapping in RL (short rollouts + bootstrapped tail),
	•	strategy steering (mean-optimal vs low-variance vs “high-ceiling” play).

Architecture options
	•	QR head: K=51–101 quantiles with Quantile Huber loss (QR-DQN style).  ￼
	•	IQN head: sample quantile fractions per forward pass for better generalization.  ￼
	•	Implement with TorchRL distributional objectives/utilities.  ￼

Inputs
	•	Use state tensor (fast). Optionally concat action embedding for “after-action” scoring.

Targets
	•	From large playout corpora (see §6). Store the full empirical distribution per state (or large Monte Carlo sample) and fit quantiles/expectiles.

Risk-conditioned policy
	•	Add a scalar risk_tau ∈ (0,1] to the policy’s observation (broadcast).
	•	At action time, score actions by Q_tau(s,a) = E_{θ}[T_clear | τ] from the evaluator’s quantiles (or CVaR@τ).
	•	Swap τ to play conservative (higher-quantile minimization) vs spiky (minimize lower quantile) without retraining.

⸻

5) Evaluation harness (for your skill estimator)

For each seed = (layout, pill sequence):
	•	Run N = 100–1000 episodes with fixed seeds. Log T, success %, chains, inputs per second, and full T distribution.
	•	Compute E[T], Var[T], P(T ≤ t*), and CVaRα(T) for α ∈ {5%, 25%}.
	•	Compare policies at different risk_tau.
	•	Save parquet + plots; keep savestate replay traces to reproduce anomalies.

⸻

6) Data generation & curriculum

6.1 RAM-labeled corpus (for evaluator + pixel→state)
	•	Use Mesen or FCEUX with Lua/trace to extract: bottle grid (color IDs), falling pill halves, orientation, gravity/lock counters, timers, level. (Mesen/FCEUX docs + Code/Data Logger aid reverse-mapping quickly.)  ￼
	•	Generate datasets:
	•	Single-move solves (near-terminal) → short stacks → full level 0.
	•	For each position, roll out M random/heuristic continuations to get an empirical T distribution.
	•	Train evaluator on this corpus first; then plug into RL as a bootstrap value model.

6.2 Pixel→state translator
	•	Train a light UNet/ViT-tiny on emulator frames with RAM-derived labels, then later fine-tune on HDMI capture frames (domain gap).

⸻

7) Multi-agent (2P vs. battle)

Wrap your env as PettingZoo ParallelEnv:
	•	Two agents "player_0", "player_1", simultaneous moves each 60 Hz.
	•	Observations: each gets its own view; optionally include opponent summary (virus count, chain meter).
	•	Rewards: head-to-head (win/lose/time), or keep time-to-clear as primary with sparse win bonus.
	•	Works with TorchRL or RLlib self-play/league later.  ￼

⸻

8) Extreme scale plan (when/if you need 10–100× more FPS)
	•	Re-implement just Dr. Mario’s rules (grid update, gravity, lock delay, clear, chain settle, pill RNG) as a C++ env hosted in EnvPool.
	•	Maintain a golden parity suite: replay input traces from Stable-Retro and assert identical board hashes & clear times.
	•	Expect multi-million step/sec on modern servers if fully CPU-vectorized.  ￼

⸻

9) Real-console I/O bridge (later)

Video in: HDMI capture from Analogue Nt mini/FPGA clone, or composite→HDMI upscaler for original NES. Feed frames to pixel→state net.

Controller out: Microcontroller (e.g., RP2040/Arduino) emulating the NES gamepad shift-register protocol (latch/clock/data on $4016/$4017); many public references exist for timing/logic; NesDev’s controller docs are definitive.  ￼
(You’ll generate button bits synchronized to VBlank at 60 Hz. Tons of community examples emulate or read the 4021 protocol for NES pads.)  ￼

⸻

10) Reverse-engineering plan for Dr. Mario (ROM)

Purpose: document RNG, virus generation, pill sequence, timers, and RAM layout to (a) lock down deterministic seeds and (b) validate the bespoke EnvPool sim.

Legal note: Don’t distribute ROMs; use a cartridge or dump you legally own. You can make disassemblies/notes for research.

Mapper facts: Dr. Mario uses MMC1 (SEROM); 32 KB PRG / 32 KB CHR typical; no battery. Helps you set Ghidra mapping windows.  ￼

Toolchain
	•	FCEUX (Trace Logger + Code/Data Logger) to separate code vs data regions and catch calls at level select.  ￼
	•	Mesen debugger/trace + conditional logging (fast iterative RE).  ￼
	•	Ghidra with 6502 processor module / NES plugin to build a labeled disassembly; or export from FCEUX/CDL → ca65 project. (Several 6502 modules & guides exist.)  ￼
	•	NesDev wiki to cross-check controller and MMC1 behavior/timing.  ￼

Targets (what to find first)
	1.	Frame counter source of randomness at level select; confirm TAS claim that it fixes layout + pill sequence. Hook at level-select routine and watch reads of counters/PPU timing.  ￼
	2.	Virus placement routine — verify constraints (e.g., no 2nd neighbor same-color).  ￼
	3.	Pill sequence RNG — pattern often: LFSR/LCG or shift-xor loops (look for EOR/LSR/ROL sequences or multiply/add constants). (NES RNG design background.)  ￼
	4.	RAM map — bottle grid, falling pill, orientation, gravity/lock counters, timers. Community RAM maps exist; verify for your dump.  ￼
	5.	Input read — confirm $4016/$4017 read cadence and debouncing logic (helps when driving real hardware).  ￼

Repeatable workflow
	•	Play to level select, set breakpoint on writes to bottle RAM; log call tree.
	•	Enable CDL while doing a full playout of a few levels → export .cdl → use as mask for disassembler to label true code vs data.  ￼
	•	Build labeled disasm project (ca65 or Ghidra) with segments for fixed bank ($C000–$FFFF) and switchable bank ($8000–$BFFF), per MMC1 notes.  ￼
	•	Unit-test: capture RNG state at selection; regenerate a level layout with your reimplementation and assert byte-for-byte equivalence.

⸻

11) Repository layout (monorepo)

drmario-rl/
  envs/
    retro/                  # Stable-Retro wrappers, import helpers, seeds
    pettingzoo/             # ParallelEnv wrapper for 2P vs
    specs/                  # RAM offsets, RNG doc, unit tests
  models/
    policy/                 # PPO nets (CNN/LSTM), risk conditioning
    evaluator/              # QR/IQN heads, training scripts
    pixel2state/            # supervised translator
  training/
    sf_configs/             # Sample Factory configs
    launches/               # cluster/SLURM/Ray launchers
  data/
    states/                 # savestates seeds, fixtures
    datasets/               # parquet playouts, RAM-labeled frames
  eval/
    harness/                # E[T], Var[T], CVaR eval; seed sweeps
    plots/
  sim-envpool/              # (later) C++ Dr. Mario rules env for EnvPool
  re/                       # Disassembly projects (Ghidra/ca65), CDL files
  io-bridge/                # microcontroller scripts for controller-out
  docs/
    DESIGN.md
    RNG.md
    CONTRIBUTING.md


⸻

12) Concrete “Day-1” commands (macOS Apple Silicon first)

# System deps
brew install cmake pkg-config lua@5.1        # stable-retro Mac prerequisite
# Python env
uv venv && source .venv/bin/activate  # or: python -m venv venv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu  # macOS MPS wheels also available via pip
pip install stable-retro sample-factory gymnasium numpy opencv-python rich
# Get a libretro core (install RetroArch or fetch core .dylib)
# Place mesen_libretro.dylib or nestopia_libretro.dylib in a known path.
# Import game (from your legally-owned ROM dump)
python -m retro.import ~/ROMs/NES
# Quick sanity-run (python snippet you’ll have in envs/retro/demo.py)

(Stable-Retro and its macOS Lua 5.1 requirement documented here. Libretro core docs for Mesen/Nestopia here.)  ￼

⸻

13) Exact evaluator head spec (QR version)
	•	Network:
	•	Encoder: for state tensor → 3×(Conv 32/64/64, 3×3, stride2) → Flatten → MLP(256).
	•	Quantile head: linear to K × 1 outputs (e.g., K=101 quantiles at fixed τ_i).
	•	Loss: Quantile Huber, τ_i ∈ {0.01 … 0.99}.
	•	Targets: Monte-Carlo T_clear samples (truncate at T_max with censoring flag).
	•	Inference: compute mean (speedrun) or pick τ for risk-aware (e.g., τ=0.75 for safer play). TorchRL distributional modules can be used for loss scaffolding.  ￼

⸻

14) Risk-aware RL integration
	•	Add risk_tau scalar to observation.
	•	Action scoring: for each candidate action a, roll one-step lookahead (env sim) → evaluator predicts {Q_τ(s', ·)}; pick argmin expected quantile or CVaRα (mean of worst α-tail) computed from quantiles.
	•	For PPO, keep value head as expected time (mean) or add a distributional critic if you want true variance-sensitive advantages.

⸻

15) RNG & seed catalogs (for your distribution studies)
	•	Catalog seeds at level select across a grid of frame counts (e.g., every frame in [0…5 s] post-reset). Save (savestate, seed_meta) for each.
	•	For each seed, dump the first N pills for reproducibility and store the virus grid hash.
	•	TAS note backing this: layout & pill sequence determined by frame count at selection; constant thereafter. Validate by trace/disasm.  ￼

⸻

16) Vs. mode spec (PettingZoo)
	•	agents = ["player_0", "player_1"]
	•	observation_spaces per player (same as single-agent).
	•	action_spaces per player (same as single-agent).
	•	parallel_step(actions_dict) → env applies both; includes garbage/attack rules if present.
	•	Rewards: default zero-sum (win=+1/−1), plus optional shaped time signals. PettingZoo Parallel API docs here.  ￼

⸻

17) What the coding agent should do (task list)

Env & training
	1.	Implement Stable-Retro Dr. Mario env wrapper with:
	•	Pixel and state-tensor observation modes.
	•	Deterministic seed registry (savestates + frame offset).
	•	Action map (10 discrete actions) with 60 Hz stepping and optional hold macros.
	•	Reward function flags: {speedrun, mean_safe, cvar_alpha} and shaping params.
	2.	Wire to Sample Factory:
	•	Configs for pixel and state modes; enable LSTM.
	•	Add policy conditioning on risk_tau.
	•	Logging: wandb or TensorBoard; record T distributions per seed.
	3.	Build evaluator (QR head) and pixel→state translator with training scripts.
	4.	Evaluation harness:
	•	Seed sweeps; dump parquet; generate distribution plots (E[T], Var[T], CVaR).

Reverse-engineering
5) Set up FCEUX/Mesen projects:
	•	Use Code/Data Logger to produce .cdl, covering: boot→level select→several levels.  ￼
	•	Export trace logs around level select & pill generation; identify RNG calls.

	6.	Build Ghidra project (6502) for Dr. Mario:
	•	Memory map banks per MMC1/SEROM. Label fixed vs switchable banks.  ￼
	•	Identify and document: frame counter, virus placer, pill RNG, timers, RAM layout.
	•	Write unit tests that call your re-implemented RNG/placer to reproduce in-game layouts.

Fast sim (optional, after parity)
7) Implement C++ Dr. Mario core rules and host inside EnvPool with a Gym API; run parity tests vs emulator traces.  ￼

Real-hardware (later)
8) Prototype controller-out with RP2040/Arduino: emulate 4021 shift-register responses to console’s latch/clock (docs at NesDev).  ￼
9) Build a capture path; fine-tune pixel→state on HDMI frames.

⸻

18) Risks & mitigations
	•	macOS build quirks (Lua vs LuaJIT). Use Lua 5.1 per Stable-Retro docs.  ￼
	•	Throughput bottlenecks: Retro+Python stepping is CPU-bound. Use many actors (SF) first; later switch to EnvPool sim.  ￼
	•	RNG correctness: Don’t trust forum rules blindly; verify by disassembly and trace. (Community pointers help, but you need byte-exact parity.)  ￼
	•	ROM/legal: No redistribution; keep disasm private to the project.

⸻

19) References (practical)
	•	Stable-Retro docs & GitHub (install/mac notes, imports).  ￼
	•	Sample Factory site/GitHub (PPO at scale).  ￼
	•	EnvPool docs/paper (multi-million FPS envs).  ￼
	•	PettingZoo AEC/Parallel APIs.  ￼
	•	TorchRL distributional modules.  ￼
	•	FCEUX Code/Data Logger & Trace Logger.  ￼
	•	Mesen debugger + cross-platform emulator.  ￼
	•	Dr. Mario mapper & RAM map (for RE).  ￼
	•	Community notes on layout/pill sequence determinism.  ￼
	•	NES controller protocol (for hardware bridge).  ￼
'''
'''initial_user_request_prompt
I want to learn contemporary RL for game playing by implementing a self-play learning system for NES Dr Mario (speed running first, 2 player vs battle later) - so, I want to both learn about RL & other techniques in this space, and I want to produce measurements for my Dr Mario skill estimator project - learning what the variability for a given agent is on different starting layouts and different pill sequences to have sense of the distribution and manner of RNG effect on per-performance play; maybe even look at that for players of different skills by deliberately using either early snapshots or deliberately degraded models. Have been thinking about scoring / reward function. We can treat different levels even in the same run as totally independent - no visible game state transfers. Within a level, ultimately we are trying to minimize the time taken to clear all of the viruses. For competitive speed play, we probably want to minimize the expected time, but for a weak opponent we might want to stick to a lower variance, lower expected speed strategy that's still fast enough to beat them but less likely to have an unusually bad luck long level. Or for a strong opponent maybe optimize for bigger chance of exceptional performance at the expense of either speed or stability. For speed running vs the clock, again, the optimization might vary for level-by-level vs level-block play - more skewed toward exceptionally fast for level-by-level and only moderately skewed such for level-block, for example. Anyway, for the reward function we may want to be able to update with lower latency and higher frequency than just waiting to score based on the overall level time. So maybe we develop a position estimator network that tries to estimate some statistical properties of the expected time to full clear from a given position; it is trained on lots of playouts of various positions, and during play/play-learning rollouts from the current position are done for various possible actions and the position estimator is used to score the resulting positions so you don't have to play all the way to the end of the level. Training of both policy network and evaluation network will start from scenarios where only a single movement is required to score, and gradually build to full level 0 and so on, if that strategy. Would like this to be able to play off a video feed of a real monitor, driving real controllers for the future, so want to train on a time base at least as fast as the NES input clock and train based on visible game state and available controller actions. Also want this to be fast enough to run very large training runs. What's the "best" (from a total project speed and success standpoint) framework/infra for doing this in October 2025?
'''
'''initial_agent_response
Awesome project. Here’s the stack I’d pick today (Oct 2025) for fastest path to working self-play on NES Dr. Mario, with room to scale to huge runs, real-console I/O, and 2-player later.

TL;DR “Best” stack
	•	Emulator & env wrapper (now): Farama’s stable-retro (active fork of Gym Retro) + a Libretro NES core (Mesen/Nestopia). It gives you Gymnasium-style envs, savestates, memory access, and reproducible RNG—crucial for your pill/virus distribution experiments.  ￼
	•	High-throughput RL runner (now): Sample Factory 2.x (PyTorch) for PPO-style self-play at very high actor throughput on CPU-bound environments (Retro is CPU-bound). It’s battle-tested for millions of frames/hour, easy to scale across machines.  ￼
	•	Multi-agent API (for vs. mode): Wrap your env with PettingZoo (Parallel API) so 2-player Dr. Mario “just works” with multi-agent algorithms later, or plug into RLlib’s MA tools if you prefer Ray for cluster scale.  ￼
	•	Scripting/ground-truth state (dev helpers): BizHawk/Mesen + Lua during instrumentation phases to read/write game RAM (board grid, falling pill, timers), build your position dataset, and validate equivalence. BizHawk’s Lua hooks are well-documented and reliable.  ￼
	•	Algorithm/tooling (now): PyTorch 2.x + TorchRL (or Sample Factory’s built-ins) for PPO/IMPALA-style training; TorchRL slots in nicely for distributional heads and risk-sensitive objectives. Use Gymnasium wrappers for preprocessing.  ￼

“Phase 2” for extreme scale
	•	GPU-native simulator (optional but ideal): Re-implement just Dr. Mario’s rules (grid update, gravity, clears, spawn/rotation, lock delay) as a C++ env and host it in EnvPool. You’ll get 10–100× more FPS vs. a general NES emulator, with Gym/dm_env API compatibility and out-of-the-box vectorization. Validate bit-for-bit against Retro/Lua traces before switching training to it.  ￼
	•	Cluster scale: If you want turnkey autoscaling/HP search, run actors via Ray RLlib while keeping your custom env wrapped in PettingZoo/Gymnasium. RLlib’s multi-agent and distributed experience pipelines are mature.  ￼

Why this stack fits your goals
	1.	Self-play & speedrunning objective

	•	Actor–learner layout (Sample Factory) keeps thousands of Retro instances stepping in parallel on CPU while learners train on GPU—excellent for pixel games where env, not model, is the bottleneck.  ￼
	•	Use savestates to branch from the same starting layout + pill sequence seeds to empirically map performance distributions and RNG sensitivity. That’s a first-class feature in Retro/Libretro workflows.  ￼

	2.	Risk-sensitive policy control (variance vs. mean time)

	•	Implement a distributional time-to-clear head (e.g., quantile regression / IQN-style head) so you can tune for: minimum expected time (speedrun), lower-variance “safe” play (weak opponent), or heavier upper-tail mass (“spiky” exceptional runs vs strong opponents). TorchRL makes distributional/value-transform heads straightforward.  ￼
	•	At action time, condition the policy on a “risk parameter” (e.g., target CVaR quantile) to smoothly dial strategy from conservative to YOLO without retraining.

	3.	Learned position evaluator for dense feedback

	•	Train a position-to-(time-to-clear distribution) network from large playout corpora you generate by:
a) instrumenting emulator RAM via Lua to snapshot true board state + timers; b) running many rollouts per state; c) fitting a distributional regressor. (BizHawk/Mesen Lua gives you fast RAM scrapes; later you switch to your C++ env for playout generation.)  ￼
	•	In RL, use this model for n-step bootstrapping and lookahead scoring of candidate actions to avoid full level rollouts each step (huge speedup).

	4.	2-player vs. battle (later)

	•	Export the single-agent env as PettingZoo ParallelEnv with two controllers and shared RNG. Then you can train self-play (independent PPO or centralized critics) or evaluation vs frozen snapshots. PettingZoo keeps you portable across TorchRL/RLlib.  ￼

	5.	Real-console I/O (video in, controller out)

	•	Perception: Once your policy/evaluator are great on emulator states, train a pixel→state network with supervised labels from emulator RAM (board grid, pill type/orientation, fall height, level timer). Then swap emulator frames for HDMI capture frames from a real NES/Analogue/FPGA clone.
	•	Control: Drive real controllers using a microcontroller (e.g., RP2040) as a USB-HID→NES bridge; the RL actor emits button states at 60 Hz locked to VBlank. (Your algorithm stack is agnostic here; it’s an I/O layer.)
	•	You can still evaluate with fixed seeds by recording HDMI + controller stream and comparing to emulator outputs for drift.

Concrete build plan (you can start today)

Week 0–1: Environment & datasets
	•	Spin up stable-retro with a Mesen/Nestopia core for Dr. Mario. Define scenario states, action set, and deterministic seeds for: virus layout + 128-pill sequence chunks (you already have details on the LFSR; wire that into seeding). Build Gymnasium env with savestates + controlled RNG.  ￼
	•	Use BizHawk Lua to dump RAM structs for: 8×16 grid, pill halves, orientation, gravity/lock counters, score/time; verify offsets across levels. Create a position corpus (state → final clear time) via scripted playouts.  ￼

Week 1–2: Baselines & evaluator
	•	Train PPO baseline in Sample Factory from pixels (84×84 grayscale or 128×128 RGB) with simple shaped reward: −1 per frame, +C on clear, small shaping for reducing remaining viruses. Use n-step returns.  ￼
	•	In parallel, train the position evaluator (distributional head predicting time-to-clear quantiles). Start curriculum: single-move solves → short stacks → full level 0. Plug this into the actor as a bootstrap/value-improvement oracle to cut rollout length.

Week 2–3: Risk-aware policy & metrics
	•	Add risk parameter to the policy input; implement action selection under different quantile targets (e.g., mean, CVaR@25%, CVaR@5%).
	•	Build your evaluation harness: for each fixed (layout, pill-sequence) seed set, run 100–1000 trials to estimate E[T], Var[T], upper-tail probabilities, and collect full distributions to feed your skill estimator project.

Week 3+: Scale and/or speed up
	•	If environment FPS is the bottleneck, clone rules into a C++ env in EnvPool for a 10–100× throughput jump; keep your Python API identical (Gymnasium / PettingZoo). Validate against Retro by replaying identical action/seed traces.  ￼
	•	For cluster scale or 2-player self-play leagues, adopt RLlib atop your PettingZoo wrapper (or stay in Sample Factory and roll your own league scheduler).  ￼

Practical choices & trade-offs
	•	PyTorch/TorchRL vs JAX/Acme/RLax: For Retro/Libretro workloads the env is CPU-bound, not learner-bound, so PyTorch + Sample Factory usually wins on time-to-results and tooling. If you later move to a GPU-native Dr. Mario (EnvPool + custom CUDA kernels), JAX can shine with massive batching, but it’s extra engineering.  ￼
	•	Retro vs bespoke sim: Retro is accurate and fast to prototype; bespoke C++ sim is how you reach very large runs (100M+ frames/day) and fine-grained RNG control. Start Retro; switch once you have parity tests.  ￼
	•	Multi-agent interfaces: Standardize on PettingZoo so you’re not locked into any one RL library for vs. mode.  ￼

References you’ll actually use
	•	stable-retro (active Gym Retro fork).  ￼
	•	Gym Retro (archival context).  ￼
	•	Sample Factory (throughput-oriented RL).  ￼
	•	EnvPool (C++ high-FPS env engine) + paper & docs.  ￼
	•	PettingZoo (multi-agent API), AEC/Parallel docs.  ￼
	•	TorchRL (distributional/value transforms, collectors).  ￼
	•	BizHawk Lua (RAM hooks, TAS-style instrumentation).  ￼

⸻

If you want, I can sketch the exact env spec (action space, observation channels, RAM offsets), the evaluator’s distributional head, and a minimal Sample Factory config that will train from pixels and log the E[T]/Var[T]/CVaR curves you care about.
'''
'''user_followup_request_prompt
Great, sounds awesome. Can you fully survey community knowledge / plans in all the dependencies and spec out the exact environment and how to stand it up? cross-platform to macOS is preferred, but windows or linux required is fine if the project will be limited otherwise. Then sketch out the full spec for the project as you mentioned, including plans for a tool using coding agent to disassemble/decompile the NES Dr Mario code from .ROM and other community efforts, the details on the heads - everything needed to hand off to a team of me and a coding (openai codex w/ GPT 5 & network & github access) agent.
'''
'''detailed_response
(see spec.md above)
'''

⸻

Discussion & Decisions (live log)

- Observation (state tensor channels):
  - Final: 14 channels over 16×8. [viruses R/Y/B=3, fixed pill R/Y/B=3, falling halves R/Y/B=3, orientation=1, gravity_scalar=1, level_scalar=1, time_since_spawn=1, settle_flag=1]. 4‑frame stack retained.
- Action macros (hold behavior):
  - Final: 10 discrete actions at 60 Hz. Hold actions latch until opposite tap, a new hold, or lock. Repeat at 60 Hz (no DAS emulation initially).
- Rewards:
  - Final: r_t = −1 + 8·ΔV + 0.5·chain_bonus − 0.1·settle_penalty; terminal +500.
- T_max:
  - Final: Global cap T_max=8000. Levels: L0:4000; L5:6000; L10:7000; L15+:8000.
- Risk conditioning:
  - Final: Train with τ ~ U(0.25, 0.75); evaluate at {0.25, 0.5, 0.75}. Decision rule: minimize τ‑quantile T_clear. CVaR α ∈ {5%, 25%} optional.
- Seed catalog:
  - Final: per‑level subdirs, 120 seeds per level (2s at 60 Hz), dump first 128 pills, store virus grid hash. Registry schema locked.
- Training defaults (SF):
  - Final: Single GPU baseline: workers=8, envs/worker=32 (≈256 actors), rollout=128, batch≈32K, LSTM=128, frame_skip=1, pixel 128×128 RGB, state C=14.
- Reverse‑engineering scope first pass:
  - Final: Confirm frame counter read sites; identify virus placer; log pill RNG; map bottle grid RAM; defer sound/input debounce.

Action Items
- [ ] Approve observation channel schema and encoding details
- [ ] Approve action macro semantics and frame‑skip policy
- [ ] Confirm reward coefficients and terminal bonus
- [ ] Select T_max policy per level
- [ ] Approve risk_tau interface and defaults
- [ ] Define seed catalog size and naming convention
- [ ] Confirm SF baseline scale for target hardware
- [ ] Prioritize RE tasks for Week 0–1
