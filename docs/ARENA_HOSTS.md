# Arena hosts: distributed trainer studies

`tools.trainer_arena_distributed` splits one `trainer-planning-arena` study
config across processes and machines without changing its answer. One
coordinator owns the output directory; workers lease seed-pair batches over
authenticated HTTP, fetch checkpoints by SHA-256, and upload rows plus move
journals. The coordinator writes `games.jsonl`, `moves/`, `arena.sqlite` and
`results.json` exactly as `tools.trainer_planning_arena` would.

## What is exact

- **Same batches.** Leases are the single-host batches: `pairs`, sequential
  `look_games` and identity-probe sizes are unchanged. A worker plays a batch
  with the same `ArenaRuntime.play` call the single-host arena makes.
- **Same order.** Batches are accepted per comparison in order; sequential
  stopping and identity probes are evaluated after each accepted batch, and
  batches played speculatively past a decision are discarded. The journal is
  written in schedule order. Move traces are gzip'd with a fixed header, so
  replayed games are byte-identical files.
- **Frame-runner statistics** now stop at each game's terminal frame, so a
  game's row no longer depends on how long other games in its batch run.
  Earlier journals can differ by one `locks`/`early_requests` count in the
  last-finishing game of a batch; moves and outcomes are unaffected.

Measured on the Mac (MPS, strict FP32, frozen `19f292c` natives): a single
process and `local --workers 2` produced byte-identical journals and traces for
a frame-runner `lock_safe` study (frame_perfect + sloth, 16 games, pairs 4) and
an asynchronous event-runner spawn study (frame_perfect + normal, 32 games,
pairs 8). CPU and MPS produced identical move journals on 8 lock_safe
frame_perfect games.

What is not guaranteed: FP32 network outputs are not bitwise identical across
devices, kernel choices or batch shapes (the preview-branch measurement found
logit differences up to 1.6e-5 from a shape change alone), and the event
runner's asynchronous batches are timing-dependent even on one host. Games stay
identical as long as no decision is that close to a tie. Every batch holds whole
seed pairs, so both side-swapped games of a seed always run on the same worker
and hardware effects apply to both entrants alike. Other numerics classes
(device, CPU model, torch build) are validated rather than assumed:

- `--calibration-games N` makes each new numerics class replay N
  already-journaled games before it may contribute. `--trust PREFIX` admits
  the reference class (e.g. `mps/`) without replay.
- `--replicate-every K` (default 16, about 6% of batches) replays sampled
  batches on a second worker, preferring another numerics class.
- `--fidelity tolerant` (default, exploratory studies) admits a class whose
  decisions agree with the reference at `--min-agreement` (0.99) of compared
  decisions, counting each game up to and including its first divergence;
  divergent games are expected and fine. `--fidelity strict` (confirmation
  runs) requires byte-identical games. A failing calibration or replica
  rejects the untrusted class. Every comparison, with per-class cumulative
  agreement and divergent-game rates, is appended to
  `distributed/audit.jsonl` and shown in `/api/v1/study/status`.
- `tools.arena_host_selftest` checks the host's native engine and planner
  builds against the Mac reference without torch or checkpoints.

Workers must run the coordinator's exact source revision (commit plus a digest
of uncommitted `drmc_rl`/`tools`/`reach_native` changes). Native libraries are
per host and must be built from the study's `native_commit`.

## Coordinator (Mac)

```bash
mkdir -p ~/.config/drmc-rl
python3 -c 'import secrets; print(secrets.token_hex(24))' > ~/.config/drmc-rl/study-worker.token
chmod 600 ~/.config/drmc-rl/study-worker.token
export DRMARIO_REACH_LIB=$FROZEN/libdrm_reach_full.dylib   # the study's frozen natives
uv run python -m tools.trainer_arena_distributed serve --config STUDY.json \
  --host 0.0.0.0 --port 8099 --calibration-games 8 --trust mps/   # add --fidelity strict for confirmation
```

Bind `0.0.0.0` (or the LAN address, e.g. 192.168.157.114) only on a trusted
LAN; every endpoint needs the bearer token but traffic is plain HTTP. macOS
asks once to allow incoming connections for Python; with the application
firewall in stealth mode, allow it explicitly:
`sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add $(uv run python -c 'import sys;print(sys.executable)')`.
Check from another host with
`curl -H "Authorization: Bearer $TOKEN" http://192.168.157.114:8099/api/v1/study/status`.
The status shows batches, live leases, per-worker identity and calibration.
A worker's first SIGTERM/SIGINT abandons its batch within about five seconds and
releases the lease so another worker takes it at once; a second signal exits
immediately. For a killed or hung worker, release its batch by hand:
`curl -X POST -H "Authorization: Bearer $TOKEN" -d '{"batch":"BATCH_KEY"}' http://HOST:8099/api/v1/study/release`
(batch keys are listed under `leases` in the status).
Restarting the coordinator is safe: finished batches are spooled under
`distributed/spool/` and workers retry for up to 15 minutes.

mombox can run the coordinator (it is I/O and bookkeeping only), but it is a
4-core production web server and must not run workers.

## Mac workers

MPS is the Mac's network device; one process leaves the M3 Max GPU idle
between its serial frame-loop steps, so several worker processes scale well.
Measured with `local`, frame_perfect, both sides the mixed-v2 core, under load
average 63–122 from unrelated jobs (`runs/review-20260909/arena-hosts-mac-benchmark.json`):

| study | 1 process | 3 workers | 4 workers | 6 workers |
|---|---|---|---|---|
| frames, lock_safe vs spawn, pairs 16 | 344 games/h | 729 | 800 | 971 |
| events, spawn vs spawn, pairs 32, async | 1,557 | 2,695 | | |

One CPU-only process at one thread was 9x slower than MPS (39 vs 356 games/h;
conv2d was 90% of its time), so the Mac should not add CPU workers while other
jobs occupy its cores. Use 4–6 MPS workers when the machine is otherwise idle
and 3 when it is shared.

```bash
uv run python -m tools.trainer_arena_distributed local --config STUDY.json \
  --workers 3 --device mps --threads 1 --native-library $FROZEN/libdrmario_pool.dylib \
  --reach-library $FROZEN/libdrm_reach_full.dylib
```

`local` runs the coordinator on loopback and N workers of this checkout that
read checkpoints in place (`--shared-artifacts`). To combine Mac and remote
workers, run `serve` on the LAN instead and start Mac workers with `worker`.

## Windows 11 Xeon ("green") via WSL2

WSL2 is the supported route. The native libraries are C/C++ built by
`tools.build_reach_native` (clang) and `make -C vendor/drmario_native
libdrmario_pool` (g++ plus clang for the C planner object), and every tool and
service here is POSIX shell. A native Windows build is feasible — both build
scripts have a clang/LLVM `win32` branch exercised by CI on `windows-latest` —
but it gives no speed advantage and would need PowerShell equivalents of the
bootstrap and supervisor. Use it only if WSL2 is unavailable.

Green: Xeon E5-2680 v4 (Broadwell, 14 cores/28 threads, AVX2, no AVX-512),
GTX 1080 Ti (Pascal, sm_61, 11 GB), Ubuntu 22.04 under WSL2 with NAT
networking. The coordinator is on the same LAN, so workers only connect
outbound; no inbound port is needed on green.

### One-time Windows setup (PowerShell as administrator)

```powershell
wsl --install -d Ubuntu-22.04        # already present on green
wsl --update
# Optional: let WSL use more of the 100 GB and keep the VM up while workers run.
@"
[wsl2]
memory=80GB
processors=28
vmIdleTimeout=-1
"@ | Set-Content "$env:USERPROFILE\.wslconfig"
wsl --shutdown
```

The NVIDIA Windows driver provides CUDA inside WSL2; do not install a Linux
driver in the distro.

### Bootstrap (inside Ubuntu)

`drmc-rl` clones anonymously; the `vendor/drmario_native` submodule is private.
Authenticate once (`gh auth login` then `gh auth setup-git`), or pass
`--native-bundle URL|PATH` to a `git bundle` containing the needed commit
(it must contain the study's `native_commit`, e.g. `19f292c`, not only the
pinned `afaf62a`).

```bash
curl -fsSL https://raw.githubusercontent.com/EthanOConnor/drmc-rl/trainer/arena-distributed/tools/arena_host_bootstrap.sh -o bootstrap.sh
bash bootstrap.sh --coordinator http://192.168.157.114:8099 --token "$(cat token.txt)" --gpu auto
~/.config/drmc-rl/start-workers.sh
```

The script installs `git build-essential clang curl`, installs uv, clones
`trainer/arena-distributed`, checks out the study's `native_commit` (read from
the coordinator), runs `uv sync --locked --extra inference` (Python 3.14.7 is
uv-managed; the system 3.10 is not used), builds both native libraries, runs
`tools.arena_host_selftest` and the distributed-study tests, and writes
`~/.config/drmc-rl/study-worker.env` plus `start-workers.sh`.

### Pascal GPU

The locked `rl` extra resolves `torch 2.14.0+cu130`; CUDA 13 wheels list
`sm_75`–`sm_120` only, so the 1080 Ti fails with "no kernel image". The
inference extra used by workers is CPU torch on Linux. With `--gpu auto` the
bootstrap installs the same torch version from the `cu126` index into this
venv only, requires `sm_61` in `torch.cuda.get_arch_list()` and a small matmul
check, and otherwise restores CPU torch. After that, always use
`uv run --no-sync` (a plain `uv run` would resync to the locked CPU wheel).
Pascal has no TF32, so `strict_fp32` changes nothing there, but cuDNN kernels
still differ from MPS and x86 CPU; the coordinator's calibration replay and
replicate audits are the fidelity evidence for `cuda/…GTX_1080_Ti…` results.

### Choosing processes and threads

The network dominates CPU runs (conv2d was 90% of a one-thread CPU profile);
the frame runner's Python loop dominates GPU runs. Defaults on green: three
CUDA worker processes sharing the GPU (about 1–1.5 GB each) and
`(14 − 2×3)/2 = 4` CPU workers at two threads each. Keep processes × threads at
or below the 14 physical cores; hyperthreads add little to FP32 convolution.
oneDNN selects AVX2 kernels on Broadwell automatically; if CPU workers on
different x86 machines must agree bitwise, pin `ONEDNN_MAX_CPU_ISA=AVX2` on all
of them. To measure, start one configuration, let each worker finish two
batches, and read `games_per_hour` from its log or
`/api/v1/study/status`; then change `DRMC_GPU_WORKERS`, `DRMC_CPU_WORKERS` and
`DRMC_THREADS` in `study-worker.env` and restart `start-workers.sh`. A quick
standalone check is `worker --max-batches 2`.

### Run at startup

With `vmIdleTimeout=-1`, a scheduled task keeps the workers alive in a WSL
session that holds the VM open:

```powershell
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu-22.04 -- bash -lc ~/.config/drmc-rl/start-workers.sh"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "drmc-study-workers" -Action $action -Trigger $trigger -RunLevel Highest
```

Alternatively enable systemd (`/etc/wsl.conf`: `[boot]` `systemd=true`) and
run `start-workers.sh` from a user service with `loginctl enable-linger`.

### Remote shell into WSL2

Simplest: Windows OpenSSH server, then `ssh user@green wsl -d Ubuntu-22.04`.

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd; Set-Service sshd -StartupType Automatic
```

Or sshd inside the distro (`sudo apt install openssh-server`, `Port 2222`) with
mirrored networking (`networkingMode=mirrored` in `.wslconfig`, then
`New-NetFirewallHyperVRule -Name wsl-ssh -DisplayName "WSL SSH" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 2222`),
or with NAT a port proxy to the current WSL address:
`netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=2222 connectaddress=$(wsl hostname -I)`
plus a matching inbound firewall rule. The NAT address changes on restart.

## tf3090 while training

A CPU-only worker on tf3090 is possible when the GPU is busy: same bootstrap
with `--gpu cpu` and `--cpu-workers` sized to cores the trainer leaves idle
(the trainer's planner workers and data path also use CPU). Do not run a CUDA
worker there during training.

## Other networks

Off the LAN, use Tailscale (coordinator `serve --host <tailscale-ip>`) or an
SSH tunnel from the worker, `ssh -N -L 8099:127.0.0.1:8099 mac`, with
`--coordinator http://127.0.0.1:8099`.
