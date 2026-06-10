# Benchmark Notes - 2026-05-08

Benchmarks were run on the local macOS `.venv` after reinstalling this checkout
with `.[dev,rl,viz]`. The environment was on:

- NumPy `2.4.4`
- `opencv-python` `4.13.0.92`
- Torch `2.9.0`
- pytest `9.0.3`

`python -m pip check` reported no broken requirements.

## Policy Network Cost

Command:

```bash
python -m tools.bench_policy --source cpp-pool --batch-size 16 --repeats 10 \
  --warmup 2 --candidate-max 128 --patch-kernel 9 \
  --json-out runs/benchmarks/policy_cpp_pool_conn_b16.json
```

Results on real `cpp-pool` decision observations using the current
`bitplane_bottle_conn_mask` default:

| Policy | Params | Forward ms | p95 ms | Decisions/sec | Feasible n | Packed n | Truncation |
|---|---:|---:|---:|---:|---:|---:|---:|
| `heatmap_dense_b0` | 156,837 | 7.306 | 7.866 | 2,190 | n/a | n/a | n/a |
| `heatmap_dense_b32` | 2,520,229 | 140.533 | 142.909 | 114 | n/a | n/a | n/a |
| `candidate_cnn` | 363,233 | 5.450 | 6.316 | 2,936 | 41.3 | 41.3 | 0% |
| `candidate_col_tx` | 1,162,465 | 6.933 | 7.965 | 2,308 | 41.3 | 41.3 | 0% |

Interpretation:

- The old 32-block heatmap default is not defensible as a forward default.
- Dense heatmap with zero extra blocks is fastest, but it spends model capacity
  producing a full 512-action map and relies on masking after the fact.
- `candidate_cnn` is the best default tradeoff: it is only modestly slower than
  the tiny heatmap, directly models planner-feasible choices, uses
  cost-to-lock as input, and had no truncation at `Kmax=128` on this sample.
- `candidate_col_tx` is viable but slower and larger; keep it as an ablation.

## Native Pool Throughput

Command:

```bash
python -m tools.bench_multienv --backend cpp-pool --vectorization sync \
  --num-envs 1,2,4,8,16,32 --duration-sec 0.5 --warmup-steps 5 \
  --repeats 3 --action-mode first \
  --json-out runs/benchmarks/multienv_cpp_pool_sync.json
```

| Envs | FPS | Decisions/sec | Avg tau | Env step ms | p95 ms | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,543 | 102.5 | 44.3 | 9.745 | 12.819 | 1.00 |
| 2 | 7,986 | 174.2 | 46.0 | 11.470 | 13.615 | 0.88 |
| 4 | 13,575 | 283.5 | 48.2 | 14.131 | 16.553 | 0.75 |
| 8 | 20,779 | 435.7 | 47.8 | 18.301 | 22.253 | 0.57 |
| 16 | 26,889 | 582.9 | 46.1 | 27.372 | 31.720 | 0.37 |
| 32 | 30,189 | 660.9 | 45.8 | 48.461 | 55.398 | 0.21 |

Interpretation:

- The native pool scales well enough to keep `num_envs=16` as the default.
- `num_envs=32` gives only about 13% more env-only decisions/sec than 16 in this
  run, while nearly doubling batch latency.
- For PPO collection, 16 envs is the better default unless update overhead or
  accelerator batching changes the balance.

## Worker Count Sweep

Commands used `DRMARIO_POOL_WORKERS={1,4,8,16}` with env counts 8/16/32.

At `num_envs=16`:

| Workers | FPS | Decisions/sec | Env step ms |
|---:|---:|---:|---:|
| 1 | 5,028 | 109.8 | 145.787 |
| 4 | 15,570 | 354.9 | 45.001 |
| 8 | 24,512 | 544.4 | 29.324 |
| 16 | 27,544 | 620.1 | 25.718 |

Interpretation:

- Parallel native planning is essential.
- Auto worker selection is acceptable on this machine, but explicit
  `DRMARIO_POOL_WORKERS=16` is a good troubleshooting/tuning knob.

## Harness Overhead

Sequential 16-env checks:

- First-feasible action mode: ~29.4k FPS, ~645 decisions/sec, ~0.3% harness
  overhead.
- Random-feasible action mode: ~40.7k FPS, ~680 decisions/sec, ~0.7% harness
  overhead, but average tau differs, so FPS is not directly comparable.
- Emitting board/raw info at 16 envs did not materially change throughput in
  this short run.

## Decision

The default training config now uses candidate scoring with the CNN board
encoder and `Kmax=128`. A separate heatmap baseline config exists at
`training/configs/smdp_ppo_heatmap.yaml`.

Recommended starting run:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml \
  --ui tui --backend cpp-pool --num_envs 16 --wandb
```

Use `tools.bench_multienv` for simulator scaling and `tools.bench_policy` for
network cost before changing env counts, worker counts, or policy shape.
