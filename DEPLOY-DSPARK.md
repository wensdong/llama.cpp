# DSpark-style confidence-scheduled speculative decoding — deploy notes

Personal branch for the qbox llama-server (Qwen3.6-27B-MTP). Based on upstream
commit `588f0dc` (the exact commit the host binary at
`~/github/llama.cpp/build/bin/llama-server` was built from).

**Do not open an upstream PR from this branch** — llama.cpp does not accept
AI-generated contributions (see AGENTS.md; private forks are exempt).

## What it does

`--spec-sched` replaces the static `--spec-draft-p-min` stop rule of the
`draft-mtp` drafter with a confidence-scheduled draft length
(DSpark, DeepSeek-AI 2026, sec 3.2 / Algorithm 1 at batch-of-1):

- per-position acceptance calibration, learned online from verification
  outcomes (no training, no validation set needed);
- an online-profiled cost model (draft-step time, cycle time per verify size);
- draft length chosen to maximize expected generated tokens per second,
  causally (the choice at draft position k never depends on later tokens),
  so the target distribution is untouched.

For the first ~1 minute of traffic (warmup) it behaves exactly like the
static p_min rule while it gathers statistics.

## Deploy on the host

```bash
cd ~/github/llama.cpp
git remote add dspark https://github.com/wensdong/llama.cpp   # once
git fetch dspark
git checkout -b dspark-sched dspark/dspark-sched
cmake --build build -j        # same build dir/flags as before
```

Restart llama-server with the previous arguments plus:

```
--spec-sched --spec-draft-n-max 12
```

(`--spec-draft-n-max 12` raises the hard cap so the scheduler has room; it
will pick the profitable depth per step on its own. Keep whatever
`--spec-draft-p-min` was in use — it still governs the warmup phase.)

If the fetch route is inconvenient, the same change is in `patches/`
(`git am patches/*.patch` on any tree near 588f0dc).

Note: the branch bundles the prebuilt web-UI assets (`tools/ui/dist/`,
HF bucket version b9541, checksum-verified). This sidesteps a failure mode of
shallow clones: the UI provisioning derives the asset version from
`git rev-list --count`, which is wrong in a shallow tree (b51), so the
download fails and the build dies on an empty generated `ui.cpp`.

## Verify

- Logs: on shutdown (or SIGINT) `common_speculative_print_stats` now prints a
  per-position calibration table (`p_pred` vs `p_emp`), the measured cost
  curve `t_rest(n)`, and a histogram of chosen draft lengths.
- Speed: `timings` in `/completion` responses (`predicted_per_second`,
  `draft_n`, `draft_n_accepted`), or the benchmark script used for the
  baseline (claudebox: `scratchpad/bench_spec.py`).

Baseline before this change (2026-07-02, Q3_K_XL, single slot):

| domain | tok/s | draft acceptance | tokens/cycle |
|--------|-------|------------------|--------------|
| code   | 84.6  | 82%              | 2.62         |
| math   | 89.6  | 90%              | 2.78         |
| chat   | 77.8  | 72%              | 2.41         |

## Rollback

Restart without `--spec-sched` (default off; the code path is then identical
to stock 588f0dc), or check out the previous commit and rebuild.
