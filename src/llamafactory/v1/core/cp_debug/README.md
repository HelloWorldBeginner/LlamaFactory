# CP Precision Debug (CP1 vs CP2)

Vendored from the `cp-precision-debug` skill. Locates the first layer where CP1
(`cp_size=1`, CP off) and CP2 (`cp_size>1`, CP on) numerically diverge.

Workflow: **hook records per-layer in/out/weight/grad → run CP1 and CP2 once
each → offline compare → first FAIL layer is the lesion.**

## Integration points (already wired in this branch)

- `src/llamafactory/v1/core/cp_debug/` — the tool (`hooks.py`, `utils.py`, `compare.py`).
- `BaseTrainer.__init__` — calls `_register_cp_debug_hooks(model)` after the CP
  plugin is applied. No-op unless `CP_DEBUG=1`.
- `BaseTrainer.fit` — wraps `loss.backward()` with `set_in_backward(True/False)`
  and calls `collect_param_gradients()` after backward (reads `param.grad`,
  NPU-safe, no `full_backward_hook`).
- `pad_and_truncate` — when `CP_DEBUG=1`, pads every sample to `cutoff_len` so
  CP1 (full seq) and CP2 (all-gathered seq) shapes align. Disable with
  `CP_DEBUG_PAD_TO_CUTOFF=0` (then you must keep seq len uniform yourself).

## Requirements for a clean comparison

- `cutoff_len % cp_size == 0` (CP must evenly split the sequence). The example
  config uses `cutoff_len: 4096`, `cp_size: 2` — OK.
- Same model, same data, same seed for both runs. Only `cp_size` (and the
  process count) differs.
- `flash_attn: flash_attention_2` (SP already requires this).

## 1. Run CP1 (CP off)

Use a config with `cp_size: 1` (or run the same config on fewer ranks so the CP
dim is size 1). `CP_DEBUG=1` enables hooks; `CP_DEBUG_DUMP_DIR` sets the output.

```bash
CP_DEBUG=1 \
CP_DEBUG_DUMP_DIR=./cp1_dumps \
CP_DEBUG_MAX_STEPS=1 \
USE_V1=1 \
torchrun --nproc_per_node=4 -m llamafactory.cli train examples/v1/train_full/train_full_ulysses_cp.yaml
# (with cp_size: 1 in the yaml for the CP1 run)
```

## 2. Run CP2 (CP on)

Same config but `cp_size: 2` (and the matching process count):

```bash
CP_DEBUG=1 \
CP_DEBUG_DUMP_DIR=./cp2_dumps \
CP_DEBUG_MAX_STEPS=1 \
USE_V1=1 \
torchrun --nproc_per_node=8 -m llamafactory.cli train examples/v1/train_full/train_full_ulysses_cp.yaml
# (with cp_size: 2 in the yaml for the CP2 run)
```

Only rank 0 writes dumps. `CP_DEBUG_MAX_STEPS=1` records just step 0 (first
micro-batch forward + its gradients) — that is the aligned unit to compare.

## 3. Compare

```bash
python -m llamafactory.v1.core.cp_debug.compare ./cp1_dumps ./cp2_dumps \
    --step 0 --diff-only --gradients
```

Useful flags: `--threshold 1e-5` (default), `--detail` (full diff for FAILs),
`--step N` (other steps if you raised `CP_DEBUG_MAX_STEPS`).

## 4. Read the result

- **Summary → First FAIL** is the first divergent layer.
- Trace upstream: if `module.out0` FAILs, check the same module's `in0` — if
  `in0` also FAILs the lesion is further upstream; if not, the lesion is in this
  module's op.
- `SHAPE_MISMATCH` → all-gather didn't align: re-check `cutoff_len % cp_size`
  and that `CP_DEBUG_PAD_TO_CUTOFF` is on.
- `ONLY_IN_CP1` / `ONLY_IN_CP2` → a hook didn't fire on one side (module
  filtered, or module naming differs).

## Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `CP_DEBUG` | `0` | Master switch (`1` enables). |
| `CP_DEBUG_DUMP_DIR` | `./cp_debug_dumps` | Dump directory. |
| `CP_DEBUG_MAX_STEPS` | `1` (here) | Number of forwards to record. |
| `CP_DEBUG_SEQ_LEN` | `cutoff_len` | Full sequence length for all-gather dim detection. |
| `CP_DEBUG_RECORD` | `both` | `forward` / `backward` / `both`. |
| `CP_DEBUG_MODE` | `dump` | `print` / `dump` / `both`. |
| `CP_DEBUG_MODULE_FILTER` | (none) | Regex, e.g. `layers\.0\.` to scope one layer. |
| `CP_DEBUG_PAD_TO_CUTOFF` | `1` when `CP_DEBUG=1` | Pad samples to `cutoff_len`. |

## Design constraints (do not regress)

- No `register_full_backward_hook` — NPU async backward conflicts; read
  `param.grad` after backward instead.
- Weights / param grads do NOT go through all-gather (params aren't CP-sharded).
- Only rank 0 writes to disk (all-gather is collective, all ranks participate).
- Forward hooks are skipped during backward (`set_in_backward`) to avoid
  recompute / gradient-checkpointing re-running forward.

Full methodology and casebook live in the upstream skill's `references/`.
