# Handoff: GPU testing for per-head QK-norm gains (`qk_norm_per_head_gains`)

## What was implemented

A new opt-in flag, `qk_norm_per_head_gains`, was added to OLMo-core's attention module. Background: with `use_head_qk_norm=True`, the QK norm is applied to the `(B, T, n_heads, head_dim)` tensor, so normalization statistics are per-head, but the learnable gain was a single `(head_dim,)` vector shared across heads. The new flag gives each head its own gains: `q_norm.weight` becomes `(n_heads, head_dim)` and `k_norm.weight` becomes `(n_kv_heads, head_dim)`, applied by broadcasting after normalization. Statistics are unchanged (still computed over `head_dim` only). The flag requires `use_head_qk_norm=True` (raises `OLMoConfigurationError` otherwise). Defaults are unchanged, so existing configs/checkpoints (e.g. Qwen3, Gemma) are unaffected.

Changed files (all changes are in the current working tree):

- `src/olmo_core/nn/layer_norm.py`: `LayerNorm.__init__` (and subclasses) accept an optional `weight_shape` that decouples the weight/bias shape from `normalized_shape`. `LayerNorm.forward` applies the affine manually via broadcasting when the shapes differ (since `F.layer_norm` requires them to match). `RMSNorm`/`QwenRMSNorm` needed no forward changes (they already broadcast `self.weight * x`). `FusedRMSNorm` and `CuTeRMSNorm` raise `NotImplementedError` for a non-default `weight_shape` (their kernels only take a 1D weight). `LayerNormConfig.build()` threads `weight_shape` through.
- `src/olmo_core/nn/attention/__init__.py`: new `qk_norm_per_head_gains` field on `AttentionConfig` and param on `Attention.__init__`, validation, per-head norm construction, and updated `AttentionConfig.num_params()` accounting.
- `src/test/nn/attention/attention_test.py`: new parametrized cases (`head-qk-norm-per-head-gains` in `test_attention`, `headwise-qk-layernorm-per-head-gains` in `test_tensor_parallel_attention`, GQA cases in `test_attention_builder_config`) and a new `test_qk_norm_per_head_gains` unit test.

## What was already verified (CPU-only machine)

- All new and existing QK-norm-related tests pass on CPU: `pytest src/test/nn/attention/attention_test.py -k 'qk or layernorm'` (26 passed; everything GPU-marked was skipped).
- `test_tensor_parallel_attention[headwise-qk-layernorm-per-head-gains-backend=GLOO]` passes (TP on CPU via gloo).
- Numerical sanity check: for `default`, `rms`, and `qwen_rms` norm types, a `(n_heads, head_dim)` weight with distinct per-row gains produces exactly `per_head_gain * shared_reference_output` per head.
- `isort`, `black`, `ruff`, `mypy` all clean.

**Everything GPU-related was skipped**: flash-attn 2/3/4 backends, TransformerEngine backend, bf16, NCCL tensor parallelism, `FusedRMSNorm`/`CuTeRMSNorm` kernel paths. That's your job.

## Your tasks

Use the repo's venv/environment as set up on the machine (`pytest -v src/...` per `AGENTS.md`). Note the GPU is a B300 (Blackwell) — if flash-attn 2 lacks sm_103 support in the installed build, note which backends were actually testable rather than silently skipping.

### 1. Run existing GPU-marked tests for the new flag

```bash
# All per-head-gains parametrizations across backends (flash_2/3/4, te, torch-SDPA) and dtypes:
pytest -v src/test/nn/attention/attention_test.py -k 'per-head-gains'

# Full QK-norm regression sweep (existing shared-gain behavior must not regress):
pytest -v src/test/nn/attention/attention_test.py -k 'qk or layernorm'

# NCCL tensor-parallel case (needs >= 2 GPUs; skip and note it if only 1 is available):
pytest -v src/test/nn/attention/attention_test.py -k 'tensor_parallel and per-head-gains'

# Layer norm suite, including the fused/CuTe kernel tests that were skipped on CPU:
pytest -v src/test/nn/layer_norm_test.py
```

### 2. Verify the fused/CuTe rejection on a machine where the kernels exist

On this machine flash-attn (and possibly quack) should be installed, so actually exercise the guards:

```python
from olmo_core.nn.layer_norm import FusedRMSNorm
# Should raise NotImplementedError mentioning 'weight_shape':
FusedRMSNorm(size=16, weight_shape=(8, 16))
```

Same for `CuTeRMSNorm` if quack is installed. Also confirm `LayerNormConfig(name="fused_rms").build(size=16, weight_shape=(8, 16))` fails cleanly.

### 3. Parity tests (the most important part)

Write and run a throwaway script (or extend the test file if the results are worth keeping) that checks, on CUDA in both fp32 and bf16 (autocast), for `Attention` with `n_heads=8, n_kv_heads=2, d_model=128`, `qk_norm=LayerNormConfig(name="rms", bias=False)`, `use_head_qk_norm=True`:

a. **Init-time parity**: with freshly initialized weights (all norm gains are ones), a per-head-gains model given identical projection weights must produce bit-identical (fp32) or allclose (bf16) outputs to a shared-gains model. Test across every available backend (`torch`, `flash_2`, `te`, ...).

b. **Broadcast parity**: copy a random shared `(head_dim,)` gain into every row of the per-head `(n_heads, head_dim)` weight; outputs must match the shared-gain model with that same gain.

c. **Per-head effect**: set distinct gains per row and verify each head's pre-RoPE normalized queries/keys scale independently (e.g. hook on `q_norm` output), and that final outputs differ from the shared model.

d. **Gradients**: run a backward pass and confirm `q_norm.weight.grad` has shape `(n_heads, head_dim)` with rows that differ from each other (i.e. per-head gradient signal actually flows), and similarly `(n_kv_heads, head_dim)` for `k_norm`.

e. **Backend cross-parity**: for the per-head-gains model, outputs across backends (`torch` vs `flash_2` vs `te`) should agree within bf16 tolerances (the repo uses `BF16_RTOL`/`BF16_ATOL` from `olmo_core.testing` — see how `test_attention` and `test_sdpa` in `src/test/nn/attention/attention_test.py` compare backends).

### 4. Model-level smoke test

Build a small transformer via `TransformerConfig` (e.g. adapt `TransformerConfig.olmo2_190M` or any factory, overriding the attention options with `qk_norm=..., use_head_qk_norm=True, qk_norm_per_head_gains=True`), run a few forward/backward steps on GPU in bf16 with a real backend, and confirm losses are finite and decreasing over ~20 steps on random/synthetic data. Also verify `TransformerConfig.num_params` matches `sum(p.numel() ...)` for that model.

### 5. Optional if time permits

- `torch.compile` the per-head-gains `Attention` module and check outputs match eager (the manual-affine branch in `LayerNorm.forward` is new code under compile).
- Distributed checkpoint save/load round-trip of a per-head-gains model (`save_model_and_optim_state` / `load_model_and_optim_state`, as used in `_run_tensor_parallel_attention`).

## Reporting

Report: which backends were actually exercised on the B300, full pass/fail per task, any tolerance violations with the actual max abs/rel errors, and anything that had to be skipped and why. If a failure looks like a bug in the new code rather than an environment issue, include a minimal repro.
