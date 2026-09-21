"""Parallelism-aware checkpoint sizing (Mark meeting 2026-07-23, decision D2).

Kills the per-node "every node writes X GB" magic number. A training job is
described by a PARALLELISM CONFIG (model size + DP/TP/PP/ZeRO); this module maps
that config to

  (i)  the per-GPU checkpoint SHARD (bytes one rank must persist), and
  (ii) the DP REPLICATION STRUCTURE — which fraction of a rank's state is a
       duplicate of a data-parallel peer (REPLICATED) vs held only by this rank
       (UNIQUE) — the quantity the DP-aware controller exploits (parallelism.py
       is imported by both the simulator driver and gp_policy so the solver and
       the sim price the *same* bytes).

Pure arithmetic, deterministic, no state — safe to import anywhere.

MIXED-PRECISION LLM TRAINING STATE (bytes per parameter)
--------------------------------------------------------
Standard Adam mixed-precision training keeps, PER PARAMETER of the model:

    bf16 parameters .......... 2 B   (weights used in fwd/bwd)
    bf16 gradients ........... 2 B
    fp32 master parameters ... 4 B   (the high-precision copy Adam updates)
    fp32 Adam m (1st moment) . 4 B
    fp32 Adam v (2nd moment) . 4 B
    ---------------------------------
    TOTAL .................... 16 B / param

This is the canonical "16 bytes/param" figure: Rajbhandari et al., "ZeRO:
Memory Optimizations Toward Training Trillion Parameter Models" (SC'20), §5 /
Table 1 gives 2+2+K with K = 12 for Adam (fp32 master + m + v); see also the
Megatron-LM and DeepSpeed memory accounting. The checkpoint state we size is
this full training state (params + grads + optimizer), because resuming Adam
needs the moments; recomputing them from scratch loses the optimizer's history.

WHAT EACH PARALLELISM AXIS DOES TO THE STATE
--------------------------------------------
* TP (tensor) and PP (pipeline) PARTITION the model: a rank in a given
  (tp-slot, pp-slot) holds P/(tp*pp) parameters' worth of state, and that
  partition is UNIQUE across the whole job (no other rank holds it).
* DP (data) REPLICATES a partition across `dp` ranks. ZeRO decides how much of
  that replicated state is actually sharded across the DP group:
      zero_stage 0  params + grads + optimizer FULLY REPLICATED across DP;
      zero_stage 1  OPTIMIZER (12 B/param) sharded across DP (each rank unique),
                    params + grads (4 B/param) still replicated;
      zero_stage 2  (grad partitioning) — treated as stage 1 for checkpoint-state
                    math (the optimizer split is what matters to persisted bytes;
                    grads are transient). Documented approximation.
      zero_stage 3  EVERYTHING sharded across DP (each rank fully unique).

So per GPU, in bytes/param of its P/(tp*pp) partition:

    stage 0:  unique = 0        replicated = 16
    stage 1:  unique = 12/dp    replicated = 4
    stage 3:  unique = 16/dp    replicated = 0

CHECKPOINT NEED (per job) = one copy of every UNIQUE shard + one copy of every
REPLICATED shard (NOT the `dp` duplicate copies). That deduplicated need is
16 B/param * P for EVERY stage — the model's full training state, exactly once.
Stage 0 STORES `dp` redundant copies of it; stage 3 stores it once, striped
across the DP group. This is precisely the redundancy DP-aware checkpointing
(dp_dedup) removes: it rotates a 1/dp slice of the replicated bytes to each DP
rank, so per-rank persisted bytes fall to  unique + replicated/dp  =  16/dp
B/param  regardless of stage — one deduplicated copy spread across the DP group.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- mixed-precision Adam state, bytes per parameter (see module docstring) ---
BYTES_PARAMS_BF16 = 2
BYTES_GRADS_BF16 = 2
BYTES_MASTER_FP32 = 4
BYTES_ADAM_M_FP32 = 4
BYTES_ADAM_V_FP32 = 4
BYTES_PER_PARAM = (BYTES_PARAMS_BF16 + BYTES_GRADS_BF16 + BYTES_MASTER_FP32
                   + BYTES_ADAM_M_FP32 + BYTES_ADAM_V_FP32)          # = 16
# ZeRO shards the OPTIMIZER (fp32 master + m + v); params+grads stay replicated
# until stage 3. These two sum to BYTES_PER_PARAM.
BYTES_OPTIMIZER = BYTES_MASTER_FP32 + BYTES_ADAM_M_FP32 + BYTES_ADAM_V_FP32   # 12
BYTES_PARAMS_GRADS = BYTES_PARAMS_BF16 + BYTES_GRADS_BF16                      # 4

# params are given in BILLIONS; a shard in GB (1e9 bytes) is then just
# bytes_per_param * params_billion / (tp*pp)  (the 1e9 factors cancel).
DEFAULT_HBM_BUDGET_GB = 80.0    # per-GPU HBM sanity bound for training state


@dataclass(frozen=True)
class ModelConfig:
    """Normalized per-class parallelism config (from a scenario `model:` block)."""
    name: str
    params_b: float          # model size, BILLIONS of parameters
    dp: int                  # data-parallel degree
    tp: int                  # tensor-parallel degree
    pp: int                  # pipeline-parallel degree
    zero_stage: int          # 0 | 1 | 3 (2 normalized to 1 for state math)

    @property
    def gpus(self) -> int:
        return self.dp * self.tp * self.pp

    @property
    def partition_params_b(self) -> float:
        """Billions of params in ONE (tp,pp) model-parallel partition."""
        return self.params_b / (self.tp * self.pp)


def parse_model(spec: dict) -> ModelConfig:
    """Validate + normalize a scenario `model:` mapping into a ModelConfig.

    Keys: name (str, optional), params_b (>0), dp/tp/pp (>=1 ints), zero_stage
    (0/1/2/3; 2 -> 1). Raises ValueError on anything malformed so a typo in a
    scenario fails loudly rather than silently mis-sizing checkpoints."""
    if not isinstance(spec, dict):
        raise ValueError(f"model: block must be a mapping, got {type(spec).__name__}")
    try:
        params_b = float(spec["params_b"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("model.params_b (billions of params) is required and "
                         "must be numeric") from exc
    if not (params_b > 0):
        raise ValueError(f"model.params_b must be > 0, got {params_b}")
    dims = {}
    for key in ("dp", "tp", "pp"):
        val = int(spec.get(key, 1))
        if val < 1:
            raise ValueError(f"model.{key} must be >= 1, got {val}")
        dims[key] = val
    stage = int(spec.get("zero_stage", 0))
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"model.zero_stage must be in {{0,1,2,3}}, got {stage}")
    if stage == 2:                     # grad partitioning: persisted-state math
        stage = 1                      # matches stage 1 (see module docstring)
    return ModelConfig(name=str(spec.get("name", "")), params_b=params_b,
                       dp=dims["dp"], tp=dims["tp"], pp=dims["pp"],
                       zero_stage=stage)


def _bytes_per_param(stage: int, dp: int) -> tuple[float, float]:
    """(unique_bpp, replicated_bpp) for one GPU's partition, by ZeRO stage.

    unique = bytes only this rank holds; replicated = bytes duplicated on the
    other dp-1 ranks of its DP group. unique+replicated is the per-GPU shard."""
    if stage == 0:
        return 0.0, float(BYTES_PER_PARAM)                    # (0, 16)
    if stage == 1:
        return BYTES_OPTIMIZER / dp, float(BYTES_PARAMS_GRADS)  # (12/dp, 4)
    if stage == 3:
        return BYTES_PER_PARAM / dp, 0.0                      # (16/dp, 0)
    raise ValueError(f"unexpected normalized zero_stage {stage}")


def shard_report(model: ModelConfig) -> dict:
    """Full per-GPU byte breakdown for a config (GB, 1e9-byte convention).

    Keys:
      per_gpu_shard_gb   bytes one rank persists WITHOUT dedup (= HBM state)
      unique_gb          of that, bytes unique to this rank
      replicated_gb      of that, bytes duplicated across the DP group
      effective_shard_gb bytes one rank persists WITH dp_dedup (unique+repl/dp)
      per_gpu_state_gb   HBM footprint of training state (== per_gpu_shard_gb)
      checkpoint_need_gb whole-JOB deduplicated need (= 16 B/param * P)
      dp/tp/pp/zero_stage echoed for callers
      hbm_ok             per_gpu_state_gb <= DEFAULT_HBM_BUDGET_GB
    """
    pp_b = model.partition_params_b
    uniq_bpp, repl_bpp = _bytes_per_param(model.zero_stage, model.dp)
    unique_gb = uniq_bpp * pp_b
    replicated_gb = repl_bpp * pp_b
    per_gpu_shard_gb = unique_gb + replicated_gb
    effective_shard_gb = unique_gb + replicated_gb / model.dp
    return {
        "name": model.name,
        "params_b": model.params_b,
        "dp": model.dp, "tp": model.tp, "pp": model.pp,
        "zero_stage": model.zero_stage,
        "gpus": model.gpus,
        "partition_params_b": pp_b,
        "unique_gb": unique_gb,
        "replicated_gb": replicated_gb,
        "per_gpu_shard_gb": per_gpu_shard_gb,
        "per_gpu_state_gb": per_gpu_shard_gb,
        "effective_shard_gb": effective_shard_gb,
        "replicated_frac": (replicated_gb / per_gpu_shard_gb
                            if per_gpu_shard_gb > 0 else 0.0),
        "checkpoint_need_gb": BYTES_PER_PARAM * model.params_b,
        "hbm_ok": per_gpu_shard_gb <= DEFAULT_HBM_BUDGET_GB,
    }


def per_gpu_shard_gb(model: ModelConfig) -> float:
    """Per-GPU checkpoint shard WITHOUT dedup (the bytes baselines write, and the
    per-GPU HBM training-state footprint)."""
    uniq_bpp, repl_bpp = _bytes_per_param(model.zero_stage, model.dp)
    return (uniq_bpp + repl_bpp) * model.partition_params_b


def effective_shard_gb(model: ModelConfig, dedup: bool) -> float:
    """Per-GPU checkpoint shard. dedup=False -> full shard (unique+replicated);
    dedup=True -> DP-aware (unique + replicated/dp), i.e. each DP rank persists a
    rotating 1/dp slice of the replicated bytes. Collapses to 16/dp B/param for
    every ZeRO stage (one deduplicated copy spread across the DP group)."""
    uniq_bpp, repl_bpp = _bytes_per_param(model.zero_stage, model.dp)
    repl = repl_bpp / model.dp if dedup else repl_bpp
    return (uniq_bpp + repl) * model.partition_params_b


def jit_dram_recoverable(model: ModelConfig | None) -> bool:
    """Can Just-In-Time checkpointing reconstruct a lost rank's FULL training
    state from a single surviving DP peer's DRAM? Only when a DP peer holds an
    intact replica, i.e. dp>1 AND zero_stage==0 (full replication). Under
    ZeRO-1 the failed rank's optimizer slice lives ONLY on that rank (sharded,
    unique) so no peer has it; under ZeRO-3 the entire shard is unique. JIT
    keeps no periodic checkpoint, so in those cases a lost rank is unrecoverable
    from survivors -> initial_state (see baselines.JITStrategy / the fairness
    audit). model=None (legacy scenario, no parallelism block) keeps the
    original survivor-drain behavior unchanged."""
    if model is None:
        return True                                # legacy: unchanged
    return model.dp > 1 and model.zero_stage == 0


def apply_model_shards(sc: dict) -> dict:
    """Wire parallelism configs into a scenario dict, IN PLACE (and return it).

    For every class carrying a `model:` block, set its `checkpoint_gb_per_rank`
    to the calculator's NO-DEDUP per-GPU shard, so the whole existing pipeline
    (build_config, baselines, capture/flush accounting, the solver) reads the
    parallelism-correct shard through the same field it always used. Idempotent.

    Classes WITHOUT a `model:` block are left EXACTLY as-is, so every legacy
    scenario is byte-identical (zero regression). The model block is authoritative
    when present: it overrides any hand-set checkpoint_gb_per_rank (which for a
    model-configured class would be a contradiction), and stores the breakdown
    under `_model_shard` for reporting."""
    for spec in sc.get("classes", {}).values():
        mdl = spec.get("model")
        if not mdl:
            continue
        model = parse_model(mdl)
        rep = shard_report(model)
        spec["checkpoint_gb_per_rank"] = rep["per_gpu_shard_gb"]
        spec["_model_shard"] = rep          # breakdown for reporting / dedup
    return sc
