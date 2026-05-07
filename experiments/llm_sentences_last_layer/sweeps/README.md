# LLM Top-k SAE W&B Sweeps

These sweeps are the LLM-punctuation analogue of the embedding-model Top-k SAE sweeps in `experiments/sae_sentences_topk/sweeps/`. They train on cached final-layer punctuation-token hidden states for the SVO sentence task.

## Entry Points

- Training script: `src/train_sae.py`
- W&B launcher: `scripts/wandb_script_launcher.py`
- Slurm launcher: `experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_sweeps.sbatch`
- Fan-out Slurm launcher: `experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_fanout.sbatch`
- Fan-out agent worker: `experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_cpu_agent.sbatch`
- Sweep configs:
  - `experiments/llm_sentences_last_layer/sweeps/sae_topk_qwen3_8b.yaml`
  - `experiments/llm_sentences_last_layer/sweeps/sae_topk_olmo_13b.yaml`
  - `experiments/llm_sentences_last_layer/sweeps/sae_topk_gpt_oss_20b.yaml`

## Experimental Contract

All three sweeps use:

- `dataset_type='sentences'`
- `sentences_path='data/sentences'`
- `embedding_cache_path='data/sentences'`
- `load_dataset_with_embeddings.encoder_model_type='decoder-only-punct'`
- `role_scheme='svo'`
- `feature_mode='filler_role'`
- `random_seed=42`
- 200 training epochs
- W&B metric `avg_feature_well_rankedness` with goal `maximize`

The hidden-state input dimensions differ by model cache, but the SAE search space is intentionally shared:

- `sae_lr`: log-uniform from `1e-5` to `5e-3`
- `sae_resample_times`: `[2, 5, 8]`
- `sae_k`: `[8, 16, 32, 64, 128]`
- `sae_hidden_dim`: `[512, 1024, 2048, 4096]`
- `sae_batch`: `[128, 256]`

This preserves the embedding-model sweep protocol while widening `k` and hidden width for the larger LLM hidden states. Use `SWEEP_COUNT=20` for the old embedding-model budget; use the default `SWEEP_COUNT=40` when the goal is to tune until the Top-k SAE baseline is no longer obviously undertuned.

## Running

Dry-run schedulability:

```bash
sbatch --test-only experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_sweeps.sbatch
```

Submit one W&B sweep per LLM:

```bash
sbatch experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_sweeps.sbatch
```

Submit with the embedding-model run budget:

```bash
SWEEP_COUNT=20 sbatch experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_sweeps.sbatch
```

Each array task creates a sweep, parses the resulting W&B sweep path, and runs one local agent for `SWEEP_COUNT` trials. To run multiple agents in parallel for an already-created sweep, copy the `wandb agent ...` path from the corresponding log in `experiments/llm_sentences_last_layer/sweeps/logs/` and launch another short `day` or `devel` job with that path.

Fan out each sweep across four concurrent CPU agents while keeping 40 total runs per model:

```bash
TOTAL_RUNS_PER_SWEEP=40 RUNS_PER_AGENT=1 MAX_PARALLEL_AGENTS=4 \
  sbatch experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_fanout.sbatch
```

The fan-out coordinator creates one fresh W&B sweep per model, records the sweep paths under `experiments/llm_sentences_last_layer/sweeps/logs/fanout-<jobid>-sweeps.tsv`, and submits one Slurm agent array per sweep. With the defaults, each model gets `40` one-run agents throttled to `4` concurrent agents, so the maximum active load is `12` jobs, `384` CPUs, and `1536G` RAM across the three models.

## Artifacts

- Checkpoints: `experiments/llm_sentences_last_layer/checkpoints/sae_baselines/topk_wandb/<model_slug>/run-<wandb_run_id>/`
- Slurm logs: `experiments/llm_sentences_last_layer/sweeps/logs/`
- W&B projects:
  - `sae-llm-punctuation-topk-qwen3-8b`
  - `sae-llm-punctuation-topk-olmo-13b`
  - `sae-llm-punctuation-topk-gpt-oss-20b`
