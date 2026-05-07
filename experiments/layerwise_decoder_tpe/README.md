# Layerwise decoder-only TPE playbook

This folder packages runnable configs + Slurm wrappers for `experiments/layerwise_decoder_tpe/train_layerwise_tpe.py`, which trains one TPE per decoder layer against flattened hidden states of all tokens in the active+passive IOI prompts. Defaults mirror the best punctuated Qwen3-8B TPE (filler_dim=128, role_dim=4, cosine LR=0.002, 100 epochs, batch 256, load_best_model_at_end).

## Quickstart (local or single GPU node)
```
uv run python experiments/layerwise_decoder_tpe/train_layerwise_tpe.py \
  experiments/layerwise_decoder_tpe/configs/layerwise_tpe_base.gin \
  --main.embedding_model_name='Qwen/Qwen3-8B' \
  --main.layer_indices='[0]' \
  --tpe/TrainingArguments.output_dir='checkpoints/layerwise_tpe/qwen3-8b/layer0' \
  --tpe/TrainingArguments.per_device_train_batch_size=256
```
This uses the same decoder-only model as the activation patching runs (`Qwen/Qwen3-8B`), pulls sentences from `data/sentences`, and caches per-layer flattened hidden states to `data/sentences/embeddings_Qwen_Qwen3-8B_layer{idx}.index.json`.

## Slurm job array (all Qwen3-8B layers)
```
sbatch experiments/layerwise_decoder_tpe/run_layerwise_tpe.sbatch
```
The script fans out a Slurm array over layers 0–35 with the base gin config, writing each layer to `checkpoints/layerwise_tpe/qwen3-8b/layer{idx}` and logging to `logs/layerwise_tpe/%x-%A_%a.out`. Pass extra gin overrides after `--` to the `sbatch` call if you need to tweak batch size or seeds.

## WandB hyperparameter sweep
- The sweep launcher mirrors `scripts/run_decoder_punct_sweep.py` but targets `train_layerwise_tpe.py`. Update `experiments/layerwise_decoder_tpe/run_layerwise_tpe_sweep.py` with your `WANDB_ENTITY`/project prefix and the model set (e.g., `Qwen/Qwen3-8B`).
- Create and run the sweep:
  ```
  WANDB_ENTITY=<entity> uv run python experiments/layerwise_decoder_tpe/run_layerwise_tpe_sweep.py \
    --model Qwen/Qwen3-8B \
    --base_config experiments/layerwise_decoder_tpe/configs/layerwise_tpe_base.gin \
    --sentences_path data/sentences \
    --cache_path data/sentences \
    --output_root checkpoints/layerwise_tpe \
    --sweep_count 16
  ```
  The script builds a W&B sweep over LR/weight_decay/warmup/filler_dim/role_dim and spawns agents that call `train_layerwise_tpe.py` with per-layer caches. Per-run outputs land in `checkpoints/layerwise_tpe/<safe-model>/layerwise/`.
```
sbatch experiments/layerwise_decoder_tpe/run_layerwise_tpe_sweep.sbatch
```
to launch the sweep agent on GPU nodes.

## Notes
- Per-layer caching keeps memory/lightweight runs: only the requested layer is encoded and cached.
- Prompts use single-token occupations and verbs with passive participles; failures surface early if tokenization produces mixed lengths.
- Enable WandB via gin overrides (`--main.use_wandb=True --main.wandb_project=...`) or through the sweep launcher; the script sets `TrainingArguments.report_to='wandb'` and logs explained variance per layer.***
