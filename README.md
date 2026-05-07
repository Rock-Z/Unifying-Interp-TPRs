# Inverting TPR

Code and experiment release for **A Unifying Perspective on Language Model
Interpretability**.

The paper studies whether neural representations can be approximated as
linearly transformed Tensor Product Representations (TPRs), then uses that
single structure to derive or explain four interpretability methods:

- additive analogies
- linear probes
- sparse autoencoders
- activation patching

Experiments cover fixed-length synthetic digit copy/reverse tasks, structured
SVO sentence embeddings, decoder-only LLM period-token representations, and
filler-role holdout generalization.

## Setup

This project uses Python 3.11+ and `uv`.

```bash
uv sync
uv run pytest
```

Large model experiments require Hugging Face access for the referenced models
and enough CPU/GPU memory to cache embeddings or hidden states. Use the provided 
Slurm scripts for any training, LLM encoding, SAE sweeps, or activation patching run.

## Repository Map

- `src/`: core data generation, seq2seq models, TPE training, probing,
  analogy, SAE, and activation-patching utilities.
- `configs/`: legacy and shared gin configs.
- `scripts/analogy`: digit and sentence analogy evaluation scripts.
- `experiments/digits/`: paper digit copy/reverse experiments.
- `experiments/sentences/`: SVO sentence embedding-model experiments.
- `experiments/llm_sentences_last_layer/`: decoder-only LLM period-token TPE,
  analogy, probe, and SAE experiments.
- `experiments/activation_patching/`: baseline activation patching scripts.
- `experiments/layerwise_decoder_tpe/`: layerwise decoder-only TPE training
  used by activation-patching analysis.
- `data/`: included generated split files for the paper-scoped datasets.
- `experiments/tpe_activation_patching/`: TPE-constructed activation patching.
- `experiments/sae_sentences_topk/`: embedding-model Top-k SAE configs and
  sweep launchers.
- `MANIFEST.md` and `PROBLEMS.md`: release scope and known open choices.

## Reproducing Main Experiments

The release includes source, configs, runners, and dataset split files. It does
not include checkpoints, embedding caches, logs, or result directories.

### Synthetic Digit Tasks

The paper's digit setup is length-6, vocabulary-20 copy/reverse with one-layer
RNN/GRU/LSTM models using embedding size 64 and hidden size 256.

```bash
for cfg in experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned/configs/{copy,reverse}_{rnn,gru,lstm}.gin; do
  uv run python src/train.py "$cfg"
done

for cfg in experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized/{copy,reverse}_{rnn,gru,lstm}_probe_all_pos.gin; do
  uv run python src/invert_tpr.py "$cfg"
done

# These analogy configs expect role_dim=64 TPE outputs under
# `experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned_role64/`.
# Generate those with the role-dim runner before running the analogy evals.
for cfg in experiments/analogy_digits/configs/1le64h256_fixedlen_l2r_aligned_role64/digits_{copy,reverse}_{rnn,gru,lstm}_eval.gin; do
  PYTHONPATH=src uv run python scripts/analogy/evaluate_digits_analogy.py "$cfg"
done
```

Filler-role holdout runs:

```bash
bash experiments/digits/filler_role_holdout/run_experiment.sh
sbatch experiments/digits/filler_role_holdout/run_experiment.sbatch
```

### SVO Embedding Models

The paper's embedding-model SVO setup uses
`data/sentences_multiple/`, with 77 occupation nouns, 5 verbs, and the template
`the <subject> will <verb> the <object>.`

```bash
uv run python src/train_sentences.py experiments/sentences/configs/modernbert_multiple_verbs.gin
uv run python src/train_sentences.py experiments/sentences/configs/embeddinggemma_multiple_verbs.gin
uv run python src/train_sentences.py experiments/sentences/configs/qwen3_8b_embedding_multiple_verbs.gin
```

SVO filler-role holdout:

```bash
bash experiments/sentences/filler_role_holdout/run_experiment.sh
bash experiments/sentences/filler_role_holdout/analogy/run_analogies.sh
```

### Decoder-Only LLM SVO Representations

The LLM setup analyzes the final-layer hidden state at the sentence-final
period token for Qwen3-8B, OLMo-2-13B, and GPT-OSS-20B.

```bash
sbatch experiments/llm_sentences_last_layer/scripts/run_tpe_sweep_array.sbatch
sbatch experiments/llm_sentences_last_layer/scripts/run_downstream_array.sbatch
sbatch experiments/llm_sentences_last_layer/scripts/run_sae_sweep_array.sbatch
```

Summaries and paper-table generation:

```bash
uv run python experiments/llm_sentences_last_layer/scripts/summarize_results.py
uv run python experiments/llm_sentences_last_layer/scripts/write_paper_tables.py
```

Existing reported LLM results are summarized in
`experiments/llm_sentences_last_layer/RESULTS.md`.

### Sparse Autoencoders

Embedding-model SAE construction:

```bash
bash experiments/sae/run_all.sh
```

Embedding-model Top-k SAE training configs and W&B sweep files are under
`experiments/sae_sentences_topk/`.

LLM SAE baselines and sweeps are under
`experiments/llm_sentences_last_layer/configs/sae*` and the Slurm launchers in
`experiments/llm_sentences_last_layer/scripts/`.

### Activation Patching

Baseline token-level activation patching:

```bash
uv run python experiments/activation_patching/activation_patching.py \
  experiments/activation_patching/configs/activation_patching_sentences.gin
```

TPE-constructed activation patching:

```bash
uv run python experiments/tpe_activation_patching/tpe_activation_patching.py \
  experiments/tpe_activation_patching/configs/tpe_activation_patching_olmo13b_all_layers.gin
```

Layerwise decoder-only TPE training for patching analysis:

```bash
sbatch experiments/layerwise_decoder_tpe/scripts/run_layerwise_tpe.sbatch
```
