# SVO filler-role holdout

This experiment mirrors `experiments/digits/filler_role_holdout/` for the multi-verb SVO sentence task. It tests whether a TPE trained on frozen sentence embeddings still approximates representations that contain noun-role pairs never seen during TPE training.

Status:
- Implemented in:
  - `src/generate_sentences.py`
  - `src/sentences.py`
  - `experiments/sentences/filler_role_holdout/evaluate_holdout.py`
  - `experiments/sentences/filler_role_holdout/run_experiment.sh`
  - `experiments/sentences/filler_role_holdout/analogy/`
- Base dataset: `data/sentences_multiple/`, which currently contains `77` nouns x `5` verbs = `29,645` active-voice sentences.
- Embedding models in scope:
  - `nomic-ai/modernbert-embed-base`
  - `google/embeddinggemma-300m`
  - `Qwen/Qwen3-Embedding-8B`

Frozen baselines to compare against:
- `experiments/sentences/checkpoints/modernbert/tpe/best_model`
- `experiments/sentences/checkpoints/embeddinggemma/tpe/best_model`
- `experiments/sentences/checkpoints/qwen3-8B/tpe/best_model`

Current baseline test-set TPE `R^2`:
- ModernBERT: `0.9396`
- EmbeddingGemma: `0.9155`
- Qwen3-Embedding-8B: `0.9000`

## Holdout definition

Use noun-role holdout, not verb-role holdout. In the SVO setup, verbs only appear in the verb slot, so hiding a verb-role pair would just create unseen-verb lexical OOD rather than the digits-style filler-role generalization test.

Proposed held-out subject nouns:
- `professor`, `artist`, `ambassador`, `coach`, `gardener`, `musician`, `referee`

Proposed held-out object nouns:
- `secretary`, `engineer`, `barber`, `drummer`, `linguist`, `physicist`, `technician`

Selection rule:
- Take every 11th noun from `data/sentences_multiple/data.nouns`.
- Use offset `0` for subject holdouts and offset `5` for object holdouts.

Split rule:
- `generalization`: every sentence whose subject is in the subject-holdout set or whose object is in the object-holdout set.
- `train` / `valid` / `test`: the remaining in-distribution sentences, split `80/10/10`.

Expected split sizes:
- `generalization = 5,145`
- Remaining in-distribution pool = `24,500`
- `train = 19,600`, `valid = 2,450`, `test = 2,450`

## Implementation plan

1. Data generation
- `src/generate_sentences.py` now supports:
  - `--holdout_step`
  - `--subject_holdout_offset`
  - `--object_holdout_offset`
  - `--seed`
- Write a new dataset directory `data/sentences_multiple_filler_role_holdout/` containing:
  - `data.train`
  - `data.valid`
  - `data.test`
  - `data.generalization`
  - `data.nouns`
  - `data.verbs`
  - `data.holdout_metadata.json`
- Unlike digits, this dataset is exhaustive, so partition the full noun x verb x noun grid directly instead of rejection sampling.

2. Loader support
- `src/sentences.py` now reads `data.generalization` when present.
- Keep the existing `DatasetDict` interface so downstream embedding caching continues to work unchanged.

3. TPE training and evaluation
- Model-specific training configs live under `experiments/sentences/filler_role_holdout/configs/` and override only:
  - `main.sentences_path`
  - `main.embedding_cache_path`
  - output directories
- Train on `train`, early-stop/select on `valid`, and evaluate on both `test` and `generalization`.
- `experiments/sentences/filler_role_holdout/evaluate_holdout.py` reports:
  - MSE
  - cosine similarity
  - `R^2`
  - explained variance
  - probe accuracy on `subj`, `verb`, and `obj` for both splits

4. Probe comparison
- Extend `src/invert_svo.py` or wrap it with a holdout-specific evaluator so probe outputs are split-aware, e.g. `subj_test`, `subj_generalization`, etc.
- Keep cache keys split-qualified to avoid colliding with existing `trained_probe_results_svo.json` files.

5. Launcher
- `run_experiment.sh` uses the same interface as digits:
  - `FORCE_DATASET`
  - `FORCE_RETRAIN`
  - `FORCE_EVAL`
- Run order:
  - generate the holdout dataset once
  - ModernBERT
  - EmbeddingGemma
  - Qwen3-Embedding-8B

## Planned outputs

- Dataset: `data/sentences_multiple_filler_role_holdout/`
- Checkpoints: `experiments/sentences/filler_role_holdout/checkpoints/<model>/tpe/`
- Holdout metrics: `.../holdout_eval_metrics.json`
- Split-aware probe metrics: `.../probe_compare_results_svo.json`

## Run

```bash
bash experiments/sentences/filler_role_holdout/run_experiment.sh
```

## Heldout TPE additive analogies

Analogy configs live under `experiments/sentences/filler_role_holdout/analogy/configs/`.
They construct analogy quadruples from the held-out `generalization` split and
rank against `train`, `valid`, `test`, and `generalization` candidates.

```bash
bash experiments/sentences/filler_role_holdout/analogy/run_analogies.sh
```

On Misha, prefer:

```bash
sbatch experiments/sentences/filler_role_holdout/analogy/run_analogies.sbatch
```

## Notes

- Keep the holdout dataset in its own directory so embedding caches are separate from `data/sentences_multiple/`.
- Qwen3-Embedding-8B may need a smaller embedding batch size if the new cache must be built from scratch.
- If we want unseen-verb generalization later, that should be a separate lexical OOD experiment rather than part of this filler-role holdout.
