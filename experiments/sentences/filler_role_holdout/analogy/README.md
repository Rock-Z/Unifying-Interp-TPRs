# Heldout SVO TPE Additive Analogies

This experiment evaluates additive analogies on the SVO filler-role holdout dataset.

The configs construct analogy quadruples from the `generalization` split and rank targets against all available heldout-dataset sentences:

- `train`
- `valid`
- `test`
- `generalization`

The entry point is `evaluate_holdout_analogies.py`. Each analogy uses the existing TPE additive manipulation: start from sentence embedding `A`, subtract the TPE binding for the varied filler-role pair in `B`, add the TPE binding for the corresponding pair in `C`, and rank the intended target `D` by cosine similarity.

Run all three heldout TPE checkpoints:

```bash
bash experiments/sentences/filler_role_holdout/analogy/run_analogies.sh
```

On Misha, prefer the batch wrapper:

```bash
sbatch experiments/sentences/filler_role_holdout/analogy/run_analogies.sbatch
```

Set `FORCE_ANALOGY=1` to overwrite existing result files.

## Results

Run completed on 2026-05-06. Each model evaluated `20,580` analogies built from the `generalization` split (`10,174` subject analogies and `10,406` object analogies). Candidates were all `29,645` heldout-dataset sentences across `train`, `valid`, `test`, and `generalization`.

| Model | Top-1 | Top-3 | Top-5 | Top-10 | Mean rank | Median rank | Mean target sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModernBERT | 0.9369 | 0.9907 | 0.9935 | 0.9956 | 1.2240 | 1.0 | 0.9726 |
| EmbeddingGemma | 0.8767 | 0.9571 | 0.9724 | 0.9835 | 1.8126 | 1.0 | 0.9868 |
| Qwen3-Embedding-8B | 0.9218 | 0.9785 | 0.9835 | 0.9884 | 1.6578 | 1.0 | 0.9677 |

Role breakdown:

| Model | Role | Count | Top-1 | Top-3 | Top-5 | Top-10 | Mean rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ModernBERT | subject | 10,174 | 0.9396 | 0.9910 | 0.9929 | 0.9948 | 1.2686 |
| ModernBERT | object | 10,406 | 0.9344 | 0.9905 | 0.9941 | 0.9964 | 1.1804 |
| EmbeddingGemma | subject | 10,174 | 0.8673 | 0.9511 | 0.9660 | 0.9793 | 1.9550 |
| EmbeddingGemma | object | 10,406 | 0.8859 | 0.9629 | 0.9788 | 0.9876 | 1.6735 |
| Qwen3-Embedding-8B | subject | 10,174 | 0.9173 | 0.9773 | 0.9820 | 0.9875 | 1.7450 |
| Qwen3-Embedding-8B | object | 10,406 | 0.9261 | 0.9796 | 0.9849 | 0.9893 | 1.5726 |

Result files:

- `results/modernbert_tpe_generalization.json`
- `results/embeddinggemma_tpe_generalization.json`
- `results/qwen3_8b_tpe_generalization.json`
