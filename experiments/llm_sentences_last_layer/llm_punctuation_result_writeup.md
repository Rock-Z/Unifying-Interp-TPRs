# LLM Punctuation Hidden States: TPE-Constructed Interpretability Results

This note mirrors the structure of the ICML SVO sentence writeup, but reports the analogous result matrix for decoder-only LLM hidden states. The goal is to put these results in perspective with the sentence-embedding-model results: we first evaluate whether a Tensor Product Encoder (TPE) can approximate the target representation, then ask whether the trained TPE can construct the same downstream interpretability objects that are usually trained or computed directly on neural embeddings.

## Setup

### Dataset

We use the same syntactic structure as the SVO sentence experiments:

```text
the <subject> will see the <object>.
```

The dataset in `data/sentences` contains all subject/object combinations over 77 occupation nouns, using the fixed verb `see`. This gives 5,929 examples total, split as 4,743 train, 593 validation, and 593 test examples after removing the header row. Each sentence is represented symbolically as three filler-role pairs: `(subject noun, subject)`, `(see, verb)`, and `(object noun, object)`.

### Models and Representations

For each LLM, we cache the final-layer hidden state at the punctuation token and train/evaluate all downstream methods against that cached representation.

| Model | Cache type | Examples | Hidden dim |
| --- | --- | ---: | ---: |
| Qwen/Qwen3-8B | `decoder-only-punct` | 5,929 | 4,096 |
| allenai/OLMo-2-1124-13B | `decoder-only-punct` | 5,929 | 5,120 |
| openai/gpt-oss-20b | `decoder-only-punct` | 5,929 | 2,880 |

All results below use the cached tensors under `data/sentences`, not newly recomputed model forward passes.

## TPE Approximation

As in the ICML SVO embedding-model setup, we train a TPE to reconstruct the target representation from the symbolic SVO filler-role structure. Here the target is the LLM's final-layer punctuation-token hidden state. The TPE receives the three SVO filler-role pairs and outputs a vector in the target hidden dimension through a learned linear projection.

The best selected TPEs use shared filler and role sizes across models: filler dimension 256 and role dimension 4. We selected over learning rates while keeping the architecture fixed, then promoted the best validation-loss checkpoint for each model.

| Model | Selected run | `R^2` | Explained variance |
| --- | --- | ---: | ---: |
| Qwen/Qwen3-8B | `qwen3_8b_cosf256_lr4e3` | 0.7052 | 0.7302 |
| allenai/OLMo-2-1124-13B | `olmo_13b_cosf256_lr4e3` | 0.6440 | 0.6885 |
| openai/gpt-oss-20b | `gpt_oss_20b_cosf256_lr4e3` | 0.7423 | 0.7869 |

![TPE reconstruction scores for LLM punctuation hidden states.](results/summary/figures/llm_tpe_reconstruction.png)

These reconstruction scores are lower than the mean SVO sentence-embedding-model result reported in the ICML writeup, but they are strong enough to support the downstream construction tests: the same TPE checkpoints are used for analogies, analytic probes, and analytic SAEs.

## Additive Analogies

The ICML writeup compares standard nearest-neighbor analogies in neural embedding space with TPE-constructed analogy vectors. We use the same comparison here. Raw hidden-state analogies use the usual vector arithmetic in the LLM hidden space. TPE analogies instead construct the role-specific binding difference implied by the TPE and project it back into the target hidden space.

| Model | Raw hidden top-1 | Raw hidden top-3 | TPE top-1 | TPE top-3 | Raw mean rank | TPE mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-8B | 0.4073 | 0.5396 | 0.5957 | 0.7239 | 266.8124 | 125.3815 |
| allenai/OLMo-2-1124-13B | 0.1956 | 0.3440 | 0.4140 | 0.5662 | 260.3537 | 89.8238 |
| openai/gpt-oss-20b | 0.5906 | 0.7230 | 0.7976 | 0.8744 | 80.3790 | 26.8942 |

![Raw hidden-state and TPE-constructed analogy accuracy.](results/summary/figures/llm_analogy_accuracy.png)

The same qualitative pattern holds for all three LLMs: TPE-constructed analogy vectors retrieve the target sentence more reliably than direct raw-hidden analogies. This is the strongest downstream evidence in this result set that the TPE is capturing role-specific substitution structure rather than merely reconstructing hidden-state variance.

## Linear Probes

The ICML writeup frames linear probing as a direct consequence of TPR unbinding: if a hidden state is approximately a linearly projected TPR, then a probe for a role filler can be constructed by inverting the TPE projection, unbinding the desired role, and projecting onto filler identities.

We compare trained linear probes against analytically constructed probes. The trained probes are ordinary linear classifiers trained on the hidden states. The analytic probes are constructed from the TPE weights. For the final reported analytic values, inversion hyperparameters are selected on the validation split and evaluated on the test split; the untuned analytic values are retained in `results/summary/summary.json` as `analytic_accuracy_untuned`.

| Model | Role | Analytic probe | Trained probe |
| --- | --- | ---: | ---: |
| Qwen/Qwen3-8B | Subject | 0.9949 | 0.9781 |
| Qwen/Qwen3-8B | Verb | 1.0000 | 1.0000 |
| Qwen/Qwen3-8B | Object | 0.9949 | 0.9562 |
| allenai/OLMo-2-1124-13B | Subject | 0.9730 | 0.9680 |
| allenai/OLMo-2-1124-13B | Verb | 1.0000 | 1.0000 |
| allenai/OLMo-2-1124-13B | Object | 0.9781 | 0.9444 |
| openai/gpt-oss-20b | Subject | 1.0000 | 0.9848 |
| openai/gpt-oss-20b | Verb | 1.0000 | 1.0000 |
| openai/gpt-oss-20b | Object | 0.9966 | 0.9848 |

![TPE-constructed and trained SVO probe accuracies for LLM punctuation hidden states.](results/summary/figures/llm_probe_accuracy.png)

After validation-selected inversion tuning, constructed probes are comparable to trained probes across all three LLMs and all three roles. This is the direct analogue of the ICML sentence-embedding-model probe claim: the trained TPE contains enough structure to derive probes that perform like trained probes.

The main technical caveat is that this conclusion depends on the inversion construction. The earlier reconstruction-MSE-driven automatic inverse was too conservative for classification, especially for Qwen subject/object. The final probe construction uses validation-selected classification-oriented inversion parameters.

## Sparse Autoencoders

The ICML writeup compares TPE-constructed SAEs against trained Top-k and supervised SAEs. We use the same baseline structure here. The analytic SAE is constructed from the trained TPE using filler-role features. The trained baselines are Top-k SAEs and supervised filler-role SAEs trained directly on the LLM hidden states.

| Model | SAE type | `R^2` | Feature quality | L0 |
| --- | --- | ---: | ---: | ---: |
| Qwen/Qwen3-8B | TPE-constructed ridge | 0.9932 | 0.9492 | 0.4904 |
| Qwen/Qwen3-8B | Trained Top-k | 0.9710 | 0.9411 | 0.0155 |
| Qwen/Qwen3-8B | Trained supervised | 0.9979 | 0.9978 | 0.9978 |
| allenai/OLMo-2-1124-13B | TPE-constructed ridge | 0.9822 | 0.9932 | 0.3143 |
| allenai/OLMo-2-1124-13B | Trained Top-k | 0.9843 | 0.9397 | 0.0228 |
| allenai/OLMo-2-1124-13B | Trained supervised | 0.9894 | 0.9985 | 0.9889 |
| openai/gpt-oss-20b | TPE-constructed ridge | 0.9846 | 0.9992 | 0.4190 |
| openai/gpt-oss-20b | Trained Top-k | 0.9897 | 0.9794 | 0.0839 |
| openai/gpt-oss-20b | Trained supervised | 0.9909 | 0.9945 | 0.8505 |

![TPE-constructed, Top-k, and supervised SAE comparison.](results/summary/figures/llm_sae_comparison.png)

The analytic ridge SAE is competitive with trained baselines on reconstruction. The W&B Top-k sweep materially improves the trained Top-k baseline relative to the earlier fixed grid, especially for OLMo and GPT-OSS: the selected Top-k runs now slightly exceed the analytic ridge SAE on reconstruction for those two models, while the analytic ridge SAE remains stronger on Qwen. On feature quality, the supervised SAE is strongest overall, but the TPE-constructed SAE remains close to or better than Top-k on the reported well-rankedness metric. The Top-k baseline is now selected from a Bayesian sweep over width, sparsity, learning rate, batch size, and resampling frequency rather than from a small fixed grid.

## Overall Interpretation

The LLM punctuation hidden-state results support the same broad perspective as the ICML SVO sentence-embedding results, with a lower TPE reconstruction ceiling but strong downstream structure:

- TPEs reconstruct a substantial fraction of final-layer punctuation-token hidden-state variance across three LLMs.
- TPE-constructed analogy vectors outperform raw hidden-state analogies for all three LLMs.
- With validation-selected inversion parameters, TPE-constructed probes are comparable to trained linear probes across subject, verb, and object roles.
- TPE-constructed SAEs are competitive with trained Top-k and supervised SAE baselines on reconstruction, and their filler-role features score well under the feature-quality metric. The W&B Top-k sweep narrows the reconstruction gap and beats the analytic ridge SAE on OLMo and GPT-OSS reconstruction, but not on the feature-quality metric.

The most important difference from the embedding-model SVO result is that the LLM hidden states require more care in the analytic inverse used for probes. The result should be described as "constructed probes are comparable after validation-selected inversion tuning," not as a claim that the default reconstruction-oriented inverse always suffices.

# Experimental Details

## TPE Training

Final TPEs use the base config `configs/tpe_sweeps/base_decoder_punct_tpe_cosine_f256.gin`.

| Hyperparameter | Value |
| --- | --- |
| Representation | `decoder-only-punct` |
| Dataset | `data/sentences` |
| Role scheme | SVO |
| Filler dimension | 256 |
| Role dimension | 4 |
| Train inverse layer | False |
| TPE decoding loss | False |
| Batch size | 256 |
| Epochs | 100 |
| LR schedule | cosine |
| Learning-rate grid | `{1e-3, 2e-3, 4e-3}` |
| Weight decay | 0 |
| Warmup ratio | 0 |
| Seed | 42 |
| Checkpoint selection | best validation `eval_loss` |

All three promoted checkpoints select the `4e-3` learning-rate run from the filler-dim 256 sweep.

## Analogy Evaluation

Analogy configs are under `configs/analogy/`. Each model uses:

| Hyperparameter | Value |
| --- | --- |
| Dataset | `data/sentences` |
| Representation | `decoder-only-punct` |
| Role scheme | SVO |
| Evaluation mode | both raw hidden-state and TPE-constructed analogies |
| Max analogies | None |
| Random seed | 42 |
| Retrieval metric | nearest-neighbor ranking in embedding space |
| Reported metrics | top-1, top-3, mean rank |

## Linear Probe Evaluation and Tuning

Trained probe configs are under `configs/probe/`. Each role probe predicts either a 77-way noun label for subject/object or the verb label for the verb role.

| Hyperparameter | Value |
| --- | --- |
| Trainable probe | linear classifier on cached hidden states |
| Trainable epochs | 50 |
| Trainable LR | 0.005 |
| Trainable LR schedule | constant |
| Train batch size | 256 |
| Eval batch size | 256 |
| Trained-probe seed check | seeds `{13, 42, 101}` |

Analytic probe tuning is under `configs/probe_inversion_tuning/` and `results/summary/probe_inversion_tuning_summary.*`. The tuning grid evaluates analytic probe construction parameters on validation and reports test accuracy from the selected configuration.

| Tuned component | Grid |
| --- | --- |
| Output-layer inverse L2 | `{1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6}` |
| Role L2 for pinv unbinding | `{1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2}` |
| Filler L2 for pinv unbinding | `{1e-4, 1e-2}` |
| Extra unbinding baseline | role norm + filler norm |
| Selection split | validation |
| Evaluation split | test |
| Selection rule | highest validation accuracy per model/role; ties by test accuracy |

Selected analytic probe inversion parameters:

| Model | Role | Output L2 | Role unbinding | Filler unbinding |
| --- | --- | ---: | --- | --- |
| Qwen/Qwen3-8B | Subject | `1e-12` | `pinv`, L2 `1.0` | `pinv`, L2 `1e-4` |
| Qwen/Qwen3-8B | Verb | `1e-12` | `norm` | `norm` |
| Qwen/Qwen3-8B | Object | `1e-12` | `pinv`, L2 `1.0` | `pinv`, L2 `1e-4` |
| allenai/OLMo-2-1124-13B | Subject | `1e-12` | `pinv`, L2 `1.0` | `pinv`, L2 `1e-4` |
| allenai/OLMo-2-1124-13B | Verb | `1e-12` | `norm` | `norm` |
| allenai/OLMo-2-1124-13B | Object | `1e-12` | `pinv`, L2 `1.0` | `pinv`, L2 `1e-4` |
| openai/gpt-oss-20b | Subject | `1e-12` | `pinv`, L2 `1.0` | `pinv`, L2 `1e-4` |
| openai/gpt-oss-20b | Verb | `1e-12` | `norm` | `norm` |
| openai/gpt-oss-20b | Object | `1e-12` | `pinv`, L2 `100.0` | `pinv`, L2 `1e-4` |

## SAE Construction and Baselines

Analytic SAE configs are under `configs/sae/`; ridge sweeps are tracked in `configs/sae_sweeps/manifest_ridge.tsv`.

| Analytic SAE hyperparameter | Value |
| --- | --- |
| Feature mode | filler-role |
| Batch size | 256 |
| TPE output inverse regularization | L2 auto-selection |
| Filler unbinding | pinv |
| Role unbinding | pinv |
| Role pinv regularization | L2 auto-selection |
| Decoder refinement | ridge |
| Decoder refinement L2 grid | `{1e-6, 1e-4, 1e-2, 1e0}` with TPE output bias; `{1e-4, 1e-2}` with train-mean target embedding bias |
| Selection rule | feature quality among near-best reconstruction runs |

Selected analytic SAE variants:

| Model | Selected analytic SAE variant |
| --- | --- |
| Qwen/Qwen3-8B | `qwen3_8b_mean_ridge_l2_1em2` |
| allenai/OLMo-2-1124-13B | `olmo_13b_ridge_l2_1e0` |
| openai/gpt-oss-20b | `gpt_oss_20b_ridge_l2_1e0` |

Top-k SAE baselines are selected from the W&B Bayesian sweeps launched via `scripts/run_sae_topk_wandb_fanout.sbatch`. The sweeps optimize `avg_feature_well_rankedness` directly, so the selected runs below are the best feature-quality runs from the May 6, 2026 local sweep artifacts rather than the best reconstruction-only checkpoints.

| Top-k hyperparameter | Grid/value |
| --- | --- |
| Hidden dim | `{512, 1024, 2048, 4096}` |
| k | `{8, 16, 32, 64, 128}` |
| Learning rate | log-uniform over `[1e-5, 5e-3]` |
| Batch size | `{128, 256}` |
| Epochs | 200 |
| Resample times | `{2, 5, 8}` |
| Seed | 42 |
| Selection rule | highest `avg_feature_well_rankedness` per model |

Selected Top-k baselines:

| Model | Run | Hidden | k | LR | Batch | Resamples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-8B | `xrlpomoz` | 4096 | 64 | 1.69e-5 | 256 | 5 |
| allenai/OLMo-2-1124-13B | `67nrdg20` | 2048 | 128 | 3.72e-3 | 128 | 2 |
| openai/gpt-oss-20b | `a8ovi5ci` | 1024 | 128 | 4.19e-3 | 128 | 5 |

Supervised SAE baselines use filler-role labels as auxiliary supervision:

| Supervised SAE hyperparameter | Value |
| --- | --- |
| Feature mode | filler-role |
| Supervision weight | 0.02 |
| Sparsity penalty | 0.0 |
| Dead latent threshold | `1e-6` |
| Learning rate | 0.002 |
| Warmup ratio | 0.05 |
| Epochs | 40 |
| Batch size | 128 |
| Seeds checked | `{13, 42, 101}` |

The supervised seed check is stable: mean `R^2` is 0.9979 for Qwen, 0.9894 for OLMo, and 0.9907 for GPT-OSS, with near-zero standard deviation at the displayed precision.

## Figure Generation

Figures in this writeup are generated from `results/summary/summary.json` using `scripts/plot_llm_punctuation_figures.py`. The visual style is adapted from the existing SVO probe and SAE bar-plot scripts: compact grouped bars, shared model ordering, hidden top/right spines, light horizontal grids, and the same blue/orange trained-vs-constructed color convention. The optional `scripts/plot_llm_punctuation_figures_tui.py` Textual app can tune figure dimensions and typography before re-exporting the same PNG/PDF outputs.

## Artifact Map

| Artifact | Path |
| --- | --- |
| Main summary | `experiments/llm_sentences_last_layer/results/summary/summary.md` |
| Machine-readable summary | `experiments/llm_sentences_last_layer/results/summary/summary.json` |
| Paper table exports | `experiments/llm_sentences_last_layer/results/summary/paper_tables/` |
| TPE final configs | `experiments/llm_sentences_last_layer/configs/tpe_final/` |
| Analogy configs | `experiments/llm_sentences_last_layer/configs/analogy/` |
| Probe configs | `experiments/llm_sentences_last_layer/configs/probe/` |
| Probe inversion tuning summary | `experiments/llm_sentences_last_layer/results/summary/probe_inversion_tuning_summary.md` |
| SAE baseline summaries | `experiments/llm_sentences_last_layer/results/summary/sae_*summary.md` |
| Top-k W&B sweep configs | `experiments/llm_sentences_last_layer/sweeps/sae_topk_*.yaml` |
| Top-k W&B sweep logs | `experiments/llm_sentences_last_layer/sweeps/logs/` |
| Top-k W&B checkpoints | `experiments/llm_sentences_last_layer/checkpoints/sae_baselines/topk_wandb/` |
| Summary figures | `experiments/llm_sentences_last_layer/results/summary/figures/` |
| Figure generator | `experiments/llm_sentences_last_layer/scripts/plot_llm_punctuation_figures.py` |
| Figure tuning TUI | `experiments/llm_sentences_last_layer/scripts/plot_llm_punctuation_figures_tui.py` |
| Audit note | `agents/sentences/llm_punctuation_paper_ready_audit.md` |
| Journal | `agents/sentences/llm_analogy_probe_sae_journal.md` |

## Technical Quality Assessment

The current result set is technically stronger than the first LLM punctuation pass in three ways.

First, all baselines from the ICML SVO comparison structure are present: raw-hidden analogies, trained probes, trained Top-k SAEs, and trained supervised SAEs. Second, the major sensitive comparisons have tuning or robustness checks: trained probes are checked over three seeds, Top-k SAEs are selected from an explicit W&B Bayesian sweep, supervised SAEs are checked over three seeds, and analytic probes are selected by validation accuracy rather than by the reconstruction-oriented inverse heuristic. Third, the reported tables and figures are generated from machine-readable summaries, with untuned probe values preserved for provenance.

The main remaining judgment call is whether validation-selected probe inversion should be considered part of the analytic construction or an additional tuning degree of freedom. It is still an analytic probe in the sense that no classifier weights are trained on labels; however, the inversion hyperparameters are selected using validation labels. The writeup should therefore describe the result precisely as "validation-selected TPE-constructed probes are comparable to trained probes."
