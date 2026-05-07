# Digits filler-role holdout

This is the in-place holdout rerun for the fixed-length-6 L2R-aligned digits experiments. It now covers all three recurrent backbones (`rnn`, `gru`, `lstm`) for both `copy` and `reverse`, and it supersedes the earlier mixed-role holdout run.

## Setup

- Frozen seq2seq checkpoints reused from `experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned/checkpoints/seq2seq/{copy,reverse}/{rnn,gru,lstm}`
- Heldout TPEs follow the aligned role-64 family: `hidden_size = 256` for `rnn` and `gru`, `hidden_size = 512` for `lstm`, `filler_dim = 64`, `role_dim = 64`, `role_scheme = l2r`
- Data prefixes:
  - `copy_vocab_20_length_6_fixed_filler_role_holdout`
  - `reverse_vocab_20_length_6_fixed_filler_role_holdout`
- Held-out filler-position pairs: `(2,1)`, `(7,2)`, `(13,3)`, `(4,4)`, `(18,5)`, `(10,6)`
- Split sizes: `train=40000`, `valid=5000`, `test=5000`, `generalization=5000`
- Holdout coverage: `train/valid/test` contain zero held-out pairs, and all `5000` generalization examples contain at least one

## Run

```bash
bash experiments/digits/filler_role_holdout/run_experiment.sh
sbatch experiments/digits/filler_role_holdout/run_experiment.sbatch
```

## Outputs

- TPE checkpoints: `experiments/digits/filler_role_holdout/checkpoints/tpe/{copy,reverse}_{rnn,gru,lstm}/best_model`
- Split metrics: `experiments/digits/filler_role_holdout/checkpoints/tpe/{copy,reverse}_{rnn,gru,lstm}/holdout_eval_metrics.json`
- Dataset metadata: `data/digits/*_filler_role_holdout.holdout_metadata.json`
- Probe results: `experiments/digits_probe/results/filler_role_holdout/{copy,reverse}_{rnn,gru,lstm}/probe_compare_results.json`
- Analogy results: `experiments/analogy_digits/results/filler_role_holdout/digits_{copy,reverse}_{rnn,gru,lstm}_eval.json`

## Approximation Results

| Task | Arch | Test seq acc | Test TPE seq acc | Test R2 | Gen seq acc | Gen TPE seq acc | Gen R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| copy | rnn | `1.0000` | `1.0000` | `0.9863` | `1.0000` | `1.0000` | `0.9835` |
| copy | gru | `1.0000` | `0.9994` | `0.9600` | `1.0000` | `0.8230` | `0.9005` |
| copy | lstm | `1.0000` | `1.0000` | `0.9949` | `1.0000` | `0.9634` | `0.9882` |
| reverse | rnn | `0.9998` | `1.0000` | `0.8344` | `1.0000` | `1.0000` | `0.8238` |
| reverse | gru | `1.0000` | `0.9998` | `0.9497` | `1.0000` | `0.9098` | `0.9000` |
| reverse | lstm | `1.0000` | `0.9994` | `0.9888` | `1.0000` | `0.9464` | `0.9758` |

## Inversion Results

### Probes

| Task | Arch | Test analytic mean | Test trained mean | Heldout analytic mean | Heldout trained mean |
|---|---|---:|---:|---:|---:|
| copy | rnn | `0.9585` | `0.9990` | `0.6774` | `0.0000` |
| copy | gru | `0.9602` | `0.9851` | `0.0198` | `0.0000` |
| copy | lstm | `0.9001` | `0.9384` | `0.1016` | `0.0000` |
| reverse | rnn | `0.9999` | `1.0000` | `0.9982` | `0.0000` |
| reverse | gru | `0.8682` | `0.9045` | `0.1417` | `0.0000` |
| reverse | lstm | `0.8044` | `0.8464` | `0.1667` | `0.0000` |

Probe takeaway:
- The analytic probe derived from the heldout-trained TPE stays strong only for the `rnn` runs.
- `gru` and `lstm` collapse badly on the matched heldout probe splits even when their IID probe accuracy remains decent.
- The trained linear probe collapses to `0.0` on every matched heldout subset because those filler labels never appeared at that position during probe training.

### Probe Regularization Check

Using the same trained heldout TPEs, I re-ran the analytic probes with `main.role_pinv_l2_lambda=None` so digits would use the same ternary-search path as the sentence probes for role-unbinding regularization. Results were written to `experiments/digits_probe/results/filler_role_holdout_auto_role_pinv`.

| Task | Arch | Fixed `role_pinv_l2_lambda` | Auto-selected lambda | Fixed heldout analytic mean | Auto heldout analytic mean | Delta |
|---|---|---:|---:|---:|---:|---:|
| copy | rnn | `1.0e-2` | `2.59e-9` | `0.6774` | `0.6607` | `-0.0168` |
| copy | gru | `1.0e-2` | `4.65e-10` | `0.0198` | `0.0063` | `-0.0136` |
| copy | lstm | `1.0e-2` | `2.32e-10` | `0.1016` | `0.0061` | `-0.0955` |
| reverse | rnn | `1.0e-2` | `5.80e-11` | `0.9982` | `0.9705` | `-0.0277` |
| reverse | gru | `1.0e-2` | `9.30e-10` | `0.1417` | `0.0022` | `-0.1396` |
| reverse | lstm | `1.0e-2` | `5.80e-11` | `0.1667` | `0.0024` | `-0.1643` |

Regularization takeaway:
- This does not help the digits heldout probes; it hurts every architecture, sometimes catastrophically.
- The search objective in `src/probing.py` minimizes TPE-internal filler reconstruction MSE on a small IID train batch, not heldout probe accuracy.
- For these digits TPEs that objective consistently picks an almost unregularized role pseudoinverse (`1e-11` to `1e-9`), which removes the damping that the fixed `1e-2` setting was providing.

### Analogies

Quartet construction:
- Reuses the standard digits analogy generator unchanged.
- `test_iid_clean`: sampled from `test`, then filtered so none of `A/B/C/D` contain any held-out pair.
- `generalization_changed_holdout`: sampled from `generalization`, then filtered so at least one changed position contains the held-out pair on one side of the change.
- Each split keeps `1000` accepted quartets and uses the unique `A/B/C/D` strings from those quartets as its retrieval pool.

| Task | Arch | Split | NN Top-1 | NN Top-3 | TPE Top-1 | TPE Top-3 |
|---|---|---|---:|---:|---:|---:|
| copy | rnn | test_iid_clean | `1.000` | `1.000` | `1.000` | `1.000` |
| copy | rnn | generalization_changed_holdout | `1.000` | `1.000` | `1.000` | `1.000` |
| copy | gru | test_iid_clean | `1.000` | `1.000` | `1.000` | `1.000` |
| copy | gru | generalization_changed_holdout | `1.000` | `1.000` | `1.000` | `1.000` |
| copy | lstm | test_iid_clean | `1.000` | `1.000` | `1.000` | `1.000` |
| copy | lstm | generalization_changed_holdout | `1.000` | `1.000` | `1.000` | `1.000` |
| reverse | rnn | test_iid_clean | `0.949` | `1.000` | `0.977` | `1.000` |
| reverse | rnn | generalization_changed_holdout | `0.973` | `1.000` | `0.950` | `1.000` |
| reverse | gru | test_iid_clean | `1.000` | `1.000` | `1.000` | `1.000` |
| reverse | gru | generalization_changed_holdout | `1.000` | `1.000` | `1.000` | `1.000` |
| reverse | lstm | test_iid_clean | `1.000` | `1.000` | `1.000` | `1.000` |
| reverse | lstm | generalization_changed_holdout | `1.000` | `1.000` | `1.000` | `1.000` |

Analogy takeaway:
- Additive analogies remain completely intact for `copy` across all three backbones.
- `reverse_gru` and `reverse_lstm` are also perfect on the sampled quartet sets.
- `reverse_rnn` is the only case with noticeable Top-1 degradation, but Top-3 stays perfect on both splits.
