#!/usr/bin/env bash
set -euo pipefail

# Regenerate the heldout datasets.
uv run src/generate_digits.py experiments/digits/filler_role_holdout/configs/data_copy.gin
uv run src/generate_digits.py experiments/digits/filler_role_holdout/configs/data_reverse.gin

# Train heldout TPEs and run holdout approximation evals.
uv run src/train.py experiments/digits/filler_role_holdout/configs/copy_rnn.gin --main.skip_seq2seq=True --tpe/TrainingArguments.overwrite_output_dir=True
uv run experiments/digits/filler_role_holdout/evaluate_holdout.py experiments/digits/filler_role_holdout/configs/copy_rnn.gin

uv run src/train.py experiments/digits/filler_role_holdout/configs/copy_gru.gin --main.skip_seq2seq=True --tpe/TrainingArguments.overwrite_output_dir=True
uv run experiments/digits/filler_role_holdout/evaluate_holdout.py experiments/digits/filler_role_holdout/configs/copy_gru.gin

uv run src/train.py experiments/digits/filler_role_holdout/configs/copy_lstm.gin --main.skip_seq2seq=True --tpe/TrainingArguments.overwrite_output_dir=True
uv run experiments/digits/filler_role_holdout/evaluate_holdout.py experiments/digits/filler_role_holdout/configs/copy_lstm.gin

uv run src/train.py experiments/digits/filler_role_holdout/configs/reverse_rnn.gin --main.skip_seq2seq=True --tpe/TrainingArguments.overwrite_output_dir=True
uv run experiments/digits/filler_role_holdout/evaluate_holdout.py experiments/digits/filler_role_holdout/configs/reverse_rnn.gin

uv run src/train.py experiments/digits/filler_role_holdout/configs/reverse_gru.gin --main.skip_seq2seq=True --tpe/TrainingArguments.overwrite_output_dir=True
uv run experiments/digits/filler_role_holdout/evaluate_holdout.py experiments/digits/filler_role_holdout/configs/reverse_gru.gin

uv run src/train.py experiments/digits/filler_role_holdout/configs/reverse_lstm.gin --main.skip_seq2seq=True --tpe/TrainingArguments.overwrite_output_dir=True
uv run experiments/digits/filler_role_holdout/evaluate_holdout.py experiments/digits/filler_role_holdout/configs/reverse_lstm.gin

# Run heldout probe comparisons.
uv run src/invert_tpr.py experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized/copy_rnn_probe_all_pos_holdout.gin --trainable/TrainingArguments.overwrite_output_dir=True
uv run src/invert_tpr.py experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized/copy_gru_probe_all_pos_holdout.gin --trainable/TrainingArguments.overwrite_output_dir=True
uv run src/invert_tpr.py experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized/copy_lstm_probe_all_pos_holdout.gin --trainable/TrainingArguments.overwrite_output_dir=True
uv run src/invert_tpr.py experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized/reverse_rnn_probe_all_pos_holdout.gin --trainable/TrainingArguments.overwrite_output_dir=True
uv run src/invert_tpr.py experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized/reverse_gru_probe_all_pos_holdout.gin --trainable/TrainingArguments.overwrite_output_dir=True
uv run src/invert_tpr.py experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized/reverse_lstm_probe_all_pos_holdout.gin --trainable/TrainingArguments.overwrite_output_dir=True

# Run heldout analogy evals.
PYTHONPATH=src uv run scripts/analogy/evaluate_digits_analogy.py experiments/analogy_digits/configs/1le64h256_fixedlen/digits_copy_rnn_holdout_eval.gin
PYTHONPATH=src uv run scripts/analogy/evaluate_digits_analogy.py experiments/analogy_digits/configs/1le64h256_fixedlen/digits_copy_gru_holdout_eval.gin
PYTHONPATH=src uv run scripts/analogy/evaluate_digits_analogy.py experiments/analogy_digits/configs/1le64h256_fixedlen/digits_copy_lstm_holdout_eval.gin
PYTHONPATH=src uv run scripts/analogy/evaluate_digits_analogy.py experiments/analogy_digits/configs/1le64h256_fixedlen/digits_reverse_rnn_holdout_eval.gin
PYTHONPATH=src uv run scripts/analogy/evaluate_digits_analogy.py experiments/analogy_digits/configs/1le64h256_fixedlen/digits_reverse_gru_holdout_eval.gin
PYTHONPATH=src uv run scripts/analogy/evaluate_digits_analogy.py experiments/analogy_digits/configs/1le64h256_fixedlen/digits_reverse_lstm_holdout_eval.gin
