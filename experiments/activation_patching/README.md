# Baseline activation patching

Run token-level activation patching on SVO prompts with:

```
uv run experiments/activation_patching/activation_patching.py experiments/activation_patching/configs/activation_patching_sentences.gin
```

Outputs mirror the original script: `token_heatmap.(npy|csv|html)`, `trial_summaries.jsonl`, and `summary.json` under the configured `output_dir`.
