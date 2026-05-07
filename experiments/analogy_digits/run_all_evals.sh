#!/usr/bin/env bash

for f in experiments/analogy_digits/configs/*.gin; do
  PYTHONPATH=src uv run scripts/analogy/evaluate_digits_analogy.py "$f"
done
