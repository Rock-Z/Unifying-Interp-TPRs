# Training Guide for TPR Experiments

This guide covers three main experiments: **sentences**, **paired sentences**, and **digits**. Use `uv` for running all scripts.

## Directory Structure

```
src/
├── train_sentences.py          # Main training script for sentences
├── train.py                    # Main training script for digits
├── paired_sentences.py         # Paired sentences dataset loader
├── sentences.py                # Sentences dataset loader  
├── digits.py                   # Digits dataset and training utilities
├── generate_pair_sentences.py  # Generate paired sentences data
├── generate_sentences.py       # Generate sentences data
├── generate_digits.py          # Generate digits data
├── invert_tpr.py               # TPR inversion and probing analysis
├── model.py                    # Tensor Product Encoder models
├── probing.py                  # Linear probing utilities
└── utils.py                    # Shared utilities

configs/
├── train_sentences.gin         # Sentences training config
├── train_paired_sentences_small.gin  # Paired sentences config
├── train_default.gin           # Default training parameters for digits
├── seq2seq_*.gin              # Individual digit task configs (copy, reverse, sort)
└── digit_invert_tpr_all_pos.gin      # TPR inversion config for digits

data/
├── sentences/                  # Sentences datasets and embeddings
├── pair_sentences/            # Paired sentences datasets
└── digits/                    # Digits datasets (copy, reverse, sort)
```

## Experiments Overview

### 1. Sentences Experiment
**Purpose**: Train Tensor Product Encoders on natural language sentences with subject-verb-object structure.

**Main Files**:
- `src/train_sentences.py` - Main training script with TPE and probing
- `src/sentences.py` - Dataset loader for sentence data
- `configs/train_sentences.gin` - Training configuration

**Data Structure**: Sentences with SVO structure (e.g., "The doctor will examine the patient")
- Roles: Subject, Verb, Object positions
- Fillers: Actual words in each position
- Includes pre-computed embeddings from various models

**Usage**:
```bash
uv run src/train_sentences.py configs/train_sentences.gin
```

### 2. Paired Sentences Experiment
**Purpose**: Train on paired occupation-verb sentences to study compositional generalization.

**Main Files**:
- `src/train_sentences.py` - Uses paired sentences loader
- `src/paired_sentences.py` - Specialized dataset loader
- `src/generate_pair_sentences.py` - Data generation script
- `configs/train_paired_sentences_small.gin` - Training config

**Data Structure**: Paired sentences (e.g., "The doctor will examine. The teacher will explain.")
- Roles: Verbs (examine, explain)
- Fillers: Occupations (doctor, teacher)
- Tests compositional generalization across occupation-verb pairs

**Usage**:
```bash
# Generate data first
uv run src/generate_pair_sentences.py --prefix pair_sentences/data

# Then train
uv run src/train_sentences.py configs/train_paired_sentences_small.gin
```

### 3. Digits Experiment
**Purpose**: Train on sequence-to-sequence tasks with digit sequences (copy, reverse, sort).

**Main Files**:
- `src/train.py` - Main training script for digit experiments
- `src/digits.py` - Dataset loader and training utilities
- `src/generate_digits.py` - Data generation script
- `src/invert_tpr.py` - TPR inversion and probing analysis
- `configs/seq2seq_*.gin` - Individual task training configs
- `configs/digit_invert_tpr_all_pos.gin` - TPR inversion config

**Data Structure**: Digit sequences with various transformations
- Tasks: copy, reverse, sort_ascending, sort_descending, interleave
- Input: Sequences of digits (vocab size 20, length 6)
- Output: Transformed sequences based on task
- Role schemes: left-to-right (l2r), right-to-left (r2l), bag-of-words (bow)

**Usage**:
```bash
# Generate data first
uv run src/generate_digits.py

# Train models for different tasks
uv run src/train.py configs/train_default.gin configs/seq2seq_copy.gin
uv run src/train.py configs/train_default.gin configs/seq2seq_reverse.gin
uv run src/train.py configs/train_default.gin configs/seq2seq_sort_ascending.gin

# Run TPR inversion analysis
uv run src/invert_tpr.py configs/digit_invert_tpr_all_pos.gin
```
