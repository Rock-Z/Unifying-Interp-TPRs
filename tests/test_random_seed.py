import random
import numpy as np
import torch
import json
import subprocess
from pathlib import Path
import shutil
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import set_random_seed
from generate_sentences import generate_sentences, split_data as split_simple, save_sentences as save_simple_sentences, save_word_lists as save_word_lists_simple
from sentences import load_sentences
from transformers import TrainingArguments


def test_set_random_seed_basic():
    set_random_seed(123)
    val1 = random.random()
    nval1 = np.random.rand()
    tval1 = torch.rand(1)

    set_random_seed(123)
    assert random.random() == val1
    assert np.random.rand() == nval1
    assert torch.allclose(torch.rand(1), tval1)


@pytest.mark.slow
def test_train_sentences_seed_reproducibility(tmp_path):
    # Generate a tiny dataset
    nouns = ["a", "b", "c"]
    sentences = generate_sentences(nouns, ["see"])
    train, valid, test = split_simple(sentences)
    sentences_root = tmp_path / "sentences"
    sentences_root.mkdir(parents=True, exist_ok=True)
    prefix = str(sentences_root / "data")
    save_simple_sentences(prefix, train, valid, test)
    save_word_lists_simple(prefix, nouns, ["see"])

    def _dummy_embeddings(dataset, *args, **kwargs):
        dim = 4
        for split in dataset:
            dataset[split] = dataset[split].add_column(
                "target_embeddings", [[0.0] * dim] * len(dataset[split])
            )
        return dataset, dim

    import train_sentences as train_mod
    train_mod.load_dataset_with_embeddings = _dummy_embeddings

    def _dummy_add_labels_for_role(ds_split, role_id, allow_missing=True):
        if "labels" in ds_split.column_names:
            ds_split = ds_split.remove_columns("labels")
        return ds_split.add_column("labels", [0] * len(ds_split))

    train_mod.add_labels_for_role = _dummy_add_labels_for_role

    def run_once(seed):
        output_dir = tmp_path / f"ckpt_{seed}"
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            seed=seed,
        )
        train_mod.main(
            sentences_path=str(sentences_root) + "/",
            embedding_model_name="dummy",
            embedding_cache_path=str(tmp_path),
            tpe_config={"filler_dim": 4, "role_dim": 4, "n_roles": 3, "hidden_size": 4},
            tpe_training_args=args,
            skip_trainable_probe=True,
            skip_analytic_probe=True,
            random_seed=seed,
            dataset_loader=load_sentences,
        )
        with open(output_dir / "eval_results_tpe.json") as f:
            return json.load(f)

    metrics1 = run_once(42)
    metrics2 = run_once(42)
    assert metrics1 == metrics2

