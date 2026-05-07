import os
import json
import gin
import torch
from hypothesis import given, strategies as st
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import digits

@given(st.lists(st.integers(min_value=0, max_value=9)))
def test_interleaved_preserves_elements(seq):
    result = digits.interleaved(seq)
    assert sorted(result) == sorted(seq)
    assert len(result) == len(seq)

@given(st.lists(st.integers(min_value=0, max_value=9)))
def test_transform_copy(seq):
    assert digits.transform(seq, "copy") == seq
    assert digits.transform(seq, "reverse") == list(reversed(seq))
    assert digits.transform(seq, "sort_ascending") == sorted(seq)
    assert digits.transform(seq, "sort_descending") == sorted(seq)[::-1]

def test_generate_examples_unique():
    gin.clear_config()
    gin.bind_parameter('transform.task', 'copy')
    examples = digits.generate_examples(1, 3, 5, 5)
    assert len(examples) == 5
    seen = set()
    for inp, tgt in examples:
        assert tuple(inp) not in seen
        seen.add(tuple(inp))


def test_generate_examples_with_filler_role_holdout():
    gin.clear_config()
    gin.bind_parameter('transform.task', 'copy')
    gin.bind_parameter('generate_examples.min_seq_length', 3)
    gin.bind_parameter('generate_examples.max_seq_length', 3)
    gin.bind_parameter('generate_examples.vocab_size', 5)

    holdout_pairs = [(2, 1), (3, 3)]
    splits = digits.generate_examples_with_filler_role_holdout(
        n_train=8,
        n_valid=3,
        n_test=4,
        n_generalization=5,
        filler_role_pairs=holdout_pairs,
    )

    assert set(splits.keys()) == {"train", "valid", "test", "generalization"}
    assert len(splits["train"]) == 8
    assert len(splits["valid"]) == 3
    assert len(splits["test"]) == 4
    assert len(splits["generalization"]) == 5

    in_distribution = splits["train"] + splits["valid"] + splits["test"]
    assert all(
        not digits.contains_filler_role_pair(seq, holdout_pairs)
        for seq, _ in in_distribution
    )
    assert all(
        digits.contains_filler_role_pair(seq, holdout_pairs)
        for seq, _ in splits["generalization"]
    )

    all_sequences = [tuple(seq) for split in splits.values() for seq, _ in split]
    assert len(all_sequences) == len(set(all_sequences))

@given(st.lists(st.integers(min_value=1, max_value=9), min_size=1, max_size=5))
def test_get_roles_l2r(seq):
    ids = torch.tensor([seq])
    attn = torch.ones_like(ids)
    fillers, roles = digits.get_roles(ids, attn, role_scheme="l2r")
    assert fillers.shape == ids.shape
    assert roles.shape == ids.shape
    assert (roles[0, :len(seq)] == torch.arange(1, len(seq)+1)).all()


def test_get_roles_content_ignores_special_tokens():
    # 1=<bos>, 3=<sep>, 0=<pad>; content digits are assigned positions 1..6.
    ids = torch.tensor([[1, 9, 8, 7, 6, 5, 4, 3]])
    attn = torch.ones_like(ids)
    special_token_ids = [0, 1, 2, 3]  # pad, bos, eos, sep

    _, l2r_roles = digits.get_roles(
        ids,
        attn,
        role_scheme="l2r_content",
        pad_token_id=0,
        special_token_ids=special_token_ids,
    )
    _, r2l_roles = digits.get_roles(
        ids,
        attn,
        role_scheme="r2l_content",
        pad_token_id=0,
        special_token_ids=special_token_ids,
    )

    assert torch.equal(l2r_roles[0], torch.tensor([0, 1, 2, 3, 4, 5, 6, 0]))
    assert torch.equal(r2l_roles[0], torch.tensor([0, 6, 5, 4, 3, 2, 1, 0]))


def test_generate_digits_script(tmp_path):
    gin.clear_config()
    gin.bind_parameter('transform.task', 'copy')
    out_dir = tmp_path / "data"
    from generate_digits import main as gen_main
    import digits as digits_module
    import generate_digits as gen_mod
    def patched_gen_examples(*args, **kwargs):
        return digits_module.generate_examples.__wrapped__(1,2,5,*args, **kwargs)
    gen_mod.generate_examples = patched_gen_examples
    orig_transform = digits_module.transform.__wrapped__
    def patched_transform(seq):
        return orig_transform(seq, "copy")
    digits_module.transform = patched_transform
    gen_main.__wrapped__(str(out_dir)+"/", "test", 1, 1, 1, 1)
    # check files created
    assert (out_dir/"test.train").exists()


def test_generate_digits_script_with_holdout(tmp_path):
    gin.clear_config()
    gin.bind_parameter('transform.task', 'copy')
    gin.bind_parameter('generate_examples.min_seq_length', 3)
    gin.bind_parameter('generate_examples.max_seq_length', 3)
    gin.bind_parameter('generate_examples.vocab_size', 5)

    out_dir = tmp_path / "data"
    from generate_digits import main as gen_main

    gen_main.__wrapped__(
        str(out_dir) + "/",
        "holdout_test",
        8,
        2,
        2,
        1,
        n_generalization=3,
        holdout_pairs=[(2, 1)],
    )

    assert (out_dir / "holdout_test.train").exists()
    assert (out_dir / "holdout_test.valid").exists()
    assert (out_dir / "holdout_test.test").exists()
    assert (out_dir / "holdout_test.generalization").exists()
    assert (out_dir / "holdout_test.holdout_metadata.json").exists()

    with open(out_dir / "holdout_test.holdout_metadata.json", "r") as f:
        metadata = json.load(f)
    with open(out_dir / "holdout_test.dataset_creation_args.gin", "r") as f:
        config_text = f.read()

    assert metadata["examples_with_holdout_pair"]["train"] == 0
    assert metadata["examples_with_holdout_pair"]["valid"] == 0
    assert metadata["examples_with_holdout_pair"]["test"] == 0
    assert metadata["examples_with_holdout_pair"]["generalization"] == 3
    assert "generate_examples.max_seq_length = 3" in config_text
    assert "generate_examples.min_seq_length = 3" in config_text
    assert "generate_examples.vocab_size = 5" in config_text
