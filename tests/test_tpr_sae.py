from pathlib import Path
import sys

import pytest
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpr_sae import find_observed_filler_role_pairs, filter_filler_role_pairs_by_presence


def test_find_observed_filler_role_pairs_returns_unique_sorted_pairs():
    split = Dataset.from_dict(
        {
            "filler_ids": [[3, 7, 7], [1, 3]],
            "role_ids": [[2, 1, 1], [0, 2]],
        }
    )
    pairs = find_observed_filler_role_pairs(split)
    assert pairs == [(1, 0), (7, 1), (3, 2)]


def test_find_observed_filler_role_pairs_requires_columns():
    split = Dataset.from_dict({"x": [1], "y": [2]})
    with pytest.raises(ValueError):
        find_observed_filler_role_pairs(split)


def test_filter_filler_role_pairs_by_presence_removes_always_on_pairs():
    split = Dataset.from_dict(
        {
            "filler_ids": [[10, 1, 2], [10, 3, 4], [10, 5, 6]],
            "role_ids": [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        }
    )
    pairs = find_observed_filler_role_pairs(split)
    kept, removed = filter_filler_role_pairs_by_presence(split, pairs, max_presence=1.0)
    assert (10, 0) in removed
    assert (10, 0) not in kept
    assert len(kept) + len(removed) == len(pairs)
