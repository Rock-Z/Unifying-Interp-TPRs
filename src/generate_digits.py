import os
import random
import json
import gin
import numpy as np
from random import shuffle

from digits import (
    contains_filler_role_pair,
    generate_examples,
    generate_examples_with_filler_role_holdout,
    pairs_to_file,
)
from utils import parse_args_for_gin


def _pair_key(filler: int, position: int) -> str:
    return f"{int(filler)}@{int(position)}"


def _build_holdout_metadata(
    splits: dict[str, list[tuple[tuple[int, ...], list[int]]]],
    holdout_pairs: list[tuple[int, int]],
) -> dict:
    """Summarize held-out pair coverage across generated splits."""
    metadata = {
        "holdout_pairs": [
            {"filler": int(filler), "position": int(position)}
            for filler, position in holdout_pairs
        ],
        "split_sizes": {split_name: len(split_examples) for split_name, split_examples in splits.items()},
        "examples_with_holdout_pair": {},
        "heldout_pair_occurrences": {},
    }

    for split_name, split_examples in splits.items():
        sequences = [tuple(int(x) for x in seq) for seq, _ in split_examples]
        metadata["examples_with_holdout_pair"][split_name] = int(
            sum(contains_filler_role_pair(seq, holdout_pairs) for seq in sequences)
        )
        pair_counts = {}
        for filler, position in holdout_pairs:
            pair_counts[_pair_key(filler, position)] = int(
                sum(
                    len(seq) >= int(position) and int(seq[int(position) - 1]) == int(filler)
                    for seq in sequences
                )
            )
        metadata["heldout_pair_occurrences"][split_name] = pair_counts

    return metadata


@gin.configurable
def main(data_dir : str, 
         prefix : str,
         n_train : int,
         n_valid : int,
         n_test : int,
         random_seed : int | None = None,
         n_generalization : int = 0,
         holdout_pairs : list[tuple[int, int]] | None = None):

    if random_seed is None:
        random_seed = random.randint(0,1000)
    random.seed(random_seed)
    np.random.seed(random_seed)

    use_holdout_split = holdout_pairs is not None or n_generalization > 0
    if use_holdout_split and (not holdout_pairs or n_generalization <= 0):
        raise ValueError(
            "Holdout generation requires both holdout_pairs and a positive n_generalization."
        )

    if use_holdout_split:
        splits = generate_examples_with_filler_role_holdout(
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            n_generalization=n_generalization,
            filler_role_pairs=holdout_pairs,
        )
        train_set = splits["train"]
        valid_set = splits["valid"]
        test_set = splits["test"]
        generalization_set = splits["generalization"]
    else:
        # We generate the train, validation, and test examples all at once like
        # this so that there won't be any differences in length distribution
        # across the three sets
        iid_examples = generate_examples(num_examples_needed=n_train+n_valid+n_test, ood=False)
        shuffle(iid_examples)
        train_set = iid_examples[:n_train]
        valid_set = iid_examples[n_train:n_train + n_valid]
        test_set = iid_examples[n_train + n_valid:]
        generalization_set = []

    # Generate dirs if do not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the sequences to files
    prefix_root = os.path.join(data_dir, prefix)
    if n_train > 0:
        pairs_to_file(train_set, prefix_root + ".train")
    if n_valid > 0: 
        pairs_to_file(valid_set, prefix_root + ".valid")
    if n_test > 0:
        pairs_to_file(test_set, prefix_root + ".test")
    if use_holdout_split and n_generalization > 0:
        pairs_to_file(generalization_set, prefix_root + ".generalization")

    with open(prefix_root + ".dataset_creation_args.gin", 'w') as fo:
        fo.write(gin.operative_config_str())
    if use_holdout_split and holdout_pairs is not None:
        metadata = _build_holdout_metadata(
            {
                "train": train_set,
                "valid": valid_set,
                "test": test_set,
                "generalization": generalization_set,
            },
            holdout_pairs=holdout_pairs,
        )
        with open(prefix_root + ".holdout_metadata.json", "w") as fo:
            json.dump(metadata, fo, indent=2)

if __name__ == "__main__":
    parse_args_for_gin()
    main()
