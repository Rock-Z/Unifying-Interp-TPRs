from typing import Literal, Tuple, Optional, Sequence
from datasets import Dataset, DatasetDict
import numpy as np
import gin

from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

import torch

# The task of interleaving a sequence
# E.g., interleaved([1,2,3,4,5,6]) = [1,6,2,5,3,4]
def interleaved(sequence, start_right=False):
    if len(sequence) <= 1:
        return list(sequence)
    else:
        if start_right:
            return [sequence[-1], sequence[0]] + interleaved(sequence[1:-1], start_right=start_right)
        else:
            return [sequence[0], sequence[-1]] + interleaved(sequence[1:-1], start_right=start_right)

# Mapping an input sequence to the output
# predicted by the task
@gin.configurable
def transform(sequence, task):
    if task == "copy":
        return sequence
    if task == "reverse":
        return sequence[::-1]
    if task == "sort_ascending":
        return sorted(sequence)
    if task == "sort_descending":
        return sorted(sequence)[::-1]
    if task == "interleave":
        return interleaved(sequence)
    if task == "interleave_right":
        return interleaved(sequence, start_right=True)

        # Creates a list of num_examples_needed examples
# Each example consists of a sequence of digits of
# length seq_length, where each digit is randomly
# drawn from 0 to (vocab_size - 1)
# If ood is False, all examples must be in-distribution
# Else, all examples must be out-of-distribution
# All examples must be unique
@gin.configurable
def generate_examples(min_seq_length, max_seq_length, vocab_size, num_examples_needed, ood=False, doubly_ood=False):

    list_examples = []
    dict_examples = {}

    num_examples = 0
    while num_examples < num_examples_needed:
        seq_length = min_seq_length + np.random.randint(max_seq_length - min_seq_length + 1)
        seq = tuple(np.random.randint(vocab_size,size=seq_length))
        if seq not in dict_examples:
            list_examples.append((seq, transform(seq))) 
            dict_examples[seq] = 1
            num_examples += 1

    return list_examples


def contains_filler_role_pair(
    sequence: Sequence[int],
    filler_role_pairs: Sequence[tuple[int, int]],
) -> bool:
    """Check whether a sequence contains any held-out filler-position pair.

    Args:
        sequence: Raw digit sequence without special tokens.
        filler_role_pairs: Held-out `(filler, position)` pairs where positions
            are 1-indexed over content digits from left to right.

    Returns:
        True if any held-out pair occurs in the sequence.

    Example:
        >>> contains_filler_role_pair((2, 4, 6), [(2, 1), (9, 2)])
        True
    """
    held_out = {(int(filler), int(position)) for filler, position in filler_role_pairs}
    for position, filler in enumerate(sequence, start=1):
        if (int(filler), int(position)) in held_out:
            return True
    return False


def generate_examples_with_filler_role_holdout(
    n_train: int,
    n_valid: int,
    n_test: int,
    n_generalization: int,
    filler_role_pairs: Sequence[tuple[int, int]],
    max_attempts_multiplier: int = 100,
) -> dict[str, list[tuple[tuple[int, ...], list[int]]]]:
    """Generate unique digits splits with held-out filler-position pairs.

    The train/valid/test splits exclude every held-out `(filler, position)`
    pair, while every example in the generalization split includes at least one
    held-out pair. Positions are defined over the raw content sequence and are
    1-indexed from left to right.

    Args:
        n_train: Number of train examples.
        n_valid: Number of validation examples.
        n_test: Number of evaluation examples.
        n_generalization: Number of held-out generalization examples.
        filler_role_pairs: Held-out `(filler, position)` pairs.
        max_attempts_multiplier: Safety factor that bounds rejection sampling.

    Returns:
        Dict mapping split names to lists of `(input_seq, target_seq)` pairs.

    Example:
        >>> splits = generate_examples_with_filler_role_holdout(
        ...     n_train=4,
        ...     n_valid=1,
        ...     n_test=1,
        ...     n_generalization=2,
        ...     filler_role_pairs=[(2, 1)],
        ... )
        >>> sorted(splits.keys())
        ['generalization', 'test', 'train', 'valid']
    """
    if n_generalization < 0:
        raise ValueError("n_generalization must be non-negative")
    if not filler_role_pairs:
        raise ValueError("filler_role_pairs must be provided when using holdout generation")

    normalized_pairs = [(int(filler), int(position)) for filler, position in filler_role_pairs]
    if any(position <= 0 for _, position in normalized_pairs):
        raise ValueError("Held-out positions must be 1-indexed positive integers")

    in_distribution_examples: list[tuple[tuple[int, ...], list[int]]] = []
    generalization_examples: list[tuple[tuple[int, ...], list[int]]] = []
    seen_sequences: set[tuple[int, ...]] = set()

    num_in_distribution_needed = n_train + n_valid + n_test
    num_generalization_needed = n_generalization
    total_needed = num_in_distribution_needed + num_generalization_needed
    max_attempts = max(total_needed * max_attempts_multiplier, 1)
    batch_size = min(max(total_needed, 1), 1024)

    attempts = 0
    while (
        len(in_distribution_examples) < num_in_distribution_needed
        or len(generalization_examples) < num_generalization_needed
    ):
        candidate_examples = generate_examples(num_examples_needed=batch_size, ood=False)
        attempts += len(candidate_examples)
        if attempts > max_attempts:
            raise RuntimeError(
                "Could not generate enough unique examples for the requested held-out split sizes."
            )
        for seq, target in candidate_examples:
            seq = tuple(int(x) for x in seq)
            if seq in seen_sequences:
                continue

            has_held_out_pair = contains_filler_role_pair(seq, normalized_pairs)
            if has_held_out_pair:
                if len(generalization_examples) >= num_generalization_needed:
                    continue
                generalization_examples.append((seq, target))
            else:
                if len(in_distribution_examples) >= num_in_distribution_needed:
                    continue
                in_distribution_examples.append((seq, target))
            seen_sequences.add(seq)

            if (
                len(in_distribution_examples) >= num_in_distribution_needed
                and len(generalization_examples) >= num_generalization_needed
            ):
                break

    np.random.shuffle(in_distribution_examples)
    np.random.shuffle(generalization_examples)

    train_set = in_distribution_examples[:n_train]
    valid_set = in_distribution_examples[n_train:n_train + n_valid]
    test_set = in_distribution_examples[n_train + n_valid:]

    return {
        "train": train_set,
        "valid": valid_set,
        "test": test_set,
        "generalization": generalization_examples,
    }


def pairs_to_file(pairs, fo):
    """
    Writes pairs of sequences to a file in a tab-separated format.
    Args:
        pairs (list of tuples): A list where each element is a tuple containing two sequences.
        fo (str): The file path where the output will be written.
    The output file will have two columns: 'input_seq' and 'target_seq', with each pair of sequences
    written on a new line, separated by a tab.
    """

    with open(fo, "w") as file_obj:
        file_obj.write("input_seq\ttarget_seq\n")
        for pair in pairs:
            file_obj.write(
                " ".join(str(x) for x in pair[0]) +
                "\t" +
                " ".join(str(x) for x in pair[1]) +
                "\n"
            )


# Optimized implementation
def get_roles(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_scheme: Literal["l2r", "r2l", "bow", "bidirectional", "l2r_content", "r2l_content"],
        padding_side: Literal["left", "right"] = "right",
        pad_token_id: Optional[int] = 0,
        special_token_ids: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple of (filler ids, role ids) based on the specified role scheme and input ids.
    """

    if role_scheme == "bow":
        roles = torch.zeros_like(input_ids)
        roles[attention_mask != 0] = 1
        fillers = input_ids.clone()
    elif role_scheme in ["l2r", "r2l"]:
        batch_size, seq_len = input_ids.shape
        seq_lengths = attention_mask.sum(dim=1)
        roles = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
        positions = torch.arange(1, seq_len + 1).expand(batch_size, seq_len)
        if padding_side == "right":
            mask = positions < (seq_lengths.unsqueeze(1) + 1)
            if role_scheme == "l2r":
                role_values = positions
            else:  # r2l
                role_values = (seq_lengths.unsqueeze(1) + 1) - positions
        else:  # padding_side == "left"
            starting_indices = seq_len - seq_lengths.unsqueeze(1)
            mask = positions >= starting_indices
            if role_scheme == "l2r":
                role_values = positions - starting_indices
            else:  # r2l
                role_values = (seq_lengths.unsqueeze(1) + 1) - (positions - starting_indices)

        roles = torch.where(mask, role_values.to(roles.dtype), roles)
        fillers = input_ids.clone()
    elif role_scheme in ["l2r_content", "r2l_content"]:
        # Content-only positional roles: assign positions only to non-special,
        # non-padding tokens. Special tokens receive the padding role id.
        roles = torch.full_like(input_ids, pad_token_id)
        fillers = input_ids.clone()
        valid_mask = attention_mask != 0
        non_special_mask = valid_mask.clone()
        if special_token_ids:
            for token_id in special_token_ids:
                if token_id is not None:
                    non_special_mask &= input_ids != int(token_id)

        content_ranks = torch.cumsum(non_special_mask.to(torch.long), dim=1)
        if role_scheme == "l2r_content":
            role_values = content_ranks
        else:
            content_lengths = non_special_mask.sum(dim=1, keepdim=True)
            role_values = content_lengths - content_ranks + 1

        roles = torch.where(non_special_mask, role_values.to(roles.dtype), roles)
    elif role_scheme == "bidirectional":
        l2r_fillers, l2r_roles = get_roles(
            input_ids,
            attention_mask,
            role_scheme="l2r",
            padding_side=padding_side,
            pad_token_id=pad_token_id
            )
        r2l_fillers, r2l_roles = get_roles(
            input_ids,
            attention_mask,
            role_scheme="r2l",
            padding_side=padding_side,
            pad_token_id=pad_token_id
            )

        # left-to-right roles and right-to-left roles need
        # to be offset by the sequence length so they're distinct
        r2l_roles[r2l_roles != pad_token_id] += input_ids.shape[-1]

        roles = torch.concat([l2r_roles, r2l_roles], dim=-1)
        fillers = torch.concat([l2r_fillers, r2l_fillers], dim=-1)
    else:
        raise ValueError(f"Unknown role scheme: {role_scheme}")

    return fillers, roles


def load_digits(file_paths: dict,
                pad_token: str = "<pad>",
                bos_token: str = "<bos>",
                sep_token: str = "<sep>",
                eos_token: str = "<eos>",
                add_special_tokens: bool = True,
                ) -> Tuple[DatasetDict, PreTrainedTokenizerFast]:

    dataset_splits = {}
    unique_digits = set()

    for split_name, split_path in file_paths.items():
        inputs = []
        labels = []

        # load data from file
        with open(split_path, "r") as f:
            # read first line and assert format
            header = f.readline().strip()
            assert header == "input_seq\ttarget_seq", f"Expected header 'input\tlabel' but got {header}"

            # parse data
            for line in f:
                input, label = line.strip().split("\t")
                # update digits set
                unique_digits.update(input.split())
                unique_digits.update(label.split())
                # remove whitespace between digits
                if add_special_tokens:
                    inputs.append(bos_token + input + " " + sep_token)
                    labels.append(label + " " + eos_token)
                else:
                    inputs.append(input + " ")
                    labels.append(label + " ")


        dataset_splits[split_name] = Dataset.from_dict({"input": inputs,"label": labels})

    # create a tokenizer for the datasets
    tokenizer = Tokenizer(BPE())
    def _sort_key(token: str) -> tuple[int, object]:
        if token.isdigit():
            return (0, int(token))
        return (1, token)
    tokenizer.add_tokens(sorted(unique_digits, key=_sort_key))
    # add special tokens
    tokenizer.add_special_tokens([pad_token, bos_token, eos_token, sep_token])

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                        bos_token=bos_token,
                                        eos_token=eos_token,
                                        sep_token=sep_token,
                                        pad_token=pad_token,)

    return DatasetDict(dataset_splits), tokenizer


def tokenize_function(
    examples: list,  # Changed from dict to list of dicts
    tokenizer,
    format: Literal["seq2seq", "tpe", "tpe_eval"] = "seq2seq",
    role_scheme: Optional[Literal["l2r", "r2l", "bow", "bidirectional", "l2r_content", "r2l_content"]] = None,
):
    """
    This function replaces both the tokenizer and the data collator.
    It takes a list of examples, tokenizes them, and returns a dict
    suitable for feeding to the model.
    """
    # Extract batch inputs/labels from list of examples
    inputs = [ex["input"] for ex in examples]
    labels = [ex["label"] for ex in examples]

    tokenized = {}

    # Common tokenizer parameters
    tokenizer_args = {
        "padding": "longest",
        "return_token_type_ids": False,
        "return_tensors": "pt",
    }

    # Batch tokenization
    tokenized_input = tokenizer(inputs, **tokenizer_args)
    tokenized_labels = tokenizer(labels, **tokenizer_args)

    if format == "seq2seq":
        tokenized.update({
            "input_ids": tokenized_input["input_ids"],
            "input_lengths": torch.sum(tokenized_input["attention_mask"], dim=1),
            "labels": tokenized_labels["input_ids"],
        })

    # TPE variants handling
    if format in {"tpe", "tpe_eval"}:
        if role_scheme is None:
            raise ValueError("role_scheme must be provided when format is 'tpe' or 'tpe_eval'")
        input_ids = tokenized_input["input_ids"]
        special_token_ids = [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
            tokenizer.eos_token_id,
        ]
        filler_ids, role_ids = get_roles(
            input_ids,
            tokenized_input["attention_mask"].clone(),
            role_scheme=role_scheme,
            pad_token_id=0,
            special_token_ids=special_token_ids,
        )
        tokenized.update({
            "filler_ids": filler_ids,
            "role_ids": role_ids,
        })

        if format == "tpe":
            tokenized.update({
                "embedding_model_input_ids": input_ids,
                "embedding_model_input_lengths": torch.sum(tokenized_input["attention_mask"], dim=1),
            })

        if format == "tpe_eval":
            tokenized["labels"] = tokenized_labels["input_ids"]

    return tokenized
