from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence, cast

import pandas as pd
from transformers import PreTrainedTokenizerBase

from sentences import is_single_token
from vocabulary import PASSIVE_PARTICIPLES


@dataclass
class TokenPatchTrial:
    """Container for a single clean/corrupted prompt pair."""

    clean_prompt: str
    corrupted_prompt: str
    correct_token_id: int
    competitor_token_id: int
    token_ids: list[int]
    token_strings: list[str]
    meta: dict[str, str]


def _render_prompt(subject: str, verb: str, dobj: str, *, vbn: str) -> str:
    """Build a two-clause prompt that ends right before the passive subject."""

    first = f"the {subject} will {verb} the {dobj}."
    second = f" the {dobj} will be {vbn} by the"
    return f"{first}{second}"


def build_trial(
    tokenizer: PreTrainedTokenizerBase,
    subject: str,
    verb: str,
    dobj: str,
    corrupted_subject: str,
) -> TokenPatchTrial:
    """Construct a token-level patching trial or raise on invalid inputs."""

    assert verb in PASSIVE_PARTICIPLES, f"Verb has no passive participle mapping: {verb!r}"
    vbn = PASSIVE_PARTICIPLES[verb]
    clean_prompt = _render_prompt(subject, verb, dobj, vbn=vbn)
    corrupted_prompt = _render_prompt(corrupted_subject, verb, dobj, vbn=vbn)

    correct_token = cast(Sequence[int], tokenizer(" " + subject, add_special_tokens=False)["input_ids"])
    competitor_token = cast(
        Sequence[int], tokenizer(" " + corrupted_subject, add_special_tokens=False)["input_ids"]
    )
    assert len(correct_token) == 1 and len(competitor_token) == 1, (
        "Subject or corrupted subject does not tokenize to a single token: "
        f"{subject!r} or {corrupted_subject!r}"
    )

    encoding = tokenizer(corrupted_prompt, add_special_tokens=False)
    token_ids = list(cast(Sequence[int], encoding["input_ids"]))
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore[attr-defined]

    return TokenPatchTrial(
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        correct_token_id=int(correct_token[0]),
        competitor_token_id=int(competitor_token[0]),
        token_ids=token_ids,
        token_strings=token_strings,
        meta={
            "subject": subject,
            "object": dobj,
            "verb": verb,
            "corrupted_subject": corrupted_subject,
        },
    )


def _parse_sentence(sentence: str) -> tuple[str, str, str]:
    s = sentence.strip()
    parts = s[:-1].strip().split()
    if len(parts) != 6:
        raise ValueError(f"Sentence does not match 6-token template: {sentence!r}")
    if parts[0] != "the" or parts[2] != "will" or parts[4] != "the":
        raise ValueError(
            f"Sentence template mismatch (expected 'the _ will _ the _'): {sentence!r}"
        )
    return parts[1], parts[3], parts[5]


def load_sentence_trials(
    tokenizer: PreTrainedTokenizerBase,
    sentences_path: str,
    max_trials: Optional[int],
    seed: int,
) -> list[TokenPatchTrial]:
    """Sample token-level trials from the SVO sentence list."""

    df = pd.read_csv(sentences_path)
    candidates = df["sentence"].dropna().tolist()
    rng = random.Random(seed)
    rng.shuffle(candidates)

    parsed_candidates: list[tuple[str, str, str]] = []
    for sentence in candidates:
        parsed_candidates.append(_parse_sentence(sentence))

    subject_pool = [
        subject
        for subject, _, dobj in parsed_candidates
        if subject != dobj and is_single_token(tokenizer, subject)
    ]

    trials: list[TokenPatchTrial] = []
    for subject, verb, dobj in parsed_candidates:
        if subject == dobj:
            # Skip: trivial swap would not create a meaningful contrast
            continue
        if not is_single_token(tokenizer, subject) or not is_single_token(tokenizer, dobj):
            # Skip: subject or object must be a single token for clean token-level attribution
            continue

        alternate_subjects = [
            alt for alt in subject_pool if alt != subject and alt != dobj
        ]
        if not alternate_subjects:
            continue
        corrupted_subject = rng.choice(alternate_subjects)

        trial = build_trial(tokenizer, subject, verb, dobj, corrupted_subject)
        trials.append(trial)
        if max_trials is not None and len(trials) >= max_trials:
            break
    return trials


__all__ = [
    "TokenPatchTrial",
    "build_trial",
    "load_sentence_trials",
]
