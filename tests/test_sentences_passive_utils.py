import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import random

from datasets import Dataset, DatasetDict

from sentences import (
    build_active_passive_prompts_from_svo,
    build_active_passive_trial,
    filter_words_single_token,
    load_sentences,
    sample_trials,
)  # noqa: E402
from vocabulary import PASSIVE_PARTICIPLES, NOUNS_SG  # noqa: E402

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


def test_filter_single_token_words():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    filtered = filter_words_single_token(tok, ["astronaut", "professor"])
    assert all(filtered)
    assert all(tok(" " + w, add_special_tokens=False)["input_ids"] for w in filtered)


def test_build_active_passive_trial_fields():
    trial = build_active_passive_trial("astronaut", "professor", "see")
    assert trial["prompt"].startswith("the astronaut will see the professor.")
    assert "target_correct" in trial and "target_competitor" in trial


def test_sample_trials_uses_participles():
    nouns = NOUNS_SG[:3]
    verbs = list(PASSIVE_PARTICIPLES.keys())
    trials = sample_trials(nouns, verbs, n=None, seed=0)
    assert trials
    for t in trials:
        assert t["verb"] in PASSIVE_PARTICIPLES


def test_sample_trials_rejects_missing_participle():
    with pytest.raises(ValueError):
        sample_trials(["a"], ["verb_missing"], n=None, seed=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_build_active_passive_prompts_from_svo(seed):
    data_path = str(Path(__file__).resolve().parents[1] / "data" / "sentences")
    dataset, assigner = load_sentences(data_path, role_scheme="svo")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    filtered_nouns = filter_words_single_token(tok, assigner.nouns_sg)
    if len(filtered_nouns) < 2:
        pytest.skip("Tokenizer did not yield enough single-token nouns")
    rng = random.Random(seed)
    subj, obj = rng.sample(filtered_nouns, 2)
    verb = "see"
    subj_id = assigner.noun_filler2idx[subj]
    obj_id = assigner.noun_filler2idx[obj]
    verb_id = assigner.verb_filler2idx[verb] + len(assigner.nouns_sg)
    role_ids = (
        assigner.role2idx["subject"],
        assigner.role2idx["object"],
        assigner.role2idx["verb"],
    )
    prompt = f"the {subj} will {verb} the {obj}."
    ds = Dataset.from_dict(
        {
            "sentence": [prompt],
            "filler_ids": [(subj_id, obj_id, verb_id)],
            "role_ids": [role_ids],
        }
    )
    ds_dict = DatasetDict({"train": ds, "valid": ds, "test": ds})
    converted = build_active_passive_prompts_from_svo(
        dataset=ds_dict,
        role_assigner=assigner,
        tokenizer=tok,
        max_examples_per_split=None,
    )
    assert len(converted["train"]) == 1
    row = converted["train"][0]
    expected = f"the {subj} will {verb} the {obj}. the {obj} will be {PASSIVE_PARTICIPLES[verb]} by the {subj}"
    assert row["sentence"] == expected
    assert tuple(row["filler_ids"]) == (subj_id, obj_id, verb_id)
    assert tuple(row["role_ids"]) == role_ids
