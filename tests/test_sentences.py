import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_sentences import (
    extract_subject_and_object,
    generate_sentences,
    save_split,
    save_word_lists,
    split_data,
    split_data_with_holdout,
)
from sentences import load_sentences

def test_generate_sentences_count():
    nouns = ["a", "b"]
    verbs = ["v"]
    sentences = generate_sentences(nouns, verbs)
    assert len(sentences) == len(nouns)**2 * len(verbs)


def test_split_data_sizes():
    sentences = [f"s{i}" for i in range(10)]
    train, valid, test = split_data(sentences)
    assert set(train + valid + test) == set(sentences)
    assert len(train) + len(valid) + len(test) == len(sentences)


def test_generate_sentences_no_space_before_period():
    nouns = ["dog"]
    verbs = ["see"]
    sents = generate_sentences(nouns, verbs, voice="active")
    assert all(s.endswith(".") for s in sents)
    assert all(" ." not in s for s in sents)


def test_split_data_with_holdout_routes_subject_and_object_pairs():
    nouns = ["alpha", "beta", "gamma"]
    verbs = ["see"]
    sentences = generate_sentences(nouns, verbs, voice="active")

    splits = split_data_with_holdout(
        sentences,
        subject_holdout_nouns=["alpha"],
        object_holdout_nouns=["gamma"],
        voice="active",
        seed=0,
    )

    for split_name in ("train", "valid", "test"):
        for sentence in splits[split_name]:
            subj, obj = extract_subject_and_object(sentence, voice="active")
            assert subj != "alpha"
            assert obj != "gamma"

    assert splits["generalization"]
    for sentence in splits["generalization"]:
        subj, obj = extract_subject_and_object(sentence, voice="active")
        assert subj == "alpha" or obj == "gamma"


def test_load_sentences_reads_generalization_split(tmp_path):
    prefix = str(tmp_path / "data")
    nouns = ["alpha", "beta"]
    verbs = ["see"]
    save_word_lists(prefix, nouns, verbs)
    save_split(prefix, "train", ["the alpha will see the alpha."])
    save_split(prefix, "valid", ["the alpha will see the beta."])
    save_split(prefix, "test", ["the beta will see the alpha."])
    save_split(prefix, "generalization", ["the beta will see the beta."])

    dataset, _ = load_sentences(str(tmp_path), role_scheme="svo")
    assert set(dataset.keys()) == {"train", "valid", "test", "generalization"}
    assert len(dataset["generalization"]) == 1
