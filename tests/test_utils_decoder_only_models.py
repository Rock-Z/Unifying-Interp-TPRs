import sys
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from src.utils import encode_decoder_only_models  # noqa: E402

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


def test_punct_encoding_shape_matches_hidden_size():
    sentences = ["hello world."]
    config = AutoConfig.from_pretrained(MODEL_ID)
    hidden = int(config.hidden_size)

    vecs, dim = encode_decoder_only_models(
        model_name=MODEL_ID,
        sentences=sentences,
        decoder_layer_indices=[-1],
        tokens="punct",
        device="cpu",
    )
    assert vecs.shape == (1, hidden)
    assert dim == hidden
    assert not np.any(np.isnan(vecs))


def test_full_sequence_encoding_flattens_layer():
    sentences = ["the cat sat."]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    seq_len = len(tok(sentences[0], add_special_tokens=True)["input_ids"])
    hidden = int(AutoConfig.from_pretrained(MODEL_ID).hidden_size)

    vecs, dim = encode_decoder_only_models(
        model_name=MODEL_ID,
        sentences=sentences,
        decoder_layer_indices=[0],
        tokens="all",
        device="cpu",
    )
    assert vecs.shape == (1, seq_len * hidden)
    assert dim == seq_len * hidden


def test_full_sequence_multi_layer_concatenation():
    sentences = ["a b c ."]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    seq_len = len(tok(sentences[0], add_special_tokens=True)["input_ids"])
    hidden = int(AutoConfig.from_pretrained(MODEL_ID).hidden_size)

    vecs, dim = encode_decoder_only_models(
        model_name=MODEL_ID,
        sentences=sentences,
        decoder_layer_indices=[0, 1],
        tokens="all",
        device="cpu",
    )
    assert vecs.shape == (1, seq_len * hidden * 2)
    assert dim == seq_len * hidden * 2
