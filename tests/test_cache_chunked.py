import os
import json
import numpy as np

import types

from src.utils import load_dataset_with_embeddings, get_cache_path


class DummyST:
    def __init__(self, name):
        self.name = name
        self._dim = 8

    def eval(self):
        return self

    def encode(self, texts, batch_size=1, show_progress_bar=False):
        arr = []
        for t in texts:
            rng = abs(hash(t)) % (10 ** 6)
            vec = np.array([(rng + i) % 97 / 97.0 for i in range(self._dim)], dtype=np.float32)
            arr.append(vec)
        return np.stack(arr, axis=0)

    def get_sentence_embedding_dimension(self):
        return self._dim


def _fake_dataset(sentences):
    class Split:
        def __init__(self, sents):
            self._sents = list(sents)
            self.column_names = ["sentence"]

        def __getitem__(self, key):
            if key == "sentence":
                return self._sents
            return []

        def add_column(self, name, values, new_fingerprint=None):
            self.column_names.append(name)
            return self

    return {"train": Split(sentences), "valid": Split(sentences), "test": Split(sentences)}


def test_chunked_cache_roundtrip(tmp_path, monkeypatch):
    # Monkeypatch SentenceTransformer to our dummy
    import src.utils as utils_mod
    monkeypatch.setattr(utils_mod, "SentenceTransformer", DummyST, raising=True)

    sentences = ["the a will see the b.", "the b will see the a."]
    ds = _fake_dataset(sentences)
    dataset_path = str(tmp_path)
    cache_flag = dataset_path  # any truthy value triggers caching path
    model_name = "dummy/model"

    ds1, dim1 = load_dataset_with_embeddings(
        dataset=ds,
        dataset_path=dataset_path,
        embedding_model_name=model_name,
        embedding_cache_path=cache_flag,
        embedding_column_name="target_embeddings",
        add_prefix=None,
        entries_per_chunk=1,  # force multiple chunks
    )

    # Check index and chunk files exist
    index_path = get_cache_path(dataset_path, model_name)
    assert os.path.exists(index_path)
    with open(index_path) as f:
        idx = json.load(f)
    assert "items" in idx and len(idx["items"]) == 2
    # Now reload to hit cache path (no encoding)
    ds2, dim2 = load_dataset_with_embeddings(
        dataset=_fake_dataset(sentences),
        dataset_path=dataset_path,
        embedding_model_name=model_name,
        embedding_cache_path=cache_flag,
        embedding_column_name="target_embeddings",
        add_prefix=None,
        entries_per_chunk=1,
    )
    assert dim1 == 8 and dim2 == 8

