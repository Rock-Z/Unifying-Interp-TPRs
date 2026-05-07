import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from analogy_utils import evaluate_analogy



def test_evaluate_analogy_rank():
    emb = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 1.0]),
        "C": np.array([1.0, 1.0]),
        "D": np.array([2.0, 0.0]),
    }
    all_items = list(emb.keys())
    all_embs = np.stack([emb[k] for k in all_items])
    res = evaluate_analogy(emb["A"], emb["B"], emb["C"], emb["D"], all_embs, all_items)
    assert res["rank"] == 1
    assert res["top_1_correct"]
