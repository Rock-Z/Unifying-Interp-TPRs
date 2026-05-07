import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from probing import tikhonov_pinv, tsvd_pinv, invert_output_layer, construct_unbinding_vectors, apply_analytic_probe
from model import TensorProductEncoder, TensorProductEncoderConfig
from invert_tpr import _is_content_only_role_scheme, _map_probe_position_to_tpe_role_id


def make_tpencoder():
    cfg = TensorProductEncoderConfig(
        hidden_size=4,
        n_fillers=3,
        n_roles=2,
        filler_dim=2,
        role_dim=2,
        filler_pad_token_id=0,
        role_pad_token_id=0,
    )
    return TensorProductEncoder(cfg)


def test_tikhonov_equivalence():
    mat = torch.randn(4, 4)
    assert torch.allclose(tikhonov_pinv(mat, 0.0), torch.linalg.pinv(mat))


def test_tsvd_shape():
    mat = torch.randn(5, 3)
    pinv = tsvd_pinv(mat, k=3)
    assert pinv.shape == (3, 5)


def test_invert_output_layer_identity():
    tp = make_tpencoder()
    W_inv, bias = invert_output_layer(tp, atol=1e-4)
    W = tp.output_layer.weight
    eye = W_inv @ W
    assert torch.allclose(eye, torch.eye(4), atol=1e-4)
    assert bias.shape[0] == W_inv.shape[0]


def test_construct_unbinding_vectors():
    tp = make_tpencoder()
    W_probe = construct_unbinding_vectors(tp, role_id=0)
    assert W_probe.shape[0] == tp.filler_embedding.num_embeddings


def test_apply_analytic_probe():
    tp = make_tpencoder()
    filler_ids = torch.tensor([[1, 2]])
    role_ids = torch.tensor([[1, 1]])
    agg = tp(filler_ids, role_ids).last_hidden_state
    logits = apply_analytic_probe(agg, tp, role_id=1, atol=1e-4)
    assert logits.shape[0] == 1


def test_probe_position_role_mapping_content_schemes():
    assert _is_content_only_role_scheme("l2r_content")
    assert _is_content_only_role_scheme("r2l_content")
    assert not _is_content_only_role_scheme("l2r")
    assert not _is_content_only_role_scheme(None)

    assert _map_probe_position_to_tpe_role_id(1, "l2r") == 2
    assert _map_probe_position_to_tpe_role_id(1, "l2r_content") == 1
    assert _map_probe_position_to_tpe_role_id(-3, "l2r_content") == 3
