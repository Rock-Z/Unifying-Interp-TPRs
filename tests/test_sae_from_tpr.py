import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sae import SparseAutoencoder
from model import TensorProductEncoder, TensorProductEncoderConfig

def build_simple_tpe():
    cfg = TensorProductEncoderConfig(
        hidden_size=4,
        n_fillers=2,
        n_roles=2,
        filler_dim=2,
        role_dim=2,
        has_linear_layer=True,
    )
    tpe = TensorProductEncoder(cfg)
    with torch.no_grad():
        tpe.filler_embedding.weight.copy_(torch.eye(2))
        tpe.role_embedding.weight.copy_(torch.eye(2))
        tpe.output_layer.weight.copy_(torch.eye(4))
        tpe.output_layer.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    return tpe

def test_feature_count_default_role_invariant():
    tpe = build_simple_tpe()
    # Default behavior constructs role-invariant features only
    sae = SparseAutoencoder.from_tensor_product_encoder(tpe)
    assert sae.config.hidden_dim == tpe.config.n_fillers
    assert sae.config.input_dim == tpe.config.hidden_size

def test_feature_count_pairwise_when_disabled_role_invariant():
    tpe = build_simple_tpe()
    # When role_invariant is False, construct all filler x role features
    sae = SparseAutoencoder.from_tensor_product_encoder(tpe, role_invariant=False)
    assert sae.config.hidden_dim == tpe.config.n_fillers * tpe.config.n_roles

def test_weights_match_expected_binding_pairwise():
    tpe = build_simple_tpe()
    # Use pairwise construction to test binding weights
    sae = SparseAutoencoder.from_tensor_product_encoder(tpe, role_invariant=False)
    # With kron(filler, role) ordering, column 1 corresponds to the
    # third basis element [f1*r0] in a 2x2 layout: [f0*r0, f0*r1, f1*r0, f1*r1]
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert torch.allclose(sae.decoder.weight[:, 1], expected)

def test_decoder_bias_copied():
    tpe = build_simple_tpe()
    sae = SparseAutoencoder.from_tensor_product_encoder(tpe)
    assert torch.allclose(sae.decoder.bias, tpe.output_layer.bias)


def test_decoder_bias_anchor_override():
    tpe = build_simple_tpe()
    anchor = torch.tensor([9.0, 8.0, 7.0, 6.0])
    sae = SparseAutoencoder.from_tensor_product_encoder(tpe, bias_anchor=anchor)
    assert torch.allclose(sae.decoder.bias, anchor)

def test_config_overrides_respected():
    tpe = build_simple_tpe()
    sae = SparseAutoencoder.from_tensor_product_encoder(
        tpe, sae_config={"sparsity_penalty": 0.5}
    )
    assert sae.config.sparsity_penalty == 0.5

def test_reconstruction_matches_input_pairwise():
    tpe = build_simple_tpe()
    # Pairwise features should exactly reconstruct for identity setup
    sae = SparseAutoencoder.from_tensor_product_encoder(tpe, role_invariant=False)
    filler_ids = torch.tensor([[1]])
    role_ids = torch.tensor([[0]])
    with torch.no_grad():
        h = tpe(filler_ids, role_ids).last_hidden_state.squeeze(1)
        recon = sae.reconstruct(h)
    assert torch.allclose(recon, h)


def test_decoder_pinv_l2_regularization_changes_decoder_weights():
    tpe = build_simple_tpe()
    sae_none = SparseAutoencoder.from_tensor_product_encoder(
        tpe,
        role_invariant=False,
        second_layer_construction="pinv-unbinding",
        decoder_pinv_regularization="none",
    )
    sae_l2 = SparseAutoencoder.from_tensor_product_encoder(
        tpe,
        role_invariant=False,
        second_layer_construction="pinv-unbinding",
        decoder_pinv_regularization="l2",
        decoder_pinv_l2_lambda=1.0,
    )
    assert torch.isfinite(sae_l2.decoder.weight).all()
    assert not torch.allclose(sae_none.decoder.weight, sae_l2.decoder.weight)
