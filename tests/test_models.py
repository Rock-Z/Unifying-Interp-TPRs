import torch
import pytest
from transformers import AutoModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from src.model import (
    TensorProductEncoderConfig,
    TensorProductEncoder,
    TensorProductEncoderForPretraining,
    RecurrentEncoderConfig,
    RecurrentDecoderConfig,
    RecurrentEncoder,
    RecurrentDecoder,
    RecurrentEncoderDecoderModel,
)


def make_tpe_config(**kwargs):
    base = dict(
        hidden_size=4,
        n_fillers=5,
        n_roles=3,
        filler_dim=2,
        role_dim=2,
        filler_pad_token_id=0,
        role_pad_token_id=0,
    )
    base.update(kwargs)
    return TensorProductEncoderConfig(**base)


def random_ids(batch=2, seq=3, vocab=5):
    return torch.randint(1, vocab, (batch, seq))


def test_autoregistration():
    config = RecurrentEncoderConfig()
    model = AutoModel.from_config(config)
    from src.model import RecurrentEncoder
    assert isinstance(model, RecurrentEncoder)
    assert model.config.model_type == "recurrent_encoder"


@pytest.mark.parametrize(
    "return_bindings,aggregation,has_linear",
    [
        (False, "sum", True),
        (True, "sum", True),
        (False, "mean", False),
        (True, "mean", False),
    ],
)
def test_tpe_forward(return_bindings, aggregation, has_linear):
    cfg = make_tpe_config(
        return_bindings=return_bindings,
        aggregation=aggregation,
        has_linear_layer=has_linear,
    )
    model = TensorProductEncoder(cfg)
    filler_ids = random_ids(seq=3)
    role_ids = random_ids(seq=3, vocab=cfg.n_roles)
    out = model(filler_ids, role_ids)
    assert out.last_hidden_state.shape[0] == filler_ids.size(0)
    if has_linear:
        assert out.last_hidden_state.shape[-1] == cfg.hidden_size
    else:
        assert out.last_hidden_state.shape[-1] == cfg.filler_dim * cfg.role_dim


def test_tpe_pretraining_loss():
    cfg = make_tpe_config()
    model = TensorProductEncoderForPretraining(cfg)
    filler_ids = random_ids()
    role_ids = random_ids(vocab=cfg.n_roles)
    target = torch.randn(filler_ids.size(0), 1, cfg.filler_dim * cfg.role_dim)
    out = model(filler_ids, role_ids, target_embeddings=target)
    assert out.loss is not None


def test_recurrent_encoder_decoder():
    enc_cfg = RecurrentEncoderConfig(vocab_size=10)
    dec_cfg = RecurrentDecoderConfig(vocab_size=10)
    model = RecurrentEncoderDecoderModel.from_encoder_decoder_pretrained(
        RecurrentEncoder(enc_cfg), RecurrentDecoder(dec_cfg)
    )
    input_ids = random_ids(vocab=10)
    labels = random_ids(vocab=10)
    out = model(
        input_ids=input_ids,
        input_lengths=torch.tensor([3, 3]),
        filler_ids=None,
        role_ids=None,
        labels=labels,
    )
    assert out.logits.shape[0] == input_ids.size(0)


def test_tpe_pretraining_save_load_roundtrip(tmp_path):
    cfg = make_tpe_config()
    model = TensorProductEncoderForPretraining(cfg)

    with torch.no_grad():
        filler_weights = torch.arange(cfg.n_fillers * cfg.filler_dim, dtype=torch.float32).view(cfg.n_fillers, cfg.filler_dim)
        role_weights = torch.arange(cfg.n_roles * cfg.role_dim, dtype=torch.float32).view(cfg.n_roles, cfg.role_dim) / 10.0
        output_weights = torch.eye(cfg.hidden_size, dtype=torch.float32)
        output_bias = torch.arange(cfg.hidden_size, dtype=torch.float32)

        model.encoder.filler_embedding.weight.copy_(filler_weights)
        model.encoder.role_embedding.weight.copy_(role_weights)
        if model.encoder.output_layer is not None:
            model.encoder.output_layer.weight.copy_(output_weights)
            model.encoder.output_layer.bias.copy_(output_bias)

    save_dir = tmp_path / "tpe"
    model.save_pretrained(save_dir)

    bin_path = save_dir / "pytorch_model.bin"
    safetensor_path = save_dir / "model.safetensors"
    if bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    elif safetensor_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensor_path))
    else:
        raise AssertionError("No saved checkpoint weights found in save_pretrained output")
    allowed_prefixes = ("encoder.", "embedding_model.", "loss_fn.")
    assert all(key.startswith(allowed_prefixes) for key in state_dict.keys())

    loaded = TensorProductEncoderForPretraining.from_pretrained(save_dir)

    assert loaded is not model
    assert loaded.encoder is not model.encoder
    loaded_config = loaded.config.to_dict()
    original_config = model.config.to_dict()
    loaded_config['_name_or_path'] = ''
    original_config['_name_or_path'] = ''
    assert loaded_config == original_config

    assert torch.allclose(loaded.encoder.filler_embedding.weight, model.encoder.filler_embedding.weight)
    assert torch.allclose(loaded.encoder.role_embedding.weight, model.encoder.role_embedding.weight)
    if loaded.encoder.output_layer is not None:
        assert torch.allclose(loaded.encoder.output_layer.weight, model.encoder.output_layer.weight)
        assert torch.allclose(loaded.encoder.output_layer.bias, model.encoder.output_layer.bias)

    filler_ids = random_ids()
    role_ids = random_ids(vocab=cfg.n_roles)
    target = torch.zeros(filler_ids.size(0), 1, cfg.hidden_size)
    out = loaded(filler_ids, role_ids, target_embeddings=target)
    assert out.loss is not None
