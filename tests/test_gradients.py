import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import (
    RecurrentEncoderConfig,
    RecurrentDecoderConfig,
    RecurrentEncoder,
    RecurrentDecoder,
    RecurrentEncoderDecoderModel,
    TensorProductEncoderConfig,
    TensorProductEncoder,
    TensorProductEncoderForPretraining,
)
from probing import LinearProbe, LinearProbeConfig


def random_ids(batch=2, seq=3, vocab=5):
    return torch.randint(1, vocab, (batch, seq))


def test_recurrent_models_backward():
    archs = ["RNN", "GRU", "LSTM"]
    for arch in archs:
        enc_cfg = RecurrentEncoderConfig(architecture=arch, vocab_size=10, hidden_size=4)
        dec_cfg = RecurrentDecoderConfig(architecture=arch, vocab_size=10, hidden_size=4)
        enc = RecurrentEncoder(enc_cfg)
        dec = RecurrentDecoder(dec_cfg)
        model = RecurrentEncoderDecoderModel.from_encoder_decoder_pretrained(enc, dec)

        input_ids = random_ids(vocab=10)
        lengths = torch.tensor([3, 3])
        labels = random_ids(vocab=10)

        out = model(input_ids=input_ids, input_lengths=lengths, labels=labels)
        assert out.loss is not None
        out.loss.backward()


def test_tensor_product_encoder_backward():
    cfg = TensorProductEncoderConfig(
        hidden_size=4,
        n_fillers=5,
        n_roles=3,
        filler_dim=2,
        role_dim=2,
        filler_pad_token_id=0,
        role_pad_token_id=0,
        has_linear_layer=True,
        return_bindings=True,
        aggregation="sum",
    )
    model = TensorProductEncoder(cfg)
    filler_ids = random_ids(seq=3)
    role_ids = random_ids(seq=3, vocab=cfg.n_roles)
    target = torch.randn(filler_ids.size(0), 1, cfg.hidden_size)
    out = model(filler_ids, role_ids)
    loss = torch.nn.MSELoss()(out.last_hidden_state, target)
    loss.backward()


def test_tpe_pretraining_backward_with_gru():
    embed_cfg = RecurrentEncoderConfig(architecture="GRU", vocab_size=10, hidden_size=4)
    embedding_model = RecurrentEncoder(embed_cfg)
    cfg = TensorProductEncoderConfig(
        hidden_size=4,
        n_fillers=5,
        n_roles=3,
        filler_dim=2,
        role_dim=2,
        filler_pad_token_id=0,
        role_pad_token_id=0,
    )
    model = TensorProductEncoderForPretraining(cfg, embedding_model=embedding_model)
    filler_ids = random_ids()
    role_ids = random_ids(vocab=cfg.n_roles)
    emb_ids = random_ids(vocab=10)
    emb_len = torch.tensor([3, 3])
    out = model(
        filler_ids,
        role_ids,
        embedding_model_input_ids=emb_ids,
        embedding_model_input_lengths=emb_len,
    )
    assert out.loss is not None
    out.loss.backward()


def test_tpe_pretraining_backward_with_lstm():
    embed_cfg = RecurrentEncoderConfig(architecture="LSTM", vocab_size=10, hidden_size=4)
    embedding_model = RecurrentEncoder(embed_cfg)
    cfg = TensorProductEncoderConfig(
        hidden_size=8,
        n_fillers=5,
        n_roles=3,
        filler_dim=2,
        role_dim=2,
        filler_pad_token_id=0,
        role_pad_token_id=0,
    )
    model = TensorProductEncoderForPretraining(cfg, embedding_model=embedding_model)
    filler_ids = random_ids()
    role_ids = random_ids(vocab=cfg.n_roles)
    emb_ids = random_ids(vocab=10)
    emb_len = torch.tensor([3, 3])
    out = model(
        filler_ids,
        role_ids,
        embedding_model_input_ids=emb_ids,
        embedding_model_input_lengths=emb_len,
    )
    assert out.loss is not None
    out.loss.backward()


def test_linear_probe_backward():
    cfg = TensorProductEncoderConfig(
        hidden_size=4,
        n_fillers=5,
        n_roles=2,
        filler_dim=2,
        role_dim=2,
        filler_pad_token_id=0,
        role_pad_token_id=0,
    )
    tpencoder = TensorProductEncoder(cfg)
    probe_cfg = LinearProbeConfig(encoder_hidden_size=cfg.hidden_size, num_labels=cfg.n_fillers)
    dummy_encoder = RecurrentEncoder(RecurrentEncoderConfig(vocab_size=10, hidden_size=cfg.hidden_size))
    probe = LinearProbe(probe_cfg, dummy_encoder)
    filler_ids = random_ids()
    role_ids = random_ids(vocab=cfg.n_roles)
    agg = tpencoder(filler_ids, role_ids).last_hidden_state.squeeze(1)
    labels = torch.randint(0, cfg.n_fillers, (agg.size(0),))
    out = probe(labels=labels, hidden_states=agg)
    assert out.loss is not None
    out.loss.backward()
