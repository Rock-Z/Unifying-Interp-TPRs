import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from src.model import (
    RecurrentEncoderConfig,
    RecurrentDecoderConfig,
    RecurrentEncoder,
    RecurrentDecoder,
    OuterProduct,
    SumFlattenedOuterProduct,
)
from src.loss import WeightedMSELoss


def random_ids(batch=2, seq=3, vocab=5):
    return torch.randint(1, vocab, (batch, seq))


def test_recurrent_encoder_output_shapes():
    cfg = RecurrentEncoderConfig(vocab_size=10, hidden_size=8, architecture="GRU")
    enc = RecurrentEncoder(cfg)
    inp = random_ids()
    lengths = torch.tensor([3, 2])
    out = enc(inp, lengths, return_hidden_states=True)
    assert out.last_hidden_state.shape == (inp.size(0), cfg.n_layers, cfg.hidden_size)
    assert out.hidden_states.shape[:2] == (inp.size(0), inp.size(1))


def test_recurrent_decoder_shapes():
    enc_cfg = RecurrentEncoderConfig(vocab_size=10, hidden_size=8, architecture="GRU")
    dec_cfg = RecurrentDecoderConfig(vocab_size=10, hidden_size=8, architecture="GRU")
    enc = RecurrentEncoder(enc_cfg)
    dec = RecurrentDecoder(dec_cfg)

    inp = random_ids()
    lengths = torch.tensor([3, 2])
    hidden = enc(inp, lengths).last_hidden_state

    dec_inp = random_ids(seq=2, vocab=10)
    dec_len = torch.tensor([2, 2])
    out = dec(dec_inp, dec_len, hidden)
    assert out.logits.shape[0] == dec_inp.size(0)
    assert out.logits.shape[-1] == dec_cfg.vocab_size


def test_outer_product_equivalence():
    batch, seq, fd, rd = 2, 3, 2, 2
    x1 = torch.randn(batch, seq, fd)
    x2 = torch.randn(batch, seq, rd)
    bindings, agg = OuterProduct("sum")(x1, x2)
    manual = torch.einsum("blf,blr->blfr", x1, x2).view(batch, seq, -1)
    manual_sum = manual.sum(dim=1).unsqueeze(1)
    assert torch.allclose(bindings, manual)
    assert torch.allclose(agg, manual_sum)

    _, sf_agg = SumFlattenedOuterProduct()(x1, x2)
    assert torch.allclose(sf_agg, manual_sum)


def test_weighted_mse_loss():
    loss_fn = WeightedMSELoss(eps=0.0)
    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([1.0, 3.0])
    loss = loss_fn(y_pred, y_true)
    expected = ((y_pred - y_true) ** 2 / (y_true ** 2)).mean()
    assert torch.allclose(loss, expected)
