import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from sae import SparseAutoencoder, SAEConfig


def test_sae_forward_returns_dead_mask():
    cfg = SAEConfig(input_dim=4, hidden_dim=2, dead_latent_threshold=0.1)
    model = SparseAutoencoder(cfg)
    x = torch.zeros(3, 4)
    out = model(inputs_embeds=x)
    assert 'dead_mask' in out
    assert out['dead_mask'].shape[0] == cfg.hidden_dim
    assert torch.allclose(out['dead_ratio'], out['dead_mask'].float().mean())
    assert torch.allclose(model.last_dead_mask.float(), out['dead_mask'].float())
    assert torch.isclose(model.last_dead_ratio, out['dead_ratio'])
