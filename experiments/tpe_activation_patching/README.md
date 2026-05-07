# TPE-driven activation patching

Run token-level activation patching where the corrupted prompt is patched with hidden states synthesized by a trained layerwise TPE.

## Usage

- TPE patching (uses the existing Qwen3-8B layer-0 TPE checkpoint):  
  `uv run experiments/tpe_activation_patching/tpe_activation_patching.py experiments/tpe_activation_patching/configs/tpe_activation_patching_qwen_layer0.gin`

- Baseline activation patching (original method, for comparison):  
  `uv run experiments/activation_patching/activation_patching.py experiments/activation_patching/configs/activation_patching_sentences.gin`

Outputs are written under `results/tpe_activation_patching/` for the TPE run and `results/activation_patching/` for the baseline. The TPE script patches `layers[tpe_layer].output`, matching the layer the TPE was trained to approximate.
