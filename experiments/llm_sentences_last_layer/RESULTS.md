# LLM Last-Layer Sentence Results

Representation: decoder-only punctuation-token hidden states on `data/sentences`.

Baseline coverage: analogy compares nearest-neighbor retrieval in raw hidden-state space against TPE space; probes compare analytic TPE probes against trained linear probes; SAE compares analytic ridge construction against trained TopK and supervised filler-role SAEs.

## TPE Selection

| Model | Run | R2 | EV |
| --- | --- | ---: | ---: |
| Qwen/Qwen3-8B | qwen3_8b_cosf256_lr4e3 | 0.7052 | 0.7302 |
| allenai/OLMo-2-1124-13B | olmo_13b_cosf256_lr4e3 | 0.6440 | 0.6885 |
| openai/gpt-oss-20b | gpt_oss_20b_cosf256_lr4e3 | 0.7423 | 0.7869 |

## Nearest-Neighbor Analogy

| Model | Raw hidden NN top1 | Raw hidden NN top3 | TPE NN top1 | TPE NN top3 | Raw hidden mean rank | TPE mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-8B | 0.4073 | 0.5396 | 0.5957 | 0.7239 | 266.8124 | 125.3815 |
| allenai/OLMo-2-1124-13B | 0.1956 | 0.3440 | 0.4140 | 0.5662 | 260.3537 | 89.8238 |
| openai/gpt-oss-20b | 0.5906 | 0.7230 | 0.7976 | 0.8744 | 80.3790 | 26.8942 |

## Linear Probes

Analytic accuracies use validation-selected inversion parameters when `probe_inversion_tuning_summary.*` is present; untuned values are retained in `summary.json`.

| Model | Role | Analytic acc | Trained acc |
| --- | --- | ---: | ---: |
| Qwen/Qwen3-8B | subj | 0.9949 | 0.9781 |
| Qwen/Qwen3-8B | verb | 1.0000 | 1.0000 |
| Qwen/Qwen3-8B | obj | 0.9949 | 0.9562 |
| allenai/OLMo-2-1124-13B | subj | 0.9730 | 0.9680 |
| allenai/OLMo-2-1124-13B | verb | 1.0000 | 1.0000 |
| allenai/OLMo-2-1124-13B | obj | 0.9781 | 0.9444 |
| openai/gpt-oss-20b | subj | 1.0000 | 0.9848 |
| openai/gpt-oss-20b | verb | 1.0000 | 1.0000 |
| openai/gpt-oss-20b | obj | 0.9966 | 0.9848 |

## Probe Seed Check

| Model | Role | N | Trained mean | Trained std | Trained min | Trained max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-8B | obj | 3 | 0.9354 | 0.0237 | 0.9022 | 0.9562 |
| Qwen/Qwen3-8B | subj | 3 | 0.9837 | 0.0068 | 0.9781 | 0.9933 |
| Qwen/Qwen3-8B | verb | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| allenai/OLMo-2-1124-13B | obj | 3 | 0.9680 | 0.0168 | 0.9444 | 0.9815 |
| allenai/OLMo-2-1124-13B | subj | 3 | 0.9674 | 0.0062 | 0.9595 | 0.9747 |
| allenai/OLMo-2-1124-13B | verb | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| openai/gpt-oss-20b | obj | 3 | 0.9691 | 0.0112 | 0.9595 | 0.9848 |
| openai/gpt-oss-20b | subj | 3 | 0.9809 | 0.0068 | 0.9713 | 0.9865 |
| openai/gpt-oss-20b | verb | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## SAE

| Model | Method | MSE | R2 | Cosine | L0 | Purity | Accuracy | Well-rankedness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-8B | analytic ridge | 0.0545 | 0.9932 | 0.9967 | 0.4904 | 0.0949 | 0.5220 | 0.9492 |
| Qwen/Qwen3-8B | trained topk | 0.0939 | 0.9882 | 0.9940 | 0.0152 | 0.0264 | 0.0406 | 0.9535 |
| Qwen/Qwen3-8B | trained supervised | 0.0167 | 0.9979 | 0.9989 | 0.9978 | 0.0338 | 0.0152 | 0.9978 |
| allenai/OLMo-2-1124-13B | analytic ridge | 0.0220 | 0.9822 | 0.9911 | 0.3143 | 0.2757 | 0.6985 | 0.9932 |
| allenai/OLMo-2-1124-13B | trained topk | 0.0420 | 0.9659 | 0.9828 | 0.0300 | 0.0280 | 0.3150 | 0.9134 |
| allenai/OLMo-2-1124-13B | trained supervised | 0.0130 | 0.9894 | 0.9947 | 0.9889 | 0.0518 | 0.0240 | 0.9985 |
| openai/gpt-oss-20b | analytic ridge | 0.3193 | 0.9846 | 0.9924 | 0.4190 | 0.2945 | 0.5940 | 0.9992 |
| openai/gpt-oss-20b | trained topk | 1.6826 | 0.9188 | 0.9603 | 0.1109 | 0.0383 | 0.5674 | 0.8458 |
| openai/gpt-oss-20b | trained supervised | 0.1895 | 0.9909 | 0.9954 | 0.8505 | 0.0447 | 0.1625 | 0.9945 |
