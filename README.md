# matryoshka-toy

A toy reimplementation of the core experiment in *Matryoshka LM Suites* (Godey & Artzi, 2026, [arXiv 2608.09703](https://arxiv.org/abs/2608.09703)):
one transformer with three exit points (2 layers @64 → +2 @128 → +2 @256), trained once, against three separate models of matched parameter count.

Written for the Weizmann course *Active LLM Paper Reading*. Write-up: [course site post](https://active-ai-reading.github.io/).

## Run

```
pip install torch matplotlib
python matryoshka_toy.py
```

CPU only, about 24 minutes in total. Options: `--steps`, `--alpha` (distillation weight, default 0.3), `--no-rescale` (drop the norm matching in the junction), `--seed`, `--tag`.

## What is in the file

| section | what |
|---|---|
| 1. corpus | character-level text from a small grammar with two long-range dependencies (subject–verb agreement across a relative clause, a name copied from the start of a sentence to its end) |
| 2. transformer | `TransformerBlock`, `LayerStack` (layers + a fresh embedding at one width) |
| 3. `VanillaLM` | one independent model per size — the baseline |
| 4. `MatryoshkaLM` | one stack, three exits, the inter-model junction between blocks (`forward`), and `matryoshka_loss` = Σ (1−α)·CE + α·KD with `stop_grad` on the teacher |
| 5. training | AdamW, warmup, gradient clipping; the same loop for both kinds of model |
| 6. experiment | parameter-matched baselines, per-size validation loss, agreement between model pairs, `results.json` + `curves.png` |

## Results (seed 0, 1,500 steps)

| size | separate | nested | gap |
|---|---|---|---|
| small | 0.345 | 0.358 | +3.8% |
| medium | 0.325 | 0.327 | +0.6% |
| large | 0.315 | 0.318 | +1.2% |

Parameters: 2.82M (three separate) vs 2.15M (nested), 24% fewer. Training time: 812 s vs 620 s.

Agreement on the next character — medium/large: 91.7% (separate) vs 94.7% (nested).

`curves_v1.png` is the first run, with a baseline that was not parameter-matched and did not converge; kept for the record.
