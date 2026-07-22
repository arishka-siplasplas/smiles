# VICReg / VICReg-exp / VICReg-ctr — embedding-quality sweep (CIFAR-100, ResNet-18)

My part of the reproduction: 3 methods × 5 hyperparameter runs = **15 models**, each with a
final linear-eval accuracy, so we can check whether a **label-free geometric metric**
(RankMe / effective rank) predicts downstream accuracy.

> **Why train and not just download?** Model zoos give *one* tuned checkpoint per method —
> the opposite of what this study needs (a *spread* of qualities). No zoo has VICReg-exp/ctr
> at all (they're from the RankMe / "duality" paper, code never released — only pseudocode).
> On CIFAR-100 + ResNet-18 a run is a few hours on one GPU, so this is cheap — the ImageNet
> pain doesn't apply here.

## Layout
```
vicreg_quality/
├── generate_configs.py     # regenerates all 15 yaml configs (edit the sweep here)
├── results_table.md        # <-- fill this as runs finish
├── vicreg/configs/         # 5 configs, stock solo-learn method
├── vicreg_exp/             # loss.py + method.py + 5 configs
└── vicreg_ctr/             # loss.py + method.py + 5 configs
```

## The 5-model sweep (same for all three methods)
**Varied hyperparameter: pretraining learning rate.** Chosen because it is *universal across
every SSL method* (so the whole group varies the same knob → cross-method comparison is
apples-to-apples), and it is the axis RankMe itself sweeps. Everything else is fixed at
solo-learn's tuned CIFAR values (ResNet-18, 50 epochs, batch 256, LARS, projector 2048,
loss weights sim=25 / var=25 / cov=1).

| run     | lr  | note |
|---------|-----|------|
| `lr0p1` | 0.1 | low LR |
| `lr0p2` | 0.2 | below default |
| `lr0p3` | 0.3 | default / reference (solo-learn tuned) |
| `lr0p5` | 0.5 | above default |
| `lr1p0` | 1.0 | high LR (expected degraded) |

To vary a different hyperparameter instead (e.g. `cov_loss_weight` or projector dim), edit
`VARIANTS` in `generate_configs.py` and rerun it.

## Setup
```bash
git clone https://github.com/vturrisi/solo-learn.git
cd solo-learn && pip install -e .[dali,umap,h5]   # dali optional; CPU/CIFAR fine without
```
CIFAR-100 auto-downloads to `./datasets` on first run.

## Running

**Plain VICReg** (no code changes — stock method):
```bash
python main_pretrain.py \
  --config-path /Users/arinabazukina/smiles/vicreg_quality/vicreg/configs \
  --config-name cifar100_base
```
Repeat for `cifar100_cov0`, `cifar100_cov10`, `cifar100_var5`, `cifar100_dim256`.

**VICReg-exp / VICReg-ctr** — the method must be registered first. Easiest: copy the two
folders into the solo-learn repo root and add one import at the top of `main_pretrain.py`:
```python
import vicreg_exp.method   # noqa: F401  registers "vicreg_exp"
import vicreg_ctr.method    # noqa: F401  registers "vicreg_ctr"
```
Then run exactly like above, pointing `--config-path` at `vicreg_exp/configs` (or `vicreg_ctr/configs`).

> `tau` (temperature of the exp/ctr LogSumExp term) is set to 0.5 in the configs. **Verify it
> against Appendix L of the paper** — relative ordering across runs is robust, absolute loss
> scale is not.

## Linear evaluation (the accuracy that goes in the table)
solo-learn logs an **online** linear-classifier accuracy during pretraining (the number in
its CIFAR-100 table, ~68.5% for base VICReg) — read it from the last epoch. For a clean
offline number use `main_linear.py` with the frozen backbone checkpoint from `trained_models/`.

## Then: the geometric metric
For each of the 15 frozen backbones, extract embeddings on the CIFAR-100 train set and compute
the geometric quality metric (RankMe = smooth effective rank of the embedding matrix, or
whatever metric the group settled on), and correlate it against the linear accuracies in
`results_table.md`. That correlation is the actual result.

## References
- solo-learn: https://github.com/vturrisi/solo-learn
- VICReg (Bardes et al. 2022): https://github.com/facebookresearch/vicreg
- VICReg-exp / VICReg-ctr — "On the duality between contrastive and non-contrastive SSL"
  (Garrido et al. 2022): https://arxiv.org/abs/2206.02574
- RankMe (Garrido et al. 2023): https://proceedings.mlr.press/v202/garrido23a/garrido23a.pdf
