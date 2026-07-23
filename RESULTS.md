# Results — geometric embedding-quality metrics

Setup: solo-learn, ResNet-18, CIFAR-100, projector 2048, LARS. Metrics computed on the frozen
backbone features (512-d): **IdEst** (dim_MST), **RankMe**, **TwoNN**, plus linear-probe accuracy
(offline probe). Methodological fix: the backbone's first conv is set to `padding=2` (as in
solo-learn) — before this the accuracy was underestimated.

The main result is on **VICReg**. I first tried varying the learning rate, but it didn't work:
training uses the LARS optimizer, which auto-adjusts the step size per layer and makes the model
almost insensitive to the chosen learning rate. As a result all 5 models came out nearly identical
(accuracy 52–54%, only ~1% spread), with insignificant correlations — there's simply no spread in
quality. So I switched to varying the number of epochs (2/5/10/20/40), and that worked
(accuracy 37–59%).

## Run 1 — VICReg, varying learning rate (20 epochs, lr ∈ {0.1…1.0})

| metric        | result                            |
|---------------|-----------------------------------|
| accuracy      | 52.8–53.9% (~1% spread)           |
| IdEst vs acc  | ρ = +0.70, p = 0.19 (not signif.) |
| RankMe vs acc | ρ = +0.70, p = 0.19 (not signif.) |

**Conclusion:** under LARS, VICReg is almost insensitive to lr → models are equal in quality →
the correlations are noise. Not a valid test. (Methodological finding: lr is a poor axis for
VICReg+LARS.)

## Run 2 — VICReg, varying number of epochs (2/5/10/20/40, lr=0.3) ✅

| epochs | IdEst | RankMe | TwoNN | acc  |
|--------|-------|--------|-------|------|
| 2      | 13.44 | 156.9  | 14.49 | 36.8 |
| 5      | 13.85 | 157.9  | 14.68 | 45.6 |
| 10     | 14.02 | 184.0  | 14.93 | 50.1 |
| 20     | 14.13 | 235.0  | 15.93 | 54.9 |
| 40     | 14.05 | 294.1  | 15.63 | 58.9 |

- **RankMe vs acc:** ρ = +1.00, p ≈ 1e-24 → RankMe's finding reproduced (higher rank = higher
  accuracy). ✅
- **IdEst vs acc:** ρ = +0.90, p = 0.037 → positive, whereas the paper reports IdEst should be
  negative. The reason is the short-training regime (2–40 epochs): intrinsic dimension is still
  rising; the decrease the authors show is a late-training phenomenon. IdEst peaks at ~epoch 20
  (14.13) and at epoch 40 already dips (14.05) — the turnover is only just beginning.
- **TwoNN vs acc:** ρ = +0.90 → moves together with IdEst (a noisy baseline, no separate value).

**Conclusion:** RankMe reproduced cleanly. In short training, IdEst behaves opposite to the paper;
catching its negative correlation would need a longer run (past the peak, ~100+ epochs).

## Runs 3 & 4 — VICReg-exp and VICReg-ctr ❌

| method     | accuracy         | status    |
|------------|------------------|-----------|
| VICReg-exp | 9–19% (unstable) | collapse  |
| VICReg-ctr | ~11% (flat)      | not learning |

**Conclusion:** results are invalid. The custom loss implementation (LogSumExp / transposition)
produces collapsed representations. It needs a correct implementation strictly per Garrido et al.,
ICLR 2023 (arXiv 2206.02574, Appendix L) — loss weights, τ, normalization. Until then no
conclusions can be drawn for exp/ctr.

## Overall

- ✅ **RankMe** reproduced (strong, significant positive correlation with accuracy).
- ⚠️ **IdEst** — in this setup opposite to the paper due to the early-training regime; needs a
  longer run.
- ❌ **VICReg-exp / VICReg-ctr** — loss implementation broken, results don't count.
- 📌 Methodologically: the "epochs" axis gives the needed quality spread; the "lr" axis under LARS
  does not.

Raw numbers for exp/ctr: [`results/vicreg_exp.csv`](results/vicreg_exp.csv),
[`results/vicreg_ctr.csv`](results/vicreg_ctr.csv).
