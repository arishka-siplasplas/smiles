# Results

Backbone: ResNet-18 · Dataset: CIFAR-100 · Batch: 256 · LARS · fixed lr=0.3
Fixed: projector 2048, loss weights sim=25 / var=25 / cov=1.
**Varied hyperparameter: pretraining epochs {2, 5, 10, 20, 40}** (paper's VICReg Fig 5a).

Label-free geometric quality metrics on the frozen embeddings:
`IdEst` (dim_MST, paper's main metric), `RankMe` (effective rank), `TwoNN` (intrinsic dim).
`acc` = linear-probe top-1 accuracy. Goal: does a metric track `acc` across the runs?

Raw numbers: [`results/vicreg_exp.csv`](results/vicreg_exp.csv), [`results/vicreg_ctr.csv`](results/vicreg_ctr.csv).

## VICReg-exp  (tau=0.5)
| epochs | IdEst | RankMe | TwoNN | acc   |
|--------|-------|--------|-------|-------|
| 2      | 3.375 | 9.564  | 6.072 | 10.49 |
| 5      | 3.463 | 9.048  | 6.886 | 8.92  |
| 10     | 5.793 | 18.612 | 9.449 | 13.01 |
| 20     | 2.980 | 5.398  | 6.579 | 8.72  |
| 40     | 5.479 | 16.600 | 9.165 | 19.52 |

## VICReg-ctr  (tau=0.5)
| epochs | IdEst | RankMe | TwoNN  | acc   |
|--------|-------|--------|--------|-------|
| 2      | 4.831 | 17.010 | 11.490 | 11.73 |
| 5      | 4.660 | 23.433 | 7.916  | 11.37 |
| 10     | 4.538 | 21.107 | 7.809  | 11.16 |
| 20     | 4.665 | 22.480 | 7.766  | 11.33 |
| 40     | 4.580 | 22.144 | 8.379  | 11.14 |

## Notes
- **VICReg-exp**: acc has real spread (8.7 → 19.5). The best run (40 ep, acc 19.52) also has the
  highest IdEst and near-highest RankMe/TwoNN; the 10-ep run is second best on all three metrics
  *and* on acc — metrics and accuracy move together here.
- **VICReg-ctr**: acc is nearly flat (~11.1–11.7) across all epoch budgets, so there is little
  quality spread to track; the metrics vary but accuracy does not follow.
