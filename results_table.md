# Results — fill as runs finish

Backbone: ResNet-18 · Dataset: CIFAR-100 · Epochs: 50 · Batch: 256 · LARS
Fixed: projector 2048, loss weights sim=25 / var=25 / cov=1. **Varied hyperparameter: pretraining LR.**

`geom_metric` = label-free geometric quality metric (RankMe / effective rank) on the frozen
embeddings. Goal: does `geom_metric` track `linear_acc` across the runs?

## VICReg
| run       | lr  | linear_acc | RankMe | stable_rank | ckpt |
|-----------|-----|------------|--------|-------------|------|
| lr0p1     | 0.1 |            |        |             |      |
| lr0p2     | 0.2 |            |        |             |      |
| lr0p3     | 0.3 |            |        |             |      |
| lr0p5     | 0.5 |            |        |             |      |
| lr1p0     | 1.0 |            |        |             |      |

## VICReg-exp  (tau=0.5)
| run       | lr  | linear_acc | RankMe | stable_rank | ckpt |
|-----------|-----|------------|--------|-------------|------|
| lr0p1     | 0.1 |            |        |             |      |
| lr0p2     | 0.2 |            |        |             |      |
| lr0p3     | 0.3 |            |        |             |      |
| lr0p5     | 0.5 |            |        |             |      |
| lr1p0     | 1.0 |            |        |             |      |

## VICReg-ctr  (tau=0.5)
| run       | lr  | linear_acc | RankMe | stable_rank | ckpt |
|-----------|-----|------------|--------|-------------|------|
| lr0p1     | 0.1 |            |        |             |      |
| lr0p2     | 0.2 |            |        |             |      |
| lr0p3     | 0.3 |            |        |             |      |
| lr0p5     | 0.5 |            |        |             |      |
| lr1p0     | 1.0 |            |        |             |      |
