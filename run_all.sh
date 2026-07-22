#!/usr/bin/env bash
# Launch all 15 pretrain runs + RankMe, on a machine with a CUDA GPU.
#
# Prereqs (do once):
#   git clone https://github.com/vturrisi/solo-learn.git && cd solo-learn && pip install -e .
#   cp -r /path/to/vicreg_quality solo-learn/           # so the folders sit in the repo root
#   # add these two imports near the top of solo-learn/main_pretrain.py:
#   #     import vicreg_quality.vicreg_exp.method   # noqa: F401
#   #     import vicreg_quality.vicreg_ctr.method   # noqa: F401
#
# Run from the solo-learn repo root:  bash vicreg_quality/run_all.sh
set -euo pipefail

QDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../vicreg_quality
VARIANTS=(base cov0 cov10 var5 dim256)
METHODS=(vicreg vicreg_exp vicreg_ctr)

echo "=== pretraining (15 runs) ==="
for m in "${METHODS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    echo ">>> $m / $v"
    python main_pretrain.py \
      --config-path "$QDIR/$m/configs" \
      --config-name "cifar100_$v"
  done
done

echo "=== RankMe on every checkpoint ==="
# solo-learn writes checkpoints under trained_models/<name>/<...>.ckpt
for m in "${METHODS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    ckpt="$(ls -t trained_models/${m}-cifar100-${v}/*.ckpt 2>/dev/null | head -1 || true)"
    if [ -z "$ckpt" ]; then
      echo "!!! no checkpoint for ${m}-cifar100-${v} — skipping"
      continue
    fi
    echo ">>> RankMe: $m / $v  ($ckpt)"
    python "$QDIR/compute_rankme.py" --ckpt "$ckpt" --data ./datasets --n 25600
  done
done

echo "=== done. Fill linear_acc + geom_metric into $QDIR/results_table.md ==="
