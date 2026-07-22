"""Registers `vicreg_exp` as a solo-learn method.

It reuses the ENTIRE VICReg method (projector, LARS optimizer, warmup-cosine schedule,
online linear classifier, training loop) and only swaps the loss function. Import this
module once, before the model is built, in your training entrypoint:

    import vicreg_exp.method   # noqa: F401  (registers "vicreg_exp")

Then `method: "vicreg_exp"` in the yaml just works.

Why the monkeypatch: solo-learn's VICReg.training_step looks up the module-global name
`vicreg_loss_func` in `solo.methods.vicreg`. Rebinding that name there (with tau bound
from cfg) makes the stock training loop use our loss without copying its body — so this
stays robust across solo-learn versions. Verify the import path below matches your repo
layout (or drop these files inside the solo package).
"""
import functools

import solo.methods.vicreg as vicreg_module
from solo.methods import METHODS
from solo.methods.vicreg import VICReg

try:
    from .loss import vicreg_exp_loss_func
except ImportError:  # when run as a top-level module
    from loss import vicreg_exp_loss_func


class VICRegExp(VICReg):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.tau = float(cfg.method_kwargs.get("tau", 0.5))

    def training_step(self, batch, batch_idx):
        # Bind tau, then delegate to stock VICReg.training_step, which calls the
        # (now patched) module-global vicreg_loss_func.
        vicreg_module.vicreg_loss_func = functools.partial(
            vicreg_exp_loss_func, tau=self.tau
        )
        return super().training_step(batch, batch_idx)


METHODS["vicreg_exp"] = VICRegExp
