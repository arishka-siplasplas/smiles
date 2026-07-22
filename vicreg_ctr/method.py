"""Registers `vicreg_ctr` as a solo-learn method.

Reuses the entire VICReg method and swaps only the loss for the contrastive (transposed)
variant. Import once before the model is built:

    import vicreg_ctr.method   # noqa: F401  (registers "vicreg_ctr")

See vicreg_exp/method.py for the rationale behind the monkeypatch approach.
"""
import functools

import solo.methods.vicreg as vicreg_module
from solo.methods import METHODS
from solo.methods.vicreg import VICReg

try:
    from .loss import vicreg_ctr_loss_func
except ImportError:
    from loss import vicreg_ctr_loss_func


class VICRegCtr(VICReg):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.tau = float(cfg.method_kwargs.get("tau", 0.5))

    def training_step(self, batch, batch_idx):
        vicreg_module.vicreg_loss_func = functools.partial(
            vicreg_ctr_loss_func, tau=self.tau
        )
        return super().training_step(batch, batch_idx)


METHODS["vicreg_ctr"] = VICRegCtr
