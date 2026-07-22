"""VICReg-exp loss (drop-in for solo-learn).

Replaces VICReg's squared-Frobenius covariance term with the LogSumExp
("exponential") repulsive criterion from Garrido et al., "On the duality between
contrastive and non-contrastive self-supervised learning" (arXiv:2206.02574):

    c_exp(K) = (1/d) * sum_i  log( sum_{j != i} exp( Cov(K)_{i,j} / tau ) )

Invariance and variance terms are identical to standard VICReg. The signature matches
solo.losses.vicreg.vicreg_loss_func plus an extra `tau`.

NOTE 1 (verify): cross-check the exact normalization of Cov and the value of `tau`
against Appendix L of the paper before trusting absolute numbers — the *relative*
ordering across the 5 hyperparameter runs (which is what the quality study needs) is
robust to these details.
NOTE 2 (multi-GPU): this computes cov on the local batch only. For >1 GPU, gather z1/z2
across devices first (solo-learn's real vicreg_loss_func uses `gather`). On CIFAR /
single GPU (devices: [0]) nothing to do.
"""
import torch
import torch.nn.functional as F


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    return torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))


def _exp_covariance(z: torch.Tensor, tau: float) -> torch.Tensor:
    """LogSumExp covariance criterion on a (batch, dim) matrix -> scalar."""
    N, D = z.size()
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)                                   # (D, D)
    off_diag = ~torch.eye(D, dtype=torch.bool, device=z.device)
    logits = (cov / tau).masked_fill(~off_diag, float("-inf"))  # drop the diagonal
    # log sum_{j!=i} exp(cov_ij / tau), then (1/d) * sum_i  == mean over rows
    return torch.logsumexp(logits, dim=1).mean()


def exp_covariance_loss(z1, z2, tau: float = 0.5) -> torch.Tensor:
    return _exp_covariance(z1, tau) + _exp_covariance(z2, tau)


def vicreg_exp_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
    tau: float = 0.5,
) -> torch.Tensor:
    sim_loss = invariance_loss(z1, z2)
    var_loss = variance_loss(z1, z2)
    cov_loss = exp_covariance_loss(z1, z2, tau=tau)
    return sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    z1 = torch.randn(256, 2048)
    z2 = torch.randn(256, 2048)
    loss = vicreg_exp_loss_func(z1, z2)
    print("vicreg-exp loss:", float(loss), "| finite:", bool(torch.isfinite(loss)))
