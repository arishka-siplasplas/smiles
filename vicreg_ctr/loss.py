"""VICReg-ctr loss (drop-in for solo-learn).

Same LogSumExp criterion as VICReg-exp, but applied to the TRANSPOSED embedding matrix.
Transposing turns the dimension-contrastive penalty into a sample-contrastive one (a
Gram matrix over samples), making the method conceptually close to SimCLR. From Garrido
et al., arXiv:2206.02574:

    L_ctr = lambda * invariance(K, K')          # unchanged, on samples
          + mu     * ( v(K^T) + v(K'^T) )        # variance on transposed matrix
          + nu     * ( c_exp(K^T) + c_exp(K'^T) ) # exp-covariance on transposed matrix

See NOTE 1 / NOTE 2 in vicreg_exp/loss.py (tau/normalization + multi-GPU gather).
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
    cov = (z.T @ z) / (N - 1)
    off_diag = ~torch.eye(D, dtype=torch.bool, device=z.device)
    logits = (cov / tau).masked_fill(~off_diag, float("-inf"))
    return torch.logsumexp(logits, dim=1).mean()


def vicreg_ctr_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
    tau: float = 0.5,
) -> torch.Tensor:
    # invariance term stays on the sample rows, exactly like VICReg
    sim_loss = invariance_loss(z1, z2)
    # variance + exp-covariance act on the TRANSPOSED (sample-contrastive) matrix
    var_loss = variance_loss(z1.T, z2.T)
    cov_loss = _exp_covariance(z1.T, tau) + _exp_covariance(z2.T, tau)
    return sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    z1 = torch.randn(256, 2048)
    z2 = torch.randn(256, 2048)
    loss = vicreg_ctr_loss_func(z1, z2)
    print("vicreg-ctr loss:", float(loss), "| finite:", bool(torch.isfinite(loss)))
