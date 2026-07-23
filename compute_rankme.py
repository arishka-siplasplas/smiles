"""Compute RankMe (and a couple of cheap sibling metrics) for the frozen backbones.

RankMe (Garrido et al., 2023, https://proceedings.mlr.press/v202/garrido23a/garrido23a.pdf)
is the *smooth effective rank* of the embedding matrix — a label-free score that tracks
downstream accuracy:

    p_k    = sigma_k / sum_j sigma_j  + eps           (sigma = singular values of Z)
    RankMe = exp( - sum_k p_k * log p_k )

Two layers, so this stays usable/testable WITHOUT solo-learn installed:
  1. pure metric functions on an embedding matrix Z  (N x D)
  2. extraction: build the CIFAR ResNet-18, load a solo-learn checkpoint, run over CIFAR-100.

Examples
--------
    # smoke test, no dataset / no checkpoint needed:
    python compute_rankme.py --random

    # real run on a trained backbone:
    python compute_rankme.py --ckpt trained_models/vicreg-cifar100-base/xxx.ckpt \
                             --data ./datasets --n 25600
"""
import argparse

import torch


# --------------------------------------------------------------------------- #
# metrics (pure — operate on an embedding matrix Z of shape (N, D))
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rankme(Z: torch.Tensor, eps: float = 1e-7) -> float:
    """Smooth effective rank (RankMe). Higher = richer / less collapsed embeddings."""
    Z = Z.to(torch.float64)
    s = torch.linalg.svdvals(Z)                    # singular values, length min(N, D)
    p = s / (s.sum() + eps) + eps
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


@torch.no_grad()
def stable_rank(Z: torch.Tensor) -> float:
    """||Z||_F^2 / ||Z||_2^2 — cheap secondary sanity metric."""
    Z = Z.to(torch.float64)
    s = torch.linalg.svdvals(Z)
    return float((s.pow(2).sum()) / (s.max().pow(2)))


# --------------------------------------------------------------------------- #
# intrinsic-dimension metrics (computed on a random subsample for tractability)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def twonn(Z: torch.Tensor, n_sub: int = 2000, discard_frac: float = 0.1) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al., 2017). Uses the ratio of the
    2nd- to 1st-nearest-neighbour distance per point, then a through-origin fit of
    -log(1 - F(mu)) vs log(mu). Lower = lower intrinsic dimension."""
    Z = Z.to(torch.float32)
    if Z.size(0) > n_sub:
        Z = Z[torch.randperm(Z.size(0))[:n_sub]]
    D = torch.cdist(Z, Z)
    D.fill_diagonal_(float("inf"))
    r1, a1 = D.min(dim=1)
    D[torch.arange(D.size(0)), a1] = float("inf")   # mask the 1st NN to get the 2nd
    r2, _ = D.min(dim=1)
    mu = r2 / r1.clamp_min(1e-12)
    mu = mu[torch.isfinite(mu) & (mu > 1)]
    mu, _ = torch.sort(mu)
    n = mu.numel()
    F = torch.arange(1, n + 1, dtype=torch.float32, device=mu.device) / n
    keep = int(n * (1 - discard_frac))              # drop the noisy tail
    x = torch.log(mu[:keep])
    y = -torch.log((1 - F[:keep]).clamp_min(1e-12))
    return float((x * y).sum() / (x * x).sum())     # slope through origin = d


@torch.no_grad()
def mle_id(Z: torch.Tensor, k: int = 20, n_sub: int = 2000) -> float:
    """Levina-Bickel MLE intrinsic-dimension estimate over k nearest neighbours."""
    Z = Z.to(torch.float32)
    if Z.size(0) > n_sub:
        Z = Z[torch.randperm(Z.size(0))[:n_sub]]
    D = torch.cdist(Z, Z)
    D.fill_diagonal_(float("inf"))
    knn, _ = torch.sort(D, dim=1)
    knn = knn[:, :k].clamp_min(1e-12)
    logs = torch.log(knn[:, -1:] / knn[:, :-1]).sum(dim=1)
    m = (k - 1) / logs.clamp_min(1e-12)
    return float(m[torch.isfinite(m)].mean())


# --------------------------------------------------------------------------- #
# backbone
# --------------------------------------------------------------------------- #
def build_cifar_resnet18() -> torch.nn.Module:
    """torchvision ResNet-18 with the CIFAR stem solo-learn uses (3x3 conv1, no maxpool),
    fc replaced by Identity -> 512-d features."""
    import torch.nn as nn
    from torchvision.models import resnet18

    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net


def load_backbone_from_checkpoint(ckpt_path: str) -> torch.nn.Module:
    """Load a solo-learn Lightning .ckpt and pull the `backbone.*` weights into the net."""
    net = build_cifar_resnet18()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    backbone_state = {
        k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(
            f"No 'backbone.*' keys in {ckpt_path}. Keys look like: "
            f"{list(state.keys())[:5]} ..."
        )
    missing, unexpected = net.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"[warn] missing keys when loading backbone: {missing[:5]} ...")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]} ...")
    return net


# --------------------------------------------------------------------------- #
# embedding extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples: int) -> torch.Tensor:
    model.eval().to(device)
    feats, seen = [], 0
    for x, _ in loader:
        z = model(x.to(device)).flatten(1).cpu()
        feats.append(z)
        seen += z.size(0)
        if seen >= max_samples:
            break
    return torch.cat(feats, 0)[:max_samples]


def cifar100_loader(root: str, batch_size: int):
    from torchvision import datasets, transforms

    # RankMe uses the *representation* on clean eval-transformed images
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    ds = datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="solo-learn .ckpt; omit with --random")
    ap.add_argument("--data", default="./datasets")
    ap.add_argument("--n", type=int, default=25600, help="samples for the SVD")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--random", action="store_true",
                    help="smoke test: random ResNet-18 on random images, no data/ckpt")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    if args.random:
        model = build_cifar_resnet18()
        n = min(args.n, 4096)  # keep the smoke test quick
        x = torch.randn(n, 3, 32, 32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, torch.zeros(n)),
            batch_size=args.batch_size,
        )
    else:
        if not args.ckpt:
            ap.error("provide --ckpt (or use --random)")
        model = load_backbone_from_checkpoint(args.ckpt)
        loader = cifar100_loader(args.data, args.batch_size)

    Z = extract_embeddings(model, loader, device, args.n)
    print(f"embeddings: {tuple(Z.shape)}")
    print(f"RankMe:       {rankme(Z):8.3f}")
    print(f"stable_rank:  {stable_rank(Z):8.3f}")
    print(f"TwoNN_id:     {twonn(Z):8.3f}")
    print(f"MLE_id:       {mle_id(Z):8.3f}")


if __name__ == "__main__":
    main()
