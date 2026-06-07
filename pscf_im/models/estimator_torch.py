"""PyTorch PSCF-IM-VAE -- the production estimator used for the paper's numbers.

Architecture (Fig. 2, Stage 1):

* **GNN encoder** : ``L`` rounds of mean-aggregation message passing producing a
  node embedding ``h_v``.
* **Fair / unfair heads** : two Gaussian posteriors ``q(U_f|.) = N(mu_f, sig_f)``
  and ``q(U_b|.) = N(mu_b, sig_b)``.
* **Decoder** : ``p(Y | U_f, U_b, G_f^v)`` with the fair-only neighbour exposure
  ``G_f^v`` (mean of active in-neighbours' fair propensities).
* **Adversarial discriminator** behind a GRL : ``p_d(U_b | U_f, S)`` -> enforces
  ``U_f ⟂ U_b | S``.
* **Weak supervision** : Bernoulli head ``p(R_b | U_b)`` on flagged nodes.

Objective (Eq. 9 + regularisers)::

    L = E_q[log p(Y|U_f,U_b,G_f)] + E_q[log p(R_b|U_b)]
        - KL[q(U_f|.) || p(U_f|S,X)] - KL[q(U_b|.) || p(U_b|S)]
        + lambda_a * L_adv

Two-phase schedule (Alg. 1): phase 1 fits ``U_b`` on the flags and freezes the
unfair head; phase 2 learns the residual fair structure under the adversary.

This module imports torch lazily and is *not* needed for the NumPy reference
path; ``build_torch_estimator`` raises a clear error if torch is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TorchVAEConfig:
    latent_dim: int = 64
    hidden_dim: int = 64
    n_layers: int = 2
    lambda_adv: float = 0.1
    lambda_rb: float = 1.0
    lr: float = 1e-3
    epochs_phase1: int = 200    # T1: pin U_b on flags
    epochs_phase2: int = 200    # T2: learn residual U_f
    seed: int = 0


def build_torch_estimator(cfg: "TorchVAEConfig | None" = None):
    """Construct the PSCF-IM-VAE; requires PyTorch.

    Returns an object with ``.fit(graph, features, S, X, Rb, Rb_mask, cascades)``
    and ``.infer()`` -> ``(U_f, U_b)`` numpy arrays.  Kept inside a factory so
    importing :mod:`pscf_im` never hard-depends on torch.
    """
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The torch backend requires PyTorch. Install it (see requirements.txt) "
            "or use backend='numpy' (ReferenceEstimator)."
        ) from exc

    from .grl import make_grl

    cfg = cfg or TorchVAEConfig()
    GradientReversal = make_grl()

    class MeanGNNEncoder(nn.Module):
        """L-layer mean-aggregation GNN producing node embeddings."""

        def __init__(self, in_dim, hidden, layers):
            super().__init__()
            dims = [in_dim] + [hidden] * layers
            self.lins = nn.ModuleList(
                nn.Linear(2 * dims[i], dims[i + 1]) for i in range(layers)
            )
            self.act = nn.ReLU()

        def forward(self, x, A_norm):
            h = x
            for lin in self.lins:
                agg = A_norm @ h                      # mean of neighbours
                h = self.act(lin(torch.cat([h, agg], dim=-1)))
            return h

    class GaussianHead(nn.Module):
        def __init__(self, in_dim, latent):
            super().__init__()
            self.mu = nn.Linear(in_dim, latent)
            self.logvar = nn.Linear(in_dim, latent)

        def forward(self, h):
            mu, logvar = self.mu(h), self.logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return z, mu, logvar

    class PSCFIMVAE(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.encoder = MeanGNNEncoder(in_dim, cfg.hidden_dim, cfg.n_layers)
            self.fair_head = GaussianHead(cfg.hidden_dim, cfg.latent_dim)
            self.unfair_head = GaussianHead(cfg.hidden_dim, cfg.latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(3 * cfg.latent_dim, cfg.hidden_dim), nn.ReLU(),
                nn.Linear(cfg.hidden_dim, 1),
            )
            self.rb_head = nn.Linear(cfg.latent_dim, 1)      # p(R_b | U_b)
            self.grl = GradientReversal(1.0)
            self.disc = nn.Sequential(                        # p_d(U_b | U_f, S)
                nn.Linear(cfg.latent_dim + 1, cfg.hidden_dim), nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            )

        def fair_exposure(self, u_f, A_active):
            """G_f^v: mean of active in-neighbours' fair propensities."""
            return A_active @ u_f

        def forward(self, x, A_norm, A_active, S):
            h = self.encoder(x, A_norm)
            u_f, mu_f, lv_f = self.fair_head(h)
            u_b, mu_b, lv_b = self.unfair_head(h)
            g_f = self.fair_exposure(u_f, A_active)
            y_logit = self.decoder(torch.cat([u_f, u_b, g_f], dim=-1)).squeeze(-1)
            rb_logit = self.rb_head(u_b).squeeze(-1)
            u_b_pred = self.disc(torch.cat([self.grl(u_f), S[:, None]], dim=-1))
            return dict(u_f=u_f, mu_f=mu_f, lv_f=lv_f, u_b=u_b, mu_b=mu_b,
                        lv_b=lv_b, y_logit=y_logit, rb_logit=rb_logit,
                        u_b_pred=u_b_pred)

    class TorchEstimator:
        """Thin training/inference wrapper around :class:`PSCFIMVAE`."""

        def __init__(self):
            torch.manual_seed(cfg.seed)
            self.model = None
            self._cache = None

        def _prep(self, graph, features, S):
            import numpy as np
            n = graph.number_of_nodes()
            A = np.zeros((n, n), dtype="float32")
            for u, v in graph.edges():
                A[v, u] = 1.0
            A_norm = A + np.eye(n, dtype="float32")
            A_norm /= A_norm.sum(1, keepdims=True)
            A_active = A / np.maximum(A.sum(1, keepdims=True), 1)
            return (torch.tensor(features, dtype=torch.float32),
                    torch.tensor(A_norm), torch.tensor(A_active),
                    torch.tensor(S, dtype=torch.float32))

        def fit(self, graph, features, S, y_target, Rb, Rb_mask):
            x, A_norm, A_active, St = self._prep(graph, features, S)
            self.model = PSCFIMVAE(in_dim=features.shape[1])
            opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
            y = torch.tensor(y_target, dtype=torch.float32)
            rb = torch.tensor(Rb, dtype=torch.float32)
            mask = torch.tensor(Rb_mask, dtype=torch.bool)
            bce = nn.BCEWithLogitsLoss()

            def step(use_adv: bool, freeze_unfair: bool):
                opt.zero_grad()
                out = self.model(x, A_norm, A_active, St)
                loss = bce(out["y_logit"], y)
                if mask.any():
                    loss = loss + cfg.lambda_rb * bce(out["rb_logit"][mask], rb[mask])
                # KL to standard-normal priors (proportional surrogate)
                for mu, lv in [(out["mu_f"], out["lv_f"]), (out["mu_b"], out["lv_b"])]:
                    loss = loss + (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean())
                if use_adv:
                    loss = loss + cfg.lambda_adv * (
                        out["u_b_pred"].detach() - out["u_b"].detach()
                    ).pow(0).mean() * 0  # placeholder keeps graph; adv via GRL term
                    loss = loss + cfg.lambda_adv * \
                        nn.functional.mse_loss(out["u_b_pred"], out["u_b"].detach())
                if freeze_unfair:
                    for p in list(self.model.unfair_head.parameters()) + \
                            list(self.model.rb_head.parameters()):
                        if p.grad is not None:
                            p.grad = None
                loss.backward()
                if freeze_unfair:
                    for p in list(self.model.unfair_head.parameters()) + \
                            list(self.model.rb_head.parameters()):
                        p.grad = None
                opt.step()
                return float(loss.item())

            for _ in range(cfg.epochs_phase1):       # phase 1: pin U_b
                step(use_adv=False, freeze_unfair=False)
            for _ in range(cfg.epochs_phase2):       # phase 2: residual U_f
                step(use_adv=True, freeze_unfair=True)
            self._cache = (x, A_norm, A_active, St)
            return self

        @torch.no_grad()
        def infer(self):
            import numpy as np
            x, A_norm, A_active, St = self._cache
            out = self.model(x, A_norm, A_active, St)
            u_f = out["u_f"].mean(dim=-1).numpy()
            u_b = out["u_b"].mean(dim=-1).numpy()

            def _mm(z):
                r = z.max() - z.min()
                return (z - z.min()) / r if r > 1e-12 else np.zeros_like(z)

            return _mm(u_f), _mm(u_b)

    return TorchEstimator()
