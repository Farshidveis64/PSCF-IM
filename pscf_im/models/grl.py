"""Gradient Reversal Layer (Ganin & Lempitsky) for adversarial disentanglement.

During the forward pass it is the identity; during the backward pass it negates
and scales the gradient by ``lambda_``.  Placing a GRL before the discriminator
``p_d(U_b | U_f, S)`` makes the encoder learn fair propensities ``U_f`` that the
discriminator *cannot* use to predict ``U_b`` -- the empirical realisation of
``U_f ⟂ U_b | S`` (Prop. 2).  Imported lazily so the package works without torch.
"""
from __future__ import annotations


def make_grl():  # pragma: no cover - exercised only when torch is installed
    """Return a ``GradientReversal`` ``nn.Module`` (raises if torch is absent)."""
    import torch
    from torch import nn
    from torch.autograd import Function

    class _GradReverse(Function):
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambda_, None

    class GradientReversal(nn.Module):
        def __init__(self, lambda_: float = 1.0):
            super().__init__()
            self.lambda_ = lambda_

        def forward(self, x):
            return _GradReverse.apply(x, self.lambda_)

    return GradientReversal
