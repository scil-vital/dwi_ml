# -*- coding: utf-8 -*-
from collections.abc import Iterator

import torch


def compute_gradient_norm(parameters: Iterator):
    """Compute the gradient norm of the provided iterable parameters.
    (Machine learning gradient descent, not dwi gradients!)

    Parameters
    ----------
    parameters : Iterable
        Model parameters after loss.backwards() has been called. All parameters
        p must have a p.grad attribute. Ex: using model.parameters() returns
        an Iterable.

    Returns
    -------
    total_norm : float
        The total gradient norm of the parameters, i.e. sqrt(sum(params^2)),
        similarly as done in
        torch.nn.utils_to_refactor.clip_grad.clip_grad_norm.
    """
    norm_type = 2.0
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
        norm_type)

    return total_norm.cpu()
