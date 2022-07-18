# -*- coding: utf-8 -*-
from collections import Iterator


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
    total_norm = 0.
    for p in parameters:
        # Possibly add: if hasattr(p, 'grad'):
        param_norm = p.grad.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm **= 1. / 2
    return total_norm
