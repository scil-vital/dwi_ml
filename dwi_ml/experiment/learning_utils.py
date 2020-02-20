def compute_gradient_norm(parameters):
    """Compute the gradient norm of the provided iterable parameters.
    (Machine learning gradient descent, not dwi gradients!)

    Parameters
    ----------
    parameters : list of torch.Tensor
        Model parameters after loss.backwards() has been called

    Returns
    -------
    total_norm : float
        The total gradient norm of the parameters
    """
    total_norm = 0.
    for p in parameters:
        param_norm = p.grad.as_tensor.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 3)
    return total_norm
