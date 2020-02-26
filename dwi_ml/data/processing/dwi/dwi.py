
from dipy.data import get_sphere
from scilpy.reconst.raw_signal import compute_sh_coefficients
import numpy as np


def compute_sh_coefficients_resampled(dwi, gradient_table, sh_order=8,
                                      basis_type='descoteaux07', smooth=0.006,
                                      use_attenuation=False,
                                      force_b0_threshold=False, mask=None,
                                      sphere=None):
    """Just a small function to remember our default sphere for SH resampling
    on raw DWI. By default, we use "repulsion100".
    """
    if sphere is None:
        sphere = get_sphere("repulsion100")

    compute_sh_coefficients(dwi, gradient_table, sh_order, basis_type, smooth,
                            use_attenuation, force_b0_threshold, mask,
                            sphere=sphere)


def standardize_data(data: np.ndarray, mask: np.ndarray = None,
                     independant: bool = False):
    """Apply classic data standardization (centralized, normalized = zero-
    centering and variance to 1).

    Parameters
    ----------
    data : np.ndarray with shape (X, Y, Z, #modalities)
        Volume to normalize along each modality.
    mask : binary np.ndarray with shape (X, Y, Z)
        3D mask defining which voxels should be used for normalization. If None,
        all non-zero voxels will be used. Voxels outside mask will be set to
        nan.
    independant: bool
        If true, will normalize each modality independantly (last axis). Else,
        will normalize with the mean and variance of all data. There is a
        big reflexion to have here. Typical approach in machine learning is to
        normalize each input X separately (each modality). But our data is not
        independant. Ex, with peaks, the peak in one direction and the peak in
        another must probably belong to the same distribution to mean something.
        We recommand using independant = False for your dwi data.

    Returns
    -------
    normalized_data : np.ndarray with shape (X, Y, Z, #modalities)
        Normalized data volume, with zero-mean and unit variance along each
        axis of the last dimension.
    """
    if mask is None:
        # If no mask is given, use non-zero data voxels
        mask = np.all(data != 0, axis=-1)
    else:
        # Mask resolution must fit DWI resolution
        assert mask.shape == data.shape[:3], "Normalization mask resolution " \
                                             "does not fit data..."

    # Computing mean and std.
    # Also dealing with extreme cases where std=0. Shouldn't happen. It means
    # that this data is meaningless for your model. Here, we won't divide the
    # data, just move its mean = value in all voxels will now be 0.
    if independant:
        mean = np.mean(data[mask], axis=0)
        std = np.std(data[mask], axis=0)
        std[std == 0] = 1
    else:
        mean = np.mean(data[mask])
        std = np.std(data[mask])
        if std == 0:
            std = 1

    normalized_data = (data - mean) / std
    normalized_data[~mask] = np.nan

    return normalized_data
