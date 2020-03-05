# -*- coding: utf-8 -*-
from dipy.core.gradients import gradient_table
import numpy as np

# Note. If you want to resample DWI: you may use
# from scilpy.reconst.raw_signal import compute_sh_coefficients
# with the sphere
# from dipy.data import get_sphere
# sphere = get_sphere("repulsion100")

# Example of tools that can be useful.
from scilpy.reconst.fodf import compute_fodf
from scilpy.reconst.raw_signal import compute_sh_coefficients
from scilpy.reconst.frf import compute_ssst_frf
from scilpy.tracking.tools import (
    filter_streamlines_by_length, resample_streamlines_step_size)
from scilpy.utils.streamlines import compress_sft
from scilpy.image.resample_volume import resample_volume


def resample_raw_dwi_from_sh(dwi_image, bvals, bvecs, sh_order):
    # Brings to SH and then back to directions.
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
    output = compute_sh_coefficients(dwi_image, gtab, sh_order=sh_order)

    # Then what?
    raise NotImplementedError

    return output


def standardize_data(data: np.ndarray, mask: np.ndarray = None,
                     independent: bool = False):
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
    independent: bool
        If true, will normalize each modality independently (last axis). Else,
        will normalize with the mean and variance of all data. There is a
        big reflexion to have here. Typical approach in machine learning is to
        normalize each input X separately (each modality). But our data is not
        independent. Ex, with peaks, the peak in one direction and the peak in
        another must probably belong to the same distribution to mean something.
        We recommend using independent = False for your dwi data.

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
    if independent:
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
