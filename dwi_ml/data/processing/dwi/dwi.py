# -*- coding: utf-8 -*-
from dipy.core.gradients import GradientTable
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_lookup
import nibabel as nib
import numpy as np
from scilpy.io.utils import validate_sh_basis_choice
from scilpy.reconst.raw_signal import compute_sh_coefficients


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


def resample_raw_dwi_from_sh(dwi_image: nib.Nifti1Image,
                             gradient_table: GradientTable,
                             sh_basis: str = 'descoteaux07',
                             sphere: Sphere = None, sh_order: int = 8,
                             smooth: float = 0.006):
    """Resample a diffusion signal according to a set of directions using
    spherical harmonics.

    Parameters
    ----------
    dwi_image : nib.Nifti1Image object
        Diffusion signal as weighted images (4D).
    gradient_table : GradientTable
        Dipy object that contains all bvals and bvecs.
    sh_basis: str
        Either 'tournier07' or 'descoteaux07'. Default: descoteaux07.
    sphere : dipy.core.sphere.Sphere, optional
        Directions the diffusion signal will be resampled to. Directions are
        assumed to be on the whole sphere, not the hemisphere like bvecs.
        If omitted, 100 directions evenly distributed on the sphere will be
        used (Dipy's "repulsion100").
    sh_order : int, optional
        SH order to fit, by default 8.
    smooth : float, optional
        Lambda-regularization coefficient in the SH fit, by default 0.006.

    Returns
    -------
    resampled_dwi : np.ndarray (4D)
        Resampled "raw" diffusion signal.
    """
    validate_sh_basis_choice(sh_basis)

    # Get the "real" SH fit
    # sphere = None, so it computes the sh coefficients based on the bvecs.
    data_sh = compute_sh_coefficients(dwi_image, gradient_table, sh_order,
                                      basis_type=sh_basis, smooth=smooth)

    # Get new directions
    if sphere is None:
        sphere = get_sphere("repulsion100")
    sh_basis = sph_harm_lookup.get(sh_basis)

    # Resample data
    # B.T contains the new sampling scheme and B*data_sh projects to the sphere.
    # B : 2-D array; real harmonics sampled at (\theta, \phi)
    # m : array; degree of the sampled harmonics.
    # l : array; order of the sampled harmonics.
    B, _, _ = sh_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, B.T)

    return data_resampled
