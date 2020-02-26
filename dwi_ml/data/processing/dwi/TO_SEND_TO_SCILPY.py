"""Signal and anatomy related utilities."""

import logging
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from dipy.core.gradients import GradientTable, \
    gradient_table as create_gradient_table
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response
from dipy.reconst.shm import smooth_pinv, sph_harm_lookup
from dipy.segment.mask import applymask


def compute_sh_coefficients(dwi: nib.Nifti1Image, gradient_table: GradientTable,
                            sh_order: int = 8, regul_coeff: float = 0.006):
    """Fit a diffusion signal with spherical harmonics coefficients.

    Parameters
    ----------
    dwi : nib.Nifti1Image object
        Diffusion signal as weighted images (4D).
    gradient_table : GradientTable
        Dipy object that contains all bvals and bvecs.
    sh_order : int, optional
        SH order to fit, by default 8. If zero, return dwi weights after b0
        attenuation.
    regul_coeff : float, optional
        Lambda-regularization coefficient in the SH fit, by default 0.006.

    Returns
    -------
    sh_coeffs : np.ndarray with shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number
        of coefficients depends on `sh_order`.
    """
    bvecs = gradient_table.bvecs
    dwi_weights = dwi.get_fdata(dtype=np.float32)

    # Exract the averaged b0.
    b0_mask = gradient_table.b0s_mask
    b0 = dwi_weights[..., b0_mask].mean(axis=3)

    # Extract diffusion weights and compute attenuation using the b0.
    bvecs = bvecs[np.logical_not(b0_mask)]
    weights = dwi_weights[..., np.logical_not(b0_mask)]

    weights = compute_dwi_attenuation(weights, b0)

    if sh_order > 0:
        # Get cartesian coords from bvecs
        raw_sphere = Sphere(xyz=bvecs)

        # Get spherical harmonics basis function
        sph_harm_basis = sph_harm_lookup.get('tournier07')

        # Words "degree" and "order" chosen according to (Descoteaux's thesis);
        # DIPY seems to reverse the definitions...
        # B : 2-D array; real harmonics sampled at (\theta, \phi)
        # m : array; degree of the sampled harmonics.
        # l : array; order of the sampled harmonics.
        B, m, l = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)

        # Compute regularization matrix
        diag_L = -l * (l + 1)

        # Compute pseudo-inverse
        invB = smooth_pinv(B, np.sqrt(regul_coeff) * diag_L)

        # Fit coefficients to signal
        data = np.dot(weights, invB.T)
    else:
        data = weights

    return data


def compute_dwi_attenuation(dwi_weights: np.ndarray, b0: np.ndarray):
    """ Compute signal attenuation by dividing the dwi signal with the b0.

    Parameters:
    -----------
    dwi_weights : np.ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : np.ndarray of shape (X, Y, Z)
        B0 image.

    Returns
    -------
    dwi_attenuation : np.ndarray
        Signal attenuation (Diffusion weights normalized by the B0).
    """
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    erroneous_voxels = np.any(dwi_weights > b0, axis=3)
    nb_erroneous_voxels = np.sum(erroneous_voxels)
    if nb_erroneous_voxels != 0:
        logging.info("# of voxels where `dwi_signal > b0` in any direction: "
                     "{}".format(nb_erroneous_voxels))
        dwi_weights = np.minimum(dwi_weights, b0)

    # Compute attenuation
    dwi_attenuation = dwi_weights / b0
    dwi_attenuation[np.logical_not(np.isfinite(dwi_attenuation))] = 0.

    return dwi_attenuation


def resample_dwi(dwi_image: nib.Nifti1Image, gradient_table: GradientTable,
                 directions: Sphere = None, sh_order: int = 8,
                 regul_coeff: float = 0.006):
    """Resample a diffusion signal according to a set of directions using
    spherical harmonics.

    Parameters
    ----------
    dwi_image : nib.Nifti1Image object
        Diffusion signal as weighted images (4D).
    gradient_table : GradientTable
        Dipy object that contains all bvals and bvecs.
    directions : dipy.core.sphere.Sphere, optional
        Directions the diffusion signal will be resampled to. Directions are
        assumed to be on the whole sphere, not the hemisphere like bvecs.
        If omitted, 100 directions evenly distributed on the sphere will be
        used.
    sh_order : int, optional
        SH order to fit, by default 8.
        attenuation.
    regul_coeff : float, optional
        Lambda-regularization coefficient in the SH fit, by default 0.006.

    Returns
    -------
    resampled_dwi : np.ndarray (4D)
        Resampled diffusion signal
    """
    # Get SH fit
    data_sh = compute_sh_coefficients(dwi_image, gradient_table, sh_order,
                                      regul_coeff)

    if directions is not None:
        sphere = directions
    else:
        # Get 100 directions sphere
        sphere = get_sphere("repulsion100")

    sh_basis = sph_harm_lookup.get("tournier07")

    # B : 2-D array; real harmonics sampled at (\theta, \phi)
    # m : array; degree of the sampled harmonics.
    # l : array; order of the sampled harmonics.
    B, m, l = sh_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, B.T)

    return data_resampled

