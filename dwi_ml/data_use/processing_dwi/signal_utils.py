"""Signal and anatomy related utilities."""
from __future__ import annotations

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
                            sh_order: int = 8, regul_coeff: float = 0.006)\
        -> np.ndarray:
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


def compute_dwi_attenuation(dwi_weights: np.ndarray, b0: np.ndarray)\
        -> np.ndarray:
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


def normalize_data_volume(data: np.ndarray, mask: Optional[np.ndarray] = None)\
        -> np.ndarray:
    """Apply classic data standardization (zero-centering and variance
    normalization) to every modality (last axis), restricted to a mask if
    provided.

    Parameters
    ----------
    data : np.ndarray with shape (X, Y, Z, #modalities)
        Volume to normalize along each modality.
    mask : binary np.ndarray with shape (X, Y, Z)
        3D mask defining which voxels should be used for normalization. If None,
        all non-zero voxels will be used.

    Returns
    -------
    normalized_data : np.ndarray with shape (X, Y, Z, #modalities)
        Normalized data volume, with zero-mean and unit variance along each
        axis of the last dimension.
    """
    # Normalization in each direction (zero mean and unit variance)
    if mask is None:
        # If no mask is given, use non-zero data voxels
        mask = np.zeros(data.shape[:3], dtype=np.int)
        nonzero_idx = np.nonzero(data.sum(axis=-1))
        mask[nonzero_idx] = 1
    else:
        # Mask resolution must fit DWI resolution
        assert mask.shape == data.shape[:3], "Normalization mask resolution " \
                                             "does not fit data..."

    idx = np.nonzero(mask)
    mean = np.mean(data[idx], axis=0)
    std = np.std(data[idx], axis=0)

    normalized_data = (data - mean) / (std + 1e-3)

    return normalized_data


def filter_bvalue(dwi_image: nib.Nifti1Image, gradient_table: GradientTable,
                  bval_filter: int) -> Tuple[nib.Nifti1Image, GradientTable]:
    """Filter a Nifti image and a GradientTable for the given b-value (and keep
    the b0).

    Parameters
    ----------
    dwi_image : nib.Nifti1Image
        The input image.
    gradient_table : dipy.io.gradients.GradientTable
        The input GradientTable.
    bval_filter : int
        The b-value to use as filter.

    Returns
    -------
    filtered_image : nib.Nifti1Image
        The filtered weights as a Nifti image.
    filtered_gradient_table : dipy.io.gradients.GradientTable
        The filtered gradient table.
    """
    eps = 10.
    bvals = gradient_table.bvals
    bvecs = gradient_table.bvecs

    bvals_mask = np.logical_and(bvals > (bval_filter - eps),
                                bvals < (bval_filter + eps))
    bvals_and_b0_mask = np.logical_or(gradient_table.b0s_mask, bvals_mask)
    filtered_bvals = bvals[bvals_and_b0_mask]
    filtered_bvecs = bvecs[bvals_and_b0_mask]
    filtered_weights = \
        dwi_image.get_fdata(dtype=np.float32)[..., bvals_and_b0_mask]

    filtered_image = nib.Nifti1Image(filtered_weights, dwi_image.affine,
                                     dwi_image.header)
    filtered_gradient_table = create_gradient_table(filtered_bvals,
                                                    filtered_bvecs)

    return filtered_image, filtered_gradient_table


def resample_dwi(dwi_image: nib.Nifti1Image, gradient_table: GradientTable,
                 directions: Sphere = None, sh_order: int = 8,
                 regul_coeff: float = 0.006) -> np.ndarray:
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


def compute_ssst_frf(dwi_image: nib.Nifti1Image, gradient_table: GradientTable,
                     wm_mask_image: nib.Nifti1Image = None) -> np.ndarray:
    """Compute a single Fiber Response Function from a DWI.

    A DTI fit is made, and voxels containing a single fiber population are
    found using a threshold on the FA.
    This function was mostly taken from from
    `www.github.com/scilus/scilpy/scripts/scil_compute_ssst_frf.py
    We could instead import the main, but easier to copy. Not so many chances
    that scil's function will change strongly.

    Parameters
    ----------
    dwi_image : nib.Nifti1Image
        Diffusion signal as weighted images (4D).
    gradient_table : GradientTable
        Dipy object that contains all bvals and bvecs.
    wm_mask_image : nib.Nifti1Image
        Binary WM mask. Only the data inside this mask will be used to
        estimate the fiber response function.

    Returns
    -------
    full_response : np.ndarray with shape (4,)
        Fiber response function
    """
    if wm_mask_image:
        if dwi_image.shape[:3] != wm_mask_image.shape:
            raise ValueError("DWI and mask shape do not match! "
                             "Got: dwi.shape={}; "
                             "mask.shape={}".format(dwi_image.shape,
                                                    wm_mask_image.shape))
        data = applymask(dwi_image.get_fdata(), wm_mask_image.get_fdata())
    else:
        data = dwi_image.get_fdata()

    # Use default parameters
    start_fa_thresh = fa_thresh = 0.7
    min_fa_thresh = 0.5
    min_nvox = 300
    roi_radius_vox = 10
    max_roi_radius_vox = 15

    # Iteratively try to fit at least 300 voxels.
    # Lower the FA threshold when it doesn't work.
    # Fail if the fa threshold is smaller than the min_threshold.
    # We use an epsilon since the -= 0.05 might incur numerical imprecisions.
    nvox = 0
    while nvox < min_nvox and roi_radius_vox <= max_roi_radius_vox:
        while nvox < min_nvox and fa_thresh >= min_fa_thresh - 0.00001:
            response, ratio, nvox = auto_response(gradient_table, data,
                                                  roi_radius=roi_radius_vox,
                                                  fa_thr=fa_thresh,
                                                  return_number_of_voxels=True)

            logging.debug(
                'Number of voxels is {} with FA threshold of {} and radius of '
                '{} vox'.format(nvox, fa_thresh, roi_radius_vox))
            fa_thresh -= 0.05
        fa_thresh = start_fa_thresh
        roi_radius_vox += 1

    if nvox < min_nvox:
        raise ValueError(
            "Could not find at least {} voxels with sufficient FA "
            "to estimate the FRF!".format(min_nvox))

    logging.debug("Found %i voxels with FA threshold %f for FRF estimation",
                  nvox, fa_thresh + 0.05)
    logging.debug("FRF eigenvalues: %s", str(response[0]))
    logging.debug("Ratio for smallest to largest eigen value is %f", ratio)
    logging.debug("Mean of the b=0 signal for voxels used for FRF: %f",
                  response[1])

    full_response = np.array([response[0][0], response[0][1],
                              response[0][2], response[1]])

    return full_response


def compute_fodf(dwi_image: nib.Nifti1Image, gradient_table: GradientTable,
                 full_frf: np.ndarray, sh_order: int, n_peaks: int = 3,
                 mask_image: nib.Nifti1Image = None, return_sh: bool = True):
    data = dwi_image.get_fdata()
    mask_data = None
    if mask_image:
        mask_data = mask_image.get_fdata()

    # Raise warning for sh order if there is not enough DWIs
    if data.shape[-1] < (sh_order + 1) * (sh_order + 2) / 2:
        logging.warning(
            'We recommend having at least {} unique DWIs volumes, but you '
            'currently have {} volumes. Try lowering the parameter --sh_order '
            'in case of non convergence.'.format(
                (sh_order + 1) * (sh_order + 2) / 2, data.shape[-1]))

    frf = full_frf[:3]
    mean_b0_val = full_frf[3]

    reg_sphere = get_sphere('symmetric362')
    peaks_sphere = get_sphere('symmetric724')

    csd_model = ConstrainedSphericalDeconvModel(
        gradient_table, (frf, mean_b0_val),
        reg_sphere=reg_sphere,
        sh_order=sh_order)

    # Run in parallel, using the default number of processes (default: CPU
    # count)
    peaks_csd = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=peaks_sphere,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 mask=mask_data,
                                 return_sh=return_sh,
                                 sh_basis_type="tournier07",
                                 sh_order=sh_order,
                                 normalize_peaks=True,
                                 npeaks=n_peaks,
                                 parallel=True)

    return peaks_csd
