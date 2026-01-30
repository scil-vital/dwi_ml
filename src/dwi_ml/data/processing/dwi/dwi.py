# -*- coding: utf-8 -*-
from dipy.core.gradients import GradientTable
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_lookup
import nibabel as nib
import numpy as np

from scilpy.io.utils import validate_sh_basis_choice
from scilpy.reconst.sh import compute_sh_coefficients

eps = 1e-6


class OnlineMeanAndVariance:
    """
    Class to help standardize data volumes. This procedure computes the mean and 
    variance of the training set; a single volume (single subject) at once. Can
    be computed during the preparation of the hdf5, for instance, where we 
    iterate over volumes, rather than needing to load all data at once in
    memory.

    At each iteration, we update the mean and variance from a batch of data (all
    voxels in that volume).

    Solution 1:  (method = 'welford')
    If we wanted to update a single observation at the time:  See the Welford 
    Variance algorithm:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    So we can loop over voxels and use that algorithm (slow).

    Solution 2:  (method = 'batch')
    See mathematics here: 
    https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

    """
    def __init__(self, nb_features, method='batch'):
        assert method in ['batch', 'welford'] 

        # Count = the total number of voxels across all subjects, all volumes
        self.count = 0    
        self.mean = 0.0
        self._variance = 0.0
        self.method = method

        # We will compute the mean and variance over all voxels, all features
        # This is to help supervise shape.
        self.nb_features = nb_features

    def update_from_new_subject(self, x: np.ndarray):
        """
        Parameters
        ----------
        x: np.ndarray
            A volume for one subject.
        """
        assert (x.shape[-1] == self.nb_features)
        x = x.flatten()

        if self.method == 'batch':
            self.add_variable_batch(x)
        else:
            self.add_variable_welford(x)

    def add_variable_batch(self, x):
        m = self.count
        mu1 = self.mean
        s1 = self._variance

        n = len(x)
        mu2 = np.mean(x)
        s2 = np.std(x) ** 2

        self.count = m + n
        self.mean = m/self.count*mu1 + n/self.count*mu2
        self._variance = m/self.count*s1 + n/self.count*s2 + \
            m*n/(self.count**2) * (mu1 - mu2)**2

    def add_variable_welford(self, x):
        # Currently looping over voxels. Not sure if it can be accelerated.
        for xi in x:
            self.count += 1
            delta = (x - self.mean)
            self.mean += delta / self.count
            self._variance += delta * (x - self.mean)
       
    @property
    def variance(self) -> float:
        if self.method == 'batch':
            return self._variance
        else:
            return self._variance / self.count
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    

def standardize_data(data: np.ndarray, mask: np.ndarray = None,
                     independent: bool = False):
    """Apply classic data standardization (centralized, normalized = zero-
    centering and variance to 1) to a single volume.

    Parameters
    ----------
    data : np.ndarray with shape (X, Y, Z, #modalities)
        Volume to normalize along each modality.
    mask : binary np.ndarray with shape (X, Y, Z)
        3D mask defining which voxels should be used for normalization. If
        None, all non-zero voxels will be used. Voxels outside mask could be
        set to NaN, but then all streamlines touching these voxels (ex, at
        their extremities) would have associated inputs containing NaN. Simply
        standardizing them with the computed mean and std.
    independent: bool
        If true, will normalize each modality independently (last axis). Else,
        will normalize with the mean and variance of all data. There is a
        big reflexion to have here. Typical approach in machine learning is to
        normalize each input X separately (each modality). But our data is not
        independent. Ex, with peaks, the peak in one direction and the peak in
        another must probably belong to the same distribution to mean
        something. We recommend using independent = False for your dwi data.

    Returns
    -------
    standardized_data : np.ndarray with shape (X, Y, Z, #modalities)
        Standardized data volume, with zero-mean and unit variance along each
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
        # data[mask] becomes a 2D array. Taking axis 0 = the voxels.
        mean = np.mean(data[mask], axis=0)
        std = np.std(data[mask], axis=0)
    else:
        mean = np.mean(data[mask])
        std = np.std(data[mask])

    # If std ~ 0, replace by eps.
    std = np.maximum(std, eps)

    standardized_data = (data - mean) / std
    # standardized_data[~mask] = np.nan

    return standardized_data


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
    # var_b : 2-D array; real harmonics sampled at (\theta, \phi)
    #      var_b.T contains the new sampling scheme and var_b*data_sh projects
    #      to the sphere.
    # degree_m : array; degree of the sampled harmonics.
    # order_l : array; order of the sampled harmonics.
    var_b, degree_m, order_l = sh_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, var_b.T)

    return data_resampled
