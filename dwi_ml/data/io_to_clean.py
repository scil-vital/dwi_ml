"""I/O related utilities."""
import pathlib

from dipy.core.gradients import GradientTable, \
    gradient_table as load_gradient_table
import nibabel as nib
import numpy as np
from scilpy.image.resample_volume import resample_volume

def load_dwi(dwi_file: str) -> nib.Nifti1Image:
    """Loads a dMRI Nifti file along with the bvals/bvecs.
    Parameters
    ----------
    dwi_file : str
        Path to the dwi file
    Returns
    -------
    dwi_image : nib.Nifti1Image object
        The loaded Nifti image
    """
    try:
        dwi_image = nib.load(dwi_file)
    except FileNotFoundError:
        # Try another suffix (change .nii to .nii.gz and vice-versa)
        orig_path = pathlib.Path(dwi_file)
        if len(orig_path.suffixes) > 1:
            # case .nii.gz : Remove .gz
            alt_path = orig_path.with_suffix('')
        else:
            # case .nii : Add .gz
            alt_path = orig_path.with_suffix('.gz')
        dwi_image = nib.load(str(alt_path))

    return dwi_image


def load_gradients_from_dwi_filename(dwi_fname: str) -> GradientTable:
    """Load bvals/bvecs from a given DWI filename by guessing the right
    extension.
    Parameters
    ----------
    dwi_fname : str
        DWI NIfTI filename whose basename is used to infer the name of the
        diffusion gradient data filenames.
    Returns
    -------
    gradient_table : GradientTable
        An object that contains the b-values and b-vectors of the diffusion
        signal, along with a b0 mask.
    """
    # Remove suffix twice to account for .nii.gz
    base_path = pathlib.Path(dwi_fname).with_suffix('').with_suffix('')
    try:
        bvals = str(base_path.with_suffix('.bvals'))
        bvecs = str(base_path.with_suffix('.bvecs'))
        gradient_table = load_gradient_table(bvals, bvecs)
    except FileNotFoundError:
        bvals = str(base_path.with_suffix('.bval'))
        bvecs = str(base_path.with_suffix('.bvec'))
        gradient_table = load_gradient_table(bvals, bvecs)
    return gradient_table


def load_volume_with_ref(fname: str, ref: nib.Nifti1Image) -> nib.Nifti1Image:
    """Load a Nifti volume that should fit the shape of a reference volume,
    resampling if necessary.
    Parameters
    ----------
    fname : str
        Path to the volume to load.
    ref : nib.Nifti1Image or np.ndarray
        Reference volume.
    Returns
    -------
    output_volume : nib.Nifti1Image
        Loaded volume, resampled to the shape of the reference volume if
        necessary.
    """
    volume_image = nib.load(fname)
    ref_shape = ref.shape[:len(
        volume_image.shape)]  # See dipy.io.utils_to_refactor.is_reference_info_valid et
    #  get_reference_info

    if volume_image.shape != ref_shape:
        output_volume = resample_volume(volume_image, ref,
                                        enforce_dimensions=True)
    else:
        output_volume = volume_image

    return output_volume
