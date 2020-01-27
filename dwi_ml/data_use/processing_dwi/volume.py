"""Volume transformation utilities.

public functions:
    resample_volume
    get_neighborhood_directions
"""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
from dipy.align.reslice import reslice


def resample_volume(image: nib.Nifti1Image, ref_image: nib.Nifti1Image,
                    interp: str = 'lin',
                    enforce_dimensions: bool = False) -> nib.Nifti1Image:
    """Resample a Nifti image to match the resolution of another reference
    image.
    
    Taken from scil_resample_volume.py
    
    Parameters
    ----------
    image : nib.Nifti1Image
        Image to resample.
    ref_image : nib.Nifti1Image
        Reference image to resample to.
    interp : str
        Interpolation mode.
        choices=['nn','lin','quad','cubic']
            nn: nearest neighbor
            lin: linear
            quad: quadratic
            cubic: cubic
        Defaults to 'linear'.
    enforce_dimensions : bool
        Enforce the reference volume dimension.

    Returns
    -------
    output_image : nib.Nifti1Image
        Resampled volume.
    """

    interpolation_code_to_order = {'nn': 0, 'lin': 1, 'quad': 2, 'cubic': 3}

    data = image.get_fdata()
    affine = image.affine
    original_zooms = image.header.get_zooms()[:3]

    new_zooms = ref_image.header.get_zooms()[:3]

    logging.debug('Data shape: {0}'.format(data.shape))
    logging.debug('Data affine: {0}'.format(affine))
    logging.debug('Data affine setup: {0}'.format(nib.aff2axcodes(affine)))

    logging.debug('Resampling data to {0} '.format(new_zooms) +
                  'with mode {0}'.format(interp))

    interp = interpolation_code_to_order[interp]
    data2, affine2 = reslice(data, affine, zooms=original_zooms, order=interp,
                             new_zooms=new_zooms)

    logging.debug('Resampled data shape: {0}'.format(data2.shape))
    logging.debug('Resampled data affine: {0}'.format(affine2))
    logging.debug('Resampled data affine setup: {0}'
                  .format(nib.aff2axcodes(affine2)))

    computed_dims = data2.shape
    ref_dims = ref_image.shape[:3]

    if enforce_dimensions and computed_dims != ref_dims:
        fix_dim_volume = np.zeros(ref_dims)
        x_dim = min(computed_dims[0], ref_dims[0])
        y_dim = min(computed_dims[1], ref_dims[1])
        z_dim = min(computed_dims[2], ref_dims[2])

        fix_dim_volume[0:x_dim, 0:y_dim, 0:z_dim] = \
            data2[0:x_dim, 0:y_dim, 0:z_dim]
        output_image = nib.Nifti1Image(fix_dim_volume, affine2)
    else:
        output_image = nib.Nifti1Image(data2, affine2)

    return output_image