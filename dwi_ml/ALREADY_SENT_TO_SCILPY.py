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


def resample_streamlines(streamlines: Iterable, step_size: float,
                         convert_mm_to_vox: bool = False,
                         affine: np.ndarray = None) -> List[np.ndarray]:
    """Resample streamlines to fit a constant step size from a list of
    streamlines. See also the equivalent for tractograms in sft.

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence or list of np.ndarray
        Streamlines to resample.
    step_size : float
        Step size that all streamlines should have.
    convert_mm_to_vox: bool
        If set, we consider that the streamlines are in voxel space and
        convert the step_size first. [False]
    affine: np.ndarray
        Needed if convert_noise_space is True. Ex : affine_vox2rasmm

    Returns
    -------
    streamlines_resampled : list of np.ndarray
        Resampled streamlines.
    """

    # Convert RASmm step size to iso VOX space
    if convert_mm_to_vox:
        step_size = convert_mm2vox(step_size, affine)

    # Resample
    lengths = length(streamlines)
    nb_points = np.ceil(lengths / step_size).astype(int)
    streamlines_resampled = [set_number_of_points(s, n) for s, n in
                             zip(streamlines, nb_points)]
    return streamlines_resampled


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


def remove_short_streamlines_from_sft(tractogram: StatefulTractogram,
                                      min_length_mm: float):
    # When ready for python 3.7: -> StatefulTractogram:
    """Remove all streamlines shorter than the minimum length in mm.

    Parameters
    ----------
    tractogram : StatefulTractogram
        Tractogram to filter.
    min_length_mm : float
        Streamlines shorter than this length will be removed.

    Returns
    -------
    tractogram : StatefulTractogram
        A tractogram without short streamlines.
    """
    # Make sure we are in world space
    orig_space = tractogram.space
    tractogram.to_rasmm()

    lengths = length(tractogram.streamlines)
    filtered_streamlines = [s for (s, l) in zip(tractogram.streamlines, lengths)
                            if l > min_length_mm]
    output_tractogram = StatefulTractogram(
        filtered_streamlines, tractogram, Space.RASMM,
        shifted_origin=tractogram.shifted_origin)

    # Return to original space
    if orig_space == Space.VOX:
        output_tractogram.to_vox()
    elif orig_space == Space.VOXMM:
        output_tractogram.to_voxmm()

    return output_tractogram


def resample_sft(tractogram: StatefulTractogram,
                 step_size: float) -> StatefulTractogram:
    """Resample streamlines to have a constant step size, from a stateful
    tractogram. See also: resample_streamlines_step_size.

    Parameters
    ----------
    tractogram : dipy.io.stateful_tractogram.StatefulTractogram
        Tractogram to resample.
    step_size : float
        Step size that all streamlines should have.
    Returns
    -------
    output_tractogram : dipy.io.stateful_tractogram.StatefulTractogram
        Tractogram with all streamlines resampled with a constant step size.
    """
    # Make sure we are in world space
    orig_space = tractogram.space
    tractogram.to_rasmm()

    # Resample streamlines
    lengths = length(tractogram.streamlines)
    nb_points = np.ceil(lengths / step_size).astype(int)
    if np.any(nb_points == 1):
        # Some streamlines are too short;
        # make sure there is always at least 2 points
        logging.warning("Some streamlines are shorter than the provided "
                        "step size...")
        nb_points[nb_points == 1] = 2
    streamlines_resampled = [set_number_of_points(s, n) for s, n in
                             zip(tractogram.streamlines, nb_points)]
    output_tractogram = StatefulTractogram(streamlines_resampled, tractogram,
                                           Space.RASMM,
                                           tractogram.shifted_origin)
    # Return to original space
    if orig_space == Space.VOX:
        tractogram.to_vox()
    elif orig_space == Space.VOXMM:
        tractogram.to_voxmm()

    return output_tractogram


def compress_sft(tractogram: StatefulTractogram):
    # When ready for python 3.7: -> StatefulTractogram:
    """Compress the streamlines of a stateful tractogram

    Parameters
    ----------
    tractogram : StatefulTractogram

    Returns
    -------
    output_tractogram : StatefulTractogram
    """
    orig_space = tractogram.space
    tractogram.to_rasmm()
    compressed_streamlines = compress_streamlines(tractogram.streamlines)

    output_tractogram = StatefulTractogram(compressed_streamlines, tractogram,
                                           tractogram.space,
                                           tractogram.shifted_origin)

    if orig_space == Space.VOXMM:
        output_tractogram.to_voxmm()
    elif orig_space == Space.VOX:
        output_tractogram.to_vox()
    elif orig_space == Space.RASMM:
        pass
    else:
        raise ValueError

    return output_tractogram


def apply_transform_to_streamlines(streamlines: Iterable, affine: np.ndarray):
    """Apply an affine transformation on a set of streamlines

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence or List of np.ndarray
        Streamlines to transform.
    affine : np.ndarray with shape (4,4)
        Affine tranformation to apply on the streamlines.

    Returns
    -------
    nib.streamlines.ArraySequence
        Transformed streamlines.
    """
    tractogram = nib.streamlines.Tractogram(streamlines)
    tractogram.apply_affine(affine)
    return tractogram.streamlines
