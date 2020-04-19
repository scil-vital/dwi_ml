# -*- coding: utf-8 -*-
import glob
import logging
import os
from pathlib import Path
from typing import List

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import length
import nibabel as nib
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft

from dwi_ml.data.processing.dwi.dwi import standardize_data


def verify_subject_lists(dwi_ml_folder: Path, chosen_subjs: List[str]):
    """
    Raises error if some subjects in training set or validation set don't
    exist.

    Prints logging info if some subjects in dwi_ml folder were not chosen
    in training set nor validation set.
    """

    # Find list of existing subjects from folders
    all_subjs = [str(f.name) for f in dwi_ml_folder.iterdir() if f.is_dir()]
    if len(all_subjs) == 0:
        raise ValueError('No subject found in dwi_ml folder: '
                         '{}'.format(dwi_ml_folder))

    # Checking chosen_subjs
    non_existing_subjs = [s for s in chosen_subjs if s not in all_subjs]
    if len(non_existing_subjs) > 0:
        raise ValueError('Following subjects were chosen either for '
                         'training set or validation set but their folders '
                         'were not found in dwi_ml: '
                         '{}'.format(non_existing_subjs))

    ignored_subj = [s for s in all_subjs if s not in chosen_subjs]
    if len(ignored_subj) > 0:
        logging.info("Careful! NOT processing subjects {} "
                     "because they were not included in training set nor "
                     "validation set!".format(ignored_subj))

    # Note. good_chosen_subjs = [s for s in all_subjs if s in chosen_subjs]


def _load_and_check_4d_nii_data(data_file):
    """Load nibabel data, and perform some checks:
    - Data must be Nifti
    - Data must be at least 3D
    Final data will most probably be 4D (3D + features). Sending loaded data to
    4D if it is 3D.
    """
    _, ext = os.path.splitext(data_file)
    if ext != '.gz' and ext != '.nii':
        raise ValueError('All data files should be nifti (.nii or .nii.gz) but '
                         'you provided {}. Please check again your config '
                         'file.'.format(data_file))

    img = nib.load(data_file)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header  # Note. affine is header.get_sform().
    img.uncache()

    if len(data.shape) < 3:
        raise NotImplementedError('Why would a data be less than 3D? We did '
                                  'not plan this, you will have to change the'
                                  'code to use this data.')
    elif len(data.shape) == 3:
        # Adding a fourth dimension
        data = data.reshape((*data.shape, 1))

    return data, affine, header


def process_group(group: str, file_list: List[str], save_intermediate: bool,
                  subj_input_path: Path, subj_output_path: Path,
                  subj_mask_data: np.ndarray = None):
    """Process each group from the json config file:
    - Load data from each file of the group and combine them.
    - Standardize data

    Parameters
    ----------
    group: str
        Group name.
    file_list: List[str]
        List of the files names that must be merged into a group.
    save_intermediate: bool
        If true, intermediate files will be saved.
    subj_input_path: Path
        Path where the files from file_list should be found.
    subj_output_path: Path
        Path where to save the intermediate files.
    subj_mask_data: np.ndarray of bools, optional
        Binary mask that will be used for data standardization.

    Returns
    -------
    standardized_group_data: np.ndarray
        Group data created by concatenating all files, standardized.
    group_affine: np.ndarray
        Affine for the group.
    """
    # First file will define data dimension and affine
    first_file = subj_input_path.joinpath(file_list[0])
    group_data, group_affine, group_header = \
        _load_and_check_4d_nii_data(first_file)

    # Other files must fit
    for data_file in file_list[1:]:
        data, _, _ = _load_and_check_4d_nii_data(data_file)
        try:
            group_data = np.append(group_data, data, axis=-1)
        except ImportError:
            raise ImportError('Data file {} could not be added to'
                              'data group {}. Wrong dimensions?'
                              ''.format(data_file, group))

    # Save unnormalized data
    if save_intermediate:
        group_image = nib.Nifti1Image(group_data, group_affine)
        output_fname = "{}_unnormalized.nii.gz".format(group)
        nib.save(group_image,
                 str(subj_output_path.joinpath(output_fname)))

    # Standardize data
    standardized_group_data = standardize_data(group_data, subj_mask_data)

    # Save standardized data
    if save_intermediate:
        standardized_img = nib.Nifti1Image(standardized_group_data,
                                           group_affine)
        output_fname = "{}_normalized.nii.gz".format(group)
        nib.save(standardized_img,
                 str(subj_output_path.joinpath(output_fname)))

    return standardized_group_data, group_affine, group_header


def process_streamlines(bundles_dir: Path, bundles,
                        header: nib.Nifti1Header, step_size: float,
                        space: Space):
    """Load and process a group of bundles and merge all streamlines
    together.

    Parameters
    ----------
    bundles_dir : Path
        Path to bundles folder.
    bundles: List[str]
        List of the bundles filenames to load.
    header : nib.Nifti1Header
        Reference used to load and send the streamlines in voxel space and to
        create final merged SFT.
    step_size: float
        Step size to resample streamlines. If none, compress streamlines.
    space: Space
        Space to place the tractograms.

    Returns
    -------
    output_tractogram : StatefulTractogram
        All streamlines in voxel space.
    output_lengths : List[float]
        The euclidean length of each streamline
    """

    # Initialize
    output_tractogram = None
    output_lengths = []

    # Find official bundles list
    if bundles is None:
        # Take everything found in subject bundle folder
        bundles = [str(p) for p in bundles_dir.glob('*')]
        if len(bundles) == 0:
            raise ValueError("No bundle found in the bundles folder!")

    for bundle_name in bundles:
        # Load bundle
        bundle_path = glob.glob(bundles_dir.joinpath(bundle_name + '*'))
        if len(bundle_path) == 0:
            raise ValueError('Bundle {} not found!'.format(bundle_name))
        elif len(bundle_path) > 1:
            raise ValueError(
                'More than one file with name {}. Clean your bundles '
                'folder.'.format(bundle_name + '*'))
        bundle = load_tractogram(bundle_path, header)

        # Send to wanted space
        bundle.to_space(space)

        # Resample or compress streamlines
        if step_size:
            bundle = resample_streamlines_step_size(bundle, step_size)
            logging.debug("Resampled streamlines' step size to {}mm"
                          .format(step_size))
        else:
            bundle = compress_sft(bundle)

        # Compute euclidean lengths
        output_lengths.extend(length(bundle.streamlines))

        # Add processed bundle to output tractogram
        if output_tractogram is None:
            output_tractogram = bundle
        else:
            output_tractogram.streamlines.extend(bundle.streamlines)

    # Transfer the streamlines to the reference space before bringing them
    # to VOX space. NOTE: This is done in case the streamlines were tracked
    # in a different space than the provided dataset reference
    if output_tractogram is None:
        output_streamlines_rasmm = []
    else:
        output_streamlines_rasmm = output_tractogram.streamlines
    output_tractogram = StatefulTractogram(output_streamlines_rasmm,
                                           header, space=Space.RASMM)

    # Internal validation check
    output_tractogram.remove_invalid_streamlines()
    logging.debug("Ran internal tractogram validation; "
                  "Remaining: {}".format(len(output_tractogram)))

    # Send to VOX space and make sure the origin is at the CENTER of the
    # voxel. NOTE: This is really important, otherwise interpolation will
    # be off by half a voxel.
    output_tractogram.to_vox()
    output_tractogram.to_center()

    return output_tractogram, output_lengths
