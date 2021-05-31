# -*- coding: utf-8 -*-
import glob
import logging
import os
from pathlib import Path
from typing import List

from dipy.io.stateful_tractogram import (set_sft_logger_level, Space)
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
        logging.info("    Careful! NOT processing subjects {} "
                     "because they were not included in training set nor "
                     "validation set!".format(ignored_subj))

    # Note. good_chosen_subjs = [s for s in all_subjs if s in chosen_subjs]


def _load_and_check_volume_to4d(data_file, group_affine=None,
                                group_header=None):
    """Load nibabel data, and perform some checks:
    - Data must be Nifti
    - Data must be at least 3D
    - If group_affine and group_header are given, loaded data must have the
    same affine and header.
    Final data will most probably be 4D (3D + features). Sending loaded data to
    4D if it is 3D.
    """
    ext = data_file.suffix

    if ext != '.gz' and ext != '.nii':
        raise ValueError('All data files should be nifti (.nii or .nii.gz) '
                         'but you provided {}. Please check again your config '
                         'file.'.format(data_file))

    img = nib.load(data_file)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header  # Note. affine is header.get_sform().
    img.uncache()

    if len(data.shape) < 3:
        raise NotImplementedError('Data less than 3D is not handled.')
    elif len(data.shape) == 3:
        # Adding a fourth dimension
        data = data.reshape((*data.shape, 1))

    return data, affine, header


def process_group(group: str, file_list: List[str], save_intermediate: bool,
                  subj_input_path: Path, subj_output_path: Path,
                  subj_std_mask_data: np.ndarray = None):
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
    subj_std_mask_data: np.ndarray of bools, optional
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
    logging.debug('    Loading {}'.format(first_file))
    group_data, group_affine, group_header = \
        _load_and_check_volume_to4d(first_file)

    # Other files must fit (data shape, header, affine)
    for data_name in file_list[1:]:
        data_file = subj_input_path.joinpath(data_name)
        logging.debug('    Loading {}'.format(data_file))
        data, _, _ = _load_and_check_volume_to4d(data_file, group_affine,
                                                 group_header)
        try:
            group_data = np.append(group_data, data, axis=-1)
        except ImportError:
            raise ImportError('Data file {} could not be added to data group '
                              '{}. Wrong dimensions?'.format(data_file, group))

    # Save unnormalized data
    if save_intermediate:
        logging.debug('  Saving intermediate non standardized files.')
        group_image = nib.Nifti1Image(group_data, group_affine)
        output_fname = "{}_nonstandardized.nii.gz".format(group)
        nib.save(group_image,
                 str(subj_output_path.joinpath(output_fname)))

    # Standardize data
    logging.debug('  Standardizing data')
    standardized_group_data = standardize_data(group_data, subj_std_mask_data)

    # Save standardized data
    if save_intermediate:
        logging.debug('  Saving intermediate standardized files.')
        standardized_img = nib.Nifti1Image(standardized_group_data,
                                           group_affine)
        output_fname = "{}_standardized.nii.gz".format(group)
        nib.save(standardized_img,
                 str(subj_output_path.joinpath(output_fname)))

    return standardized_group_data, group_affine, group_header


def process_streamlines(bundles_dir: Path, bundles,
                        enforce_bundles_presence: bool,
                        header: nib.Nifti1Header, step_size: float,
                        space: Space):
    """Load and process a group of bundles and merge all streamlines
    together.

    Parameters
    ----------
    bundles_dir : Path
        Path to bundles folder.
    bundles: List[str]
        List of the bundles filenames to load. If none, all bundles will be
        used.
    enforce_bundles_presence: bool
        If true, the process will stop if one bundle is missing for one
        subject.
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
    # Silencing SFT's logger if our logging is in DEBUG mode
    set_sft_logger_level('WARNING')

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
        # Find bundle name
        # Note. glob uses str. Other possibility: Path.cwd().glob but creates
        # a generator object.
        logging.debug('    Loading bundle {}'.format(bundle_name))
        bundle_name = str(bundles_dir.joinpath(bundle_name + '*'))
        bundle_real_name = glob.glob(bundle_name)
        if len(bundle_real_name) == 0 & enforce_bundles_presence:
            raise FileNotFoundError('Bundle {} not found!'.format(bundle_name))
        elif len(bundle_real_name) > 1:
            raise ValueError(
                'More than one file with name {}. Clean your bundles '
                'folder.'.format(bundle_name))
        else:
            bundle_real_name = bundle_real_name[0]

        # Check bundle extension
        _, file_extension = os.path.splitext(bundle_real_name)
        if file_extension not in ['.trk', '.tck']:
            raise ValueError("We do not support bundle's type: {}. We "
                             "only support .trk and .tck files."
                             .format(bundle_real_name))

        # Loading bundle and sending to wanted space
        bundle = load_tractogram(bundle_real_name[0], header)
        bundle.to_center()

        # Resample or compress streamlines
        # Note. No matter the chosen space, resampling is done in mm.
        if step_size:
            logging.debug('  Resampling')
            bundle = resample_streamlines_step_size(bundle, step_size)
            logging.debug("Resampled streamlines' step size to {}mm"
                          .format(step_size))
        else:
            logging.debug('  Compressing')
            bundle = compress_sft(bundle)

        # Compute euclidean lengths (rasmm space)
        bundle.to_space(Space.RASMM)
        output_lengths.extend(length(bundle.streamlines))

        # Sending to wanted space
        bundle.to_space(space)

        # Add processed bundle to output tractogram
        if output_tractogram is None:
            output_tractogram = bundle
        else:
            output_tractogram.streamlines.extend(bundle.streamlines)

    # Removing invalid streamlines
    logging.debug('...Done. Total: {:,.0f} streamlines. Now removing invalid '
                  'streamlines.'.format(len(output_tractogram)))
    output_tractogram.remove_invalid_streamlines()
    logging.debug("...Done. Remaining: {:,.0f} streamlines."
                  "".format(len(output_tractogram)))

    return output_tractogram, output_lengths
