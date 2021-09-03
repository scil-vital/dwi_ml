# -*- coding: utf-8 -*-
import glob
import logging
import os
from pathlib import Path
from typing import List

from dipy.io.stateful_tractogram import (set_sft_logger_level, Space)
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.utils import length
import nibabel as nib
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft, concatenate_sft

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


def _load_volume_to4d(data_file):
    """Load nibabel data, and perform some checks:
    - Data must be Nifti
    - Data must be at least 3D
    Final data will most probably be 4D (3D + features). Sending loaded data to
    4D if it is 3D, with last dimension 1.
    """
    ext = data_file.suffix

    if ext != '.gz' and ext != '.nii':
        raise ValueError('All data files should be nifti (.nii or .nii.gz) '
                         'but you provided {}. Please check again your config '
                         'file.'.format(data_file))

    logging.debug('    Loading {}'.format(data_file))
    img = nib.load(data_file)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header  # Note. affine is header.get_sform()
    img.uncache()
    voxel_size = header.get_zooms()[:3]

    if len(data.shape) < 3:
        raise NotImplementedError('Data less than 3D is not handled.')
    elif len(data.shape) == 3:
        # Adding a fourth dimension
        data = data.reshape((*data.shape, 1))

    return data, affine, voxel_size, header


def process_group(group: str, file_list: List[str], save_intermediate: bool,
                  subj_input_path: Path, subj_output_path: Path,
                  subj_std_mask_data: np.ndarray = None,
                  independent_modalities: bool = True):
    """Process each group from the json config file:
    - Load data from each file of the group and combine them. All datasets from
      a given group must have the same affine, voxel resolution and data shape.
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
    independent_modalities: bool
        If True, modalities will be standardized separately.

    Returns
    -------
    group_data: np.ndarray
        Group data created by concatenating all files, standardized.
    group_affine: np.ndarray
        Affine for the group.
    """
    # First file will define data dimension and affine
    first_file = subj_input_path.joinpath(file_list[0])
    group_data, group_affine, group_res, group_header = \
        _load_volume_to4d(first_file)

    # Other files must fit (data shape, affine, voxel size)
    # It is not a promise that data has been correctly registered, but it
    # is a minimal check.
    for data_name in file_list[1:]:
        data_file = subj_input_path.joinpath(data_name)
        data, affine, res, _ = _load_volume_to4d(data_file)

        if not np.array_equal(affine, group_affine):
            raise ValueError('Data file {} does not have the same affine as '
                             'other files in group {}. Data from each group '
                             'will be concatenated, and should have the same '
                             'affine and voxel resolution.'
                             .format(data_name, group))

        if not np.array_equal(res, group_res):
            raise ValueError('Data file {} does not have the same resolution '
                             'as other files in group {}. Data from each '
                             'group will be concatenated, and should have the '
                             'same affine and voxel resolution.'
                             .format(data_name, group))

        try:
            group_data = np.append(group_data, data, axis=-1)
        except ImportError:
            raise ImportError('Data file {} could not be added to data group '
                              '{}. Wrong dimensions?'.format(data_name, group))

    # Save unnormalized data
    if save_intermediate:
        output_fname = subj_output_path.joinpath(group +
                                                 "_nonstandardized.nii.gz")
        logging.debug('      *Saving intermediate non standardized files into '
                      '{}.'.format(output_fname))
        group_image = nib.Nifti1Image(group_data, group_affine)
        nib.save(group_image, output_fname)

    # Standardize data (per channel)
    logging.debug('      *Standardizing data')
    group_data = standardize_data(group_data, subj_std_mask_data,
                                  independent=independent_modalities)

    # Save standardized data
    if save_intermediate:
        output_fname = subj_output_path.joinpath(group +
                                                 "_standardized.nii.gz")
        logging.debug('      *Saving intermediate standardized files into {}.'
                      .format(output_fname))
        standardized_img = nib.Nifti1Image(group_data, group_affine)
        nib.save(standardized_img, output_fname)

    return group_data, group_affine, group_header


def process_streamlines(bundles_dir: Path, bundles, header: nib.Nifti1Header,
                        step_size: float, space: Space,
                        save_intermediate: bool, subj_output_path: Path):
    """Load and process a group of bundles and merge all streamlines
    together.

    Parameters
    ----------
    bundles_dir : Path
        Path to bundles folder.
    bundles: List[str]
        List of the bundles filenames to load. If none, all bundles will be
        used.
    header : nib.Nifti1Header
        Reference used to load and send the streamlines in voxel space and to
        create final merged SFT. If the file is a .trk, 'same' is used instead.
    step_size: float
        Step size to resample streamlines. If none, compress streamlines.
    space: Space
        Space to place the tractograms.
    save_intermediate: bool
        If true, intermediate files will be saved.
    subj_output_path: Path
        Path where to save the intermediate files.

    Returns
    -------
    final_tractogram : StatefulTractogram
        All streamlines in voxel space.
    output_lengths : List[float]
        The euclidean length of each streamline
    """
    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    # Initialize
    final_sft = None
    output_lengths = []

    # If not bundles in the config, taking everything in the subject's folder
    if bundles is None:
        # Take everything found in subject bundle folder
        bundles = [str(p) for p in bundles_dir.glob('*')]

    for bundle_name in bundles:
        # Find bundle name
        logging.debug('      *Loading bundle {}'.format(bundle_name))

        # Completing name, ex if no extension was given or to allow suffixes
        bundle_path = str(bundles_dir.joinpath(bundle_name + '*'))
        bundle_complete_name = glob.glob(bundle_path)
        if len(bundle_complete_name) == 0:
            logging.debug("      Skipping bundle {} because it was not found "
                          "in this subject's folder".format(bundle_name))
            # Note: if args.enforce_bundles_presence was set to true, this case
            # is not possible, already checked in create_hdf5_dataset.py.
        else:
            bundle_complete_name = bundle_complete_name[0]

            # Check bundle extension
            _, file_extension = os.path.splitext(bundle_complete_name)
            if file_extension not in ['.trk', '.tck']:
                raise ValueError("We do not support bundle's type: {}. We "
                                 "only support .trk and .tck files."
                                 .format(bundle_complete_name))
            if file_extension == '.trk':
                header = 'same'

            # Loading bundle and sending to wanted space
            logging.info("          - Loading")
            sft = load_tractogram(bundle_complete_name, header)
            sft.to_center()

            # Resample or compress streamlines
            # Note. No matter the chosen space, resampling is done in mm.
            if step_size:
                logging.info("          - Resampling")
                sft = resample_streamlines_step_size(sft, step_size)
                logging.debug("      *Resampled streamlines' step size to {}mm"
                              .format(step_size))
            else:
                logging.info("          - Compressing")
                sft = compress_sft(sft)

            # Compute euclidean lengths (rasmm space)
            sft.to_space(Space.RASMM)
            output_lengths.extend(length(sft.streamlines))

            # Sending to wanted space
            sft.to_space(space)

            # Add processed bundle to output tractogram
            if final_sft is None:
                final_sft = sft
            else:
                final_sft = concatenate_sft([final_sft, sft],
                                            erase_metadata=False)

            if save_intermediate:
                bundle_name = os.path.splitext(bundle_name)[0]
                output_fname = subj_output_path.joinpath(bundle_name + '.trk')
                logging.debug(
                    '      *Saving intermediate bundle {} into {}.'
                    .format(bundle_name, output_fname))
                # Note. Do not remove the str below. Does not work well with
                # Path.
                save_tractogram(sft, str(output_fname))

    # Removing invalid streamlines
    logging.debug('      *Total: {:,.0f} streamlines. Now removing invalid '
                  'streamlines.'.format(len(final_sft)))
    final_sft.remove_invalid_streamlines()
    logging.debug("      *Remaining: {:,.0f} streamlines."
                  "".format(len(final_sft)))

    return final_sft, output_lengths
