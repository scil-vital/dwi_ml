# -*- coding: utf-8 -*-
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
    all_subjs = [str(s.name) for s in Path(dwi_ml_folder).iterdir()
                 if s.is_dir()]
    if len(all_subjs) == 0:
        raise ValueError('No subject found in dwi_ml folder: '
                         '{}'.format(dwi_ml_folder))

    # Checking chosen_subjs
    non_existing_subjs = [s for s in chosen_subjs if s not in all_subjs]
    if len(non_existing_subjs) > 0:
        raise ValueError('Following subjects were chosen either for '
                         'training set or validation set but their folders '
                         'were not found in dwi_ml_ready: '
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


def process_volumes(group: str, file_list: List[str], subj_id,
                    save_intermediate: bool, subj_input_path: Path,
                    subj_output_path: Path, standardization: str,
                    subj_std_mask_data: np.ndarray = None):
    """Process each group from the json config file:
    - Load data from each file of the group and combine them. All datasets from
      a given group must have the same affine, voxel resolution and data shape.
    - Standardize data

    Parameters
    ----------
    group: str
        Group name.
    file_list: List[str]
        List of the files names that must be merged into a group. Wildcards
        will be replaced by the subject id.
    subj_id: str
        The subject's id.
    save_intermediate: bool
        If true, intermediate files will be saved.
    subj_input_path: Path
        Path where the files from file_list should be found.
    subj_output_path: Path
        Path where to save the intermediate files.
    standardization: str
        One of ['all', 'independent', 'per_file', 'none'].
    subj_std_mask_data: np.ndarray of bools, optional
        Binary mask that will be used for data standardization.

    Returns
    -------
    group_data: np.ndarray
        Group data created by concatenating all files, standardized.
    group_affine: np.ndarray
        Affine for the group.
    """
    # First file will define data dimension and affine
    file_name = file_list[0].replace('*', subj_id)
    first_file = subj_input_path.joinpath(file_name)
    logging.info("       - Processing file {}".format(file_name))
    group_data, group_affine, group_res, group_header = \
        _load_volume_to4d(first_file)

    if standardization == 'per_file':
        logging.debug('      *Standardizing sub-data')
        group_data = standardize_data(group_data, subj_std_mask_data,
                                      independent=False)

    # Other files must fit (data shape, affine, voxel size)
    # It is not a promise that data has been correctly registered, but it
    # is a minimal check.
    for file_name in file_list[1:]:
        file_name = file_name.replace('*', subj_id)
        data_file = subj_input_path.joinpath(file_name)

        logging.info("       - Processing file {}".format(file_name))

        if not data_file.is_file():
            logging.debug("      Skipping volume {} because it was not found "
                          "in this subject's folder".format(file_name))
            # Note: if args.enforce_files_presence was set to true, this case
            # is not possible, already checked in create_hdf5_dataset.py.

        data, affine, res, _ = _load_volume_to4d(data_file)

        if not np.allclose(affine, group_affine, atol=1e-5):
            # Note. When keeping default options on tolerance, we have ran into
            # some errors in some cases, depending on how data has been
            # processed. Now accepting bigger error.
            raise ValueError('Data file {} does not have the same affine as '
                             'other files in group {}. Data from each group '
                             'will be concatenated, and should have the same '
                             'affine and voxel resolution.\n'
                             'Affine: {}\n'
                             'Group affine: {}\n'
                             'Biggest difference: {}'
                             .format(file_name, group, affine, group_affine,
                                     np.max(affine - group_affine)))

        if not np.allclose(res, group_res):
            raise ValueError('Data file {} does not have the same resolution '
                             'as other files in group {}. Data from each '
                             'group will be concatenated, and should have the '
                             'same affine and voxel resolution.\n'
                             'Resolution: {}\n'
                             'Group resolution: {}'
                             .format(file_name, group, res, group_res))

        if standardization == 'per_file':
            logging.debug('      *Standardizing sub-data')
            data = standardize_data(data, subj_std_mask_data,
                                    independent=False)

        try:
            group_data = np.append(group_data, data, axis=-1)
        except ImportError:
            raise ImportError('Data file {} could not be added to data group '
                              '{}. Wrong dimensions?'.format(file_name, group))

    # Standardize data (per channel)
    if standardization == 'independent':
        logging.debug('      *Standardizing data on each feature.')
        group_data = standardize_data(group_data, subj_std_mask_data,
                                      independent=True)
    elif standardization == 'all':
        logging.debug('      *Standardizing data as a whole.')
        group_data = standardize_data(group_data, subj_std_mask_data,
                                      independent=False)
    elif standardization not in ['none', 'per_file']:
        raise ValueError("standardization must be one of "
                         "['all', 'independent', 'per_file', 'none']")

    # Save standardized data
    if save_intermediate:
        output_fname = subj_output_path.joinpath(group + ".nii.gz")
        logging.debug('      *Saving intermediate files into {}.'
                      .format(output_fname))
        standardized_img = nib.Nifti1Image(group_data, group_affine)
        nib.save(standardized_img, str(output_fname))

    return group_data, group_affine, group_header, group_res


def process_streamlines(
        subj_dir: Path, group: str, bundles: List[str], subj_id: str,
        header: nib.Nifti1Header, step_size: float, compress: bool,
        space: Space, save_intermediate: bool, subj_output_path: Path):
    """Load and process a group of bundles and merge all streamlines
    together.

    Parameters
    ----------
    subj_dir : Path
        Path to bundles folder.
    group: str
        group name
    bundles: List[str]
        List of the bundles filenames to load. Wildcards will be replaced by
        the subject id. If the list is folderx/ALL, all bundles in the folderx
        will be used.
    subj_id: str
        The subject's id.
    header : nib.Nifti1Header
        Reference used to load and send the streamlines in voxel space and to
        create final merged SFT. If the file is a .trk, 'same' is used instead.
    step_size: float
        Step size to resample streamlines.
    compress: bool
        Compress streamlines.
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
    if step_size and compress:
        raise ValueError("Only one option can be chosen: either resampling "
                         "to step_size or compressing, not both.")

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    # Initialize
    final_sft = None
    output_lengths = []

    for tmp_bundle_name in bundles:
        if tmp_bundle_name.endswith('/ALL'):
            bundles_dir = tmp_bundle_name.split('/ALL')
            bundles_dir = ''.join(bundles_dir[:-1])
            bundles2 = [str(p) for p in subj_dir.glob(bundles_dir + '/*')]
        else:
            bundles2 = [tmp_bundle_name]

        # Either a loop on "ALL" or a loop on only one file, tmp_bundle_name.
        for bundle_name in bundles2:
            bundle_name = bundle_name.replace('*', subj_id)

            bundle_file = subj_dir.joinpath(bundle_name)
            if not bundle_file.is_file():
                logging.debug("      Skipping bundle {} because it was not "
                              "found in this subject's folder"
                              .format(bundle_name))
                # Note: if args.enforce_files_presence was set to true, this
                # case is not possible, already checked in create_hdf5_dataset
            else:
                # Check bundle extension
                _, file_extension = os.path.splitext(str(bundle_file))
                if file_extension not in ['.trk', '.tck']:
                    raise ValueError("We do not support bundle's type: {}. We "
                                     "only support .trk and .tck files."
                                     .format(bundle_file))
                if file_extension == '.trk':
                    header = 'same'

                # Loading bundle and sending to wanted space
                logging.info("       - Processing bundle {}"
                             .format(os.path.basename(bundle_name)))
                sft = load_tractogram(str(bundle_file), header)
                sft.to_center()

                # Resample or compress streamlines
                # Note. No matter the chosen space, resampling is done in mm.
                if step_size:
                    logging.info("          - Resampling")
                    sft = resample_streamlines_step_size(sft, step_size)
                    logging.debug("      *Resampled streamlines' step size to "
                                  "{}mm".format(step_size))
                elif compress:
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
        output_fname = subj_output_path.joinpath(group + '.trk')
        logging.debug('      *Saving intermediate bundle {} into '
                      '{}.'.format(group, output_fname))
        # Note. Do not remove the str below. Does not work well
        # with Path.
        save_tractogram(final_sft, str(output_fname))

    # Removing invalid streamlines
    logging.debug('      *Total: {:,.0f} streamlines. Now removing invalid '
                  'streamlines.'.format(len(final_sft)))
    final_sft.remove_invalid_streamlines()
    logging.debug("      *Remaining: {:,.0f} streamlines."
                  "".format(len(final_sft)))

    return final_sft, output_lengths
