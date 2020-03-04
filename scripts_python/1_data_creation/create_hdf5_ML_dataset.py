#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import shutil
import os
from pathlib import Path
from typing import Dict

import h5py
import nibabel as nib
import numpy as np
from dipy.io.streamline import save_tractogram

from dwi_ml.data.creation.dataset_creator import DatasetCreator
from dwi_ml.data.processing.dwi.dwi import standardize_data
from dwi_ml.data.creation.subjects_validation import validate_subject_list


DESCRIPTION = """Script to combine multiple diffusion MRI volumes and their 
streamlines into a single .hdf5 file.

Your folders should follow organization described in 
dwi_ml.data.creation.description_data_structure.FOLDER_DESCRIPTION.

A hdf5 folder will be created alongside dwi_ml_ready. It will contain the .hdf5
file and possibly intermediate files created during bundle processing.
"""

def _parse_args():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('path', type=str,
                   help="Path to database folder, which should contain a "
                        "dwi_ml_ready folder.")
    p.add_argument('config_file', type=str,
                   help="Path to the json config file defining the groups "
                        "wanted in your hdf5. Should follow description in "
                        "dwi_ml.data.creation.description_data_structure's "
                        "CONFIG_DESCRIPTION")
    p.add_argument('dwi', type=str,
                   help="Name of the DWI data that should be found inside "
                        "each subject's dwi folder in dwi_ml_ready.")
    p.add_argument('training_subj_ids', type=str,
                   help="txt file containing the list of subjects ids to use "
                        "for training.")
    p.add_argument('validation_subj_ids', type=str,
                   help="txt file containing the list of subjects ids to use "
                        "for validation.")
    p.add_argument('--name', type=str,
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--mask', type=str,
                   help="Mask defining the voxels used for data "
                        "standardization. If none is given, all non-zero "
                        "voxels will be used. mask should be the name of "
                        "the mask file inside dwi_ml_ready/{subj_id}/mask.")
    p.add_argument('--logging', type=str,
                   choices=['error', 'warning', 'info', 'debug'],
                   default='warning',
                   help="Choose logging level. [warning]")
    p.add_argument('--save_intermediate', action="store_true",
                   help="If set, save intermediate processing files for "
                        "each subject inside")
    p.add_argument('-f', '--force', action='store_true',
                   help="If set, overwrite existing hdf5 folder.")

    arguments = p.parse_args()

    return arguments


def verify_subject_lists(dwi_ml_folder, chosen_subjs):
    """
    Raises error if some subjects in training set or validation set don't
    exist.

    Prints logging info if some subjects in dwi_ml folder were not chosen
    in training set nor validation set.
    """
    # Find list of existing subjects from folders
    all_subjs = [str(f.name) for f in dwi_ml_folder.iterdir()]
    if len(all_subjs) == 0:
        raise ValueError('No subject found in dwi_ml folder!')

    # Checking chosen_subjs
    non_existing_subjs, good_chosen_subjs, ignored_subj = \
            validate_subject_list(all_subjs, chosen_subjs)
    if len(non_existing_subjs) > 0:
        raise ValueError('Following subjects were chosen either for '
                         'training set or validation set but their folders '
                         'were not found in dwi_ml: '
                         '{}'.format(non_existing_subjs))
    if len(ignored_subj) > 0:
        logging.info("Careful! NOT processing subjects {} "
                     "because they were not included in training set nor "
                     "validation set!".format(ignored_subj))


def main():
    """Parse arguments, generate hdf5 dataset and save it on disk."""
    args = _parse_args()

    # Initialize logger
    logging.basicConfig(level=str(args.logging).upper())
    logging.info(args)

    dwi_ml_folder = Path(args.path, "dwi_ml_ready")

    # Verify that subjects exist and that no subjects are forgotten
    chosen_subjs = args.training_subj_ids + args.validation_subj_ids
    verify_subject_lists(dwi_ml_folder, chosen_subjs)

    # Read group information from the json file
    json_file = open(args.config_file, 'r')
    groups_config = json.load(json_file)

    # Create dataset from config and save
    dataset_config = DatasetCreator(args.training_subj_ids, args.valid_subj_ids,
                                    dwi_ml_folder,
                                    EVERYTHING_CONCERNING_BUNDLES)
    _generate_dataset(args.path, args.name, dataset_config, groups_config,
                      args.mask, save_intermediate=args.save_intermediate,
                      force=args.force)


def load_and_check_data(data_file):
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
    img.uncache()

    if len(data.shape)<3:
        raise NotImplementedError('Why would a data be less than 3D? We did '
                                  'not plan this, you will have to change the'
                                  'code to use this data.')
    elif len(data.shape)==3:
        # Adding a fourth dimension
        data = data.reshape((*data.shape, 1))

    return data, affine


def process_group(group, file_list, save_intermediate, subj_input_path,
                  subj_output_path, subj_mask_data):
    """Process each group from the json config file:
     - Load data from each file of the group and combine them.
     - Standardize data
    """
    # First file will define data dimension and affine
    first_file = subj_input_path.joinpath(file_list[0])
    group_data, group_affine = load_and_check_data(first_file)

    # Other files must fit
    for data_file in file_list[1:]:
        data, _ = load_and_check_data(data_file)
        try:
            group_data = np.append(group_data, data, axis=-1)
        except:
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

    return standardized_group_data


def process_streamlines(subject_id: str, subject_data_path: Path,
                     output_path: Path,
                     dataset_creator: DatasetCreator,
                     save_intermediate: bool = False):

    tractogram, lengths = dataset_creator.load_process_and_merge_bundles(
        subject_data_path.joinpath("bundles"), dwi_image)
    if save_intermediate:
        save_tractogram(tractogram, str(output_path.joinpath(
            "{}_all_streamlines.tck".format(subject_id))))

    return tractogram.streamlines, lengths


def _generate_dataset(path: str, name: str,
                      dataset_creator: DatasetCreator, groups_config: Dict,
                      mask: str, save_intermediate: bool = False,
                      force: bool = False):
    """
    Generate a dataset from a group of dMRI subjects with multiple bundles.
    All bundles are merged as a single whole-brain dataset in voxel space.
    If wished, all intermediate steps are saved on disk in the hdf5 folder.

    Parameters
    ----------
    path : str
        Path to dataset folder, which contains dwi_ml_ready.
    name : str
        Dataset name used to save the .hdf5 file. If not given, guessed from
        path name.
    dataset_creator : DatasetCreator
        Configuration for the dataset to generate.
    groups_config: Dict
        List of groups and associated inputs from the json file.
    mask: str or None
        Name of the mask for each subject (in their dwi_ml mask folder).
    save_intermediate : bool
        Save intermediate processing files for each subject.
    force : bool
        Overwrite an existing dataset if it exists.
    """
    # Create hdf5 folder or clean existing one
    hdf5_path = Path(path, "hdf5")
    if hdf5_path.is_dir():
        if force:
            print("Deleting existing hdf5 data folder: {}"
                  .format(hdf5_path))
            shutil.rmtree(str(hdf5_path))
        else:
            raise FileExistsError("hdf5 data folder already exists: {}. "
                                  "Use force to allow overwrite"
                                  .format(hdf5_path))
    hdf5_path.mkdir()

    # Initialize database
    if name:
        dataset_name = name
    else:
        dataset_name = "{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    dataset_file = hdf5_path.joinpath("{}.hdf5".format(dataset_name))

    # Combine data into a hdf file.
    dwi_ml_folder = Path(path, "dwi_ml_ready")
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 1
        hdf_file.attrs.update(dataset_creator.get_state_dict())
        hdf_file.attrs.update(groups_config)

        # Starting the subjects processing
        logging.info("Processing {} subjects : {}"
                     .format(len(dataset_creator.chosen_subjs),
                             dataset_creator.chosen_subjs))
        for subj_id in dataset_creator.train_subjs:
            logging.info("Processing subject: {}".format(subj_id))
            subj_input_path = dwi_ml_folder.joinpath(subj_id)

            # Add subject to hdf database
            subj_hdf = hdf_file.create_group(subj_id)

            # Create subject's hdf5 data folder
            subj_output_path = hdf5_path.joinpath(subj_id)
            subj_output_path.mkdir()

            # Find subject's mask
            if mask is not None:
                subj_mask_file = subj_input_path.joinpath('masks/' + mask)
                subj_mask_data = nib.load(subj_mask_file).get_fdata(type=bool)

            # Add the subj data based on groups in the json config file
            # (inputs and others. All nifti)
            for group in groups_config:
                file_list = groups_config[group]
                group_data = process_group(group, file_list, save_intermediate,
                                           subj_input_path, subj_output_path,
                                           subj_mask_data)
                subj_hdf.create_dataset(group, data=group_data)

            # Add the streamlines data
            streamlines, lengths = process_streamlines(subj_id, subj_input_path,
                                 subj_output_path, dataset_creator,
                                 save_intermediate=save_intermediate)
            if streamlines is not None:
                streamlines_group = AJOUTER LES SPACE ATTRIBUTES
                streamlines_group = subj_hdf.create_group('streamlines')
                # Accessing private Dipy values, but necessary
                streamlines_group.create_dataset('data',
                                                 data=streamlines._data)
                streamlines_group.create_dataset('offsets',
                                                 data=streamlines._offsets)
                streamlines_group.create_dataset('lengths',
                                                 data=streamlines._lengths)
                streamlines_group.create_dataset('euclidean_lengths',
                                                 data=lengths)

    print("Saved dataset : {}".format(dataset_file))


if __name__ == '__main__':
    main()
