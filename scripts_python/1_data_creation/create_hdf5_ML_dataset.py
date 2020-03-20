#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Main script for users
# Creates a .hdf5 dataset.
# Public functions:
# main


#                                                                                                       ToDo Proposition. Ajouter la possibilité de loader des données
#                                                                                                        preprocessed ailleurs pour les volumes au lieu de processer les volumes ici?

#                                                                                                       ToDO À ajouter! Prise en charge des bids

import argparse
import datetime
import logging
import os
import shutil
from pathlib import Path
from typing import List

from dipy.io.streamline import save_tractogram
import h5py
import nibabel as nib
import numpy as np

from dwi_ml.data.creation.hdf5_creator import (
    HDF5CreatorAbstract,
    HDF5CreatorDWI,
    HDF5CreatorDwiSH,
    HDF5CreatorFODFPeaks,
    HDF5CreatorFodfSH)
                                                                                                            # toDO NOTE. Creators.load_and_process have changed. We need to use bval and bvecs now
                                                                                                            #  instead of gradient table. So we need to load things a bit differently from the beginnig.
# À ARRANGER
from scil_vital.shared.code.data.description_data_structure import (
    FOLDER_DESCRIPTION,
    CONFIG_DECRIPTION)
from scil_vital.shared.code.io.io_utils import (
    load_dwi_and_gradients,
    load_volume_with_ref)
from scil_vital.shared.code.signal.signal_utils import (
    filter_bvalue,
    normalize_data_volume)

DESCRIPTION = """Script to process multiple diffusion MRI volumes and their 
streamlines into a single .hdf5 file. Please follow the following 
structures: 

""" \
              + FOLDER_DESCRIPTION + CONFIG_DECRIPTION + """
              
==== In this script (create_hdf5_ML_dataset.py), a `processed` folder will be created 
alongside the `raw` folder in the dataset folder. It will contain the .hdf5 file 
and the intermediate files.
"""


def _parse_args():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('path', type=str, help="Path to dataset folder.")
    p.add_argument('config_file', type=str, help="Path to the json "
                                                 "config file.")

    # Raw, raw-SH or FODF-sh of FODF-peaks
    subp = p.add_subparsers(dest='model_input',
                            help="Choice of signal that will be used "
                                 "as input by the model.")

    # RAW arguments
    dwi_subp = subp.add_parser('dwi',
                               help="Use the '"'raw'"' diffusion signal. "
                                    "If chosen, following options are "
                                    "available:\n --resample")
    dwi_subp.add_argument('--resample', action="store_true",
                          help="If set, resample the diffusion signal to 100 "
                               "directions evenly sampled on the sphere.")

    # DWI-SH arguments
    dwi_sh_subp = subp.add_parser('dwi-sh',
                                  help="Fit spherical harmonics onto "
                                       "the diffusion signal, and use "
                                       "the coefficients as input."
                                       "If chosen, following options are "
                                       "available:\n --sh-order")
    dwi_sh_subp.add_argument('--sh-order', type=int, choices=[2, 4, 6, 8],
                             default=6, help="Order of the spherical "
                                             "harmonics to fit. [6]")

    # FODF-SH arguments
    fodf_sh_subp = subp.add_parser('fodf-sh',
                                   help="Fit fODFs, and use the SH "
                                        "coefficients as input. If chosen, "
                                        "following options are available:\n"
                                        "--sh-order.")
    fodf_sh_subp.add_argument('--sh-order', type=int, choices=[2, 4, 6, 8],
                              default=6, help="Order of the spherical "
                                              "harmonics to fit. [6]")

    # FODF peaks arguments
    fodf_peaks_subp = subp.add_parser("fodf-peaks",
                                      help="Fit fODFs, and use the "
                                           "main peaks as input. If chosen, "
                                           "following options are available:\n"
                                           "--sh-order\n -nn-peaks.")
    fodf_peaks_subp.add_argument('--sh-order', type=int, choices=[2, 4, 6, 8],
                                 default=6, help="Order of the spherical "
                                                 "harmonics to fit. [6]")
    fodf_peaks_subp.add_argument('--n-peaks', type=int, choices=[1, 2, 3],
                                 default=3, help="Number of peaks to use as "
                                                 "input to the model. [3]")

    # Other facultative arguments
    p.add_argument('--subject_ids', type=str, nargs='+',
                   help="List of subjects ids to use for training. Ex:"
                        "(EXAMPLE TO GIVE). If not given, we will process all "
                        "the subjects in the raw folder.")                                              # ToDo ANTOINE: was a positional argument
                                                                                                        #  PHILIPPE: did not exist.
    p.add_argument('--name', type=str, help="Dataset name [Default uses "
                                            "date and time of processing].")
    p.add_argument('--logging', type=str,
                   choices=['error', 'warning', 'info', 'debug'],
                   default='warning',
                   help="Choose logging level. [warning]")
    p.add_argument('--save_intermediate', action="store_true",
                   help="If set, save intermediate processing files for "
                        "each subject.")
    p.add_argument('-f', '--force', action='store_true',
                   help="If set, overwrite existing folder. CAREFUL!! It "
                        "will delete everything in that folder!")

    arguments = p.parse_args()

    return arguments


def main():
    """Parse arguments, generate hdf5 dataset and save it on disk."""
    args = _parse_args()
                                                                                                # ToDo NOTE: OTHER VERIFICATIONS ARE MADE DIRECTLY IN data_loader. SHOULD BE
                                                                                                #  DONE HERE INSTEAD??  (see all the validate_config methods)

    # Initialize logger
    logging.basicConfig(level=str(args.logging).upper())
    logging.info(args)

    # Choose processor class based on user's model choice
    cls_choice = {"dwi": HDF5CreatorDWI,
                  "dwi-sh": HDF5CreatorDwiSH,
                  "fodf-sh": HDF5CreatorFodfSH,
                  "fodf-peaks": HDF5CreatorFODFPeaks}
    dataset_cls = cls_choice[args.model_input]

    # Prepare other parameters
    cls_args = vars(args).copy()
    del cls_args["path"]
    del cls_args["config_file"]
    del cls_args["model_input"]
    del cls_args["name"]
    del cls_args["logging"]
    del cls_args["force"]
    del cls_args["save_intermediate"]

    # Create dataset from config and save
    raw_path = Path(args.path, "raw")
    dataset_config = dataset_cls.from_json(args.config_file, raw_path,
                                           **cls_args)
                                                                                                # ToDo ANTOINE uses with Timer("Generating dataset", newline=True):
    _generate_dataset(args.path, args.name, dataset_config,
                      save_intermediate=args.save_intermediate,
                      force=args.force)


def _generate_dataset(path: str, name: str,
                      dataset_creator: HDF5CreatorAbstract,
                      save_intermediate: bool = False,
                      force: bool = False):
    """Generate a dataset from a group of dMRI subjects with multiple bundles.
    All bundles are merged as a single whole-brain dataset in voxel space.
    All intermediate steps are saved on disk in a separate "processed" folder.

    Nomenclature used throughout the code:
    *_path : Path to folder
    *_file : Path to file
    *_image : `Nifti1Image` object
    *_volume : Numpy 3D/4D array

    Parameters
    ----------
    path : str
        Path to dataset folder.
    name : str
        Dataset name used to save the .hdf5 file. If not given, guessed from
        path name.
    dataset_creator : DatasetConfig
        Configuration for the dataset to generate.
    chosen_subject_ids: List[str]
        User-selected subjects to process. Will be compared to json list and to
        subjects list from directory to avoid errors.
    save_intermediate : bool
        Save intermediate processing files for each subject.
    force : bool
        Overwrite an existing dataset if it exists.
    """

    # Prepare folder/file name
    if name:
        dataset_name = name
        processed_path = Path(path, "processed_{}".format(dataset_name))
    else:
        dataset_name = os.path.basename(path)
        processed_path = Path(path, "processed_default_{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")))

    # Create processed folder or clean existing one
    if processed_path.is_dir():
        if force:
            print("Deleting existing processed data folder: {}"
                  .format(processed_path))
            shutil.rmtree(str(processed_path))                                                              # toDo We should probably avoid removing stuff that
                                                                                                            #  the person could have placed in the processed_... folder.
                                                                                                            #  ex: maybe the user created images or notes and want to keep them

                                                                                                            # NOTE. Antoine's version was:
                                                                                                            #  Remove processed subjects
                                                                                                            #  for subject_id in os.listdir(processed_path):
                                                                                                            #    if subject_id in subject_ids:
                                                                                                            #        subject_path = pjoin(processed_path, subject_id)
                                                                                                            #        shutil.rmtree(subject_path)
                                                                                                            #  but you also want to remove all other subjects event if they are not in
                                                                                                            #  subject_ids: you don't want to subjects processed with different parameters
                                                                                                            #  in processed_path.
        else:
            raise FileExistsError("Processed data folder already exists: {}. "
                                  "Use force to allow overwrite"
                                  .format(processed_path))
    processed_path.mkdir()

    # Initialize database
    dataset_file = processed_path.joinpath("{}.hdf5".format(dataset_name))
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 3                                                                   # ToDo ANTOINE's version was 2. Any difference?
        hdf_file.attrs.update(dataset_creator.get_state_dict())                                         #  ANTOINE a aussi ["bundles"].
                                                                                                        #  et hdf_file.attrs.update({k: v for k, v in config_dict.items()
                                                                                                        #     if v is not None})

        # Starting the subjects processing
        logging.info("Processing {} subjects : {}"
                     .format(len(dataset_creator.final_subjs),
                             dataset_creator.final_subjs))
        raw_path = Path(path, "raw")                                                            # toDo Antoine used pjoin. Any difference?
        for subject_id in dataset_creator.final_subjs:
                                                                                                # toDO ANTOINE A:
                                                                                                #  with Timer(
                                                                                                #        "Processing subject {}".format(
                                                                                                #            subject_id),
                                                                                                #        newline=True,
                                                                                                #        color='blue'
                                                                                                #  ):
            subject_data_path = raw_path.joinpath(subject_id)

            # Create subject's processed data folder
            subject_output_path = processed_path.joinpath(subject_id)
            subject_output_path.mkdir()
            logging.info("Processing subject: {}".format(subject_id))
            model_input, streamlines, lengths = \
                _process_subject(subject_id, subject_data_path,
                                 subject_output_path, dataset_creator,
                                 save_intermediate=save_intermediate)

            # Add subject to hdf database
            hdf_subject = hdf_file.create_group(subject_id)
            hdf_input_volume = hdf_subject.create_group('input_volume')                                             # ANTOINE a un group input_volume, peaks_volume, odfs_volume
            hdf_input_volume.attrs['vox2rasmm'] = model_input.affine
            hdf_input_volume.create_dataset(
                'data', data=model_input.get_fdata(dtype=np.float32))
            if streamlines is not None:
                streamlines_group = hdf_subject.create_group('streamlines')
                # Accessing private Dipy values, but necessary
                streamlines_group.create_dataset('data',
                                                 data=streamlines._data)
                streamlines_group.create_dataset('offsets',
                                                 data=streamlines._offsets)
                streamlines_group.create_dataset('lengths',
                                                 data=streamlines._lengths)
                streamlines_group.create_dataset('euclidean_lengths',
                                                 data=lengths)
                                                                                                                # À AJOUTER POUR ANTOINE:
                                                                                                 # streamlines_group.create_dataset('rewards', data=streamlines_data['rewards'])
    print("Saved dataset : {}".format(dataset_file))


def _process_subject(subject_id: str, subject_data_path: Path,
                     output_path: Path,
                     dataset_creator: HDF5CreatorAbstract,
                     save_intermediate: bool = False):
    """Process a subject to extract normalized input data and all streamlines.

    Parameters
    ----------
    subject_id : str
        Subject unique identifier.
    subject_data_path : pathlib.Path
        Path to the subject's data.
    output_path : pathlib.Path
        Path to the processed folder where intermediate data should be saved.
    dataset_creator : HDF5CreatorAbstract
        Processor for dataset volumes and streamlines.
    save_intermediate : bool
        Save intermediate processing files for each subject.


    Returns
    -------
    model_input : nib.Nifti1Image
        Processed model input data normalized along every modality (last axis).
    streamlines : nib.streamlines.ArraySequence
        All subject streamlines in voxel space.
    streamlines_lengths : List[float]
        Euclidean length of each streamline.
    """
                                                                                                    # ANTOINE: with Timer("Fitting SH to DWI", newline=True):
    # Get diffusion data
    dwi_file = subject_data_path.joinpath("dwi",
                                          "{}_dwi.nii.gz".format(subject_id))
    dwi_image, gradient_table = load_dwi_and_gradients(str(dwi_file))

    if dataset_creator.bval:
        dwi_image, gradient_table = filter_bvalue(dwi_image, gradient_table,
                                                  dataset_creator.bval)
    else:
        # Verify if multiple b-values are present
        bvals = gradient_table.bvals[np.logical_not(gradient_table.b0s_mask)]
        assert np.allclose(bvals, bvals[0]), \
            "Multiple b-values detected for subject {}, please use the " \
            "--bval-filter option".format(subject_id)

    # Get WM mask                                                                                   # Does it have to be WM? Could rename to... what? Tracking mask? normalization mask?
    wm_mask_file = subject_data_path.joinpath(
        "masks", "{}_wm.nii.gz".format(subject_id))
    if not wm_mask_file.exists():
        raise ValueError("WM mask is required! : {}".format(wm_mask_file))
    wm_mask_image = load_volume_with_ref(str(wm_mask_file), dwi_image)

    # Load and process data based on model choice
    model_input_volume = \
        dataset_creator.load_and_process_volume(dwi_image, bvals, bvecs, frf,
                                wm_mask_image, output_path)

    # Free some memory, we don't need the data anymore
    dwi_image.uncache()

    # Save unnormalized processed data
    if save_intermediate:                                                                       # Est-ce que le nom pourrait refléter mieux le modèle?
        model_input_image = nib.Nifti1Image(model_input_volume,
                                            dwi_image.affine)
        output_fname = "{}_model_input_unnormalized.nii.gz".format(subject_id)
        nib.save(model_input_image, str(output_path.joinpath(output_fname)))                            # Antoine enregistrait dwi ET sh.

    # Create and save normalized volume (using WM mask)
                                                                                                    # Antoine: with Timer("Normalizing input", newline=True):
    normalized_volume = normalize_data_volume(model_input_volume,
                                              wm_mask_image.get_fdata())                                    # Antoine a WM mask et normalization mask qui sont différents.
                                                                                                        # Erreur de type??
    normalized_image = nib.Nifti1Image(normalized_volume, dwi_image.affine)
    if save_intermediate:
        output_fname = "{}_model_input_normalized.nii.gz".format(subject_id)
        nib.save(normalized_image, str(output_path.joinpath(output_fname)))

    # Load, process and save streamlines
    tractogram, lengths = dataset_creator.load_process_and_merge_bundles(
        subject_data_path.joinpath("bundles"), dwi_image)
    if save_intermediate:
        save_tractogram(tractogram, str(output_path.joinpath(
            "{}_all_streamlines.tck".format(subject_id))))

    return normalized_image, tractogram.streamlines, lengths


if __name__ == '__main__':
    main()
