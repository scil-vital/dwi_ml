# -*- coding: utf-8 -*-
import logging
import os
import pathlib
import re
from typing import List

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import (load_tractogram, save_tractogram)
from dipy.tracking.utils import length
import nibabel as nib
import numpy as np
from scilpy.tracking.tools import (filter_streamlines_by_length,
                                   resample_streamlines_step_size)
from scilpy.io.streamlines import compress_sft
from scilpy.? import subsample_sft_francois

from dwi_ml.data.creation.subjects_validation import validate_subject_list
from dwi_ml.data.processing.dwi.dwi import standardize_data
from dwi_ml.experiment.timer import Timer

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


def _load_and_check_4D_nii_data(data_file):
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
    group_data, group_affine = _load_and_check_4D_nii_data(first_file)

    # Other files must fit
    for data_file in file_list[1:]:
        data, _ = _load_and_check_4D_nii_data(data_file)
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


class BundleConfig:
    """Bundle configuration parameters."""

    def __init__(self, name: str, clustering_threshold_mm: float = None,
                 removal_distance_mm: float = None):
        """
        Parameters
        ----------
        name : str
            The name of the bundle.
        clustering_threshold_mm : float
            The clustering threshold applied before removing similar
            streamlines.
        removal_distance_mm : float
            The removal threshold used to remove similar streamlines.
        """
        self.name = name
        self.clustering_threshold_mm = clustering_threshold_mm
        self.removal_distance_mm = removal_distance_mm


def _arrange_bundles(bundles):
    """
    ?
    """
    arranged_bundles = []
    for name, config_dict in bundles.items():
        try:
            bundle_config = BundleConfig(name,
                                         config_dict["clustering_threshold_mm"],
                                         config_dict["removal_distance_mm"])
            arranged_bundles.append(bundle_config)
        except KeyError as e:
            raise ValueError("Bundle {} is missing configuration "
                             "parameters: {}".format(name, e))
    return arranged_bundles


def _load_and_process_one_bundle(bundle_config: BundleConfig,
                                 available_bundles, bundles_path: pathlib.Path,
                                 dwi_ref: nib.Nifti1Image, minimum_length_mm,
                                 step_size):
    """
    ?
    """
    # Find the bundle
    regex = re.compile(".*_{}.t([rc])k".format(bundle_config.name))
    matches = [b for b in available_bundles if re.match(regex, str(b))]
    if len(matches) == 0:
        logging.warning("Bundle {} was not found in "
                        "path: {}".format(bundle_config.name,
                                          str(bundles_path)))
        return None
    if len(matches) > 1:
        raise ValueError("Bundle {} has matched "
                         "multiple files: {}"
                         .format(bundle_config.name, matches))
    bundle_file = matches[0]

    # Load the bundle
    logging.info("Processing bundle: {}".format(bundle_file))
    bundle = load_tractogram(str(bundle_file), reference=dwi_ref,
                             to_space=Space.RASMM,
                             trk_header_check=False,
                             bbox_valid_check=False)
    if len(bundle) == 0:
        logging.warning("Bundle {} contains 0 streamlines, "
                        "skipping...".format(str(bundle_file)))
        return None

    # Keep count of the original number of streamlines
    bundle_original_count = len(bundle)
    logging.debug("Bundle contains {} streamlines"
                  .format(bundle_original_count))

    # Remove streamlines that are too short
    bundle = filter_streamlines_by_length(bundle, minimum_length_mm)
    logging.debug("Removed streamlines under "                                                  
                  "{}mm; Remaining: {}".format(minimum_length_mm, len(bundle)))

    # Subsample bundles to keep only the closest to centroid (only if we
    # have bundle information, i.e. not wholebrain)
    if (bundle_config.clustering_threshold_mm is not None
            and bundle_config.removal_distance_mm is not None):
        bundle = subsample_sft_francois(bundle,
                                        bundle_config.clustering_threshold_mm,
                                        bundle_config.removal_distance_mm)
        logging.debug("Subsampled bundle using clustering "
                      "threshold of  {}mm and a removal distance of "
                      "{}mm; Remaining: {}"
                      .format(bundle_config.clustering_threshold_mm,
                              bundle_config.removal_distance_mm,
                              len(bundle)))

    # Resample streamlines to have all the same step size
    if step_size:
        bundle = resample_streamlines_step_size(bundle, step_size)
        logging.debug("Resampled streamlines' step size to {}mm"
                      .format(step_size))
    else:  # If no step size is defined, compress the streamlines
        bundle = compress_sft(bundle)

    return bundle, bundle_original_count


def _load_process_and_merge_bundles(bundles_path: pathlib.Path, bundles,
                                    dwi_ref: nib.Nifti1Image):
    """Load and process a group of bundles and merge all streamlines
    together.

    Parameters
    ----------
    bundles_path : pathlib.Path
        Path to bundles folder.
    dwi_ref : np.ndarray
        Reference used to load and send the streamlines in voxel space.

    Returns
    -------
    output_tractogram : StatefulTractogram
        All streamlines in voxel space.
    output_lengths : List[float]
        The euclidean length of each streamline
    """

    with Timer("Processing streamlines", newline=True):

        # Initialize
        output_tractogram = None
        output_lengths = []
        n_original_streamlines = 0
        if not bundles:
            # If no bundles described in the json file, we will treat the files
            # found in bundles as wholebrain tractograms
            chosen_bundles_config = [BundleConfig(p.stem) for p in
                                     bundles_path.glob('*')]
            if len(chosen_bundles_config) == 0:
                raise ValueError("No bundles found in the boundles folder!")
        else:
            chosen_bundles_config = bundles
        available_bundles = list(bundles_path.iterdir())

        for bundle_config in chosen_bundles_config:
            bundle, bundle_original_count = _load_and_process_one_bundle(
                bundle_config, available_bundles, bundles_path, dwi_ref)
            if bundle is None:
                continue

            # Keep track of original count
            n_original_streamlines += bundle_original_count

            # Compute euclidean lengths
            output_lengths.extend(length(bundle.streamlines))

            # Add processed bundle to output tractogram
            if output_tractogram is None:
                output_tractogram = bundle
            else:
                # Validate that tractograms are in the same space
                # Function doesnt exist anymore but should not be necessary
                # if we use SFT.
                assert are_tractograms_in_same_space(output_tractogram,
                                                     bundle),\
                    "Inconsistent tractogram space: {}".format(bundle)
                output_tractogram.streamlines.extend(bundle.streamlines)

        # Transfer the streamlines to the reference space before bringing them
        # to VOX space. NOTE: This is done in case the streamlines were tracked
        # in a different space than the provided dataset reference
        if output_tractogram is None:
            output_streamlines_rasmm = []
        else:
            output_streamlines_rasmm = output_tractogram.streamlines
        output_tractogram = StatefulTractogram(output_streamlines_rasmm,
                                               dwi_ref,
                                               space=Space.RASMM)

        # Internal validation check
        output_tractogram.remove_invalid_streamlines()
        logging.debug("Ran internal tractogram validation; "
                      "Remaining: {}".format(len(output_tractogram)))

        # Final nb of streamlines
        logging.info("Final number of streamlines : "
                     "{} / {}".format(len(output_tractogram),
                                      n_original_streamlines))

        # Send to VOX space and make sure the origin is at the CENTER of the
        # voxel. NOTE: This is really important, otherwise interpolation will
        # be off by half a voxel.
        output_tractogram.to_vox()
        output_tractogram.to_center()

    return output_tractogram, output_lengths


def process_streamlines(subject_id: str, subject_data_path: pathlib.Path,
                     output_path: pathlib.Path,
                     save_intermediate: bool = False):

    tractogram, lengths = _load_process_and_merge_bundles(
        subject_data_path.joinpath("bundles"), dwi_image)
    if save_intermediate:
        save_tractogram(tractogram, str(output_path.joinpath(
            "{}_all_streamlines.tck".format(subject_id))))

    return tractogram.streamlines, lengths