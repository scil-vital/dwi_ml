import logging
import pathlib
import re
from typing import Dict, List

from dipy.core.gradients import (gradient_table)
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import length
import nibabel as nib
import numpy as np
from scilpy.tracking.tools import (filter_streamlines_by_length,
                                   resample_streamlines_step_size)
from scilpy.io.streamlines import compress_sft
from scilpy.? import subsample_sft_francois

from dwi_ml.data.creation.subjects_validation import validate_subject_list
from dwi_ml.experiment.timer import Timer

class BundleConfig(object):
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


class DatasetCreator(object):
    def __init__(self, training_subjects: List[str],
                 validation_subjects: List[str], dwi_ml_folder: pathlib.Path,
                 minimum_length_mm: float = None, step_size_mm: float = None,
                 bundles: Dict = None):
        """
        Parameters
        ----------
        training_subjects: List[str]
            List of the subjects in the training set.
        validation_subjects: List[str]
            List of the subjects in the validation set.
        dwi_ml_folder: pathlib.Path
            Path to your dwi_ml_ready folder. Folders should follow description
            in dwi_ml.data.creation.description_data_structure.py.
        minimum_length_mm : float
            Remove streamlines shorter than this length.
        step_size : float
            Step size to resample streamlines (in mm).
        bundles : dict
            Bundle-wise parameters; these should include the name and
            subsampling parameters OR If empty, datasets will be treated as
            wholebrain tractograms.
        """
        self.minimum_length_mm = minimum_length_mm
        self.step_size = step_size_mm
        self.train_subjs = training_subjects
        self.valid_subjs = validation_subjects
        self.dwi_ml_folder = dwi_ml_folder

        # Check if chosen subjects exist in dwi_ml folder.
        self.chosen_subjs = training_subjects + validation_subjects
        self.verify_subject_lists()

        if bundles:
            # Bundle-specific options
            self.bundles = []
            for name, config_dict in bundles.items():
                try:
                    bundle_config = BundleConfig(
                        name,
                        config_dict["clustering_threshold_mm"],
                        config_dict["removal_distance_mm"]
                    )
                    self.bundles.append(bundle_config)
                except KeyError as e:
                    raise ValueError("Bundle {} is missing configuration "
                                     "parameters: {}".format(name, e))
        else:
            # Datasets will be treated as wholebrain tractograms in
            # load_and_process_streamlines
            self.bundles = None

    def verify_subject_lists(self):
        """
        Raises error if some subjects in training set or validation set don't
        exist.

        Prints logging info if some subjects in dwi_ml folder were not chosen
        in training set nor validation set.
        """
        # Find list of existing subjects from folders
        all_subjs = [str(f.name) for f in self.dwi_ml_folder.iterdir()]
        if len(all_subjs) == 0:
            raise ValueError('No subject found in dwi_ml folder!')

        # Checking chosen_subjs
        non_existing_subjs, good_chosen_subjs, ignored_subj = \
                validate_subject_list(all_subjs, self.chosen_subjs)
        if len(non_existing_subjs) > 0:
            raise ValueError('Following subjects were chosen either for '
                             'training set or validation set but their folders '
                             'were not found in dwi_ml: '
                             '{}'.format(non_existing_subjs))
        if len(ignored_subj) > 0:
            logging.info("Careful! NOT processing subjects {} "
                         "because they were not included in training set nor "
                         "validation set!".format(ignored_subj))

    def get_state_dict(self):
        """
        Get a dictionary representation. Useful if this object is used to
        create a HDF file (see
        dwi_ml.scripts_python.1_data_creation.create_hdf5_ML_dataset.py)
        """
        return {'training_subject_ids':
                self.train_subjs,
                'validation_subject_ids':
                self.valid_subjs,
                'minimum_length_mm':
                self.minimum_length_mm if self.minimum_length_mm else "",
                'step_size_mm':
                self.step_size if self.step_size else "",
                'bundles':
                [b.name for b in self.bundles] if self.bundles else ""}

    def load_process_and_merge_bundles(self,
                                       bundles_path: pathlib.Path,
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
            if not self.bundles:
                # If no bundles described in the json file, we will treat the files
                # found in bundles as wholebrain tractograms
                chosen_bundles_config = [BundleConfig(p.stem) for p in
                                         bundles_path.glob('*')]
                if len(chosen_bundles_config) == 0:
                    raise ValueError("No bundles found in the boundles folder!")
            else:
                chosen_bundles_config = self.bundles
            available_bundles = list(bundles_path.iterdir())

            for bundle_config in chosen_bundles_config:
                bundle, bundle_original_count = self._load_and_process_one_bundle(
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

    def _load_and_process_one_bundle(self, bundle_config: BundleConfig,
                                     available_bundles,
                                     bundles_path: pathlib.Path,
                                     dwi_ref: nib.Nifti1Image):

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
        bundle = filter_streamlines_by_length(bundle,                                                                           # toDo. Bundle has to be a sft.
                                              self.minimum_length_mm)
        logging.debug("Removed streamlines under "                                                  
                      "{}mm; Remaining: {}".format(self.minimum_length_mm,
                                                   len(bundle)))

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
        if self.step_size:
            bundle = resample_streamlines_step_size(bundle, self.step_size)
            logging.debug("Resampled streamlines' step size to {}mm"
                          .format(self.step_size))
        else:  # If no step size is defined, compress the streamlines
            bundle = compress_sft(bundle)

        return bundle, bundle_original_count

