# -*- coding: utf-8 -*-
import datetime
import logging
import os
from pathlib import Path
from typing import List

from dipy.io.stateful_tractogram import set_sft_logger_level, Space
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.utils import length
import h5py
from nested_lookup import nested_lookup
import nibabel as nib
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft, concatenate_sft

from dwi_ml.data.io import load_file_to4d
from dwi_ml.data.processing.dwi.dwi import standardize_data


def _load_and_verify_file(file_name, subj_input_path, group, group_affine,
                          group_res):
    data_file = subj_input_path.joinpath(file_name)

    logging.info("       - Processing file {}".format(file_name))

    if not data_file.is_file():
        logging.debug("      Skipping file {} because it was not "
                      "found in this subject's folder".format(file_name))
        # Note: if args.enforce_files_presence was set to true, this
        # case is not possible, already checked in
        # create_hdf5_dataset.py.
        return None

    data, affine, res, _ = load_file_to4d(data_file)

    if not np.allclose(affine, group_affine, atol=1e-5):
        # Note. When keeping default options on tolerance, we have ran
        # into some errors in some cases, depending on how data has
        # been processed. Now accepting bigger error.
        raise ValueError(
            'Data file {} does not have the same affine as other '
            'files in group {}. Data from each group will be '
            'concatenated, and should have the same affine and voxel '
            'resolution.\n'
            'Affine: {}\n'
            'Group affine: {}\n'
            'Biggest difference: {}'
            .format(file_name, group, affine, group_affine,
                    np.max(affine - group_affine)))

    if not np.allclose(res, group_res):
        raise ValueError(
            'Data file {} does not have the same resolution as other '
            'files in group {}. Data from each group will be '
            'concatenated, and should have the same affine and voxel '
            'resolution.\n'
            'Resolution: {}\n'
            'Group resolution: {}'
            .format(file_name, group, res, group_res))

    return data


class HDF5Creator:
    """
    Creates a hdf5 file with:
    - One group per subject
            - One group per 'volume' group in the config file, where 4D-data is
            the concatenation of every MRI volume listed for this group.
                    - Volumes are standardized as set in the config file.
            - One group per 'streamlines' group where data is the decomposed
            SFT containing concatenation of all tractograms listed for this
            group.
                    - SFT was resampled / compressed as stated in argument.

    See the doc for an example of config file.
    """
    HDF_DATABASE_VERSION = 2

    def __init__(self, root_folder: Path, saving_name,
                 training_subjs: List[str], validation_subjs: List[str],
                 testing_subjs: List[str], groups_config: dict,
                 std_mask: str, step_size: float, compress: bool,
                 space: Space, enforce_files_presence: bool = True,
                 save_intermediate: bool = False):
        """
        Params
        ------
        root_folder: Path
            Path to the dwi_ml_ready folder containing all data. See the doc
            for the suggested data organization.
        saving_path: Path
            Path + filename where to save the final hdf5 file.
        training_subjs: List[str],
        validation_subjs: List[str]
        testing_subj: List[str]
            List of subject names for each data set.
        groups_config: dict
            Information from json file loaded as a dict.
        std_mask: str
            Name of the standardization mask inside each subject's folder.
        standardization: str
            One of ['all', 'independent', 'per_file', 'none'].
        step_size: float
            Step size to resample streamlines.
        compress: bool
            Compress streamlines.
        space: Space
            Space to place the tractograms.
        enforce_file_presence: bool
            If true, will stop if some files are not available for a subject.
            Default: True.
        save_intermediate: bool
            If true, intermediate files will be saved for debugging purposes.
            Default: False.
        """

        # Mandatory
        self.root_folder = root_folder
        self.saving_path = saving_name
        self.training_subjs = training_subjs
        self.validation_subjs = validation_subjs
        self.testing_subjs = testing_subjs
        self.groups_config = groups_config
        self.step_size = step_size
        self.compress = compress
        self.space = space

        # Optional
        self.std_mask = std_mask  # (could be None)
        self.save_intermediate = save_intermediate
        self.enforce_files_presence = enforce_files_presence

        # ------- Reading groups config

        self.volume_groups, self.streamline_groups = \
            self._analyse_config_file()

        # -------- Performing checks

        # Check that all subjects exist.
        self.all_subjs = training_subjs + validation_subjs + testing_subjs
        self._verify_subjects_list()

        # Check that all files exist
        if enforce_files_presence:
            self._check_files_presence()

    def _analyse_config_file(self):
        """
        Reads the groups config json file and find:
        - List of groups. Their type should be one of 'volume' or 'streamlines'
        - For volume groups: 'standardization' value should be provided and one
          of 'all', 'independent', 'per_file' or 'none'.

        Returns the list of volume groups and streamline groups.
        """
        volume_groups = []
        streamline_groups = []
        for group in self.groups_config.keys():
            if 'type' not in self.groups_config[group]:
                raise KeyError("Group {}'s type was not defined. It should be "
                               "the group type (either 'volume' or "
                               "'streamlines'). See the doc for a "
                               "groups_config.json example.".format(group))
            if 'files' not in self.groups_config[group]:
                raise KeyError(
                    "Group {}'s files were not defined. It should list "
                    "the files to load and concatenate for this group. "
                    "See the doc for a groups_config.json example."
                    .format(group))

            # Volume groups
            if self.groups_config[group]['type'] == 'volume':
                std_choices = ['all', 'independent', 'per_file', 'none']
                if 'standardization' not in self.groups_config[group]:
                    raise KeyError(
                        "Group {}'s 'standardization' was not defined. It "
                        "should be one of {}. See the doc for a "
                        "groups_config.json example."
                        .format(group, std_choices))
                if self.groups_config[group]['standardization'] not in \
                        std_choices:
                    raise KeyError(
                        "Group {}'s 'standardization' should be one of {}, "
                        "but we got {}. See the doc for a groups_config.json "
                        "example."
                        .format(group, std_choices,
                                self.groups_config[group]['standardization']))
                volume_groups.append(group)

            # Streamline groups
            elif self.groups_config[group]['type'] == 'streamlines':
                streamline_groups.append(group)

            else:
                raise ValueError(
                    "Group {}'s type should be one of volume or streamlines "
                    "but got {}"
                    .format(group, self.groups_config[group]['type']))

        logging.info("Volume groups: {}".format(volume_groups))
        logging.info("Streamline groups: {}".format(streamline_groups))
        return volume_groups, streamline_groups

    def _verify_subjects_list(self):
        """
        Raises error if some subjects do not exit in the root folder. Prints
        logging info if some subjects in the root folder were not chosen.
        """
        # Find list of subjects existing  inside folder
        all_subjs = [str(s.name) for s in Path(self.root_folder).iterdir()
                     if s.is_dir()]
        if len(all_subjs) == 0:
            raise ValueError('No subject found in dwi_ml folder: '
                             '{}'.format(self.root_folder))

        # Checking that chosen subjects exist.
        non_existing_subjs = [s for s in all_subjs if s not in all_subjs]
        if len(non_existing_subjs) > 0:
            raise ValueError(
                'Following subjects were chosen for the hdf5 file but their '
                'folders were not found in {}:\n {}'
                .format(self.root_folder, non_existing_subjs))

        # Checking if some existing subjects were not chosen.
        ignored_subj = [s for s in all_subjs if s not in all_subjs]
        if len(ignored_subj) > 0:
            logging.info("    Careful! NOT processing subjects {} "
                         "because they were not included in training set nor "
                         "validation set!".format(ignored_subj))

    def _check_files_presence(self):
        """
        Verifying now the list of files. Prevents stopping after a long
        processing time if a file does not exist.

        The list of files to verify for each subject is :
         - the standardization mask
         - all files in the group_config file
        """
        logging.debug("Verifying files presence")

        # concatenating files from all groups files:
        config_file_list = sum(nested_lookup('files', self.groups_config), [])

        for subj_id in self.all_subjs:
            subj_input_dir = Path(self.root_folder).joinpath(subj_id)

            # Find subject's standardization mask
            for sub_mask in self.std_mask:
                sub_std_mask_file = subj_input_dir.joinpath(
                    sub_mask.replace('*', subj_id))
                if not sub_std_mask_file.is_file():
                    raise FileNotFoundError(
                        "Standardization mask {} not found for subject {}!"
                        .format(sub_std_mask_file, subj_id))

            # Find subject's files from group_config
            for this_file in config_file_list:
                this_file = this_file.replace('*', subj_id)
                if this_file.endswith('/ALL'):
                    logging.debug(
                        "    Keyword 'ALL' detected; we will load all "
                        "files in the folder '{}'"
                        .format(this_file.replace('/ALL', '')))
                else:
                    this_file = subj_input_dir.joinpath(this_file)
                    if not this_file.is_file():
                        raise FileNotFoundError(
                            "File from groups_config ({}) not found for "
                            "subject {}!".format(this_file, subj_id))

    def create_database(self):
        """
        Generate a dataset from a group of dMRI subjects with multiple bundles.
        All data from each group are concatenated.
        All bundles are merged as a single whole-brain dataset in voxel space.
        If wished, all intermediate steps are saved on disk in the hdf5 folder.
        """
        with h5py.File(self.saving_path, 'w') as hdf_handle:
            # Save version and configuration
            hdf_handle.attrs['version'] = self.HDF_DATABASE_VERSION
            now = datetime.datetime.now()
            hdf_handle.attrs['data_and_time'] = now.strftime('%d %B %Y %X')
            hdf_handle.attrs['chosen_subjs'] = self.all_subjs
            hdf_handle.attrs['groups_config'] = str(self.groups_config)
            hdf_handle.attrs['training_subjs'] = self.training_subjs
            hdf_handle.attrs['validation_subjs'] = self.validation_subjs
            hdf_handle.attrs['testing_subjs'] = self.testing_subjs
            hdf_handle.attrs['step_size'] = self.step_size if \
                self.step_size is not None else 'Not defined by user'
            hdf_handle.attrs['space'] = self.space.name

            # Add data one subject at the time
            nb_processed = 0
            nb_subjs = len(self.all_subjs)
            logging.debug("Processing {} subjects : {}"
                          .format(nb_subjs, self.all_subjs))
            for subj_id in self.all_subjs:
                nb_processed += 1
                logging.info("*Processing subject {}/{}: {}"
                             .format(nb_processed, nb_subjs, subj_id))
                self._create_one_subj(subj_id, hdf_handle)

        logging.info("Saved dataset : {}".format(self.saving_path))

    def _create_one_subj(self, subj_id, hdf_handle):
        """
        Creating one subject's data as a hdf5 group.
        """
        subj_input_dir = self.root_folder.joinpath(subj_id)

        subj_hdf_group = hdf_handle.create_group(subj_id)

        # Prepare subject folder for intermediate files
        subj_intermediate_path = self.saving_path.parent.joinpath(
            subj_id + "_intermediate")
        if self.save_intermediate:
            subj_intermediate_path.mkdir()

        # Find subject's standardization mask
        subj_std_mask_data = None
        for sub_mask in self.std_mask:
            sub_mask = sub_mask.replace('*', subj_id)
            logging.info("    - Loading standardization mask {}"
                         .format(sub_mask))
            sub_mask_file = subj_input_dir.joinpath(sub_mask)
            sub_mask_img = nib.load(sub_mask_file)
            sub_mask_data = np.asanyarray(sub_mask_img.dataobj) > 0
            if subj_std_mask_data is None:
                subj_std_mask_data = sub_mask_data
            else:
                subj_std_mask_data = np.logical_or(sub_mask_data,
                                                   subj_std_mask_data)

        # Add the subj data based on groups in the json config file
        ref = self._create_volume_groups(
            subj_id, subj_input_dir, subj_intermediate_path,
            subj_std_mask_data, subj_hdf_group)

        self._create_streamline_groups(ref, subj_input_dir, subj_id,
                                       subj_intermediate_path, subj_hdf_group)

    def _create_volume_groups(self, subj_id, subj_input_dir,
                              subj_intermediate_path, subj_std_mask_data,
                              subj_hdf_group):
        """
        Loop on all volume groups for a given subject and create the hdf5
        group.
        """
        group_header = None
        for group in self.volume_groups:
            logging.info("    - Processing volume group '{}'...".format(group))

            (group_data, group_affine,
             group_header, group_res) = self._process_one_volume_group(
                group, subj_id, subj_input_dir, subj_intermediate_path,
                subj_std_mask_data)
            logging.debug('      *Done. Now creating dataset from group.')
            hdf_group = subj_hdf_group.create_group(group)
            hdf_group.create_dataset('data', data=group_data)
            logging.debug('      *Done.')

            # Saving data information.
            subj_hdf_group[group].attrs['affine'] = group_affine
            subj_hdf_group[group].attrs['type'] = self.groups_config[group][
                'type']
            subj_hdf_group[group].attrs['voxres'] = group_res

            # Adding the shape info separately to access it without loading
            # the data (useful for lazy data!).
            subj_hdf_group[group].attrs['nb_features'] = group_data.shape[-1]
        return group_header

    def _process_one_volume_group(self, group: str, subj_id: str,
                                  subj_input_path: Path,
                                  subj_output_path: Path,
                                  subj_std_mask_data: np.ndarray = None):
        """
        Process each group from the json config file:
        - Load data from each file of the group and combine them. All datasets
          from a given group must have the same affine, voxel resolution and
          data shape.
          Note. Wildcards will be replaced by the subject id.
        - Standardize data

        Parameters
        ----------
        group: str
            Group name.
        subj_id: str
            The subject's id.
        subj_input_path: Path
            Path where the files from file_list should be found.
        subj_output_path: Path
            Path where to save the intermediate files.
        subj_std_mask_data: np.ndarray of bools, optional
            Binary mask that will be used for data standardization.

        Returns
        -------
        group_data: np.ndarray
            Group data created by concatenating all files, standardized.
        group_affine: np.ndarray
            Affine for the group.
        """
        standardization = self.groups_config[group]['standardization']
        file_list = self.groups_config[group]['files']

        # First file will define data dimension and affine
        file_name = file_list[0].replace('*', subj_id)
        first_file = subj_input_path.joinpath(file_name)
        logging.info("       - Processing file {}".format(file_name))
        group_data, group_affine, group_res, group_header = load_file_to4d(
            first_file)

        if standardization == 'per_file':
            logging.debug('      *Standardizing sub-data')
            group_data = standardize_data(group_data, subj_std_mask_data,
                                          independent=False)

        # Other files must fit (data shape, affine, voxel size)
        # It is not a promise that data has been correctly registered, but it
        # is a minimal check.
        for file_name in file_list[1:]:
            file_name = file_name.replace('*', subj_id)
            data = _load_and_verify_file(file_name, subj_input_path, group,
                                         group_affine, group_res)

            if standardization == 'per_file':
                logging.debug('      *Standardizing sub-data')
                data = standardize_data(data, subj_std_mask_data,
                                        independent=False)

            # Append file data to hdf group.
            try:
                group_data = np.append(group_data, data, axis=-1)
            except ImportError:
                raise ImportError(
                    'Data file {} could not be added to data group {}. '
                    'Wrong dimensions?'.format(file_name, group))

        # Standardize data (per channel) (if not done 'per_file' yet).
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
        if self.save_intermediate:
            output_fname = subj_output_path.joinpath(group + ".nii.gz")
            logging.debug('      *Saving intermediate files into {}.'
                          .format(output_fname))
            standardized_img = nib.Nifti1Image(group_data, group_affine)
            nib.save(standardized_img, str(output_fname))

        return group_data, group_affine, group_header, group_res

    def _create_streamline_groups(self, ref, subj_input_dir, subj_id,
                                  subj_intermediate_path, subj_hdf_group):
        for group in self.streamline_groups:

            # Add the streamlines data
            logging.info('    - Processing bundles...')

            if ref is None:
                logging.debug(
                    "No group_header! This means no 'volume' group was added "
                    "in the config_file. If all bundles are .trk, we can use "
                    "ref 'same' but if some bundles were .tck, we need a ref!")
            sft, lengths = self._process_one_streamline_group(
                subj_input_dir, group, subj_id, ref, subj_intermediate_path)
            streamlines = sft.streamlines

            if streamlines is None:
                logging.warning('Careful! Total tractogram for subject {} '
                                'contained no streamlines!'.format(subj_id))
            else:
                streamlines_group = subj_hdf_group.create_group('streamlines')
                streamlines_group.attrs['type'] = 'streamlines'

                # The hdf5 can only store numpy arrays (it is actually the
                # reason why it can fetch only precise streamlines from
                # their ID). We need to deconstruct the sft and store all
                # its data separately to allow reconstructing it later.
                (a, d, vs, vo) = sft.space_attributes
                streamlines_group.attrs['space'] = str(sft.space)
                streamlines_group.attrs['affine'] = a
                streamlines_group.attrs['dimensions'] = d
                streamlines_group.attrs['voxel_sizes'] = vs
                streamlines_group.attrs['voxel_order'] = vo

                if len(sft.data_per_point) > 0:
                    logging.warning('sft contained data_per_point. Data '
                                    'not kept.')
                if len(sft.data_per_streamline) > 0:
                    logging.warning('sft contained data_per_streamlines. '
                                    'Data not kept.')

                # Accessing private Dipy values, but necessary.
                # We need to deconstruct the streamlines into arrays with
                # types recognizable by the hdf5.
                streamlines_group.create_dataset('data',
                                                 data=streamlines._data)
                streamlines_group.create_dataset('offsets',
                                                 data=streamlines._offsets)
                streamlines_group.create_dataset('lengths',
                                                 data=streamlines._lengths)
                streamlines_group.create_dataset('euclidean_lengths',
                                                 data=lengths)

    def _process_one_streamline_group(
            self, subj_dir: Path, group: str, subj_id: str,
            header: nib.Nifti1Header, subj_output_path: Path):
        """
        Load and process a group of bundles and merge all streamlines
        together.

        Note. Wildcards will be replaced by the subject id. If the list is
        folder/ALL, all bundles in the folder will be used.

        Parameters
        ----------
        subj_dir : Path
            Path to bundles folder.
        group: str
            group name
        subj_id: str
            The subject's id.
        header : nib.Nifti1Header
            Reference used to load and send the streamlines in voxel space and
            to create final merged SFT. If the file is a .trk, 'same' is used
            instead.
        subj_output_path: Path
            Path where to save the intermediate files.

        Returns
        -------
        final_tractogram : StatefulTractogram
            All streamlines in voxel space.
        output_lengths : List[float]
            The euclidean length of each streamline
        """
        bundles = self.groups_config[group]['files']

        if self.step_size and self.compress:
            raise ValueError(
                "Only one option can be chosen: either resampling to "
                "step_size or compressing, not both.")

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

            # Either a loop on "ALL" or a loop on only one file,
            # tmp_bundle_name.
            for bundle_name in bundles2:
                bundle_name = bundle_name.replace('*', subj_id)
                bundle_file = subj_dir.joinpath(bundle_name)
                sft = self._load_and_process_sft(bundle_file, bundle_name,
                                                 header)

                if sft is not None:
                    # Compute euclidean lengths (rasmm space)
                    sft.to_space(Space.RASMM)
                    output_lengths.extend(length(sft.streamlines))

                    # Sending to wanted space
                    sft.to_space(self.space)

                    # Add processed bundle to output tractogram
                    if final_sft is None:
                        final_sft = sft
                    else:
                        final_sft = concatenate_sft([final_sft, sft],
                                                    erase_metadata=False)

        if self.save_intermediate:
            output_fname = subj_output_path.joinpath(group + '.trk')
            logging.debug('      *Saving intermediate bundle {} into '
                          '{}.'.format(group, output_fname))
            # Note. Do not remove the str below. Does not work well
            # with Path.
            save_tractogram(final_sft, str(output_fname))

        # Removing invalid streamlines
        logging.debug('      *Total: {:,.0f} streamlines. Now removing '
                      'invalid streamlines.'.format(len(final_sft)))
        final_sft.remove_invalid_streamlines()
        logging.debug("      *Remaining: {:,.0f} streamlines."
                      "".format(len(final_sft)))

        return final_sft, output_lengths

    def _load_and_process_sft(self, bundle_file, bundle_name, header):
        if not bundle_file.is_file():
            logging.debug(
                "      Skipping bundle {} because it was not found in this "
                "subject's folder".format(bundle_name))
            # Note: if args.enforce_files_presence was set to true,
            # this case is not possible, already checked in
            # create_hdf5_dataset
            return None

        # Check bundle extension
        _, file_extension = os.path.splitext(str(bundle_file))
        if file_extension not in ['.trk', '.tck']:
            raise ValueError(
                "We do not support bundle's type: {}. We only support .trk "
                "and .tck files.".format(bundle_file))
        if file_extension == '.trk':
            # overriding given header.
            header = 'same'

        # Loading bundle and sending to wanted space
        logging.info("       - Processing bundle {}"
                     .format(os.path.basename(bundle_name)))
        sft = load_tractogram(str(bundle_file), header)
        sft.to_center()

        # Resample or compress streamlines
        # Note. No matter the chosen space, resampling is done in
        # mm.
        if self.step_size:
            logging.info("          - Resampling")
            sft = resample_streamlines_step_size(sft, self.step_size)
            logging.debug("      *Resampled streamlines' step size to {}mm"
                          .format(self.step_size))
        elif self.compress:
            logging.info("          - Compressing")
            sft = compress_sft(sft)

        return sft
