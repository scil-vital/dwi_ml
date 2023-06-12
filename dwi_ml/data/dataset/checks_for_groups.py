# -*- coding: utf-8 -*-
import logging
from typing import Tuple

import numpy as np


def _find_groups_info_for_subj(hdf_file, subj_id: str):
    """
    Separate subject's hdf5 groups intro volume groups or streamline groups
    based on their 'type' attrs. For volume groups, verify the nb_features
    attribute.

    Params
    ------
    hdf_file: hdf handle
        Opened hdf handle to the hdf5 file.
    subj_id: str
        The subject name in the hdf5 file.

    Returns
    -------
    volume_groups: List[str]
        The list of volume groups for this subject.
    nb_features: List[int]
        The number of features for each volume (probably the size of each
        volume's last dimension).
    streamline_groups: List[str]
        The list of streamline groups for this subject.
    contains_connectivity: np.ndarray
        A list of boolean for each streamline_group stating if it contains the
        pre-computed connectivity matrices for that subject.
    """
    volume_groups = []
    nb_features = []
    streamline_groups = []
    contains_connectivity = []

    hdf_groups = hdf_file[subj_id]
    for hdf_group in hdf_groups:
        group_type = hdf_file[subj_id][hdf_group].attrs['type']
        if group_type == 'volume':
            volume_groups.append(hdf_group)
            nb_features.append(
                hdf_file[subj_id][hdf_group].attrs['nb_features'])
        elif group_type == 'streamlines':
            streamline_groups.append(hdf_group)
            found_matrix = 'connectivity_matrix' in hdf_file[subj_id][hdf_group]
            contains_connectivity.append(found_matrix)
        else:
            raise NotImplementedError(
                "So far, you can only add 'volume' or 'streamline' groups in "
                "the groups_config.json. Please see the doc for a json file "
                "example. Your hdf5 contained group of type {} for subj {}"
                .format(group_type, subj_id))

    contains_connectivity = np.asarray(contains_connectivity, dtype=bool)
    return volume_groups, nb_features, streamline_groups, contains_connectivity


def _compare_groups_info(volume_groups, nb_features, streamline_groups,
                         group_info: Tuple):
    """
    Compares the three lists (volume_groups, nb_features, streamline_groups) of
    one subject to the expected list for this database, included in group_info.
    """
    v, f, s = group_info
    if volume_groups != v:
        logging.warning("Subject's hdf5 groups with attributes 'type' set as "
                        "'volume' are not the same as expected with this "
                        "dataset! Expected: {}. Found: {}"
                        .format(v, volume_groups))
    if nb_features != f:
        logging.warning("Among subject's hdf5 groups with attributes 'type' "
                        "set as 'volume', some data to not have the same "
                        "number of features as expected for this dataset! "
                        "Expected: {}. Found: {}".format(f, nb_features))
    if streamline_groups != s:
        logging.warning("Subject's hdf5 groups with attributes 'type' set as "
                        "'streamlines' are not the same as expected with this "
                        "dataset! Expected: {}. Found: {}"
                        .format(s, streamline_groups))


def prepare_groups_info(subject_id: str, hdf_file, ref_group_info=None):
    """
    Read the hdf5 file for this subject and get the groups information
    (volume and streamlines groups names, number of features for volumes).

    If group_info is given, compare subject's information with database
    expected information.
    """
    volume_groups, nb_features, streamline_groups, contains_connectivity = \
        _find_groups_info_for_subj(hdf_file, subject_id)

    if ref_group_info is not None:
        _compare_groups_info(volume_groups, nb_features, streamline_groups,
                             ref_group_info)

    return volume_groups, nb_features, streamline_groups, contains_connectivity
