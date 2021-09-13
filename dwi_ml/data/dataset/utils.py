# -*- coding: utf-8 -*-
def find_groups_info(hdf_file, subj_id: str, log):
    """
    Separate subject's hdf5 groups intro volume groups or streamline groups
    based on their 'type' attrs.
    """
    volume_groups = []
    nb_features = []
    streamline_groups = []

    groups = hdf_file[subj_id]
    for group in groups:
        group_type = hdf_file[subj_id][group].attrs['type']
        if group_type == 'volume':
            volume_groups.append(group)
            nb_features.append(
                hdf_file[subj_id][group].attrs['nb_features'])
        elif group_type == 'streamlines':
            streamline_groups.append(group)
        else:
            raise NotImplementedError(
                "So far, you can only add volume groups in the "
                "groups_config.json. As for the streamlines, they are "
                "added through the option --bundles. Please see the doc "
                "for a json file example. You tried to add data of type: "
                "{}".format(group_type))

    log.info("Volume groups are: {}".format(volume_groups))
    log.info("Number of features in each of these groups: {}"
                  .format(nb_features))
    log.info("Streamline groups are: {}".format(streamline_groups))

    return volume_groups, nb_features, streamline_groups


def compare_groups_info(volume_groups, nb_features, streamline_groups,
                        group_info):
    v, f, s = group_info
    if volume_groups != v:
        raise Warning("Subject's hdf5 groups with attributes 'type' set as "
                      "'volume' are not the same as expected with this "
                      "dataset! Expected: {}. Found: {}"
                      .format(v, volume_groups))
    if nb_features != f:
        raise Warning("Among subject's hdf5 groups with attributes 'type' set "
                      " as 'volume', some data to not have the same number of "
                      "features as expected for this dataset! Expected: {}. "
                      "Found: {}".format(f, nb_features))
    if streamline_groups != s:
        raise Warning("Subject's hdf5 groups with attributes 'type' set as "
                      "'streamlines' are not the same as expected with this "
                      "dataset! Expected: {}. Found: {}"
                      .format(s, streamline_groups))


def prepare_groups_info(subject_id: str, log, hdf_file, group_info=None):
    volume_groups, nb_features, streamline_groups = \
        find_groups_info(hdf_file, subject_id, log)

    if group_info is not None:
        compare_groups_info(volume_groups, nb_features, streamline_groups,
                            group_info)

    return volume_groups, nb_features, streamline_groups
