# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from typing import List

from dwi_ml.io_utils import add_resample_or_compress_arg


def format_nb_blocs_connectivity(connectivity_nb_blocs) -> List:
    """
    Convert the raw option for connectivity into a list of 3 values.
    Ex: [10, 20, 10] is returned without modification.
    Ex: 20 becomes [20, 20, 20]
    With other values (ex, a list of <>3 values), an error is raised.
    """
    if connectivity_nb_blocs is None:
        # Default/const value with argparser '+' not possible.
        # Setting it manually.
        connectivity_nb_blocs = 20
    elif (isinstance(connectivity_nb_blocs, list) and
          len(connectivity_nb_blocs) == 1):
        connectivity_nb_blocs = connectivity_nb_blocs[0]

    if isinstance(connectivity_nb_blocs, List):
        assert len(connectivity_nb_blocs) == 3, \
            "Expecting to work with 3D volumes. Expecting " \
            "connectivity_nb_blocs to be a list of 3 values, " \
            "but got {}.".format(connectivity_nb_blocs)
    else:
        assert isinstance(connectivity_nb_blocs, int), \
            "Expecting the connectivity_nb_blocs to be either " \
            "a 3D list or an integer, but got {}" \
            .format(connectivity_nb_blocs)
        connectivity_nb_blocs = [connectivity_nb_blocs] * 3

    return connectivity_nb_blocs


def add_hdf5_creation_args(p: ArgumentParser):

    # Positional arguments
    p.add_argument('dwi_ml_ready_folder',
                   help="Path to the folder containing the data. \n"
                        "-> Should follow description in our doc, here: \n"
                        "-> https://dwi-ml.readthedocs.io/en/latest/"
                        "creating_hdf5.html")
    p.add_argument('out_hdf5_file',
                   help="Path and name of the output hdf5 file. \nIf "
                        "--save_intermediate is set, the intermediate files "
                        "will be saved in \nthe same location, in a folder "
                        "name based on date and hour of creation.\n"
                        "If it already exists, use -f to allow overwriting.")
    p.add_argument('config_file',
                   help="Path to the json config file defining the groups "
                        "wanted in your hdf5. \n"
                        "-> Should follow description in our doc, here: \n"
                        "-> https://dwi-ml.readthedocs.io/en/latest/"
                        "creating_hdf5.html")
    p.add_argument('training_subjs',
                   help="A txt file containing the list of subjects ids to "
                        "use for training. \n(Can be an empty file.)")
    p.add_argument('validation_subjs',
                   help="A txt file containing the list of subjects ids to use "
                        "for validation. \n(Can be an empty file.)")
    p.add_argument('testing_subjs',
                   help="A txt file containing the list of subjects ids to use "
                        "for testing. \n(Can be an empty file.)")

    # Optional arguments
    p.add_argument('--enforce_files_presence', type=bool, default=True,
                   metavar="True/False",
                   help='If True, the process will stop if one file is '
                        'missing for a subject. \nChecks are not made for '
                        'option "ALL" for streamline groups.\nDefault: True')
    p.add_argument('--save_intermediate', action="store_true",
                   help="If set, save intermediate processing files for "
                        "each subject inside the \nhdf5 folder, in sub-"
                        "folders named subjid_intermediate.\n"
                        "(Final concatenated standardized volumes and \n"
                        "final concatenated resampled/compressed streamlines.)")


def add_streamline_processing_args(p: ArgumentParser):
    g = p.add_argument_group('Streamlines processing options:')
    add_resample_or_compress_arg(g)
