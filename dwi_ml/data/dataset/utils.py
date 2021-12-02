# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.experiment_utils.prints import format_dict_to_str


def add_args_dataset(p: argparse.ArgumentParser):
    dataset_group = p.add_argument_group("Dataset:")
    dataset_group.add_argument(
        '--cache_size', type=int, metavar='s', default=1,
        help="Relevant only if lazy data is used. Size of the cache in terms "
             "of length \nof the queue (i.e. number of volumes). NOTE: Real "
             "cache size will actually be \ntwice this value as the "
             "training and validation subsets each have their cache.\n"
             "Default: 1.")
    dataset_group.add_argument(
        '--lazy', action='store_true',
        help="If set, do not load all the dataset in memory at once. Load "
             "only what \nis needed for a batch.")


def prepare_multisubjectdataset(args):
    """
    Instantiate a MultiSubjectDataset and load data.
    """
    with Timer("\n\nPreparing testing and validation sets",
               newline=True, color='blue'):
        dataset = MultiSubjectDataset(
            args.hdf5_file, cache_size=args.cache_size, lazy=args.lazy,
            # toDo
            taskman_managed=args.taskman_managed)
        dataset.load_data()

        logging.info("Dataset attributes: \n" +
                     format_dict_to_str(dataset.params))

    return dataset
