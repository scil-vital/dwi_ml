# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment_utils.timer import Timer


def add_dataset_args(p: argparse.ArgumentParser):
    """
    Optional arguments that should be added to an argparser in order to use a
    MultisubjectDataset.
    """
    dataset_group = p.add_argument_group("Dataset")
    dataset_group.add_argument(
        '--cache_size', type=int, metavar='s', default=1,
        help="Relevant only if lazy data is used. Size of the cache in terms\n"
             "of length of the queue (i.e. number of volumes). NOTE: Real \n"
             "cache size will actually be twice this value as the "
             "training \nand validation subsets each have their cache. [1]")
    dataset_group.add_argument(
        '--lazy', action='store_true',
        help="If set, do not load all the dataset in memory at once. Load \n"
             "only what is needed for a batch.")


def prepare_multisubjectdataset(args, load_training=True, load_validation=True,
                                load_testing=True):
    """
    Instantiates a MultiSubjectDataset AND loads data.

    Params
    ------
    args: Namespace
        Must contain 'hdf5_File, 'taskman_managed', 'lazy' and 'cache_size'
    """
    with Timer("\nPreparing testing and validation sets",
               newline=True, color='blue'):
        dataset = MultiSubjectDataset(
            args.hdf5_file, taskman_managed=args.taskman_managed,
            lazy=args.lazy, subset_cache_size=args.cache_size)
        dataset.load_data(load_training, load_validation, load_testing)

        logging.info("Number of subjects loaded: \n"
                     "      Training: {}\n"
                     "      Validation: {}\n"
                     "      Testing: {}"
                     .format(dataset.training_set.nb_subjects,
                             dataset.validation_set.nb_subjects,
                             dataset.testing_set.nb_subjects))

    return dataset
