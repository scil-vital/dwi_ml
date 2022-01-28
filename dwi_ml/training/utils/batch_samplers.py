# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.batch_samplers import DWIMLBatchSampler


def add_args_batch_sampler(p: argparse.ArgumentParser):
    # BATCH SIZE
    g_batch_size = p.add_argument_group("Batch sampler: batch size")

    g_batch_size.add_argument(
        '--batch_size', type=int, default=100, metavar='s',
        help="Batch size. Can be defined in number of streamlines or in total "
             "length_mm (specified through batch_size_units).\n The total "
             "number your computer will accept depends on the type of input "
             "data. \nYou will need to test this value. Suggestion: in "
             "nb_streamlines: 100. In length_mm: 10000. Default: 100.")
    g_batch_size.add_argument(
        '--batch_size_units', type=str, metavar='u',
        choices={'nb_streamlines', 'length_mm'},
        help="'nb_streamlines' or 'length_mm' (which should hopefully be "
             "correlated to the number of input data points).")
    g_batch_size.add_argument(
        '--nb_streamlines_per_chunk', type=int, default=256, metavar='s',
        help="In the case of a batch size in terms of 'length_mm', chunks of "
             "n streamlines are sampled at once,\n and then their size is "
             "checked, either removing streamlines if exceeded, or else "
             "sampling a new chunk of ids.\n Default: 256, for no good "
             "reason.")
    g_batch_size.add_argument(
        '--nb_subjects_per_batch', type=int, metavar='n',
        help="Maximum number of different subjects from which to load data in "
             "each batch. \nThis should help avoid loading too many inputs in "
             "memory, particularly for \nlazy data. If not set, we will use "
             "true random sampling. \nSuggestion, 5. Hint: Will influence the "
             "cache size if the cache_manager is used.")
    g_batch_size.add_argument(
        '--cycles', type=int, metavar='c',
        help="Relevant only if training:batch:nb_subject_per_batch is set.\n"
             "Number of cycles before changing to new subjects (and thus new "
             "volumes).")


def prepare_batchsamplers_train_valid(dataset, args_training, args_validation):
    with Timer("\nPreparing batch samplers...", newline=True, color='green'):
        logging.info("Training batch sampler...")
        training_batch_sampler = _prepare_batchsampler(
            dataset.training_set, args_training)

        if dataset.validation_set.nb_subjects > 0:
            logging.info("Validation batch sampler...")
            validation_batch_sampler = _prepare_batchsampler(
                dataset.validation_set, args_validation)

        else:
            validation_batch_sampler = None

    return training_batch_sampler, validation_batch_sampler


def _prepare_batchsampler(subset, args):
    batch_sampler = DWIMLBatchSampler(
        subset, streamline_group_name=args.streamline_group_name,
        # BATCH SIZE
        batch_size=args.batch_size, batch_size_units=args.batch_size_units,
        nb_streamlines_per_chunk=args.nb_streamlines_per_chunk,
        nb_subjects_per_batch=args.nb_subjects_per_batch, cycles=args.cycles,
        # other
        rng=args.rng)

    logging.info("\nSampler user-defined parameters: \n" +
                 format_dict_to_str(batch_sampler.params))
    return batch_sampler
