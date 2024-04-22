# -*- coding: utf-8 -*-
import argparse

from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler


def add_args_batch_sampler(p: argparse.ArgumentParser):
    # BATCH SIZE
    g_batch_size = p.add_argument_group("Batch sampler: batch size")

    g_batch_size.add_argument(
        '--batch_size_training', type=int, default=100, metavar='s',
        help="Batch size. Unit must be spectified (through batch_size_units). "
             "The total size \nyour computer will accept depends on the "
             "type of input data. You will need to \ntest this value. "
             "Suggestion: in nb_streamlines: 100. In length_mm: 10,000. [100]")
    g_batch_size.add_argument(
        '--batch_size_validation', type=int, default=100, metavar='s',
        help="Idem; batch size during validation.")
    g_batch_size.add_argument(
        '--batch_size_units', type=str, metavar='u', default='nb_streamlines',
        choices={'nb_streamlines', 'length_mm'},
        help="One of 'nb_streamlines' or 'length_mm' (which should hopefully "
             "be correlated \nto the number of input data points).")
    g_batch_size.add_argument(
        '--nb_streamlines_per_chunk', type=int, default=None, metavar='n',
        help="Only used with batch_size_units='length_mm'. Chunks of "
             "n streamlines are sampled \nat once, their size is checked, "
             "and then number of streamlines is ajusted until \nbelow "
             "batch_size.")
    g_batch_size.add_argument(
        '--nb_subjects_per_batch', type=int, metavar='n',
        help="Maximum number of different subjects from which to load data "
             "in each batch. This \nshould help avoid loading too many inputs "
             "in memory, particularly for lazy data. \nIf not set, we will "
             "use true random sampling. Suggestion, 5.\n"
             "**Note: Will influence the cache if the cache_manager is used.")
    g_batch_size.add_argument(
        '--cycles', type=int, metavar='c',
        help="Relevant only if nb_subject_per_batch is set. Number of cycles "
             "before changing \nto new subjects (and thus loading new "
             "volumes).")


def prepare_batch_sampler(dataset, args, sub_loggers_level):
    with Timer("\nPreparing batch sampler...", newline=True, color='green'):
        batch_sampler = DWIMLBatchIDSampler(
            dataset=dataset, streamline_group_name=args.streamline_group_name,
            batch_size_training=args.batch_size_training,
            batch_size_validation=args.batch_size_validation,
            batch_size_units=args.batch_size_units,
            nb_streamlines_per_chunk=args.nb_streamlines_per_chunk,
            nb_subjects_per_batch=args.nb_subjects_per_batch,
            cycles=args.cycles,
            rng=args.rng, log_level=sub_loggers_level)

    return batch_sampler
