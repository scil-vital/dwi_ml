# -*- coding: utf-8 -*-
import argparse


def add_args_batch_sampler(p: argparse.ArgumentParser):
    # BATCH SIZE
    g_batch_size = p.add_argument_group("Batch sampler: batch size")

    g_batch_size.add_argument(
        '--batch_size_training', type=int, default=100, metavar='s',
        help="Batch size. Unit must be spectified (through batch_size_units)."
             "\nThe total size your computer will accept depends on the "
             "type of \ninput data. You will need to test this value. [100]\n"
             "Suggestion: in nb_streamlines: 100. In length_mm: 10000. \n")
    g_batch_size.add_argument(
        '--batch_size_validation', type=int, default=100, metavar='s',
        help="Idem; batch size during validation.")
    g_batch_size.add_argument(
        '--batch_size_units', type=str, metavar='u',
        choices={'nb_streamlines', 'length_mm'},
        help="One of 'nb_streamlines' or 'length_mm' (which should hopefully "
             "\nbe correlated to the number of input data points).")
    g_batch_size.add_argument(
        '--nb_streamlines_per_chunk', type=int, default=None, metavar='n',
        help="Only used with  batch_size_units='length_mm'. \nChunks of "
             "n streamlines are sampled at once, their size is \nchecked, "
             "and then number of streamlines is ajusted until below \n"
             "batch_size.")
    g_batch_size.add_argument(
        '--nb_subjects_per_batch', type=int, metavar='n',
        help="Maximum number of different subjects from which to load data \n"
             "in each batch. This should help avoid loading too many inputs \n"
             "in memory, particularly for lazy data. If not set, we will "
             "use \ntrue random sampling. Suggestion, 5. \n"
             "**Note: Will influence the cache if the cache_manager is used.")
    g_batch_size.add_argument(
        '--cycles', type=int, metavar='c',
        help="Relevant only if nb_subject_per_batch is set. Number of cycles\n"
             "before changing to new subjects (and thus loading new volumes).")
