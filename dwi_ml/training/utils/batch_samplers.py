# -*- coding: utf-8 -*-
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler


def get_args_batch_sampler():
    args = {
        '--batch_size_training': {
            'type': int, 'default': 1000, 'metavar': 's',
            'help': "Batch size. Unit must be spectified (through "
                    "batch_size_units). The total size \nyour computer will "
                    "accept depends on the type of input data. You will "
                    "need to \ntest this value. Default: 100. Suggestion: in "
                    "nb_streamlines: 100. In \nlength_mm: 10000."},
        '--batch_size_validation': {
            'type': int, 'default': 1000, 'metavar': 's',
            'help': "Idem; batch size during validation."},
        '--batch_size_units': {
            'metavar': 'u', 'choices': {'nb_streamlines', 'length_mm'},
            'default': 'nb_streamlines',
            'help': "One of 'nb_streamlines' or 'length_mm'."},
        '--nb_streamlines_per_chunk': {
            'type': int, 'metavar': 'n',
            'help': "Only used with  batch_size_units in 'length_mm'. Chunks "
                    "of n streamlines are \nsampled at once, their size is "
                    "checked, and then number of streamlines is \najusted "
                    "until below batch_size."},
        '--nb_subjects_per_batch': {
            'type': int, 'metavar': 'n',
            'help': "Maximum number of different subjects from which to load "
                    "data in each batch. \nThis should help avoid loading too "
                    "many inputs in memory, particularly for lazy \ndata. If "
                    "not set, we will use true random sampling. "
                    "Suggestion: 5."},
        '--cycles': {
            'type': int, 'metavar': 'c',
            'help': "Relevant only if nb_subject_per_batch is set. Number of "
                    "cycles before \nchanging to new subjects (and thus "
                    "loading new volumes)."
        }
    }
    return args


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

        logging.info("Batch sampler's user-defined parameters: " +
                     format_dict_to_str(batch_sampler.params_for_checkpoint))

    return batch_sampler
