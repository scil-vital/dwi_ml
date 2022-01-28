# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.batch_samplers import DWIMLBatchSampler


def add_args_batch_sampler(p: argparse.ArgumentParser):
    # BATCH SIZE
    g_batch_size = p.add_argument_group("Batch sampler: batch size")

    gg_batch_size = g_batch_size.add_mutually_exclusive_group()
    gg_batch_size.add_argument(
        '--max_batch_size_in_nb_streamlines', type=int, default=100,
        metavar='s',
        help="Number of streamlines per batch. The total number your \n"
             "computer will accept depends on the type of input data. \nYou "
             "will need to test this value. Default: 100.")
    gg_batch_size.add_argument(
        '--max_batch_size_in_nb_points', type=int, metavar='s',
        help="Alternative choice: You can define the batch size in terms of "
             "number of points per batch.")

    g_batch_size.add_argument(
        '--chunk_size', type=int, default=256, metavar='s',
        help="Number of streamlines to sample together while creating the "
             "batches. \nIf the size of the streamlines is known in terms of "
             "number of points \n(resampling has been done, and no "
             "compressing is done in the batch \nsampler), we iteratively add "
             "chunk_size streamlines to the batch until \nthe total number of "
             "sampled timepoint reaches the max_batch_size. \nElse, the total "
             "number of streamlines in the batch will be 1*chunk_size.\n"
             "Default: 256, for no good reason.")
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

    # STREAMLINES PREPROCESSING
    # toDo Should not be necessary for the batch sampler. Will be added in the
    #  batch loader. Batch size to be cleaned.


def prepare_batchsamplers_train_valid(dataset, args_training, args_validation):
    with Timer("\nPreparing batch samplers...", newline=True, color='green'):
        logging.info("Training batch sampler...")
        training_batch_sampler = _prepare_batchsampler(
            dataset.training_set, args_training)

        if dataset.validation_set.nb_subjects > 0:
            logging.info("Validation batch sampler...")
            validation_batch_sampler = _prepare_batchsampler(
                dataset.training_set, args_validation)

        else:
            validation_batch_sampler = None

    return training_batch_sampler, validation_batch_sampler


def _prepare_batchsampler(subset, args):
    if args.step_size and args.step_size <= 0:
        raise ValueError("Step size can't be 0 or less!")
        # Note. When using
        # scilpy.tracking.tools.resample_streamlines_step_size, a warning
        # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
        # that the value is suspicious. Not raising the same warnings here
        # as you may be wanting to test weird things to understand better
        # your model.

    batch_sampler = DWIMLBatchSampler(
        subset, streamline_group_name=args.streamline_group_name,
        # BATCH SIZE
        max_batch_size=args.max_batch_size,
        batch_size_unit=args.batch_size_unit, max_chunk_size=args.chunk_size,
        nb_subjects_per_batch=args.nb_subjects_per_batch, cycles=args.cycles,
        # STREAMLINES PREPROCESSING
        step_size=args.step_size, compress=args.compress,
        # other
        rng=args.rng)

    logging.info("\nSampler user-defined parameters: \n" +
                 format_dict_to_str(batch_sampler.params))
    return batch_sampler
