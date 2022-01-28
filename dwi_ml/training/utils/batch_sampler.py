# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.training.batch_samplers import \
    BatchStreamlinesSamplerOneInput
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer


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
             "volumes)."
    )

    # STREAMLINES PREPROCESSING
    g_streamlines_preprocessing = p.add_argument_group(
        "Batch sampler: streamlines preprocessing")
    sub = g_streamlines_preprocessing.add_mutually_exclusive_group()
    sub.add_argument(
        '--step_size', type=float, metavar='s',
        help="Resample all streamlines to this step size (in mm). If not set, "
             "will keep \nstreamlines as they are. Note that you probably may "
             "have already resampled or \ncompressed your data creating your "
             "dataset, but you can use a different \nchoice in the batch "
             "sampler if you wish.")
    sub.add_argument(
        '--compress', action='store_true',
        help="If set, compress streamlines. Once again, the choice can be "
             "different in the \nbatch sampler than chosen when creating the "
             "hdf5.")

    # STREAMLINES AUGMENTATION
    g_data_augmentation = p.add_argument_group(
        "Batch sampler: streamlines data augmentation")
    g_data_augmentation.add_argument(
        '--noise_size', type=float, metavar='v', dest='noise_gaussian_size',
        help="If set, add random Gaussian noise to streamline coordinates "
             "with given variance. \nThis corresponds to the std of the "
             "Gaussian. If step_size is not given, make sure \nit is smaller "
             "than your step size to avoid flipping direction. \nEx, you "
             "could choose 0.1 * step-size. \nNoise is truncated to "
             "+/-(2*noise_sigma) and to +/-(0.5 * step-size).")
    g_data_augmentation.add_argument(
        '--noise_variability', type=float, metavar='v',
        dest='noise_gaussian_variability',
        help="If set, a variation is applied to the noise_size to have very "
             "noisy streamlines and \nless noisy streamlines. This means that "
             "the real gaussian_size will be a random \nnumber between "
             "[size - variability, size + variability].")
    g_data_augmentation.add_argument(
        '--split_ratio', type=float, metavar='r',
        help="Percentage of streamlines to randomly split into 2, in each "
             "batch (keeping both \nsegments as two independent streamlines). "
             "The reason for cutting is to help the \nML algorithm to track "
             "from the middle of WM by having already seen \nhalf-"
             "streamlines. If you are using interface seeding, this is not "
             "necessary.")
    g_data_augmentation.add_argument(
        '--reverse_ratio', type=float, metavar='r',
        help="Percentage of streamlines to randomly reverse in each batch.")


def prepare_batchsamplers_oneinput(dataset, args_training, args_validation,
                                   neighborhood_points):
    with Timer("\nPreparing batch samplers...", newline=True, color='green'):
        logging.info("Training batch sampler...")
        training_batch_sampler = _prepare_batchsampler_oneinput(
            dataset.training_set, args_training, neighborhood_points)

        if dataset.validation_set.nb_subjects > 0:
            logging.info("Validation batch sampler...")
            validation_batch_sampler = _prepare_batchsampler_oneinput(
                dataset.training_set, args_validation, neighborhood_points)

        else:
            validation_batch_sampler = None

    return training_batch_sampler, validation_batch_sampler


def _prepare_batchsampler_oneinput(subset, args, neighborhood_points):
    if args.step_size and args.step_size <= 0:
        raise ValueError("Step size can't be 0 or less!")
        # Note. When using
        # scilpy.tracking.tools.resample_streamlines_step_size, a warning
        # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
        # that the value is suspicious. Not raising the same warnings here
        # as you may be wanting to test weird things to understand better
        # your model.

    if args.split_ratio > 1 or args.split_ratio < 0:
        raise ValueError("split_ratio must be a ratio (i.e. between 0 "
                         "and 1).")

    if args.reverse_ratio > 1 or args.reverse_ratio < 0:
        raise ValueError("reverse_ratio must be a ratio (i.e. between 0 "
                         "and 1).")

    batch_sampler = BatchStreamlinesSamplerOneInput(
        subset, input_group_name=args.input_group_name,
        streamline_group_name=args.streamline_group_name,
        # BATCH SIZE
        max_chunk_size=args.chunk_size, max_batch_size=args.max_batch_size,
        nb_subjects_per_batch=args.nb_subjects_per_batch, cycles=args.cycles,
        # STREAMLINES PREPROCESSING
        step_size=args.step_size, compress=args.compress,
        # STREAMLINES AUGMENTATION
        noise_gaussian_size=args.noise_gaussian_size,
        noise_gaussian_variability=args.noise_gaussian_variability,
        reverse_ratio=args.reverse_ratio, split_ratio=args.split_ratio,
        # NEIGHBORHOOD
        neighborhood_points=neighborhood_points,
        rng=args.rng, wait_for_gpu=args.wait_for_gpu
    )

    logging.info(
        "\nSampler user-defined parameters: \n" +
        format_dict_to_str(batch_sampler.params))
    return batch_sampler
