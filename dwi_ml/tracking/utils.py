# -*- coding: utf-8 -*-
import logging

from dipy.io.stateful_tractogram import (Space, Origin, set_sft_logger_level,
                                         StatefulTractogram)
import nibabel as nib
import numpy as np
from dipy.io.streamline import save_tractogram

from scilpy.io.utils import add_processes_arg
from scilpy.tracking.seed import SeedGenerator

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.tracking.tracking_mask import TrackingMask


def add_mandatory_options_tracking(p):
    """
    Similar to scilpy.tracking.utils.add_generic_options_tracking but the
    sh are not used for tracking. Rather, the learned model must be loaded
    together with the according input.
    """
    p.add_argument('experiment_path',
                   help='Path to the directory containing the experiment.\n'
                        '(Should contain a model subdir with a file \n'
                        'parameters.json and a file best_model_state.pkl.)')
    p.add_argument('hdf5_file',
                   help="Path to the hdf5 file.")
    p.add_argument('subj_id',
                   help="Subject id to use for tractography.\n"
                        "Will also be added as prefix to add to the "
                        "out_tractogram name.")
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')
    p.add_argument('seeding_mask_group',
                   help="Seeding mask's volume group in the hdf5.")
    p.add_argument('input_group',
                   help="Model's input's volume group in the hdf5.")
    p.add_argument('--subset', default='testing',
                   choices=['training', 'validation', 'testing'],
                   help="Subject id should probably come come the "
                        "'testing' set but you can\n modify this to "
                        "'training' or 'validation'.")


def add_tracking_options(p):
    """
    Similar to scilpy.tracking.utils.add_generic_options_tracking but
    - no algo (det/prob) anymore. Rather, propagation depends on the model.
    - no sf_threshold or sh_basis args.
    """
    track_g = p.add_argument_group('  Tracking options')
    track_g.add_argument('--algo', choices=['det', 'prob'], default='det',
                         help="Tracking algorithm (det or prob). Must be "
                              "implemented in the chosen model. [det]")
    track_g.add_argument('--step', dest='step_size', type=float, default=0.5,
                         help='Step size in mm. [%(default)s]')
    track_g.add_argument('--min_length', type=float, default=10.,
                         metavar='m',
                         help='Minimum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         metavar='M',
                         help='Maximum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--tracking_mask_group',
                         help="Tracking mask's volume group in the hdf5.")

    # Additional tracking options compared to scil_compute_local_tracking:
    track_g.add_argument('--theta', metavar='t', type=float,
                         default=90,
                         help="The tracking direction at each step being "
                              "defined by the model, \ntheta arg can't define "
                              "allowed directions in the tracking field.\n"
                              "Rather, this new equivalent angle, is used as "
                              "\na stopping criterion during propagation: "
                              "tracking \nis stopped when a direction is more "
                              "than an angle t from preceding direction")
    track_g.add_argument('--max_invalid_len', metavar='M', type=float,
                         default=0.001,
                         help="Maximum length without valid direction, in mm. "
                              "[%(default)s]")
    track_g.add_argument('--track_forward_only', action='store_true',
                         help="If set, tracks in one direction only (forward) "
                              "given the initial \nseed. The direction is "
                              "randomly drawn from the ODF.")

    track_g.add_argument('--mask_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Mask interpolation: nearest-neighbor or "
                              "trilinear. [%(default)s]")
    track_g.add_argument('--data_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Input data interpolation: nearest-neighbor or "
                              "trilinear. [%(default)s]")

    # As in scilpy:
    r_g = p.add_argument_group('  Random seeding options')
    r_g.add_argument('--rng_seed', type=int,
                     help='Initial value for the random number generator. '
                          '[%(default)s]')
    r_g.add_argument('--skip', type=int, default=0,
                     help="Skip the first N random number. \n"
                          "Useful if you want to create new streamlines to "
                          "add to \na previously created tractogram with a "
                          "fixed --rng_seed.\nEx: If tractogram_1 was created "
                          "with -nt 1,000,000, \nyou can create tractogram_2 "
                          "with \n--skip 1,000,000.")

    # Preparing upcoming GPU option:
    m_g = p.add_argument_group('  Memory options')
    ram_options = m_g.add_mutually_exclusive_group()
    # Parallel processing or GPU processing
    add_processes_arg(ram_options)
    ram_options.add_argument('--use_gpu', action='store_true',
                             help="If set, use GPU for processing. Cannot be "
                                  "used \ntogether with --processes.")
    m_g.add_argument('--simultaneous_tracking', type=int, default=1,
                     help='Track n streamlines at the same time. Intended for '
                          'GPU usage. Default = 1 (no simultaneous tracking).')

    return track_g


def prepare_dataset_one_subj(args):
    # Right now, we con only track on one subject at the time. We could
    # instantiate a LazySubjectData directly but we want to use the cache
    # manager (suited better for multiprocessing)
    dataset = MultiSubjectDataset(args.hdf5_file, lazy=args.lazy,
                                  cache_size=args.cache_size,
                                  log_level=logging.WARNING)

    if args.subset == 'testing':
        # Most logical choice.
        dataset.load_data(load_training=False, load_validation=False,
                          subj_id=args.subj_id)
        subset = dataset.testing_set
    elif args.subset == 'training':
        dataset.load_data(load_validation=False, load_testing=False,
                          subj_id=args.subj_id)
        subset = dataset.training_set
    elif args.subset == 'validation':
        dataset.load_data(load_training=False, load_testing=False,
                          subj_id=args.subj_id)
        subset = dataset.validation_set
    else:
        raise ValueError("Subset must be one of 'training', 'validation' "
                         "or 'testing.")

    if args.subj_id not in subset.subjects:
        raise ValueError("Subject {} does not belong in hdf5's {} set."
                         .format(args.subj_id, args.subset))
    subj_idx = subset.subjects.index(args.subj_id)  # Should be 0.

    return subset, subj_idx


def prepare_seed_generator(parser, args, hdf_handle):
    """
    Prepares a SeedGenerator from scilpy's library. Returns also some header
    information to allow verifications.
    """
    seeding_group = hdf_handle[args.subj_id][args.seeding_mask_group]
    seed_data = np.array(seeding_group['data'], dtype=np.float32)
    seed_res = np.array(seeding_group.attrs['voxres'], dtype=np.float32)
    affine = np.array(seeding_group.attrs['affine'], dtype=np.float32)
    ref = nib.Nifti1Image(seed_data, affine)

    seed_generator = SeedGenerator(seed_data, seed_res, space=Space.VOX,
                                   origin=Origin('corner'))

    if len(seed_generator.seeds_vox) == 0:
        parser.error('Seed mask "{}" does not have any voxel with value > 0.'
                     .format(args.in_seed))

    if args.npv:
        # Note. Not really nb seed per voxel, just in average.
        nbr_seeds = len(seed_generator.seeds_vox) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds_vox)

    seed_header = nib.Nifti1Image(seed_data, affine).header

    return seed_generator, nbr_seeds, seed_header, ref


def prepare_tracking_mask(args, hdf_handle):
    """
    Prepare the tracking mask as a DataVolume from scilpy's library. Returns
    also some header information to allow verifications.
    """
    if args.tracking_mask_group not in hdf_handle[args.subj_id]:
        raise KeyError("HDF group '{}' not found for subject {} in hdf file {}"
                       .format(args.tracking_mask_group, args.subj_id,
                               hdf_handle))
    tm_group = hdf_handle[args.subj_id][args.tracking_mask_group]
    mask_data = np.array(tm_group['data'], dtype=np.float64).squeeze()
    # mask_res = np.array(tm_group.attrs['voxres'], dtype=np.float32)
    affine = np.array(tm_group.attrs['affine'], dtype=np.float32)
    ref = nib.Nifti1Image(mask_data, affine)

    mask = TrackingMask(mask_data.shape, mask_data, args.mask_interp)

    return mask, ref


def prepare_step_size_vox(step_size, res):
    if step_size:
        # Our step size is in voxel space = we require isometric voxels.
        if not res[0] == res[1] == res[2]:
            raise ValueError("We are expecting isometric voxels.")

        step_size_vox_space = step_size / res[0]
        normalize_directions = True
        logging.info("Step size in voxel space will be {}"
                     .format(step_size_vox_space))
    else:
        # Step size 0 is understood here as 'keep model output as is', i.e.
        # multiplying by 1 works.
        step_size_vox_space = 1
        normalize_directions = False

    return step_size_vox_space, normalize_directions


def track_and_save(tracker, args, ref):
    with Timer("\nTracking...", newline=True, color='blue'):
        streamlines, seeds = tracker.track()

        logging.debug("Tracked {} streamlines (out of {} seeds). Now saving..."
                      .format(len(streamlines), tracker.nbr_seeds))

    if len(streamlines) == 0:
        logging.warning("No streamlines created! Not saving tractogram!")
        return

    # save seeds if args.save_seeds is given
    # Seeds must be saved in voxel space (ok!), but origin: center, if we want
    # to use scripts such as scil_compute_seed_density_map.
    if args.save_seeds:
        print("Saving seeds in data_per_streamline.")
        seeds = [np.asarray(seed) - 0.5 for seed in seeds]  # to_center
        data_per_streamline = {'seeds': seeds}
    else:
        data_per_streamline = {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    logging.info("Saving resulting tractogram to {}"
                 .format(args.out_tractogram))
    sft = StatefulTractogram(streamlines, ref, space=Space.VOX,
                             origin=Origin('corner'),
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram, bbox_valid_check=False)
