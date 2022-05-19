# -*- coding: utf-8 -*-


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
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')
    p.add_argument('algo', choices=['det', 'prob'],
                   help="Tracking algorithm (det or prob). Must be "
                        "implemented in the chosen model.")

    sm_g = p.add_argument_group('Loading params: seeding mask: CHOOSE ONE')
    seeding_mask = sm_g.add_mutually_exclusive_group()
    seeding_mask.add_argument('--sm_from_hdf5', metavar='group',
                              help="Seeding mask's volume group in the hdf5.\n"
                                   "Must correspond to the config_file used "
                                   "when creating the hdf5.")
    seeding_mask.add_argument('--sm_from_data', metavar='file',
                              help="Seeding mask's volume.")

    tm_g = p.add_argument_group('Loading params: tracking mask: CHOOSE ONE')
    tracking_mask = tm_g.add_mutually_exclusive_group()
    tracking_mask.add_argument('--tm_from_hdf5', metavar='group',
                               help="Tracking mask's volume group in the hdf5."
                                    "\nMust correspond to the config_file "
                                    "used when creating the hdf5.")
    tracking_mask.add_argument('--tm_from_data', metavar='file',
                               help="Tracking mask's volume.")

    i_g = p.add_argument_group('Loading params: input: CHOOSE ONE')
    inputs = i_g.add_mutually_exclusive_group()
    inputs.add_argument('--input_from_hdf5', metavar='group',
                        help="Model's input's volume group in the hdf5.\n"
                             "Must correspond to the config_file used "
                             "when creating the hdf5.")
    inputs.add_argument('--input_from_data', metavar='file',
                        help="Model's input's volume name.")

    l_g = p.add_argument_group('Loading params')
    l_g.add_argument('--subj_id', metavar='ID',
                     help="For data loaded through hdf5, the subject id "
                          "to use for tractography.\n"
                          "For data loaded directly, prefix to add to the "
                          "out_tractogram name.")
    l_g.add_argument('--hdf5_file', metavar='f',
                     help="hdf5 file path, in case any option above uses data "
                          "loaded from a hdf5 file.")


def add_tracking_options(p):
    """
    Similar to scilpy.tracking.utils.add_generic_options_tracking but
    - no algo (det/prob) anymore. Rather, propagation depends on the model.
    - no sf_threshold or sh_basis args.
    """
    track_g = p.add_argument_group('  Tracking options')

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
    track_g.add_argument('--rk_order', metavar="K", type=int, default=2,
                         choices=[1, 2, 4],
                         help="The order of the Runge-Kutta integration used "
                              "for the \nstep function [%(default)s]. As a "
                              "rule of thumb, doubling the rk_order \nwill "
                              "double the computation time in the worst case.")

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
                         default=1,
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
    return track_g
