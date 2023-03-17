# -*- coding: utf-8 -*-
import logging

from dwi_ml.models.direction_getter_models import keys_to_direction_getters


def add_direction_getter_args(p, gaussian_fisher_args=True):
    # For the direction getters
    dg_keys = list(keys_to_direction_getters.keys())
    p.add_argument(
        '--dg_key', choices=dg_keys, metavar='key',
        default='cosine-regression',
        help="Model for the direction getter layer. One of {cosine-regression,"
             "l2-regression, \nsphere-classification, "
             "smooth-sphere-classification, gaussian, \ngaussian-mixture, "
             "fisher-von-mises, fisher-von-mises-mixture.}")
    p.add_argument(
        '--dg_dropout', type=float, metavar='r', default=0.,
        help="Dropout ratio for the direction getter. Default: 0.")

    # Gaussian models, Fisher-von-Mises models
    if gaussian_fisher_args:
        p.add_argument(
            '--dg_nb_gaussians', type=int, metavar='n',
            help="Number of gaussians in the case of a Gaussian Mixture model "
                 "for the direction \ngetter. [3]")
        p.add_argument(
            '--dg_nb_clusters', type=int,
            help="Number of clusters in the case of a Fisher von Mises "
                 "Mixture model for the direction \ngetter. [3].")
    p.add_argument(
        '--normalize_targets', const=1., nargs='?', type=float,
        help="For REGRESSION models:  If set, target directions will be "
             "normalized before \ncomputing the loss. Default norm: 1.")
    p.add_argument(
        '--normalize_outputs', const=1., nargs='?', type=float,
        help="For REGRESSION models:  If set, model outputs will be "
             "normalized. Default norm: 1.")

    # EOS
    p.add_argument(
        '--add_eos', action='store_true',
        help="If true, adds EOS during the loss computation.\n"
             "  1) Then, the last coordinate of streamlines has a target \n"
             "  (EOS), requiring one more input point.\n"
             "  2) In REGRESSION models: Adds a fourth dimension during "
             "prediction.\n"
             "     In CLASSIFICATION models, adds an additional EOS class.\n")


def check_args_direction_getter(args):
    dg_args = {'dropout': args.dg_dropout,
               'add_eos': args.add_eos}

    # Gaussian additional arg = nb_gaussians.
    if args.dg_key == 'gaussian-mixture':
        if args.dg_nb_gaussians:
            dg_args.update({'nb_gaussians': args.dg_nb_gaussians})
    elif args.dg_nb_gaussians:
        logging.warning("You have provided a value for --dg_nb_gaussians but "
                        "the chosen direction getter is not the gaussian "
                        "mixture. Ignored.")

    # Fisher additional arg = nb_clusters
    if args.dg_key == 'fisher-von-mises-mixture':
        if args.dg_nb_clusters:
            dg_args.update({'n_cluster': args.dg_nb_clusters})
    elif args.dg_nb_clusters:
        logging.warning("You have provided a value for --dg_nb_clusters but "
                        "the chosen direction getter is not the Fisher von "
                        "Mises mixture. Ignored.")

    # Regression and normalisation
    if 'regression' in args.dg_key:
        dg_args.update({
            'normalize_targets': args.normalize_targets,
            'normalize_outputs': args.normalize_outputs,
        })

    return dg_args
