# -*- coding: utf-8 -*-
import logging

from dwi_ml.models.direction_getter_models import keys_to_direction_getters


def add_direction_getter_args(p, dropout_arg=True,
                              gaussian_fisher_args=True):
    # For the direction getters
    p.add_argument(
        '--dg_key', choices=keys_to_direction_getters.keys(), metavar='key',
        default='cosine-regression',
        help="Model for the direction getter layer. One of {cosine-regression,"
             "l2-regression, \nsphere-classification, gaussian, "
             "gaussian-mixture, fisher-von-mises, fisher-von-\nmises-mixture.")
    if dropout_arg:
        p.add_argument(
            '--dg_dropout', type=float, metavar='r',
            help="Dropout ratio for the direction getter. Default: the value "
                 "of --dropout option \nor, if not given, 0.")

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
             "normalized before computing the loss. Default norm: 1.")
    p.add_argument(
        '--normalize_outputs', const=1., nargs='?', type=float,
        help="For REGRESSION models:  If set, model outputs will be "
             "normalized. Default norm: 1.")


def check_args_direction_getter(args):
    dg_args = {}

    if args.dg_dropout:
        dg_args.update({'dropout': args.dg_dropout})

    if args.dg_key == 'gaussian-mixture':
        if args.dg_nb_gaussians:
            dg_args.update({'nb_gaussians': args.dg_nb_gaussians})
    elif args.dg_nb_gaussians:
        logging.warning("You have provided a value for --dg_nb_gaussians but "
                        "the chosen direction getter is not the gaussian "
                        "mixture. Ignored.")

    if args.dg_key == 'fisher-von-mises-mixture':
        if args.dg_nb_clusters:
            dg_args.update({'n_cluster': args.dg_nb_clusters})
    elif args.dg_nb_clusters:
        logging.warning("You have provided a value for --dg_nb_clusters but "
                        "the chosen direction getter is not the Fisher von "
                        "Mises mixture. Ignored.")

    if 'regression' in args.dg_key:
        dg_args.update({
            'normalize_targets': args.normalize_targets,
            'normalize_outputs': args.normalize_outputs,
        })
    return dg_args
