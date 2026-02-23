# -*- coding: utf-8 -*-
import logging
from argparse import ArgumentParser

from dwi_ml.models.direction_getter_models import keys_to_direction_getters


def add_direction_getter_args(p: ArgumentParser, gaussian_fisher_args=True):
    # For the direction getters
    dg_keys = list(keys_to_direction_getters.keys())
    p.add_argument(
        '--dg_key', choices=dg_keys, metavar='key',
        default='cosine-regression',
        help="Model for the direction getter layer. One of {cosine-regression,"
             "l2-regression, \nsphere-classification, "
             "smooth-sphere-classification, gaussian, \ngaussian-mixture, "
             "fisher-von-mises, fisher-von-mises-mixture.}\n"
             "With sphere classification, the default sphere is "
             "symmetric724.")
    p.add_argument(
        '--dg_dropout', type=float, metavar='r', default=0.,
        help="Dropout ratio for the direction getter. Default: 0.")

    # Gaussian models, Fisher-von-Mises models
    if gaussian_fisher_args:
        p.add_argument(
            '--add_entropy_to_gauss', nargs='?', const=1.0, type=float,
            metavar='f',
            help="For GAUSSIAN models: If set, adds the entropy to the negative "
                 "log-likelihood \nloss. By defaut, weight is 1.0, but a "
                 "value >1 can be added \n to increase its influence.")
        p.add_argument(
            '--dg_nb_gaussians', type=int, metavar='n',
            help="For GAUSSIAN models: Number of gaussians in the case of a "
                 "mixture model. [3]")
        p.add_argument(
            '--dg_nb_clusters', type=int,
            help="For FISHER VON MISES models: Number of clusters in the case "
                 "of a mixture model for the direction \ngetter. [3]")
    p.add_argument(
        '--normalize_targets', const=1., nargs='?', type=float,
        metavar='norm',
        help="For REGRESSION models: If set, target directions will be "
             "normalized before \ncomputing the loss. Default norm: 1.")
    p.add_argument(
        '--normalize_outputs', const=1., nargs='?', type=float,
        metavar='norm',
        help="For REGRESSION models: If set, model outputs will be "
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
    p.add_argument(
        '--eos_weight', type=float, default=1.0, metavar='w',
        help="In the case of regression, Gaussian and Fisher von Mises models: "
             "defines the \nweight of the EOS loss: "
             "final_loss = loss + weight * eos_loss")


def check_args_direction_getter(args):
    dg_args = {'dropout': args.dg_dropout,
               'add_eos': args.add_eos,
               'eos_weight': args.eos_weight,
               }

    if args.dg_dropout < 0 or args.dg_dropout > 1:
        raise ValueError('The dg dropout rate must be between 0 and 1.')

    # Gaussian additional arg = nb_gaussians and entropy_weight.
    if args.dg_key == 'gaussian-mixture':
        if args.dg_nb_gaussians:
            dg_args.update({'nb_gaussians': args.dg_nb_gaussians})
        if args.add_entropy_to_gauss:
            dg_args.update({'entroy_weight': args.add_entropy_to_gauss})

    else:
        if args.dg_nb_gaussians:
            logging.warning("You have provided a value for --dg_nb_gaussians "
                            "but the chosen direction getter is not the "
                            "gaussian mixture. Ignored.")
        if args.add_entropy_to_gauss:
            logging.warning("You have provided a value for --add_entropy_to_gauss "
                            "but the chosen direction getter is not the "
                            "gaussian mixture. Ignored.")

    # Fisher additional arg = nb_clusters
    if args.dg_key == 'fisher-von-mises-mixture':
        if args.dg_nb_clusters:
            dg_args.update({'n_cluster': args.dg_nb_clusters})
    elif args.dg_nb_clusters:
        logging.warning("You have provided a value for --dg_nb_clusters but "
                        "the chosen direction getter is not the Fisher von "
                        "Mises mixture. Ignored.")

    # Regression and normalisation
    if 'regression' in args.dg_key or 'gaussian' in args.dg_key:
        dg_args['normalize_targets'] = args.normalize_targets
    elif args.normalize_targets:
        raise ValueError("--normalize_targets is only an option for "
                         "regression and gaussian models.")

    if 'regression' in args.dg_key:
        dg_args['normalize_outputs'] = args.normalize_outputs
    elif args.normalize_outputs is not None:
        raise ValueError("--normalize_outputs is only an option for "
                         "regression models.")

    return dg_args
