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

    p.add_argument(
        '--EOS_type_loss',
        choices=['as_label', 'as_zeros', 'as_class'],
        help="Choice of EOS addition during the loss computation.\n"
             "A EOS choice must also be added during forward so that the "
             "model's outputs will have the right length.\n"
             "  - as_label: To be used with REGRESSION models\n"
             "    Adds a fourth dimension during prediction.\n"
             "  - as_zeros: To be used with a REGRESSION model.\n"
             "    A [0,0,0] direction is added to the targets. We also "
             "    verify that no normalization is performed to ensure that "
             "    the model can learn zeros.\n"
             "  - as_class: To be used with CLASSIFICATION models.\n"
             "    Adds an additional EOS class.\n")
    p.add_argument(
        '--smooth_labels', action='store_true',
        help="To be used with CLASSIFICATION models.\n"
             "If set, applies a Gaussian to smooth the labels, as "
             "suggested in Deeptract (Benou and Raviv, 2019).")


def check_args_direction_getter(args):
    dg_args = {}

    if args.dg_dropout:
        dg_args.update({'dropout': args.dg_dropout})

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

    # Classification and smoothing
    if args.smooth_labels and 'classification' not in args.dg_key:
        raise ValueError("smooth_labels can only be used together with "
                         "a classification model.")

    # EOS:
    if args.EOS_type_loss is not None and args.EOS_type_foward is None:
        logging.warning("The EOS added during loss computation will probably "
                        "fail: no EOS is added during the forward method!")
    if args.EOS_type_loss == 'as_label':
        if 'regression' not in args.dg_key:
            raise ValueError("--EOS_type_loss 'as_label' is intented for "
                             "regression models.")
        dg_args.update({
            'add_eos_label': True
        })
    elif args.EOS_type_loss == 'as_classification':
        if 'classification' not in args.dg_key:
            raise ValueError("--EOS_type_loss 'as_class' is intented for "
                             "classification models.")
        dg_args.update({
            'add_eos_class': True
        })
    elif args.EOS_type_loss == 'as_zeros':
        if 'regression' not in args.dg_key:
            raise ValueError("--EOS_type_loss 'as_zeros' is intented for "
                             "regression models.")
        if args.normalize_outputs or args.normalize_targets:
            logging.warning("You probably shouldn't use --EOS_type_loss "
                            "'as_zeros' together with --normalize_targets nor "
                            "--normalize_outputs.")

    return dg_args
