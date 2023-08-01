# -*- coding: utf-8 -*-
import logging
from argparse import ArgumentParser

from dwi_ml.models.direction_getter_models import keys_to_direction_getters


def get_direction_getter_args(gaussian_fisher_args=True):
    dg_keys = list(keys_to_direction_getters.keys())

    args = {
        '--dg_key': {
            'choices': dg_keys, 'metavar': 'key',
            'default': 'cosine-regression',
            'help':
                "Model for the direction getter layer. One of {cosine-regression,"
                "l2-regression, \nsphere-classification, "
                "smooth-sphere-classification, gaussian, \ngaussian-mixture, "
                "fisher-von-mises, fisher-von-mises-mixture.}\n"
                "With sphere classification, the default sphere is "
                "symmetric724."},
        '--dg_dropout': {
            'type': float, 'metavar': 'r', 'default': 0.,
            'help': "Dropout ratio for the direction getter. Default: 0."},
        '--compress_loss': {
            'metavar': 'eps', 'nargs': '?', 'const': 1e-3, 'type': float,
            'help':
                "If set, compress the loss. \nCan be used independently from "
                "options on the input streamlines such as --compress.\n"
                "Compression ratio (eps) can be given: as long as the angle is \n"
                "smaller than eps (in degree), the next points' loss are "
                "averaged together."},
        '--weight_loss_with_angle': {
            'action': 'store_true',
            'help': "If set, weight the loss at each coordinate with the angle "
                    "with previous dir."},
        '--normalize_targets': {
            'const': 1., 'nargs': '?', 'type': float, 'metavar': 'norm',
            'help': "For REGRESSION models:  If set, target directions will be "
                    "normalized before \ncomputing the loss. Default norm: 1."},
        '--normalize_outputs': {
            'const': 1., 'nargs': '?', 'type': float, 'metavar': 'norm',
            'help': "For REGRESSION models:  If set, model outputs will be "
                    "normalized. Default norm: 1."},
        '--add_eos': {
            'action': 'store_true',
            'help':
                "If true, adds EOS during the loss computation.\n"
                "  1) Then, the last coordinate of streamlines has a target \n"
                "  (EOS), requiring one more input point.\n"
                "  2) In REGRESSION models: Adds a fourth dimension during "
                "prediction.\n"
                "     In CLASSIFICATION models, adds an additional EOS class.\n"
        }
    }

    # Gaussian models, Fisher-von-Mises models
    if gaussian_fisher_args:
        args.update({
            '--dg_nb_gaussians': {
                'type': int, 'metavar': 'n',
                'help': "Number of gaussians in the case of a Gaussian Mixture "
                        "model for the direction \ngetter. [3]"},
            '--dg_nb_clusters': {
                'type': int, 'metavar': 'k',
                'help': "Number of clusters in the case of a Fisher von Mises "
                        "Mixture model for the direction \ngetter. [3]."
            }
        })
    return args


def check_args_direction_getter(args):
    dg_args = {'dropout': args.dg_dropout,
               'add_eos': args.add_eos,
               'compress_loss': args.compress_loss is not None,
               'compress_eps': args.compress_loss,
               'weight_loss_with_angle': args.weight_loss_with_angle,
               }

    if args.dg_dropout < 0 or args.dg_dropout > 1:
        raise ValueError('The dg dropout rate must be between 0 and 1.')

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
