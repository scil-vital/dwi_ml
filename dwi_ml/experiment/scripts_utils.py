# -*- coding: utf-8 -*-
from os import path


def add_dwi_ml_positional_args(p):
    p.add_argument('train_database_path', type=str,
                   help="Path to the model_and_training set (.hdf5).")
    p.add_argument('valid_database_path', type=str,
                   help="Path to the validation set (.hdf5).")
    return p


def add_dwi_ml_optional_args(p):
    p.add_argument('--add-streamline-noise', action="store_true",
                   help="Add random gaussian noise to streamline coordinates "
                        "on-the-fly. Noise variance is 0.1 * step-size, "
                        "or 0.1mm if no step size is used.")
    p.add_argument('--streamlines-cut-ratio', type=float,
                   help="Cut a percentage of streamline at a random point in "
                        "each batch. [None]")
    p.add_argument('--step-size', type=float,
                   help="Resample all streamlines to this step size. If None, "
                        "train on streamlines as they are (e.g. compressed). "
                        "[None]")
    p.add_argument('--neighborhood-dist-mm', type=float,
                   help="Distance (in mm) at which to get neighborhood "
                        "information to concatenate to the input vector. If "
                        "None, no neighborhood is added. [None]")
    p.add_argument('--nb-neighborhood-axes', type=int,
                   help="Nb of axes at which to get neighborhood distance. "
                        "Default = 6 (up, down, left, right, front, back).")
    p.add_argument('--add-previous-dir', action="store_true",
                   help="Concatenate previous streamline direction to the "
                        "input vector.")
    p.add_argument('--lazy', action="store_true",
                   help="Do not load all the model_and_training dataset in  "
                        "memory at once. Load only what is needed for a"
                        "batch.")
    p.add_argument('--batch-size', type=int, default=20000,
                   help="Number of streamline points per batch. [20000]")
    p.add_argument('--volumes-per-batch', type=int,
                   help="Limits the number of volumes used in a batch. Also "
                        "determines the cache size if --cache-manager is "
                        "used. If None, use true random sampling. [None]")
    p.add_argument('--cycles-per-volume-batch', type=int,
                   help="Relevant only if --volumes-per-batch is used. Number "
                        "of update cycles before chaging to new volumes. [1]")
    p.add_argument('--n-epoch', type=int, default=100,
                   help="Maximum number of epochs. [100]")
    p.add_argument('--seed', type=int, default=1234,
                   help="Random experiment seed. [1234]")
    p.add_argument('--patience', type=int, default=20,
                   help="Use early stopping. Defines the number of epochs "
                        "after which the model should stop model_and_training "
                        "if the loss hasn't improved. [20].")
    p.add_argument('--use-gpu', action="store_true",
                   help="Train using the GPU.")
    p.add_argument('--num-workers', type=int, default=0,
                   help="Number of parallel CPU workers. [0]")
    p.add_argument('--worker-interpolation', action='store_true',
                   help="If using --num-workers > 0, interpolation will be "
                        "done on CPU by the workers instead of on the main "
                        "thread using the chosen device. [False]")
    p.add_argument('--cache-manager', action="store_true",
                   help="Relevant only if --lazy is used. Cache volumes and "
                        "streamlines in-memory instead of fetching from the "
                        "disk everytime. Cache size is determined by "
                        "--volumes-per-batch.")
    p.add_argument('--taskman-managed', action="store_true",
                   help="Instead of printing progression, print taskman-"
                        "relevant data.")
    p.add_argument('--logging', type=str,
                   choices=['error', 'warning', 'info', 'debug'],
                   default='warning', help="Activate debug mode")

    return p


def check_train_valid_args_path(args):
    if not path.exists(args.train_database_path):
        raise ValueError(
            'The model_and_training set path seems to be wrong!')
    if not path.exists(args.valid_database_path):
        raise ValueError('The validation set path seems to be wrong!')
